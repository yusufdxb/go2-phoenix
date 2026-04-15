"""Pre-deployment parity gate: ONNX vs TorchScript on a real parquet.

Catches silently-broken exports (wrong actor weights, obs-dim mismatch,
runtime numeric drift) before the policy is run on hardware. Invoked
via ``python -m phoenix.sim2real.verify_deploy --parquet ... --deploy-cfg ...``.

The core :func:`verify_parity` is pure numpy + injected callables so CI
can exercise it without onnxruntime or torch. The CLI wrapper loads
both backends from the paths in ``configs/sim2real/deploy.yaml``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from phoenix.replay.trajectory_reader import TrajectoryReader
from phoenix.sim2real.observation import JointOrder, ObservationBuilder

logger = logging.getLogger("phoenix.sim2real.verify_deploy")

InferenceFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ParityReport:
    steps_checked: int
    max_abs_diff: float
    mean_abs_diff: float
    per_step_max: tuple[float, ...]
    tol: float

    @property
    def passed(self) -> bool:
        return self.max_abs_diff <= self.tol


def verify_parity(
    obs_iter: Iterable[np.ndarray],
    onnx_infer: InferenceFn,
    torch_infer: InferenceFn,
    tol: float,
) -> ParityReport:
    """Run each observation through both inference fns, compare action outputs."""
    per_step: list[float] = []
    running_sum = 0.0
    for obs in obs_iter:
        a_onnx = np.asarray(onnx_infer(obs), dtype=np.float32).reshape(-1)
        a_torch = np.asarray(torch_infer(obs), dtype=np.float32).reshape(-1)
        if a_onnx.shape != a_torch.shape:
            raise ValueError(
                f"Action shape mismatch: onnx={a_onnx.shape} torch={a_torch.shape}"
            )
        diff = np.abs(a_onnx - a_torch)
        step_max = float(diff.max())
        per_step.append(step_max)
        running_sum += float(diff.mean())

    if not per_step:
        raise ValueError("verify_parity needs at least one observation")

    return ParityReport(
        steps_checked=len(per_step),
        max_abs_diff=max(per_step),
        mean_abs_diff=running_sum / len(per_step),
        per_step_max=tuple(per_step),
        tol=tol,
    )


def build_obs_from_parquet(
    parquet_path: str | Path,
    obs_builder: ObservationBuilder,
    pad_zeros: int = 0,
    max_steps: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield policy obs vectors reconstructed from a trajectory parquet.

    Uses the same ObservationBuilder as the deploy node so this gate
    catches obs-builder bugs too (not just model-export bugs).
    """
    reader = TrajectoryReader(parquet_path)
    n = len(reader) if max_steps is None else min(max_steps, len(reader))
    base_lin_vel = reader.column("base_lin_vel_body")
    base_ang_vel = reader.column("base_ang_vel_body")
    joint_pos = reader.column("joint_pos")
    joint_vel = reader.column("joint_vel")
    command_vel = reader.column("command_vel")
    base_quat = reader.column("base_quat")  # (N, 4) xyzw
    action = reader.column("action")

    last_action = np.zeros(len(obs_builder.joint_order), dtype=np.float32)
    for i in range(n):
        proprio = obs_builder.build(
            base_lin_vel=base_lin_vel[i].astype(np.float32),
            base_ang_vel=base_ang_vel[i].astype(np.float32),
            projected_gravity=_projected_gravity_from_quat_xyzw(base_quat[i]),
            velocity_command=command_vel[i].astype(np.float32),
            joint_pos=joint_pos[i].astype(np.float32),
            joint_vel=joint_vel[i].astype(np.float32),
            last_action=last_action,
        )
        if pad_zeros > 0:
            obs = np.concatenate(
                [proprio, np.zeros(pad_zeros, dtype=np.float32)], axis=-1
            )
        else:
            obs = proprio
        yield obs
        last_action = action[i].astype(np.float32)


def _projected_gravity_from_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    """World-frame gravity (0, 0, -1) rotated into the body frame.

    Matches Isaac Lab's ``mdp.projected_gravity`` observation term.
    """
    x, y, z, w = (float(v) for v in quat_xyzw)
    # Rotate (0, 0, -1) by the inverse quaternion (== conjugate for unit quats).
    # Inverse rotation of v by q is equivalent to rotating v by q_conj.
    # Direct closed-form: g_body = R(q)^T @ g_world.
    gx = 2.0 * (x * z - w * y) * -1.0
    gy = 2.0 * (y * z + w * x) * -1.0
    gz = (1.0 - 2.0 * (x * x + y * y)) * -1.0
    return np.asarray([gx, gy, gz], dtype=np.float32)


# -------------------- CLI plumbing ------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--deploy-cfg", type=Path, required=True)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--max-steps", type=int, default=200)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI plumbing
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    args = parse_args(argv)

    import yaml

    cfg = yaml.safe_load(args.deploy_cfg.read_text())
    onnx_path = Path(cfg["policy"]["onnx_path"])
    torch_path = Path(cfg["policy"]["torchscript_path"])
    pad_zeros = int(cfg.get("policy", {}).get("obs_pad_zeros", 0))

    builder = ObservationBuilder(
        JointOrder(tuple(cfg["joint_order"])), cfg["control"]["default_joint_pos"]
    )

    import onnxruntime as ort
    import torch

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    def onnx_infer(obs: np.ndarray) -> np.ndarray:
        out = sess.run(None, {in_name: obs[None, :].astype(np.float32)})[0]
        return np.asarray(out).reshape(-1)

    module = torch.jit.load(str(torch_path), map_location="cpu")
    module.eval()

    def torch_infer(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
            out = module(t)
            if isinstance(out, (tuple, list)):
                out = out[0]
        return out.cpu().numpy().reshape(-1)

    obs_iter = build_obs_from_parquet(
        args.parquet, builder, pad_zeros=pad_zeros, max_steps=args.max_steps
    )
    report = verify_parity(list(obs_iter), onnx_infer, torch_infer, tol=args.tol)

    logger.info(
        "Parity: steps=%d max_diff=%.3e mean_diff=%.3e tol=%.1e -> %s",
        report.steps_checked,
        report.max_abs_diff,
        report.mean_abs_diff,
        report.tol,
        "PASS" if report.passed else "FAIL",
    )
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
