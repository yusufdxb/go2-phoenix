"""Post-export sanity bench: feed a canonical-stand observation to the
exported policy and assert the action magnitude is small.

Motivation: the 2026-04-15 training shipped an under-trained flat-v0 policy
(error_vel_xy=0.76 m/s at final iter — ~8x the 0.1 m/s target). The tell at
deploy-time is large raw action magnitudes on a zero-command, default-pose
input: a converged Flat-v0 baseline should emit |action|_infinity well below
one-third of the per-step slew clip. If that's not true, do not ship the
export.

Split into a pure-numpy :func:`check_canonical_action` (testable in CI with
a fake inference function) and a thin CLI wrapper that loads the ONNX + the
deploy yaml. This is the same split used by :mod:`phoenix.sim2real.verify_deploy`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from phoenix.sim2real.observation import JointOrder, ObservationBuilder

logger = logging.getLogger("phoenix.sim2real.bench_export")

InferenceFn = Callable[[np.ndarray], np.ndarray]

DEFAULT_MAX_ABS_ACTION = 0.3


@dataclass(frozen=True)
class BenchReport:
    obs_dim: int
    action_dim: int
    max_abs_action: float
    threshold: float

    @property
    def passed(self) -> bool:
        return self.max_abs_action < self.threshold


def build_canonical_stand_obs(
    obs_builder: ObservationBuilder,
    pad_zeros: int = 0,
) -> np.ndarray:
    """Build the canonical-stand policy observation.

    Canonical stand: body at rest, standing upright, zero velocity command,
    joints at their default (sim reset) pose, no prior action.
    """
    n = len(obs_builder.joint_order)
    proprio = obs_builder.build(
        base_lin_vel=np.zeros(3, dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        projected_gravity=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        velocity_command=np.zeros(3, dtype=np.float32),
        joint_pos=obs_builder.default_q.copy(),
        joint_vel=np.zeros(n, dtype=np.float32),
        last_action=np.zeros(n, dtype=np.float32),
    )
    if pad_zeros > 0:
        return np.concatenate([proprio, np.zeros(pad_zeros, dtype=np.float32)])
    return proprio


def check_canonical_action(
    infer: InferenceFn,
    obs: np.ndarray,
    threshold: float = DEFAULT_MAX_ABS_ACTION,
) -> BenchReport:
    """Run ``infer`` once on ``obs`` and summarize the action magnitude."""
    action = np.asarray(infer(obs), dtype=np.float32).reshape(-1)
    if action.size == 0:
        raise ValueError("infer returned an empty action vector")
    return BenchReport(
        obs_dim=int(obs.shape[-1]),
        action_dim=int(action.size),
        max_abs_action=float(np.max(np.abs(action))),
        threshold=float(threshold),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--onnx", type=Path, default=None)
    p.add_argument("--deploy-cfg", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=DEFAULT_MAX_ABS_ACTION)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    args = parse_args(argv)

    import yaml

    cfg = yaml.safe_load(args.deploy_cfg.read_text())
    onnx_path = args.onnx or Path(cfg["policy"]["onnx_path"])
    pad_zeros = int(cfg.get("policy", {}).get("obs_pad_zeros", 0))

    builder = ObservationBuilder(
        JointOrder(tuple(cfg["joint_order"])), cfg["control"]["default_joint_pos"]
    )
    obs = build_canonical_stand_obs(builder, pad_zeros=pad_zeros)

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name

    def infer(x: np.ndarray) -> np.ndarray:
        out = session.run(None, {in_name: x[None, :].astype(np.float32)})[0]
        return np.asarray(out).reshape(-1)

    report = check_canonical_action(infer, obs, threshold=args.threshold)
    logger.info(
        "Canonical-stand bench: obs_dim=%d action_dim=%d |a|_inf=%.4f threshold=%.3f -> %s",
        report.obs_dim,
        report.action_dim,
        report.max_abs_action,
        report.threshold,
        "PASS" if report.passed else "FAIL",
    )
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
