"""Parity gate for an exported Phoenix policy, against the SOURCE checkpoint.

Why this exists alongside ``phoenix.sim2real.verify_deploy``
-----------------------------------------------------------
``verify_deploy`` compares ONNX against the TorchScript fallback. Both are
produced by the same ``_ExportablePolicy`` wrapper in the same export run and
held in the same process, so neither side re-reads the checkpoint and neither
side re-runs the graph export. It therefore cannot see a shipped ``.onnx`` that
stopped matching the checkpoint it claims to come from.

This gate re-derives the torch side in a fresh process, straight from
``latest.pt``, and compares the SHIPPED ONNX file against that. What it catches:

* a ``policy.onnx`` or ``policy.onnx.data`` that no longer corresponds to the
  checkpoint next to it, whether from a stale export, a partial rebuild, or a
  corrupted transfer;
* a ``torch.onnx.export`` graph bug, since the ONNX side is the exported graph
  and the torch side never goes through it;
* an observation layout that drifted away from what the deploy node emits,
  because the batches are rebuilt through :class:`ObservationBuilder`.

What it does NOT catch, stated plainly: the torch side calls the same
``_extract_actor_state_dict``, ``_build_actor_mlp``, ``_load_actor_weights`` and
``_ExportablePolicy`` helpers the export path calls. A bug INSIDE those shared
helpers (normalizer stats read from the wrong buffer, a systematically wrong
layer mapping) is present identically on both sides here too, and cancels. Only
an independent reimplementation of the forward pass would close that, and this
gate is not one. Treat it as a shipped-artifact gate, not a proof of the
wrapper.

It also adds two things ``verify_deploy`` does not report:

* cosine similarity per output, both globally and as the worst single-sample
  value. A global cosine over a flattened batch is dominated by the bulk of the
  rows and can sit at 1.0 while one observation diverges badly, which for a
  control policy is exactly the sample that matters.
* a sha256 of every shipped file including ``policy.onnx.data``. The external
  data file carries all the weights, so a manifest that omits it cannot prove a
  transfer was intact.

Observation batches are REAL, never random: reconstructed from logged rollout
parquets through the same :class:`ObservationBuilder` the deploy node uses, and
from recorded sim rollout observations. That means this gate also fails if the
observation layout stops matching what the node emits.

Lazy-torch convention: no module-level torch or onnxruntime import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

logger = logging.getLogger("phoenix.parity_gate")

# The repo's historical bar, from checkpoints/*/export_report.txt.
DEFAULT_MAX_ABS_TOL = 1e-5
DEFAULT_COS_TOL = 0.9999


@dataclass
class OutputParity:
    name: str
    max_abs: float
    mean_abs: float
    cos_global: float
    cos_worst_sample: float
    max_abs_tol: float
    cos_tol: float

    @property
    def passed(self) -> bool:
        return (
            self.max_abs <= self.max_abs_tol
            and self.cos_global >= self.cos_tol
            and self.cos_worst_sample >= self.cos_tol
        )

    def failures(self) -> list[str]:
        out = []
        if self.max_abs > self.max_abs_tol:
            out.append(f"max_abs {self.max_abs:.3e} exceeds {self.max_abs_tol:.1e}")
        if self.cos_global < self.cos_tol:
            out.append(f"cos_global {self.cos_global:.9f} below {self.cos_tol}")
        if self.cos_worst_sample < self.cos_tol:
            out.append(f"cos_worst_sample {self.cos_worst_sample:.9f} below {self.cos_tol}")
        return out


@dataclass
class BatchParity:
    source: str
    n_samples: int
    outputs: list[OutputParity] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(o.passed for o in self.outputs)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flattened arrays.

    Returns 1.0 when both are all-zero (identical), which is the correct
    verdict for a parity check even though the cosine is undefined there.
    """
    af = np.asarray(a, dtype=np.float64).ravel()
    bf = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(af))
    nb = float(np.linalg.norm(bf))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(af, bf) / (na * nb))


def compare(
    name: str,
    ref: np.ndarray,
    got: np.ndarray,
    *,
    max_abs_tol: float,
    cos_tol: float,
) -> OutputParity:
    """Compare a reference output tensor against the candidate, per output."""
    ref = np.asarray(ref, dtype=np.float32)
    got = np.asarray(got, dtype=np.float32)
    if ref.shape != got.shape:
        raise ValueError(f"output '{name}' shape mismatch: ref={ref.shape} got={got.shape}")
    diff = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    worst = min(cosine(ref[i], got[i]) for i in range(ref.shape[0]))
    return OutputParity(
        name=name,
        max_abs=float(diff.max()),
        mean_abs=float(diff.mean()),
        cos_global=cosine(ref, got),
        cos_worst_sample=worst,
        max_abs_tol=max_abs_tol,
        cos_tol=cos_tol,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shipped_manifest(checkpoint: Path, onnx: Path) -> dict[str, dict]:
    """sha256 and size of every file a deploy transfer has to carry intact.

    ``policy.onnx.data`` is the external-data sidecar and holds all the weights,
    so a manifest that lists only ``policy.onnx`` cannot prove a transfer was
    intact. The TorchScript fallback is included when present because the deploy
    node falls back to it.
    """

    candidates = [checkpoint, onnx, onnx.with_suffix(onnx.suffix + ".data")]
    fallback = onnx.with_name("policy.pt")
    if fallback != checkpoint:
        candidates.append(fallback)
    manifest: dict[str, dict] = {}
    for path in candidates:
        if path.exists():
            manifest[str(path)] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    return manifest


# ---------------------------------------------------------------- real batches


def obs_from_parquet(parquet: Path, cfg: dict, max_steps: int) -> np.ndarray:
    """Real logged rollout -> the exact obs vector the deploy node would emit."""
    from phoenix.sim2real.observation import JointOrder, ObservationBuilder
    from phoenix.sim2real.verify_deploy import build_obs_from_parquet

    builder = ObservationBuilder(
        JointOrder(tuple(cfg["joint_order"])), cfg["control"]["default_joint_pos"]
    )
    pad = int(cfg.get("policy", {}).get("obs_pad_zeros", 0))
    rows = list(build_obs_from_parquet(parquet, builder, pad_zeros=pad, max_steps=max_steps))
    return np.stack(rows).astype(np.float32)


def obs_from_npz(npz: Path, key: str, max_steps: int) -> np.ndarray:
    """Recorded sim rollout observations, already in policy obs layout."""
    data = np.load(npz, allow_pickle=True)
    arr = np.asarray(data[key], dtype=np.float32)
    arr = arr.reshape(-1, arr.shape[-1])
    return arr[:max_steps]


# ---------------------------------------------------------------- the two sides


def build_reference_policy(checkpoint: Path, device: str = "cpu") -> tuple[Any, int, int]:
    """Rebuild the actor from the rsl_rl checkpoint, independent of the export run."""
    import torch

    from phoenix.sim2real import export as ex

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    actor_sd = ex._extract_actor_state_dict(ckpt)
    layer_keys = sorted(
        (k for k in actor_sd if k.endswith(".weight") and ("mlp" in k or "actor" in k)),
        key=ex._layer_index,
    )
    if not layer_keys:
        raise RuntimeError(f"No MLP weights in actor state_dict. Keys: {list(actor_sd)}")
    obs_dim = int(actor_sd[layer_keys[0]].shape[1])
    action_dim = int(actor_sd[layer_keys[-1]].shape[0])
    hidden = [int(actor_sd[k].shape[0]) for k in layer_keys[:-1]]

    actor = ex._build_actor_mlp(obs_dim, action_dim, hidden)
    ex._load_actor_weights(actor, actor_sd, layer_keys)
    actor.to(device).eval()
    policy = ex._ExportablePolicy(actor, ckpt, actor_sd, tap_indices=None).to(device).eval()
    if policy.normalizer is None:
        raise RuntimeError(
            "checkpoint has no obs normalizer; the exported policy would see raw observations"
        )
    return policy, obs_dim, action_dim


def run_torch(policy: Any, obs: np.ndarray) -> np.ndarray:
    import torch

    with torch.inference_mode():
        out = policy(torch.from_numpy(obs))
    if isinstance(out, tuple):
        out = out[0]
    return out.cpu().numpy()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parity-gate an exported policy vs its checkpoint.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--deploy-cfg", type=Path, required=True)
    p.add_argument("--parquet", type=Path, action="append", default=[])
    p.add_argument("--npz", type=Path, action="append", default=[])
    p.add_argument("--npz-key", type=str, default="onset_obs")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--tol", type=float, default=DEFAULT_MAX_ABS_TOL)
    p.add_argument("--cos-tol", type=float, default=DEFAULT_COS_TOL)
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    args = parse_args(argv)

    import onnxruntime as ort
    import yaml

    cfg = yaml.safe_load(args.deploy_cfg.read_text())

    policy, obs_dim, action_dim = build_reference_policy(args.checkpoint)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    out_names = [o.name for o in session.get_outputs()]

    onnx_in_dim = session.get_inputs()[0].shape[-1]
    logger.info(
        "checkpoint obs_dim=%d action_dim=%d | onnx in '%s'%s out %s",
        obs_dim,
        action_dim,
        in_name,
        session.get_inputs()[0].shape,
        [(o.name, o.shape) for o in session.get_outputs()],
    )
    if isinstance(onnx_in_dim, int) and onnx_in_dim != obs_dim:
        logger.error("ONNX input dim %s != checkpoint obs_dim %d", onnx_in_dim, obs_dim)
        return 2

    # The node's contract: 48 proprio dims + configured padding.
    pad = int(cfg.get("policy", {}).get("obs_pad_zeros", 0))
    node_dim = 48 + pad
    if isinstance(onnx_in_dim, int) and node_dim != onnx_in_dim:
        logger.error(
            "SHAPE CONTRACT FAIL: %s emits 48+%d=%d but ONNX expects %s",
            args.deploy_cfg,
            pad,
            node_dim,
            onnx_in_dim,
        )
        return 2
    logger.info(
        "shape contract OK: node emits 48+%d=%d, onnx expects %s", pad, node_dim, onnx_in_dim
    )

    manifest = shipped_manifest(args.checkpoint, args.onnx)
    for path, entry in sorted(manifest.items()):
        logger.info("shipped %s  %d bytes  sha256=%s", path, entry["bytes"], entry["sha256"])
    if not any(k.endswith(".onnx.data") for k in manifest):
        logger.info("no policy.onnx.data sidecar; weights are inline in the onnx file")

    batches: list[tuple[str, np.ndarray]] = []
    for pq in args.parquet:
        batches.append((f"parquet:{pq}", obs_from_parquet(pq, cfg, args.max_steps)))
    for nz in args.npz:
        batches.append(
            (f"npz:{nz}[{args.npz_key}]", obs_from_npz(nz, args.npz_key, args.max_steps))
        )
    if not batches:
        logger.error("no real input batch given; refusing to gate on random data")
        return 2

    results: list[BatchParity] = []
    for source, obs in batches:
        if obs.shape[-1] != obs_dim:
            logger.error("batch %s has obs dim %d, expected %d", source, obs.shape[-1], obs_dim)
            return 2
        ref = run_torch(policy, obs)
        got = session.run(out_names, {in_name: obs})
        bp = BatchParity(source=source, n_samples=int(obs.shape[0]))
        ref_tuple = (ref,) if not isinstance(ref, tuple) else ref
        for name, r, g in zip(out_names, ref_tuple, got, strict=True):
            bp.outputs.append(compare(name, r, g, max_abs_tol=args.tol, cos_tol=args.cos_tol))
        results.append(bp)

    ok = True
    for bp in results:
        for o in bp.outputs:
            verdict = "PASS" if o.passed else "FAIL"
            logger.info(
                "%s n=%d out=%s max_abs=%.3e mean_abs=%.3e cos=%.9f cos_worst=%.9f -> %s",
                bp.source,
                bp.n_samples,
                o.name,
                o.max_abs,
                o.mean_abs,
                o.cos_global,
                o.cos_worst_sample,
                verdict,
            )
            if not o.passed:
                ok = False
                for f in o.failures():
                    logger.error("  %s: %s", o.name, f)

    if args.json_out:
        payload = {
            "checkpoint": str(args.checkpoint),
            "onnx": str(args.onnx),
            "deploy_cfg": str(args.deploy_cfg),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "tol_max_abs": args.tol,
            "tol_cos": args.cos_tol,
            "shipped_manifest": manifest,
            "passed": ok,
            "batches": [
                {
                    "source": b.source,
                    "n_samples": b.n_samples,
                    "outputs": [vars(o) for o in b.outputs],
                }
                for b in results
            ],
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        logger.info("wrote %s", args.json_out)

    logger.info("PARITY %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
