"""End-to-end deploy gate: does the on-robot path reproduce the studied one?

Everything upstream of this is a component check. This is the one that matters,
and it is the check that would have caught the class of bug that ruins sim-to-real
transfers: the shield can be perfectly calibrated and still be worthless if the
latent the *robot* computes lives in a slightly different space than the latent
the monitor was *fit* on. An action-only export parity check cannot see that —
the robot would walk correctly and the monitor would score nonsense.

So this replays real recorded observations from the Phase 3 rollouts through the
exported ONNX policy (the artifact the Jetson actually loads), and compares, in
order:

1. **Latent parity** — ONNX latent vs the latent recorded during the Isaac study.
2. **Score parity** — the OOD score from each, through the deployed float32
   monitor.
3. **Decision parity** — the only one with a pass/fail budget of zero: for every
   frame, do the two paths agree on whether it is above the trip threshold, and
   does the full arbiter produce an identical engage/disengage trace?

Fails closed and loudly on any decision disagreement.

Usage::

    PYTHONPATH=src $HOME/Sim/isaac-sim-venv/bin/python scripts/reliability_verify_deploy.py \
        --onnx deploy/stand_v3_latent.onnx --artifact deploy/shield_stand_v3.npz \
        --raw-dir reliability_eval/raw_stand
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

from phoenix.reliability.deploy import build_shield, load_artifact  # noqa: E402


def replay(shield, latents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the arbiter over a latent sequence; return (blend, raw score) traces."""
    shield.reset()
    blends = np.empty(len(latents), dtype=np.float64)
    scores = np.empty(len(latents), dtype=np.float64)
    for i, row in enumerate(latents):
        decision = shield.step(row)
        blends[i] = decision.blend
        scores[i] = decision.raw_score
    return blends, scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="deploy/stand_v3_latent.onnx")
    ap.add_argument("--artifact", default="deploy/shield_stand_v3.npz")
    ap.add_argument("--raw-dir", default="reliability_eval/raw_stand")
    ap.add_argument("--envs", type=int, default=16, help="Envs replayed per condition")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import onnxruntime as ort

    monitor, op, meta = load_artifact(args.artifact)
    print(f"[verify] artifact latent_dim={monitor.dim} trip={op.trip_threshold:.1f} K={op.trip_persistence}")

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in session.get_outputs()]
    if "latent" not in out_names:
        raise SystemExit(
            f"FAIL CLOSED: {args.onnx} has outputs {out_names} — no 'latent'. "
            "Re-export with `--emit-latent`; the shield cannot run on this policy."
        )
    onnx_latent_dim = session.get_outputs()[out_names.index("latent")].shape[-1]
    if isinstance(onnx_latent_dim, int) and onnx_latent_dim != monitor.dim:
        raise SystemExit(
            f"FAIL CLOSED: ONNX emits latent_dim={onnx_latent_dim}, artifact expects {monitor.dim}."
        )

    results = {}
    worst_latent = worst_score = 0.0
    total_decision_mismatch = 0
    total_blend_mismatch = 0
    total_frames = 0

    for path in sorted(glob.glob(f"{args.raw_dir}/*.npz")):
        name = Path(path).stem
        data = np.load(path)
        obs = data["obs"][:, : args.envs, :]
        ref_latent = data["latent"][:, : args.envs, :]
        steps, envs, obs_dim = obs.shape

        flat_obs = np.ascontiguousarray(obs.reshape(-1, obs_dim), dtype=np.float32)
        onnx_latent = session.run(["latent"], {"obs": flat_obs})[0]
        flat_ref = ref_latent.reshape(-1, ref_latent.shape[-1])

        latent_diff = float(np.max(np.abs(onnx_latent - flat_ref)))
        s_onnx = monitor.score(onnx_latent)
        s_ref = monitor.score(flat_ref)
        finite = np.isfinite(s_onnx) & np.isfinite(s_ref)
        score_diff = float(
            np.max(np.abs(s_onnx[finite] - s_ref[finite]) / np.maximum(np.abs(s_ref[finite]), 1e-9))
        )
        decision_mismatch = int(
            np.sum((s_onnx > op.trip_threshold) != (s_ref > op.trip_threshold))
        )

        # Full arbiter trace parity, per env, on the real time ordering.
        shield_a, _, _ = build_shield(args.artifact)
        shield_b, _, _ = build_shield(args.artifact)
        blend_mismatch = 0
        onnx_seq = onnx_latent.reshape(steps, envs, -1)
        for n in range(envs):
            b_onnx, _ = replay(shield_a, onnx_seq[:, n, :])
            b_ref, _ = replay(shield_b, ref_latent[:, n, :])
            blend_mismatch += int(np.sum(b_onnx != b_ref))

        frames = steps * envs
        total_frames += frames
        total_decision_mismatch += decision_mismatch
        total_blend_mismatch += blend_mismatch
        worst_latent = max(worst_latent, latent_diff)
        worst_score = max(worst_score, score_diff)
        results[name] = {
            "frames": frames,
            "max_latent_abs_diff": latent_diff,
            "max_score_rel_diff": score_diff,
            "trip_decision_mismatch": decision_mismatch,
            "arbiter_blend_mismatch": blend_mismatch,
        }
        print(
            f"[verify] {name:<22} latent<={latent_diff:.2e}  score_rel<={score_diff:.2e}  "
            f"trip_mismatch={decision_mismatch}  blend_mismatch={blend_mismatch}"
        )

    summary = {
        "onnx": args.onnx,
        "artifact": args.artifact,
        "raw_dir": args.raw_dir,
        "total_frames": total_frames,
        "max_latent_abs_diff": worst_latent,
        "max_score_rel_diff": worst_score,
        "trip_decision_mismatch": total_decision_mismatch,
        "arbiter_blend_mismatch": total_blend_mismatch,
        "trip_threshold": op.trip_threshold,
        "per_condition": results,
    }
    out_path = Path(args.out) if args.out else Path(str(args.artifact) + ".verify.json")
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[verify] wrote {out_path}")
    print(
        f"[verify] {total_frames} frames | worst latent diff {worst_latent:.2e} | "
        f"worst score rel diff {worst_score:.2e}"
    )

    if total_decision_mismatch or total_blend_mismatch:
        print(
            f"[verify] FAIL: {total_decision_mismatch} trip-decision and "
            f"{total_blend_mismatch} arbiter-blend disagreements between the exported "
            "policy and the studied one. Do not deploy."
        )
        return 1
    print("[verify] PASS: the exported on-robot path reproduces the studied shield exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
