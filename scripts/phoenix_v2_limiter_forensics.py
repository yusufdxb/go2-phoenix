#!/usr/bin/env python3
"""Phase A: replay recorded GO2 policy requests through the three limiters.

Usage::

    PYTHONPATH=src python scripts/phoenix_v2_limiter_forensics.py \
        --run logs/payload_evidence_20260922/20260921_cc131039b662/F1_20260922T001048Z \
        [--run ...] --out results/phoenix_v2/limiter_offline \
        --rate-deltas 0.02 0.035 0.05 0.075 0.1 0.175

Writes one ``<run>.json`` per run, ``summary.json``, ``summary.csv`` and ``REPORT.md``.
See :mod:`phoenix.monitor.limiter_replay` for what the replay can and cannot say.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from phoenix.monitor.layers import _policy_to_motor, read_bridge_telemetry
from phoenix.monitor.limiter_replay import (
    SCHEMA,
    limiter_metrics,
    replay_input,
    verify_incumbent_replay,
    with_trained_action_clip,
)
from phoenix.sim2real.safety import MAX_DELTA_PER_STEP_RAD


def _q_node(path: Path, n_ticks: int) -> np.ndarray:
    """The ``q`` the policy node clipped against, per tick, motor order."""
    out = []
    with path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record", "tick") != "tick":
                continue
            pol = rec.get("policy") or {}
            if rec.get("mode") == "policy":
                out.append(_policy_to_motor(pol.get("q_policy")))
            else:
                out.append(np.full(12, np.nan))
    arr = np.vstack(out)
    assert arr.shape[0] == n_ticks, (arr.shape, n_ticks)
    return arr


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--incumbent-delta", type=float, default=MAX_DELTA_PER_STEP_RAD)
    ap.add_argument(
        "--rate-deltas", type=float, nargs="+", default=[0.02, 0.035, 0.05, 0.075, 0.1, 0.175]
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    rows, runs = [], {}
    for run in args.run:
        bridge = run / "bridge.jsonl"
        manifest = json.loads(bridge.open().readline())
        layers = read_bridge_telemetry(bridge)
        inp = replay_input(layers, _q_node(bridge, len(layers)))
        name = f"{run.parent.name}/{run.name}"
        res = {
            "schema": SCHEMA,
            "run": name,
            "bridge_sha256": _sha(bridge),
            "live": manifest.get("live"),
            "stage": manifest.get("stage"),
            "code_sha": (manifest.get("code_identity") or {}).get("sha"),
            "open_loop_caveat": (
                "Replay is open loop: recorded q was produced by the incumbent limiter. "
                "Other limiters' numbers describe the same request stream, not the "
                "robot's response to them."
            ),
            "incumbent_replay_check": verify_incumbent_replay(inp, args.incumbent_delta),
            "note_clip1": (
                "clip1+ rows measure against the TRAINED plant's request "
                "default + 0.25 * clip(raw, -1, 1); the others against the unclipped "
                "request the deploy node computed."
            ),
            "limiters": {},
        }
        configs = [("measured_q", args.incumbent_delta)]
        configs += [("prev_command", d) for d in args.rate_deltas]
        configs += [("hard_only", 0.0)]
        clipped = with_trained_action_clip(inp)
        raw = inp.raw_action
        res["raw_action_outside_trained_range_fraction"] = float((np.abs(raw) > 1.0).mean())
        res["raw_action_abs_max"] = float(np.abs(raw).max())
        variants = [("", inp)] + [("clip1+", clipped)]
        for (tag, src), (limiter, delta) in [(v, c) for v in variants for c in configs]:
            m = limiter_metrics(src, limiter, delta)
            m.pop("sent")
            key = tag + (limiter if limiter == "hard_only" else f"{limiter}@{delta:g}")
            res["limiters"][key] = m
            rows.append(
                {
                    "run": name,
                    "live": manifest.get("live"),
                    "limiter": key,
                    "ticks": m["ticks"],
                    "altered_fraction": round(m["altered_fraction"], 4),
                    "worst_joint_altered_fraction": round(
                        max(v["altered_fraction"] for v in m["per_joint"].values()), 4
                    ),
                    "rms_modification_rad": round(m["rms_modification_rad"], 4),
                    "max_modification_rad": round(m["max_modification_rad"], 4),
                    "distortion_D": round(m["distortion_D"], 4),
                    "limiter_reversals": m["limiter_reversals"],
                    "hard_envelope_clip_fraction": round(m["hard_envelope_clip_fraction"], 4),
                    "hard_limit_violations": m["hard_limit_violations"],
                    "abort_band_first_tick": m["abort_band_first_tick"],
                    "max_abs_target_rate_rad_s": round(m["max_abs_target_rate_rad_s"], 2),
                    "max_abs_pd_torque_demand_nm": round(m["max_abs_pd_torque_demand_nm"], 2),
                    "soft_clip_runs_ge3": m["soft_clip_runs_ge3"],
                    "soft_clip_runs_with_growing_gap": m["soft_clip_runs_with_growing_gap"],
                }
            )
        runs[name] = res
        (args.out / f"{run.parent.name}__{run.name}.json").write_text(
            json.dumps(res, indent=1)
        )
    summary = {"schema": SCHEMA, "commit": commit, "runs": list(runs), "rows": rows}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    with (args.out / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(
            f"{r['run']:<45} {r['limiter']:<20} alt={r['altered_fraction']:.3f} "
            f"worst={r['worst_joint_altered_fraction']:.3f} rms={r['rms_modification_rad']:.3f} "
            f"D={r['distortion_D']:.3f} rev={r['limiter_reversals']} "
            f"rate={r['max_abs_target_rate_rad_s']} abort@{r['abort_band_first_tick']}"
        )
    for name, res in runs.items():
        print(name, "incumbent replay:", res["incumbent_replay_check"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
