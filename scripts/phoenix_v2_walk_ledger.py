#!/usr/bin/env python3
"""Append one walking-candidate record to ``results/phoenix_v2/walk_ledger.jsonl``.

Amendment 6.3: every candidate is recorded, including failures. A record holds the
training run's commit, resolved env and train configs (and their hashes), seed,
checkpoint sha256, deploy-contract version, the limiter each evaluation applied, and the
headline metrics of every evaluation summary passed in.

Usage::

    python scripts/phoenix_v2_walk_ledger.py --candidate W1 --run-dir checkpoints/phoenix-walk-w1/<stamp> \
        --checkpoint model_1499.pt --eval results/phoenix_v2/walk_dev/w1/nominal ... --note "..."
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

LEDGER = Path("results/phoenix_v2/walk_ledger.jsonl")
HEADLINE = (
    "walk2_success_rate",
    "walk2_failures_by_check",
    "walk2_mean_lin_err_m_s",
    "walk2_mean_yaw_err_rad_s",
    "walk2_mean_cmd_speed_m_s",
    "walk2_median_progress_ratio",
    "walk2_p05_min_base_height_m",
    "walk2_p95_max_joint_speed_rad_s",
    "walk2_mean_target_jump_fraction",
    "success_rate",
    "survival_rate",
    "fidelity_pass_rate",
    "altered_fraction",
    "rms_modification_rad",
    "raw_out_of_range_fraction",
    "effort_saturation_fraction",
    "attitude_violation_episode_rate",
    "mean_base_height_m",
)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", required=True, help="file name inside --run-dir")
    ap.add_argument("--eval", type=Path, nargs="*", default=[])
    ap.add_argument("--contract", default="v3")
    ap.add_argument("--note", default="")
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    a = ap.parse_args(argv)
    ckpt = a.run_dir / a.checkpoint
    import yaml  # noqa: PLC0415

    train = yaml.safe_load((a.run_dir / "train.yaml").read_text())
    rec = {
        "schema": "phoenix-v2-walk-ledger/v1",
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate": a.candidate,
        "run_dir": str(a.run_dir),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": _sha(ckpt),
        "train_yaml_sha256": _sha(a.run_dir / "train.yaml"),
        "env_resolved_sha256": _sha(a.run_dir / "env_resolved.yaml"),
        "train_config": train,
        "env_resolved": yaml.safe_load((a.run_dir / "env_resolved.yaml").read_text()),
        "seed": train["run"]["seed"],
        "contract": a.contract,
        "ledger_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "note": a.note,
        "evaluations": [],
    }
    for d in a.eval:
        s = json.loads((d / "summary.json").read_text())
        if s.get("checkpoint_sha256") != rec["checkpoint_sha256"]:
            raise SystemExit(f"{d}: evaluated a different checkpoint")
        rec["evaluations"].append({
            "path": str(d),
            "label": s.get("label"),
            "seed": s.get("seed"),
            "env_config": s.get("env_config"),
            "eval_commit": s.get("commit"),
            "eval_dirty": s.get("dirty"),
            "limiter": s.get("limiter"),
            **{k: s.get(k) for k in HEADLINE},
        })
    a.ledger.parent.mkdir(parents=True, exist_ok=True)
    with a.ledger.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ("candidate", "checkpoint_sha256", "seed")}
                     | {"evals": [(e["label"], e["walk2_success_rate"]) for e in rec["evaluations"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
