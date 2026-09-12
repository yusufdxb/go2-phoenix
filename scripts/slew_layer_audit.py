"""Re-derive slew-clip figures from a logged capture, under every definition in use.

For each parquet it prints:

* the LEGACY figure (``legacy_raw_action_delta_saturation_rate``, consecutive raw
  action deltas against 0.175), which is what every sim slew percentage before
  2026-09-11 used;
* the deploy-equivalent policy-node clip activation (target against the measured
  joint position at the same tick), which is what the deploy path physically does;
* what a second, lagged clip does to the already clipped command
  (:func:`phoenix.training.slew.clip_layer_audit`), which is the question the
  policy node plus LowCmd bridge double limiter raises.

The captures do not store the target, so it is rebuilt as
``default_q + action_scale * action`` from ``--deploy-cfg``. A capture recorded
before the hip-pose fix ran with every hip at 0.0; pass ``--hips-zero`` to reproduce
the pose that run actually used.

Usage:
    python3 scripts/slew_layer_audit.py --deploy-cfg configs/sim2real/deploy_stand_v2.yaml \\
        --hips-zero data/failures/gate7_live_2026-04-21_18-33-17.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phoenix.replay.trajectory_reader import TrajectoryReader  # noqa: E402
from phoenix.sim2real.safety import MAX_DELTA_PER_STEP_RAD  # noqa: E402
from phoenix.training.slew import (  # noqa: E402
    clip_layer_audit,
    legacy_raw_action_delta_saturation_rate,
)


def audit_parquet(path: Path, cfg: dict, hips_zero: bool) -> dict:
    reader = TrajectoryReader(path)
    q = reader.column("joint_pos").astype(np.float64)
    actions = reader.column("action").astype(np.float64)
    t = reader.column("timestamp_s").astype(np.float64)
    default_q = np.asarray(
        [cfg["control"]["default_joint_pos"][n] for n in cfg["joint_order"]], dtype=np.float64
    )
    if hips_zero:
        default_q[[i for i, n in enumerate(cfg["joint_order"]) if n.endswith("hip_joint")]] = 0.0
    legacy = float(
        np.mean(
            [
                legacy_raw_action_delta_saturation_rate(
                    actions[i - 1 : i], actions[i : i + 1], MAX_DELTA_PER_STEP_RAD
                )
                for i in range(1, len(actions))
            ]
        )
    )
    audit = clip_layer_audit(
        actions=actions,
        measured_q=q,
        default_q=default_q,
        action_scale=float(cfg["control"]["action_scale"]),
    )
    return {
        "parquet": str(path),
        "rows": int(len(reader)),
        "rate_hz": float((len(t) - 1) / (t[-1] - t[0])) if len(t) > 1 else None,
        "hips_zero": hips_zero,
        "legacy_raw_action_delta_pct": 100.0 * legacy,
        **audit,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("parquets", nargs="+", type=Path)
    p.add_argument("--deploy-cfg", type=Path, required=True)
    p.add_argument("--hips-zero", action="store_true")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args(argv)
    cfg = yaml.safe_load(args.deploy_cfg.read_text())
    results = [audit_parquet(path, cfg, args.hips_zero) for path in args.parquets]
    for r in results:
        print(
            f"{r['parquet']}: rows={r['rows']} rate={r['rate_hz']:.1f} Hz  "
            f"legacy={r['legacy_raw_action_delta_pct']:.2f}%  "
            f"policy-layer clip={r['policy_layer_pct']:.2f}%  "
            f"second-layer (lag {r['lag_steps']})={r['second_layer_pct']:.2f}% "
            f"(where first did not: {r['second_only_pct']:.2f}%, "
            f"mean adjust {r['mean_second_adjust_rad'] * 1e3:.2f} mrad)  "
            f"end-to-end={r['end_to_end_pct']:.2f}%  "
            f"joint step median {r['median_joint_step_rad'] * 1e3:.2f} mrad"
        )
    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
