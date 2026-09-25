"""Rank the eval_sweep.sh results and print a table."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SWEEP_DIR = Path("logs/eval_sweep")


def main() -> int:
    rows = []
    for f in sorted(SWEEP_DIR.glob("seed*_iter*.json")):
        m = re.match(r"seed(\d+)_iter(\d+)\.json", f.name)
        if not m:
            continue
        seed, it = int(m.group(1)), int(m.group(2))
        data = json.loads(f.read_text())
        vt = data["velocity_tracking"]
        lin_errs = [vt[k]["mean_lin_err_mps"] for k in ("stand", "slow", "fast") if vt[k]["mean_lin_err_mps"] is not None]
        yaw_errs = [vt[k]["mean_yaw_err_rad_s"] for k in ("stand", "slow", "fast") if vt[k]["mean_yaw_err_rad_s"] is not None]
        mean_lin_err = sum(lin_errs) / len(lin_errs) if lin_errs else float("nan")
        mean_yaw_err = sum(yaw_errs) / len(yaw_errs) if yaw_errs else float("nan")
        headroom = data["torque_headroom"]
        max_headroom = max(v["peak_over_limit"] for v in headroom.values())
        rows.append(
            {
                "seed": seed,
                "iter": it,
                "checkpoint": data["checkpoint"],
                "fall_rate": data["fall_rate"],
                "mean_lin_err": mean_lin_err,
                "mean_yaw_err": mean_yaw_err,
                "stand_height": data["standing_base_height_m"]["mean"],
                "joint_near_limit": data["joint_near_limit_fraction"],
                "max_torque_over_limit": max_headroom,
                "action_clip_rate": data["action_clip_rate"],
            }
        )

    # Score: lower is better. fall_rate dominates (x10), then tracking error,
    # then a small penalty for torque headroom / joint-limit proximity.
    def score(r: dict) -> float:
        h = r["stand_height"] if r["stand_height"] == r["stand_height"] else 0.0  # nan check
        height_penalty = abs(h - 0.30) if h else 1.0
        return (
            10.0 * r["fall_rate"]
            + r["mean_lin_err"]
            + r["mean_yaw_err"]
            + r["joint_near_limit"]
            + max(0.0, r["max_torque_over_limit"] - 0.9)
            + 0.5 * height_penalty
        )

    for r in rows:
        r["score"] = score(r)
    rows.sort(key=lambda r: r["score"])

    header = f"{'rank':4} {'seed':4} {'iter':5} {'fall':6} {'lin_err':8} {'yaw_err':8} {'height':7} {'jnt_lim':8} {'torq%':6} {'clip%':6} {'score':7}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows, 1):
        print(
            f"{i:4} {r['seed']:4} {r['iter']:5} {r['fall_rate']:6.3f} {r['mean_lin_err']:8.4f} "
            f"{r['mean_yaw_err']:8.4f} {r['stand_height']:7.3f} {r['joint_near_limit']:8.4f} "
            f"{r['max_torque_over_limit']*100:6.1f} {r['action_clip_rate']*100:6.1f} {r['score']:7.4f}"
        )

    print()
    print("Top 2 per seed:")
    for seed in sorted({r["seed"] for r in rows}):
        seed_rows = sorted([r for r in rows if r["seed"] == seed], key=lambda r: r["score"])
        for r in seed_rows[:2]:
            print(f"  seed{seed} iter{r['iter']}: {r['checkpoint']} (score {r['score']:.4f})")

    if rows:
        best = rows[0]
        print()
        print(f"BEST OVERALL: seed{best['seed']} iter{best['iter']}: {best['checkpoint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
