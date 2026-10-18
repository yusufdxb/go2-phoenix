"""Check kinematic leg odometry against tape on walks that have both.

    python3 scripts/leg_odometry_validate.py --out results/.../leg_odometry_validation.json \\
        RUN_DIR=TAPE_FWD_M[:TAPE_LAT_M] ...

A run directory holding operator_measurement.json (the free-floor harness writes it) may be
passed without =TAPE; its forward_cm / lateral_cm are used. Reports, per run, the leg-odometry
rear-feet forward displacement (same convention as the tape), its ratio to tape, and a
leave-one-out check: the scale fitted on the other runs, applied to this one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from phoenix.sim2real import leg_odometry as lo  # noqa: E402


def parse(spec: str) -> tuple[Path, float, float | None, str]:
    if "=" in spec:
        d, v = spec.split("=", 1)
        fwd, _, lat = v.partition(":")
        return Path(d), float(fwd), float(lat) if lat else None, "command line"
    m = json.loads((Path(spec) / "operator_measurement.json").read_text())
    lat = m.get("lateral_cm")
    return Path(spec), m["forward_cm"] / 100, None if lat is None else lat / 100, f"operator_measurement.json ({m.get('by')})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rows = []
    for spec in args.runs:
        d, tape, lat, src = parse(spec)
        e = lo.estimate_run(d)
        rows.append({"run_dir": str(d), "tape_fwd_m": tape, "tape_lat_m": lat, "tape_source": src,
                     "odom_rear_fwd_m": e["rear_feet_forward_m"], "odom_rear_lat_m": e["rear_feet_lateral_m"],
                     "odom_yaw_deg": e["yaw_deg"], "ratio": e["rear_feet_forward_m"] / tape})
    ratios = np.array([r["ratio"] for r in rows])
    for i, r in enumerate(rows):
        others = np.delete(ratios, i)
        if len(others):
            k = float(others.mean())
            r["loo_scale"] = k
            r["loo_pred_fwd_m"] = r["odom_rear_fwd_m"] / k
            r["loo_err_m"] = r["loo_pred_fwd_m"] - r["tape_fwd_m"]
            r["loo_err_frac"] = r["loo_err_m"] / r["tape_fwd_m"]
    doc = {
        "schema": "phoenix-leg-odometry-validation/v1",
        "estimator": "phoenix.sim2real.leg_odometry (diagonal-pair stance, no slip, IMU attitude)",
        "runs": rows,
        "ratio_mean": float(ratios.mean()),
        "ratio_std": float(ratios.std(ddof=1)) if len(ratios) > 1 else None,
        "loo_abs_err_frac_max": max((abs(r["loo_err_frac"]) for r in rows if "loo_err_frac" in r), default=None),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    for r in rows:
        print(f"{Path(r['run_dir']).name:34s} tape {r['tape_fwd_m']:.3f}  odom {r['odom_rear_fwd_m']:.3f}  ratio {r['ratio']:.3f}  "
              f"LOO err {r.get('loo_err_m', float('nan')):+.3f} m ({100 * r.get('loo_err_frac', float('nan')):+.1f} %)")
    print(f"ratio mean {doc['ratio_mean']:.3f} std {doc['ratio_std']:.3f}; LOO max |err| {100 * doc['loo_abs_err_frac_max']:.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
