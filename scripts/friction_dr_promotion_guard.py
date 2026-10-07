"""Compute the friction-DR prereg promotion guards exactly as written (post-study, no reinterpretation).

docs/preregistration/2026-10-07-joint-friction-dr.md, "Promotion guards": a T checkpoint can be
proposed only if no fall in any replay condition, frictionless (`nominal`) replay within 0.10 of the
paired C on vx03_4s and yaw05_3s, sim2sim gate v3 PASS, no health-guard abort. "Within 0.10" is
evaluated as written: |T - C| <= 0.10. Reads only committed study outputs.

    python3 scripts/friction_dr_promotion_guard.py   # writes results/friction_dr/promotion_guard.json
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/friction_dr"
SEEDS = (50, 51, 52, 53, 54)


def _rows(label: str) -> dict:
    rows = json.loads((OUT / label / "replay" / "fric_results.json").read_text())
    return {(r["label"], r["preset"]): r for r in rows}, any(r["fell"] for r in rows)


def main() -> int:
    report = {
        "rule": "nominal |T-C| <= 0.10 on vx03_4s fwd and yaw05_3s yaw; no falls; gate v3 PASS; no health abort",
        "seeds": {},
    }
    for s in SEEDS:
        t, t_fell = _rows(f"fric_T_s{s}")
        c, _ = _rows(f"fric_C_s{s}")
        d_fwd = t[("nominal", "vx03_4s")]["fwd_m"] / 1.2 - c[("nominal", "vx03_4s")]["fwd_m"] / 1.2
        d_yaw = (
            t[("nominal", "yaw05_3s")]["yaw_deg"] / 86.0
            - c[("nominal", "yaw05_3s")]["yaw_deg"] / 86.0
        )
        gate = json.loads((OUT / f"fric_T_s{s}" / "gate_v3" / "gate_report.json").read_text())[
            "verdict"
        ]
        abort = any((REPO / "checkpoints/friction_dr" / f"fric_T_s{s}").glob("health_abort.json"))
        checks = {
            "no_fall": not t_fell,
            "nominal_fwd_within_0p10": abs(d_fwd) <= 0.10,
            "nominal_yaw_within_0p10": abs(d_yaw) <= 0.10,
            "gate_v3_pass": gate == "PASS",
            "no_health_abort": not abort,
        }
        report["seeds"][s] = {
            "d_nominal_fwd": d_fwd,
            "d_nominal_yaw": d_yaw,
            "checks": checks,
            "pass": all(checks.values()),
        }
    report["any_seed_passes"] = any(v["pass"] for v in report["seeds"].values())
    (OUT / "promotion_guard.json").write_text(json.dumps(report, indent=2))
    for s, v in report["seeds"].items():
        print(
            s,
            "PASS" if v["pass"] else "FAIL",
            f"d_fwd {v['d_nominal_fwd']:+.3f} d_yaw {v['d_nominal_yaw']:+.3f}",
            [k for k, ok in v["checks"].items() if not ok],
        )
    print("any seed passes:", report["any_seed_passes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
