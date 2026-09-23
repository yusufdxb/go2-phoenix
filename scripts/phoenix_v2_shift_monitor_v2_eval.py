#!/usr/bin/env python3
"""Calibrate detector v2 on nominal development sessions and apply the frozen gate.

EXPERIMENT.md amendment 17. Applied ONCE, to fresh seeds that took no part in detector
development or in the failed v1 gate.

    PYTHONPATH=src python scripts/phoenix_v2_shift_monitor_v2_eval.py \
        --calibration <dir> --nominal <dir> \
        --degraded <dir>:rear:0.70 [--degraded <dir>:front:0.70 ...] \
        --gate-condition <name> --out <dir>

``--degraded`` takes ``path:true_group:applied_scale`` so the truth is declared on the
command line rather than inferred from a directory name. ``--gate-condition`` names the
one condition the pass/fail criteria are read from; every other degraded set is reported
as a declared secondary and cannot change the verdict.

Calibration and nominal validation sets must be disjoint; overlapping them is refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from phoenix.monitor.layers import read_bridge_telemetry, tracking_pairs
from phoenix.monitor.residual import calibrate, window_stats
from phoenix.monitor.response_shift_v2 import V2_GROUPS, assess_v2, calibrate_v2

np.seterr(all="ignore")

# The frozen gate, amendment 15.5, unchanged for v2.
GATE = {
    "max_false_flag": 0.05,
    "min_detection": 0.80,
    "min_correct_group": 0.70,
    "max_severity_bias": 0.10,
    "min_range_coverage": 0.70,
}


def sessions(root: Path, seeds: list[str] | None = None) -> list[Path]:
    out = sorted(root.glob("seed*/bridge/robot*.jsonl"))
    if seeds:
        want = {s if s.startswith("seed") else f"seed{s}" for s in seeds}
        out = [p for p in out if p.parent.parent.name in want]
    return out


def stats_for(paths: list[Path]):
    return [window_stats(tracking_pairs(read_bridge_telemetry(p))) for p in paths]


def score(paths, base, v2, true_group=None, applied=None):
    rows = []
    for p, st in zip(paths, stats_for(paths), strict=True):
        r = assess_v2(st, base, v2)
        rows.append(
            {
                "session": str(p),
                **r.to_dict(),
                "group_correct": bool(true_group is not None and r.group == true_group),
            }
        )
    n = max(len(rows), 1)
    det = [r for r in rows if r["shifted"]]
    sev = [r["severity"] for r in det if r["severity"] is not None]
    cov = [
        r
        for r in det
        if r["range"] and applied is not None and r["range"][0] <= applied <= r["range"][1]
    ]
    return {
        "n": len(rows),
        "true_group": true_group,
        "applied_scale": applied,
        "flagged_rate": len(det) / n,
        "detection_rate": len(det) / n,
        "correct_group_rate": sum(r["group_correct"] for r in rows) / n,
        "unresolved_rate": sum(1 for r in det if r["group"] is None) / n,
        "severity_median": float(np.median(sev)) if sev else None,
        "severity_bias": (float(np.median(sev)) - applied) if (sev and applied) else None,
        "range_coverage": (len(cov) / len(det)) if det else 0.0,
        "reported_groups": {
            g: sum(1 for r in det if r["group"] == g) for g in list(V2_GROUPS) + [None]
        },
        "sessions": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--nominal", type=Path, required=True)
    ap.add_argument("--degraded", action="append", default=[], help="path:group:scale")
    ap.add_argument("--gate-condition", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="GATE-v2")
    a = ap.parse_args()

    cal, nom = sessions(a.calibration), sessions(a.nominal)
    if set(map(str, cal)) & set(map(str, nom)):
        print("calibration and nominal sets overlap", file=sys.stderr)
        return 2
    if len(cal) < 4 or len(nom) < 4:
        print("need at least 4 sessions in each set", file=sys.stderr)
        return 1

    base = calibrate(stats_for(cal), regime="walk")
    v2 = calibrate_v2(stats_for(cal), base, regime="walk", sessions=[str(p) for p in cal])

    nom_row = score(nom, base, v2)
    out = {
        "schema": "phoenix-v2-shift-monitor-v2-eval/v1",
        "label": a.label,
        "gate": GATE,
        "baseline": v2.to_dict(),
        "calibration_sessions": [str(p) for p in cal],
        "nominal": nom_row,
        "conditions": {},
        "gate_condition": a.gate_condition,
    }
    print(f"[{a.label}] calibrated on {len(cal)} nominal sessions; {len(nom)} held-out nominal")
    print(f"  stage1 tau_global {v2.tau_global:.4f}   stage2 tau_fw {v2.tau_fw:.3f}")
    print(f"  nominal false-flag rate {nom_row['flagged_rate']:.4f}")
    if nom_row["flagged_rate"]:
        print(f"    reported: { {k:v for k,v in nom_row['reported_groups'].items() if v} }")

    for spec in a.degraded:
        path, grp, scale = spec.rsplit(":", 2)
        r = score(sessions(Path(path)), base, v2, grp, float(scale))
        name = Path(path).name
        out["conditions"][name] = r
        sev = f"{r['severity_median']:.3f}" if r["severity_median"] is not None else "n/a"
        print(
            f"  {name:12s} true={grp:6s}@{scale}  detect {r['detection_rate']:.3f}  "
            f"correct-group {r['correct_group_rate']:.3f}  severity {sev}  "
            f"cover {r['range_coverage']:.3f}  reported "
            f"{ {k:v for k,v in r['reported_groups'].items() if v} }"
        )

    g = out["conditions"].get(a.gate_condition)
    if g is None:
        print(f"gate condition {a.gate_condition!r} not among the degraded sets", file=sys.stderr)
        return 3
    checks = {
        "G1_nominal_false_flag<=0.05": (
            nom_row["flagged_rate"],
            GATE["max_false_flag"],
            nom_row["flagged_rate"] <= GATE["max_false_flag"],
        ),
        "G2_detection>=0.80": (
            g["detection_rate"],
            GATE["min_detection"],
            g["detection_rate"] >= GATE["min_detection"],
        ),
        "G3_correct_group>=0.70": (
            g["correct_group_rate"],
            GATE["min_correct_group"],
            g["correct_group_rate"] >= GATE["min_correct_group"],
        ),
        "G4a_severity_bias<=0.10": (
            abs(g["severity_bias"]) if g["severity_bias"] is not None else float("inf"),
            GATE["max_severity_bias"],
            g["severity_bias"] is not None and abs(g["severity_bias"]) <= GATE["max_severity_bias"],
        ),
        "G4b_range_coverage>=0.70": (
            g["range_coverage"],
            GATE["min_range_coverage"],
            g["range_coverage"] >= GATE["min_range_coverage"],
        ),
    }
    passed = all(v[2] for v in checks.values())
    out["gate_result"] = {k: {"value": v[0], "bar": v[1], "pass": v[2]} for k, v in checks.items()}
    out["passed"] = passed

    print(f"\n=== FROZEN GATE on '{a.gate_condition}' ===")
    for k, (val, bar, ok) in checks.items():
        print(f"  {k:30s} {val:.4f}  bar {bar:.2f}   {'PASS' if ok else 'FAIL'}")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "shift_monitor_v2.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}/shift_monitor_v2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
