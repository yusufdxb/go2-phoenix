#!/usr/bin/env python3
"""Calibrate the group-shift monitor on nominal sessions and score it on degraded ones.

Phase I / Phase J. The SAME script runs the development pass (seeds 7001-7003) and the
frozen validation gate (fresh seeds); which it is depends only on the directories given
and is recorded in the output.

    PYTHONPATH=src python scripts/phoenix_v2_shift_monitor_eval.py \
        --calibration <dir> --nominal <dir> --degraded <dir> [--degraded <dir> ...] \
        --true-group all --out <dir> [--label dev|gate]

``--calibration`` and ``--nominal`` must be disjoint session sets: the first sets the
thresholds, the second measures the false-alarm rate against them. Passing the same
directory to both is refused, because a false-alarm rate measured on the sessions that
set the threshold is meaningless.

Reports, per condition: the fraction of sessions flagged, whether the selected group is
the one that was actually degraded, the severity estimate against the applied scale, and
the detection latency in windows. It applies no pass/fail verdict; the gate thresholds
live in the preregistration and are applied by the caller.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from phoenix.monitor.layers import read_bridge_telemetry, tracking_pairs
from phoenix.monitor.residual import authority_ratio, calibrate, window_stats
from phoenix.monitor.response_shift import (
    CANDIDATE_GROUPS,
    MULTI_JOINT_GROUPS,
    ShiftState,
    _group_series,
    assess_shift,
    calibrate_groups,
)

np.seterr(all="ignore")


def sessions(root: Path, seeds: list[str] | None = None) -> list[Path]:
    """Telemetry sessions under ``root``, optionally restricted to given seeds.

    Each simulated robot is an independent session: its own initial state and its own
    command sequence.
    """
    out = sorted(root.glob("seed*/bridge/robot*.jsonl"))
    if seeds:
        want = {f"seed{s}" if not s.startswith("seed") else s for s in seeds}
        out = [p for p in out if p.parent.parent.name in want]
    return out


def stats_for(paths: list[Path]):
    return [window_stats(tracking_pairs(read_bridge_telemetry(p))) for p in paths]


def first_flag_window(st, base, gbase, group: str) -> int | None:
    """Windows until ``group`` first satisfies its persistence rule. Detection latency."""
    cfg = gbase.cfg
    s = authority_ratio(st, base)
    series = _group_series(s, CANDIDATE_GROUPS[group])
    thr = gbase.threshold[group]
    vals = [v for v in series if np.isfinite(v)]
    for i in range(cfg.min_usable, len(vals) + 1):
        recent = vals[max(0, i - cfg.n) : i]
        if sum(1 for v in recent if v < thr) >= min(cfg.k_of_n, len(recent)):
            return i
    return None


def score(paths, base, gbase, true_group, applied, groups=None):
    rows = []
    for p, st in zip(paths, stats_for(paths), strict=True):
        rep = assess_shift(st, base, gbase, groups=groups)
        sel = rep.selected
        rows.append(
            {
                "session": str(p),
                "flagged": rep.shifted,
                "selected_group": None if sel is None else sel.group,
                "selected_is_true": bool(sel is not None and sel.group == true_group),
                "severity": None if sel is None else sel.shift,
                "range": None if sel is None else [sel.lo, sel.hi],
                "true_group_state": (
                    next(g.state for g in rep.groups if g.group == true_group)
                    if true_group
                    else None
                ),
                "true_group_shift": (
                    next(g.shift for g in rep.groups if g.group == true_group)
                    if true_group
                    else None
                ),
                "latency_windows": (
                    first_flag_window(st, base, gbase, true_group) if true_group else None
                ),
            }
        )
    flagged = [r for r in rows if r["flagged"]]
    tg = [r["true_group_shift"] for r in rows if r["true_group_shift"] is not None]
    lat = [r["latency_windows"] for r in rows if r["latency_windows"] is not None]
    return {
        "sessions": rows,
        "n": len(rows),
        "flagged_rate": len(flagged) / max(len(rows), 1),
        "selected_is_true_rate": sum(r["selected_is_true"] for r in rows) / max(len(rows), 1),
        "true_group_detected_rate": sum(
            r["true_group_state"] == ShiftState.SHIFTED.value for r in rows
        )
        / max(len(rows), 1),
        "severity_median": float(np.median(tg)) if tg else None,
        "severity_range": [float(np.min(tg)), float(np.max(tg))] if tg else None,
        "applied_scale": applied,
        "severity_bias": (float(np.median(tg)) - applied) if (tg and applied) else None,
        "latency_windows_median": float(np.median(lat)) if lat else None,
        "latency_detected_fraction": len(lat) / max(len(rows), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument(
        "--nominal", type=Path, required=True, help="held-out nominal, for false alarms"
    )
    ap.add_argument("--degraded", type=Path, action="append", default=[])
    ap.add_argument("--true-group", default=None, help="the group actually degraded")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="dev")
    ap.add_argument("--calibration-seed", action="append", default=None)
    ap.add_argument("--nominal-seed", action="append", default=None)
    ap.add_argument(
        "--multi-joint-only",
        action="store_true",
        help="drop the twelve singleton candidates, leaving only the physical groups; "
        "correct when every intervention under study is multi-joint",
    )
    a = ap.parse_args()

    cal = sessions(a.calibration, a.calibration_seed)
    nom = sessions(a.nominal, a.nominal_seed)
    if not cal or not nom:
        print("need calibration and nominal sessions", file=sys.stderr)
        return 1
    if set(map(str, cal)) & set(map(str, nom)):
        print(
            "calibration and nominal session sets overlap; a false-alarm rate measured "
            "on the sessions that set the threshold is meaningless",
            file=sys.stderr,
        )
        return 2

    groups = MULTI_JOINT_GROUPS if a.multi_joint_only else CANDIDATE_GROUPS
    base = calibrate(stats_for(cal), regime="walk")
    gbase = calibrate_groups(stats_for(cal), base, regime="walk", sessions=[str(p) for p in cal])

    out = {
        "schema": "phoenix-v2-shift-monitor-eval/v1",
        "label": a.label,
        "config": gbase.to_dict()["config"],
        "true_group": a.true_group,
        "hypothesis_space": sorted(groups),
        "calibration_sessions": [str(p) for p in cal],
        "group_baseline": gbase.to_dict(),
        "nominal": score(nom, base, gbase, a.true_group, None, groups),
        "conditions": {},
    }
    n = out["nominal"]
    print(f"[{a.label}] calibrated on {len(cal)} sessions, {len(nom)} held-out nominal")
    print(f"  nominal false-flag rate {n['flagged_rate']:.3f} ({n['n']} sessions)")
    if n["flagged_rate"]:
        bad = {r["selected_group"] for r in n["sessions"] if r["flagged"]}
        print(f"    groups falsely flagged: {sorted(bad)}")

    for d in a.degraded:
        applied = None
        tail = d.name.rsplit("_", 1)[-1]
        if tail.startswith("0p"):
            applied = float(tail.replace("p", "."))
        r = score(sessions(d), base, gbase, a.true_group, applied, groups)
        out["conditions"][d.name] = r
        sev = f"{r['severity_median']:.3f}" if r["severity_median"] is not None else "n/a"
        print(
            f"  {d.name:22s} detect {r['true_group_detected_rate']:.3f} "
            f"correct-group {r['selected_is_true_rate']:.3f} "
            f"severity {sev} (true {applied}) "
            f"latency {r['latency_windows_median']}"
        )

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "shift_monitor.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}/shift_monitor.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
