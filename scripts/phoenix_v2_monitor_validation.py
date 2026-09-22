#!/usr/bin/env python3
"""Phase 1a analysis: calibrate on the nominal calibration sessions, assess every other
session with the frozen monitor settings, and apply the preregistered gate:
the injected joint localised (DEGRADED, and no other joint DEGRADED) in >= 8 of 10
sessions per injected condition at s_train, and no DEGRADED joint in the false-positive set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phoenix.monitor.fidelity import fidelity_report
from phoenix.monitor.health import HealthMonitor, JointState, PersistenceConfig
from phoenix.monitor.layers import read_bridge_telemetry, tracking_pairs
from phoenix.monitor.residual import WindowConfig, authority_ratio, calibrate, window_stats

INJECTED = {"rr_thigh_s_train": "RR_thigh_joint", "fl_calf_s_train": "FL_calf_joint",
            "rl_hip_s_train": "RL_hip_joint"}


def assess(path: Path, base, cfg):
    layers = read_bridge_telemetry(path)
    st = window_stats(tracking_pairs(layers), cfg)
    s = authority_ratio(st, base, cfg)
    mon = HealthMonitor(base, PersistenceConfig())
    report = mon.report()
    first = {}
    for i in range(st.n_windows):
        report = mon.update(s[i], st.torque_gain_ratio[i])
        for h in report:
            if h.state in (JointState.SUSPECT, JointState.DEGRADED) and (h.joint, h.state) not in first:
                first[(h.joint, h.state)] = i + 1  # seconds, 1 s windows
    return report, first, fidelity_report(layers)["verdict"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--s-train", type=float, required=True)
    a = ap.parse_args()
    cfg = WindowConfig()
    cal_files = sorted((a.root / "calibration" / "bridge").glob("*.jsonl"))
    stats = [window_stats(tracking_pairs(read_bridge_telemetry(p)), cfg) for p in cal_files]
    base = calibrate(stats, regime="stand", cfg=cfg, source=[str(p) for p in cal_files])
    (a.root / "baseline.json").write_text(json.dumps(base.to_dict(), indent=1, default=float))
    out = {"schema": "phoenix-v2-monitor-validation-sim/v1", "s_train": a.s_train,
           "calibration_sessions": len(cal_files), "conditions": {}}
    for cond in ("false_positive", "rr_thigh_1p0", *INJECTED):
        rows = []
        for p in sorted((a.root / cond / "bridge").glob("*.jsonl")):
            report, first, fid = assess(p, base, cfg)
            deg = [h.joint for h in report if h.state == JointState.DEGRADED]
            glob_ = [h.joint for h in report if h.state == JointState.GLOBAL_SHIFT]
            rows.append({
                "session": p.name, "fidelity": fid, "degraded": deg, "global_shift": glob_,
                "states": {h.joint: getattr(h.state, "value", h.state) for h in report},
                "s_hat": {h.joint: h.s_hat for h in report},
                "first_suspect_s": {j: t for (j, st), t in first.items() if st == JointState.SUSPECT},
                "first_degraded_s": {j: t for (j, st), t in first.items() if st == JointState.DEGRADED},
            })
        target = INJECTED.get(cond)
        c = {"sessions": len(rows), "rows": rows,
             "any_degraded": sum(bool(r["degraded"]) for r in rows),
             "global_shift_sessions": sum(bool(r["global_shift"]) for r in rows)}
        if target:
            loc = [r for r in rows if r["degraded"] == [target]]
            c["localised"] = len(loc)
            c["false_joint_sessions"] = sum(any(j != target for j in r["degraded"]) for r in rows)
            t_s = [r["first_suspect_s"].get(target) for r in loc]
            t_d = [r["first_degraded_s"].get(target) for r in loc]
            c["median_time_to_suspect_s"] = sorted(x for x in t_s if x)[len(t_s) // 2] if loc else None
            c["median_time_to_degraded_s"] = sorted(x for x in t_d if x)[len(t_d) // 2] if loc else None
            c["s_hat_target_median"] = sorted(r["s_hat"][target] for r in loc if r["s_hat"][target] is not None)[len(loc) // 2] if loc else None
        out["conditions"][cond] = c
    fp_ok = out["conditions"]["false_positive"]["any_degraded"] == 0
    loc_ok = all(out["conditions"][c]["localised"] >= 8 for c in INJECTED)
    out["gate"] = {"false_positive_ok": fp_ok, "localisation_ok": loc_ok, "passed": fp_ok and loc_ok}
    (a.root / "validation.json").write_text(json.dumps(out, indent=1, default=str))
    for cond, c in out["conditions"].items():
        print(cond, {k: v for k, v in c.items() if k != "rows"})
    print("GATE", out["gate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
