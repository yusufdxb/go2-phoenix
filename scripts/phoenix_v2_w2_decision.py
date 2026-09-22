#!/usr/bin/env python3
"""Apply the amendment 8.2 decision rule to W2's dev evaluation and directional curve.

    python scripts/phoenix_v2_w2_decision.py <dev_eval_dir> <curve_dir> [--out decision.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GATE = {"nominal": 0.90, "dr": 0.80, "stand": 0.90}


def f_of(curve: Path, it: int) -> dict:
    b = json.loads((curve / f"it{it:04d}" / "directional.json").read_text())["bins"]
    sf, sb = b["strong_forward"], b["strong_backward"]
    return {"F": sf["vx_achieved"] / sf["vx_cmd"], "B": sb["vx_achieved"] / sb["vx_cmd"],
            "fwd_success": sf["success"], "bwd_success": sb["success"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dev", type=Path)
    ap.add_argument("curve", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    gate = {c: json.loads((a.dev / c / "summary.json").read_text())["walk2_success_rate"] for c in GATE}
    passed = all(gate[c] >= GATE[c] for c in GATE)
    its = sorted(int(p.name[2:]) for p in a.curve.glob("it*") if (p / "directional.json").exists())
    curve = {it: f_of(a.curve, it) for it in its}
    f_end, f_2600 = curve[2999]["F"], curve[2600]["F"]
    if passed:
        branch = "a: PASS dev Gate W-H -> freeze, fresh seeds, Phase 7"
    elif f_end >= 0.5 and f_end - f_2600 >= 0.05:
        branch = "b: still improving -> extension study E"
    else:
        branch = "c: W3 symmetric curriculum"
    out = {"rule": "amendment 8.2", "gate": gate, "gate_thresholds": GATE, "gate_pass": passed,
           "F_2999": f_end, "F_2600": f_2600, "curve": curve, "branch": branch}
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "curve"}, indent=1))
    for it, v in curve.items():
        print(it, {k: round(x, 3) for k, x in v.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
