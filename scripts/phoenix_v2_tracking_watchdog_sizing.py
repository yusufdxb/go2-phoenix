#!/usr/bin/env python3
"""Size the catastrophic tracking-error watchdog by the amendment 2 rule.

Rule: the smallest multiple of 0.05 rad that is >= 1.25 x the largest 0.2 s-sustained
``|sent - q|`` on any joint in the development runs given (``steps.npz`` from
``phoenix_v2_sim_stand.py --save-steps``). "Sustained" = the minimum of the gap over
10 consecutive valid 50 Hz ticks, i.e. the largest level the gap stayed above for 0.2 s,
which is exactly what trips the gate's latch.

    python scripts/phoenix_v2_tracking_watchdog_sizing.py <run_dir>... --out sizing.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def sustained_max(gap: np.ndarray, valid: np.ndarray, win: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Per-joint max over (env, window) of the windowed minimum gap; gap (T, N, 12)."""
    g = np.where(valid[..., None], gap, -np.inf)
    t = g.shape[0]
    mins = np.stack([g[k : t - win + 1 + k] for k in range(win)]).min(axis=0)  # (T-win+1, N, 12)
    per_joint = mins.max(axis=(0, 1))
    k, e, _ = np.unravel_index(np.argmax(mins), mins.shape)
    return per_joint, float(mins.max()), (int(k), int(e))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--window-ticks", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = {"rule": "smallest multiple of 0.05 >= 1.25 x max 0.2s-sustained |sent-q|", "runs": {}}
    overall = 0.0
    for r in a.runs:
        z = np.load(r / "steps.npz")
        gap = np.abs(z["sent"] - z["q0"])
        pj, m, at = sustained_max(gap, z["valid"].astype(bool), a.window_ticks)
        res["runs"][str(r)] = {"max_sustained_gap_per_joint": [round(float(v), 4) for v in pj],
                               "max": m, "at_step_env": list(at),
                               "max_instant": float(np.where(z["valid"][..., None], gap, 0).max())}
        overall = max(overall, m)
    res["max_sustained_gap_rad"] = overall
    res["tracking_abort_rad"] = math.ceil(round(1.25 * overall / 0.05, 9)) * 0.05
    res["effort_saturation_gap_rad_kp25"] = 23.5 / 25.0
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
