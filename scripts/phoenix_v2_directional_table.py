#!/usr/bin/env python3
"""Directional table (diagnostic) for one or more eval dirs holding steps.npz + episodes.jsonl.

Writes <dir>/directional.json and prints one row per bin.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from phoenix.monitor.walk_directional import directional_report

COLS = ("n_segments", "vx_cmd", "vx_achieved", "abs_vx_err", "success", "trunk_contact_episode",
        "attitude_violation", "mean_pitch", "mean_abs_action", "raw_out_of_range", "p95_joint_speed",
        "torque_saturation", "action_rate_l2", "torque_l2", "front_duty_minus_rear", "thigh_contact",
        "min_height")

for d in sys.argv[1:]:
    z = dict(np.load(f"{d}/steps.npz"))
    eps = [json.loads(line) for line in open(f"{d}/episodes.jsonl")]
    rep = directional_report(z, eps, dt=0.02)
    with open(f"{d}/directional.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"== {d}")
    print("bin".ljust(16) + "".join(c[:10].rjust(11) for c in COLS))
    for name, row in rep["bins"].items():
        print(name.ljust(16) + "".join(
            (f"{row[c]:11.3f}" if isinstance(row.get(c), float) else str(row.get(c, "-")).rjust(11)) for c in COLS))
