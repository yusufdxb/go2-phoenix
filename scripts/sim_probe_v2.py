"""Run the v2 friction probe against the fixed-base MuJoCo GO2 (positive control P1).

The probe's own state machine (``friction_probe_node.ProbeSession``, protocol v2) drives a
simulated "robot" whose friction is a chosen model (default: Model B-measured), so replaying
the resulting ``fp_trace.npz`` through the same models must recover that model. 1 ms physics,
50 Hz probe ticks, targets held between ticks, sensor noise optional. Before the probe's first
command the simulated bridge holds the start pose under the probe gains.

    PYTHONPATH=src python scripts/sim_probe_v2.py --scene <scene.xml> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actuator_replay import actuator, build_fixed_base, set_joint_friction  # noqa: E402

from phoenix.sim2real import friction_probe as fp  # noqa: E402
from phoenix.sim2real.actuator_replay import joint_type  # noqa: E402
from phoenix.sim2real.friction_probe_node import (  # noqa: E402
    AbortLimits,
    ProbeSession,
    Snapshot,
    write_npz,
)
from phoenix.sim2real.go2_model import UNITREE_MOTOR_ORDER  # noqa: E402

Q0 = np.tile([0.0, 0.8, -1.5], 4)
TICK_STEPS = 20  # 1 ms physics, 50 Hz probe


def run(scene: str, model_yaml: str | None, out: Path, repeats: int, joints: list[str],
        noise_rad: float, seed: int, profile: str = "isaac_matched") -> dict:
    import mujoco

    m, qadr, dadr, _ = build_fixed_base(scene, profile, 0.001)
    if model_yaml:
        cfg = yaml.safe_load(Path(model_yaml).read_text())
        c = cfg.get("constraint", {})
        set_joint_friction(m, dadr, {k: float(v) for k, v in cfg["frozen_nm"].items()},
                           c.get("solref"), c.get("solimp"))
    d = mujoco.MjData(m)
    d.qpos[qadr] = Q0
    mujoco.mj_forward(m, d)
    calf = np.array([joint_type(n) == "calf" for n in UNITREE_MOTOR_ORDER])
    rng = np.random.default_rng(seed)
    total_s = ProbeSession(joints, AbortLimits(), protocol="v2", repeats=repeats).total_s
    probe = ProbeSession(joints, AbortLimits(wall_cap_s=total_s + 10.0), protocol="v2", repeats=repeats)
    target, app = Q0.copy(), np.zeros(12)
    rows, end = [], None
    for k in range(int((probe.total_s + 10.0) * 1000 / TICK_STEPS) + 1):
        t_s = k * TICK_STEPS * 0.001
        q = d.qpos[qadr] + rng.normal(0.0, noise_rad, 12)
        snap = Snapshot(q, d.qvel[dadr].copy(), app.copy(), 0.0, False, 0.0)
        res = probe.step(t_s, snap, int(round(t_s * 1e9)))
        if res.row is not None:
            rows.append(res.row)
        if res.phase in ("abort", "done"):
            end = res.reason or res.phase
            break
        if res.target is not None:
            target = np.asarray(res.target, dtype=float)
        for _ in range(TICK_STEPS):
            _, app = actuator(target, d.qpos[qadr], d.qvel[dadr], fp.KP, fp.KD, calf)
            d.qfrc_applied[:] = 0
            d.qfrc_applied[dadr] = app
            mujoco.mj_step(m, d)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"q0": probe.q0 if probe.q0 is not None else Q0, "probe_order": list(UNITREE_MOTOR_ORDER),
            "protocol": "v2"}
    write_npz(out / "fp_trace.npz", rows, meta)
    info = {"end": end, "rows": len(rows), "total_s": probe.total_s, "model": model_yaml,
            "noise_rad": noise_rad, "seed": seed, "repeats": repeats, "joints": joints,
            "scene": scene, "profile": profile}
    (out / "sim_probe.json").write_text(json.dumps(info, indent=1))
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="configs/sim2real/actuator_model_B_measured.yaml",
                    help="model-B style YAML for the simulated robot; '' for no added friction")
    ap.add_argument("--repeats", type=int, default=fp.V2_REPEATS)
    ap.add_argument("--joints", default=",".join(UNITREE_MOTOR_ORDER))
    ap.add_argument("--noise-rad", type=float, default=3e-5, help="q sensor noise sd (10-07 settle)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    info = run(args.scene, args.model or None, Path(args.out), args.repeats, args.joints.split(","),
               args.noise_rad, args.seed)
    print(json.dumps(info))
    return 0 if info["end"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
