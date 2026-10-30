#!/usr/bin/env python3
"""Physics checks for model B (dof frictionloss) on the fixed-base GO2, no hardware data.

1. holding: constant torque on one joint, gravity off, 2 s; drift vs torque/F
   for several constraint softness settings (creep of the soft constraint)
2. ramp sweep: PD (kp 25, kd 0.5) tracks a 0.05 rad/s up/down ramp under gravity;
   hysteresis (tracking error at breakaway, up + down) vs F; Coulomb predicts 2F/kp
3. stick quality: velocity sign changes and peak |v| while the joint is held
4. step convergence: the ramp at dt 1 ms vs 0.5 ms, same constraint time constant

    PYTHONPATH=src python scripts/model_b_physics_checks.py --scene <scene.xml> \
        --config configs/sim2real/actuator_model_B.yaml --out logs/actuator_s2r/model_B/physics_checks
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

from phoenix.sim2real.actuator_replay import breakaway, joint_type  # noqa: E402
from phoenix.sim2real.go2_model import UNITREE_MOTOR_ORDER  # noqa: E402

Q0 = np.tile([0.0, 0.8, -1.5], 4)  # hung sit-like pose, inside every joint range
CALF = np.array([joint_type(n) == "calf" for n in UNITREE_MOTOR_ORDER])


def holding(scene, dt, fric, solref, solimp, joints=(0, 1, 2), fracs=(0.5, 0.9, 1.1), dur=2.0):
    import mujoco

    m, qadr, dadr, _ = build_fixed_base(scene, "isaac_matched", dt)
    m.opt.gravity[:] = 0
    set_joint_friction(m, dadr, fric, solref, solimp)
    out = {}
    for j in joints:
        f = fric[joint_type(UNITREE_MOTOR_ORDER[j])]
        for frac in fracs:
            d = mujoco.MjData(m)
            d.qpos[qadr] = Q0
            for _ in range(int(round(dur / dt))):
                d.qfrc_applied[:] = 0
                d.qfrc_applied[dadr[j]] = frac * f
                mujoco.mj_step(m, d)
            out[f"{UNITREE_MOTOR_ORDER[j]}@{frac}F"] = float(d.qpos[qadr[j]] - Q0[j])
    return out


def ramp(scene, dt, fric, solref, solimp, j, rate=0.05, amp=0.1, kp=25.0, kd=0.5):
    """Settle 1 s, ramp up by amp, ramp back down; returns t, q_j, target_j, friction torque."""
    import mujoco

    m, qadr, dadr, _ = build_fixed_base(scene, "isaac_matched", dt)
    set_joint_friction(m, dadr, fric, solref, solimp)
    d = mujoco.MjData(m)
    d.qpos[qadr] = Q0
    tr = amp / rate
    t = np.arange(0, 1.0 + 2 * tr, dt)
    off = np.where(t < 1, 0, np.where(t < 1 + tr, (t - 1) * rate, amp - (t - 1 - tr) * rate))
    fric_dof = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
    kpv, kdv = np.full(12, kp), np.full(12, kd)
    q, fr = np.zeros(len(t)), np.zeros(len(t))
    for k, ok in enumerate(off):
        tgt = Q0.copy()
        tgt[j] += ok
        _, app = actuator(tgt, d.qpos[qadr], d.qvel[dadr], kpv, kdv, CALF)
        d.qfrc_applied[:] = 0
        d.qfrc_applied[dadr] = app
        mujoco.mj_step(m, d)
        q[k] = d.qpos[qadr[j]]
        sel = (d.efc_type == fric_dof) & (d.efc_id == dadr[j])
        fr[k] = d.efc_force[sel].sum()
    return t + dt, q, Q0[j] + off, fr, tr


def ramp_summary(t, q, tgt, fr, tr, f, kp=25.0):
    up = breakaway(q, tgt, t, 1.0, 1.0 + tr, 1.0)
    dn = breakaway(q, tgt, t, 1.0 + tr, 1.0 + 2 * tr, -1.0)
    v = np.gradient(q, t)
    # held = second half of the stick interval between ramp reversal and downward breakaway
    # (the first half still contains the decelerating upward slide)
    held = (t > 1.0 + tr + dn[0] / 2) & (t < 1.0 + tr + dn[0]) if np.isfinite(dn[0]) else np.zeros_like(t, bool)
    vh = v[held]
    return {"break_up_s": up[0], "break_down_s": dn[0], "err_up": up[1], "err_down": dn[1],
            "hysteresis_rad": up[1] + dn[1], "coulomb_pred_rad": 2 * f / kp,
            "held_median_abs_v": float(np.median(np.abs(vh))) if len(vh) else float("nan"),
            "held_frac_below_1mrad_s": float((np.abs(vh) < 1e-3).mean()) if len(vh) else float("nan"),
            "held_v_sign_changes": int((np.diff(np.sign(vh[np.abs(vh) > 1e-6])) != 0).sum()) if len(vh) else 0,
            "held_max_abs_fric_nm": float(np.abs(fr[held]).max()) if held.any() else float("nan"),
            "slide_mean_abs_fric_nm": float(np.abs(fr[(t > 1.0 + 0.8 * tr) & (t < 1.0 + tr)]).mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    solref, solimp = cfg["constraint"]["solref"], cfg["constraint"]["solimp"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = {"hip": 0.3, "thigh": 0.3, "calf": 0.8}  # mid training-measured range, check values only
    res = {"reference_friction_nm": ref, "holding": {}, "ramp_sweep": {}, "dt_convergence": {}}
    for name, (sr, si) in {"mujoco_default": ([0.02, 1.0], [0.9, 0.95, 0.001]),
                           "stiff_time_default_imp": ([0.002, 1.0], [0.9, 0.95, 0.001]),
                           "registered": (solref, solimp)}.items():
        res["holding"][name] = {"solref": sr, "solimp": si,
                                "drift_rad_2s": holding(args.scene, 0.001, ref, sr, si)}
        print(name, {k: round(v * 1e3, 3) for k, v in res["holding"][name]["drift_rad_2s"].items()}, "mrad")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for c, j in enumerate((0, 1, 2)):
        jt = joint_type(UNITREE_MOTOR_ORDER[j])
        for f in (0.0, 0.5 * ref[jt], ref[jt], 1.5 * ref[jt]):
            fr_ = {k: (f if k == jt else ref[k]) for k in ref}
            t, q, tgt, fr, tr = ramp(args.scene, 0.001, fr_, solref, solimp, j)
            r = ramp_summary(t, q, tgt, fr, tr, f)
            res["ramp_sweep"][f"{UNITREE_MOTOR_ORDER[j]}@F={f:.3f}"] = r
            print(UNITREE_MOTOR_ORDER[j], f"F={f:.3f}", {k: round(v, 4) if isinstance(v, float) else v
                                                          for k, v in r.items()})
            axes[0, c].plot(t, tgt - q, label=f"F={f:.2f} N m")
            axes[1, c].plot(t, fr, label=f"F={f:.2f} N m")
        axes[0, c].set_title(f"{UNITREE_MOTOR_ORDER[j]}: ramp 0.05 rad/s, kp 25 kd 0.5")
        axes[0, c].set_ylabel("target - q [rad]")
        axes[1, c].set_ylabel("friction constraint torque [N m]")
        axes[1, c].set_xlabel("t [s]")
        axes[0, c].legend(fontsize=7)
        # step convergence at the reference friction
        a = ramp(args.scene, 0.001, ref, solref, solimp, j)
        b = ramp(args.scene, 0.0005, ref, solref, solimp, j)
        qb = np.interp(a[0], b[0], b[1])
        res["dt_convergence"][UNITREE_MOTOR_ORDER[j]] = {
            "max_abs_q_diff_rad": float(np.abs(a[1] - qb).max()),
            "summary_1ms": ramp_summary(*a, ref[jt]), "summary_0p5ms": ramp_summary(*b, ref[jt])}
        print(UNITREE_MOTOR_ORDER[j], "dt 1 ms vs 0.5 ms max |dq| rad",
              round(res["dt_convergence"][UNITREE_MOTOR_ORDER[j]]["max_abs_q_diff_rad"], 6))
    fig.tight_layout()
    fig.savefig(out / "ramp_sweep.png", dpi=110)
    (out / "physics_checks.json").write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
