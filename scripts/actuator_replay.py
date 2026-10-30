#!/usr/bin/env python3
"""Replay recorded GO2 motor commands through a fixed-base MuJoCo GO2 and compare.

Input: hung-robot sessions (``FP_*`` dirs with ``bridge.jsonl`` and optionally
``fp_trace.npz``) listed in a split config. For each session the simulator is
driven by the logged ``final_target``/``kp``/``kd`` (zero-order hold, optional
command delay) through the training actuator law (per-joint DCMotor, as in
``phoenix.velocity.env_cfg``), reset to the measured state every ``--reset-s``.

Output (``--out``): ``metrics.json``, ``summary.md`` and per-session plots.

    PYTHONPATH=src python scripts/actuator_replay.py \
        --split configs/sim2real/actuator_replay_split.yaml \
        --data-root <hw_sessions dir> --scene <unitree_go2/scene.xml> \
        --out logs/actuator_s2r/baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from phoenix.sim2real.actuator_replay import (
    JOINT_TYPES,
    SIGNAL_STATUS,
    Session,
    joint_metrics,
    joint_type,
    load_cache,
    load_session,
    probe_torque_check,
    save_cache,
    session_tilt_deg,
    stick_slip_metrics,
)
from phoenix.sim2real.go2_model import UNITREE_MOTOR_ORDER
from phoenix.sim2sim.dc_motor import DCMotorParams, dc_motor_torque

# Training actuator (src/phoenix/velocity/env_cfg.py): hip/thigh vs calf DCMotor.
HIP_THIGH = DCMotorParams(25.0, 0.5, 23.5, 23.5, 30.0)
CALF = DCMotorParams(25.0, 0.5, 45.43, 45.43, 15.7)
GAP_RESET_S = 0.05


def build_fixed_base(scene: str, profile: str, dt: float):
    """Fixed-base GO2 carrying the profiled model's inertias and joint params."""
    import mujoco

    from phoenix.sim2sim.model import load_go2_model

    ref, _, applied = load_go2_model(profile, scene_xml=scene, timestep=dt)
    spec = mujoco.MjSpec.from_file(scene)
    for j in list(spec.body("base").joints):
        spec.delete(j)
    for k in list(spec.keys):
        spec.delete(k)
    m = spec.compile()
    for b in range(m.nbody):
        rb = mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_BODY, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b))
        m.body_mass[b], m.body_ipos[b] = ref.body_mass[rb], ref.body_ipos[rb]
        m.body_iquat[b], m.body_inertia[b] = ref.body_iquat[rb], ref.body_inertia[rb]
    for n in UNITREE_MOTOR_ORDER:
        a = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        r = ref.jnt_dofadr[mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_JOINT, n)]
        m.dof_damping[a], m.dof_frictionloss[a] = ref.dof_damping[r], ref.dof_frictionloss[r]
        m.dof_armature[a] = ref.dof_armature[r]
    m.opt.timestep = dt
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    mujoco.mj_setConst(m, mujoco.MjData(m))
    qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in UNITREE_MOTOR_ORDER])
    dadr = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in UNITREE_MOTOR_ORDER])
    return m, qadr, dadr, applied


def set_joint_friction(model, dadr, frictionloss: dict[str, float], solref=None, solimp=None) -> None:
    """Model B: per-joint-type Coulomb friction as MuJoCo's dof frictionloss constraint.

    The constraint holds a joint at rest while the net torque on it is below F and
    opposes sliding with F (static = Coulomb). ``solref``/``solimp`` set the softness
    of that constraint, i.e. how much a held joint creeps under subthreshold torque.
    """
    for j, n in enumerate(UNITREE_MOTOR_ORDER):
        model.dof_frictionloss[dadr[j]] = frictionloss[joint_type(n)]
        if solref is not None:
            model.dof_solref[dadr[j]] = solref
        if solimp is not None:
            model.dof_solimp[dadr[j], : len(solimp)] = solimp


def tilt_gravity(model, roll_deg: float = 0.0, pitch_deg: float = 0.0) -> None:
    """Gravity seen by the welded base when the real base is tilted by roll then pitch."""
    r, p = np.radians(roll_deg), np.radians(pitch_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    model.opt.gravity[:] = (ry @ rx).T @ np.array([0.0, 0.0, -9.81])


def actuator(q_des, q, qd, kp, kd, calf_mask):
    out = []
    for p in (HIP_THIGH, CALF):
        out.append(dc_motor_torque(q_des, q, qd, p, stiffness_scale=kp / p.stiffness, damping_scale=kd / p.damping))
    computed = np.where(calf_mask, out[1][0], out[0][0])
    applied = np.where(calf_mask, out[1][1], out[0][1])
    return computed, applied


def replay(s: Session, model, qadr, dadr, delay_s: float, reset_s: float):
    """Open-loop replay; returns sim q, dq, clip flag, applied motor torque at the probe
    times and the dof friction constraint torque, all at tick times."""
    import mujoco

    d = mujoco.MjData(model)
    h = model.opt.timestep
    calf = np.array([joint_type(n) == "calf" for n in UNITREE_MOTOR_ORDER])
    gaps = np.flatnonzero(np.diff(s.t) > GAP_RESET_S) + 1
    seg_start = np.concatenate([[0], gaps])
    seg_end = np.concatenate([gaps, [len(s.t)]])
    tt, qq, vv, cc, uu, ff = [], [], [], [], [], []
    fric_dof = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
    dof_to_j = np.full(model.nv, -1)
    dof_to_j[dadr] = np.arange(12)
    for a, b in zip(seg_start, seg_end, strict=True):
        if b - a < 2:
            continue
        ts = s.t[a:b]
        grid = np.arange(ts[0], ts[-1] + 1e-9, h)
        next_reset = ts[0]
        for tk in grid:
            if tk >= next_reset - 1e-9:
                i = min(np.searchsorted(ts, tk), len(ts) - 1) + a
                d.qpos[qadr], d.qvel[dadr] = s.q[i], s.dq[i]
                d.qacc_warmstart[:] = 0
                next_reset += reset_s
            i = max(np.searchsorted(ts, tk - delay_s, side="right") - 1, 0) + a
            q, qd = d.qpos[qadr].copy(), d.qvel[dadr].copy()
            comp, app = actuator(s.target[i], q, qd, s.kp[i], s.kd[i], calf)
            d.qfrc_applied[:] = 0
            d.qfrc_applied[dadr] = app
            mujoco.mj_step(model, d)
            tt.append(tk + h)
            qq.append(d.qpos[qadr].copy())
            vv.append(d.qvel[dadr].copy())
            cc.append(np.abs(comp - app) > 1e-9)
            uu.append(app)
            fr = np.zeros(12)
            sel = d.efc_type == fric_dof
            if sel.any():
                np.add.at(fr, dof_to_j[d.efc_id[sel]], d.efc_force[sel])
            ff.append(fr)
    tt = np.asarray(tt)
    qq, vv, cc, uu, ff = map(np.asarray, (qq, vv, cc, uu, ff))

    def at(x, times):
        return np.stack([np.interp(times, tt, x[:, j]) for j in range(12)], axis=1)

    nearest = np.clip(np.searchsorted(tt, s.t) - 1, 0, len(tt) - 1)
    tau_probe = None
    if s.probe is not None:
        tau_probe = uu[np.clip(np.searchsorted(tt, s.probe["t"]) - 1, 0, len(tt) - 1)]
    return at(qq, s.t), at(vv, s.t), cc[nearest], tau_probe, ff[nearest]


def plot_session(s: Session, q_sim: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharex=True)
    for j, n in enumerate(UNITREE_MOTOR_ORDER):
        ax = axes[j // 3, j % 3]
        ax.plot(s.t, s.target[:, j], color="0.6", lw=0.8, label="target (cmd)")
        ax.plot(s.t, s.q[:, j], color="#1f4e8c", lw=1.0, label="q hw")
        ax.plot(s.t, q_sim[:, j], color="#c0502b", lw=0.9, ls="--", label="q sim")
        ax.set_title(n, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("t [s]", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_probe(s: Session, q_sim: np.ndarray, tau_sim: np.ndarray | None, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = s.probe
    active = [n for n in UNITREE_MOTOR_ORDER if np.any(p["active"] == n)]
    if not active:
        return
    fig, axes = plt.subplots(len(active), 2, figsize=(13, 2.4 * len(active)), squeeze=False)
    for r, n in enumerate(active):
        j = UNITREE_MOTOR_ORDER.index(n)
        sel = p["active"] == n
        t0, t1 = p["t"][sel][0] - 0.5, p["t"][sel][-1] + 0.5
        w = (s.t >= t0) & (s.t <= t1)
        ax = axes[r, 0]
        ax.plot(s.t[w], s.target[w, j] - s.q[w, j], color="#1f4e8c", lw=1, label="target - q hw")
        ax.plot(s.t[w], s.target[w, j] - q_sim[w, j], color="#c0502b", lw=1, ls="--", label="target - q sim")
        ax.set_title(f"{n} tracking error [rad]", fontsize=9)
        ax.legend(fontsize=7)
        ax = axes[r, 1]
        ax.plot(p["t"][sel], p["tau_est"][sel, j], ".", ms=3, color="#1f4e8c", label="tau_est hw (measured)")
        if tau_sim is not None:
            ax.plot(p["t"][sel], tau_sim[sel, j], ".", ms=3, color="#c0502b", label="tau sim (applied)")
        ax.set_title(f"{n} torque [N m]", fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def by_type(per_joint: dict, key: str) -> dict:
    out = {}
    for jt in JOINT_TYPES:
        v = [m[key] for n, m in per_joint.items()
             if joint_type(n) == jt and key in m and np.isfinite(m[key]) and "excluded" not in m]
        if v:
            out[jt] = float(np.mean(v))
    return out


def write_summary(res: dict, path: Path) -> None:
    lines = [f"# Actuator replay: {res['model']['label']}", "",
         f"profile `{res['model']['profile']}`, dt {res['model']['dt']} s, command delay "
         f"{res['model']['delay_s']} s, reset every {res['model']['reset_s']} s", "",
         "## Signal status", "", "| signal | status |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in SIGNAL_STATUS.items()]
    for split in ("train", "heldout"):
        lines += ["", f"## {split}", "",
              "| session | stage | code sha | ckpt sha256 | ticks | pos RMSE hip/thigh/calf [rad] | vel RMSE hip/thigh/calf [rad/s] |",
              "|---|---|---|---|---|---|---|"]
        for name, r in res["sessions"].items():
            if r["split"] != split:
                continue
            idn = r["identity"]
            pr, vr = r["by_type"]["pos_rmse"], r["by_type"]["vel_rmse"]
            fmt = lambda d: " / ".join(f"{d.get(t, float('nan')):.4f}" for t in JOINT_TYPES)  # noqa: E731
            lines.append(f"| {name} | {idn.get('stage')} | {str(idn.get('code_sha'))[:8]} | "
                     f"{str(idn.get('checkpoint_sha256'))[:8]} | {r['n_ticks']} | {fmt(pr)} | {fmt(vr)} |")
        lines += ["", "Timing and steady state by joint type (hip / thigh / calf; whole session, exclusions dropped):", ""]
        for name, r in res["sessions"].items():
            if r["split"] != split:
                continue
            b = r["by_type"]
            f3 = lambda k, b=b: " / ".join(f"{b[k].get(t, float('nan')):+.3f}" for t in JOINT_TYPES)  # noqa: E731
            lines.append(f"- {name}: lag target->dq hw {f3('lag_hw_s')} s, sim {f3('lag_sim_s')} s; "
                     f"steady target-q hw {f3('sse_hw')} rad, sim {f3('sse_sim')} rad")
        lines += ["", "Per mode (pos RMSE hip / thigh / calf):", ""]
        for name, r in res["sessions"].items():
            if r["split"] != split:
                continue
            for mode, row in r["metrics"]["per_mode"].items():
                lines.append(f"- {name} `{mode}` kp {row['kp']:.0f} kd {row['kd']:.1f} n={row['n']}: "
                         + " / ".join(f"{row[f'pos_rmse_{t}']:.4f}" for t in JOINT_TYPES))
    lines += ["", "## Probe torque check (tau_est vs PD law on the published target)", "",
          "| session | joint | n | gain scale a | offset b [N m] | resid RMSE | tau_sim - tau_est RMSE | excluded |",
          "|---|---|---|---|---|---|---|---|"]
    for name, r in res["sessions"].items():
        for j, row in r["probe_torque"].items():
            lines.append(f"| {name} | {j} | {row['n']} | {row['gain_scale']:.3f} | {row['offset_nm']:+.3f} | "
                     f"{row['resid_rmse_nm']:.3f} | {row.get('tau_sim_minus_tau_est_rmse_nm', float('nan')):.3f} | "
                     f"{row.get('excluded', '')} |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="isaac_matched")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--delay-s", type=float, default=0.0)
    ap.add_argument("--reset-s", type=float, default=2.0)
    ap.add_argument("--label", default="A: existing training actuator")
    ap.add_argument("--cache", default=None, help="dir for parsed-session npz caches")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--save-sim", action="store_true", help="write sim q (float32) per session for overlays")
    ap.add_argument("--model-b", default=None, help="model-B YAML (constraint softness, optional 'frozen' F)")
    ap.add_argument("--frictionloss", default=None, help="hip,thigh,calf Coulomb friction [N m]; implies model B")
    ap.add_argument("--splits", default="train,heldout")
    ap.add_argument("--tilt", default="level", choices=["level", "pitch+", "pitch-", "roll+", "roll-"],
                    help="apply each session's median logged tilt magnitude about this axis")
    args = ap.parse_args(argv)

    split = yaml.safe_load(Path(args.split).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    model, qadr, dadr, applied = build_fixed_base(args.scene, args.profile, args.dt)
    fric = None
    if args.model_b or args.frictionloss:
        cfg = yaml.safe_load(Path(args.model_b).read_text()) if args.model_b else {}
        if args.frictionloss:
            fric = dict(zip(JOINT_TYPES, map(float, args.frictionloss.split(",")), strict=True))
        else:
            fric = {k: float(v) for k, v in cfg["frozen_nm"].items()}
        c = cfg.get("constraint", {})
        set_joint_friction(model, dadr, fric, c.get("solref"), c.get("solimp"))
    res = {"model": {"label": args.label, "profile": args.profile, "dt": args.dt, "delay_s": args.delay_s,
                     "reset_s": args.reset_s, "actuator": {"hip_thigh": HIP_THIGH.__dict__, "calf": CALF.__dict__},
                     "applied_profile": {k: v for k, v in applied.items() if k != "scene_xml"},
                     "frictionloss_nm": fric, "model_b_config": args.model_b, "tilt": args.tilt,
                     "dof_solref": model.dof_solref[dadr[0]].tolist(), "dof_solimp": model.dof_solimp[dadr[0]].tolist()},
           "split_file": args.split, "sessions": {}}
    for sp in args.splits.split(","):
        for rel in split[sp]:
            name = Path(rel).name
            cp = cache / f"{name}.npz"
            s = load_cache(cp) if cp.exists() else load_session(Path(args.data_root) / rel)
            if not cp.exists():
                save_cache(s, cp)
            excl = (split.get("exclusions") or {}).get(name, {})
            tilt = float("nan")
            if args.tilt != "level":
                tp = cache / f"{name}.tilt.json"
                if not tp.exists():
                    tp.write_text(json.dumps(session_tilt_deg(Path(args.data_root) / rel)))
                tilt = json.loads(tp.read_text())
                sign = 1.0 if args.tilt.endswith("+") else -1.0
                tilt_gravity(model, **{f"{args.tilt[:-1]}_deg": sign * tilt})
            q_sim, dq_sim, clip, tau_probe, tau_fric = replay(s, model, qadr, dadr, args.delay_s, args.reset_s)
            tilt_gravity(model)
            met = joint_metrics(s, q_sim, dq_sim, clip, exclude=excl)
            for n, row in stick_slip_metrics(s, q_sim, excl).items():
                met["per_joint"][n].update({k: v for k, v in row.items() if k != "excluded"})
            for j, n in enumerate(UNITREE_MOTOR_ORDER):
                met["per_joint"][n]["fric_abs_mean_nm"] = float(np.abs(tau_fric[:, j]).mean())
                met["per_joint"][n]["fric_abs_max_nm"] = float(np.abs(tau_fric[:, j]).max())
            r = {"split": sp, "path": rel, "identity": s.identity, "n_ticks": int(len(s.t)), "tilt_deg": tilt,
                 "duration_s": float(s.t[-1]), "metrics": met,
                 "by_type": {k: by_type(met["per_joint"], k) for k in
                             ("pos_rmse", "vel_rmse", "lag_hw_s", "lag_sim_s", "lag_hw_vs_sim_s", "sse_hw", "sse_sim",
                              "overshoot_hw", "overshoot_sim", "sse_diff", "breakaway_hw_s", "breakaway_sim_s",
                              "breakaway_abs_err_s", "err_at_break_hw", "err_at_break_sim", "stuck_frac_hw",
                              "stuck_frac_sim", "stuck_agree", "dir_resid_split", "fric_abs_mean_nm")},
                 "probe_torque": probe_torque_check(s, tau_probe, excl)}
            res["sessions"][name] = r
            if args.save_sim:
                np.savez_compressed(out / f"{name}_sim.npz", t=s.t, q_sim=q_sim.astype(np.float32),
                                    tau_fric=tau_fric.astype(np.float32))
            if not args.no_plots:
                plot_session(s, q_sim, out / f"{name}_overlay.png", f"{name} ({sp}) {args.label}")
                if s.probe is not None and len(s.probe["t"]) >= 50:
                    plot_probe(s, q_sim, tau_probe, out / f"{name}_probe.png")
            print(f"{name} [{sp}] ticks={len(s.t)} pos_rmse={r['by_type']['pos_rmse']}", flush=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=float))
    write_summary(res, out / "summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
