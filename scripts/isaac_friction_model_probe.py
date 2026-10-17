"""Runtime check that the joint friction MODEL reaches Isaac's PhysX dynamics (SIM ONLY).

Same rig as scripts/isaac_friction_probe.py (PhoenixVelocity env, other DR neutralised,
root fixed at 0.8 m, legs in the air) but driven through the training code path:
phoenix.velocity.isaac_terms.randomize_joint_friction_model (one degenerate distribution
per env) and update_load_dependent_friction (called every control step, as the interval
event does in training). Three envs:

* e0 coulomb: measured x1.0                -> F_hat(0.05 rad/s) = F0_j
* e1 viscous: measured x1.0 + b per group   -> slope over 0.5/1/2 rad/s of F_hat(e1) - F_hat(e0) = b
* e2 load:    measured x1.0 + k_load 0.1    -> F_hat = F0_j + k * mean|tau| (gravity load)

F_hat(v) = (mean tau_up - mean tau_down) / 2 over samples moving at 0.5-1.5 v. Probed joints
are the front-right leg and RL_calf; tolerance max(0.05 N m, 15 %), viscous max(0.02, 25 %).

    ~/Sim/isaac-sim-venv/bin/python scripts/isaac_friction_model_probe.py --out results/friction_physics/isaac_model_probe.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PROBE_JOINTS = ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "RL_calf_joint")
VISCOUS = {"hip": 0.05, "thigh": 0.08, "calf": 0.10}
LOAD_K = 0.10
SPAN_RAD = 0.30
SETTLE_S = 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    from phoenix.sim_app_exit import run_isaac_main

    return run_isaac_main(lambda: _run(args), app, label="isaac_friction_model_probe")


def _run(args: argparse.Namespace) -> int:
    import gymnasium as gym
    import torch
    import warp as wp

    from phoenix.physics.joint_friction import FrictionDistribution, load_measured
    from phoenix.velocity import env_cfg as pv_env_cfg
    from phoenix.velocity.isaac_terms import randomize_joint_friction_model, update_load_dependent_friction
    from phoenix.velocity.spec import TASK_ID, vnext3_spec

    pv_env_cfg.register()
    spec = vnext3_spec()
    dr = dataclasses.replace(
        spec.domain_randomization,
        base_mass_add_kg=(0.0, 0.0),
        base_com_enabled=False,
        motor_stiffness_scale=(1.0, 1.0),
        motor_damping_scale=(1.0, 1.0),
        joint_armature_add=(0.0, 0.0),
        actuator_latency_physics_steps=(0, 0),
        reset_joint_pos_offset=(0.0, 0.0),
        push_enabled=False,
    )
    spec = spec.replace(domain_randomization=dr, sim=dataclasses.replace(spec.sim, num_envs=3))
    cfg = pv_env_cfg.build_velocity_env_cfg(spec)
    cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.8)
    env = gym.make(TASK_ID, cfg=cfg, render_mode=None)
    env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]
    sim = u.sim
    dt = float(sim.get_physics_dt())
    decim = int(round(float(u.step_dt) / dt))
    dev = u.device
    names = list(robot.joint_names)
    meas = load_measured()

    dists = [
        FrictionDistribution("hardware", scale=(1.0, 1.0)),
        FrictionDistribution("hardware", scale=(1.0, 1.0), viscous_hip=(VISCOUS["hip"],) * 2,
                             viscous_thigh=(VISCOUS["thigh"],) * 2, viscous_calf=(VISCOUS["calf"],) * 2),
        FrictionDistribution("hardware", scale=(1.0, 1.0), load=(LOAD_K, LOAD_K)),
    ]
    for e, fd in enumerate(dists):
        randomize_joint_friction_model(u, torch.tensor([e], device=dev), fd.to_dict(), asset_name="robot")
    readback = {
        "static": wp.to_torch(robot.data.joint_friction_coeff).clone(),
        "viscous": wp.to_torch(robot.data.joint_viscous_friction_coeff).clone(),
    }
    q_default = wp.to_torch(robot.data.default_joint_pos).clone()
    n_phys = [0]

    def step(target: torch.Tensor) -> None:
        robot.set_joint_position_target_index(target=target)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
        n_phys[0] += 1
        if n_phys[0] % decim == 0:  # the interval event runs once per control step
            update_load_dependent_friction(u, None, asset_name="robot")

    def ramp(j: int, speed: float) -> tuple[torch.Tensor, torch.Tensor]:
        base = q_default.clone()
        n = int(SPAN_RAD / speed / dt)
        traj = [base[0, j] - SPAN_RAD / 2 + k * speed * dt for k in range(n)]
        target = base.clone()
        target[:, j] = traj[0]
        for _ in range(int(SETTLE_S / dt)):
            step(target)
        taus: dict[str, list] = {"up": [], "down": []}
        skip = int(0.25 * n)
        for leg, seq in (("up", traj), ("down", list(reversed(traj)))):
            for k, qt in enumerate(seq):
                target[:, j] = qt
                step(target)
                if k < skip:
                    continue
                qd = wp.to_torch(robot.data.joint_vel)[:, j]
                tau = wp.to_torch(robot.data.applied_torque)[:, j]
                sgn = 1.0 if leg == "up" else -1.0
                moving = (sgn * qd > 0.5 * speed) & (sgn * qd < 1.5 * speed)
                taus[leg].append(torch.where(moving, tau, torch.nan))
            for _ in range(int(0.5 / dt)):
                step(target)
        up = torch.stack(taus["up"])
        dn = torch.stack(taus["down"])
        f_hat = (torch.nanmean(up, dim=0) - torch.nanmean(dn, dim=0)) / 2
        mean_abs = (torch.nanmean(up.abs(), dim=0) + torch.nanmean(dn.abs(), dim=0)) / 2
        target = base.clone()
        for _ in range(int(SETTLE_S / dt)):
            step(target)
        return f_hat, mean_abs

    rows = []
    for jn in PROBE_JOINTS:
        j = names.index(jn)
        g = jn.split("_")[1]
        f0 = meas[jn]["static"]
        fs, mabs = ramp(j, 0.05)
        e0, e2 = float(fs[0]), float(fs[2])
        truth_load = f0 + LOAD_K * float(mabs[2])
        sp = (0.5, 1.0, 2.0)
        fr = [ramp(j, v)[0] for v in sp]
        # differential: env1 (viscous) minus env0 (Coulomb only) at the same speed cancels the
        # inertial / PD-tracking part of the up-down torque difference at the faster ramps
        fv = [float(f[1] - f[0]) for f in fr]
        import numpy as np

        slope, icpt = np.polyfit(sp, fv, 1)
        b = VISCOUS[g]
        rows.append({
            "joint": jn,
            "readback_static_e0": float(readback["static"][0, j]),
            "readback_viscous_e1": float(readback["viscous"][1, j]),
            "coulomb_set": f0, "coulomb_est": e0,
            "coulomb_ok": bool(abs(e0 - f0) <= max(0.05, 0.15 * f0)),
            "viscous_set": b, "viscous_est": float(slope), "viscous_intercept": float(icpt), "viscous_f_hat_diff": fv,
            "viscous_ok": bool(abs(slope - b) <= max(0.02, 0.25 * b)),
            "load_set": truth_load, "load_est": e2, "load_mean_abs_tau": float(mabs[2]),
            "load_ok": bool(abs(e2 - truth_load) <= max(0.05, 0.15 * truth_load)),
        })
        print(rows[-1], flush=True)
    verdict = "PASS" if all(r["coulomb_ok"] and r["viscous_ok"] and r["load_ok"] for r in rows) else "FAIL"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"verdict": verdict, "physics_dt": dt, "control_decimation": decim, "rows": rows}, indent=2))
    print(f"VERDICT {verdict}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
