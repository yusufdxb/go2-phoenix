"""Positive control for the joint-friction DR: does Isaac's PhysX joint friction act in N m?

SIM ONLY. Builds the PhoenixVelocity env (vnext3, all other DR neutralised), hangs the
robot (fixed root, 0.8 m), writes a KNOWN Coulomb friction F (static = dynamic) on every
joint through phoenix.velocity.isaac_terms.randomize_joint_friction_nm with a degenerate
range (F, F), one F per env, then repeats the hardware FP method: ramp one joint's PD
target at 0.05 rad/s up then down over the same span and estimate

    Fs_hat = (mean tau_up - mean tau_down) / 2        (gravity cancels over the same span)

using the actuator's applied torque while the joint is moving (|qd| > 0.01 rad/s).
PASS when |Fs_hat - F| <= max(0.05 N m, 0.15 F) for every probed joint and env.

    ~/Sim/isaac-sim-venv/bin/python scripts/isaac_friction_probe.py --out results/friction_dr/isaac_probe.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FRICTIONS_NM = (0.0, 0.3, 0.6, 1.2)
PROBE_JOINTS = ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "RL_calf_joint")
RAMP_RAD_S = 0.05
SPAN_RAD = 0.15
SETTLE_S = 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    from phoenix.sim_app_exit import run_isaac_main

    return run_isaac_main(lambda: _run(args), app, label="isaac_friction_probe")


def _run(args: argparse.Namespace) -> int:
    import gymnasium as gym
    import torch
    import warp as wp

    from phoenix.velocity import env_cfg as pv_env_cfg
    from phoenix.velocity.isaac_terms import randomize_joint_friction_nm
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
    spec = spec.replace(
        domain_randomization=dr,
        sim=dataclasses.replace(spec.sim, num_envs=len(FRICTIONS_NM)),
    )
    cfg = pv_env_cfg.build_velocity_env_cfg(spec)
    cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.8)
    env = gym.make(TASK_ID, cfg=cfg, render_mode=None)
    env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]
    sim = u.sim
    dt = float(sim.get_physics_dt())
    dev = u.device
    names = list(robot.joint_names)
    n_env = u.num_envs

    for i, f in enumerate(FRICTIONS_NM):
        randomize_joint_friction_nm(
            u, torch.tensor([i], device=dev), (f, f), (f, f), asset_name="robot"
        )
    readback = wp.to_torch(robot.data.joint_friction_coeff).clone()
    readback_dyn = wp.to_torch(robot.data.joint_dynamic_friction_coeff).clone()

    q_default = wp.to_torch(robot.data.default_joint_pos).clone()

    def step(target: torch.Tensor) -> None:
        robot.set_joint_position_target_index(target=target)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)

    target = q_default.clone()
    for _ in range(int(SETTLE_S / dt)):
        step(target)

    results = []
    for jn in PROBE_JOINTS:
        j = names.index(jn)
        base = q_default.clone()
        n_half = int(SPAN_RAD / RAMP_RAD_S / dt)
        traj = [base[0, j] - SPAN_RAD / 2 + k * RAMP_RAD_S * dt for k in range(n_half)]
        legs = {"up": traj, "down": list(reversed(traj))}
        # start at the bottom of the span and settle
        target = base.clone()
        target[:, j] = traj[0]
        for _ in range(int(SETTLE_S / dt)):
            step(target)
        taus: dict[str, list[torch.Tensor]] = {"up": [], "down": []}
        for leg in ("up", "down"):
            for qt in legs[leg]:
                target[:, j] = qt
                step(target)
                qd = wp.to_torch(robot.data.joint_vel)[:, j]
                tau = wp.to_torch(robot.data.applied_torque)[:, j]
                moving = (qd.abs() > 0.01) & ((qd > 0) if leg == "up" else (qd < 0))
                taus[leg].append(torch.where(moving, tau, torch.nan))
            for _ in range(int(0.5 / dt)):
                step(target)
        up = torch.nanmean(torch.stack(taus["up"]), dim=0)
        dn = torch.nanmean(torch.stack(taus["down"]), dim=0)
        frac_up = torch.stack(taus["up"]).isfinite().float().mean(dim=0)
        frac_dn = torch.stack(taus["down"]).isfinite().float().mean(dim=0)
        for e in range(n_env):
            f = FRICTIONS_NM[e]
            est = float((up[e] - dn[e]) / 2)
            tol = max(0.05, 0.15 * f)
            results.append(
                {
                    "joint": jn,
                    "set_nm": f,
                    "readback_static": float(readback[e, j]),
                    "readback_dynamic": float(readback_dyn[e, j]),
                    "est_nm": est,
                    "moving_frac_up": float(frac_up[e]),
                    "moving_frac_down": float(frac_dn[e]),
                    "pass": abs(est - f) <= tol,
                }
            )
        # back to default before the next joint
        target = base.clone()
        for _ in range(int(SETTLE_S / dt)):
            step(target)

    verdict = "PASS" if all(r["pass"] for r in results) else "FAIL"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"verdict": verdict, "physics_dt": dt, "results": results}, indent=2)
    )
    for r in results:
        print(
            f"{r['joint']:15s} set {r['set_nm']:.2f}  est {r['est_nm']:+.3f}  "
            f"readback {r['readback_static']:.2f}/{r['readback_dynamic']:.2f}  "
            f"moving {r['moving_frac_up']:.2f}/{r['moving_frac_down']:.2f}  {'ok' if r['pass'] else 'MISS'}",
            flush=True,
        )
    print(f"VERDICT {verdict}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
