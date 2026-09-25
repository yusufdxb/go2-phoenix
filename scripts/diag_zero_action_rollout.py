"""Zero-action rollout diagnostic: isolate why Phoenix falls near-instantly vs stock.

No policy involved: every step commands action=0 (target = default_q exactly).
A correctly-configured quadruped at its trained default pose, with correct PD
gains, should just stand there. This checks, per step, for the first ~200
steps of a fresh 64-env rollout:
  * actuator stiffness/damping actually read off the constructed actuators
    AFTER reset (the known randomize_actuator_gains-zeroes-explicit-gains
    trap: phoenix.sim_env scale_explicit_actuator_gains was written BECAUSE
    of this bug; confirms it did not regress here)
  * root height and per-body contact force (base, Head_upper, Head_lower
    separately) so a false-positive "Head_.*" trigger is visible on its own
  * which termination term fires first and at which step, per env
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("phoenix.velocity.diag_zero_action")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--zero-reset-noise", action="store_true", help="also zero reset perturbation ranges")
    p.add_argument("--action-std", type=float, default=0.0, help="Gaussian noise std added to the zero action (0 = literally zero)")
    p.add_argument("--headless", action="store_true", default=True)
    return p.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    args = parse_args(argv)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app
    from phoenix.sim_app_exit import run_isaac_main

    return run_isaac_main(lambda: _run(args, simulation_app), simulation_app, label="diag_zero_action")


def _run(args, simulation_app):  # noqa: ANN001
    import dataclasses

    import gymnasium as gym
    import torch

    from phoenix.velocity import env_cfg as pv_env_cfg
    from phoenix.velocity.mdp import as_torch
    from phoenix.velocity.spec import TASK_ID, default_spec

    pv_env_cfg.register()

    spec = default_spec().replace(sim=dataclasses.replace(default_spec().sim, num_envs=args.num_envs))
    if args.zero_reset_noise:
        dr = spec.domain_randomization
        spec = spec.replace(
            domain_randomization=dataclasses.replace(
                dr,
                reset_lin_vel=(0.0, 0.0),
                reset_ang_vel=(0.0, 0.0),
                reset_joint_pos_offset=(0.0, 0.0),
                push_enabled=False,
            )
        )
    env_cfg = pv_env_cfg.build_velocity_env_cfg(spec)
    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
    unwrapped = env.unwrapped

    obs, _ = env.reset()
    device = unwrapped.device
    n = unwrapped.num_envs

    robot = unwrapped.scene["robot"]
    actuator_report = {}
    for name, act in robot.actuators.items():
        stiff = as_torch(act.stiffness)
        damp = as_torch(act.damping)
        eff = act.effort_limit
        eff_t = as_torch(eff) if not isinstance(eff, (int, float)) else torch.tensor([float(eff)])
        actuator_report[name] = {
            "stiffness_mean": float(stiff.mean()),
            "stiffness_min": float(stiff.min()),
            "stiffness_max": float(stiff.max()),
            "damping_mean": float(damp.mean()),
            "damping_min": float(damp.min()),
            "damping_max": float(damp.max()),
            "effort_limit_mean": float(eff_t.mean()),
        }

    root_height_at_reset = as_torch(robot.data.root_pos_w)[:, 2].clone()
    default_q = as_torch(robot.data.default_joint_pos).clone()
    actual_q_at_reset = as_torch(robot.data.joint_pos).clone()
    joint_offset_at_reset = (actual_q_at_reset - default_q).abs()

    contact_sensor = unwrapped.scene.sensors["contact_forces"]
    body_names = contact_sensor.body_names
    base_idx = [i for i, b in enumerate(body_names) if b == "base"]
    head_idx = [i for i, b in enumerate(body_names) if b.startswith("Head_")]

    term_mgr = unwrapped.termination_manager
    still_alive = torch.ones(n, dtype=torch.bool, device=device)
    first_term_step = torch.full((n,), -1, dtype=torch.long, device=device)
    first_term_reason = ["none"] * n
    max_base_force = torch.zeros(n, device=device)
    max_head_force = torch.zeros(n, device=device)
    min_height = root_height_at_reset.clone()

    action_dim = unwrapped.action_manager.total_action_dim
    zero_action = torch.zeros(n, action_dim, device=device)

    for step in range(args.num_steps):
        step_action = zero_action
        if args.action_std > 0:
            step_action = torch.randn(n, action_dim, device=device) * args.action_std
        obs, rew, terminated, truncated, extras = env.step(step_action)
        forces = as_torch(contact_sensor.data.net_forces_w_history)  # (envs, hist, bodies, 3)
        peak = torch.linalg.norm(forces, dim=-1).amax(dim=1)  # (envs, bodies)
        if base_idx:
            max_base_force = torch.maximum(max_base_force, peak[:, base_idx].amax(dim=1))
        if head_idx:
            max_head_force = torch.maximum(max_head_force, peak[:, head_idx].amax(dim=1))
        min_height = torch.minimum(min_height, as_torch(robot.data.root_pos_w)[:, 2])

        done = (terminated | truncated) & still_alive
        if bool(done.any()):
            ids = done.nonzero(as_tuple=True)[0]
            first_term_step[ids] = step
            for name in ("trunk_contact", "bad_orientation", "numerical_failure", "time_out"):
                try:
                    fired = term_mgr.get_term(name)
                except Exception:  # noqa: BLE001
                    continue
                for i in ids.tolist():
                    if bool(fired[i]) and first_term_reason[i] == "none":
                        first_term_reason[i] = name
            still_alive = still_alive & ~done

    result = {
        "num_envs": n,
        "num_steps": args.num_steps,
        "zero_reset_noise": args.zero_reset_noise,
        "action_std": args.action_std,
        "actuators_after_reset": actuator_report,
        "root_height_at_reset": {
            "mean": float(root_height_at_reset.mean()),
            "min": float(root_height_at_reset.min()),
            "max": float(root_height_at_reset.max()),
        },
        "joint_offset_at_reset_rad": {
            "mean": float(joint_offset_at_reset.mean()),
            "max": float(joint_offset_at_reset.max()),
        },
        "min_height_over_rollout": {
            "mean": float(min_height.mean()),
            "min": float(min_height.min()),
        },
        "max_base_contact_force_n": {"mean": float(max_base_force.mean()), "max": float(max_base_force.max())},
        "max_head_contact_force_n": {"mean": float(max_head_force.mean()), "max": float(max_head_force.max())},
        "body_names": body_names,
        "fraction_terminated_by_step_end": float((first_term_step >= 0).float().mean()),
        "first_term_step_mean_over_terminated": (
            float(first_term_step[first_term_step >= 0].float().mean())
            if bool((first_term_step >= 0).any())
            else None
        ),
        "first_term_step_min": int(first_term_step[first_term_step >= 0].min()) if bool((first_term_step >= 0).any()) else None,
        "first_term_reason_counts": {
            r: first_term_reason.count(r) for r in set(first_term_reason)
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    logger.info("diag written -> %s", args.out)
    print(json.dumps(result, indent=2), flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
