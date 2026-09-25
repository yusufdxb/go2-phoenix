"""Held-out evaluation of a PhoenixVelocity checkpoint with DR on.

Runs the actual PhoenixVelocity env (proprio-only 45-D actor, real DR), not a
replay, so numbers reflect what training actually produced. Commands are the
env's own curriculum-final sampling (DR on); results are BINNED by commanded
speed (stand / slow / fast) rather than forced to an exact grid, because
forcing exact commands means reaching into the UniformVelocityCommand term's
internal buffer, which is more fragile than reading what the env already
samples. Flagged here rather than silently simplified.

Metrics:
  * velocity tracking error, binned by |cmd| (stand / slow / fast)
  * standing base height at near-zero command
  * fall rate (trunk_contact + bad_orientation termination fraction)
  * fraction of joint-steps within 5% of a soft limit
  * torque headroom per actuator group: peak |tau| / effort_limit
  * action-clip rate: fraction of |raw action| components > 1.0 (pre rsl_rl clip)

2026-09-25 addition: the random-command joint_near_limit_fraction above
averages over all 12 joints and whatever commands the env happens to sample,
which HIDES a real, narrow failure the sim2sim agent found directly: with
random commands most steps do not stress any one joint, so an isolated
per-joint, per-command violation gets diluted into an unremarkable-looking
average. seed42@3000 drove FR/RL calves to their HARD position limit during
counter-clockwise yaw (+0.6 rad/s, worse at +0.8), 0.25-0.31 rad past the
stop. Added: per-calf-joint hard-limit proximity (fraction of steps ANY calf
is within 0.05 rad of its HARD limit, not the 0.9-factor soft one), measured
under a FIXED-command sweep (yaw +0.6/+0.8/-0.6 rad/s, vx 0.5 m/s) that
forces the exact commands the gate's fixed scenarios use, not random
sampling, so this eval and the gate measure the same thing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("phoenix.velocity.eval")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a PhoenixVelocity checkpoint.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=1000, help="policy steps (50 Hz -> 20s)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--headless", action="store_true", default=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    args = parse_args(argv)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    from phoenix.sim_app_exit import run_isaac_main

    return run_isaac_main(lambda: _run(args, simulation_app), simulation_app, label="eval_velocity")


def _run(args: argparse.Namespace, simulation_app) -> int:  # noqa: ANN001
    import dataclasses

    import gymnasium as gym
    import torch

    from phoenix.velocity import env_cfg as pv_env_cfg
    from phoenix.velocity.mdp import as_torch
    from phoenix.velocity.spec import TASK_ID, default_spec

    pv_env_cfg.register()

    spec = default_spec().replace(
        sim=dataclasses.replace(default_spec().sim, num_envs=args.num_envs)
    )
    env_cfg = pv_env_cfg.build_velocity_env_cfg(spec)
    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
    unwrapped = env.unwrapped

    ckpt = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
    # rsl_rl >=3.0 stores actor_state_dict directly (mlp.N.weight/bias plus
    # distribution.std_param, which is excluded below by the ".weight" filter).
    # Older rsl_rl used model_state_dict with an "actor." prefix. Mirrors
    # phoenix.sim2real.export._extract_actor_state_dict (deploy-owned, not
    # imported here since sim2real is a different file-ownership boundary).
    if "actor_state_dict" in ckpt:
        actor_sd = ckpt["actor_state_dict"]
    elif "model_state_dict" in ckpt:
        actor_sd = {
            k[len("actor.") :]: v
            for k, v in ckpt["model_state_dict"].items()
            if k.startswith("actor.")
        }
    else:
        raise KeyError(f"checkpoint has no actor weights; keys={list(ckpt)}")
    layer_keys = sorted(
        (k for k in actor_sd if k.endswith(".weight") and "mlp" in k),
        key=lambda k: int(k.split(".")[1]),
    )
    obs_dim = int(actor_sd[layer_keys[0]].shape[1])
    action_dim = int(actor_sd[layer_keys[-1]].shape[0])
    hidden = [int(actor_sd[k].shape[0]) for k in layer_keys[:-1]]
    actor = torch.nn.Sequential(
        *sum(
            (
                [torch.nn.Linear(([obs_dim] + hidden)[i], (hidden + [action_dim])[i])]
                + ([torch.nn.ELU()] if i < len(hidden) else [])
                for i in range(len(hidden) + 1)
            ),
            [],
        )
    ).to("cuda:0")
    state = {}
    lin_idx = 0
    for name, mod in actor.named_children():
        if isinstance(mod, torch.nn.Linear):
            src = layer_keys[lin_idx]
            state[f"{name}.weight"] = actor_sd[src]
            state[f"{name}.bias"] = actor_sd[src.replace(".weight", ".bias")]
            lin_idx += 1
    actor.load_state_dict(state)
    actor.eval()

    limits = {}
    for aname, act in unwrapped.scene["robot"].actuators.items():
        eff = act.effort_limit
        limits[aname] = float(eff.flatten()[0].item()) if hasattr(eff, "flatten") else float(eff)

    obs_dict, _ = env.reset()
    n = unwrapped.num_envs

    sum_lin_err = {"stand": 0.0, "slow": 0.0, "fast": 0.0}
    cnt = {"stand": 0, "slow": 0, "fast": 0}
    sum_yaw_err = {"stand": 0.0, "slow": 0.0, "fast": 0.0}
    heights_stand = []
    near_limit_frac_sum = 0.0
    clip_rate_sum = 0.0
    torque_peak = {k: 0.0 for k in limits}
    terminations = {"trunk_contact": 0, "bad_orientation": 0, "time_out": 0, "numerical_failure": 0}
    total_episodes = 0
    steps_done = 0

    with torch.no_grad():
        for _ in range(args.num_steps):
            obs = obs_dict["policy"]
            action = actor(obs)
            # Diagnostic rate at a fixed |a|>1 threshold,
            # separate from the clip actually applied below.
            clip_rate_sum += float((action.abs() > 1.0).float().mean())
            # Step with the SAME clip the policy was trained/deployed with
            # (spec.action.clip_actions), not a hardcoded 1.0. A mismatched
            # tight clip here silently changes the env dynamics away from what
            # training actually saw, especially for late-training checkpoints
            # where the action distribution can be far outside [-1, 1].
            action_clipped = action.clamp(-spec.action.clip_actions, spec.action.clip_actions)
            obs_dict, _rew, terminated, truncated, extras = env.step(action_clipped)
            steps_done += 1

            cmd = unwrapped.command_manager.get_command("base_velocity")
            v = as_torch(unwrapped.scene["robot"].data.root_lin_vel_b)
            w = as_torch(unwrapped.scene["robot"].data.root_ang_vel_b)
            lin_err = torch.linalg.norm(cmd[:, :2] - v[:, :2], dim=1)
            yaw_err = (cmd[:, 2] - w[:, 2]).abs()
            lin_mag = torch.linalg.norm(cmd[:, :2], dim=1)
            stand = lin_mag < 0.15
            fast = lin_mag > 0.5
            slow = ~stand & ~fast
            for name, mask in (("stand", stand), ("slow", slow), ("fast", fast)):
                if int(mask.sum()) > 0:
                    sum_lin_err[name] += float(lin_err[mask].sum())
                    sum_yaw_err[name] += float(yaw_err[mask].sum())
                    cnt[name] += int(mask.sum())
            if int(stand.sum()) > 0:
                h = as_torch(unwrapped.scene["robot"].data.root_pos_w)[:, 2]
                heights_stand.extend(h[stand].tolist())

            q = as_torch(unwrapped.scene["robot"].data.joint_pos)
            soft = as_torch(unwrapped.scene["robot"].data.soft_joint_pos_limits)
            rng = (soft[..., 1] - soft[..., 0]).clamp(min=1e-6)
            near = ((q - soft[..., 0]) / rng < 0.05) | ((soft[..., 1] - q) / rng < 0.05)
            near_limit_frac_sum += float(near.float().mean())

            applied = as_torch(unwrapped.scene["robot"].data.applied_torque)
            for aname, act in unwrapped.scene["robot"].actuators.items():
                ids = act.joint_indices
                peak = float(applied[:, ids].abs().max())
                torque_peak[aname] = max(torque_peak[aname], peak)

            done = terminated | truncated
            if bool(done.any()):
                total_episodes += int(done.sum())
                term = unwrapped.termination_manager
                for name in terminations:
                    try:
                        terminations[name] += int(term.get_term(name)[done].sum())
                    except Exception:  # noqa: BLE001
                        pass

    result = {
        "checkpoint": str(args.checkpoint),
        "num_envs": n,
        "num_steps": steps_done,
        "total_episodes": total_episodes,
        "velocity_tracking": {
            k: {
                "mean_lin_err_mps": (sum_lin_err[k] / cnt[k]) if cnt[k] else None,
                "mean_yaw_err_rad_s": (sum_yaw_err[k] / cnt[k]) if cnt[k] else None,
                "samples": cnt[k],
            }
            for k in ("stand", "slow", "fast")
        },
        "standing_base_height_m": {
            "mean": (sum(heights_stand) / len(heights_stand)) if heights_stand else None,
            "n": len(heights_stand),
        },
        "fall_rate": (
            (
                terminations["trunk_contact"]
                + terminations["bad_orientation"]
                + terminations["numerical_failure"]
            )
            / max(total_episodes, 1)
        ),
        "termination_breakdown": terminations,
        "joint_near_limit_fraction": near_limit_frac_sum / max(steps_done, 1),
        "torque_headroom": {
            k: {
                "peak_torque_nm": torque_peak[k],
                "limit_nm": limits[k],
                "peak_over_limit": torque_peak[k] / limits[k],
            }
            for k in limits
        },
        "action_clip_rate": clip_rate_sum / max(steps_done, 1),
    }
    result["fixed_command_sweep"] = _fixed_command_sweep(unwrapped, actor, spec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    logger.info("eval written -> %s", args.out)
    env.close()
    return 0


def _fixed_command_sweep(unwrapped, actor, spec, num_steps: int = 250) -> dict:
    """Force fixed commands (matching the gate's scenarios) and measure
    per-calf-joint HARD-limit proximity directly, instead of relying on
    random commands to happen to stress the right joint. Writes directly to
    the UniformVelocityCommand term's vel_command_b buffer every step so the
    env's own resampling never overrides it mid-sweep.
    """
    import torch

    from phoenix.velocity.contract import JOINT_ORDER
    from phoenix.velocity.mdp import as_torch

    calf_idx = [i for i, j in enumerate(JOINT_ORDER) if j.endswith("_calf_joint")]
    robot = unwrapped.scene["robot"]
    hard_limits = as_torch(robot.data.joint_pos_limits)  # (envs, joints, 2)
    cmd_term = unwrapped.command_manager.get_term("base_velocity")
    device = unwrapped.device
    n = unwrapped.num_envs

    scenarios = {
        "yaw_p0.6": (0.0, 0.0, 0.6),
        "yaw_p0.8": (0.0, 0.0, 0.8),
        "yaw_m0.6": (0.0, 0.0, -0.6),
        "vx_0.5": (0.5, 0.0, 0.0),
    }
    out: dict = {}
    for name, (vx, vy, wz) in scenarios.items():
        obs_dict, _ = unwrapped.reset()
        fixed = torch.tensor([vx, vy, wz], device=device).expand(n, 3).clone()
        cmd_term.vel_command_b[:] = fixed
        near_hard_any_step = torch.zeros(n, dtype=torch.bool, device=device)
        max_overshoot_rad = 0.0
        with torch.no_grad():
            for _ in range(num_steps):
                obs = obs_dict["policy"]
                action = actor(obs)
                action_clipped = action.clamp(-spec.action.clip_actions, spec.action.clip_actions)
                obs_dict, _rew, terminated, truncated, extras = unwrapped.step(action_clipped)
                cmd_term.vel_command_b[:] = fixed  # re-force every step
                q = as_torch(robot.data.joint_pos)[:, calf_idx]
                lo = hard_limits[:, calf_idx, 0]
                hi = hard_limits[:, calf_idx, 1]
                near = ((q - lo).abs() < 0.05) | ((hi - q).abs() < 0.05)
                near_hard_any_step |= near.any(dim=1)
                overshoot = torch.clamp(lo - q, min=0.0) + torch.clamp(q - hi, min=0.0)
                max_overshoot_rad = max(max_overshoot_rad, float(overshoot.max()))
        out[name] = {
            "fraction_envs_calf_near_hard_limit": float(near_hard_any_step.float().mean()),
            "max_calf_hard_limit_overshoot_rad": max_overshoot_rad,
        }
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    sys.exit(main())
