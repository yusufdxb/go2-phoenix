"""Held-out evaluation of a PhoenixVelocity checkpoint with DR on.

Runs the actual PhoenixVelocity env (proprio-only 45-D actor, real DR), not a
replay, so numbers reflect what training actually produced. Commands are the
env's own curriculum-final sampling (DR on); results are BINNED by commanded
speed (stand / slow / fast) rather than forced to an exact grid, because
forcing exact commands means reaching into the UniformVelocityCommand term's
internal buffer, which is more fragile than reading what the env already
samples. Flagged here rather than silently simplified.

Metrics (see task spec item 7):
  * velocity tracking error, binned by |cmd| (stand / slow / fast)
  * standing base height at near-zero command
  * fall rate (trunk_contact + bad_orientation termination fraction)
  * fraction of joint-steps within 5% of a soft limit
  * torque headroom per actuator group: peak |tau| / effort_limit
  * action-clip rate: fraction of |raw action| components > 1.0 (pre rsl_rl clip)
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
    from phoenix.velocity.spec import TASK_ID, default_spec

    pv_env_cfg.register()

    spec = default_spec().replace(sim=dataclasses.replace(default_spec().sim, num_envs=args.num_envs))
    env_cfg = pv_env_cfg.build_velocity_env_cfg(spec)
    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
    unwrapped = env.unwrapped

    ckpt = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
    actor_sd = {k[len("actor.") :]: v for k, v in ckpt["model_state_dict"].items() if k.startswith("actor.")}
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
    device = unwrapped.device
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
            clip_rate_sum += float((action.abs() > 1.0).float().mean())
            action_clipped = action.clamp(-1.0, 1.0)
            obs_dict, _rew, terminated, truncated, extras = env.step(action_clipped)
            steps_done += 1

            cmd = unwrapped.command_manager.get_command("base_velocity")
            v = unwrapped.scene["robot"].data.root_lin_vel_b
            w = unwrapped.scene["robot"].data.root_ang_vel_b
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
                h = unwrapped.scene["robot"].data.root_pos_w[:, 2]
                heights_stand.extend(h[stand].tolist())

            q = unwrapped.scene["robot"].data.joint_pos
            soft = unwrapped.scene["robot"].data.soft_joint_pos_limits
            rng = (soft[..., 1] - soft[..., 0]).clamp(min=1e-6)
            near = ((q - soft[..., 0]) / rng < 0.05) | ((soft[..., 1] - q) / rng < 0.05)
            near_limit_frac_sum += float(near.float().mean())

            applied = unwrapped.scene["robot"].data.applied_torque
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
            (terminations["trunk_contact"] + terminations["bad_orientation"] + terminations["numerical_failure"])
            / max(total_episodes, 1)
        ),
        "termination_breakdown": terminations,
        "joint_near_limit_fraction": near_limit_frac_sum / max(steps_done, 1),
        "torque_headroom": {
            k: {"peak_torque_nm": torque_peak[k], "limit_nm": limits[k], "peak_over_limit": torque_peak[k] / limits[k]}
            for k in limits
        },
        "action_clip_rate": clip_rate_sum / max(steps_done, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    logger.info("eval written -> %s", args.out)
    env.close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    sys.exit(main())
