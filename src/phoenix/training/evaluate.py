"""Headless rollout + video capture for a trained checkpoint.

Used by:
* ``scripts/demo.sh`` (via :mod:`phoenix.demo.benchmark`) to record
  ``sim_baseline.mp4`` / ``sim_adapted.mp4``.
* Unit smoke tests to sanity-check checkpoint loading.

Produces a ``metrics.json`` alongside the video with success rate,
average tracking error, and failure count under the Phoenix
failure-detection rules.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("phoenix.training.evaluate")


@dataclass
class RolloutMetrics:
    num_episodes: int
    mean_episode_return: float
    mean_episode_length_s: float
    success_rate: float
    failure_rate: float
    mean_lin_vel_error: float
    mean_ang_vel_error: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a Phoenix checkpoint and record video.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-config", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--num-episodes", type=int, default=64)
    p.add_argument("--video-out", type=Path, default=None)
    p.add_argument(
        "--video-length", type=int, default=500, help="Video length in env steps (50 Hz)"
    )
    p.add_argument("--metrics-out", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=args.video_out is not None)
    simulation_app = app_launcher.app
    try:
        return _run(args, simulation_app)
    finally:
        simulation_app.close()


def _run(args: argparse.Namespace, simulation_app) -> int:  # noqa: ANN001
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from phoenix.sim_env import build_env_cfg, load_layered_config

    env_cfg_loaded = load_layered_config(args.env_config)
    env_cfg = build_env_cfg(env_cfg_loaded)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed

    task_name = env_cfg_loaded.to_container()["env"]["task_name"]
    render_mode = "rgb_array" if args.video_out else None
    env = gym.make(task_name, cfg=env_cfg, render_mode=render_mode)

    if args.video_out:
        args.video_out.parent.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(args.video_out.parent),
            name_prefix=args.video_out.stem,
            episode_trigger=lambda ep: ep == 0,
            video_length=args.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    # Use the upstream rsl_rl cfg to stay compatible with whatever version
    # of rsl_rl is installed (actor/critic cfg shape changed in 3.0).
    import importlib.metadata as metadata

    import yaml  # noqa: E402
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    from phoenix.training.agent_cfg import build_runner_cfg

    eval_yaml = {
        "run": {
            "name": "eval",
            "output_dir": "/tmp",
            "log_interval": 1,
            "save_interval": 1,
            "max_iterations": 1,
            "seed": args.seed,
            "device": args.device,
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "runner": {"num_steps_per_env": 24, "empirical_normalization": True},
    }
    _ = yaml  # quiet unused-import warning (yaml reserved for future config loading)
    runner_cfg = build_runner_cfg(eval_yaml, task_name)
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(args.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=args.device)

    # ---- Rollout -----------------------------------------------------------
    obs, _ = env.get_observations()
    episode_return = torch.zeros(args.num_envs, device=args.device)
    episode_length = torch.zeros(args.num_envs, device=args.device)

    returns: list[float] = []
    lengths: list[float] = []
    successes = 0
    failures = 0
    lin_err_acc = 0.0
    ang_err_acc = 0.0
    n_steps = 0

    dt_ctrl = env_cfg.decimation * env_cfg.sim.dt  # seconds per env step

    with torch.inference_mode():
        while len(returns) < args.num_episodes:
            actions = policy(obs)
            obs, reward, dones, extras = env.step(actions)
            episode_return += reward
            episode_length += 1
            n_steps += 1

            # tracking error from the env's command buffer, if exposed
            unwrapped = env.unwrapped
            if hasattr(unwrapped, "command_manager"):
                cmd = unwrapped.command_manager.get_command("base_velocity")
                base_lin = unwrapped.scene["robot"].data.root_lin_vel_b[:, :2]
                base_ang_z = unwrapped.scene["robot"].data.root_ang_vel_b[:, 2]
                lin_err_acc += float(
                    torch.mean(torch.linalg.norm(cmd[:, :2] - base_lin, dim=-1)).item()
                )
                ang_err_acc += float(torch.mean(torch.abs(cmd[:, 2] - base_ang_z)).item())

            done_idx = dones.nonzero(as_tuple=False).flatten()
            if len(done_idx) > 0:
                for i in done_idx.tolist():
                    returns.append(float(episode_return[i].item()))
                    ep_len = float(episode_length[i].item())
                    lengths.append(ep_len * dt_ctrl)
                    # termination reason via the time_out buffer
                    time_out = bool(extras.get("time_outs", torch.zeros_like(dones))[i].item())
                    if time_out:
                        successes += 1
                    else:
                        failures += 1
                    episode_return[i] = 0.0
                    episode_length[i] = 0.0

    n_eps = max(len(returns), 1)
    metrics = RolloutMetrics(
        num_episodes=n_eps,
        mean_episode_return=float(np.mean(returns)),
        mean_episode_length_s=float(np.mean(lengths)),
        success_rate=successes / n_eps,
        failure_rate=failures / n_eps,
        mean_lin_vel_error=lin_err_acc / max(n_steps, 1),
        mean_ang_vel_error=ang_err_acc / max(n_steps, 1),
    )
    logger.info("Metrics: %s", metrics)

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(asdict(metrics), indent=2))

    env.close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    sys.exit(main())
