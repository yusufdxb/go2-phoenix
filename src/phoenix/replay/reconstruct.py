"""Reconstruct a real-world failure trajectory in Isaac Sim.

Spawns ``per_trajectory`` copies of the Phoenix GO2 env, each with
perturbed physics drawn from the variation sampler, initialised from the
logged pre-failure state. Used by:

* ``scripts/replay.sh`` — interactive inspection of a single failure.
* :mod:`phoenix.adaptation.fine_tune` — as a training-time seed source.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("phoenix.replay.reconstruct")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay a real failure trajectory in Isaac Sim.")
    p.add_argument("--trajectory", type=Path, required=True, help="Parquet from real_world logger")
    p.add_argument("--variations-config", type=Path, required=True)
    p.add_argument("--env-config", type=Path, default=Path("configs/env/rough.yaml"))
    p.add_argument("--variations", type=int, default=None, help="Override per_trajectory")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output-dir", type=Path, default=Path("media/renders/replay"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    args = parse_args(argv)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app
    try:
        return _run(args, simulation_app)
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


def _run(args: argparse.Namespace, simulation_app) -> int:  # noqa: ANN001
    import gymnasium as gym
    import torch
    import yaml

    from phoenix.replay import VariationSampler, load_initial_state
    from phoenix.sim_env import build_env_cfg, load_layered_config

    var_cfg = yaml.safe_load(args.variations_config.read_text())
    n_variations = args.variations or int(var_cfg["variations"]["per_trajectory"])

    initial = load_initial_state(args.trajectory, row=0)
    logger.info("Loaded initial state from %s", args.trajectory)

    sampler = VariationSampler(
        bounds=var_cfg["variations"]["dr_bounds"],
        seed=int(var_cfg["variations"]["seed"]),
    )
    variations = sampler.sample(n_variations)
    logger.info("Sampled %d variations", len(variations))

    env_cfg_loaded = load_layered_config(args.env_config)
    env_cfg = build_env_cfg(env_cfg_loaded)
    env_cfg.scene.num_envs = n_variations
    env_cfg.sim.device = args.device

    task_name = env_cfg_loaded.to_container()["env"]["task_name"]
    env = gym.make(task_name, cfg=env_cfg, render_mode=None)

    # ---- Apply per-env variation & initial state ---------------------------
    env.reset()  # drive the gym wrapper out of ResetNeeded state
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]

    # Broadcast the logged initial state into every env.
    num_envs = n_variations
    pos = torch.as_tensor(initial.base_pos, device=args.device).repeat(num_envs, 1)
    quat = torch.as_tensor(initial.base_quat, device=args.device).repeat(num_envs, 1)
    jpos = torch.as_tensor(initial.joint_pos, device=args.device).repeat(num_envs, 1)
    jvel = torch.as_tensor(initial.joint_vel, device=args.device).repeat(num_envs, 1)

    robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
    robot.write_joint_state_to_sim(jpos, jvel)

    # ---- Roll out the horizon ---------------------------------------------
    horizon_s = float(var_cfg["replay"]["horizon_s"])
    ctrl_dt = float(var_cfg["replay"]["dt"])
    n_steps = int(horizon_s / ctrl_dt)

    # Zero-action rollout: the purpose is *state discovery under perturbation*,
    # not policy evaluation. A subsequent training run will replace the zero
    # action with the current policy once the adaptation loop is wired up.
    action = torch.zeros(num_envs, int(env.action_space.shape[-1]), device=args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"variations": [v.__dict__ for v in variations], "horizon_steps": n_steps}

    for _ in range(n_steps):
        env.step(action)

    (args.output_dir / "replay_summary.json").write_text(str(summary))
    logger.info("Wrote replay summary to %s", args.output_dir)
    env.close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    sys.exit(main())
