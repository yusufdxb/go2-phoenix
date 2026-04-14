"""Baseline + adaptation PPO training loop.

Invoked from ``scripts/train.sh`` inside Isaac Lab's Python context. Loads a
layered env YAML (e.g. ``configs/env/rough.yaml``) and a training YAML
(``configs/train/ppo.yaml``), builds the env + rsl_rl runner, then writes
checkpoints to ``checkpoints/<run_name>/``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("phoenix.training.ppo_runner")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Phoenix GO2 policy with PPO (rsl_rl).")
    p.add_argument(
        "--config", type=Path, required=True, help="Training YAML (e.g. configs/train/ppo.yaml)"
    )
    p.add_argument(
        "--env-config",
        type=Path,
        default=None,
        help="Override env YAML (else taken from training YAML)",
    )
    p.add_argument("--num-envs", type=int, default=None, help="Override number of parallel envs")
    p.add_argument(
        "--max-iterations", type=int, default=None, help="Override training iteration count"
    )
    p.add_argument(
        "--resume", type=Path, default=None, help="Resume from an existing checkpoint .pt"
    )
    p.add_argument(
        "--load-optimizer", action="store_true", help="Load optimizer state when resuming"
    )
    p.add_argument(
        "--headless", action="store_true", default=True, help="Run without GUI (default)"
    )
    p.add_argument("--device", type=str, default=None, help="Override torch device (e.g. cuda:0)")
    p.add_argument("--seed", type=int, default=None, help="Override seed")
    p.add_argument("--run-name", type=str, default=None, help="Override experiment/run name")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    args = parse_args(argv)
    logger.info("parsed args: %s", args)

    # Launch Isaac Sim's AppLauncher *before* importing anything from isaaclab.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app
    logger.info("Isaac Sim launched")

    try:
        return _run(args, simulation_app)
    finally:
        simulation_app.close()


def _run(args: argparse.Namespace, simulation_app) -> int:  # noqa: ANN001
    import importlib.metadata as metadata

    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from omegaconf import OmegaConf
    from rsl_rl.runners import OnPolicyRunner

    from phoenix.sim_env import build_env_cfg, load_layered_config
    from phoenix.training.agent_cfg import build_runner_cfg

    # ---- Load training YAML ------------------------------------------------
    train_cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    assert isinstance(train_cfg, dict)

    # CLI overrides
    if args.num_envs is not None:
        train_cfg["run"].setdefault("_cli_num_envs", args.num_envs)
    if args.max_iterations is not None:
        train_cfg["run"]["max_iterations"] = args.max_iterations
    if args.device is not None:
        train_cfg["run"]["device"] = args.device
    if args.seed is not None:
        train_cfg["run"]["seed"] = args.seed
    if args.run_name is not None:
        train_cfg["run"]["name"] = args.run_name

    # ---- Build env cfg -----------------------------------------------------
    env_cfg_path = args.env_config or Path(train_cfg["env"]["config"])
    env_cfg_loaded = load_layered_config(env_cfg_path)
    env_cfg = build_env_cfg(env_cfg_loaded)

    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = train_cfg["run"]["device"]
    env_cfg.seed = train_cfg["run"]["seed"]

    task_name = env_cfg_loaded.to_container()["env"]["task_name"]

    # ---- Log directory -----------------------------------------------------
    run_name = train_cfg["run"]["name"]
    log_root = Path(train_cfg["run"]["output_dir"]) / run_name
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = log_root / stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run log dir: %s", log_dir)

    # Persist the exact configs used (replay-friendly)
    shutil.copy(args.config, log_dir / "train.yaml")
    shutil.copy(env_cfg_path, log_dir / "env.yaml")

    # ---- Create env + runner -----------------------------------------------
    print("[phoenix] before gym.make", flush=True)
    env = gym.make(task_name, cfg=env_cfg, render_mode=None)
    print("[phoenix] after gym.make, wrapping", flush=True)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    print("[phoenix] wrapped; building runner cfg", flush=True)

    runner_cfg = build_runner_cfg(train_cfg, task_name)
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, metadata.version("rsl-rl-lib"))
    print("[phoenix] creating OnPolicyRunner", flush=True)
    runner = OnPolicyRunner(
        env, runner_cfg.to_dict(), log_dir=str(log_dir), device=runner_cfg.device
    )
    print("[phoenix] runner ready", flush=True)

    if args.resume is not None:
        logger.info("Resuming from checkpoint: %s", args.resume)
        from phoenix.training.checkpoint import load_runner_checkpoint

        ckpt_info = load_runner_checkpoint(
            runner,
            args.resume,
            load_actor=True,
            load_critic=True,
            load_optimizer=bool(args.load_optimizer),
            load_iteration=False,
        )
        if not ckpt_info.get("actor_match", False):
            raise RuntimeError(f"Actor weights did not round-trip from {args.resume}: {ckpt_info}")

    # ---- Train -------------------------------------------------------------
    start = time.time()
    try:
        runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        logger.warning("Interrupted — writing final checkpoint.")
    elapsed = time.time() - start
    logger.info("Training wall-time: %.1fs", elapsed)

    # Save a stable "latest.pt" symlink so deploy/adapt don't need timestamps.
    latest = log_root / "latest.pt"
    # Numeric sort by the iteration number so model_499.pt > model_50.pt.
    ckpts = sorted(
        log_dir.glob("model_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if ckpts:
        final = ckpts[-1]
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(final.resolve())
        logger.info("latest.pt -> %s", final.name)

    env.close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    sys.exit(main())
