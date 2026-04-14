"""Fine-tune a baseline policy with failure-seeded curriculum.

Invoked from ``scripts/adapt.sh`` inside Isaac Lab's Python context. The
inner PPO training loop matches :mod:`phoenix.training.ppo_runner` so
successes reproduce; what differs is the curriculum that seeds a fraction
of environment resets from recorded real-world failure Parquets.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("phoenix.adaptation.fine_tune")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a Phoenix policy with failure curriculum.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume", type=Path, default=None, help="Override resume path")
    p.add_argument("--trajectory-dir", type=Path, default=None, help="Override curriculum dir")
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--headless", action="store_true", default=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app
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

    from phoenix.adaptation.curriculum import FailureCurriculum, TrajectoryPool
    from phoenix.sim_env import build_env_cfg, load_layered_config
    from phoenix.training.agent_cfg import build_runner_cfg

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    assert isinstance(cfg, dict)

    if args.max_iterations is not None:
        cfg["run"]["max_iterations"] = args.max_iterations
    if args.device is not None:
        cfg["run"]["device"] = args.device

    env_cfg_path = Path(cfg["env"]["config"])
    env_cfg_loaded = load_layered_config(env_cfg_path)
    env_cfg = build_env_cfg(env_cfg_loaded)
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = cfg["run"]["device"]
    env_cfg.seed = int(cfg["run"].get("seed", 42))

    task_name = env_cfg_loaded.to_container()["env"]["task_name"]

    # ---- Failure curriculum ----------------------------------------------
    traj_dir = Path(args.trajectory_dir or cfg["curriculum"]["trajectory_dir"])
    pool = TrajectoryPool.from_directory(traj_dir)
    curriculum = FailureCurriculum(
        pool, failure_fraction=float(cfg["curriculum"]["failure_sample_fraction"])
    )
    if pool.empty():
        logger.warning(
            "Curriculum trajectory dir %s is empty — adaptation will behave like plain fine-tune.",
            traj_dir,
        )
    else:
        logger.info("Curriculum loaded %d failure trajectories from %s", len(pool), traj_dir)

    # ---- Logging / checkpoint dirs ---------------------------------------
    run_name = cfg["run"]["name"]
    log_root = Path(cfg["run"]["output_dir"]) / run_name
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = log_root / stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, log_dir / "adapt.yaml")
    shutil.copy(env_cfg_path, log_dir / "env.yaml")

    # ---- Env + runner ----------------------------------------------------
    env = gym.make(task_name, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner_cfg = build_runner_cfg(cfg, task_name)
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(
        env, runner_cfg.to_dict(), log_dir=str(log_dir), device=runner_cfg.device
    )

    resume_path = args.resume or Path(cfg["resume"]["path"])
    if not resume_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {resume_path}")
    logger.info("Resuming baseline from %s", resume_path)
    runner.load(str(resume_path), load_optimizer=bool(cfg["resume"].get("load_optimizer", False)))

    # Install the reset bridge so curriculum assignments actually take effect.
    from phoenix.adaptation.reset_bridge import install as install_reset_bridge

    install_reset_bridge(env, curriculum)

    start = time.time()
    try:
        runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        logger.warning("Interrupted — writing final checkpoint.")
    logger.info("Adaptation wall-time: %.1fs", time.time() - start)

    latest = log_root / "latest.pt"
    ckpts = sorted(
        log_dir.glob("model_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if ckpts:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(ckpts[-1].resolve())
    env.close()
    return 0


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    sys.exit(main())
