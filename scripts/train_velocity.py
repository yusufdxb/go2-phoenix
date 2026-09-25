"""Train the PhoenixVelocity walking policy (Isaac Lab + rsl_rl).

Invoke with the Isaac Sim venv python directly (do not rely on isaaclab.sh,
see project_isaac_sim_venv memory note):

    OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \\
    PYTHONPATH=src /home/yusuf/Sim/isaac-sim-venv/bin/python scripts/train_velocity.py \\
        --num-envs 64 --max-iterations 6 --run-name smoke --headless

The spec (phoenix.velocity.spec.default_spec / smoke_spec) is the single
source of truth; this script only overrides num_envs / max_iterations / seed /
run-name on top of it, and writes the realized spec next to the checkpoints.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("phoenix.velocity.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PhoenixVelocity (walking) with PPO.")
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--action-clip", type=float, default=None, help="override ActionSpec.clip_actions")
    p.add_argument("--save-interval", type=int, default=None, help="override PPOSpec.save_interval (checkpoint cadence)")
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/velocity"))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--smoke", action="store_true", help="Use spec.smoke_spec() instead of default_spec()")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    args = parse_args(argv)
    logger.info("parsed args: %s", args)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app
    logger.info("Isaac Sim launched")

    from phoenix.sim_app_exit import run_isaac_main

    return run_isaac_main(lambda: _run(args, simulation_app), simulation_app, label="train_velocity")


def _run(args: argparse.Namespace, simulation_app) -> int:  # noqa: ANN001
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    import importlib.metadata as metadata

    from phoenix.velocity import env_cfg as pv_env_cfg
    from phoenix.velocity.spec import PLAY_TASK_ID, TASK_ID, VelocityTaskSpec, default_spec, smoke_spec

    pv_env_cfg.register()

    spec: VelocityTaskSpec = smoke_spec() if args.smoke else default_spec()
    if args.num_envs is not None:
        spec = spec.replace(sim=dataclasses.replace(spec.sim, num_envs=args.num_envs))
    if args.max_iterations is not None:
        spec = spec.replace(ppo=dataclasses.replace(spec.ppo, max_iterations=args.max_iterations))
    if args.seed is not None:
        spec = spec.replace(seed=args.seed)
    if args.action_clip is not None:
        spec = spec.replace(action=dataclasses.replace(spec.action, clip_actions=args.action_clip))
    if args.save_interval is not None:
        spec = spec.replace(ppo=dataclasses.replace(spec.ppo, save_interval=args.save_interval))
    spec = spec.replace(ppo=dataclasses.replace(spec.ppo, experiment_name=args.run_name))

    problems = spec.validate()
    if problems:
        raise ValueError("spec invalid:\n  " + "\n  ".join(problems))

    env_cfg = pv_env_cfg.build_velocity_env_cfg(spec)
    if args.device is not None:
        env_cfg.sim.device = args.device

    run_name = args.run_name
    log_root = args.output_dir / run_name
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = log_root / stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run log dir: %s", log_dir)
    (log_dir / "spec.json").write_text(spec.to_json())

    print("[phoenix.velocity] before gym.make", flush=True)
    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
    print("[phoenix.velocity] after gym.make, wrapping", flush=True)

    # Post-construction realized dump: what the live event manager / actuators
    # ACTUALLY carry, not what the cfg asked for. See reference_phoenix_unwired_config_blocks.
    dump = _realized_dump(env)
    dump_path = log_dir / "realized_dr_dump.json"
    dump_path.write_text(json.dumps(dump, indent=2, default=str))
    logger.info("realized DR/actuator dump written: %s", dump_path)

    # clip_actions from the spec (see ActionSpec.clip_actions for the 2026-09-25
    # bug this default replaces: 1.0 with scale=0.25 pinned every joint offset at
    # 0.25 rad, too small for a trot, and stalled the command curriculum at level 0).
    env = RslRlVecEnvWrapper(env, clip_actions=spec.action.clip_actions)

    runner_cfg = pv_env_cfg.build_ppo_runner_cfg(spec)
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, metadata.version("rsl-rl-lib"))
    print("[phoenix.velocity] creating OnPolicyRunner", flush=True)
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=str(log_dir), device=runner_cfg.device)
    print("[phoenix.velocity] runner ready", flush=True)

    if args.resume is not None:
        logger.info("Resuming from checkpoint: %s", args.resume)
        from phoenix.training.checkpoint import load_runner_checkpoint

        ckpt_info = load_runner_checkpoint(
            runner, args.resume, load_actor=True, load_critic=True, load_optimizer=False, load_iteration=False
        )
        if not ckpt_info.get("actor_match", False):
            raise RuntimeError(f"Actor weights did not round-trip from {args.resume}: {ckpt_info}")

    start = time.time()
    try:
        runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        logger.warning("Interrupted, writing final checkpoint.")
    elapsed = time.time() - start
    logger.info("Training wall-time: %.1fs (%.3f it/s)", elapsed, runner_cfg.max_iterations / max(elapsed, 1e-6))

    latest = log_root / "latest.pt"
    ckpts = sorted(log_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if ckpts:
        final = ckpts[-1]
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(final.resolve())
        logger.info("latest.pt -> %s", final.name)
        (log_root / "spec.json").write_text(spec.to_json())

    env.close()
    return 0


def _realized_dump(env) -> dict:
    """What the constructed env's event manager and actuators ACTUALLY carry."""
    unwrapped = env.unwrapped
    em = unwrapped.event_manager
    events = {}
    for mode in ("startup", "reset", "interval"):
        try:
            names = em.active_terms[mode]
        except Exception:  # noqa: BLE001
            names = []
        for name in names:
            cfg = em.get_term_cfg(name)
            events[name] = {
                "mode": mode,
                "func": getattr(cfg.func, "__name__", getattr(cfg.func, "__class__", type(cfg.func)).__name__),
                "params": {k: repr(v) for k, v in dict(cfg.params).items()},
            }
    robot = unwrapped.scene["robot"]
    actuators = {}
    for name, act in robot.actuators.items():
        actuators[name] = {
            "class": type(act).__name__,
            "joint_names": list(getattr(act, "joint_names", [])),
            "effort_limit": _to_py(getattr(act, "effort_limit", None)),
            "velocity_limit": _to_py(getattr(act, "velocity_limit", None)),
            "stiffness": _to_py(getattr(act, "stiffness", None)),
            "damping": _to_py(getattr(act, "damping", None)),
            "min_delay": getattr(getattr(act, "cfg", None), "min_delay", None),
            "max_delay": getattr(getattr(act, "cfg", None), "max_delay", None),
        }
    dims = unwrapped.observation_manager.group_obs_dim
    return {
        "events": events,
        "actuators": actuators,
        "obs_dims": {k: list(v) if isinstance(v, tuple) else v for k, v in dims.items()},
        "num_envs": unwrapped.num_envs,
    }


def _to_py(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.flatten()[:8].tolist()
    except Exception:  # noqa: BLE001
        pass
    return x


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    sys.exit(main())
