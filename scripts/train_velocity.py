"""Train the PhoenixVelocity walking policy (Isaac Lab + rsl_rl).

Invoke with the Python interpreter from an environment containing Isaac Lab:

    OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \\
    PYTHONPATH=src python scripts/train_velocity.py \
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
    p.add_argument(
        "--action-clip", type=float, default=None, help="override ActionSpec.clip_actions"
    )
    p.add_argument(
        "--joint-pos-limits-weight",
        type=float,
        default=None,
        help=(
            "override the joint_pos_limits reward weight (default_spec() keeps "
            "-1.0). 2026-09-25: sim2sim found seed42@3000 drives FR/RL calves "
            "to their hard limit during CCW yaw; legged_gym/unitree_rl_gym "
            "convention is -10.0 with the same 0.9 soft-limit factor Phoenix "
            "already has (see spec.py's joint_pos_limits term)."
        ),
    )
    p.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="override PPOSpec.save_interval (checkpoint cadence)",
    )
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/velocity"))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument(
        "--resume-continue-iterations",
        action="store_true",
        help=(
            "With --resume: restore the checkpoint's iteration counter (rsl_rl "
            "load_cfg['iteration']=True) and run only the REMAINING iterations so "
            "the run stops at the same --max-iterations TOTAL target instead of "
            "running --max-iterations MORE on top of the checkpoint's count. "
            "Without this flag (the old behavior), --resume restarts the "
            "iteration counter at 0, matching phoenix.training.ppo_runner's "
            "fine-tune semantics."
        ),
    )
    p.add_argument(
        "--smoke", action="store_true", help="Use spec.smoke_spec() instead of default_spec()"
    )
    p.add_argument(
        "--curriculum",
        choices=["on", "off"],
        default="on",
        help=(
            "on (default): performance-gated command-range curriculum, starts at "
            "the narrow initial ranges. off: CurriculumSpec.enabled=False, which "
            "starts CommandCurriculum at level=num_levels so the env trains on the "
            "FINAL command ranges from iteration 0 (matches stock Isaac Lab Go2, "
            "which has no curriculum at all). 2026-09-25: seed 42 got stuck at "
            "level 0 (yaw_score 0.31 < 0.5 threshold at iteration ~1972 despite low "
            "termination and decent absolute tracking error), which would fail "
            "gate_v2's command-range coverage check if it never advances."
        ),
    )
    p.add_argument(
        "--arm-label",
        type=str,
        default=None,
        help="Free-text label recorded in the manifest/curriculum-state file (e.g. clip100-nocurriculum)",
    )
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

    return run_isaac_main(
        lambda: _run(args, simulation_app), simulation_app, label="train_velocity"
    )


def _run(args: argparse.Namespace, simulation_app) -> int:  # noqa: ANN001
    import importlib.metadata as metadata

    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    from phoenix.velocity import env_cfg as pv_env_cfg
    from phoenix.velocity.spec import (
        TASK_ID,
        VelocityTaskSpec,
        default_spec,
        smoke_spec,
    )

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
    if args.curriculum == "off":
        spec = spec.replace(curriculum=dataclasses.replace(spec.curriculum, enabled=False))
    if args.joint_pos_limits_weight is not None:
        spec = spec.with_reward_weight("joint_pos_limits", args.joint_pos_limits_weight)
    spec = spec.replace(ppo=dataclasses.replace(spec.ppo, experiment_name=args.run_name))

    run_name = args.run_name
    log_root = args.output_dir / run_name

    # Resume must keep the ORIGINAL arm's config, not whatever this invocation's
    # CLI/code defaults happen to be (a watchdog relaunch must not silently
    # change curriculum on/off or clip_actions mid-run). If a prior spec.json
    # exists for this run, the arm-defining fields must match exactly.
    if args.resume is not None:
        # log_root/spec.json is only written on a CLEAN finish (see the
        # latest.pt block below); a crash-resume (the case this check exists
        # for) never wrote it, so look at the run's timestamped subdirs
        # instead and take the earliest (the original invocation's own spec).
        prior_specs = sorted(log_root.glob("*/spec.json"))
        prior_spec_path = prior_specs[0] if prior_specs else None
        if prior_spec_path is not None and prior_spec_path.exists():
            import json as _json

            prior = _json.loads(prior_spec_path.read_text())
            mismatches = []
            if bool(prior["curriculum"]["enabled"]) != bool(spec.curriculum.enabled):
                mismatches.append(
                    f"curriculum.enabled: prior={prior['curriculum']['enabled']} this run={spec.curriculum.enabled}"
                )
            if abs(float(prior["action"]["clip_actions"]) - float(spec.action.clip_actions)) > 1e-9:
                mismatches.append(
                    f"action.clip_actions: prior={prior['action']['clip_actions']} this run={spec.action.clip_actions}"
                )
            prior_jpl = next(
                (r["weight"] for r in prior["rewards"] if r["name"] == "joint_pos_limits"), None
            )
            this_jpl = spec.reward("joint_pos_limits").weight
            if prior_jpl is not None and abs(float(prior_jpl) - float(this_jpl)) > 1e-9:
                mismatches.append(f"joint_pos_limits weight: prior={prior_jpl} this run={this_jpl}")
            if mismatches:
                raise ValueError(
                    f"resume config mismatch for {run_name}: "
                    + "; ".join(mismatches)
                    + " -- a resumed run must use the SAME arm config as the original invocation"
                )

    problems = spec.validate()
    if problems:
        raise ValueError("spec invalid:\n  " + "\n  ".join(problems))

    env_cfg = pv_env_cfg.build_velocity_env_cfg(spec)
    if args.device is not None:
        env_cfg.sim.device = args.device

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = log_root / stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run log dir: %s", log_dir)
    (log_dir / "spec.json").write_text(spec.to_json())
    if args.arm_label:
        (log_root / "arm_label.txt").write_text(args.arm_label)

    print("[phoenix.velocity] before gym.make", flush=True)
    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
    print("[phoenix.velocity] after gym.make, wrapping", flush=True)

    # Keep a direct reference to the live curriculum object (set by
    # VelocityCommandCurriculum.__init__ as an attribute on the unwrapped env)
    # so its ACTUAL widest-ranges-reached can be recorded after training, not
    # assumed from the spec. Grabbed before RslRlVecEnvWrapper in case wrapping
    # changes attribute access.
    from phoenix.velocity.isaac_terms import ENV_CURRICULUM_ATTR

    curriculum_obj = getattr(env.unwrapped, ENV_CURRICULUM_ATTR, None)

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
    runner = OnPolicyRunner(
        env, runner_cfg.to_dict(), log_dir=str(log_dir), device=runner_cfg.device
    )
    print("[phoenix.velocity] runner ready", flush=True)

    num_learning_iterations = runner_cfg.max_iterations
    if args.resume is not None:
        logger.info("Resuming from checkpoint: %s", args.resume)
        from phoenix.training.checkpoint import load_runner_checkpoint

        ckpt_info = load_runner_checkpoint(
            runner,
            args.resume,
            load_actor=True,
            load_critic=True,
            load_optimizer=False,
            load_iteration=bool(args.resume_continue_iterations),
        )
        if not ckpt_info.get("actor_match", False):
            raise RuntimeError(f"Actor weights did not round-trip from {args.resume}: {ckpt_info}")
        if args.resume_continue_iterations:
            # rsl_rl's learn(num_learning_iterations=X) runs X MORE iterations from
            # runner.current_learning_iteration (on_policy_runner.py:77-78:
            # start_it = self.current_learning_iteration; total_it = start_it + X).
            # runner.current_learning_iteration was just restored from the
            # checkpoint's "iter" field (load_iteration=True above), so pass the
            # REMAINING count, not the total target, or the run overshoots by
            # `runner.current_learning_iteration` extra iterations.
            already_done = int(runner.current_learning_iteration)
            num_learning_iterations = max(0, runner_cfg.max_iterations - already_done)
            logger.info(
                "resume-continue-iterations: checkpoint at iteration %d, target %d, running %d more",
                already_done,
                runner_cfg.max_iterations,
                num_learning_iterations,
            )
            if num_learning_iterations == 0:
                logger.info(
                    "checkpoint already at or past the target iteration count, nothing to do"
                )

    start = time.time()
    try:
        if num_learning_iterations > 0:
            runner.learn(
                num_learning_iterations=num_learning_iterations, init_at_random_ep_len=True
            )
    except KeyboardInterrupt:
        logger.warning("Interrupted, writing final checkpoint.")
    elapsed = time.time() - start
    logger.info(
        "Training wall-time: %.1fs (%.3f it/s)",
        elapsed,
        num_learning_iterations / max(elapsed, 1e-6),
    )

    # Record the curriculum's ACTUAL state (widest ranges reached, level), not
    # the ranges assumed from the spec. A curriculum that stalls below the
    # final ranges (2026-09-25: seed 42 stuck at level 0, yaw_score 0.31 < 0.5
    # threshold at iteration ~1972) must not have its manifest silently claim
    # it trained on the final ranges -- that is exactly what a deploy-side
    # command-range safety gate reads to decide what commands are safe.
    # Written unconditionally (even after a KeyboardInterrupt) so a crash still
    # leaves the truth on disk for postrun to read.
    if curriculum_obj is not None:
        curriculum_state_path = log_root / "curriculum_final_state.json"
        curriculum_state_path.write_text(json.dumps(curriculum_obj.manifest_dict(), indent=2))
        logger.info("curriculum final state written: %s", curriculum_state_path)

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
                "func": getattr(
                    cfg.func, "__name__", getattr(cfg.func, "__class__", type(cfg.func)).__name__
                ),
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
