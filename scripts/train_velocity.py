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
    p.add_argument("--save-interval", type=int, default=None, help="override PPOSpec.save_interval (checkpoint cadence)")
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
    p.add_argument("--smoke", action="store_true", help="Shrink the chosen --spec with spec.smoke_spec()")
    p.add_argument(
        "--spec",
        choices=["default", "vnext"],
        default="default",
        help=(
            "default: default_spec() (walk_v1/v2 arms, obs v1, no health guard). "
            "vnext: spec.vnext_spec() (obs v2 Unitree scales, yaw +-1 from iteration 0, "
            "lin-only Unitree curriculum gate, joint_pos_limits -10, health guard ON)."
        ),
    )
    p.add_argument(
        "--health",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Override one HealthGuardSpec field (repeatable), e.g. --health action_p99_abort=4.0",
    )
    p.add_argument(
        "--override-health-abort",
        action="store_true",
        help="Allow --resume of a run that the health guard stopped (health_abort.json present).",
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

    return run_isaac_main(lambda: _run(args, simulation_app), simulation_app, label="train_velocity")


def _run(args: argparse.Namespace, simulation_app) -> int:  # noqa: ANN001
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    import importlib.metadata as metadata

    from phoenix.velocity import env_cfg as pv_env_cfg
    from phoenix.velocity.spec import (
        PLAY_TASK_ID,
        TASK_ID,
        VelocityTaskSpec,
        default_spec,
        smoke_spec,
        vnext_spec,
    )

    pv_env_cfg.register()

    spec: VelocityTaskSpec = vnext_spec() if args.spec == "vnext" else default_spec()
    if args.smoke:
        spec = smoke_spec(spec)
    if args.health:
        from phoenix.velocity.spec import HealthGuardSpec

        types = {f.name: f.type for f in dataclasses.fields(HealthGuardSpec)}
        changes = {}
        for item in args.health:
            key, _, raw = item.partition("=")
            if key not in types:
                raise SystemExit(f"--health: unknown field {key!r}; known: {sorted(types)}")
            kind = str(types[key])
            changes[key] = (raw.lower() in ("1", "true", "yes")) if kind == "bool" else (
                int(raw) if kind == "int" else float(raw)
            )
        spec = spec.replace(health=dataclasses.replace(spec.health, **changes))
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
    abort_marker = log_root / "health_abort.json"
    if args.resume is not None and abort_marker.exists() and not args.override_health_abort:
        raise SystemExit(
            f"refusing to resume {run_name}: the health guard stopped it ({abort_marker}). "
            "Pass --override-health-abort to resume anyway."
        )
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
            prior_jpl = next((r["weight"] for r in prior["rewards"] if r["name"] == "joint_pos_limits"), None)
            this_jpl = spec.reward("joint_pos_limits").weight
            if prior_jpl is not None and abs(float(prior_jpl) - float(this_jpl)) > 1e-9:
                mismatches.append(f"joint_pos_limits weight: prior={prior_jpl} this run={this_jpl}")
            prior_obs = prior.get("obs_schema", "phoenix-velocity-obs/v1")
            if prior_obs != spec.obs_schema:
                mismatches.append(f"obs_schema: prior={prior_obs} this run={spec.obs_schema}")
            prior_gate = prior["curriculum"].get("gate", "normalized_lin_yaw")
            if prior_gate != spec.curriculum.gate:
                mismatches.append(f"curriculum.gate: prior={prior_gate} this run={spec.curriculum.gate}")
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
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=str(log_dir), device=runner_cfg.device)
    print("[phoenix.velocity] runner ready", flush=True)

    guard = None
    if spec.health.enabled:
        guard = _install_health_guard(runner, spec)
        logger.info("health guard ON: %s", dataclasses.asdict(spec.health))

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
                logger.info("checkpoint already at or past the target iteration count, nothing to do")

    from phoenix.velocity.health import TrainingHealthAbortError

    start = time.time()
    health_abort = None
    try:
        if num_learning_iterations > 0:
            runner.learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        logger.warning("Interrupted, writing final checkpoint.")
    except TrainingHealthAbortError as exc:
        health_abort = exc
        it = exc.verdict.iteration
        logger.error("HEALTH GUARD ABORT at iteration %d: %s", it, exc)
        ckpt_path = log_dir / f"model_{it}.pt"
        runner.current_learning_iteration = it
        runner.save(str(ckpt_path))
        report = guard.report(exc.verdict) if guard is not None else {"reasons": str(exc)}
        report["checkpoint"] = str(ckpt_path)
        for path in (abort_marker, log_dir / "health_abort.json"):
            path.write_text(json.dumps(report, indent=2))
        if runner.logger.writer is not None:
            runner.logger.stop_logging_writer()
    elapsed = time.time() - start
    logger.info("Training wall-time: %.1fs (%.3f it/s)", elapsed, num_learning_iterations / max(elapsed, 1e-6))

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
    # 3 = stopped by the health guard: not a crash, must not be auto-resumed.
    return 3 if health_abort is not None else 0


def _install_health_guard(runner, spec):  # noqa: ANN001, ANN202
    """Hook the fail-fast guard into rsl_rl 5.0.1's per-iteration logger call.

    Reads what the runner already computes (no change to the PPO math):
    ``logger.log(learning_rate=..., action_std=...)`` kwargs, the episode extras
    the env logged this iteration (curriculum level, action p99), the logger's
    episode-length buffer, and the KL of every minibatch, captured by wrapping
    the actor's ``get_kl_divergence`` (which PPO calls for the adaptive LR).
    Health/* scalars go to tensorboard; ABORT raises TrainingHealthAbortError after
    the iteration is logged.
    """
    import statistics

    import torch

    from phoenix.velocity.health import (
        ABORT,
        HealthSample,
        TrainingHealthAbortError,
        TrainingHealthGuard,
    )

    guard = TrainingHealthGuard(spec.health, spec.ppo.desired_kl if spec.ppo.schedule == "adaptive" else None)
    kl_values: list[float] = []

    actor = runner.alg.actor
    original_kl = actor.get_kl_divergence

    def kl_capture(*a, **kw):  # noqa: ANN002, ANN003, ANN202
        kl = original_kl(*a, **kw)
        kl_values.append(float(torch.mean(kl)))
        return kl

    actor.get_kl_divergence = kl_capture

    def extra_mean(key: str) -> float | None:
        vals = []
        for ep in runner.logger.ep_extras:
            if key in ep:
                v = ep[key]
                vals.append(float(v.float().mean()) if isinstance(v, torch.Tensor) else float(v))
        return statistics.fmean(vals) if vals else None

    original_log = runner.logger.log
    last_status = {"s": None}

    def guarded_log(*a, **kw):  # noqa: ANN002, ANN003, ANN202
        it = int(kw["it"])
        std = kw["action_std"].detach().float()
        lens = runner.logger.lenbuffer
        sample = HealthSample(
            iteration=it,
            std_max=float(std.max()),
            std_mean=float(std.mean()),
            action_p99=extra_mean("Curriculum/telemetry/action_abs_p99"),
            episode_length=statistics.fmean(lens) if len(lens) else None,
            kl=statistics.fmean(kl_values) if kl_values else None,
            learning_rate=float(kw["learning_rate"]),
            curriculum_level=extra_mean("Curriculum/command_curriculum/level"),
        )
        kl_values.clear()
        verdict = guard.update(sample)
        writer = runner.logger.writer
        if writer is not None:
            for k, v in verdict.scalars.items():
                writer.add_scalar(f"Health/{k}", v, it)
        out = original_log(*a, **kw)
        if verdict.status != last_status["s"] or it % 100 == 0 or verdict.status == ABORT:
            msg = "; ".join(verdict.abort_reasons or verdict.warnings) or "ok"
            print(f"[phoenix.velocity.health] it {it} {verdict.status}: {msg} | {sample}", flush=True)
            last_status["s"] = verdict.status
        if verdict.status == ABORT:
            raise TrainingHealthAbortError(verdict)
        return out

    runner.logger.log = guarded_log
    return guard


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
