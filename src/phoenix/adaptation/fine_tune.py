"""Fine-tune a baseline policy with failure-seeded curriculum.

Invoked from ``scripts/adapt.sh`` inside Isaac Lab's Python context. The
inner PPO training loop matches :mod:`phoenix.training.ppo_runner` so
successes reproduce; what differs is the curriculum that seeds a fraction
of environment resets from recorded real-world failure Parquets.

Three seeds, three separate quantities, never conflated:

``--seed``
    The TRAINING seed. Sets ``env_cfg.seed`` and ``cfg["run"]["seed"]``,
    overriding the YAML, so a multi-seed sweep is actually multiple runs. Until
    this flag existed, ``scripts/loop_closure.sh`` looped over three seeds,
    labelled three output directories, and passed none of them: all three runs
    read ``cfg["run"]["seed"]`` and were one run reported three times.
``--curriculum-seed``
    The FailureCurriculum RNG seed, which decides WHICH envs get a failure seed
    and which trajectory each draws. Defaults to ``--seed``. It must never fall
    back to 0; this repo has already shipped a run where the curriculum RNG sat
    at its 0 default while the caller believed it was varying.
The EVALUATION seed
    Belongs to ``phoenix.training.evaluate --seed`` and is deliberately not
    settable here. A policy trained under one seed and evaluated under another
    is the normal case, and folding them together hides it.

Every run writes ``<run_dir>/seeds.json``. The artifact, not the caller's
intent, is the record of what the run used.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("phoenix.adaptation.fine_tune")

#: Filename of the per-run seed record, read back by multi-seed orchestration.
SEEDS_ARTIFACT = "seeds.json"

#: Exact key set of that artifact. Consumers assert against this shape.
SEEDS_FIELDS = ("training_seed", "curriculum_seed", "config_seed", "resolved_from")

#: Seed used when neither the CLI nor the config names one. Historical runs all
#: used this value through ``cfg["run"].get("seed", 42)``, so it is recorded as
#: the config seed rather than pretending the config declared something.
DEFAULT_CONFIG_SEED = 42


def resolve_failure_reset_fraction(config):
    """Read the accurate name with explicit legacy YAML aliases."""
    names = ("failure_reset_fraction", "failure_sample_fraction", "failure_fraction")
    present = [name for name in names if name in config]
    if len(present) != 1:
        raise ValueError(f"Specify exactly one failure reset fraction key, found {present}")
    value = float(config[present[0]])
    if not 0 <= value <= 1:
        raise ValueError("failure_reset_fraction must be in [0,1]")
    return value


def _checked_seed(value, name):
    if isinstance(value, bool) or not isinstance(value, (int,)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}")
    return int(value)


def resolve_seeds(config, *, seed=None, curriculum_seed=None):
    """Resolve the training and curriculum RNG seeds as separate quantities.

    Returns exactly the record written to :data:`SEEDS_ARTIFACT`.
    ``curriculum_seed`` falls back to the TRAINING seed, never to 0, so a
    caller that varies ``--seed`` alone still varies which environments are
    failure-seeded. ``resolved_from`` is ``"cli"`` when either seed came from
    the command line and ``"config"`` when the run was seeded entirely by the
    YAML.
    """
    run = config.get("run") or {}
    config_seed = _checked_seed(run.get("seed", DEFAULT_CONFIG_SEED), "run.seed")
    training = config_seed if seed is None else _checked_seed(seed, "--seed")
    curriculum = (
        training if curriculum_seed is None else _checked_seed(curriculum_seed, "--curriculum-seed")
    )
    return {
        "training_seed": training,
        "curriculum_seed": curriculum,
        "config_seed": config_seed,
        "resolved_from": "config" if seed is None and curriculum_seed is None else "cli",
    }


def write_seeds(run_dir, seeds):
    """Write the seed record for one run; refuse to disagree with an existing one."""
    if tuple(sorted(seeds)) != tuple(sorted(SEEDS_FIELDS)):
        raise ValueError(f"Seed record must hold exactly {sorted(SEEDS_FIELDS)}")
    path = Path(run_dir) / SEEDS_ARTIFACT
    payload = json.dumps(seeds, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise FileExistsError(f"Refusing to overwrite a different seed record: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(payload)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a Phoenix policy with failure curriculum.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume", type=Path, default=None, help="Override resume path")
    p.add_argument("--trajectory-dir", type=Path, default=None, help="Override curriculum dir")
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Training seed. Sets env_cfg.seed and cfg['run']['seed'], overriding the YAML.",
    )
    p.add_argument(
        "--curriculum-seed",
        type=int,
        default=None,
        help="FailureCurriculum RNG seed. Defaults to --seed, never to 0.",
    )
    p.add_argument("--headless", action="store_true", default=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    args = parse_args(argv)
    print("[adapt] args:", args, flush=True)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app
    print("[adapt] app launched", flush=True)
    try:
        return _run(args, simulation_app)
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


def _run(args: argparse.Namespace, simulation_app) -> int:  # noqa: ANN001
    import importlib.metadata as metadata

    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from omegaconf import OmegaConf
    from rsl_rl.runners import OnPolicyRunner

    from phoenix.adaptation.curriculum import DEFAULT_STRATA, FailureCurriculum, TrajectoryPool
    from phoenix.sim_env import build_env_cfg, load_layered_config
    from phoenix.training.agent_cfg import build_runner_cfg

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    assert isinstance(cfg, dict)

    if args.max_iterations is not None:
        cfg["run"]["max_iterations"] = args.max_iterations
    if args.device is not None:
        cfg["run"]["device"] = args.device

    # Resolve the two seeds this entry point owns BEFORE anything consumes
    # them, and write cfg["run"]["seed"] so every downstream reader
    # (build_runner_cfg, the copied adapt.yaml's consumers) sees the effective
    # training seed rather than the YAML's.
    seeds = resolve_seeds(cfg, seed=args.seed, curriculum_seed=args.curriculum_seed)
    cfg["run"]["seed"] = seeds["training_seed"]
    print(f"[adapt] seeds: {seeds}", flush=True)

    env_cfg_path = Path(cfg["env"]["config"])
    env_cfg_loaded = load_layered_config(env_cfg_path)
    env_cfg = build_env_cfg(env_cfg_loaded)
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = cfg["run"]["device"]
    env_cfg.seed = seeds["training_seed"]

    task_name = env_cfg_loaded.to_container()["env"]["task_name"]

    # ---- Failure curriculum ----------------------------------------------
    traj_dir = Path(args.trajectory_dir or cfg["curriculum"]["trajectory_dir"])
    # Optional per-cell mode-subset filter (mode-subset ablation, 2026-05-08).
    # Ashfall writes ``curriculum.failure_modes`` into the per-cell adapt
    # YAML; if present and non-empty, only parquets whose recorded
    # ``failure_mode`` column intersects this whitelist are eligible.
    # An empty list / missing field preserves legacy "all modes" behavior.
    failure_modes_cfg = cfg.get("curriculum", {}).get("failure_modes") or None
    if failure_modes_cfg:
        failure_modes_cfg = list(failure_modes_cfg)
    pool = TrajectoryPool.from_directory(traj_dir, failure_modes=failure_modes_cfg)
    # The curriculum RNG decides which envs are failure-seeded and which
    # trajectory each draws. It is a SEPARATE quantity from the training seed
    # and defaults to it, so a caller varying only --seed still varies this;
    # it never falls back to FailureCurriculum's own seed=0 default, which is
    # the failure mode that once held this stochastic input constant across a
    # whole multi-seed pilot.
    curriculum = FailureCurriculum(
        pool,
        failure_reset_fraction=resolve_failure_reset_fraction(cfg["curriculum"]),
        seed=seeds["curriculum_seed"],
        sampling=cfg["curriculum"].get("sampling", "uniform_legacy"),
        strata=tuple(cfg["curriculum"].get("strata", DEFAULT_STRATA)),
    )
    if pool.empty() and curriculum.failure_reset_fraction > 0:
        raise ValueError("Active failure reset curriculum has no trajectories")
    if pool.empty():
        logger.info(
            "Curriculum trajectory dir %s is empty (modes=%s) — "
            "adaptation will behave like plain fine-tune.",
            traj_dir,
            failure_modes_cfg,
        )
    else:
        logger.info(
            "Curriculum loaded %d failure trajectories from %s (modes=%s, sampling=%s, strata=%s)",
            len(pool),
            traj_dir,
            failure_modes_cfg or "all",
            curriculum.sampling,
            curriculum.describe_strata(),
        )

    # ---- Logging / checkpoint dirs ---------------------------------------
    run_name = cfg["run"]["name"]
    log_root = Path(cfg["run"]["output_dir"]) / run_name
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = log_root / stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, log_dir / "adapt.yaml")
    shutil.copy(env_cfg_path, log_dir / "env.yaml")
    # The copied adapt.yaml still carries the config's own seed, so the seed
    # record is what says which seeds this run actually used. Written before
    # any GPU time is spent, so an interrupted run is still identifiable.
    write_seeds(log_dir, seeds)

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
    print(f"[adapt] Resuming baseline from {resume_path}", flush=True)
    load_optim = bool(cfg["resume"].get("load_optimizer", False))

    # Use the Phoenix helper so we *verify* what actually round-tripped
    # (actor weights, the learned Gaussian ``std_param``, and the empirical
    # obs normalizer buffers if present). rsl_rl 3.x's
    # ``OnPolicyRunner.load`` is silent on partial matches when
    # ``strict=False`` — the helper re-reads the checkpoint and confirms
    # the live modules match bit-for-bit, raising if not.
    from phoenix.training.checkpoint import load_runner_checkpoint

    ckpt_info = load_runner_checkpoint(
        runner,
        resume_path,
        load_actor=True,
        load_critic=True,
        load_optimizer=load_optim,
        load_iteration=False,
    )
    if not ckpt_info.get("actor_match", False):
        raise RuntimeError(
            f"Actor weights did not round-trip from {resume_path}: "
            f"mismatched_keys={ckpt_info.get('actor_mismatched_keys')} "
            f"ckpt_only={ckpt_info.get('actor_ckpt_only_keys')} "
            f"live_only={ckpt_info.get('actor_live_only_keys')}"
        )
    std_mean = ckpt_info.get("actor_std_mean")
    print(
        f"[adapt] Baseline loaded at iter={ckpt_info.get('iter')} "
        f"std_mean={std_mean:.3f} "
        f"obs_norm_in_ckpt={ckpt_info.get('actor_obs_normalizer_in_ckpt')}",
        flush=True,
    )

    # Install the reset bridge so curriculum assignments actually take effect.
    from phoenix.adaptation.reset_bridge import install as install_reset_bridge

    reset_cfg = cfg["curriculum"]
    # A friction scenario adapter is only built when the pool declares friction
    # to restore. Building it unconditionally would claim a causal continuation
    # the sources do not carry.
    scenario_adapter = None
    if reset_cfg.get("restore_environment_parameters", False):
        from phoenix.adaptation.scenario_bridge import FrictionScenarioAdapter

        scenario_adapter = FrictionScenarioAdapter(env.unwrapped)
    install_reset_bridge(
        env,
        curriculum,
        seed_row_strategy=reset_cfg.get("seed_row_strategy", "failure_onset_minus_seconds"),
        seed_row_offset_steps=reset_cfg.get(
            "seed_row_offset_steps", reset_cfg.get("seed_row_offset_k", 0)
        ),
        seed_row_offset_seconds=float(reset_cfg.get("seed_row_offset_seconds", 0.5)),
        command_policy=reset_cfg.get("command_policy", "source_hold_to_onset"),
        command_hold_seconds=reset_cfg.get("command_hold_seconds"),
        position_frame=reset_cfg.get("position_frame"),
        history_rows=int(reset_cfg.get("history_rows", 2)),
        require_exact_replay=bool(reset_cfg.get("require_exact_replay", False)),
        scenario_adapter=scenario_adapter,
        environment_policy=reset_cfg.get("environment_policy", "require_declared"),
        telemetry_path=log_dir / "failure_resets.jsonl",
    )
    # The wrapper constructed the runner before bridge installation. Reset once
    # more so the first fresh PPO rollout also uses the intended distribution.
    env.reset()

    # Why ``init_at_random_ep_len=False`` when warm-starting: rsl_rl's
    # Logger only contributes reward values to ``rewbuffer`` when an
    # episode *terminates*. With random initial episode lengths most
    # envs time-out within the first 24-step rollout and contribute an
    # artificially low partial reward, which misleadingly looks like the
    # loaded policy forgot how to walk.
    # Starting at step 0 for every env means the metrics in iteration 0
    # reflect the actual warm-started behaviour rather than a truncated
    # window artifact.
    start = time.time()
    try:
        runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=False)
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
