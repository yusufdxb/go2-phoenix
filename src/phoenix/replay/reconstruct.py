"""Reconstruct a real-world failure trajectory in Isaac Sim.

Spawns ``per_trajectory`` copies of the Phoenix GO2 env, each with a
distinct perturbation drawn from the variation sampler, initialised
from the logged pre-failure state. Used by:

* ``scripts/replay.sh`` — interactive inspection of a single failure.
* :mod:`phoenix.adaptation.fine_tune` — as a training-time seed source.

What gets perturbed per env:

* ``push_velocity_delta`` → added to body-frame x-velocity at spawn.
* ``push_yaw_delta`` → added to body-frame z-angular-velocity at spawn.
* ``mass_delta_kg`` → added to the trunk (body 0) mass via PhysX root view.
* ``friction_delta`` → applied uniformly to the scene's physics-material
  friction range as ``range * (1 + mean_delta)``. Per-env friction needs
  the lower-level PhysX material API and is intentionally not done here;
  the per-env mass + initial-velocity sweep gives most of the variation.

The pure-numpy translation of variation samples -> per-env tensors lives
in :mod:`phoenix.replay.apply_variations` and is unit-tested in CI, and the
per-variant Parquet writing lives in :mod:`phoenix.replay.variant_writer`,
also unit-tested in CI.

What this entry point produces
------------------------------

Until 2026-09-17 it applied a ZERO action for the whole horizon and wrote only
``replay_summary.json``. Nothing was reproduced (the robot was not being
driven) and no trajectories were emitted, so ``scripts/loop_closure.sh``
aborted at its replay stage by design and the Phoenix loop had never closed.

Now ``--policy CHECKPOINT`` drives the rollout with the policy whose failure is
being replayed, and every variant env is written to its own Parquet in the
schema the adaptation curriculum already reads. The legacy behaviour is still
available but must be asked for by name with ``--zero-action-diagnostic``, and
the summary then records ``reproduction_evidence: false``. One of the two flags
is required: the old default silently produced the weaker artifact.

A capture whose ``position_frame`` has no validated inverse into simulator
coordinates (a real-robot boot-relative odometry capture) is REFUSED with exit
code 2 rather than seeded; see :mod:`phoenix.replay.state_adapter`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("phoenix.replay.reconstruct")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay a real failure trajectory in Isaac Sim.")
    p.add_argument("--trajectory", type=Path, required=True, help="Parquet from real_world logger")
    p.add_argument("--seed-row-strategy", default="failure_onset_minus_seconds",
                   choices=["first", "failure_onset", "failure_onset_minus_steps", "failure_onset_minus_seconds"])
    p.add_argument("--seed-row-offset-steps", type=int, default=0)
    p.add_argument("--seed-row-offset-seconds", type=float, default=.5)
    p.add_argument("--variations-config", type=Path, required=True)
    p.add_argument("--env-config", type=Path, default=Path("configs/env/rough.yaml"))
    p.add_argument("--variations", type=int, default=None, help="Override per_trajectory")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output-dir", type=Path, default=Path("media/renders/replay"))
    p.add_argument(
        "--policy",
        type=Path,
        default=None,
        help=(
            "rsl_rl checkpoint to DRIVE the replay. Without it the rollout applies a "
            "zero action, which reproduces nothing and is only a state-discovery "
            "diagnostic; --zero-action-diagnostic must then be passed explicitly."
        ),
    )
    p.add_argument(
        "--zero-action-diagnostic",
        action="store_true",
        help="Run the legacy zero-action rollout instead of a policy-driven replay.",
    )
    p.add_argument(
        "--variation-seed",
        type=int,
        default=None,
        help=(
            "Override the Halton seed from the variations config. Use a DIFFERENT "
            "seed for a held-out arm so its variation points are disjoint from the "
            "ones training saw; reusing the seed makes the held-out trajectory's "
            "perturbations identical to the training pool's and the arm is not held out."
        ),
    )
    p.add_argument(
        "--no-variant-trajectories",
        action="store_true",
        help="Skip writing per-variant Parquet trajectories (summary JSON only).",
    )
    args = p.parse_args(argv)
    if args.policy is None and not args.zero_action_diagnostic:
        p.error(
            "pass --policy CHECKPOINT to replay the failure with the policy that produced "
            "it, or --zero-action-diagnostic to run the legacy zero-action rollout and "
            "have the summary record reproduction_evidence=false"
        )
    if args.policy is not None and args.zero_action_diagnostic:
        p.error("--policy and --zero-action-diagnostic are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    args = parse_args(argv)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app
    from phoenix.sim_app_exit import run_isaac_main

    return run_isaac_main(lambda: _run(args, simulation_app), simulation_app, label="replay")


def _run(args: argparse.Namespace, simulation_app) -> int:  # noqa: ANN001
    import gymnasium as gym
    import numpy as np
    import torch
    import yaml

    from phoenix.replay import VariationSampler, load_initial_state
    from phoenix.replay.apply_variations import build_per_env_initial_conditions
    from phoenix.sim_env import build_env_cfg, load_layered_config

    var_cfg = yaml.safe_load(args.variations_config.read_text())
    n_variations = args.variations or int(var_cfg["variations"]["per_trajectory"])

    from phoenix.adaptation.reset_bridge import resolve_seed
    from phoenix.replay.state_adapter import VelocityCommandAdapter
    seed_record = resolve_seed(args.trajectory, args.seed_row_strategy,
                               args.seed_row_offset_steps, args.seed_row_offset_seconds)
    initial = load_initial_state(args.trajectory, row=seed_record["resolved_row"])
    logger.info("Loaded initial state from %s", args.trajectory)

    # A hardware capture is a recording, not a simulator seed: its base_pos is
    # boot-relative odometry whose z is displacement from the boot pose, not
    # height above the floor. Restoring it would spawn the trunk at a
    # fabricated height, so refuse here with the capture named rather than
    # letting the state write produce a plausible-looking rollout.
    from phoenix.replay.state_adapter import RESTORABLE_POSITION_FRAMES
    if initial.position_frame not in RESTORABLE_POSITION_FRAMES:
        logger.error(
            "%s declares position_frame=%r (source: %s), which has no validated mapping into "
            "simulator coordinates. Replay refused.",
            args.trajectory,
            initial.position_frame,
            initial.position_frame_source,
        )
        return 2

    variation_seed = (
        int(var_cfg["variations"]["seed"])
        if args.variation_seed is None
        else int(args.variation_seed)
    )
    sampler = VariationSampler(
        bounds=var_cfg["variations"]["dr_bounds"],
        seed=variation_seed,
    )
    variations = sampler.sample(n_variations)
    logger.info("Sampled %d variations", len(variations))

    per_env = build_per_env_initial_conditions(initial, variations)

    env_cfg_loaded = load_layered_config(args.env_config)
    env_cfg = build_env_cfg(env_cfg_loaded)
    env_cfg.scene.num_envs = n_variations
    env_cfg.sim.device = args.device

    # Friction sweep: shift the scene-wide physics-material range by the
    # mean of the sampled friction_scale factors. Per-env friction would
    # need the PhysX material API; documented in module docstring.
    mean_friction_scale = float(per_env["friction_scale"].mean())
    _apply_friction_scale(env_cfg, mean_friction_scale)

    task_name = env_cfg_loaded.to_container()["env"]["task_name"]
    raw_env = gym.make(task_name, cfg=env_cfg, render_mode=None)

    # The rsl_rl wrapper RESETS inside its constructor (isaaclab_rl/rsl_rl/
    # vecenv_wrapper.py: "The wrapper calls reset at the start"), so it must be
    # built BEFORE the seed state is written or the write is discarded
    # immediately. Its step() returns (obs, reward, dones, extras) and its
    # get_observations() recomputes from the observation manager, so it
    # reflects a state written by hand.
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(raw_env, clip_actions=1.0)

    # ---- Apply per-env variation & initial state ---------------------------
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    device = args.device

    def _t(arr, dtype=torch.float32):
        return torch.as_tensor(np.ascontiguousarray(arr), device=device, dtype=dtype)

    pos = _t(per_env["base_pos"])
    quat_wxyz = _t(per_env["base_quat_wxyz"])
    jpos = _t(per_env["joint_pos"])
    jvel = _t(per_env["joint_vel"])
    lin_vel = _t(per_env["base_lin_vel"])
    ang_vel = _t(per_env["base_ang_vel"])

    # Map the stored position into world coordinates using the frame the source
    # declared, so spawn poses are inside each env's tile and never displaced by
    # a partially-applied origin.
    if initial.position_frame == "env_local" and hasattr(unwrapped.scene, "env_origins"):
        pos = pos + unwrapped.scene.env_origins[: pos.shape[0]]

    robot.write_root_pose_to_sim(torch.cat([pos, quat_wxyz], dim=-1))
    robot.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1))
    robot.write_joint_state_to_sim(jpos, jvel)
    # This entry point is a fixed-horizon zero-action diagnostic, not a training
    # env, so pinning the logged command for the whole rollout is the intent
    # rather than a command-process confound. Declared explicitly.
    VelocityCommandAdapter(unwrapped).restore(
        torch.arange(n_variations, device=device),
        initial.command_vel,
        hold_seconds=float("inf"),
    )

    # Per-env mass perturbation on body[0] (trunk). Use root_physx_view if
    # the Isaac Lab build exposes it; otherwise log and skip.
    mass_applied = _apply_per_env_mass(
        robot, per_env["base_mass_delta_kg"], device=device, num_envs=n_variations
    )

    # ---- Roll out the horizon ---------------------------------------------
    horizon_s = float(var_cfg["replay"]["horizon_s"])
    ctrl_dt = float(var_cfg["replay"]["dt"])
    n_steps = int(horizon_s / ctrl_dt)
    action_dim = int(env.action_space.shape[-1])

    policy = None
    if args.policy is not None:
        policy = _load_policy(env, args, task_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_variant_trajectories:
        from phoenix.replay.variant_writer import VariantTrajectoryWriter

        writer = VariantTrajectoryWriter(
            args.output_dir, n_variations, control_dt=ctrl_dt
        )

    obs = _policy_observation(env.get_observations())
    steps_run = 0
    with torch.inference_mode():
        for step_index in range(n_steps):
            if policy is None:
                # State discovery under the sampled perturbations only. This
                # reproduces nothing: the robot is not being driven, so a
                # "the failure reproduced" claim from this arm would be false.
                action = torch.zeros(n_variations, action_dim, device=device)
            else:
                action = policy(obs)

            # Snapshot BEFORE stepping. env.step() resets whichever envs
            # terminate, so state read after the step is post-reset for exactly
            # the envs whose terminal state matters most.
            snapshot = _snapshot(unwrapped, robot, n_variations) if writer else None

            obs_td, _reward, dones, _extras = env.step(action)
            obs = _policy_observation(obs_td)
            steps_run = step_index + 1

            if writer is not None:
                writer.append_step(
                    step_index,
                    action=_np(action)[:n_variations, :12],
                    terminated=_np(dones).astype(bool).reshape(n_variations),
                    **snapshot,
                )
                if writer.active_envs == 0:
                    logger.info("every variant terminated by step %d", step_index)
                    break

    variant_results = writer.close() if writer is not None else []
    controller = "zero_action_diagnostic" if policy is None else "policy_driven_replay"
    summary = {
        "trajectory": str(args.trajectory),
        "seed": seed_record,
        "controller": controller,
        # True only when a policy actually drove the rollout AND per-variant
        # trajectories were recorded, so the claim can be checked against
        # artifacts rather than taken on trust.
        "reproduction_evidence": bool(policy is not None and variant_results),
        "policy_checkpoint": None if args.policy is None else str(args.policy),
        "position_frame": initial.position_frame,
        "position_frame_source": initial.position_frame_source,
        "replay_fidelity": "state_only_seed",
        "command_hold": "episode",
        "n_variations": n_variations,
        "variation_seed": variation_seed,
        "horizon_steps": n_steps,
        "steps_run": steps_run,
        "friction_scale_mean": mean_friction_scale,
        "mass_delta_applied": mass_applied,
        "variants_written": len(variant_results),
        "variants_with_failure": sum(1 for r in variant_results if r.failed),
        "variations": [v.__dict__ for v in variations],
    }

    (args.output_dir / "replay_summary.json").write_text(json.dumps(summary, indent=2))
    if writer is not None:
        writer.write_index(
            args.output_dir / "variants_index.json",
            extra={
                "trajectory": str(args.trajectory),
                "controller": controller,
                "policy_checkpoint": summary["policy_checkpoint"],
                "variation_seed": variation_seed,
            },
        )
    logger.info(
        "Wrote %d variant trajectories to %s (%d with a detected failure); "
        "controller=%s mass_applied=%s friction_scale=%.3f",
        len(variant_results),
        args.output_dir,
        summary["variants_with_failure"],
        controller,
        mass_applied,
        mean_friction_scale,
    )
    env.close()
    return 0


def _np(value):
    """Tensor or warp array to numpy, matching the repo's _to_numpy idiom."""
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return np.asarray(value)


def _policy_observation(obs):
    """The observation object the inference policy expects.

    Do NOT flatten to the ``policy`` group. rsl_rl's ``MlpModel.get_latent``
    does ``[obs[g] for g in self.obs_groups]``, so it wants the whole
    multi-group container; handing it the already-extracted policy tensor
    raises ``IndexError: too many indices for tensor of dimension 2``.

    ``phoenix.training.evaluate`` looks like it flattens, but its guard is
    ``isinstance(obs, dict)`` and ``TensorDict`` is NOT a dict subclass, so in
    practice it passes the container through untouched. This function makes
    that behaviour explicit instead of accidental: only a genuine tuple or a
    plain ``dict`` is unwrapped, which keeps a hand-built test double working.
    """
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        return obs.get("policy", next(iter(obs.values())))
    return obs


def _snapshot(unwrapped, robot, n: int) -> dict:
    """Env-local state for every variant env, in the Parquet schema's frames.

    Positions are made environment-local (the capture convention) and the
    quaternion is converted from Isaac Lab's wxyz to the schema's xyzw. Both
    conversions match :mod:`phoenix.real_world.synthesize_failure`.
    """
    import numpy as np

    data = robot.data
    origins = _np(unwrapped.scene.env_origins)[:n, :3]
    base_pos = _np(data.root_pos_w)[:n, :3] - origins
    quat_wxyz = _np(data.root_quat_w)[:n, :4]
    quat_xyzw = np.roll(quat_wxyz, -1, axis=-1)
    contacts = np.zeros((n, 4), dtype=np.float32)
    try:
        sensor = unwrapped.scene["contact_forces"]
        feet = [i for i, name in enumerate(sensor.body_names) if name.lower().endswith("foot")]
        if len(feet) == 4:
            contacts = np.linalg.norm(_np(sensor.data.net_forces_w)[:n, feet], axis=-1)
    except (KeyError, AttributeError):
        pass
    return {
        "base_pos": base_pos,
        "base_quat_xyzw": quat_xyzw,
        "base_lin_vel_body": _np(data.root_lin_vel_b)[:n, :3],
        "base_ang_vel_body": _np(data.root_ang_vel_b)[:n, :3],
        "joint_pos": _np(data.joint_pos)[:n, :12],
        "joint_vel": _np(data.joint_vel)[:n, :12],
        "command_vel": _np(unwrapped.command_manager.get_command("base_velocity"))[:n, :3],
        "contact_forces": contacts,
        # base_pos is env-origin-relative and the origin sits on the flat
        # ground plane, so column 2 IS a validated ground-relative height.
        # This is the one replay path that may feed the collapse detector.
        "base_height": base_pos[:, 2:3],
    }


def _load_policy(env, args, task_name: str):
    """Rebuild the rsl_rl inference policy from a checkpoint.

    Mirrors :mod:`phoenix.training.evaluate`, including resolving
    ``empirical_normalization`` FROM THE CHECKPOINT. A hardcoded ``True`` on a
    checkpoint without normalizer buffers silently shrinks every observation by
    1% (see ``phoenix.sim2real.export.checkpoint_has_obs_normalizer``), which
    would make the replay a different function of the state than the run being
    reproduced.
    """
    from importlib import metadata

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    from phoenix.sim2real.export import checkpoint_has_obs_normalizer
    from phoenix.training.agent_cfg import build_runner_cfg
    from phoenix.training.checkpoint import load_runner_checkpoint

    use_norm = checkpoint_has_obs_normalizer(args.policy)
    logger.info("empirical_normalization resolved from checkpoint: %s", use_norm)
    runner_yaml = {
        "run": {
            "name": "replay",
            "output_dir": "/tmp",
            "log_interval": 1,
            "save_interval": 1,
            "max_iterations": 1,
            "seed": 0,
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
        "runner": {"num_steps_per_env": 24, "empirical_normalization": use_norm},
    }
    runner_cfg = build_runner_cfg(runner_yaml, task_name)
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=args.device)
    info = load_runner_checkpoint(
        runner,
        args.policy,
        load_actor=True,
        load_critic=False,
        load_optimizer=False,
        load_iteration=False,
    )
    if not info.get("actor_match", False):
        raise RuntimeError(f"Actor weights did not round-trip from {args.policy}: {info}")
    return runner.get_inference_policy(device=args.device)


def _apply_friction_scale(env_cfg, scale: float) -> None:
    """Multiply the physics_material event term's friction range by ``scale``."""
    events = env_cfg.events
    events = events.default if hasattr(events, "default") else events
    pm = getattr(events, "physics_material", None)
    if pm is None:
        return
    sf_lo, sf_hi = pm.params.get("static_friction_range", (0.5, 1.0))
    df_lo, df_hi = pm.params.get("dynamic_friction_range", (0.5, 1.0))
    pm.params["static_friction_range"] = (float(sf_lo) * scale, float(sf_hi) * scale)
    pm.params["dynamic_friction_range"] = (float(df_lo) * scale, float(df_hi) * scale)


def _apply_per_env_mass(robot, mass_deltas, *, device, num_envs: int) -> bool:
    """Add ``mass_deltas[i]`` kg to body 0 of env i. Returns True if applied."""
    import torch

    physx_view = getattr(robot, "root_physx_view", None)
    if physx_view is None:
        logger.warning(
            "robot.root_physx_view not available — mass_delta_kg ignored. "
            "Friction + initial-velocity perturbations still apply."
        )
        return False
    import numpy as np

    try:
        # This block silently did NOTHING before 2026-09-17. On Isaac Sim 6 /
        # Isaac Lab 4.5 the tensor view runs a WARP backend: get_masses()
        # returns a wp.array whose .dtype is a python type, so
        # torch.as_tensor(..., dtype=masses.dtype) raised TypeError, the except
        # below swallowed it, and every variant ran with mass_delta_kg = 0
        # while replay_summary.json reported the sampled values. Handing the
        # warp frontend a torch tensor fails differently and just as quietly
        # ("issubclass() arg 1 must be a class", frontend_warp.py:102).
        #
        # So: read, modify in numpy, and hand back an array of the SAME kind
        # the view gave us, which is what omni.physics.tensors' own set_masses
        # example does (wp.from_numpy(..., dtype=wp.float32)).
        raw = physx_view.get_masses()  # (count, max_links)
        masses = _np(raw)
        new_masses = masses.copy()
        deltas = _np(mass_deltas).reshape(-1)[:num_envs]
        if new_masses.shape[0] < num_envs:
            raise ValueError(
                f"view holds {new_masses.shape[0]} articulations, need {num_envs}"
            )
        new_masses[:num_envs, 0] = new_masses[:num_envs, 0] + deltas

        if type(raw).__module__.startswith("warp"):
            import warp as wp

            device = getattr(raw, "device", "cpu")
            data = wp.from_numpy(
                np.ascontiguousarray(new_masses, dtype=np.float32),
                dtype=wp.float32,
                device=device,
            )
            idx = wp.from_numpy(
                np.arange(num_envs, dtype=np.uint32), dtype=wp.uint32, device=device
            )
        else:
            data = torch.as_tensor(new_masses, dtype=torch.float32)
            idx = torch.arange(num_envs, dtype=torch.int32)
        physx_view.set_masses(data, idx)

        # Read back: a silent no-op here is the whole point of this fix.
        applied = _np(physx_view.get_masses())[:num_envs, 0]
        if not np.allclose(applied, new_masses[:num_envs, 0], atol=1e-4):
            logger.warning(
                "set_masses accepted the write but the masses did not change "
                "(max diff %.3g), mass_delta_kg NOT applied.",
                float(np.max(np.abs(applied - new_masses[:num_envs, 0]))),
            )
            return False
    except (AttributeError, RuntimeError, TypeError, ValueError, ImportError) as exc:
        logger.warning("set_masses failed (%s), mass_delta_kg ignored.", exc)
        return False
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    sys.exit(main())
