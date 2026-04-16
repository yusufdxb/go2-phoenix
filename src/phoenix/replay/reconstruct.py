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

The pure-numpy translation of variation samples → per-env tensors lives
in :mod:`phoenix.replay.apply_variations` and is unit-tested in CI.
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
    import numpy as np
    import torch
    import yaml

    from phoenix.replay import VariationSampler, load_initial_state
    from phoenix.replay.apply_variations import build_per_env_initial_conditions
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
    env = gym.make(task_name, cfg=env_cfg, render_mode=None)

    # ---- Apply per-env variation & initial state ---------------------------
    env.reset()  # drive the gym wrapper out of ResetNeeded state
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

    # Add env_origins so spawn poses are inside each env's tile, not stacked.
    if hasattr(unwrapped.scene, "env_origins"):
        pos = pos + unwrapped.scene.env_origins[: pos.shape[0]]

    robot.write_root_pose_to_sim(torch.cat([pos, quat_wxyz], dim=-1))
    robot.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1))
    robot.write_joint_state_to_sim(jpos, jvel)

    # Per-env mass perturbation on body[0] (trunk). Use root_physx_view if
    # the Isaac Lab build exposes it; otherwise log and skip.
    mass_applied = _apply_per_env_mass(
        robot, per_env["base_mass_delta_kg"], device=device, num_envs=n_variations
    )

    # ---- Roll out the horizon ---------------------------------------------
    horizon_s = float(var_cfg["replay"]["horizon_s"])
    ctrl_dt = float(var_cfg["replay"]["dt"])
    n_steps = int(horizon_s / ctrl_dt)

    # Zero-action rollout. The purpose here is *state discovery under the
    # sampled perturbations* — a downstream training run replaces the zero
    # action with the live policy when used as a curriculum seed.
    action = torch.zeros(n_variations, int(env.action_space.shape[-1]), device=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "trajectory": str(args.trajectory),
        "n_variations": n_variations,
        "horizon_steps": n_steps,
        "friction_scale_mean": mean_friction_scale,
        "mass_delta_applied": mass_applied,
        "variations": [v.__dict__ for v in variations],
    }

    for _ in range(n_steps):
        env.step(action)

    (args.output_dir / "replay_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "Wrote replay summary to %s (mass_applied=%s, friction_scale=%.3f)",
        args.output_dir,
        mass_applied,
        mean_friction_scale,
    )
    env.close()
    return 0


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
    try:
        masses = physx_view.get_masses()  # (num_envs, num_bodies) on cpu in current Isaac Lab
        deltas = torch.as_tensor(mass_deltas, dtype=masses.dtype, device=masses.device)
        new_masses = masses.clone()
        new_masses[:num_envs, 0] = new_masses[:num_envs, 0] + deltas
        indices = torch.arange(num_envs, dtype=torch.int32, device=masses.device)
        physx_view.set_masses(new_masses, indices)
    except (AttributeError, RuntimeError, TypeError) as exc:
        logger.warning("set_masses failed (%s) — mass_delta_kg ignored.", exc)
        return False
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    sys.exit(main())
