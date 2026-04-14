"""Build a Phoenix-flavoured GO2 env cfg from a layered YAML config.

The factory produces an Isaac Lab ``ManagerBasedRLEnvCfg`` starting from the
upstream ``UnitreeGo2RoughEnvCfg``, then applies failure-oriented overrides:

* friction / restitution / mass / motor scale domain randomization
* slippery terrain overlay (friction patches)
* base push perturbations

Isaac Lab imports are done lazily so the module can still be imported in
CI (which has no ``torch`` / ``isaaclab``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config_loader import PhoenixConfig, load_layered_config

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from isaaclab.envs import ManagerBasedRLEnvCfg


def _apply_domain_randomization(env_cfg: Any, dr: dict[str, Any]) -> None:
    """Patch friction/mass/motor DR ranges in the env's Events cfg."""
    if not dr.get("enabled", True):
        return

    events = env_cfg.events
    # friction patches on the terrain material
    if hasattr(events, "physics_material") and events.physics_material is not None:
        fr_lo, fr_hi = dr["friction_range"]
        rs_lo, rs_hi = dr.get("restitution_range", [0.0, 0.0])
        events.physics_material.params["static_friction_range"] = (fr_lo, fr_hi)
        events.physics_material.params["dynamic_friction_range"] = (fr_lo, fr_hi)
        events.physics_material.params["restitution_range"] = (rs_lo, rs_hi)

    # base mass offset
    if hasattr(events, "add_base_mass") and events.add_base_mass is not None:
        m_lo, m_hi = dr["mass_offset_kg"]
        events.add_base_mass.params["mass_distribution_params"] = (float(m_lo), float(m_hi))

    # motor strength (actuator gain randomization, if the event hook exists upstream)
    if hasattr(events, "randomize_actuator_gains") and events.randomize_actuator_gains is not None:
        g_lo, g_hi = dr["motor_strength_scale"]
        events.randomize_actuator_gains.params["stiffness_distribution_params"] = (g_lo, g_hi)
        events.randomize_actuator_gains.params["damping_distribution_params"] = (g_lo, g_hi)


def _apply_perturbation(env_cfg: Any, pert: dict[str, Any]) -> None:
    """Enable or disable the push-robot event based on the YAML block."""
    events = env_cfg.events
    if not pert.get("enabled", False):
        if hasattr(events, "push_robot"):
            events.push_robot = None
        return

    if not hasattr(events, "push_robot") or events.push_robot is None:
        # Upstream cfg disabled push_robot — we do not silently re-enable it
        # to avoid resurrecting a stale function handle. Log and return.
        return

    events.push_robot.interval_range_s = (pert["push_interval_s"], pert["push_interval_s"])
    vel_xy = float(pert["push_velocity_xy"])
    vel_yaw = float(pert["push_velocity_yaw"])
    events.push_robot.params["velocity_range"] = {
        "x": (-vel_xy, vel_xy),
        "y": (-vel_xy, vel_xy),
        "yaw": (-vel_yaw, vel_yaw),
    }


def _apply_slippery_patches(env_cfg: Any, dr: dict[str, Any]) -> None:
    """Shrink DR friction ranges to a slippery regime (used by slippery.yaml)."""
    if "friction_range" not in dr:
        return
    events = env_cfg.events
    if hasattr(events, "physics_material") and events.physics_material is not None:
        fr_lo, fr_hi = dr["friction_range"]
        events.physics_material.params["static_friction_range"] = (fr_lo, fr_hi)
        events.physics_material.params["dynamic_friction_range"] = (fr_lo, fr_hi)


def _apply_commands(env_cfg: Any, cmd: dict[str, Any]) -> None:
    if not hasattr(env_cfg, "commands") or env_cfg.commands is None:
        return
    vel_cmd = env_cfg.commands.base_velocity
    vel_cmd.ranges.lin_vel_x = tuple(cmd["lin_vel_x"])
    vel_cmd.ranges.lin_vel_y = tuple(cmd["lin_vel_y"])
    vel_cmd.ranges.ang_vel_z = tuple(cmd["ang_vel_z"])
    vel_cmd.resampling_time_range = (cmd["resample_time_s"], cmd["resample_time_s"])


def build_env_cfg(config: str | Path | PhoenixConfig) -> ManagerBasedRLEnvCfg:
    """Build a GO2 env cfg, applying YAML overrides on top of the upstream task."""
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401 - registers tasks

    if not isinstance(config, PhoenixConfig):
        config = load_layered_config(config)
    data = config.to_container()
    env_blk = data["env"]

    task_name = env_blk["task_name"]
    env_cfg = gym.spec(task_name).kwargs["env_cfg_entry_point"]  # type: ignore[index]
    # Resolve the "pkg.module:ClassName" entry point to an instance.
    import importlib

    module_name, class_name = env_cfg.split(":")
    env_cfg_cls = getattr(importlib.import_module(module_name), class_name)
    cfg = env_cfg_cls()

    # Core scene + timing
    cfg.scene.num_envs = int(env_blk["num_envs"])
    cfg.episode_length_s = float(env_blk["episode_length_s"])
    cfg.decimation = int(env_blk["decimation"])
    cfg.sim.dt = float(env_blk["sim_dt"])
    cfg.seed = int(data.get("seed", 42))

    _apply_commands(cfg, data.get("command", {}))
    _apply_domain_randomization(cfg, data.get("domain_randomization", {}))
    _apply_slippery_patches(cfg, data.get("domain_randomization", {}))
    _apply_perturbation(cfg, data.get("perturbation", {}))

    return cfg


def make_gym_env(config: str | Path | PhoenixConfig, render: bool = False):
    """Create the gym env paired with the cfg. Returns ``(env, cfg, task_name)``."""
    import gymnasium as gym

    if not isinstance(config, PhoenixConfig):
        config = load_layered_config(config)
    cfg = build_env_cfg(config)
    task_name = config.to_container()["env"]["task_name"]
    env = gym.make(task_name, cfg=cfg, render_mode="rgb_array" if render else None)
    return env, cfg, task_name
