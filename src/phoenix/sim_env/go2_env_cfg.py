"""Build a Phoenix-flavoured GO2 env cfg from a layered YAML config.

The factory produces an Isaac Lab ``ManagerBasedRLEnvCfg`` starting from the
upstream ``UnitreeGo2RoughEnvCfg``, then applies failure-oriented overrides:

* friction / restitution / mass domain randomization
* slippery terrain overlay (narrowed friction range)
* base push perturbations via ``base_external_force_torque``

Isaac Lab imports are done lazily so the module can still be imported in
CI (which has no ``torch`` / ``isaaclab``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config_loader import PhoenixConfig, load_layered_config

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from isaaclab.envs import ManagerBasedRLEnvCfg


def _events_root(env_cfg: Any) -> Any:
    """Return the concrete event container.

    Isaac Lab's GO2 task wraps events in a ``PresetCfg`` (``default`` /
    ``newton`` / ``physx``). We always operate on ``events.default`` since
    that's what ``physx`` aliases to on this machine.
    """
    events = env_cfg.events
    return events.default if hasattr(events, "default") else events


def _apply_domain_randomization(env_cfg: Any, dr: dict[str, Any]) -> None:
    """Patch friction / restitution / mass DR ranges in the event terms."""
    if not dr.get("enabled", True):
        return
    events = _events_root(env_cfg)

    pm = getattr(events, "physics_material", None)
    if pm is not None:
        fr_lo, fr_hi = dr["friction_range"]
        rs_lo, rs_hi = dr.get("restitution_range", [0.0, 0.0])
        pm.params["static_friction_range"] = (float(fr_lo), float(fr_hi))
        pm.params["dynamic_friction_range"] = (float(fr_lo), float(fr_hi))
        pm.params["restitution_range"] = (float(rs_lo), float(rs_hi))

    abm = getattr(events, "add_base_mass", None)
    if abm is not None and "mass_offset_kg" in dr:
        m_lo, m_hi = dr["mass_offset_kg"]
        abm.params["mass_distribution_params"] = (float(m_lo), float(m_hi))


def _apply_perturbation(env_cfg: Any, pert: dict[str, Any]) -> None:
    """Turn perturbations on/off via ``base_external_force_torque``.

    The GO2 preset disables the velocity-style ``push_robot`` event; we
    instead modulate the reset-mode external force/torque applied to the
    base, which the upstream cfg retains. When the overlay is disabled
    we zero the ranges so behaviour matches the base config.
    """
    events = _events_root(env_cfg)
    efx = getattr(events, "base_external_force_torque", None)
    if efx is None:
        return

    if not pert.get("enabled", False):
        efx.params["force_range"] = (0.0, 0.0)
        efx.params["torque_range"] = (0.0, 0.0)
        return

    vel_xy = float(pert["push_velocity_xy"])
    vel_yaw = float(pert["push_velocity_yaw"])
    # Convert a ~1 m/s impulse intent into a proxy body-frame force spike.
    # The robot is ~15 kg — f ≈ m·Δv/Δt over one control step (0.02 s).
    push_force = 15.0 * vel_xy / 0.02
    push_torque = 2.0 * vel_yaw / 0.02
    efx.params["force_range"] = (-push_force, push_force)
    efx.params["torque_range"] = (-push_torque, push_torque)


def _apply_commands(env_cfg: Any, cmd: dict[str, Any]) -> None:
    if not cmd or not hasattr(env_cfg, "commands") or env_cfg.commands is None:
        return
    vel_cmd = getattr(env_cfg.commands, "base_velocity", None)
    if vel_cmd is None:
        return
    vel_cmd.ranges.lin_vel_x = tuple(cmd["lin_vel_x"])
    vel_cmd.ranges.lin_vel_y = tuple(cmd["lin_vel_y"])
    vel_cmd.ranges.ang_vel_z = tuple(cmd["ang_vel_z"])
    vel_cmd.resampling_time_range = (cmd["resample_time_s"], cmd["resample_time_s"])


def build_env_cfg(config: str | Path | PhoenixConfig) -> ManagerBasedRLEnvCfg:
    """Build a GO2 env cfg, applying YAML overrides on top of the upstream task."""
    import importlib

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401 - registers tasks

    if not isinstance(config, PhoenixConfig):
        config = load_layered_config(config)
    data = config.to_container()
    env_blk = data["env"]

    task_name = env_blk["task_name"]
    env_cfg_entry = gym.spec(task_name).kwargs["env_cfg_entry_point"]
    module_name, class_name = env_cfg_entry.split(":")
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
