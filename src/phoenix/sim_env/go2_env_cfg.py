"""Build a Phoenix-flavoured GO2 env cfg from a layered YAML config.

The factory produces an Isaac Lab ``ManagerBasedRLEnvCfg`` starting from the
upstream ``UnitreeGo2RoughEnvCfg``, then applies failure-oriented overrides:

* friction / restitution / mass domain randomization
* slippery terrain overlay (narrowed friction range)
* base push perturbations via ``base_external_force_torque``
* velocity-command ranges + ``rel_standing_envs``

**Which YAML sections are wired, which are not** (2026-04-17 audit):

Wired → override upstream defaults:
    env, command, domain_randomization, perturbation, reward, seed

Present in ``base.yaml`` but NOT wired (upstream Go2 defaults win):
    observation.noise, termination, robot.init_state, robot.actuator

Reward wiring added 2026-04-19 (retrain spec Phase 0); prior to this,
YAML reward.* overrides were silent no-ops. This change invalidates
v3b as a reproducible baseline — v3b checkpoint stays as the frozen
reference for comparisons but cannot be re-created from its config.

``_warn_unwired_sections`` logs a warning when an unwired section is present
in the loaded config so the drift is loud, not silent. Turning these on is a
deliberate act that changes training behavior and invalidates v3b
reproducibility, so it is a separate PR, not a quiet edit here.

Isaac Lab imports are done lazily so the module can still be imported in
CI (which has no ``torch`` / ``isaaclab``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config_loader import PhoenixConfig, load_layered_config

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from isaaclab.envs import ManagerBasedRLEnvCfg

logger = logging.getLogger("phoenix.sim_env.go2_env_cfg")

_UNWIRED_TOP_LEVEL = ("termination",)
_UNWIRED_ROBOT_SUB = ("init_state", "actuator")

# YAML reward key -> upstream Isaac Lab RewardsCfg term attribute name.
# Upstream term names live at
# IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/
#   velocity/velocity_env_cfg.py (class RewardsCfg).
# Only terms supported by UnitreeGo2RoughEnvCfg are listed. Keys in YAML
# not present here raise KeyError in _apply_rewards — we do NOT want
# silent drift reappearing.
_REWARD_TERM_MAP: dict[str, str] = {
    "track_lin_vel_xy": "track_lin_vel_xy_exp",
    "track_ang_vel_z": "track_ang_vel_z_exp",
    "lin_vel_z": "lin_vel_z_l2",
    "ang_vel_xy": "ang_vel_xy_l2",
    "joint_torque": "dof_torques_l2",
    "joint_acc": "dof_acc_l2",
    "action_rate": "action_rate_l2",
    "feet_air_time": "feet_air_time",
}


def _unwired_sections_present(data: dict[str, Any]) -> list[str]:
    """Return config-path names of sections present in ``data`` but not applied.

    Used by ``build_env_cfg`` to warn loudly at construction time when the YAML
    contains overrides we don't actually plumb into the env cfg. Pure function
    (no Isaac Lab imports) so it can be unit-tested without a sim app.
    """
    unwired: list[str] = []
    for key in _UNWIRED_TOP_LEVEL:
        if key in data:
            unwired.append(key)
    obs = data.get("observation")
    if isinstance(obs, dict) and "noise" in obs:
        unwired.append("observation.noise")
    robot = data.get("robot")
    if isinstance(robot, dict):
        for sub in _UNWIRED_ROBOT_SUB:
            if sub in robot:
                unwired.append(f"robot.{sub}")
    return unwired


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
    # Fraction of envs that get velocity_command = 0 each episode. Without this,
    # canonical stand (cmd=0) is a measure-zero event in the sampler → never
    # seen at train time → extrapolated to huge actions at deploy. 2% matches
    # Isaac Lab's legged-locomotion baseline defaults.
    if "rel_standing_envs" in cmd and hasattr(vel_cmd, "rel_standing_envs"):
        vel_cmd.rel_standing_envs = float(cmd["rel_standing_envs"])


def _apply_rewards(env_cfg: Any, rewards: dict[str, Any]) -> None:
    """Apply YAML reward overrides to Isaac Lab env cfg reward term weights.

    For each key in ``rewards``, look up the upstream term name in
    ``_REWARD_TERM_MAP`` and set ``env_cfg.rewards.<term>.weight``. An
    unknown key raises KeyError — this is deliberate, to prevent the
    silent-no-op drift that motivated adding this helper (see
    :mod:`phoenix.sim_env.go2_env_cfg` module docstring, 2026-04-19).
    """
    if not rewards:
        return
    for yaml_key, weight in rewards.items():
        if yaml_key not in _REWARD_TERM_MAP:
            raise KeyError(
                f"Unknown reward key {yaml_key!r} — add it to _REWARD_TERM_MAP "
                f"or remove from YAML. Known keys: {sorted(_REWARD_TERM_MAP)}"
            )
        term_name = _REWARD_TERM_MAP[yaml_key]
        term = getattr(env_cfg.rewards, term_name, None)
        if term is None:
            raise AttributeError(
                f"Reward term {term_name!r} (YAML key {yaml_key!r}) not present on "
                f"{type(env_cfg.rewards).__name__}. Either the upstream task omits "
                f"this term, or _REWARD_TERM_MAP is stale."
            )
        term.weight = float(weight)


def build_env_cfg(config: str | Path | PhoenixConfig) -> ManagerBasedRLEnvCfg:
    """Build a GO2 env cfg, applying YAML overrides on top of the upstream task."""
    import importlib

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401 - registers tasks

    if not isinstance(config, PhoenixConfig):
        config = load_layered_config(config)
    data = config.to_container()
    env_blk = data["env"]

    unwired = _unwired_sections_present(data)
    if unwired:
        logger.warning(
            "phoenix env cfg: YAML sections present but not applied to env_cfg — "
            "upstream Go2 defaults win: %s. See go2_env_cfg.py module docstring.",
            ", ".join(unwired),
        )

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
    _apply_rewards(cfg, data.get("reward", {}))

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
