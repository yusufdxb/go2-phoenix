"""Build a Phoenix-flavoured GO2 env cfg from a layered YAML config.

The factory produces an Isaac Lab ``ManagerBasedRLEnvCfg`` starting from the
upstream ``UnitreeGo2RoughEnvCfg``, then applies failure-oriented overrides:

* friction / restitution / mass domain randomization
* motor-strength scale DR (scales actuator stiffness and damping uniformly)
* actuator latency DR (action delay steps stored as event attribute)
* slippery terrain overlay (narrowed friction range)
* base push perturbations via ``base_external_force_torque``
* velocity-command ranges + ``rel_standing_envs``

**Which YAML sections are wired, which are not** (2026-04-17 audit,
updated 2026-06-07 wiring PR):

Wired (override upstream defaults):
    env, command, domain_randomization (friction / restitution / mass /
    motor_strength_scale / actuator_latency_steps), perturbation, reward, seed

Present in ``base.yaml`` but NOT wired (upstream Go2 defaults win):
    observation.noise, termination, robot.init_state, robot.actuator

Reward wiring added 2026-04-19 (retrain spec Phase 0); prior to this,
YAML reward.* overrides were silent no-ops. This change invalidates
v3b as a reproducible baseline — v3b checkpoint stays as the frozen
reference for comparisons but cannot be re-created from its config.

``_unwired_sections_present`` flags both unwired sections and unapplied keys
inside a wired section (e.g. ``domain_randomization.motor_strength_scale``);
``build_env_cfg`` logs a warning on each, so the drift is loud, not silent.
Turning these on is a deliberate act that changes training behavior and
invalidates v3b reproducibility, so it is a separate PR, not a quiet edit here.

Isaac Lab imports are done lazily so the module can still be imported in
CI (which has no ``torch`` / ``isaaclab``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config_loader import PhoenixConfig, load_layered_config

# Isaac Lab's RewardTermCfg — lazy-import guard so this module can
# still be imported in CI without isaaclab. The actual usage is
# gated behind _NEW_TERM_FACTORIES, which only fires at env build.
try:  # pragma: no cover - exercised only on machines with Isaac Lab
    from isaaclab.managers import RewardTermCfg as _RewTerm
except ImportError:  # pragma: no cover
    _RewTerm = None  # type: ignore[assignment]

from phoenix.sim_env.rate_limit import MAX_DELTA_PER_STEP_RAD
from phoenix.sim_env.rewards import slew_sat_hinge_l2

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from isaaclab.envs import ManagerBasedRLEnvCfg

logger = logging.getLogger("phoenix.sim_env.go2_env_cfg")

_UNWIRED_TOP_LEVEL = ("termination",)
_UNWIRED_ROBOT_SUB = ("init_state", "actuator")

# Keys inside the (wired) ``domain_randomization`` block that
# ``_apply_domain_randomization`` actually plumbs into the env cfg. Any other
# key under ``domain_randomization`` is declared-but-dropped, and
# ``_unwired_sections_present`` flags it so the drift is loud, not silent.
_APPLIED_DR_KEYS = (
    "enabled",
    "friction_range",
    "restitution_range",
    "mass_offset_kg",
    "motor_strength_scale",
    "actuator_latency_steps",
)

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

# Factories for Phoenix-owned reward terms — not in upstream
# UnitreeGo2RoughEnvCfg.rewards. When a YAML key lands here,
# _apply_rewards constructs a RewTerm via the factory and setattrs
# it onto env_cfg.rewards. Keys must not collide with
# _REWARD_TERM_MAP — see _apply_rewards dispatch.
_NEW_TERM_FACTORIES: dict[str, tuple[str, Callable[[float], Any]]] = {
    "slew_sat_hinge": (
        "slew_sat_hinge_l2",  # attribute name on env_cfg.rewards
        lambda weight: _RewTerm(
            func=slew_sat_hinge_l2,
            weight=float(weight),
            params={"threshold": 0.15},
        ),
    ),
}


def _unwired_sections_present(data: dict[str, Any]) -> list[str]:
    """Return config-path names of sections present in ``data`` but not applied.

    Covers both fully unwired sections and unapplied keys inside an otherwise
    wired section (any ``domain_randomization`` key outside ``_APPLIED_DR_KEYS``).
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
    dr = data.get("domain_randomization")
    if isinstance(dr, dict):
        for sub in dr:
            if sub not in _APPLIED_DR_KEYS:
                unwired.append(f"domain_randomization.{sub}")
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
    """Patch DR ranges into event terms and env-cfg attributes.

    Wired knobs:
    * ``friction_range`` / ``restitution_range`` — patched on
      ``events.physics_material.params``.
    * ``mass_offset_kg`` — patched on ``events.add_base_mass.params``.
    * ``motor_strength_scale`` — patched on
      ``events.scale_motor_strength.params`` (stiffness + damping, scale
      operation).  The event term is pre-created by ``_prepare_dr_event_terms``
      inside ``build_env_cfg`` (which has Isaac Lab available); for mock tests
      the attribute can be set directly on the events object.
    * ``actuator_latency_steps`` — written to
      ``events.phoenix_actuator_latency_range`` as a ``(lo, hi)`` tuple.  The
      training harness reads this attribute to configure its action-delay
      buffer; it is intentionally a plain attribute rather than an event term
      because Isaac Lab has no built-in startup event for discrete action delay.
    """
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

    # --- motor_strength_scale ------------------------------------------------
    # Scales actuator stiffness and damping uniformly per episode reset.
    # ``events.scale_motor_strength`` is created by ``_prepare_dr_event_terms``
    # in the sim context. Mock tests may set it directly.
    sms = getattr(events, "scale_motor_strength", None)
    if sms is not None and "motor_strength_scale" in dr:
        lo, hi = dr["motor_strength_scale"]
        sms.params["stiffness_distribution_params"] = (float(lo), float(hi))
        sms.params["damping_distribution_params"] = (float(lo), float(hi))

    # --- actuator_latency_steps ----------------------------------------------
    # Stores the latency range on the env_cfg ROOT (not on ``events``): Isaac's
    # EventManager._prepare_terms scans every attribute of the events cfg and
    # rejects anything that is not an EventTermCfg, so a bare tuple there crashes
    # env init. The action-delay buffer reads ``env_cfg.phoenix_actuator_latency_range``.
    # Range is (lo, hi) in steps at 200 Hz (5 ms/step → 1 step ≈ 5 ms).
    # ponytail: plain attribute, no action-delay buffer consumes it yet (no Isaac
    # primitive for discrete action delay); implement the buffer when latency DR matters.
    if "actuator_latency_steps" in dr:
        lat = dr["actuator_latency_steps"]
        if isinstance(lat, (list, tuple)):
            lo_lat, hi_lat = lat
        else:
            lo_lat = hi_lat = lat
        env_cfg.phoenix_actuator_latency_range = (int(lo_lat), int(hi_lat))


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
    """Apply YAML reward overrides to Isaac Lab env cfg.

    Upstream-term keys (in ``_REWARD_TERM_MAP``) reweight an existing
    ``env_cfg.rewards.<term>`` by setting its ``weight``.

    Phoenix-owned-term keys (in ``_NEW_TERM_FACTORIES``) construct a
    new ``RewTerm`` via the factory and attach it to
    ``env_cfg.rewards`` under the factory's attribute name.

    Unknown keys raise ``KeyError`` — this is deliberate, to prevent
    the silent-no-op drift that motivated adding this helper (see
    :mod:`phoenix.sim_env.go2_env_cfg` module docstring, 2026-04-19).
    """
    if not rewards:
        return
    for yaml_key, weight in rewards.items():
        if yaml_key in _REWARD_TERM_MAP:
            term_name = _REWARD_TERM_MAP[yaml_key]
            term = getattr(env_cfg.rewards, term_name, None)
            if term is None:
                raise AttributeError(
                    f"Reward term {term_name!r} (YAML key {yaml_key!r}) not present on "
                    f"{type(env_cfg.rewards).__name__}. Either the upstream task omits "
                    f"this term, or _REWARD_TERM_MAP is stale."
                )
            term.weight = float(weight)
        elif yaml_key in _NEW_TERM_FACTORIES:
            attr_name, factory = _NEW_TERM_FACTORIES[yaml_key]
            setattr(env_cfg.rewards, attr_name, factory(weight))
        else:
            raise KeyError(
                f"Unknown reward key {yaml_key!r} — add it to _REWARD_TERM_MAP, "
                f"add it to _NEW_TERM_FACTORIES, or remove from YAML. "
                f"Known upstream keys: {sorted(_REWARD_TERM_MAP)}; "
                f"known phoenix keys: {sorted(_NEW_TERM_FACTORIES)}"
            )


def scale_explicit_actuator_gains(
    env: Any,
    env_ids: Any,
    asset_cfg: Any,
    stiffness_distribution_params: tuple[float, float] | None = None,
    damping_distribution_params: tuple[float, float] | None = None,
) -> None:
    """Startup motor-strength DR for EXPLICIT actuators (e.g. Go2 ``DCMotor``).

    Per-env, per-joint scales the actuator MODEL's stiffness and damping in
    place by a uniform factor drawn from the given ranges.

    Why not ``isaaclab...randomize_actuator_gains``: that term resets gains to
    ``asset.data.joint_stiffness`` (the implicit sim drive, which is 0 for an
    explicit actuator since its PD is computed in software) and writes that
    back to ``actuator.stiffness`` for every actuator, ZEROING explicit-actuator
    gains. The Go2 uses ``DCMotor`` (explicit), so it collapses. The DCMotor PD
    law reads ``actuator.stiffness``/``actuator.damping`` each step, so scaling
    those directly is the correct, sim-write-free DR for it.

    Args:
        env: The environment instance.
        env_ids: Indices of environments to randomize, or None for all.
        asset_cfg: Scene-entity config selecting the articulation.
        stiffness_distribution_params: ``(lo, hi)`` scale range for stiffness.
        damping_distribution_params: ``(lo, hi)`` scale range for damping.
    """
    import torch

    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(asset.num_instances, device=asset.device)
    for actuator in asset.actuators.values():
        if stiffness_distribution_params is not None:
            lo, hi = stiffness_distribution_params
            fac = torch.empty(
                (len(env_ids), actuator.stiffness.shape[1]), device=asset.device
            ).uniform_(float(lo), float(hi))
            actuator.stiffness[env_ids] = actuator.stiffness[env_ids] * fac
        if damping_distribution_params is not None:
            lo, hi = damping_distribution_params
            fac = torch.empty(
                (len(env_ids), actuator.damping.shape[1]), device=asset.device
            ).uniform_(float(lo), float(hi))
            actuator.damping[env_ids] = actuator.damping[env_ids] * fac


def _prepare_dr_event_terms(env_cfg: Any, dr: dict[str, Any]) -> None:
    """Pre-create Phoenix-owned DR event terms that upstream GO2 cfg omits.

    Must be called inside a sim context (Isaac Lab available) before
    ``_apply_domain_randomization``, which patches params on these terms.

    Currently creates:
    * ``events.scale_motor_strength`` — a startup term that scales the explicit
      DCMotor actuator gains for ``motor_strength_scale`` (see
      :func:`scale_explicit_actuator_gains` for why the built-in
      ``randomize_actuator_gains`` cannot be used here).

    The actuator-latency term is intentionally omitted: Isaac Lab has no
    built-in startup event for discrete action delay; the range is stored
    as a plain attribute ``env_cfg.phoenix_actuator_latency_range`` by
    ``_apply_domain_randomization`` instead.
    """
    if not dr.get("enabled", True) or "motor_strength_scale" not in dr:
        return

    from isaaclab.managers import EventTermCfg as EventTerm  # type: ignore[import]
    from isaaclab.managers import SceneEntityCfg  # type: ignore[import]

    events = _events_root(env_cfg)
    if getattr(events, "scale_motor_strength", None) is None:
        term = EventTerm(
            func=scale_explicit_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "stiffness_distribution_params": (1.0, 1.0),  # placeholder; overwritten below
                "damping_distribution_params": (1.0, 1.0),
            },
        )
        events.scale_motor_strength = term


def _apply_rate_limit(env_cfg: Any, action: dict[str, Any]) -> None:
    """Swap the joint-position action term for the rate-limited variant.

    Makes the deploy-side per-step slew cap part of the MDP: the policy is
    trained against ``current_q ± max_delta_per_step``, the exact limiter the
    Jetson bridge enforces (``sim2real.safety.per_step_clip_array``). Without
    this the limiter exists only at deploy and the policy meets it for the first
    time on hardware — the closed-loop mismatch behind the 0.33%->33% slew blowup.

    Defaults to ENABLED with the canonical ``MAX_DELTA_PER_STEP_RAD``. Set
    ``action.rate_limit.enabled: false`` only to reproduce a pre-limiter baseline.
    """
    rl = (action or {}).get("rate_limit", {})
    if not rl.get("enabled", True):
        logger.warning(
            "phoenix env cfg: per-step rate limiter DISABLED "
            "(action.rate_limit.enabled=false) — sim will NOT match the deploy "
            "slew cap; do this only for a pre-limiter baseline."
        )
        return

    # Lazy import: this pulls in Isaac Lab, so it must not touch the no-sim CI path.
    from phoenix.sim_env.rate_limited_action import RateLimitedJointPositionActionCfg

    base = env_cfg.actions.joint_pos
    max_delta = float(rl.get("max_delta_per_step", MAX_DELTA_PER_STEP_RAD))
    env_cfg.actions.joint_pos = RateLimitedJointPositionActionCfg(
        asset_name=base.asset_name,
        joint_names=base.joint_names,
        scale=base.scale,
        offset=base.offset,
        use_default_offset=base.use_default_offset,
        preserve_order=base.preserve_order,
        clip=getattr(base, "clip", None),
        max_delta_per_step=max_delta,
    )
    # Warning-level on purpose: this alters the training action distribution to
    # match deploy, and given this repo's history of silent DR no-ops it must be
    # loudly visible in every run's log, not buried at info level.
    logger.warning(
        "phoenix env cfg: per-step rate limiter ACTIVE (max_delta=%.4f rad/step) "
        "— sim action pipeline now matches the deploy slew cap.",
        max_delta,
    )


def _apply_fixture(env_cfg: Any, fixture: dict[str, Any]) -> None:
    """Stash the unloaded-feet fixture config on the env cfg root.

    The fixture clamp needs the LIVE env (per-substep root writes), which does
    not exist at cfg-build time, so this only records the request as
    ``env_cfg.phoenix_fixture``. The eval/training entry point installs the
    clamp via ``phoenix.sim_env.fixture_hold.install_fixture_hold`` after the
    env is constructed. Stored on the cfg root (not on ``events``) for the same
    reason as ``phoenix_actuator_latency_range``: EventManager rejects non-
    EventTermCfg attributes on the events cfg.
    """
    if fixture and fixture.get("enabled", False):
        env_cfg.phoenix_fixture = dict(fixture)
        logger.warning(
            "phoenix env cfg: stand-fixture scenario REQUESTED "
            "(rel=%.2f, hold_height=%.2fm, roll=%.2frad) — trunk will be pinned "
            "in the air at env construction; this is an OOD eval, not ground standing.",
            float(fixture.get("rel_fixture_envs", 1.0)),
            float(fixture.get("hold_height_m", 0.55)),
            float(fixture.get("roll_rad", 0.0)),
        )


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
    # Pre-create Phoenix DR event terms that the upstream GO2 cfg omits
    # (e.g. scale_motor_strength). Must run before _apply_domain_randomization.
    _prepare_dr_event_terms(cfg, data.get("domain_randomization", {}))
    _apply_domain_randomization(cfg, data.get("domain_randomization", {}))
    _apply_perturbation(cfg, data.get("perturbation", {}))
    _apply_rewards(cfg, data.get("reward", {}))
    _apply_rate_limit(cfg, data.get("action", {}))
    _apply_fixture(cfg, data.get("fixture", {}))

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
