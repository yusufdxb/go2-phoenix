"""Regression tests for reward-section wiring in go2_env_cfg.

These tests run in CI (non-sim). They exercise the pure-Python helpers
added in Phase 0 of the 2026-04-19 phoenix retrain plan.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from phoenix.sim_env.go2_env_cfg import (
    _APPLIED_DR_KEYS,
    _REWARD_TERM_MAP,
    _apply_domain_randomization,
    _apply_rewards,
    _unwired_sections_present,
)


def test_reward_term_map_covers_phoenix_base_keys() -> None:
    """Every reward key we keep in base.yaml must map to an upstream
    Isaac Lab reward term name."""
    expected = {
        "track_lin_vel_xy": "track_lin_vel_xy_exp",
        "track_ang_vel_z": "track_ang_vel_z_exp",
        "lin_vel_z": "lin_vel_z_l2",
        "ang_vel_xy": "ang_vel_xy_l2",
        "joint_torque": "dof_torques_l2",
        "joint_acc": "dof_acc_l2",
        "action_rate": "action_rate_l2",
        "feet_air_time": "feet_air_time",
    }
    assert _REWARD_TERM_MAP == expected


class _FakeRewardTerm:
    """Stand-in for Isaac Lab RewardTermCfg, only `.weight` is exercised."""

    def __init__(self, weight: float):
        self.weight = weight


class _FakeRewards:
    """Attribute-access container matching RewardsCfg's term-as-attr pattern."""

    def __init__(self, **terms):
        for k, v in terms.items():
            setattr(self, k, v)


class _FakeEnvCfg:
    def __init__(self, rewards):
        self.rewards = rewards


def test_apply_rewards_sets_weights() -> None:
    env_cfg = _FakeEnvCfg(
        _FakeRewards(
            action_rate_l2=_FakeRewardTerm(-0.01),
            dof_acc_l2=_FakeRewardTerm(-2.5e-7),
        )
    )
    _apply_rewards(env_cfg, {"action_rate": -0.5, "joint_acc": -1.0e-6})
    assert env_cfg.rewards.action_rate_l2.weight == -0.5
    assert env_cfg.rewards.dof_acc_l2.weight == -1.0e-6


def test_apply_rewards_unknown_key_raises() -> None:
    env_cfg = _FakeEnvCfg(_FakeRewards())
    with pytest.raises(KeyError, match="bogus_term"):
        _apply_rewards(env_cfg, {"bogus_term": -1.0})


def test_apply_rewards_empty_dict_is_noop() -> None:
    env_cfg = _FakeEnvCfg(_FakeRewards(action_rate_l2=_FakeRewardTerm(-0.01)))
    _apply_rewards(env_cfg, {})
    assert env_cfg.rewards.action_rate_l2.weight == -0.01


def test_reward_no_longer_in_unwired_top_level() -> None:
    """Phase 0 of the 2026-04-19 retrain removes 'reward' from the
    unwired list. 'termination' and the robot sub-keys stay unwired
    (separate PRs)."""
    from phoenix.sim_env.go2_env_cfg import _UNWIRED_TOP_LEVEL

    assert "reward" not in _UNWIRED_TOP_LEVEL
    assert "termination" in _UNWIRED_TOP_LEVEL  # intentionally unchanged


def test_unwired_sections_does_not_flag_reward() -> None:
    unwired = _unwired_sections_present({"reward": {"action_rate": -0.5}})
    assert unwired == []


def test_unwired_sections_still_flags_termination() -> None:
    unwired = _unwired_sections_present({"termination": {"pitch_threshold_rad": 0.8}})
    assert unwired == ["termination"]


def test_wired_dr_keys_not_flagged() -> None:
    """2026-06-07 DR-wiring PR: ``motor_strength_scale`` and
    ``actuator_latency_steps`` are now wired into ``_apply_domain_randomization``
    and added to ``_APPLIED_DR_KEYS``. ``_unwired_sections_present`` must NOT
    flag them, regression guard against accidental removal from
    ``_APPLIED_DR_KEYS``."""
    dr = {
        "enabled": True,
        "friction_range": [0.4, 1.0],
        "restitution_range": [0.0, 0.1],
        "mass_offset_kg": [-1.0, 1.0],
        "motor_strength_scale": [0.9, 1.1],
        "actuator_latency_steps": 2,
    }
    unwired = _unwired_sections_present({"domain_randomization": dr})
    assert "domain_randomization.motor_strength_scale" not in unwired
    assert "domain_randomization.actuator_latency_steps" not in unwired
    # The other applied keys must also not be flagged.
    assert "domain_randomization.enabled" not in unwired
    assert "domain_randomization.friction_range" not in unwired
    assert "domain_randomization.restitution_range" not in unwired
    assert "domain_randomization.mass_offset_kg" not in unwired


def test_unwired_sections_does_not_flag_fully_applied_dr() -> None:
    """A ``domain_randomization`` block containing only applied keys produces
    no warning, the check must not be noisy on the supported config."""
    dr = {
        "enabled": True,
        "friction_range": [0.4, 1.0],
        "restitution_range": [0.0, 0.1],
        "mass_offset_kg": [-1.0, 1.0],
    }
    assert _unwired_sections_present({"domain_randomization": dr}) == []


def test_unwired_sections_does_not_flag_observation_noise() -> None:
    # observation.noise is wired (2026-06-21) by _apply_observation_noise.
    unwired = _unwired_sections_present({"observation": {"noise": {"joint_pos": 0.01}}})
    assert "observation.noise" not in unwired


def test_unwired_sections_flags_observation_include() -> None:
    unwired = _unwired_sections_present({"observation": {"include": ["joint_pos"]}})
    assert unwired == ["observation.include"]


def test_unwired_sections_flags_robot_sub_keys() -> None:
    """``robot.init_state`` and ``robot.actuator`` are present-but-unwired;
    upstream Go2 defaults win for both."""
    unwired = _unwired_sections_present(
        {"robot": {"init_state": {"pos": [0, 0, 0.4]}, "actuator": {"stiffness": 25.0}}}
    )
    assert "robot.init_state" in unwired
    assert "robot.actuator" in unwired


def test_unwired_sections_empty_config_is_clean() -> None:
    assert _unwired_sections_present({}) == []


def test_apply_rewards_missing_term_on_env_cfg_raises_with_context() -> None:
    """If the env cfg's RewardsCfg doesn't have the mapped term at all
    (e.g. a flat-env subclass dropped feet_air_time), we want a clear
    AttributeError that names both the upstream term and the YAML key,
    not a bare AttributeError from getattr."""
    env_cfg = _FakeEnvCfg(_FakeRewards())  # no reward terms at all
    with pytest.raises(AttributeError) as exc:
        _apply_rewards(env_cfg, {"feet_air_time": 0.5})
    msg = str(exc.value)
    assert "feet_air_time" in msg  # upstream term name
    assert "'feet_air_time'" in msg or "feet_air_time" in msg  # yaml key
    assert "_REWARD_TERM_MAP" in msg or "upstream task omits" in msg


def test_apply_rewards_new_term_factory_attaches_reward() -> None:
    from phoenix.sim_env.go2_env_cfg import _NEW_TERM_FACTORIES

    assert "slew_sat_hinge" in _NEW_TERM_FACTORIES

    class _StubRewTerm:
        def __init__(self, *, func, weight, params):
            self.func = func
            self.weight = weight
            self.params = params

    env_cfg = _FakeEnvCfg(_FakeRewards())
    with patch("phoenix.sim_env.go2_env_cfg._RewTerm", _StubRewTerm):
        _apply_rewards(env_cfg, {"slew_sat_hinge": -50.0})

    assert hasattr(env_cfg.rewards, "slew_sat_hinge_l2")
    assert env_cfg.rewards.slew_sat_hinge_l2.weight == -50.0
    # Sanity: the factory passed threshold=0.15 per the spec.
    assert env_cfg.rewards.slew_sat_hinge_l2.params == {"threshold": 0.15}


def test_apply_rewards_new_term_factory_mixed_with_upstream() -> None:
    class _StubRewTerm:
        def __init__(self, *, func, weight, params):
            self.func = func
            self.weight = weight
            self.params = params

    env_cfg = _FakeEnvCfg(
        _FakeRewards(action_rate_l2=_FakeRewardTerm(-0.01)),
    )
    with patch("phoenix.sim_env.go2_env_cfg._RewTerm", _StubRewTerm):
        _apply_rewards(
            env_cfg,
            {"action_rate": -0.5, "slew_sat_hinge": -50.0},
        )
    assert env_cfg.rewards.action_rate_l2.weight == -0.5
    assert env_cfg.rewards.slew_sat_hinge_l2.weight == -50.0


# ---------------------------------------------------------------------------
# DR-wiring regression tests (2026-06-07)
# These guard the two keys that were previously declared-but-dropped:
#   motor_strength_scale , wired via events.scale_motor_strength.params
#   actuator_latency_steps, wired via events.phoenix_actuator_latency_range
# ---------------------------------------------------------------------------


class _FakeEventTerm:
    """Minimal stand-in for an Isaac Lab EventTerm, only .params is used."""

    def __init__(self, **params):
        self.params: dict = dict(params)


class _FakeEvents:
    """Attribute container mimicking events.default on Go2PhysxEventsCfg."""

    def __init__(self, **terms):
        for k, v in terms.items():
            setattr(self, k, v)


class _FakeEventEnvCfg:
    """Minimal env_cfg whose .events has no .default (flat events)."""

    def __init__(self, events):
        self.events = events


def test_applied_dr_keys_contains_new_keys() -> None:
    """_APPLIED_DR_KEYS must include both newly wired keys so
    _unwired_sections_present does not flag them."""
    assert "motor_strength_scale" in _APPLIED_DR_KEYS
    assert "actuator_latency_steps" in _APPLIED_DR_KEYS


def test_apply_dr_motor_strength_scale_patches_event_params() -> None:
    """When events.scale_motor_strength exists, _apply_domain_randomization
    must set stiffness_distribution_params and damping_distribution_params
    to the configured range."""
    sms_term = _FakeEventTerm(
        stiffness_distribution_params=(1.0, 1.0),
        damping_distribution_params=(1.0, 1.0),
        operation="scale",
    )
    events = _FakeEvents(
        physics_material=_FakeEventTerm(
            static_friction_range=(0.8, 0.8),
            dynamic_friction_range=(0.6, 0.6),
            restitution_range=(0.0, 0.0),
        ),
        add_base_mass=_FakeEventTerm(mass_distribution_params=(-1.0, 3.0)),
        scale_motor_strength=sms_term,
    )
    env_cfg = _FakeEventEnvCfg(events)

    dr = {
        "enabled": True,
        "friction_range": [0.3, 1.5],
        "restitution_range": [0.0, 0.5],
        "mass_offset_kg": [-2.0, 2.0],
        "motor_strength_scale": [0.85, 1.15],
    }
    _apply_domain_randomization(env_cfg, dr)

    assert sms_term.params["stiffness_distribution_params"] == pytest.approx((0.85, 1.15))
    assert sms_term.params["damping_distribution_params"] == pytest.approx((0.85, 1.15))


def test_apply_dr_motor_strength_scale_skipped_when_term_absent() -> None:
    """If events.scale_motor_strength is not present (upstream cfg not pre-prepared),
    _apply_domain_randomization must not raise, it silently skips."""
    events = _FakeEvents(
        physics_material=_FakeEventTerm(
            static_friction_range=(0.8, 0.8),
            dynamic_friction_range=(0.6, 0.6),
            restitution_range=(0.0, 0.0),
        ),
        add_base_mass=_FakeEventTerm(mass_distribution_params=(-1.0, 3.0)),
        # no scale_motor_strength
    )
    env_cfg = _FakeEventEnvCfg(events)
    dr = {
        "enabled": True,
        "friction_range": [0.3, 1.5],
        "motor_strength_scale": [0.85, 1.15],
    }
    # Must not raise even though the term is absent.
    _apply_domain_randomization(env_cfg, dr)
    assert not hasattr(events, "scale_motor_strength")


def test_apply_dr_actuator_latency_steps_sets_range_list() -> None:
    """When actuator_latency_steps is a [lo, hi] list, the range tuple must
    be stored on env_cfg.phoenix_actuator_latency_range (NOT on events: Isaac's
    EventManager rejects non-EventTermCfg attributes on the events cfg)."""
    events = _FakeEvents()
    env_cfg = _FakeEventEnvCfg(events)
    dr = {"enabled": True, "friction_range": [0.3, 1.5], "actuator_latency_steps": [1, 5]}
    _apply_domain_randomization(env_cfg, dr)
    assert hasattr(env_cfg, "phoenix_actuator_latency_range")
    assert env_cfg.phoenix_actuator_latency_range == (1, 5)


def test_apply_dr_actuator_latency_steps_sets_range_scalar() -> None:
    """When actuator_latency_steps is a scalar, lo == hi == scalar."""
    events = _FakeEvents()
    env_cfg = _FakeEventEnvCfg(events)
    dr = {"enabled": True, "friction_range": [0.3, 1.5], "actuator_latency_steps": 2}
    _apply_domain_randomization(env_cfg, dr)
    assert env_cfg.phoenix_actuator_latency_range == (2, 2)


def test_apply_dr_actuator_latency_steps_absent_leaves_no_attr() -> None:
    """If actuator_latency_steps is not in the DR block, the attribute must
    NOT be created."""
    events = _FakeEvents()
    env_cfg = _FakeEventEnvCfg(events)
    dr = {"enabled": True, "friction_range": [0.3, 1.5]}
    _apply_domain_randomization(env_cfg, dr)
    assert not hasattr(env_cfg, "phoenix_actuator_latency_range")


def test_apply_dr_disabled_skips_all_new_wiring() -> None:
    """When DR is disabled the new wiring must not write any attributes."""
    events = _FakeEvents(
        scale_motor_strength=_FakeEventTerm(
            stiffness_distribution_params=(1.0, 1.0),
            damping_distribution_params=(1.0, 1.0),
        ),
    )
    env_cfg = _FakeEventEnvCfg(events)
    dr = {
        "enabled": False,
        "motor_strength_scale": [0.85, 1.15],
        "actuator_latency_steps": [1, 5],
    }
    _apply_domain_randomization(env_cfg, dr)
    # scale_motor_strength params must remain at placeholder defaults
    assert events.scale_motor_strength.params["stiffness_distribution_params"] == (1.0, 1.0)
    # latency attr must not have been set
    assert not hasattr(env_cfg, "phoenix_actuator_latency_range")


def test_unwired_base_yaml_no_longer_flags_motor_and_latency() -> None:
    """Regression: _unwired_sections_present on a config with both newly wired
    keys must return an empty list for the DR block (all DR keys are applied)."""
    dr_only = {
        "domain_randomization": {
            "enabled": True,
            "friction_range": [0.3, 1.5],
            "restitution_range": [0.0, 0.5],
            "mass_offset_kg": [-2.0, 2.0],
            "motor_strength_scale": [0.85, 1.15],
            "actuator_latency_steps": [1, 5],
        }
    }
    unwired = _unwired_sections_present(dr_only)
    # No DR sub-key should be flagged.
    assert not any(u.startswith("domain_randomization.") for u in unwired)
