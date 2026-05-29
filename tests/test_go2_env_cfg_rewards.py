"""Regression tests for reward-section wiring in go2_env_cfg.

These tests run in CI (non-sim). They exercise the pure-Python helpers
added in Phase 0 of the 2026-04-19 phoenix retrain plan.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from phoenix.sim_env.go2_env_cfg import (
    _REWARD_TERM_MAP,
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
    """Stand-in for Isaac Lab RewardTermCfg — only `.weight` is exercised."""

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


def test_unwired_sections_flags_dropped_dr_keys() -> None:
    """Audit 2026-05-21 (FIXLIST Critical): ``domain_randomization`` is a
    wired section, but ``motor_strength_scale`` / ``actuator_latency_steps``
    are declared in base.yaml and silently dropped by
    ``_apply_domain_randomization``. The unwired check must flag dropped keys
    *inside* a wired section, not just unwired section names, so the train/sim
    mismatch is loud. This was the most plausible mechanical cause of the
    33 percent hardware slew saturation. Locking it against regression to the
    section-name-only check that originally hid the drop."""
    dr = {
        "enabled": True,
        "friction_range": [0.4, 1.0],
        "restitution_range": [0.0, 0.1],
        "mass_offset_kg": [-1.0, 1.0],
        "motor_strength_scale": [0.9, 1.1],
        "actuator_latency_steps": 2,
    }
    unwired = _unwired_sections_present({"domain_randomization": dr})
    assert "domain_randomization.motor_strength_scale" in unwired
    assert "domain_randomization.actuator_latency_steps" in unwired
    # The genuinely applied keys must NOT be flagged.
    assert "domain_randomization.enabled" not in unwired
    assert "domain_randomization.friction_range" not in unwired
    assert "domain_randomization.restitution_range" not in unwired
    assert "domain_randomization.mass_offset_kg" not in unwired


def test_unwired_sections_does_not_flag_fully_applied_dr() -> None:
    """A ``domain_randomization`` block containing only applied keys produces
    no warning — the check must not be noisy on the supported config."""
    dr = {
        "enabled": True,
        "friction_range": [0.4, 1.0],
        "restitution_range": [0.0, 0.1],
        "mass_offset_kg": [-1.0, 1.0],
    }
    assert _unwired_sections_present({"domain_randomization": dr}) == []


def test_unwired_sections_flags_observation_noise() -> None:
    unwired = _unwired_sections_present({"observation": {"noise": {"scale": 0.1}}})
    assert unwired == ["observation.noise"]


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
    AttributeError that names both the upstream term and the YAML key —
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
