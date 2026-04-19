"""Regression tests for reward-section wiring in go2_env_cfg.

These tests run in CI (non-sim). They exercise the pure-Python helpers
added in Phase 0 of the 2026-04-19 phoenix retrain plan.
"""

from __future__ import annotations

import logging

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
    unwired = _unwired_sections_present(
        {"termination": {"pitch_threshold_rad": 0.8}}
    )
    assert unwired == ["termination"]


def test_apply_rewards_missing_term_on_env_cfg_raises_with_context() -> None:
    """If the env cfg's RewardsCfg doesn't have the mapped term at all
    (e.g. a flat-env subclass dropped feet_air_time), we want a clear
    AttributeError that names both the upstream term and the YAML key —
    not a bare AttributeError from getattr."""
    env_cfg = _FakeEnvCfg(_FakeRewards())  # no reward terms at all
    with pytest.raises(AttributeError) as exc:
        _apply_rewards(env_cfg, {"feet_air_time": 0.5})
    msg = str(exc.value)
    assert "feet_air_time" in msg              # upstream term name
    assert "'feet_air_time'" in msg or "feet_air_time" in msg  # yaml key
    assert "_REWARD_TERM_MAP" in msg or "upstream task omits" in msg
