"""Regression tests for reward-section wiring in go2_env_cfg.

These tests run in CI (non-sim). They exercise the pure-Python helpers
added in Phase 0 of the 2026-04-19 phoenix retrain plan.
"""

from __future__ import annotations

import logging

import pytest

from phoenix.sim_env.go2_env_cfg import _REWARD_TERM_MAP


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
