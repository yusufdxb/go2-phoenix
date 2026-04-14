"""Tests for the policy observation builder."""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real.observation import JointOrder, ObservationBuilder

JOINT_NAMES = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)  # fmt: skip

DEFAULTS = {n: 0.5 for n in JOINT_NAMES}


def _builder() -> ObservationBuilder:
    return ObservationBuilder(JointOrder(JOINT_NAMES), DEFAULTS)


def test_dim_is_48() -> None:
    assert _builder().dim == 48


def test_build_layout_matches_policy() -> None:
    b = _builder()
    obs = b.build(
        base_lin_vel=np.asarray([1.0, 2.0, 3.0]),
        base_ang_vel=np.asarray([4.0, 5.0, 6.0]),
        projected_gravity=np.asarray([0.0, 0.0, -1.0]),
        velocity_command=np.asarray([0.5, 0.0, 0.1]),
        joint_pos=np.ones(12) * 0.5,  # equals defaults → rel == 0
        joint_vel=np.zeros(12),
        last_action=None,
    )
    assert obs.shape == (48,)
    assert np.allclose(obs[:3], [1.0, 2.0, 3.0])
    assert np.allclose(obs[3:6], [4.0, 5.0, 6.0])
    assert np.allclose(obs[6:9], [0.0, 0.0, -1.0])
    assert np.allclose(obs[9:12], [0.5, 0.0, 0.1])
    # joint_pos - default_q all zero
    assert np.allclose(obs[12:24], 0.0)
    # joint_vel zero
    assert np.allclose(obs[24:36], 0.0)
    # last_action zero
    assert np.allclose(obs[36:48], 0.0)


def test_remap_reorders_joint_state() -> None:
    order = JointOrder(("a", "b", "c"))
    ros_names = ["c", "a", "b"]
    idx = order.remap(ros_names)
    values = np.asarray([10.0, 20.0, 30.0])  # aligned with ros_names
    assert np.allclose(values[idx], [20.0, 30.0, 10.0])  # a, b, c


def test_remap_missing_joint_raises() -> None:
    order = JointOrder(("a", "b"))
    with pytest.raises(KeyError, match="missing"):
        order.remap(["a"])


def test_build_rejects_wrong_action_dim() -> None:
    b = _builder()
    with pytest.raises(ValueError, match="dim mismatch"):
        b.build(
            base_lin_vel=np.zeros(3),
            base_ang_vel=np.zeros(3),
            projected_gravity=np.zeros(3),
            velocity_command=np.zeros(3),
            joint_pos=np.zeros(12),
            joint_vel=np.zeros(12),
            last_action=np.zeros(11),  # wrong!
        )
