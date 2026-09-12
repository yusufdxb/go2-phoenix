"""Tests for the policy observation builder and the base_lin_vel contract."""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real.observation import (
    BASE_LIN_VEL_SOURCE_ODOM,
    BASE_LIN_VEL_SOURCE_ZEROS,
    OBS_TERM_ORDER,
    BaseLinVelUnavailableError,
    JointOrder,
    ObservationBuilder,
    assemble_policy_observation,
    projected_gravity_from_quat,
    resolve_base_lin_vel,
    term_slices,
)

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


# ---------------------------------------------------------------------------
# Term layout contract
# ---------------------------------------------------------------------------


def test_term_slices_cover_the_vector_in_training_order() -> None:
    slices = term_slices(12)
    assert list(slices) == list(OBS_TERM_ORDER)
    assert slices["base_lin_vel"] == slice(0, 3)
    assert slices["base_ang_vel"] == slice(3, 6)
    assert slices["projected_gravity"] == slice(6, 9)
    assert slices["velocity_command"] == slice(9, 12)
    assert slices["joint_pos"] == slice(12, 24)
    assert slices["joint_vel"] == slice(24, 36)
    assert slices["last_action"] == slice(36, 48)
    assert _builder().term_slices() == slices


# ---------------------------------------------------------------------------
# base_lin_vel resolution. The regression: the deploy node fed the policy
# np.zeros(3) here with no config, no log, and no record in the capture.
# ---------------------------------------------------------------------------


def test_odom_source_returns_the_measurement() -> None:
    sample = resolve_base_lin_vel(
        BASE_LIN_VEL_SOURCE_ODOM,
        odom_lin_vel_body=np.asarray([0.4, -0.1, 0.02]),
        odom_valid=True,
        odom_provenance="body_passthrough",
    )
    assert np.allclose(sample.value, [0.4, -0.1, 0.02])
    assert sample.measured is True
    assert sample.provenance == "odom:body_passthrough"


def test_odom_source_raises_rather_than_zeroing() -> None:
    with pytest.raises(BaseLinVelUnavailableError, match="Refusing to substitute zeros"):
        resolve_base_lin_vel(BASE_LIN_VEL_SOURCE_ODOM, odom_lin_vel_body=None, odom_valid=False)


def test_odom_source_rejects_non_finite_measurements() -> None:
    with pytest.raises(BaseLinVelUnavailableError, match="not finite"):
        resolve_base_lin_vel(
            BASE_LIN_VEL_SOURCE_ODOM,
            odom_lin_vel_body=np.asarray([0.1, float("nan"), 0.0]),
            odom_valid=True,
        )


def test_odom_source_rejects_wrong_shape() -> None:
    with pytest.raises(BaseLinVelUnavailableError, match=r"shape"):
        resolve_base_lin_vel(
            BASE_LIN_VEL_SOURCE_ODOM, odom_lin_vel_body=np.zeros(2), odom_valid=True
        )


def test_zeros_source_is_marked_unmeasured_and_operator_selected() -> None:
    sample = resolve_base_lin_vel(BASE_LIN_VEL_SOURCE_ZEROS)
    assert np.allclose(sample.value, 0.0)
    assert sample.measured is False
    assert sample.provenance == "zeros:operator_selected"


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown base_lin_vel source"):
        resolve_base_lin_vel("imu_integration")


# ---------------------------------------------------------------------------
# Sensor -> policy assembly
# ---------------------------------------------------------------------------


def test_assemble_uses_quaternion_for_projected_gravity() -> None:
    b = _builder()
    quat = (0.0, float(np.sin(0.15)), 0.0, float(np.cos(0.15)))  # 0.3 rad pitch
    obs = assemble_policy_observation(
        b,
        base_lin_vel=np.asarray([0.2, 0.0, 0.0]),
        quat_xyzw=quat,
        base_ang_vel=np.zeros(3),
        velocity_command=np.zeros(3),
        joint_pos=np.ones(12) * 0.5,
        joint_vel=np.zeros(12),
        last_action=None,
    )
    assert obs.shape == (48,)
    assert np.allclose(obs[6:9], projected_gravity_from_quat(*quat))
    assert np.allclose(obs[:3], [0.2, 0.0, 0.0])


def test_assemble_appends_height_scan_padding() -> None:
    b = _builder()
    obs = assemble_policy_observation(
        b,
        base_lin_vel=np.zeros(3),
        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        base_ang_vel=np.zeros(3),
        velocity_command=np.zeros(3),
        joint_pos=np.ones(12) * 0.5,
        joint_vel=np.zeros(12),
        pad_zeros=187,
    )
    assert obs.shape == (235,)
    assert np.allclose(obs[48:], 0.0)
