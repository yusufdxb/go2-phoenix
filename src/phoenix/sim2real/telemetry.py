"""Pure (no-rclpy) helpers for turning raw GO2 telemetry into log-ready arrays.

Kept ROS-free, unlike :mod:`phoenix.sim2real.ros2_policy_node` and
:mod:`phoenix.sim2real.lowstate_bridge_node`, so this logic is unit-testable
in the no-ros CI suite. Nothing here touches control or safety; it exists
only to feed :class:`phoenix.real_world.trajectory_logger.TrajectoryStep`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def foot_force_to_array(raw: Sequence[int]) -> np.ndarray:
    """Convert ``unitree_go/msg/LowState.foot_force`` (``int16[4]``) to float32.

    UNITS UNVERIFIED: the public Unitree SDK does not document a calibrated
    Newton conversion for this field on the GO2 (unlike, say,
    ``imu_state.accelerometer``, which is documented as m/s^2). This function
    performs a lossless numeric cast only, no scaling factor is applied or
    assumed. Treat the result as raw per-foot sensor counts, NOT calibrated
    Newtons, until someone verifies the conversion against a known load on
    real hardware.

    Order is assumed FR, FL, RR, RL, matching the motor ordering already used
    by ``lowstate_bridge_node.MOTOR_NAMES`` for ``motor_state``. This ordering
    is NOT independently confirmed for ``foot_force`` specifically (the .msg
    file gives no per-index documentation), it is inferred from the rest of
    the message following the same per-leg grouping. Flag as ASSUMPTION.
    """
    arr = np.asarray(raw, dtype=np.float32)
    if arr.shape != (4,):
        raise ValueError(f"expected 4 foot_force values (FR,FL,RR,RL), got shape {arr.shape}")
    return arr


def rotate_world_to_body(
    v_world: np.ndarray, quat_xyzw: tuple[float, float, float, float]
) -> np.ndarray:
    """Rotate a vector from the world/odom frame into the body frame.

    ``quat_xyzw`` is the body orientation in the world frame (x, y, z, w),
    the same convention used throughout ``ros2_policy_node`` for the IMU
    orientation.

    Why this rotation exists: ``nav_msgs/Odometry.twist.twist.linear`` is,
    per REP 105, expressed in the frame named by ``child_frame_id``. Field
    notes for this robot record ``child_frame_id = "base_link"`` for
    ``/utlidar/robot_odom`` (docs/go2_field_notes.md), which if honored by
    the publisher would already make the twist body-frame, no rotation
    needed. That has NOT been independently confirmed for this specific
    publisher (some odometry sources report ``child_frame_id`` correctly
    while still publishing twist in the parent/world frame, in violation of
    REP 105). To avoid silently trusting an unverified frame claim, this
    function rotates explicitly using the IMU-derived orientation rather
    than passing the twist through as-is.

    HARDWARE-UNVERIFIED: this is the single riskiest assumption in the
    odometry telemetry path. If ``/utlidar/robot_odom`` actually already
    publishes body-frame twist, this rotation is redundant when the robot is
    level (identity-ish quaternion) but wrong whenever the robot is pitched
    or rolled. Confirm against a real bag (compare raw twist to IMU-derived
    heading over a straight walk) before trusting ``base_lin_vel_body`` from
    real captures for training.
    """
    x, y, z, w = quat_xyzw
    # Standard unit-quaternion -> rotation matrix (body -> world).
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - w * x)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    body_to_world = np.array(
        [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], dtype=np.float64
    )
    # Rotation matrices are orthonormal, so world->body is the transpose.
    v_body = body_to_world.T @ np.asarray(v_world, dtype=np.float64)
    return v_body.astype(np.float32)
