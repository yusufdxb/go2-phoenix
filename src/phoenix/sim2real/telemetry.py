"""Pure (no-rclpy) helpers for turning raw GO2 telemetry into log-ready arrays.

Kept ROS-free, unlike :mod:`phoenix.sim2real.ros2_policy_node` and
:mod:`phoenix.sim2real.lowstate_bridge_node`, so this logic is unit-testable
in the no-ros CI suite.

Two of these helpers now feed the control path as well as the log: when
``observation.base_lin_vel_source: odom`` is selected,
:func:`odom_twist_to_body` produces the ``base_lin_vel`` observation term. It
is written to fail closed (return ``valid=False``) rather than guess.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: Units of ``contact_forces`` for a capture written from the real robot.
#: ``unitree_go/msg/LowState.foot_force`` is ``int16[4]`` with no documented
#: calibration; see :func:`foot_force_to_array`.
CONTACT_FORCE_UNITS_RAW_COUNTS = "raw_counts_uncalibrated"
#: Units of ``contact_forces`` for a capture written from Isaac Lab's contact
#: sensor, which reports true Newtons.
CONTACT_FORCE_UNITS_NEWTONS = "newtons"

#: The ROS frame id the GO2's body is published under. Measured on hardware:
#: ``/utlidar/robot_odom`` has ``header.frame_id = "odom"`` and
#: ``child_frame_id = "base_link"`` (docs/go2_field_notes.md section 3).
DEFAULT_BASE_FRAME_ID = "base_link"
#: Frame ids we accept as "the parent/world frame of an odometry message".
WORLD_FRAME_IDS: tuple[str, ...] = ("odom", "map", "world")


def foot_force_to_array(raw: Sequence[int]) -> np.ndarray:
    """Convert ``unitree_go/msg/LowState.foot_force`` (``int16[4]``) to float32.

    UNITS UNVERIFIED: the public Unitree SDK does not document a calibrated
    Newton conversion for this field on the GO2 (unlike, say,
    ``imu_state.accelerometer``, which is documented as m/s^2). This function
    performs a lossless numeric cast only, no scaling factor is applied or
    assumed. Treat the result as raw per-foot sensor counts, NOT calibrated
    Newtons, until someone verifies the conversion against a known load on
    real hardware. Captures record this explicitly in the
    ``contact_forces_units`` column (:data:`CONTACT_FORCE_UNITS_RAW_COUNTS`),
    so a consumer never has to infer it from the file name.

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

    This is a pure rotation primitive. Whether an odometry twist needs it at
    all is decided by :func:`odom_twist_to_body` from the message's own
    ``child_frame_id``, NOT assumed here.
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


@dataclass(frozen=True)
class OdomTwistResult:
    """Outcome of resolving an odometry twist into the body frame."""

    #: (3,) float32 body-frame linear velocity. Zeros when ``valid`` is False;
    #: callers must check ``valid`` and must not treat those zeros as a
    #: measurement.
    lin_vel_body: np.ndarray
    #: How the value was obtained, recorded per-row in the trajectory parquet.
    provenance: str
    valid: bool


def odom_twist_to_body(
    *,
    twist_linear: Sequence[float],
    child_frame_id: str,
    header_frame_id: str,
    quat_xyzw: tuple[float, float, float, float],
    base_frame_id: str = DEFAULT_BASE_FRAME_ID,
) -> OdomTwistResult:
    """Resolve ``nav_msgs/Odometry.twist.twist.linear`` into the body frame.

    THE CONTRACT, and why this reads ``child_frame_id`` instead of assuming:

    ``nav_msgs/Odometry`` documents ``pose`` in ``header.frame_id`` and
    ``twist`` in ``child_frame_id`` (REP 105 uses the same split). On this
    robot, ``/utlidar/robot_odom`` was measured publishing
    ``header.frame_id = "odom"`` with ``child_frame_id = "base_link"``
    (docs/go2_field_notes.md section 3). Under the standard reading that
    twist is ALREADY body-frame, and rotating it world-to-body is wrong:
    it would be a no-op only while the robot is level and would corrupt the
    value under any pitch or roll. An earlier revision of this module did
    exactly that, unconditionally.

    So the decision is made from the message itself:

    * ``child_frame_id == base_frame_id``: pass the twist through unchanged.
    * ``child_frame_id`` is the parent/world frame (equal to
      ``header.frame_id`` or one of :data:`WORLD_FRAME_IDS`): rotate
      world-to-body with the IMU orientation. Publishers that do this are out
      of spec, but they exist.
    * anything else, including an empty ``child_frame_id``: return
      ``valid=False``. We do not guess a frame.

    HARDWARE-UNVERIFIED: the frame *labels* were measured, the *contents*
    were not. Nobody has yet compared a real ``/utlidar/robot_odom`` twist
    against IMU-derived heading over a straight walk to confirm the publisher
    honours its own ``child_frame_id``. Do that before trusting
    ``base_lin_vel_body`` from real captures for training, and before running
    the policy with ``base_lin_vel_source: odom``.
    """
    child = _normalize_frame(child_frame_id)
    parent = _normalize_frame(header_frame_id)
    base = _normalize_frame(base_frame_id)

    v = np.asarray(twist_linear, dtype=np.float64).reshape(-1)
    if v.shape != (3,):
        return OdomTwistResult(
            lin_vel_body=np.zeros(3, dtype=np.float32),
            provenance=f"invalid_twist_shape:{v.shape[0]}",
            valid=False,
        )
    if not np.isfinite(v).all():
        return OdomTwistResult(
            lin_vel_body=np.zeros(3, dtype=np.float32),
            provenance="non_finite_twist",
            valid=False,
        )

    if child and child == base:
        return OdomTwistResult(
            lin_vel_body=v.astype(np.float32),
            provenance="body_passthrough",
            valid=True,
        )
    if child and (child == parent or child in WORLD_FRAME_IDS):
        return OdomTwistResult(
            lin_vel_body=rotate_world_to_body(v, quat_xyzw),
            provenance="rotated_from_world",
            valid=True,
        )
    return OdomTwistResult(
        lin_vel_body=np.zeros(3, dtype=np.float32),
        provenance=f"unrecognized_child_frame:{child or '<empty>'}",
        valid=False,
    )


def _normalize_frame(frame_id: str | None) -> str:
    """Strip TF's optional leading slash and surrounding whitespace."""
    if not frame_id:
        return ""
    return str(frame_id).strip().lstrip("/")


@dataclass(frozen=True)
class OdomSample:
    """One odometry reading, resolved and labelled, or an explicit absence."""

    #: True when a message existed and was inside the freshness window.
    fresh: bool
    #: (3,) RAW ``pose.pose.position``, exactly as published. Boot-pose
    #: relative, so ``position[2]`` is NOT height above the floor. Zeros when
    #: ``fresh`` is False.
    position: np.ndarray
    #: (3,) body-frame linear velocity. Zeros unless ``twist_valid``.
    lin_vel_body: np.ndarray
    #: True only when the twist could be resolved into the body frame.
    twist_valid: bool
    #: Recorded per-row in the trajectory parquet. One of the
    #: :class:`OdomTwistResult` provenances, or ``absent`` / ``stale``.
    provenance: str


def sample_odom(
    msg,
    *,
    fresh: bool,
    quat_xyzw: tuple[float, float, float, float],
    base_frame_id: str = DEFAULT_BASE_FRAME_ID,
) -> OdomSample:
    """Extract a labelled :class:`OdomSample` from a ``nav_msgs/Odometry``.

    ``msg`` is duck-typed (``pose.pose.position``, ``twist.twist.linear``,
    ``child_frame_id``, ``header.frame_id``) so this stays importable and
    testable without rclpy.

    Deriving nothing is the point: the raw position is passed through
    untouched and no height, ground clearance, or displacement is computed
    from it here.
    """
    if msg is None or not fresh:
        return OdomSample(
            fresh=False,
            position=np.zeros(3, dtype=np.float32),
            lin_vel_body=np.zeros(3, dtype=np.float32),
            twist_valid=False,
            provenance="absent" if msg is None else "stale",
        )

    p = msg.pose.pose.position
    position = np.asarray([p.x, p.y, p.z], dtype=np.float32)
    lv = msg.twist.twist.linear
    header = getattr(msg, "header", None)
    twist = odom_twist_to_body(
        twist_linear=(lv.x, lv.y, lv.z),
        child_frame_id=getattr(msg, "child_frame_id", ""),
        header_frame_id=getattr(header, "frame_id", "") if header is not None else "",
        quat_xyzw=quat_xyzw,
        base_frame_id=base_frame_id,
    )
    return OdomSample(
        fresh=True,
        position=position,
        lin_vel_body=twist.lin_vel_body,
        twist_valid=twist.valid,
        provenance=twist.provenance,
    )
