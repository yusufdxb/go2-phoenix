"""Unit tests for phoenix.sim2real.telemetry (pure, no-rclpy helpers)."""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real.telemetry import (
    foot_force_to_array,
    odom_twist_to_body,
    rotate_world_to_body,
    sample_odom,
)


def test_foot_force_to_array_passthrough_no_scaling() -> None:
    raw = [100, -50, 0, 32767]
    out = foot_force_to_array(raw)
    assert out.dtype == np.float32
    assert np.allclose(out, [100.0, -50.0, 0.0, 32767.0])


def test_foot_force_to_array_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        foot_force_to_array([1, 2, 3])
    with pytest.raises(ValueError):
        foot_force_to_array([1, 2, 3, 4, 5])


def test_rotate_world_to_body_identity_quat_is_passthrough() -> None:
    v = np.asarray([0.3, -0.2, 0.1])
    out = rotate_world_to_body(v, (0.0, 0.0, 0.0, 1.0))
    assert np.allclose(out, v, atol=1e-6)


def test_rotate_world_to_body_90deg_yaw() -> None:
    # +90deg yaw about z: x=0, y=0, z=sin(45deg), w=cos(45deg).
    half = np.pi / 4.0
    quat = (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))
    # World +x expressed in a body frame yawed +90deg is body -y.
    # (Hand-derived from the body->world rotation matrix for this quat;
    # see phoenix.sim2real.telemetry.rotate_world_to_body docstring.)
    out = rotate_world_to_body(np.asarray([1.0, 0.0, 0.0]), quat)
    assert np.allclose(out, [0.0, -1.0, 0.0], atol=1e-6)


def test_rotate_world_to_body_roundtrip_with_inverse_quat() -> None:
    """Rotating by q then by q's inverse must return the original vector."""
    x, y, z, w = 0.1826, 0.3651, 0.5477, 0.7303  # arbitrary, will be normalized
    quat = np.asarray([x, y, z, w])
    quat = quat / np.linalg.norm(quat)
    inv_quat = (-quat[0], -quat[1], -quat[2], quat[3])

    v = np.asarray([1.0, -2.0, 0.5])
    rotated = rotate_world_to_body(v, tuple(quat))
    back = rotate_world_to_body(rotated, inv_quat)
    assert np.allclose(back, v, atol=1e-5)


# ---------------------------------------------------------------------------
# Odometry twist frame contract.
#
# nav_msgs/Odometry documents twist in child_frame_id (REP 105 splits pose and
# twist the same way). The GO2's /utlidar/robot_odom was MEASURED publishing
# header.frame_id="odom", child_frame_id="base_link", so its twist is already
# body-frame and must NOT be rotated. An earlier revision rotated it
# unconditionally, which is a no-op only while the robot is level.
# ---------------------------------------------------------------------------

_YAW_90 = (0.0, 0.0, float(np.sin(np.pi / 4)), float(np.cos(np.pi / 4)))


def test_base_link_child_frame_is_passed_through_unrotated() -> None:
    out = odom_twist_to_body(
        twist_linear=(1.0, 0.0, 0.0),
        child_frame_id="base_link",
        header_frame_id="odom",
        quat_xyzw=_YAW_90,
    )
    assert out.valid is True
    assert out.provenance == "body_passthrough"
    # A spurious rotation under 90 deg yaw would give (0, -1, 0).
    assert np.allclose(out.lin_vel_body, [1.0, 0.0, 0.0], atol=1e-6)


def test_leading_slash_in_frame_id_is_tolerated() -> None:
    out = odom_twist_to_body(
        twist_linear=(1.0, 0.0, 0.0),
        child_frame_id="/base_link",
        header_frame_id="/odom",
        quat_xyzw=_YAW_90,
    )
    assert out.valid is True and out.provenance == "body_passthrough"


def test_world_child_frame_is_rotated_into_body() -> None:
    out = odom_twist_to_body(
        twist_linear=(1.0, 0.0, 0.0),
        child_frame_id="odom",
        header_frame_id="odom",
        quat_xyzw=_YAW_90,
    )
    assert out.valid is True
    assert out.provenance == "rotated_from_world"
    assert np.allclose(out.lin_vel_body, [0.0, -1.0, 0.0], atol=1e-6)


@pytest.mark.parametrize("child", ["", "mystery_link", "camera_link"])
def test_unrecognized_child_frame_is_invalid_not_guessed(child) -> None:
    out = odom_twist_to_body(
        twist_linear=(1.0, 0.0, 0.0),
        child_frame_id=child,
        header_frame_id="odom",
        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    assert out.valid is False
    assert out.provenance.startswith("unrecognized_child_frame")
    assert np.allclose(out.lin_vel_body, 0.0)


def test_non_finite_twist_is_invalid() -> None:
    out = odom_twist_to_body(
        twist_linear=(float("nan"), 0.0, 0.0),
        child_frame_id="base_link",
        header_frame_id="odom",
        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    assert out.valid is False
    assert out.provenance == "non_finite_twist"


def _msg(position=(1.0, 2.0, 0.4), linear=(0.5, 0.0, 0.0), child="base_link", frame="odom"):
    import types

    return types.SimpleNamespace(
        header=types.SimpleNamespace(frame_id=frame),
        child_frame_id=child,
        pose=types.SimpleNamespace(
            pose=types.SimpleNamespace(
                position=types.SimpleNamespace(x=position[0], y=position[1], z=position[2])
            )
        ),
        twist=types.SimpleNamespace(
            twist=types.SimpleNamespace(
                linear=types.SimpleNamespace(x=linear[0], y=linear[1], z=linear[2])
            )
        ),
    )


def test_sample_odom_preserves_the_raw_position() -> None:
    sample = sample_odom(_msg(), fresh=True, quat_xyzw=_YAW_90)
    # Raw, underived: position[2] is boot-pose relative, not a height, and
    # nothing in this path is allowed to turn it into one.
    assert np.allclose(sample.position, [1.0, 2.0, 0.4])
    assert sample.fresh is True and sample.twist_valid is True


def test_sample_odom_absent_and_stale_are_distinguishable() -> None:
    absent = sample_odom(None, fresh=False, quat_xyzw=(0.0, 0.0, 0.0, 1.0))
    stale = sample_odom(_msg(), fresh=False, quat_xyzw=(0.0, 0.0, 0.0, 1.0))
    assert absent.provenance == "absent"
    assert stale.provenance == "stale"
    for s in (absent, stale):
        assert s.fresh is False and s.twist_valid is False
        assert np.allclose(s.position, 0.0)
        assert np.allclose(s.lin_vel_body, 0.0)


def test_sample_odom_missing_child_frame_id_is_invalid() -> None:
    import types

    msg = _msg()
    del msg.child_frame_id
    msg = types.SimpleNamespace(**vars(msg))
    sample = sample_odom(msg, fresh=True, quat_xyzw=(0.0, 0.0, 0.0, 1.0))
    assert sample.fresh is True
    assert sample.twist_valid is False
