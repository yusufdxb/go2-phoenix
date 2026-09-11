"""Unit tests for phoenix.sim2real.telemetry (pure, no-rclpy helpers)."""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real.telemetry import foot_force_to_array, rotate_world_to_body


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
