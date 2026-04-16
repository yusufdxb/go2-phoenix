"""Cross-path consistency tests for the projected-gravity helper.

The audit caught that ``ros2_policy_node._projected_gravity_from_quat``
had ``gx, gy`` sign-flipped vs ``verify_deploy._projected_gravity_from_quat_xyzw``.
That meant the deployed policy saw mirror-image gravity in its obs
vector. These tests now lock the two helpers to:

1. Each other (byte-for-byte consistency along the parity gate).
2. The closed-form Isaac Lab convention ``R(q)^T @ (0, 0, -1)`` for a
   handful of analytically known quaternions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phoenix.sim2real.ros2_policy_node import _projected_gravity_from_quat
from phoenix.sim2real.verify_deploy import _projected_gravity_from_quat_xyzw


def _q_pitch(theta: float) -> tuple[float, float, float, float]:
    """Quaternion (x,y,z,w) for a pure pitch rotation around body y-axis."""
    return (0.0, math.sin(theta / 2), 0.0, math.cos(theta / 2))


def _q_roll(theta: float) -> tuple[float, float, float, float]:
    return (math.sin(theta / 2), 0.0, 0.0, math.cos(theta / 2))


def _q_yaw(theta: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(theta / 2), math.cos(theta / 2))


# ---------------- closed-form values ---------------------------------------


def test_identity_quat_returns_world_minus_z() -> None:
    g = _projected_gravity_from_quat(0.0, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(g, [0.0, 0.0, -1.0], atol=1e-7)


def test_pitch_forward_tilts_gravity_into_x() -> None:
    # Body pitched +0.3 rad (nose up) ⇒ gravity in body frame has positive
    # gx of magnitude sin(0.3); verifies the sign convention in BOTH
    # helpers (the bug had this negative).
    theta = 0.3
    x, y, z, w = _q_pitch(theta)
    g = _projected_gravity_from_quat(x, y, z, w)
    np.testing.assert_allclose(g, [math.sin(theta), 0.0, -math.cos(theta)], atol=1e-6)


def test_roll_right_tilts_gravity_into_y() -> None:
    # Body rolled +0.4 rad ⇒ projected gravity should have negative gy
    # of magnitude sin(0.4) (gravity falls toward the lowered side).
    theta = 0.4
    x, y, z, w = _q_roll(theta)
    g = _projected_gravity_from_quat(x, y, z, w)
    np.testing.assert_allclose(g, [0.0, -math.sin(theta), -math.cos(theta)], atol=1e-6)


def test_yaw_does_not_change_projected_gravity() -> None:
    # A pure yaw rotation leaves projected_gravity at (0, 0, -1) because
    # the body z-axis stays aligned with world z.
    for theta in (0.1, -0.7, 1.5):
        x, y, z, w = _q_yaw(theta)
        g = _projected_gravity_from_quat(x, y, z, w)
        np.testing.assert_allclose(g, [0.0, 0.0, -1.0], atol=1e-6)


# ---------------- cross-helper consistency (parity-gate sentinel) ---------


@pytest.mark.parametrize(
    "quat_xyzw",
    [
        (0.0, 0.0, 0.0, 1.0),
        _q_pitch(0.3),
        _q_pitch(-0.5),
        _q_roll(0.2),
        _q_roll(-0.4),
        _q_yaw(1.0),
        # Combined small rotation
        (0.05, -0.05, 0.05, math.sqrt(1 - 0.05 * 0.05 * 3)),
    ],
)
def test_two_helpers_byte_match(quat_xyzw) -> None:
    # Both helpers compute the same expression but with the leading sign
    # reordered (`-2 * x` vs `2 * x * -1`) — float32 rounding order can
    # differ by 1 ULP. We tolerate that, but no more.
    g_node = _projected_gravity_from_quat(*quat_xyzw)
    g_verify = _projected_gravity_from_quat_xyzw(np.asarray(quat_xyzw, dtype=np.float32))
    np.testing.assert_allclose(g_node, g_verify, atol=1e-7)
