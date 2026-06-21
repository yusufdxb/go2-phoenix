"""Tests for the stand-fixture geometry helper (CI-safe, no torch)."""

from __future__ import annotations

import math

from phoenix.sim_env.fixture_math import roll_quat_wxyz


def test_zero_roll_is_identity():
    w, x, y, z = roll_quat_wxyz(0.0)
    assert (w, x, y, z) == (1.0, 0.0, 0.0, 0.0)


def test_is_unit_quaternion():
    for roll in (-0.5, -0.16, 0.16, 0.3, 1.2):
        w, x, y, z = roll_quat_wxyz(roll)
        assert math.isclose(w * w + x * x + y * y + z * z, 1.0, rel_tol=1e-12)


def test_roll_is_about_x_only():
    w, x, y, z = roll_quat_wxyz(0.16)
    assert y == 0.0 and z == 0.0
    assert math.isclose(w, math.cos(0.08))
    assert math.isclose(x, math.sin(0.08))


def test_sign_follows_roll_direction():
    assert roll_quat_wxyz(0.2)[1] > 0.0  # +roll -> +x component
    assert roll_quat_wxyz(-0.2)[1] < 0.0
