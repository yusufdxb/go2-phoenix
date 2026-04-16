"""Tests for the per-env variation translation that feeds reconstruct.py.

The Isaac Sim plumbing in ``reconstruct.py`` itself is hardware-only —
these tests cover the pure-numpy translation that decides what each
sim env actually starts from.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.replay.apply_variations import (
    _quat_xyzw_to_wxyz,
    build_per_env_initial_conditions,
)
from phoenix.replay.trajectory_reader import InitialState
from phoenix.replay.variations import VariationSample


def _initial(quat_xyzw=(0.0, 0.0, 0.0, 1.0)) -> InitialState:
    return InitialState(
        base_pos=np.asarray([1.0, 2.0, 0.4], dtype=np.float32),
        base_quat=np.asarray(quat_xyzw, dtype=np.float32),
        base_lin_vel_body=np.asarray([0.3, 0.0, 0.0], dtype=np.float32),
        base_ang_vel_body=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        joint_pos=np.linspace(-0.5, 0.5, 12, dtype=np.float32),
        joint_vel=np.zeros(12, dtype=np.float32),
        command_vel=np.asarray([0.5, 0.0, 0.0], dtype=np.float32),
    )


def _v(friction=0.0, mass=0.0, push=0.0, yaw=0.0) -> VariationSample:
    return VariationSample(
        friction_delta=friction,
        mass_delta_kg=mass,
        push_velocity_delta=push,
        push_yaw_delta=yaw,
    )


# ---------------- quaternion convention ------------------------------------


def test_quat_xyzw_to_wxyz_rolls_last_axis() -> None:
    out = _quat_xyzw_to_wxyz(np.asarray([0.1, 0.2, 0.3, 0.9]))
    np.testing.assert_allclose(out, [0.9, 0.1, 0.2, 0.3])


def test_quat_xyzw_to_wxyz_works_on_batches() -> None:
    batch = np.asarray([[0.1, 0.2, 0.3, 0.9], [0.0, 0.0, 0.0, 1.0]])
    out = _quat_xyzw_to_wxyz(batch)
    np.testing.assert_allclose(out, [[0.9, 0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0]])


def test_quat_xyzw_to_wxyz_rejects_wrong_dim() -> None:
    with pytest.raises(ValueError, match="last dim must be 4"):
        _quat_xyzw_to_wxyz(np.asarray([0.0, 0.0, 0.0]))


# ---------------- build_per_env_initial_conditions -------------------------


def test_build_per_env_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_per_env_initial_conditions(_initial(), [])


def test_build_per_env_returns_one_row_per_variation() -> None:
    initial = _initial()
    variations = [_v(push=p) for p in (-0.1, 0.0, 0.2, 0.5)]
    out = build_per_env_initial_conditions(initial, variations)
    assert out["base_pos"].shape == (4, 3)
    assert out["base_quat_wxyz"].shape == (4, 4)
    assert out["base_lin_vel"].shape == (4, 3)
    assert out["base_ang_vel"].shape == (4, 3)
    assert out["joint_pos"].shape == (4, 12)
    assert out["joint_vel"].shape == (4, 12)
    assert out["base_mass_delta_kg"].shape == (4,)
    assert out["friction_scale"].shape == (4,)


def test_push_velocity_delta_is_added_to_x_only() -> None:
    initial = _initial()
    deltas = [-0.4, 0.0, 0.6]
    out = build_per_env_initial_conditions(initial, [_v(push=d) for d in deltas])
    np.testing.assert_allclose(out["base_lin_vel"][:, 0], [0.3 - 0.4, 0.3 + 0.0, 0.3 + 0.6])
    np.testing.assert_allclose(out["base_lin_vel"][:, 1], 0.0)
    np.testing.assert_allclose(out["base_lin_vel"][:, 2], 0.0)


def test_push_yaw_delta_is_added_to_z_only() -> None:
    initial = _initial()
    deltas = [-0.3, 0.0, 0.3]
    out = build_per_env_initial_conditions(initial, [_v(yaw=d) for d in deltas])
    np.testing.assert_allclose(out["base_ang_vel"][:, 0], 0.0)
    np.testing.assert_allclose(out["base_ang_vel"][:, 1], 0.0)
    np.testing.assert_allclose(out["base_ang_vel"][:, 2], deltas)


def test_mass_delta_is_passed_through_unchanged() -> None:
    initial = _initial()
    deltas = [-1.0, 0.0, 0.5, 1.0]
    out = build_per_env_initial_conditions(initial, [_v(mass=d) for d in deltas])
    np.testing.assert_allclose(out["base_mass_delta_kg"], deltas)


def test_friction_delta_becomes_clipped_multiplicative_scale() -> None:
    initial = _initial()
    # -2.0 would push the multiplier negative; clamp protects callers
    # that multiply scene friction directly.
    deltas = [-2.0, -0.5, 0.0, 0.3]
    out = build_per_env_initial_conditions(initial, [_v(friction=d) for d in deltas])
    np.testing.assert_allclose(out["friction_scale"], [0.05, 0.5, 1.0, 1.3])


def test_quaternion_is_converted_xyzw_to_wxyz_per_env() -> None:
    initial = _initial(quat_xyzw=(0.1, 0.2, 0.3, 0.9))
    out = build_per_env_initial_conditions(initial, [_v(), _v(push=0.1)])
    expected = np.asarray([[0.9, 0.1, 0.2, 0.3], [0.9, 0.1, 0.2, 0.3]], dtype=np.float32)
    np.testing.assert_allclose(out["base_quat_wxyz"], expected)


def test_initial_state_is_not_mutated() -> None:
    initial = _initial()
    snapshot_lin = initial.base_lin_vel_body.copy()
    snapshot_ang = initial.base_ang_vel_body.copy()
    snapshot_pos = initial.base_pos.copy()
    build_per_env_initial_conditions(initial, [_v(push=0.7, yaw=0.5)])
    np.testing.assert_allclose(initial.base_lin_vel_body, snapshot_lin)
    np.testing.assert_allclose(initial.base_ang_vel_body, snapshot_ang)
    np.testing.assert_allclose(initial.base_pos, snapshot_pos)


def test_zero_variations_recover_logged_state() -> None:
    initial = _initial()
    out = build_per_env_initial_conditions(initial, [_v()])
    np.testing.assert_allclose(out["base_lin_vel"][0], initial.base_lin_vel_body)
    np.testing.assert_allclose(out["base_ang_vel"][0], initial.base_ang_vel_body)
    np.testing.assert_allclose(out["joint_pos"][0], initial.joint_pos)
    np.testing.assert_allclose(out["base_mass_delta_kg"], [0.0])
    np.testing.assert_allclose(out["friction_scale"], [1.0])
