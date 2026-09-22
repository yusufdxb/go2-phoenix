"""45-D actor observation builder: layout, quaternion convention, fail-closed inputs.

Rotations here are built from axis-angle with Rodrigues' formula, independently
of the quaternion code under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phoenix.velocity import contract as c
from phoenix.velocity.observation import (
    ObservationError,
    actions_to_joint_targets,
    build_actor_observation,
    projected_gravity_wxyz,
    quat_xyzw_to_wxyz,
    tilt_from_projected_gravity,
)

DEFAULT_Q = np.array([c.DEFAULT_JOINT_POS[j] for j in c.JOINT_ORDER])


def _quat(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.r_[math.cos(angle / 2), math.sin(angle / 2) * axis]


def _rodrigues(axis, angle) -> np.ndarray:
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


@pytest.mark.parametrize(
    "axis,deg",
    [
        ((1, 0, 0), 0),
        ((1, 0, 0), 10),
        ((1, 0, 0), -10),
        ((0, 1, 0), 10),
        ((0, 1, 0), -10),
        ((1, 0, 0), 90),
        ((0, 1, 0), 90),
        ((1, 0, 0), 180),
        ((0, 0, 1), 73),
        ((1, 1, 0), 37),
    ],
)
def test_projected_gravity_matches_independent_rotation(axis, deg) -> None:
    ang = math.radians(deg)
    expected = _rodrigues(axis, ang).T @ np.array([0, 0, -1.0])
    got = projected_gravity_wxyz(_quat(axis, ang))
    assert np.allclose(got, expected, atol=1e-12)
    # q and -q are the same rotation.
    assert np.allclose(projected_gravity_wxyz(-_quat(axis, ang)), expected, atol=1e-12)


def test_known_tilts() -> None:
    assert tilt_from_projected_gravity(projected_gravity_wxyz([1, 0, 0, 0])) == pytest.approx(0)
    for axis in ((1, 0, 0), (0, 1, 0)):
        for deg in (10, -10, 45, 90):
            g = projected_gravity_wxyz(_quat(axis, math.radians(deg)))
            assert tilt_from_projected_gravity(g) == pytest.approx(abs(math.radians(deg)))
    g = projected_gravity_wxyz(_quat((1, 0, 0), math.pi))
    assert tilt_from_projected_gravity(g) == pytest.approx(math.pi)
    # Pure yaw is not tilt.
    g = projected_gravity_wxyz(_quat((0, 0, 1), 2.0))
    assert tilt_from_projected_gravity(g) == pytest.approx(0, abs=1e-9)


def test_positive_roll_moves_gravity_to_negative_body_y() -> None:
    # Rolling right side down (+x rotation): gravity appears along -y... check sign
    # against the independent matrix so a flipped convention cannot pass.
    g = projected_gravity_wxyz(_quat((1, 0, 0), math.radians(10)))
    expected = _rodrigues((1, 0, 0), math.radians(10)).T @ np.array([0, 0, -1.0])
    assert np.sign(g[1]) == np.sign(expected[1]) != 0


def test_xyzw_conversion_is_explicit() -> None:
    q = _quat((0, 1, 0), 0.3)
    xyzw = np.r_[q[1:], q[0]]
    assert np.allclose(quat_xyzw_to_wxyz(xyzw), q)
    # Feeding xyzw data as wxyz gives a DIFFERENT (wrong) answer, not silently the same.
    assert not np.allclose(projected_gravity_wxyz(xyzw), projected_gravity_wxyz(q))


def _inputs(**over):
    base = dict(
        gyro_body=[0.1, -0.2, 0.3],
        quat_wxyz=[1, 0, 0, 0],
        command=[0.5, 0.0, -0.4],
        joint_pos=DEFAULT_Q + 0.01 * np.arange(12),
        joint_vel=np.arange(12) * -0.1,
        last_action=np.linspace(-1, 1, 12),
    )
    base.update(over)
    return base


def test_layout_matches_contract_slices() -> None:
    kw = _inputs()
    obs = build_actor_observation(**kw)
    assert obs.shape == (45,) and obs.dtype == np.float32
    s = c.obs_slices()
    assert np.allclose(obs[s["base_ang_vel"]], kw["gyro_body"])
    assert np.allclose(obs[s["projected_gravity"]], [0, 0, -1])
    assert np.allclose(obs[s["velocity_command"]], kw["command"])
    assert np.allclose(obs[s["joint_pos_rel"]], 0.01 * np.arange(12), atol=1e-6)
    assert np.allclose(obs[s["joint_vel"]], kw["joint_vel"])
    assert np.allclose(obs[s["last_action"]], kw["last_action"])


@pytest.mark.parametrize(
    "over",
    [
        {"gyro_body": [0, 0]},
        {"gyro_body": [0, float("nan"), 0]},
        {"quat_wxyz": [0, 0, 0, 0]},
        {"quat_wxyz": [2, 0, 0, 0]},
        {"quat_wxyz": [1, 0, 0, float("inf")]},
        {"command": [0, 0, 0, 0]},
        {"joint_pos": np.zeros(11)},
        {"joint_vel": np.full(12, np.nan)},
        {"last_action": np.r_[np.zeros(11), np.inf]},
    ],
)
def test_bad_inputs_fail_closed(over) -> None:
    with pytest.raises(ObservationError):
        build_actor_observation(**_inputs(**over))


def test_action_mapping_is_default_plus_scaled_action() -> None:
    a = np.linspace(-1, 1, 12)
    assert np.allclose(actions_to_joint_targets(a, 0.25), DEFAULT_Q + 0.25 * a)
    assert np.allclose(actions_to_joint_targets(np.zeros(12), 0.25), DEFAULT_Q)
    with pytest.raises(ObservationError):
        actions_to_joint_targets(np.full(12, np.nan), 0.25)
