"""Item 2 regression: capture and restore must be exact coordinate-frame inverses.

``snapshot_manager_state`` used to subtract only the environment origin's z
while ``restore_state`` added the full origin back. With any nonzero origin x
or y, which is every environment but the first in the usual grid layout, a
restored robot landed origin-x/origin-y metres away from the state that was
captured. These tests pin the round trip to the identity for nonzero x, y AND
z, so the asymmetry cannot come back.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from phoenix.replay.state_adapter import (
    DEFAULT_POSITION_FRAME,
    to_stored_position,
    to_world_position,
)
from phoenix.training.episode_outcomes import SNAPSHOT_POSITION_FRAME, snapshot_manager_state

ORIGINS = [
    [0.0, 0.0, 0.0],
    [7.5, -3.25, 0.0],
    [0.0, 0.0, 0.85],
    [12.0, 4.0, -0.6],
    [-2.5, -9.75, 1.5],
]


class _Scene(dict):
    def __init__(self, robot, env_origins):
        super().__init__(robot=robot)
        self.env_origins = np.asarray(env_origins, dtype=float)


def _fake_snapshot_env(positions_world, origins):
    n = len(positions_world)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=np.asarray(positions_world, dtype=float),
            root_lin_vel_b=np.zeros((n, 3)),
            root_ang_vel_b=np.zeros((n, 3)),
            root_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
            joint_vel=np.zeros((n, 12)),
        )
    )
    return SimpleNamespace(
        scene=_Scene(robot, origins),
        command_manager=SimpleNamespace(get_command=lambda name: np.zeros((n, 3))),
        termination_manager=SimpleNamespace(
            terminated=np.zeros(n, dtype=bool),
            active_terms=["base_contact"],
            get_term=lambda name: np.zeros(n, dtype=bool),
        ),
    )


@pytest.mark.parametrize("origin", ORIGINS)
def test_env_local_round_trip_is_the_identity(origin):
    world = np.asarray([1.25, -0.5, 0.34])
    stored = to_stored_position(world, origin, "env_local")
    assert to_world_position(stored, origin, "env_local") == pytest.approx(world)
    # And the stored value really is measured from the origin on all three axes.
    assert stored == pytest.approx(world - np.asarray(origin))


@pytest.mark.parametrize("origin", ORIGINS)
def test_world_frame_round_trip_ignores_the_origin(origin):
    world = np.asarray([3.0, 2.0, 0.4])
    stored = to_stored_position(world, origin, "world")
    assert stored == pytest.approx(world)
    assert to_world_position(stored, origin, "world") == pytest.approx(world)


def test_unknown_or_malformed_frames_are_rejected():
    with pytest.raises(ValueError, match="Unknown position_frame"):
        to_world_position([0, 0, 0], [0, 0, 0], "robot_frame")
    with pytest.raises(ValueError, match="Unknown position_frame"):
        to_stored_position([0, 0, 0], [0, 0, 0], "robot_frame")
    with pytest.raises(ValueError, match="three components"):
        to_world_position([0, 0], [0, 0, 0], "env_local")
    with pytest.raises(ValueError, match="finite"):
        to_world_position([0, 0, float("nan")], [0, 0, 0], "env_local")


def test_snapshot_position_is_env_local_in_all_three_axes():
    origins = [[7.5, -3.25, 0.85], [0.0, 0.0, 0.0]]
    world = [[8.0, -3.0, 1.15], [1.0, 2.0, 0.4]]
    env = _fake_snapshot_env(world, origins)
    state = snapshot_manager_state(env, np.asarray)
    assert SNAPSHOT_POSITION_FRAME == DEFAULT_POSITION_FRAME == "env_local"
    assert state["position"][0] == pytest.approx([0.5, 0.25, 0.3])
    assert state["position"][1] == pytest.approx([1.0, 2.0, 0.4])


@pytest.mark.parametrize("origin", ORIGINS)
def test_capture_then_restore_returns_the_same_world_pose(origin):
    """The composition the reset bridge actually performs, end to end.

    Under the old z-only capture this fails for every origin with a nonzero x
    or y: the stored x/y stayed in world coordinates and the restore added the
    origin to them a second time.
    """
    world = np.asarray([[origin[0] + 0.4, origin[1] - 1.1, origin[2] + 0.33]])
    env = _fake_snapshot_env(world, [origin])
    stored = snapshot_manager_state(env, np.asarray)["position"][0]
    restored = to_world_position(stored, origin, SNAPSHOT_POSITION_FRAME)
    assert restored == pytest.approx(world[0])


def test_restore_state_writes_the_exact_inverse(tmp_path):
    torch = pytest.importorskip("torch")
    from phoenix.replay.state_adapter import restore_state
    from phoenix.replay.trajectory_reader import load_initial_state
    from tests.test_failure_seed_v2 import capsule
    from tests.test_reset_bridge import _fake_env, _FakeRobot

    origin = [7.5, -3.25, 0.85]
    robot = _FakeRobot()
    env, _, target = _fake_env(robot, torch.tensor([origin], dtype=torch.float32), "cpu")
    state = load_initial_state(capsule(tmp_path), 40)
    record = restore_state(target, state, 0)
    written = robot.root_pose_calls[0][0][0][:3].tolist()
    expected = (np.asarray(state.base_pos) + np.asarray(origin)).tolist()
    assert written == pytest.approx(expected)
    assert record["position_frame"] == "env_local"
    assert record["env_origin"] == pytest.approx(origin)
    assert record["restored_position_world"] == pytest.approx(expected)


def test_world_frame_capsule_is_restored_without_the_origin(tmp_path):
    torch = pytest.importorskip("torch")
    from phoenix.replay.state_adapter import restore_state
    from phoenix.replay.trajectory_reader import load_initial_state
    from tests.test_failure_seed_v2 import capsule
    from tests.test_reset_bridge import _fake_env, _FakeRobot

    origin = [7.5, -3.25, 0.85]
    path = capsule(tmp_path, position_frame="world", filename="world_capsule.json")
    robot = _FakeRobot()
    env, _, target = _fake_env(robot, torch.tensor([origin], dtype=torch.float32), "cpu")
    state = load_initial_state(path)
    assert state.position_frame == "world"
    record = restore_state(target, state, 0)
    assert robot.root_pose_calls[0][0][0][:3].tolist() == pytest.approx(state.base_pos.tolist())
    assert record["position_frame"] == "world"


def test_declared_frame_conflict_is_refused_not_reconciled(tmp_path):
    from phoenix.replay.trajectory_reader import load_initial_state
    from tests.test_failure_seed_v2 import capsule

    path = capsule(tmp_path, position_frame="world", filename="world_capsule.json")
    with pytest.raises(ValueError, match="declares position_frame"):
        load_initial_state(path, position_frame="env_local")


def test_undeclared_source_reports_the_assumption(tmp_path):
    from phoenix.replay.trajectory_reader import load_initial_state
    from tests.test_reset_bridge import _write_multi_row_parquet

    path = tmp_path / "legacy.parquet"
    _write_multi_row_parquet(path, 20, 5, 0.5)
    state = load_initial_state(path, 10)
    assert state.position_frame == "env_local"
    assert state.position_frame_source == "phoenix_capture_default"
    declared = load_initial_state(path, 10, position_frame="world")
    assert declared.position_frame == "world"
    assert declared.position_frame_source == "declared_by_caller"
