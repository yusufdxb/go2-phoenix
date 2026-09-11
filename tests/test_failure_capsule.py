"""The capsule writer is the single validated producer of schema 1.1.

Round trips through the reader that the reset bridge actually uses, so the
format cannot drift between what is written and what is seeded.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from phoenix.adaptation.reset_bridge import resolve_seed
from phoenix.replay.failure_capsule import CAPSULE_SCHEMA_VERSION, write_failure_capsule
from phoenix.replay.trajectory_reader import TrajectoryReader, load_initial_state


def _frames(n=30, action=True, target=True):
    frames = []
    for i in range(n):
        frame = {
            "base_pos": [0.01 * i, -0.02 * i, 0.30],
            "base_quat": [0.0, 0.0, 0.0, 1.0],
            "base_lin_vel_body": [0.5, 0.0, 0.0],
            "base_ang_vel_body": [0.0, 0.0, 0.01 * i],
            "joint_pos": [0.1] * 12,
            "joint_vel": [0.0] * 12,
            "command_vel": [0.5, 0.0, 0.0],
        }
        if action:
            frame["action"] = [0.01 * i] * 12
        if target:
            frame["joint_target"] = [0.2 + 0.01 * i] * 12
        frames.append(frame)
    return frames


def _write(tmp_path, name="capsule.json", **kwargs):
    params = dict(
        capsule_id="cap-1",
        failure_onset_index=20,
        control_dt=0.02,
        episode_start_index=0,
        environment_parameters={"static_friction": 0.3},
        disturbances=[{"kind": "push", "vx": 1.0}],
        failure_mode="slip",
        severity="high",
        scenario_id="sc-1",
    )
    params.update(kwargs)
    frames = params.pop("frames", _frames())
    return write_failure_capsule(tmp_path / name, frames, **params)


def test_round_trip_through_the_reader_the_bridge_uses(tmp_path):
    path = _write(tmp_path)
    reader = TrajectoryReader(path)
    assert len(reader) == 30
    assert reader.metadata["schema_version"] == CAPSULE_SCHEMA_VERSION
    assert reader.declared_position_frame == "env_local"
    assert reader.environment_parameters == {"static_friction": 0.3}
    assert reader.disturbances == [{"kind": "push", "vx": 1.0}]
    assert reader.failure_indices().tolist() == [20]

    record = resolve_seed(path, "failure_onset_minus_steps", 5)
    assert record["resolved_row"] == 15
    assert record["position_frame_source"] == "declared_by_source"
    assert record["time_before_onset_seconds"] == pytest.approx(0.1)

    state = load_initial_state(path, record["resolved_row"], history_rows=-1)
    assert state.position_frame == "env_local"
    assert state.environment_parameters == {"static_friction": 0.3}
    assert len(state.controller_history) == 16
    assert state.controller_history.starts_at_episode_start is True
    assert state.controller_history.last_action == pytest.approx([0.15] * 12)
    assert state.controller_history.joint_targets[-1] == pytest.approx([0.35] * 12)


def test_absent_cause_is_absent_not_zero(tmp_path):
    path = _write(tmp_path, environment_parameters=None, disturbances=None)
    data = json.loads(path.read_text())
    assert "environment_parameters" not in data
    assert "disturbances" not in data
    record = resolve_seed(path, "failure_onset_minus_seconds", offset_seconds=0.2)
    assert record["declared_environment_parameters"] == {}
    assert record["declared_disturbances"] == []


def test_world_frame_is_writable_and_declared(tmp_path):
    path = _write(tmp_path, position_frame="world")
    assert load_initial_state(path, 10).position_frame == "world"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(frames=[]), "at least one frame"),
        (dict(position_frame="odom"), "Unknown position_frame"),
        (dict(capsule_id=""), "capsule_id is required"),
        (dict(control_dt=0.0), "control_dt"),
        (dict(failure_onset_index=99), "failure_onset_index out of range"),
        (dict(pre_failure_start_index=25), "pre_failure_start_index"),
        (dict(episode_start_index=25), "episode_start_index"),
        (dict(extra={"capsule_id": "other"}), "may not redefine"),
    ],
)
def test_malformed_capsules_refused(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _write(tmp_path, **kwargs)


def test_partial_controller_history_refused(tmp_path):
    frames = _frames()
    frames[7].pop("action")
    with pytest.raises(ValueError, match="must be complete or absent"):
        _write(tmp_path, frames=frames)


def test_non_finite_and_unnormalized_frames_refused(tmp_path):
    frames = _frames()
    frames[3]["base_lin_vel_body"] = [np.nan, 0.0, 0.0]
    with pytest.raises(ValueError, match="finite"):
        _write(tmp_path, frames=frames)
    frames = _frames()
    frames[3]["base_quat"] = [0.0, 0.0, 0.0, 0.5]
    with pytest.raises(ValueError, match="normalized xyzw"):
        _write(tmp_path, frames=frames)


def test_recorded_evidence_is_not_overwritten(tmp_path):
    _write(tmp_path)
    _write(tmp_path)  # byte-identical rewrite is a no-op
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _write(tmp_path, capsule_id="cap-2")


def test_capsule_without_controller_history_is_a_state_only_seed(tmp_path):
    path = _write(tmp_path, frames=_frames(action=False, target=False))
    assert load_initial_state(path, 10).controller_history is None
    with pytest.raises(ValueError, match="no 'action' column"):
        load_initial_state(path, 10, history_rows=2)
