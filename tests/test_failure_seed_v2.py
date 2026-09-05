"""Failure-mechanism regression tests that fail under the old row-0 bridge."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from phoenix.adaptation.curriculum import FailureCurriculum, TrajectoryPool
from phoenix.adaptation.fine_tune import resolve_failure_reset_fraction
from phoenix.adaptation.reset_bridge import install, resolve_seed
from phoenix.replay.state_adapter import VelocityCommandAdapter, body_to_world
from phoenix.replay.trajectory_reader import load_initial_state
from tests.test_reset_bridge import _fake_env, _FakeRobot, _write_multi_row_parquet


def test_pre_onset_is_row40_never_row0(tmp_path):
    path = tmp_path / "failure.parquet"
    _write_multi_row_parquet(path, 50, 10, 0.5)
    record = resolve_seed(path, "failure_onset_minus_steps", 10)
    assert record["requested_seed_row"] == record["resolved_row"] == 40
    assert record["failure_onset_row"] == 50
    assert record["time_before_onset_seconds"] == pytest.approx(0.2)
    assert (
        resolve_seed(path, "failure_onset_minus_seconds", offset_seconds=0.2)["resolved_row"] == 40
    )
    assert resolve_seed(path)["resolved_row"] == 25


def test_insufficient_history_rejected(tmp_path):
    path = tmp_path / "failure.parquet"
    _write_multi_row_parquet(path, 5, 3, 0.5)
    with pytest.raises(ValueError, match="unavailable"):
        resolve_seed(path)
    with pytest.raises(ValueError, match="nonnegative"):
        resolve_seed(path, "failure_onset_minus_steps", -1)


def test_world_velocity_90_degree_yaw():
    q = [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    assert body_to_world([1, 0, 0], q) == pytest.approx([0, 1, 0])
    assert body_to_world([0, 1, 0], q) == pytest.approx([-1, 0, 0])
    with pytest.raises(ValueError):
        body_to_world([1, 0, 0], [0, 0, 0, 0])


def capsule(tmp_path):
    frame = dict(
        base_pos=[0, 0, 0.3],
        base_quat=[0, 0, 2**-0.5, 2**-0.5],
        base_lin_vel_body=[1, 0, 0],
        base_ang_vel_body=[0, 1, 0],
        joint_pos=[0] * 12,
        joint_vel=[1] * 12,
        command_vel=[0.6, 0.1, 0.2],
    )
    data = dict(
        schema_version="1.0",
        capsule_id="capsule-fixture",
        control_dt=0.02,
        failure_onset_index=50,
        pre_failure_start_index=0,
        frames=[frame] * 60,
    )
    path = tmp_path / "capsule.json"
    path.write_text(json.dumps(data))
    return path


def test_capsule_reset_restores_world_velocities_command_and_telemetry(tmp_path):
    torch = pytest.importorskip("torch")
    path = capsule(tmp_path)
    robot = _FakeRobot()
    env, _, target = _fake_env(robot, torch.tensor([[10.0, 20.0, 0.0]]), "cpu")
    pool = TrajectoryPool([path])
    log = tmp_path / "reset.jsonl"
    install(
        env,
        FailureCurriculum(pool, failure_reset_fraction=1),
        seed_row_strategy="failure_onset_minus_steps",
        seed_row_offset_steps=10,
        telemetry_path=log,
    )
    target._reset_idx(torch.tensor([0]))
    assert robot.root_velocity_calls[0][0][0].tolist() == pytest.approx(
        [0, 1, 0, -1, 0, 0], abs=1e-6
    )
    assert robot.root_pose_calls[0][0][0][:3].tolist() == pytest.approx([10, 20, 0.3])
    term = target.command_manager.get_term("base_velocity")
    assert term.command[0].tolist() == pytest.approx([0.6, 0.1, 0.2])
    assert not term.is_heading_env[0] and not term.is_standing_env[0]
    assert torch.isinf(term.time_left[0])
    record = json.loads(log.read_text())
    assert record["capsule_id"] == "capsule-fixture"
    assert record["resolved_row"] == 40
    assert record["restored_command"] == pytest.approx([0.6, 0.1, 0.2])


def test_missing_command_not_generic_fallback(tmp_path):
    path = capsule(tmp_path)
    data = json.loads(path.read_text())
    for frame in data["frames"]:
        frame["command_vel"] = None
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Missing required seed state: command_vel"):
        load_initial_state(path, 40)
    with pytest.raises(RuntimeError, match="Cannot restore command"):
        VelocityCommandAdapter(SimpleNamespace())


@pytest.mark.parametrize(
    "key", ["failure_reset_fraction", "failure_sample_fraction", "failure_fraction"]
)
def test_fraction_config_aliases(key):
    assert resolve_failure_reset_fraction({key: 0.3}) == 0.3


def test_ambiguous_aliases_rejected():
    with pytest.raises(ValueError):
        resolve_failure_reset_fraction({"failure_reset_fraction": 0.2, "failure_fraction": 0.3})
    with pytest.raises(ValueError):
        FailureCurriculum(TrajectoryPool([]), failure_reset_fraction=0.2, failure_fraction=0.2)
