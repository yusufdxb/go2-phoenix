"""Item 3: a single state row is not the state of a delayed closed loop.

``last_action`` is a trained observation term, the rate limiter can carry a
previous target, and the DC motor delays setpoints through a buffer whose lag
is redrawn at every reset. These tests pin what the bridge restores, what it
refuses to claim, and that "exact replay" is an exception rather than a label
when any of it cannot be rebuilt.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from phoenix.replay.controller_history import (
    REPLAY_APPROXIMATE,
    REPLAY_EXACT,
    REPLAY_STATE_ONLY,
    ControllerHistory,
    inspect_controller_state,
)
from phoenix.replay.trajectory_reader import load_initial_state
from tests.test_failure_seed_v2 import capsule


def _history(rows=3, dim=12, targets=False, episode_start=True):
    actions = np.arange(rows * dim, dtype=np.float32).reshape(rows, dim) * 0.01
    return ControllerHistory(
        actions=actions,
        joint_targets=actions + 0.5 if targets else None,
        starts_at_episode_start=episode_start,
        control_dt=0.02,
        source_rows=(0, rows - 1),
    )


def _env(*, action_dim=12, clip_mode=None, delay=0, num_envs=2, array=np.zeros):
    terms = {}
    if clip_mode is not None:
        terms["joint_pos"] = SimpleNamespace(
            cfg=SimpleNamespace(clip_mode=clip_mode),
            _offset=array((num_envs, action_dim)),
        )
    actuators = {}
    if delay:
        actuators["base_legs"] = SimpleNamespace(cfg=SimpleNamespace(max_delay=delay))
    robot = SimpleNamespace(actuators=actuators)
    manager = SimpleNamespace(
        _action=array((num_envs, action_dim)),
        _prev_action=array((num_envs, action_dim)),
        _terms=terms,
    )
    manager.action = manager._action
    return SimpleNamespace(action_manager=manager, scene={"robot": robot})


# -------------------- validation (no torch) --------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(actions=np.zeros((0, 12))),
        dict(actions=np.zeros(12)),
        dict(actions=np.full((2, 12), np.nan)),
        dict(actions=np.zeros((2, 12)), joint_targets=np.zeros((3, 12))),
        dict(actions=np.zeros((2, 12)), control_dt=0.0),
        dict(actions=np.zeros((2, 12)), source_rows=(5, 1)),
    ],
)
def test_malformed_history_rejected(kwargs):
    with pytest.raises(ValueError):
        ControllerHistory(**kwargs)


def test_history_exposes_last_and_previous_action():
    history = _history(rows=3)
    assert history.last_action == pytest.approx(history.actions[2])
    assert history.previous_action == pytest.approx(history.actions[1])
    assert ControllerHistory(actions=np.zeros((1, 12))).previous_action is None


# -------------------- inspection (no torch) --------------------------------


def test_delayed_actuator_can_never_be_reconstructed():
    report = inspect_controller_state(_env(delay=4), _history(targets=True))
    assert "delayed_actuator.base_legs.delay_buffer_redrawn_at_reset" in (
        report["not_reconstructed"]
    )
    assert "action_manager.action" in report["restorable"]


def test_prev_command_rate_limiter_needs_recorded_joint_targets():
    without = inspect_controller_state(_env(clip_mode="prev_command"), _history(targets=False))
    assert "rate_limited_action.joint_pos.prev_target_no_joint_target_history" in (
        without["not_reconstructed"]
    )
    with_targets = inspect_controller_state(
        _env(clip_mode="prev_command"), _history(targets=True)
    )
    assert with_targets["not_reconstructed"] == []
    assert "rate_limited_action.joint_pos.prev_target" in with_targets["restorable"]


def test_measured_q_rate_limiter_is_stateless():
    report = inspect_controller_state(_env(clip_mode="measured_q"), _history())
    assert report["not_reconstructed"] == []


def test_truncated_window_and_short_history_are_named():
    report = inspect_controller_state(
        _env(), ControllerHistory(actions=np.zeros((1, 12)), starts_at_episode_start=False)
    )
    assert "action_manager.prev_action_history_too_short" in report["not_reconstructed"]
    assert "controller_history.window_does_not_start_at_episode_start" in (
        report["not_reconstructed"]
    )


def test_no_history_is_reported_as_no_history():
    report = inspect_controller_state(_env(), None)
    assert report["not_reconstructed"] == ["action_manager.last_action_no_history"]


# -------------------- capsule loading (no torch) ---------------------------


def test_capsule_history_rows_and_episode_start(tmp_path):
    state = load_initial_state(capsule(tmp_path), 40, history_rows=3)
    assert state.controller_history is not None
    assert len(state.controller_history) == 3
    assert state.controller_history.source_rows == (38, 40)
    assert state.controller_history.last_action == pytest.approx([0.25] * 12)
    assert state.controller_history.control_dt == pytest.approx(0.02)
    assert state.controller_history.starts_at_episode_start is False
    whole = load_initial_state(capsule(tmp_path), 40, history_rows=-1)
    assert len(whole.controller_history) == 41
    assert whole.controller_history.starts_at_episode_start is True
    assert load_initial_state(capsule(tmp_path), 40).controller_history is None


def test_capsule_with_joint_targets_carries_them(tmp_path):
    path = capsule(tmp_path)
    data = json.loads(path.read_text())
    for frame in data["frames"]:
        frame["joint_target"] = [0.75] * 12
    path.write_text(json.dumps(data))
    state = load_initial_state(path, 40, history_rows=2)
    assert state.controller_history.joint_targets[-1] == pytest.approx([0.75] * 12)


def test_source_without_actions_refuses_to_pretend(tmp_path):
    path = capsule(tmp_path)
    data = json.loads(path.read_text())
    for frame in data["frames"]:
        frame.pop("action")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="no 'action' column"):
        load_initial_state(path, 40, history_rows=2)
    assert load_initial_state(path, 40, history_rows=0).controller_history is None


# -------------------- writes (torch) ---------------------------------------


def test_restore_writes_last_action_and_reports_approximate():
    torch = pytest.importorskip("torch")
    from phoenix.replay.controller_history import restore_controller_history

    env = _env(delay=3, array=torch.zeros)
    history = _history(rows=3)
    record = restore_controller_history(env, 1, history)
    assert env.action_manager._action[1].numpy() == pytest.approx(history.actions[2])
    assert env.action_manager._prev_action[1].numpy() == pytest.approx(history.actions[1])
    assert env.action_manager._action[0].numpy() == pytest.approx(np.zeros(12))
    assert record["replay_fidelity"] == REPLAY_APPROXIMATE
    assert record["controller_state_not_reconstructed"] == [
        "delayed_actuator.base_legs.delay_buffer_redrawn_at_reset"
    ]
    assert record["controller_history_rows"] == 3


def test_exact_is_reported_only_when_nothing_is_left_over():
    torch = pytest.importorskip("torch")
    from phoenix.replay.controller_history import restore_controller_history

    env = _env(clip_mode="prev_command", array=torch.zeros)
    record = restore_controller_history(env, 0, _history(targets=True))
    assert record["replay_fidelity"] == REPLAY_EXACT
    assert record["controller_state_not_reconstructed"] == []
    assert env.action_manager._terms["joint_pos"]._prev_target[0].numpy() == pytest.approx(
        _history(targets=True).joint_targets[-1]
    )


def test_require_exact_raises_instead_of_claiming():
    torch = pytest.importorskip("torch")
    from phoenix.replay.controller_history import restore_controller_history

    env = _env(delay=2, array=torch.zeros)
    with pytest.raises(RuntimeError, match="delayed_actuator.base_legs"):
        restore_controller_history(env, 0, _history(), require_exact=True)


def test_action_width_mismatch_is_loud():
    torch = pytest.importorskip("torch")
    from phoenix.replay.controller_history import restore_controller_history

    env = _env(action_dim=8, array=torch.zeros)
    with pytest.raises(ValueError, match="does not match the environment"):
        restore_controller_history(env, 0, _history(dim=12))


def test_state_only_seed_is_labelled_and_cannot_claim_exact(tmp_path):
    torch = pytest.importorskip("torch")
    from phoenix.replay.state_adapter import restore_state
    from tests.test_reset_bridge import _fake_env, _FakeRobot

    robot = _FakeRobot()
    env, _, target = _fake_env(robot, torch.zeros(1, 3), "cpu")
    state = load_initial_state(capsule(tmp_path), 40)
    record = restore_state(target, state, 0)
    assert record["replay_fidelity"] == REPLAY_STATE_ONLY
    assert record["controller_state_not_reconstructed"] == ["controller_history_not_supplied"]
    with pytest.raises(RuntimeError, match="no controller history"):
        restore_state(target, state, 0, require_exact_replay=True)
