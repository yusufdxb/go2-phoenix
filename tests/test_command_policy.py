"""Item 11: a seeded env must not also get a different command PROCESS.

``reset_bridge.install`` used to pin the restored velocity command for the
whole episode (``command_hold_seconds=inf``) while control envs kept resampling
normally, so treatment and control differed in state AND in the command
process. The policy is now declared, bounded, and recorded in the telemetry.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phoenix.adaptation.curriculum import FailureCurriculum, TrajectoryPool
from phoenix.adaptation.reset_bridge import install
from phoenix.replay.state_adapter import resolve_command_hold
from tests.test_failure_seed_v2 import capsule


def test_source_hold_to_onset_replays_only_the_replayed_interval():
    hold, telemetry = resolve_command_hold("source_hold_to_onset", time_before_onset_seconds=0.5)
    assert hold == pytest.approx(0.5)
    assert telemetry["command_policy"] == "source_hold_to_onset"
    assert telemetry["command_hold"] == "seconds"
    assert telemetry["command_hold_seconds"] == pytest.approx(0.5)


def test_seeding_at_onset_returns_straight_to_the_normal_process():
    hold, telemetry = resolve_command_hold("source_hold_to_onset", time_before_onset_seconds=0.0)
    assert hold is None
    assert telemetry["command_hold"] == "reset_process"
    assert telemetry["command_hold_source"] == "source_hold_to_onset_zero_interval"


def test_source_hold_without_a_recorded_interval_fails_loudly():
    with pytest.raises(ValueError, match="time to onset"):
        resolve_command_hold("source_hold_to_onset")
    with pytest.raises(ValueError, match="finite and nonnegative"):
        resolve_command_hold("source_hold_to_onset", time_before_onset_seconds=-1.0)


def test_fixed_hold_requires_a_finite_positive_duration():
    hold, telemetry = resolve_command_hold("fixed_hold", hold_seconds=1.5)
    assert hold == pytest.approx(1.5)
    assert telemetry["command_hold_seconds"] == pytest.approx(1.5)
    for bad in (None, 0.0, -1.0, float("inf")):
        with pytest.raises(ValueError, match="fixed_hold"):
            resolve_command_hold("fixed_hold", hold_seconds=bad)


def test_match_control_process_leaves_the_reset_clock_alone():
    hold, telemetry = resolve_command_hold("match_control_process")
    assert hold is None
    assert telemetry["command_hold"] == "reset_process"


def test_legacy_forever_hold_is_available_but_named():
    hold, telemetry = resolve_command_hold("hold_to_episode_end_legacy")
    assert np.isinf(hold)
    assert telemetry["command_policy"] == "hold_to_episode_end_legacy"
    assert telemetry["command_hold"] == "episode"
    assert telemetry["command_hold_seconds"] is None


def test_unknown_policy_rejected():
    with pytest.raises(ValueError, match="Unknown command_policy"):
        resolve_command_hold("hold_forever")


def test_install_rejects_a_hold_the_policy_would_ignore(tmp_path: Path):
    pool = TrajectoryPool([capsule(tmp_path)])
    curriculum = FailureCurriculum(pool, failure_reset_fraction=1.0)
    with pytest.raises(ValueError, match="only applies to command_policy"):
        install(object(), curriculum, command_hold_seconds=2.0)
    with pytest.raises(ValueError, match="Unknown command_policy"):
        install(object(), curriculum, command_policy="forever")


# -------------------- installed behaviour (torch) --------------------------


def _install(tmp_path, **kwargs):
    torch = pytest.importorskip("torch")
    from tests.test_reset_bridge import _fake_env, _FakeRobot

    robot = _FakeRobot()
    env, _, target = _fake_env(robot, torch.zeros(2, 3), "cpu")
    term = target.command_manager.get_term("base_velocity")
    # The environment's own reset draws a resampling clock; a seeded env that
    # keeps it is on exactly the control envs' command process.
    term.time_left[:] = 9.0
    log = tmp_path / f"reset_{kwargs.get('command_policy', 'default')}.jsonl"
    install(
        env,
        FailureCurriculum(TrajectoryPool([capsule(tmp_path)]), failure_reset_fraction=1.0),
        seed_row_strategy="failure_onset_minus_steps",
        seed_row_offset_steps=10,
        telemetry_path=log,
        **kwargs,
    )
    target._reset_idx(torch.tensor([0]))
    return term, json.loads(log.read_text().splitlines()[0])


def test_default_policy_returns_to_the_normal_command_process(tmp_path):
    torch = pytest.importorskip("torch")
    term, record = _install(tmp_path)
    assert not torch.isinf(term.time_left[0])
    assert term.time_left[0].item() == pytest.approx(0.2)
    assert record["command_policy"] == "source_hold_to_onset"
    assert record["command_hold"] == "seconds"


def test_match_control_process_does_not_touch_the_reset_clock(tmp_path):
    pytest.importorskip("torch")
    term, record = _install(tmp_path, command_policy="match_control_process")
    assert term.time_left[0].item() == pytest.approx(9.0)
    assert record["command_hold"] == "reset_process"
    assert record["command_hold_seconds"] is None
    assert term.command[0].tolist() == pytest.approx([0.6, 0.1, 0.2])


def test_legacy_policy_still_reproduces_the_forever_hold(tmp_path):
    torch = pytest.importorskip("torch")
    term, record = _install(tmp_path, command_policy="hold_to_episode_end_legacy")
    assert torch.isinf(term.time_left[0])
    assert record["command_policy"] == "hold_to_episode_end_legacy"
    assert record["command_hold"] == "episode"


def test_fixed_hold_is_written_to_the_term_and_the_telemetry(tmp_path):
    pytest.importorskip("torch")
    term, record = _install(tmp_path, command_policy="fixed_hold", command_hold_seconds=1.25)
    assert term.time_left[0].item() == pytest.approx(1.25)
    assert record["command_hold_seconds"] == pytest.approx(1.25)
