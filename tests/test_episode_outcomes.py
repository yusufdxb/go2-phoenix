import numpy as np
import pytest

from phoenix.training.episode_outcomes import (
    EpisodeOutcome,
    PreResetCapture,
    load_outcomes,
    write_outcomes,
)
from phoenix.training.evaluate import parse_args


def outcome(**kwargs):
    return EpisodeOutcome(
        policy_id="sha256:abc",
        evaluation_seed=19,
        episode_id=0,
        success=False,
        termination_reason="base_contact",
        episode_length_steps=10,
        control_dt_s=0.02,
        episode_return=3.0,
        **kwargs,
    )


def test_roundtrip_preserves_unknowns_and_independent_seeds(tmp_path):
    value = outcome(
        training_seed=1,
        scenario_seed=2,
        scenario_id="s1",
        parameter_sample_id="p1",
        environment_parameters={"friction": 0.2},
    )
    path = write_outcomes(tmp_path / "episodes.jsonl", [value])
    assert load_outcomes(path) == [value]
    assert value.failure_modes is None
    assert value.recovery_outcome is None
    assert value.to_dict()["episode_length_s"] == 0.2


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(scenario_id="fake"),
        dict(tracking_error=float("nan")),
        dict(command=[1, 2]),
        dict(failure_events=[]),
        dict(schema_version="1.0.0"),
    ],
)
def test_invalid_outcome_rejected(kwargs):
    with pytest.raises(ValueError):
        outcome(**kwargs)


def test_duration_tampering_rejected(tmp_path):
    path = write_outcomes(tmp_path / "episodes.jsonl", [outcome()])
    path.write_text(path.read_text().replace('"episode_length_s": 0.2', '"episode_length_s": 2.0'))
    with pytest.raises(ValueError):
        load_outcomes(path)


def test_pre_reset_capture_does_not_log_generic_reset_state():
    class Env:
        def __init__(self):
            self.state = np.array([[10.0, 20.0], [30.0, 40.0]])

        def _reset_idx(self, ids):
            self.state[ids] = 0

    env = Env()
    capture = PreResetCapture(env, lambda: {"velocity": env.state.copy()})
    capture.begin_step()
    env._reset_idx(np.array([1]))
    observed = capture.overlay({"velocity": env.state.copy()})
    np.testing.assert_array_equal(observed["velocity"], [[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_array_equal(env.state[1], [0.0, 0.0])
    capture.begin_step()
    assert capture.terminal == {}
    capture.close()
    env._reset_idx([0])
    assert capture.terminal == {}


def test_unsupported_simulator_fails_loudly():
    with pytest.raises(RuntimeError):
        PreResetCapture(object(), lambda: {})


def test_cli_distinguishes_training_and_evaluation_seed():
    args = parse_args(
        [
            "--checkpoint",
            "p.pt",
            "--env-config",
            "e.yaml",
            "--seed",
            "21",
            "--training-seed",
            "7",
            "--failure-analyzer-factory",
            "ashfall.evaluation.metrics:FailureAnalyzer",
        ]
    )
    assert args.seed == 21 and args.training_seed == 7


def test_episode_evidence_cannot_be_overwritten(tmp_path):
    path = write_outcomes(tmp_path / "episodes.jsonl", [outcome()])
    write_outcomes(path, [outcome()])
    with pytest.raises(FileExistsError):
        write_outcomes(path, [outcome(training_seed=99)])
