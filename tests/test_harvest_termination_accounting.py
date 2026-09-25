"""Termination accounting in scripts/harvest_sim_failures.py (report schema 2.0).

The 1.0 harvest recorded every physical fall twice: the real window at step s,
then a one-row window on the same environment at step s + 1 with the same
``base_contact`` term. These tests pin the rule that separates a physical
episode termination from that post-reset artifact, and pin that the rule is
narrow enough never to merge away a genuine fall.

Pure python: no Isaac Lab, torch or ROS.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "harvest_sim_failures.py"
HARVEST_DIR = REPO_ROOT / "data" / "failures" / "sim_harvest"
HISTORICAL_REPORT = HARVEST_DIR / "harvest_report.json"
RECOMPUTED_REPORT = HARVEST_DIR / "harvest_report.v2_recomputed.json"
# configs/env/base.yaml episode_length_s 20.0 / control_dt_s 0.02.
MAX_EPISODE_STEPS = 1000


@pytest.fixture(scope="module")
def harvest():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    name = "harvest_sim_failures_accounting"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tick(h, env, step, length, terms=("base_contact",), generation=None):
    return h.TerminationTick(
        env_index=env,
        step_index=step,
        terms=tuple(terms),
        episode_length_steps=length,
        episode_generation=generation,
    )


# --------------------------------------------------------------------------
# The predicate
# --------------------------------------------------------------------------
def test_first_step_after_reset_with_same_terms_is_an_artifact(harvest):
    previous = _tick(harvest, 13, 72, 73)
    assert harvest.is_post_reset_artifact(_tick(harvest, 13, 73, 1), previous) is True


def test_no_previous_termination_is_never_an_artifact(harvest):
    assert harvest.is_post_reset_artifact(_tick(harvest, 13, 0, 1), None) is False


def test_back_to_back_genuine_fall_is_not_merged(harvest):
    """A second real fall shortly after the first keeps its own record."""
    previous = _tick(harvest, 4, 100, 300)
    later = _tick(harvest, 4, 160, 60)
    assert harvest.is_post_reset_artifact(later, previous) is False


def test_next_step_tick_whose_episode_is_not_one_step_old_is_not_merged(harvest):
    previous = _tick(harvest, 4, 100, 300)
    assert harvest.is_post_reset_artifact(_tick(harvest, 4, 101, 2), previous) is False


def test_different_terms_are_not_merged(harvest):
    previous = _tick(harvest, 4, 100, 300)
    other = _tick(harvest, 4, 101, 1, terms=("unattributed_termination",))
    assert harvest.is_post_reset_artifact(other, previous) is False


def test_other_environment_is_not_merged(harvest):
    previous = _tick(harvest, 1, 100, 300)
    assert harvest.is_post_reset_artifact(_tick(harvest, 2, 101, 1), previous) is False


def test_unknown_episode_length_counts_as_a_failure(harvest):
    previous = _tick(harvest, 4, 100, 300)
    assert harvest.is_post_reset_artifact(_tick(harvest, 4, 101, None), previous) is False


def test_generation_must_be_the_next_one_when_both_are_known(harvest):
    previous = _tick(harvest, 4, 100, 300, generation=3)
    assert harvest.is_post_reset_artifact(_tick(harvest, 4, 101, 1, generation=4), previous)
    assert not harvest.is_post_reset_artifact(_tick(harvest, 4, 101, 1, generation=5), previous)
    assert not harvest.is_post_reset_artifact(_tick(harvest, 4, 101, 1, generation=3), previous)


# --------------------------------------------------------------------------
# The ledger both the live loop and the recompute go through
# --------------------------------------------------------------------------
def test_ledger_reproduces_the_74_plus_74_adjacency_pattern(harvest):
    ledger = harvest.TerminationLedger()
    ticks = []
    for k in range(74):
        env, step = k % 64, 100 + 37 * k
        ticks.append(_tick(harvest, env, step, 200))
        ticks.append(_tick(harvest, env, step + 1, 1))
    for tick in sorted(ticks, key=lambda t: (t.step_index, t.env_index)):
        ledger.observe(tick)

    assert ledger.count(harvest.CLASS_GENUINE) == 74
    assert ledger.count(harvest.CLASS_POST_RESET_ARTIFACT) == 74
    for verdict in ledger.classified:
        if verdict.classification == harvest.CLASS_POST_RESET_ARTIFACT:
            assert verdict.artifact_of_step_index == verdict.tick.step_index - 1


def test_ledger_attributes_an_artifact_chain_to_its_physical_termination(harvest):
    ledger = harvest.TerminationLedger()
    ledger.observe(_tick(harvest, 7, 500, 400))
    first = ledger.observe(_tick(harvest, 7, 501, 1))
    second = ledger.observe(_tick(harvest, 7, 502, 1))
    assert first.classification == harvest.CLASS_POST_RESET_ARTIFACT
    assert second.classification == harvest.CLASS_POST_RESET_ARTIFACT
    assert first.artifact_of_step_index == 500
    assert second.artifact_of_step_index == 500
    assert ledger.count(harvest.CLASS_GENUINE) == 1


def test_ledger_keeps_back_to_back_genuine_falls(harvest):
    ledger = harvest.TerminationLedger()
    ledger.observe(_tick(harvest, 2, 300, 301))
    ledger.observe(_tick(harvest, 2, 301, 1))  # artifact of 300
    ledger.observe(_tick(harvest, 2, 340, 38))  # a real second fall
    assert ledger.count(harvest.CLASS_GENUINE) == 2
    assert ledger.count(harvest.CLASS_POST_RESET_ARTIFACT) == 1


def test_ledger_classifies_a_time_out_only_tick_separately(harvest):
    ledger = harvest.TerminationLedger()
    verdict = ledger.observe(_tick(harvest, 0, 999, 1000, terms=("time_out",)))
    assert verdict.classification == harvest.CLASS_TIME_OUT


def test_ledger_refuses_ticks_out_of_step_order(harvest):
    ledger = harvest.TerminationLedger()
    ledger.observe(_tick(harvest, 3, 50, 51))
    with pytest.raises(ValueError, match="step order"):
        ledger.observe(_tick(harvest, 3, 50, 1))


def test_build_report_refuses_accounting_that_does_not_close(harvest):
    with pytest.raises(ValueError, match="does not close"):
        harvest.build_report(
            [],
            dt_ctrl=0.02,
            min_pre_onset_rows=100,
            termination_ticks_seen=2,
            time_out_ticks_seen=0,
            unattributed_seen=0,
            post_reset_artifacts=[],
        )


# --------------------------------------------------------------------------
# Time-out resets did not clear the 1.0 window
# --------------------------------------------------------------------------
def test_time_out_reset_steps(harvest):
    assert harvest.time_out_reset_steps(0, 1102, 1000) == [999]
    assert harvest.time_out_reset_steps(0, 999, 1000) == []
    assert harvest.time_out_reset_steps(0, 2500, 1000) == [999, 1999]
    assert harvest.time_out_reset_steps(379, 1560, 1000) == [1378]


def test_window_spans_reset_boundaries(harvest):
    # env 58, step 1102, 600 rows: the window starts at 503 and holds step 999.
    assert harvest.window_spans_reset(1102, 600, [999]) is True
    # env 28, step 1674, 600 rows: the window starts at 1075, after the reset.
    assert harvest.window_spans_reset(1674, 600, [999]) is False
    # A reset on the row before the window leaves the window clean.
    assert harvest.window_spans_reset(1600, 600, [1000]) is False
    # A reset on the first row of the window splices it.
    assert harvest.window_spans_reset(1600, 600, [1001]) is True


def test_undo_post_reset_contact_read_resets_exactly_the_given_envs(harvest):
    class Sensor:
        def __init__(self):
            self.calls = []

        def reset(self, env_ids):
            self.calls.append(list(env_ids))

    class Env:
        def __init__(self, scene):
            self.scene = scene

    sensor = Sensor()
    assert harvest.undo_post_reset_contact_read(Env({"contact_forces": sensor}), [3, 9]) is True
    assert sensor.calls == [[3, 9]]
    # No contact sensor in the scene: nothing to undo, and no crash.
    assert harvest.undo_post_reset_contact_read(Env({}), [1]) is False


def test_pre_reset_capture_tracks_generation_and_resets_this_step():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from phoenix.training.episode_outcomes import PreResetCapture

    class FakeEnv:
        def __init__(self):
            self.calls = []

        def _reset_idx(self, env_ids):
            self.calls.append(list(env_ids))

    env = FakeEnv()
    capture = PreResetCapture(env, lambda: {"x": np.arange(4.0)})
    capture.begin_step()
    env._reset_idx([1, 3])
    assert capture.generation == {1: 1, 3: 1}
    assert capture.reset_this_step == {1, 3}
    capture.begin_step()
    assert capture.reset_this_step == set()
    env._reset_idx([3])
    assert capture.generation == {1: 1, 3: 2}
    assert capture.reset_this_step == {3}
    assert env.calls == [[1, 3], [3]]
    capture.close()


# --------------------------------------------------------------------------
# The historical 1.0 report, recomputed through the same ledger
# --------------------------------------------------------------------------
historical = pytest.mark.skipif(
    not HISTORICAL_REPORT.is_file(), reason="historical 1.0 harvest report not present"
)


@historical
def test_recompute_of_the_historical_report(harvest):
    before = hashlib.sha256(HISTORICAL_REPORT.read_bytes()).hexdigest()
    report = harvest.recompute_report_from_v1(
        HISTORICAL_REPORT, max_episode_length_steps=MAX_EPISODE_STEPS
    )
    after = hashlib.sha256(HISTORICAL_REPORT.read_bytes()).hexdigest()

    assert before == after, "the historical report must never be modified"
    assert report["schema_version"] == harvest.HARVEST_SCHEMA_VERSION
    assert report["recomputed_from"]["sha256"] == before
    counts = report["counts"]
    assert counts["termination_ticks_seen"] == 148
    assert counts["post_reset_artifacts"] == 74
    assert counts["physical_terminations"] == 74
    assert counts["time_out_ticks"] == 0
    assert counts["written"] == 3
    assert counts["rejected"] == {"no_usable_pre_onset_window": 71}
    assert counts["written_with_spliced_window"] == 1
    assert len(report["written_with_spliced_window"]) == 1
    assert report["written_with_spliced_window"][0].endswith(
        "sim_fall_0001_env058_step001102.parquet"
    )
    detector = report["detector"]
    assert detector["physical_terminations"] == 74
    assert detector["evaluated"] == 74
    assert detector["not_evaluable"] == 0
    assert detector["fired"] == 74
    assert detector["recall"] == 1.0
    # Every artifact is the one-row tick on the step after a physical termination.
    for artifact in report["post_reset_artifacts"]:
        assert artifact["episode_length_steps"] == 1
        assert artifact["artifact_of_step_index"] == artifact["step_index"] - 1


@historical
def test_committed_recomputed_report_is_exactly_derivable(harvest):
    if not RECOMPUTED_REPORT.is_file():
        pytest.skip("recomputed report not written yet")
    report = harvest.recompute_report_from_v1(
        HISTORICAL_REPORT, max_episode_length_steps=MAX_EPISODE_STEPS
    )
    assert json.loads(RECOMPUTED_REPORT.read_text()) == json.loads(json.dumps(report))


@historical
def test_write_recomputed_report_refuses_to_overwrite_evidence(harvest, tmp_path):
    with pytest.raises(ValueError, match="historical report"):
        harvest.write_recomputed_report(
            HISTORICAL_REPORT, HISTORICAL_REPORT, max_episode_length_steps=MAX_EPISODE_STEPS
        )
    out = tmp_path / "recomputed.json"
    out.write_text("{}\n")
    with pytest.raises(FileExistsError):
        harvest.write_recomputed_report(
            HISTORICAL_REPORT, out, max_episode_length_steps=MAX_EPISODE_STEPS
        )
    fresh = tmp_path / "fresh.json"
    harvest.write_recomputed_report(
        HISTORICAL_REPORT, fresh, max_episode_length_steps=MAX_EPISODE_STEPS
    )
    # Writing identical content again is allowed.
    harvest.write_recomputed_report(
        HISTORICAL_REPORT, fresh, max_episode_length_steps=MAX_EPISODE_STEPS
    )
