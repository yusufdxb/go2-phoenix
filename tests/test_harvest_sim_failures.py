"""Tests for scripts/harvest_sim_failures.py.

Runs without Isaac Lab, ROS 2 or torch: the script imports Isaac only inside
``main``, and everything under test here is the harvest bookkeeping.

The property under test is the one the audit found broken: the SIMULATOR is
ground truth for whether an episode failed, and the detector is a measurement
against it. A termination the detector missed must still be harvested, its
miss recorded, and detector recall reported.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "harvest_sim_failures.py"

DT = 0.02
STANDING_HEIGHT = 0.35
COLLAPSED_HEIGHT = 0.10


def _load_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("harvest_sim_failures", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves string annotations through sys.modules, so the
    # module has to be registered before it executes.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def harvest():
    return _load_module()


@pytest.fixture(scope="module")
def logger_classes():
    from phoenix.real_world.trajectory_logger import TrajectoryLogger, TrajectoryStep

    return TrajectoryLogger, TrajectoryStep


@pytest.fixture(scope="module")
def detector_factory():
    from phoenix.real_world.failure_detector import FailureDetector

    return FailureDetector


def _row(height: float) -> dict:
    return {
        "base_pos": np.array([0.0, 0.0, height], dtype=np.float32),
        "base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # xyzw identity
        "base_lin_vel_body": np.zeros(3, dtype=np.float32),
        "base_ang_vel_body": np.zeros(3, dtype=np.float32),
        "joint_pos": np.zeros(12, dtype=np.float32),
        "joint_vel": np.zeros(12, dtype=np.float32),
        "command_vel": np.zeros(3, dtype=np.float32),
        "action": np.zeros(12, dtype=np.float32),
        "contact_forces": np.zeros(4, dtype=np.float32),
        "contact_forces_units": "newtons",
    }


def _window(n_rows: int, collapse_at: int | None) -> list[dict]:
    """Upright rows, optionally collapsing (base height below floor) at a row."""
    return [
        _row(COLLAPSED_HEIGHT if collapse_at is not None and i >= collapse_at else STANDING_HEIGHT)
        for i in range(n_rows)
    ]


def _harvest_one(harvest, logger_classes, detector_factory, tmp_path, rows, **kwargs):
    logger_cls, step_cls = logger_classes
    params = dict(
        rows=rows,
        dt_ctrl=DT,
        out_dir=tmp_path,
        index=0,
        env_index=3,
        step_index=1234,
        terms=["base_contact"],
        detector=detector_factory(),
        logger_cls=logger_cls,
        step_cls=step_cls,
        min_pre_onset_rows=100,
    )
    params.update(kwargs)
    return harvest.harvest_termination(**params)


# --------------------------------------------------------------------------
# Detector fired: unchanged behaviour, plus the ground truth is now recorded.
# --------------------------------------------------------------------------
def test_detected_failure_is_written_with_both_records(
    harvest, logger_classes, detector_factory, tmp_path
):
    rows = _window(200, collapse_at=150)
    record = _harvest_one(harvest, logger_classes, detector_factory, tmp_path, rows)

    assert record.status == "written"
    # simulator ground truth
    assert record.sim_termination_terms == ["base_contact"]
    assert record.sim_termination_index == 199
    assert record.sim_termination_time_s == pytest.approx(199 * DT)
    # detector output, kept separate
    assert record.detector_fired is True
    assert record.detector_mode == "collapse"
    assert record.detector_onset_index == 150
    assert record.detector_false_negative is False
    # negative latency means the detector led the simulator's termination
    assert record.detector_latency_s == pytest.approx((150 - 199) * DT)
    assert record.onset_index == 150
    assert record.onset_source == "detector"

    table = pq.read_table(record.path)
    flags = table.column("failure_flag").to_pylist()
    modes = table.column("failure_mode").to_pylist()
    assert flags[:150] == [False] * 150
    assert all(flags[150:])
    assert modes[149] is None
    assert modes[150] == "collapse"
    # Sim provenance is labelled, never left at the "unknown" default.
    assert set(table.column("capture_source").to_pylist()) == {"sim"}
    assert set(table.column("base_lin_vel_source").to_pylist()) == {"sim_ground_truth"}
    assert set(table.column("contact_forces_units").to_pylist()) == {"newtons"}


def test_sidecar_keeps_simulator_and_detector_fields_apart(
    harvest, logger_classes, detector_factory, tmp_path
):
    rows = _window(200, collapse_at=150)
    record = _harvest_one(harvest, logger_classes, detector_factory, tmp_path, rows)
    meta = json.loads(Path(record.path).with_suffix(".meta.json").read_text())

    assert meta["schema_version"] == harvest.HARVEST_SCHEMA_VERSION
    assert meta["sim_termination_terms"] == ["base_contact"]
    assert meta["sim_termination_index"] == 199
    assert meta["detector_id"] == harvest.DETECTOR_ID
    assert meta["detector_mode"] == "collapse"
    assert meta["detector_onset_index"] == 150
    # No field mixes the two sources.
    assert set(meta["field_notes"]) >= {"sim_*", "detector_*", "onset_source"}


# --------------------------------------------------------------------------
# The bug: a genuine termination the detector missed used to be discarded.
# --------------------------------------------------------------------------
def test_detector_miss_is_retained_not_discarded(
    harvest, logger_classes, detector_factory, tmp_path
):
    rows = _window(200, collapse_at=None)  # upright the whole way: detector sees nothing
    record = _harvest_one(harvest, logger_classes, detector_factory, tmp_path, rows)

    assert record.status == "written", "a genuine simulator failure must never be dropped"
    assert record.detector_fired is False
    assert record.detector_false_negative is True
    assert record.detector_mode is None
    assert record.detector_latency_s is None
    # Onset falls back to the simulator's own termination row.
    assert record.onset_source == "simulator_termination"
    assert record.onset_index == 199

    table = pq.read_table(record.path)
    flags = table.column("failure_flag").to_pylist()
    modes = table.column("failure_mode").to_pylist()
    assert flags[:199] == [False] * 199
    assert flags[199] is True
    # failure_mode stays the detector's column: unlabelled, never borrowing the
    # simulator's termination term as if it were a detected mode.
    assert set(modes) == {None}


# --------------------------------------------------------------------------
# The structural guard that stays, under its own reason.
# --------------------------------------------------------------------------
def test_window_with_no_pre_onset_interval_is_rejected_structurally(
    harvest, logger_classes, detector_factory, tmp_path
):
    rows = _window(200, collapse_at=10)
    record = _harvest_one(harvest, logger_classes, detector_factory, tmp_path, rows)

    assert record.status == "rejected"
    assert record.rejected_reason == "no_usable_pre_onset_window"
    assert record.path is None
    assert list(tmp_path.glob("*.parquet")) == []
    # The detector measurement is still recorded for a rejected window.
    assert record.detector_fired is True
    assert record.detector_onset_index == 10


def test_short_termination_without_pre_onset_rows_is_rejected(
    harvest, logger_classes, detector_factory, tmp_path
):
    """A detector miss cannot smuggle a zero-length window past the guard."""
    rows = _window(40, collapse_at=None)
    record = _harvest_one(harvest, logger_classes, detector_factory, tmp_path, rows)
    assert record.status == "rejected"
    assert record.rejected_reason == "no_usable_pre_onset_window"


def test_single_row_window_is_rejected(harvest, logger_classes, detector_factory, tmp_path):
    record = _harvest_one(harvest, logger_classes, detector_factory, tmp_path, _window(1, None))
    assert record.status == "rejected"
    assert record.rejected_reason == "window_shorter_than_two_rows"
    assert record.sim_termination_index == 0


# --------------------------------------------------------------------------
# Recall is reported as a measurement over ALL genuine terminations.
# --------------------------------------------------------------------------
def test_evaluate_detector_reports_a_miss_without_raising(harvest, detector_factory):
    evaluation = harvest.evaluate_detector(_window(50, None), DT, detector_factory())
    assert evaluation.fired is False
    assert evaluation.mode is None
    assert evaluation.onset_index is None


def test_report_counts_rejected_trajectories_in_the_recall_denominator(
    harvest, logger_classes, detector_factory, tmp_path
):
    records = [
        _harvest_one(
            harvest, logger_classes, detector_factory, tmp_path, _window(200, 150), index=0
        ),
        _harvest_one(
            harvest, logger_classes, detector_factory, tmp_path, _window(200, None), index=1
        ),
        # Detected but structurally unusable: still a detector hit.
        _harvest_one(
            harvest, logger_classes, detector_factory, tmp_path, _window(200, 10), index=2
        ),
    ]
    report = harvest.build_report(
        records,
        dt_ctrl=DT,
        min_pre_onset_rows=100,
        terminations_seen=10,
        timeouts_seen=7,
        unattributed_seen=0,
    )

    assert report["counts"]["terminations_seen"] == 10
    assert report["counts"]["time_out_terminations"] == 7
    assert report["counts"]["genuine_failures"] == 3
    assert report["counts"]["written"] == 2
    assert report["counts"]["rejected"] == {"no_usable_pre_onset_window": 1}
    assert report["detector"]["fired"] == 2
    assert report["detector"]["false_negatives"] == 1
    assert report["detector"]["recall"] == pytest.approx(2 / 3)
    assert report["detector"]["fired_by_mode"] == {"collapse": 2}
    assert report["detector"]["mean_lead_s"] > 0.0
    assert len(report["trajectories"]) == 3


def test_report_is_json_serializable_and_reads_as_a_measurement(harvest):
    report = harvest.build_report(
        [],
        dt_ctrl=DT,
        min_pre_onset_rows=100,
        terminations_seen=0,
        timeouts_seen=0,
        unattributed_seen=0,
    )
    json.dumps(report)
    assert report["detector"]["recall"] is None
    text = harvest.format_report(report)
    assert "detector recall" in text
    assert "MEASUREMENT" in text


def test_recall_denominator_excludes_windows_the_detector_never_saw(harvest):
    """Regression: an exactly-0.500 recall was an artifact of the window guard.

    The len(rows) < 2 guard returns BEFORE evaluate_detector runs, so those
    records keep detector_fired=False by default. Dividing hits by ALL genuine
    terminations therefore scored the detector as missing windows it was never
    shown, reporting the guard's behaviour as the detector's. Recall must be
    over evaluated windows only, with the unevaluable ones reported separately.
    """
    record_cls = harvest.TerminationRecord

    def record(evaluated, fired):
        r = record_cls(
            env_index=0,
            step_index=0,
            window_rows=2 if evaluated else 1,
            sim_termination_terms=["base_contact"],
            sim_termination_index=1 if evaluated else 0,
            sim_termination_time_s=0.02,
        )
        r.detector_evaluated = evaluated
        r.detector_fired = fired
        return r

    # 4 windows shown to the detector, all detected; 4 never shown.
    records = [record(True, True) for _ in range(4)] + [record(False, False) for _ in range(4)]
    report = harvest.build_report(
        records,
        dt_ctrl=0.02,
        min_pre_onset_rows=100,
        terminations_seen=8,
        timeouts_seen=0,
        unattributed_seen=0,
    )
    det = report["detector"]
    assert det["genuine_failures"] == 8
    assert det["evaluated"] == 4
    assert det["not_evaluable"] == 4
    assert det["fired"] == 4
    # The bug reported 0.5 here. The detector saw 4 and caught 4.
    assert det["recall"] == 1.0
    assert det["false_negatives"] == 0
