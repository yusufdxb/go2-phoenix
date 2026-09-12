"""Round-trip test for the Parquet trajectory logger."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from phoenix.real_world.trajectory_logger import (
    _SCHEMA,
    CAPTURE_SOURCE_HARDWARE,
    CAPTURE_SOURCE_UNKNOWN,
    TrajectoryLogger,
    TrajectoryStep,
)


def _make_step(i: int) -> TrajectoryStep:
    return TrajectoryStep(
        step=i,
        timestamp_s=i * 0.02,
        base_pos=np.asarray([0.1 * i, 0.0, 0.4], dtype=np.float32),
        base_quat=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        base_lin_vel_body=np.asarray([0.5, 0.0, 0.0], dtype=np.float32),
        base_ang_vel_body=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        joint_pos=np.arange(12, dtype=np.float32) * 0.1,
        joint_vel=np.arange(12, dtype=np.float32) * 0.01,
        command_vel=np.asarray([0.5, 0.0, 0.0], dtype=np.float32),
        action=np.arange(12, dtype=np.float32),
        contact_forces=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        failure_flag=(i == 7),
        failure_mode="attitude" if i == 7 else None,
    )


def test_roundtrip_small(tmp_path: Path) -> None:
    p = tmp_path / "traj.parquet"
    with TrajectoryLogger(p, row_group_size=4) as log:
        for i in range(10):
            log.append(_make_step(i))
    assert p.exists() and p.stat().st_size > 0

    table = pq.read_table(p)
    assert table.num_rows == 10
    # step column is monotonically increasing
    steps = table.column("step").to_pylist()
    assert steps == list(range(10))
    # failure_flag set only at step 7
    flags = table.column("failure_flag").to_pylist()
    assert flags[7] is True
    assert sum(flags) == 1
    # base_pos x-component equals 0.1 * step
    xs = [row[0] for row in table.column("base_pos").to_pylist()]
    assert np.allclose(xs, [0.1 * i for i in range(10)], atol=1e-6)


def test_rows_written_property(tmp_path: Path) -> None:
    log = TrajectoryLogger(tmp_path / "t.parquet", row_group_size=3)
    for i in range(5):
        log.append(_make_step(i))
    assert log.rows_written == 5
    log.close()
    assert log.rows_written == 5


# ---------------------------------------------------------------------------
# Provenance columns. A consumer must be able to tell sim from hardware, and
# raw foot-force counts from calibrated Newtons, from the schema alone.
# ---------------------------------------------------------------------------


def test_provenance_defaults_are_unknown_not_guessed(tmp_path: Path) -> None:
    p = tmp_path / "defaults.parquet"
    with TrajectoryLogger(p) as log:
        log.append(_make_step(0))
    table = pq.read_table(p)
    assert table.column("capture_source").to_pylist() == [CAPTURE_SOURCE_UNKNOWN]
    assert table.column("base_lin_vel_source").to_pylist() == ["unknown"]
    assert table.column("obs_base_lin_vel_source").to_pylist() == ["unknown"]
    assert table.column("contact_forces_units").to_pylist() == ["unknown"]


def test_provenance_roundtrips(tmp_path: Path) -> None:
    p = tmp_path / "labelled.parquet"
    step = _make_step(0)
    step.capture_source = CAPTURE_SOURCE_HARDWARE
    step.base_lin_vel_source = "odom:body_passthrough"
    step.obs_base_lin_vel_source = "zeros:operator_selected"
    step.contact_forces_units = "raw_counts_uncalibrated"
    with TrajectoryLogger(p) as log:
        log.append(step)
    table = pq.read_table(p)
    assert table.column("capture_source").to_pylist() == ["hardware"]
    assert table.column("base_lin_vel_source").to_pylist() == ["odom:body_passthrough"]
    assert table.column("obs_base_lin_vel_source").to_pylist() == ["zeros:operator_selected"]
    assert table.column("contact_forces_units").to_pylist() == ["raw_counts_uncalibrated"]


def test_schema_carries_every_provenance_column() -> None:
    names = set(_SCHEMA.names)
    assert {
        "capture_source",
        "base_lin_vel_source",
        "obs_base_lin_vel_source",
        "contact_forces_units",
    } <= names
    # odom_valid stays: it is per-row validity, not provenance.
    assert "odom_valid" in names


# ---------------------------------------------------------------------------
# Simulator ground truth vs detector opinion. failure_flag/failure_mode are the
# detector's verdict; the sim_termination_* columns are what actually happened.
# The detector misses genuine falls, so a consumer must be able to tell
# "detector said not this mode" from "detector said nothing at all".
# ---------------------------------------------------------------------------


def test_hardware_shaped_row_leaves_simulator_columns_null(tmp_path: Path) -> None:
    p = tmp_path / "hardware.parquet"
    step = _make_step(0)
    step.capture_source = CAPTURE_SOURCE_HARDWARE
    step.failure_onset_source = "detector"
    with TrajectoryLogger(p) as log:
        log.append(step)
    table = pq.read_table(p)
    assert table.column("sim_termination_terms").to_pylist() == [None]
    assert table.column("sim_termination_index").to_pylist() == [None]
    assert table.column("sim_termination_time_s").to_pylist() == [None]
    assert table.column("failure_onset_source").to_pylist() == ["detector"]


def test_all_four_simulator_columns_default_to_null(tmp_path: Path) -> None:
    """Every pre-existing writer keeps working without naming these."""
    p = tmp_path / "untaught_writer.parquet"
    with TrajectoryLogger(p) as log:
        log.append(_make_step(0))
    table = pq.read_table(p)
    for name in (
        "sim_termination_terms",
        "sim_termination_index",
        "sim_termination_time_s",
        "failure_onset_source",
    ):
        assert table.column(name).to_pylist() == [None], name


def test_sim_shaped_row_preserves_the_simulator_columns(tmp_path: Path) -> None:
    p = tmp_path / "sim.parquet"
    step = _make_step(0)
    step.capture_source = "sim"
    step.sim_termination_terms = "base_contact|time_out"
    step.sim_termination_index = 417
    step.sim_termination_time_s = 8.34
    step.failure_onset_source = "simulator_termination"
    with TrajectoryLogger(p) as log:
        log.append(step)
    table = pq.read_table(p)
    assert table.column("sim_termination_terms").to_pylist() == ["base_contact|time_out"]
    assert table.column("sim_termination_index").to_pylist() == [417]
    assert table.column("sim_termination_time_s").to_pylist() == [pytest.approx(8.34)]
    assert table.column("failure_onset_source").to_pylist() == ["simulator_termination"]


def test_a_detector_miss_is_distinguishable_from_a_nominal_row(tmp_path: Path) -> None:
    """The exact case that motivated these columns.

    Row 0: the simulator terminated on base_contact and the detector said
    nothing. Row 1: nominal. Both have failure_flag=False and failure_mode
    None, so only the simulator columns separate them.
    """
    p = tmp_path / "miss.parquet"
    missed = _make_step(0)
    missed.capture_source = "sim"
    missed.failure_flag = False
    missed.failure_mode = None
    missed.sim_termination_terms = "base_contact"
    missed.sim_termination_index = 0
    missed.sim_termination_time_s = 0.0
    missed.failure_onset_source = "detector"

    nominal = _make_step(1)
    nominal.capture_source = "sim"
    nominal.failure_onset_source = "detector"

    with TrajectoryLogger(p) as log:
        log.append(missed)
        log.append(nominal)

    table = pq.read_table(p)
    assert table.column("failure_flag").to_pylist() == [False, False]
    assert table.column("failure_mode").to_pylist() == [None, None]
    assert table.column("sim_termination_terms").to_pylist() == ["base_contact", None]


def test_reader_round_trips_a_row_with_and_without_simulator_columns(tmp_path: Path) -> None:
    """The replay reader must tolerate the widened schema either way."""
    from phoenix.replay.trajectory_reader import TrajectoryReader, load_initial_state

    p = tmp_path / "roundtrip.parquet"
    hardware = _make_step(0)
    hardware.capture_source = CAPTURE_SOURCE_HARDWARE
    hardware.failure_onset_source = "detector"
    sim = _make_step(1)
    sim.capture_source = "sim"
    sim.sim_termination_terms = "base_contact"
    sim.sim_termination_index = 1
    sim.sim_termination_time_s = 0.02
    sim.failure_onset_source = "simulator_termination"
    with TrajectoryLogger(p) as log:
        log.append(hardware)
        log.append(sim)

    reader = TrajectoryReader(p)
    assert len(reader) == 2
    # Row 0 is the hardware row (null), row 1 the sim row, in append order.
    assert list(reader.column("sim_termination_terms")) == [None, "base_contact"]
    assert list(reader.column("failure_onset_source")) == [
        "detector",
        "simulator_termination",
    ]
    state = load_initial_state(p, row=0)
    assert state.base_quat.shape == (4,)
