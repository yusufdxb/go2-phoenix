"""Tests for the per-step episode-telemetry artifact.

The artifact exists to be read by a separate repository's failure-mode
detector, so these tests cover the schema contract (version stamping, units,
fail-closed validation) and the grouping helper that turns flat rows back
into per-episode signal blocks.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from phoenix.training.episode_telemetry import (
    ARROW_SCHEMA,
    REQUIRED_COLUMNS,
    SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
    UNITS,
    UNITS_KEY,
    EpisodeTelemetrySchemaError,
    EpisodeTelemetryWriter,
    TelemetryStep,
    load_episode_signals,
    quat_wxyz_to_euler,
    table_units,
    validate_table,
    write_episode_telemetry,
)


def make_step(
    step_index: int,
    *,
    run_id: str = "run_a",
    seed: int = 7,
    episode_id: int = 0,
    pitch: float = 0.0,
    roll: float = 0.0,
    height: float = 0.32,
    cmd_x: float = 0.5,
    act_x: float = 0.5,
) -> TelemetryStep:
    return TelemetryStep(
        run_id=run_id,
        episode_id=episode_id,
        seed=seed,
        env_index=1,
        step_index=step_index,
        t_s=step_index * 0.02,
        control_dt_s=0.02,
        pitch_rad=pitch,
        roll_rad=roll,
        yaw_rad=0.0,
        base_height_m=height,
        base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel_x_mps=cmd_x,
        cmd_lin_vel_y_mps=0.0,
        cmd_ang_vel_z_radps=0.0,
        actual_lin_vel_x_mps=act_x,
        actual_lin_vel_y_mps=0.0,
        actual_ang_vel_z_radps=0.0,
        joint_vel_radps=np.zeros(12, dtype=np.float32),
        contact_forces_n=np.full(4, 20.0, dtype=np.float32),
    )


# --- schema -----------------------------------------------------------------


def test_units_map_is_total_over_columns():
    assert set(UNITS) == set(REQUIRED_COLUMNS)


def test_round_trip_preserves_values(tmp_path):
    path = tmp_path / "tel.parquet"
    write_episode_telemetry(path, [make_step(i, pitch=0.01 * i) for i in range(5)])
    table = pq.read_table(path)
    assert table.num_rows == 5
    validate_table(table, source=str(path))
    assert table.column("pitch_rad").to_pylist() == pytest.approx(
        [0.0, 0.01, 0.02, 0.03, 0.04]
    )


def test_schema_version_in_metadata_and_column(tmp_path):
    path = tmp_path / "tel.parquet"
    write_episode_telemetry(path, [make_step(0)])
    table = pq.read_table(path)
    assert table.schema.metadata[SCHEMA_VERSION_KEY].decode() == SCHEMA_VERSION
    assert set(table.column("schema_version").to_pylist()) == {SCHEMA_VERSION}


def test_units_stamped_into_metadata(tmp_path):
    path = tmp_path / "tel.parquet"
    write_episode_telemetry(path, [make_step(0)])
    table = pq.read_table(path)
    assert json.loads(table.schema.metadata[UNITS_KEY].decode()) == UNITS
    assert table_units(table)["base_height_m"] == "meters"


def test_writer_flushes_multiple_row_groups(tmp_path):
    path = tmp_path / "tel.parquet"
    with EpisodeTelemetryWriter(path, row_group_size=4) as writer:
        for i in range(10):
            writer.append(make_step(i))
        assert writer.rows_written == 10
    assert pq.read_table(path).num_rows == 10


def test_context_manager_flushes_on_exception(tmp_path):
    path = tmp_path / "tel.parquet"
    with pytest.raises(RuntimeError):
        with EpisodeTelemetryWriter(path, row_group_size=100) as writer:
            writer.append(make_step(0))
            raise RuntimeError("rollout blew up")
    assert pq.read_table(path).num_rows == 1


# --- fail-closed validation -------------------------------------------------


def test_missing_file_raises(tmp_path):
    with pytest.raises(EpisodeTelemetrySchemaError, match="does not exist"):
        load_episode_signals(tmp_path / "absent.parquet")


def test_directory_raises(tmp_path):
    with pytest.raises(EpisodeTelemetrySchemaError, match="is a directory"):
        load_episode_signals(tmp_path)


def test_non_parquet_raises(tmp_path):
    path = tmp_path / "junk.parquet"
    path.write_text("not parquet")
    with pytest.raises(EpisodeTelemetrySchemaError, match="not a readable"):
        load_episode_signals(path)


def test_zero_rows_raises(tmp_path):
    path = tmp_path / "empty.parquet"
    pq.write_table(ARROW_SCHEMA.empty_table(), path)
    with pytest.raises(EpisodeTelemetrySchemaError, match="zero telemetry rows"):
        load_episode_signals(path)


def test_version_mismatch_raises(tmp_path):
    path = tmp_path / "bad_version.parquet"
    write_episode_telemetry(path, [make_step(0)])
    table = pq.read_table(path)
    bumped = table.set_column(
        table.schema.get_field_index("schema_version"),
        "schema_version",
        pa.array(["9.9.9"], type=pa.string()),
    )
    pq.write_table(bumped, path)
    with pytest.raises(EpisodeTelemetrySchemaError, match="schema version mismatch"):
        load_episode_signals(path)


def test_missing_column_raises(tmp_path):
    path = tmp_path / "short.parquet"
    write_episode_telemetry(path, [make_step(0)])
    table = pq.read_table(path)
    dropped = table.drop(["base_height_m"])
    pq.write_table(dropped, path)
    with pytest.raises(EpisodeTelemetrySchemaError, match="missing required column"):
        load_episode_signals(path)


def test_null_in_required_column_raises(tmp_path):
    path = tmp_path / "nulled.parquet"
    write_episode_telemetry(path, [make_step(0)])
    table = pq.read_table(path)
    nulled = table.set_column(
        table.schema.get_field_index("base_height_m"),
        "base_height_m",
        pa.array([None], type=pa.float64()),
    )
    pq.write_table(nulled, path)
    with pytest.raises(EpisodeTelemetrySchemaError, match="null value"):
        load_episode_signals(path)


def test_missing_units_raises(tmp_path):
    path = tmp_path / "nounits.parquet"
    write_episode_telemetry(path, [make_step(0)])
    table = pq.read_table(path)
    stripped = table.replace_schema_metadata(
        {SCHEMA_VERSION_KEY: SCHEMA_VERSION.encode()}
    )
    pq.write_table(stripped, path)
    with pytest.raises(EpisodeTelemetrySchemaError, match="units map missing"):
        load_episode_signals(path)


# --- grouping ---------------------------------------------------------------


def test_grouping_splits_by_episode_key(tmp_path):
    path = tmp_path / "tel.parquet"
    steps = [make_step(i, episode_id=0) for i in range(3)]
    steps += [make_step(i, episode_id=1) for i in range(4)]
    steps += [make_step(i, episode_id=0, run_id="run_b") for i in range(2)]
    write_episode_telemetry(path, steps)
    signals = load_episode_signals(path)
    assert set(signals) == {("run_a", 7, 0), ("run_a", 7, 1), ("run_b", 7, 0)}
    assert len(signals[("run_a", 7, 1)]["pitch_rad"]) == 4


def test_grouping_sorts_by_step_index(tmp_path):
    path = tmp_path / "tel.parquet"
    steps = [make_step(i, pitch=float(i)) for i in (3, 0, 2, 1)]
    write_episode_telemetry(path, steps)
    signals = load_episode_signals(path)
    assert signals[("run_a", 7, 0)]["pitch_rad"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_grouped_signal_shapes(tmp_path):
    path = tmp_path / "tel.parquet"
    write_episode_telemetry(path, [make_step(i) for i in range(6)])
    sig = load_episode_signals(path)[("run_a", 7, 0)]
    assert sig["dt_s"] == pytest.approx(0.02)
    assert sig["cmd_lin_vel"].shape == (6, 2)
    assert sig["actual_lin_vel"].shape == (6, 2)
    assert sig["joint_vel"].shape == (6, 12)
    assert sig["contact_forces"].shape == (6, 4)


# --- attitude convention ----------------------------------------------------


def test_identity_quaternion_is_level():
    roll, pitch, yaw = quat_wxyz_to_euler([1.0, 0.0, 0.0, 0.0])
    assert (roll, pitch, yaw) == pytest.approx((0.0, 0.0, 0.0))


def test_quaternion_is_read_as_wxyz_not_xyzw():
    # 90 degrees about x, expressed wxyz. Read as xyzw this would be a yaw.
    half = np.sqrt(0.5)
    roll, pitch, yaw = quat_wxyz_to_euler([half, half, 0.0, 0.0])
    assert roll == pytest.approx(np.pi / 2)
    assert yaw == pytest.approx(0.0)


def test_pitch_is_clamped_not_raising():
    # A slightly non-unit quaternion can push the arcsin argument past 1.
    roll, pitch, yaw = quat_wxyz_to_euler([0.7072, 0.0, -0.7072, 0.0])
    assert pitch == pytest.approx(-np.pi / 2, abs=1e-3)


def test_bad_quaternion_length_raises():
    with pytest.raises(ValueError, match="4 components"):
        quat_wxyz_to_euler([1.0, 0.0, 0.0])


# --- cross-repo contract ----------------------------------------------------


def test_signals_construct_the_downstream_detector_container(tmp_path):
    """The grouped dict must be directly constructible as Ashfall's container.

    Skipped when the consumer repo is not importable. When it is, this is the
    real cross-repo contract check: field names, shapes and dtypes all have to
    line up with no translation layer.
    """
    recurrence = pytest.importorskip("ashfall.analysis.recurrence")
    path = tmp_path / "tel.parquet"
    # A clear attitude failure: pitch past the detector's 0.8 rad threshold.
    write_episode_telemetry(
        path, [make_step(i, pitch=1.2 if i > 10 else 0.0) for i in range(30)]
    )
    signals = load_episode_signals(path)[("run_a", 7, 0)]
    telemetry = recurrence.EpisodeTelemetry(**signals)
    assert recurrence.missing_signals(telemetry) == []
    mode, reason = recurrence.detect_episode_mode(telemetry)
    assert mode == "attitude", (mode, reason)
