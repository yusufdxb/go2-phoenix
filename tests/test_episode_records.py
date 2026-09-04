"""Unit tests for the per-episode evaluation record artifact."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from phoenix.training.episode_records import (
    ARROW_SCHEMA,
    NO_FAILURE,
    REQUIRED_COLUMNS,
    SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
    UNITS,
    UNITS_KEY,
    EpisodeRecord,
    EpisodeRecordSchemaError,
    build_episode_record,
    load_episode_records,
    records_to_table,
    sha256_file,
    table_units,
    validate_table,
    write_episode_records,
)


def make_record(**overrides) -> EpisodeRecord:
    base = dict(
        run_id="run-a",
        episode_id=0,
        seed=1234,
        env_index=3,
        terrain_id="rough",
        challenge_id="ff0p5",
        success=False,
        termination_reason="failure",
        time_to_failure_steps=120,
        time_to_failure_s=2.4,
        episode_return=41.5,
        episode_length_steps=120,
        episode_length_s=2.4,
        mean_lin_vel_error_mps=0.21,
        max_lin_vel_error_mps=0.77,
        mean_ang_vel_error_radps=0.11,
        max_ang_vel_error_radps=0.42,
        control_dt_s=0.02,
        policy_path="/ckpt/model_799.pt",
        policy_sha256="ab" * 32,
    )
    base.update(overrides)
    return EpisodeRecord(**base)


class TestRoundTrip:
    def test_schema_round_trip(self, tmp_path):
        records = [make_record(episode_id=i, env_index=i) for i in range(4)]
        out = write_episode_records(tmp_path / "episodes.parquet", records)
        loaded = load_episode_records(out)
        assert loaded == records

    def test_all_required_columns_present(self, tmp_path):
        out = write_episode_records(tmp_path / "e.parquet", [make_record()])
        table = pq.read_table(out)
        for name in REQUIRED_COLUMNS:
            assert name in table.schema.names

    def test_units_preserved(self, tmp_path):
        out = write_episode_records(tmp_path / "e.parquet", [make_record()])
        table = pq.read_table(out)
        assert table_units(table) == UNITS
        assert UNITS["mean_lin_vel_error_mps"] == "m/s"
        assert UNITS["time_to_failure_s"] == "seconds"
        assert UNITS["episode_length_steps"] == "env_steps"
        # The units map is total over the schema.
        assert set(UNITS) == set(REQUIRED_COLUMNS)

    def test_units_missing_raises(self, tmp_path):
        table = records_to_table([make_record()])
        stripped = table.replace_schema_metadata({SCHEMA_VERSION_KEY: SCHEMA_VERSION.encode()})
        path = tmp_path / "nounits.parquet"
        pq.write_table(stripped, path)
        with pytest.raises(EpisodeRecordSchemaError, match="units map missing"):
            load_episode_records(path)

    def test_seed_propagates_into_record(self, tmp_path):
        records = [make_record(episode_id=i, seed=7 + i) for i in range(3)]
        out = write_episode_records(tmp_path / "e.parquet", records)
        assert [r.seed for r in load_episode_records(out)] == [7, 8, 9]


class TestFailClosed:
    def test_version_mismatch_raises(self, tmp_path):
        table = records_to_table([make_record()])
        bad = table.set_column(
            table.schema.get_field_index("schema_version"),
            "schema_version",
            pa.array(["0.0.1"], type=pa.string()),
        )
        path = tmp_path / "old.parquet"
        pq.write_table(bad, path)
        with pytest.raises(EpisodeRecordSchemaError, match="schema version mismatch"):
            load_episode_records(path)

    def test_metadata_version_mismatch_raises(self, tmp_path):
        table = records_to_table([make_record()])
        bad = table.replace_schema_metadata(
            {
                SCHEMA_VERSION_KEY: b"9.9.9",
                UNITS_KEY: json.dumps(UNITS, sort_keys=True).encode(),
            }
        )
        path = tmp_path / "meta.parquet"
        pq.write_table(bad, path)
        with pytest.raises(EpisodeRecordSchemaError, match="schema version mismatch"):
            load_episode_records(path)

    def test_missing_column_raises(self, tmp_path):
        table = records_to_table([make_record()])
        dropped = table.drop(["seed"])
        path = tmp_path / "noseed.parquet"
        pq.write_table(dropped, path)
        with pytest.raises(EpisodeRecordSchemaError, match="missing required column"):
            load_episode_records(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(EpisodeRecordSchemaError, match="does not exist"):
            load_episode_records(tmp_path / "absent.parquet")

    def test_null_version_row_raises(self):
        table = records_to_table([make_record()])
        bad = table.set_column(
            table.schema.get_field_index("schema_version"),
            "schema_version",
            pa.array([None], type=pa.string()),
        )
        with pytest.raises(EpisodeRecordSchemaError, match="null schema_version"):
            validate_table(bad)

    def test_valid_table_passes(self):
        validate_table(records_to_table([make_record()]))


class TestBuildRecord:
    def _build(self, **overrides):
        kwargs = dict(
            run_id="run-b",
            episode_id=2,
            seed=99,
            env_index=5,
            terrain_id="slippery",
            challenge_id="ff0p0",
            timed_out=False,
            termination_reason="failure",
            episode_return=12.0,
            episode_length_steps=50,
            control_dt_s=0.02,
            lin_err_sum_mps=5.0,
            lin_err_max_mps=0.9,
            ang_err_sum_radps=2.5,
            ang_err_max_radps=0.4,
            tracking_steps=50,
            policy_path="/ckpt/a.pt",
            policy_sha256="cd" * 32,
        )
        kwargs.update(overrides)
        return build_episode_record(**kwargs)

    def test_failed_episode_time_to_failure(self):
        rec = self._build()
        assert rec.success is False
        assert rec.time_to_failure_steps == 50
        assert rec.time_to_failure_s == pytest.approx(1.0)
        assert rec.episode_length_s == pytest.approx(1.0)

    def test_successful_episode_uses_no_failure_sentinel(self):
        rec = self._build(timed_out=True, termination_reason="time_out")
        assert rec.success is True
        assert rec.time_to_failure_steps == NO_FAILURE
        assert rec.time_to_failure_s == float(NO_FAILURE)

    def test_mean_errors_divide_by_tracking_steps(self):
        rec = self._build(lin_err_sum_mps=5.0, tracking_steps=50)
        assert rec.mean_lin_vel_error_mps == pytest.approx(0.1)
        assert rec.mean_ang_vel_error_radps == pytest.approx(0.05)

    def test_zero_tracking_steps_does_not_divide_by_zero(self):
        rec = self._build(lin_err_sum_mps=0.0, ang_err_sum_radps=0.0, tracking_steps=0)
        assert rec.mean_lin_vel_error_mps == 0.0

    def test_seed_survives_build(self):
        assert self._build(seed=4242).seed == 4242


class TestProvenance:
    def test_sha256_file_matches_hashlib(self, tmp_path):
        import hashlib

        payload = b"phoenix-checkpoint-bytes" * 100
        path = tmp_path / "ckpt.pt"
        path.write_bytes(payload)
        assert sha256_file(path) == hashlib.sha256(payload).hexdigest()

    def test_schema_metadata_declares_version(self):
        meta = ARROW_SCHEMA.metadata
        assert meta[SCHEMA_VERSION_KEY].decode() == SCHEMA_VERSION
