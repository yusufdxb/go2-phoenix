"""Item 12: uniform file sampling makes the curriculum a function of harvest luck.

``assign`` drew a pool FILE uniformly, so a pool of fourteen slips and one
collapse taught collapse recovery in 1/15 of seeded envs and nothing recorded
that. Stratified sampling draws the stratum first, so mechanism and command
regime are represented evenly whatever the dataset composition happens to be.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phoenix.adaptation.curriculum import (
    DEFAULT_STRATA,
    FailureCurriculum,
    TrajectoryPool,
    describe_trajectory,
)
from tests.test_failure_seed_v2 import capsule


def _write(path: Path, mode: str | None, command=(0.5, 0.0, 0.0), n_rows: int = 6) -> Path:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    schema = pa.schema(
        [
            ("step", pa.int64()),
            ("failure_mode", pa.string()),
            ("failure_flag", pa.bool_()),
            ("command_vel", pa.list_(pa.float32(), 3)),
        ]
    )
    onset = n_rows // 2
    rows = [
        {
            "step": i,
            "failure_mode": mode if i >= onset else None,
            "failure_flag": i >= onset,
            "command_vel": list(command),
        }
        for i in range(n_rows)
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


# -------------------- metadata extraction ----------------------------------


def test_describe_parquet_reads_mode_and_command_bucket(tmp_path):
    described = describe_trajectory(_write(tmp_path / "a.parquet", "slip", (0.8, 0.0, 0.0)))
    assert described.readable
    assert described.failure_mode == "slip"
    assert described.failure_modes == frozenset({"slip"})
    assert described.command_bucket == "fast"


@pytest.mark.parametrize(
    ("command", "bucket"),
    [((0.0, 0.0, 0.0), "stand"), ((0.1, 0.1, 0.0), "stand"), ((0.4, 0.0, 0.0), "slow"),
     ((0.0, 0.5, 0.0), "slow"), ((0.6, 0.0, 0.0), "fast"), ((1.0, 1.0, 0.0), "fast")],
)
def test_command_buckets_have_explicit_edges(tmp_path, command, bucket):
    path = _write(tmp_path / f"c_{bucket}_{command[0]}_{command[1]}.parquet", "slip", command)
    assert describe_trajectory(path).command_bucket == bucket


def test_capsule_metadata_supplies_severity_and_scenario(tmp_path):
    path = capsule(
        tmp_path, failure_mode="collapse", severity="high", scenario_id="s-42"
    )
    described = describe_trajectory(path)
    assert described.readable
    assert described.failure_mode == "collapse"
    assert described.severity == "high"
    assert described.scenario_id == "s-42"
    assert described.command_bucket == "fast"


def test_unreadable_file_is_unlabelled_not_fatal(tmp_path):
    (tmp_path / "broken.parquet").write_bytes(b"stub")
    described = describe_trajectory(tmp_path / "broken.parquet")
    assert described.readable is False
    assert described.failure_mode is None
    assert described.command_bucket is None


# -------------------- sampling ---------------------------------------------


def _unbalanced_pool(tmp_path) -> TrajectoryPool:
    paths = [_write(tmp_path / f"slip_{i}.parquet", "slip") for i in range(14)]
    paths.append(_write(tmp_path / "collapse_0.parquet", "collapse"))
    return TrajectoryPool(paths)


def test_uniform_legacy_mirrors_dataset_composition(tmp_path):
    pool = _unbalanced_pool(tmp_path)
    curriculum = FailureCurriculum(pool, failure_reset_fraction=1.0, seed=0)
    modes = [pool.metadata()[i].failure_mode for i in curriculum.assign(2000)]
    collapse = modes.count("collapse") / len(modes)
    assert 0.03 < collapse < 0.10  # about 1/15, i.e. whatever the harvest yielded


def test_stratified_balances_the_mechanism_regardless_of_file_counts(tmp_path):
    pool = _unbalanced_pool(tmp_path)
    curriculum = FailureCurriculum(
        pool, failure_reset_fraction=1.0, seed=0, sampling="stratified"
    )
    modes = [pool.metadata()[i].failure_mode for i in curriculum.assign(2000)]
    collapse = modes.count("collapse") / len(modes)
    assert 0.45 < collapse < 0.55


def test_strata_can_be_narrowed_and_are_reported(tmp_path):
    pool = _unbalanced_pool(tmp_path)
    curriculum = FailureCurriculum(
        pool, failure_reset_fraction=0.5, seed=3, sampling="stratified", strata=("failure_mode",)
    )
    assert curriculum.describe_strata() == {"collapse": 1, "slip": 14}
    assert curriculum.strata == ("failure_mode",)
    counts = FailureCurriculum(pool, failure_reset_fraction=0.5).describe_strata()
    assert counts == {"collapse|slow": 1, "slip|slow": 14}
    assert DEFAULT_STRATA == ("failure_mode", "command_bucket")


def test_stratified_over_unlabelled_files_still_assigns(tmp_path):
    for i in range(3):
        (tmp_path / f"stub_{i}.parquet").write_bytes(b"stub")
    pool = TrajectoryPool.from_directory(tmp_path)
    curriculum = FailureCurriculum(
        pool, failure_reset_fraction=0.5, seed=1, sampling="stratified"
    )
    assert curriculum.describe_strata() == {"unlabelled|unlabelled": 3}
    assignment = curriculum.assign(16)
    assert int((assignment >= 0).sum()) == 8
    assert assignment[assignment >= 0].max() < len(pool)


def test_stratified_assignment_is_reproducible_and_seed_sensitive(tmp_path):
    pool = _unbalanced_pool(tmp_path)

    def draw(seed):
        return FailureCurriculum(
            pool, failure_reset_fraction=0.5, seed=seed, sampling="stratified"
        ).assign(64)

    assert np.array_equal(draw(42), draw(42))
    assert not np.array_equal(draw(42), draw(7))


def test_invalid_sampling_configuration_rejected(tmp_path):
    pool = _unbalanced_pool(tmp_path)
    with pytest.raises(ValueError, match="Unknown sampling"):
        FailureCurriculum(pool, failure_reset_fraction=0.5, sampling="balanced")
    with pytest.raises(ValueError, match="Unknown strata"):
        FailureCurriculum(pool, failure_reset_fraction=0.5, strata=("weather",))
    with pytest.raises(ValueError, match="at least one stratum"):
        FailureCurriculum(pool, failure_reset_fraction=0.5, sampling="stratified", strata=())


def test_legacy_uniform_mode_is_labelled_legacy(caplog):
    pool = TrajectoryPool([Path("nonexistent.parquet")])
    with caplog.at_level("WARNING"):
        FailureCurriculum(pool, failure_reset_fraction=0.5)
    assert "uniform_legacy" in caplog.text
    assert "stratified" in caplog.text
