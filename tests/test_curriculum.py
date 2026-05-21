"""Tests for the failure-curriculum scheduler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phoenix.adaptation.curriculum import FailureCurriculum, TrajectoryPool


def _make_pool(tmp_path: Path, n: int) -> TrajectoryPool:
    for i in range(n):
        (tmp_path / f"t{i}.parquet").write_bytes(b"stub")
    return TrajectoryPool.from_directory(tmp_path)


def _write_mode_parquet(path: Path, mode: str | None, n_rows: int = 5) -> None:
    """Write a tiny parquet whose ``failure_mode`` column is ``mode``."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    schema = pa.schema(
        [
            ("step", pa.int64()),
            ("failure_mode", pa.string()),
        ]
    )
    rows = [{"step": i, "failure_mode": mode} for i in range(n_rows)]
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)


def test_empty_pool_produces_all_minus_one(tmp_path: Path) -> None:
    pool = TrajectoryPool.from_directory(tmp_path)  # empty
    c = FailureCurriculum(pool, failure_fraction=0.5)
    assignment = c.assign(64)
    assert (assignment == -1).all()


def test_zero_fraction_skips_failure_spawns(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, 3)
    c = FailureCurriculum(pool, failure_fraction=0.0)
    assignment = c.assign(32)
    assert (assignment == -1).all()


def test_fraction_respected(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, 3)
    c = FailureCurriculum(pool, failure_fraction=0.25, seed=1)
    a = c.assign(32)
    n_failure = int((a >= 0).sum())
    assert n_failure == 8  # 25% of 32


def test_assigned_indices_are_in_pool(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, 4)
    c = FailureCurriculum(pool, failure_fraction=0.5, seed=0)
    a = c.assign(32)
    failures = a[a >= 0]
    assert failures.max() < len(pool)
    assert failures.min() >= 0


def test_fraction_bounds_rejected() -> None:
    with pytest.raises(ValueError):
        FailureCurriculum(TrajectoryPool(paths=[]), failure_fraction=1.5)


def test_reproducible_assignment(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, 3)
    a1 = FailureCurriculum(pool, failure_fraction=0.5, seed=42).assign(32)
    a2 = FailureCurriculum(pool, failure_fraction=0.5, seed=42).assign(32)
    assert np.array_equal(a1, a2)


def test_different_seeds_yield_different_assignments(tmp_path: Path) -> None:
    """Multi-seed pilots rely on the curriculum RNG actually varying.

    Regression guard for the 2026-05-07 fine_tune patch that threads
    cfg.run.seed into FailureCurriculum. Without distinct draws across
    seeds, the per-env failure-vs-clean reset pattern is held constant
    across cells and one stochastic input is silently shared.
    """
    pool = _make_pool(tmp_path, 4)
    a_42 = FailureCurriculum(pool, failure_fraction=0.5, seed=42).assign(64)
    a_123 = FailureCurriculum(pool, failure_fraction=0.5, seed=123).assign(64)
    a_7 = FailureCurriculum(pool, failure_fraction=0.5, seed=7).assign(64)
    assert not np.array_equal(a_42, a_123)
    assert not np.array_equal(a_42, a_7)
    assert not np.array_equal(a_123, a_7)


def test_failure_modes_filter_none_keeps_all(tmp_path: Path) -> None:
    """``failure_modes=None`` and ``failure_modes=[]`` preserve legacy behavior."""
    _write_mode_parquet(tmp_path / "a_slip.parquet", "slip")
    _write_mode_parquet(tmp_path / "b_attitude.parquet", "attitude")
    _write_mode_parquet(tmp_path / "c_collapse.parquet", "collapse")
    pool_none = TrajectoryPool.from_directory(tmp_path, failure_modes=None)
    pool_empty = TrajectoryPool.from_directory(tmp_path, failure_modes=[])
    assert len(pool_none) == 3
    assert len(pool_empty) == 3


def test_failure_modes_filter_single_mode(tmp_path: Path) -> None:
    """A single-mode whitelist keeps only matching parquets."""
    _write_mode_parquet(tmp_path / "a_slip.parquet", "slip")
    _write_mode_parquet(tmp_path / "b_attitude.parquet", "attitude")
    _write_mode_parquet(tmp_path / "c_slip.parquet", "slip")
    pool = TrajectoryPool.from_directory(tmp_path, failure_modes=["slip"])
    assert len(pool) == 2
    assert all("slip" in p.name for p in pool.paths)


def test_failure_modes_filter_multi_mode(tmp_path: Path) -> None:
    """A multi-mode whitelist keeps any parquet matching any listed mode."""
    _write_mode_parquet(tmp_path / "a_slip.parquet", "slip")
    _write_mode_parquet(tmp_path / "b_attitude.parquet", "attitude")
    _write_mode_parquet(tmp_path / "c_collapse.parquet", "collapse")
    _write_mode_parquet(tmp_path / "d_command_mismatch.parquet", "command_mismatch")
    pool = TrajectoryPool.from_directory(tmp_path, failure_modes=["attitude", "collapse"])
    assert len(pool) == 2
    names = sorted(p.name for p in pool.paths)
    assert names == ["b_attitude.parquet", "c_collapse.parquet"]


def test_failure_modes_filter_excludes_files_with_no_mode(tmp_path: Path) -> None:
    """Parquets whose only ``failure_mode`` rows are null are excluded."""
    _write_mode_parquet(tmp_path / "a_slip.parquet", "slip")
    _write_mode_parquet(tmp_path / "b_unlabeled.parquet", None)
    pool = TrajectoryPool.from_directory(tmp_path, failure_modes=["slip"])
    assert len(pool) == 1
    assert pool.paths[0].name == "a_slip.parquet"


def test_failure_modes_filter_no_match_returns_empty(tmp_path: Path) -> None:
    _write_mode_parquet(tmp_path / "a_slip.parquet", "slip")
    pool = TrajectoryPool.from_directory(tmp_path, failure_modes=["attitude"])
    assert pool.empty()
