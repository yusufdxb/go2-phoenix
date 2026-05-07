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
