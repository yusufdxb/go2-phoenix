"""Tests for phoenix.training.slew.slew_saturation_rate.

The helper measures the fraction of per-(env, timestep, motor) action
deltas whose absolute value exceeds a threshold, matching the Jetson
dryrun saturation definition in docs/dryrun_findings_2026-04-14.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from phoenix.training.slew import slew_saturation_rate


def test_zero_delta_returns_zero() -> None:
    prev = np.zeros((4, 12), dtype=np.float32)
    curr = np.zeros((4, 12), dtype=np.float32)
    assert slew_saturation_rate(prev, curr, threshold=0.175) == 0.0


def test_all_deltas_above_threshold_returns_one() -> None:
    prev = np.zeros((3, 12), dtype=np.float32)
    curr = np.full((3, 12), 0.2, dtype=np.float32)
    assert slew_saturation_rate(prev, curr, threshold=0.175) == 1.0


def test_exact_threshold_counts_as_saturated() -> None:
    prev = np.zeros((1, 1), dtype=np.float32)
    curr = np.full((1, 1), 0.175, dtype=np.float32)
    # Spec: >= threshold is saturated (matches Jetson clip behavior).
    assert slew_saturation_rate(prev, curr, threshold=0.175) == 1.0


def test_mixed_fraction_matches_expectation() -> None:
    # 12 motors, 4 envs => 48 samples. Make 12 of them saturate.
    prev = np.zeros((4, 12), dtype=np.float32)
    curr = np.zeros((4, 12), dtype=np.float32)
    curr[:3, :4] = 0.2  # 12 samples saturate
    rate = slew_saturation_rate(prev, curr, threshold=0.175)
    assert rate == pytest.approx(12 / 48)


def test_negative_deltas_counted_by_magnitude() -> None:
    prev = np.zeros((1, 2), dtype=np.float32)
    curr = np.array([[-0.2, 0.1]], dtype=np.float32)
    # Only the -0.2 sample saturates.
    assert slew_saturation_rate(prev, curr, threshold=0.175) == 0.5


def test_shape_mismatch_raises() -> None:
    prev = np.zeros((4, 12), dtype=np.float32)
    curr = np.zeros((3, 12), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        slew_saturation_rate(prev, curr, threshold=0.175)


def test_threshold_must_be_positive() -> None:
    prev = np.zeros((1, 1), dtype=np.float32)
    curr = np.zeros((1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="threshold"):
        slew_saturation_rate(prev, curr, threshold=0.0)
