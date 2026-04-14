"""Tests for Halton-based variation sampling."""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.replay.variations import VariationSampler, halton_sequence


def test_halton_in_unit_cube() -> None:
    seq = halton_sequence(32, 4)
    assert seq.shape == (32, 4)
    assert seq.min() >= 0.0 and seq.max() < 1.0


def test_halton_is_deterministic() -> None:
    a = halton_sequence(16, 4)
    b = halton_sequence(16, 4)
    assert np.allclose(a, b)


def test_halton_covers_more_than_random() -> None:
    """Halton should produce lower discrepancy than pseudo-random for small N."""
    halton = halton_sequence(16, 4)
    # A simple coverage proxy: after binning into 4 quantiles per dim, count
    # the fraction of unique cells visited. Halton should be near-1.0.
    bins = (halton * 4).astype(int)
    unique = len({tuple(row) for row in bins})
    assert unique >= 14  # at least 14/16 distinct cells


def test_variation_sampler_respects_bounds() -> None:
    bounds = {
        "friction_delta": [-0.3, 0.3],
        "mass_delta_kg": [-1.0, 1.0],
        "push_velocity_delta": [-0.5, 0.5],
        "push_yaw_delta": [-0.3, 0.3],
    }
    sampler = VariationSampler(bounds=bounds, seed=7)
    samples = sampler.sample(16)
    assert len(samples) == 16
    for s in samples:
        assert -0.3 <= s.friction_delta <= 0.3
        assert -1.0 <= s.mass_delta_kg <= 1.0
        assert -0.5 <= s.push_velocity_delta <= 0.5
        assert -0.3 <= s.push_yaw_delta <= 0.3


def test_variation_sampler_rejects_missing_bound() -> None:
    with pytest.raises(KeyError, match="missing"):
        VariationSampler(bounds={"friction_delta": [-0.1, 0.1]})


def test_variation_sampler_is_reproducible() -> None:
    bounds = {
        "friction_delta": [-0.3, 0.3],
        "mass_delta_kg": [-1.0, 1.0],
        "push_velocity_delta": [-0.5, 0.5],
        "push_yaw_delta": [-0.3, 0.3],
    }
    a = VariationSampler(bounds, seed=17).sample(8)
    b = VariationSampler(bounds, seed=17).sample(8)
    assert a == b
