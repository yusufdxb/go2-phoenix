"""Unit tests for the reliability-layer OOD scorers (Phase 1 core).

These exercise the contract the Simplex arbiter and eval harness rely on:
scorers fit on nominal-only data assign higher scores to out-of-distribution
features, stay finite/invertible in the high-dim low-sample regime, and fail
toward SAFE (+inf) on non-finite input.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.reliability.ood_monitor import (
    KNNScorer,
    MahalanobisScorer,
    TemporalFilter,
    ledoit_wolf_shrinkage,
)


def _nominal(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Anisotropic nominal cloud so a naive Euclidean score would be wrong.
    scale = np.linspace(0.5, 3.0, d)
    return rng.standard_normal((n, d)) * scale


# --- Ledoit-Wolf shrinkage --------------------------------------------------


def test_shrinkage_intensity_in_unit_interval():
    x = _nominal(50, 20)
    cov, a = ledoit_wolf_shrinkage(x)
    assert 0.0 <= a <= 1.0
    assert cov.shape == (20, 20)
    # Symmetric and positive definite (Cholesky succeeds).
    np.testing.assert_allclose(cov, cov.T, atol=1e-9)
    np.linalg.cholesky(cov)


def test_shrinkage_keeps_covariance_invertible_when_dim_near_n():
    # 24 samples, 20 dims: empirical covariance is nearly singular; shrinkage
    # must pull it to something Cholesky can factor.
    x = _nominal(24, 20, seed=3)
    cov, a = ledoit_wolf_shrinkage(x)
    assert a > 0.0
    np.linalg.cholesky(cov)  # must not raise


# --- Mahalanobis ------------------------------------------------------------


def test_mahalanobis_scores_ood_higher_than_nominal():
    d = 12
    fit = _nominal(2000, d, seed=1)
    scorer = MahalanobisScorer.fit(fit)

    held_in = _nominal(500, d, seed=2)
    shift = np.full(d, 6.0)
    held_ood = _nominal(500, d, seed=2) + shift

    assert scorer.score(held_ood).mean() > 10.0 * scorer.score(held_in).mean()


def test_mahalanobis_nonfinite_scores_inf():
    scorer = MahalanobisScorer.fit(_nominal(500, 6))
    bad = np.zeros(6)
    bad[2] = np.nan
    assert scorer.score_one(bad) == np.inf
    assert np.isinf(scorer.score(np.array([[np.inf] * 6]))[0])


def test_mahalanobis_batch_matches_single():
    scorer = MahalanobisScorer.fit(_nominal(800, 8, seed=5))
    batch = _nominal(10, 8, seed=9)
    batch_scores = scorer.score(batch)
    for i, row in enumerate(batch):
        assert scorer.score_one(row) == pytest.approx(batch_scores[i], rel=1e-9)


# --- KNN --------------------------------------------------------------------


def test_knn_scores_ood_higher_than_nominal():
    d = 16
    fit = _nominal(3000, d, seed=7)
    scorer = KNNScorer.fit(fit, n_components=8, k=5)

    held_in = _nominal(400, d, seed=11)
    held_ood = _nominal(400, d, seed=11) + np.full(d, 5.0)

    assert scorer.score(held_ood).mean() > scorer.score(held_in).mean()


def test_knn_reference_is_subsampled_to_bound():
    scorer = KNNScorer.fit(_nominal(5000, 10), n_components=6, k=5, max_reference=1000)
    assert scorer._reference.shape[0] == 1000


def test_knn_nonfinite_scores_inf():
    scorer = KNNScorer.fit(_nominal(1000, 6), n_components=4, k=3)
    bad = np.zeros(6)
    bad[0] = np.nan
    assert scorer.score_one(bad) == np.inf


# --- Temporal filter --------------------------------------------------------


def test_ewma_smooths_and_cusum_accumulates():
    tf = TemporalFilter(alpha=0.5, drift=1.0)
    # Scores below drift keep CUSUM at zero.
    for _ in range(5):
        _, cusum = tf.update(0.5)
    assert cusum == 0.0
    # A sustained rise above drift accumulates.
    prev = 0.0
    for _ in range(5):
        _, cusum = tf.update(3.0)
        assert cusum >= prev
        prev = cusum
    assert cusum > 0.0


def test_ewma_tracks_between_samples():
    tf = TemporalFilter(alpha=0.5)
    e1, _ = tf.update(0.0)
    e2, _ = tf.update(4.0)
    assert e1 == 0.0
    assert e2 == pytest.approx(2.0)  # 0.5*4 + 0.5*0


def test_temporal_filter_saturates_on_nonfinite():
    tf = TemporalFilter(alpha=0.3, drift=0.0)
    tf.update(1.0)
    ewma, cusum = tf.update(np.inf)
    assert np.isinf(ewma)
    assert np.isinf(cusum)


def test_temporal_filter_rejects_bad_alpha():
    with pytest.raises(ValueError):
        TemporalFilter(alpha=0.0)
    with pytest.raises(ValueError):
        TemporalFilter(alpha=1.5)
