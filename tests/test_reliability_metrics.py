"""Unit tests for the reliability-layer threshold-free metrics + lead time."""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.reliability.metrics import (
    average_precision,
    bootstrap_ci,
    fpr_at_tpr,
    lead_time_seconds,
    roc_auc,
    threshold_at_fpr,
)

# --- AUROC ------------------------------------------------------------------


def test_auroc_perfect_separation():
    scores = np.array([0.0, 0.1, 0.2, 1.0, 1.1, 1.2])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert roc_auc(scores, labels) == pytest.approx(1.0)


def test_auroc_inverted_is_zero():
    scores = np.array([1.2, 1.1, 1.0, 0.2, 0.1, 0.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert roc_auc(scores, labels) == pytest.approx(0.0)


def test_auroc_chance_on_ties():
    # All identical scores -> every pair tied -> AUC exactly 0.5.
    scores = np.ones(8)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    assert roc_auc(scores, labels) == pytest.approx(0.5)


def test_auroc_known_value():
    # pos={3}, neg={1,2,4}: pos outranks 1 and 2 but not 4 -> 2/3.
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.array([0, 0, 1, 0])
    assert roc_auc(scores, labels) == pytest.approx(2.0 / 3.0)


# --- Average precision ------------------------------------------------------


def test_average_precision_perfect():
    scores = np.array([0.1, 0.2, 0.9, 1.0])
    labels = np.array([0, 0, 1, 1])
    assert average_precision(scores, labels) == pytest.approx(1.0)


def test_average_precision_known_value():
    # Ranked desc: labels [1,0,1,0]. AP = (1/1)*1 + (2/3)*1 averaged over
    # the two positives = (1.0 + 0.6667) / 2.
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([1, 0, 1, 0])
    assert average_precision(scores, labels) == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


# --- FPR@TPR and nominal-only threshold -------------------------------------


def test_fpr_at_tpr_perfect_is_zero():
    scores = np.concatenate([np.zeros(100), np.ones(100) + 5])
    labels = np.concatenate([np.zeros(100), np.ones(100)])
    assert fpr_at_tpr(scores, labels, 0.95) == pytest.approx(0.0)


def test_threshold_at_fpr_uses_nominal_only():
    rng = np.random.default_rng(0)
    nominal = rng.standard_normal(10000)
    thr = threshold_at_fpr(nominal, target_fpr=0.05)
    # ~5% of nominal should exceed the calibrated threshold.
    assert np.mean(nominal >= thr) == pytest.approx(0.05, abs=0.01)


# --- Bootstrap CI -----------------------------------------------------------


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(2)
    scores = np.concatenate([rng.standard_normal(300), rng.standard_normal(300) + 2.0])
    labels = np.concatenate([np.zeros(300), np.ones(300)])
    point, lo, hi = bootstrap_ci(roc_auc, scores, labels, n_boot=300, seed=1)
    assert lo <= point <= hi
    assert 0.5 < point < 1.0
    assert hi - lo < 0.3  # a real separation gives a reasonably tight CI


# --- Lead time --------------------------------------------------------------


def test_lead_time_positive_when_monitor_fires_early():
    # Score crosses threshold at step 40; failure oracle fires at step 50.
    scores = np.zeros(60)
    scores[40:] = 10.0
    lt = lead_time_seconds(scores, threshold=5.0, failure_step=50, dt=0.02)
    assert lt == pytest.approx((50 - 40) * 0.02)


def test_lead_time_none_when_never_fires():
    scores = np.zeros(60)
    lt = lead_time_seconds(scores, threshold=5.0, failure_step=50, dt=0.02)
    assert lt is None


def test_lead_time_nonpositive_when_late():
    scores = np.zeros(60)
    scores[55:] = 10.0  # fires after failure at 50
    lt = lead_time_seconds(scores, threshold=5.0, failure_step=50, dt=0.02)
    assert lt is not None and lt <= 0.0


def test_lead_time_requires_consecutive_run():
    scores = np.zeros(60)
    scores[30] = 10.0  # single spike, should be ignored with min_consecutive=3
    scores[45:] = 10.0
    lt = lead_time_seconds(
        scores, threshold=5.0, failure_step=50, dt=0.02, min_consecutive=3
    )
    assert lt == pytest.approx((50 - 45) * 0.02)
