"""The one-joint explanatory model: estimator accuracy and the distribution trade-off.

Short episodes and coarse grids keep this under a few seconds; the full study
(``run_study`` with defaults) is recorded in docs/research/TOY_MODEL.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.condition.toy_model import (
    JointParams,
    anchored_dist,
    cost,
    fit_policy,
    gain_ratio_from_errors,
    targeted_dist,
    tracking_errors,
)

FAST = JointParams(duration_s=2.0)
B_GRID = np.linspace(0.0, 0.25, 11)
K_GRID = np.linspace(0.0, 1.2, 4)


@pytest.mark.parametrize("s", [0.5, 0.6, 0.8])
@pytest.mark.parametrize("theta", [(0.1, 0.0), (0.1, 0.8)])
def test_gain_ratio_recovers_authority_in_linear_regime(s, theta):
    e_nom = tracking_errors(theta, 1.0, FAST, seed=1)
    e_now = tracking_errors(theta, s, FAST, seed=2)
    assert gain_ratio_from_errors(e_nom, e_now) == pytest.approx(s, rel=0.05)


def test_gain_ratio_is_one_under_no_change():
    theta = (0.1, 0.5)
    e1 = tracking_errors(theta, 1.0, FAST, seed=1)
    e2 = tracking_errors(theta, 1.0, FAST, seed=2)
    assert gain_ratio_from_errors(e1, e2) == pytest.approx(1.0, abs=0.03)


def test_gain_ratio_nan_on_zero_error():
    assert np.isnan(gain_ratio_from_errors(np.ones(3), np.zeros(3)))


def test_nominal_policy_is_brittle_and_targeted_pays_on_nominal():
    """The trade-off the gate exists for, on the model: not a claim about the GO2."""
    rng = np.random.default_rng(0)
    theta_nom, _ = fit_policy(rng.uniform(0.95, 1.05, 4), FAST, B_GRID, K_GRID)
    theta_tgt, _ = fit_policy(targeted_dist(0.6, rng, 4), FAST, B_GRID, K_GRID)
    at = lambda th, s: np.mean([cost(th, s, FAST, seed=50 + j) for j in range(2)])  # noqa: E731
    assert at(theta_tgt, 0.6) < at(theta_nom, 0.6)
    assert at(theta_tgt, 1.0) > at(theta_nom, 1.0)


def test_anchored_distribution_keeps_nominal_fraction():
    rng = np.random.default_rng(3)
    s = anchored_dist(0.6, rng, 10, nominal_fraction=0.5)
    assert s.shape == (10,)
    assert np.sum(s >= 0.95) == 5
    assert np.all((s[5:] >= 0.5) & (s[5:] <= 0.7))


def test_targeted_distribution_never_exceeds_nominal_authority():
    rng = np.random.default_rng(4)
    assert targeted_dist(0.97, rng, 200).max() <= 1.0
    assert targeted_dist(0.02, rng, 200).min() >= 0.05
