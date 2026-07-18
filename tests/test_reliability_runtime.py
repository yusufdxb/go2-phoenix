"""End-to-end integration test for the deployable reliability runtime.

Proves the whole layer behaves: on nominal features the shield stays on the
learned policy (blend 0); under sustained out-of-distribution features it
hands off to the fallback (blend ramps to 1). All in numpy, no Isaac / robot.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.reliability.arbiter import ShieldState, SimplexArbiter, SimplexArbiterCfg
from phoenix.reliability.features import policy_features
from phoenix.reliability.ood_monitor import MahalanobisScorer, TemporalFilter
from phoenix.reliability.runtime import (
    ShieldRuntime,
    calibrate_arbiter_thresholds,
)


def _actor(obs_dim, hidden, action_dim, seed=0):
    rng = np.random.default_rng(seed)
    dims = [obs_dim, *hidden, action_dim]
    return [
        (rng.standard_normal((dims[i + 1], dims[i])) * 0.3, rng.standard_normal(dims[i + 1]) * 0.1)
        for i in range(len(dims) - 1)
    ]


def _build_runtime(seed=0):
    layers = _actor(16, [64, 48, 32], 16, seed=seed)
    rng = np.random.default_rng(seed + 1)
    nominal_obs = rng.standard_normal((4000, 16))
    nominal_latent = policy_features(nominal_obs, layers)["latent"]

    scorer = MahalanobisScorer.fit(nominal_latent)
    trip, clear = calibrate_arbiter_thresholds(scorer, nominal_latent, trip_fpr=0.01, clear_fpr=0.2)
    cfg = SimplexArbiterCfg(
        trip_threshold=trip,
        clear_threshold=clear,
        trip_persistence=3,
        clear_persistence=10,
        handoff_ticks=8,
        recover_ticks=20,
        min_fallback_ticks=10,
    )
    runtime = ShieldRuntime(
        scorer,
        SimplexArbiter(cfg),
        feature_key="latent",
        temporal_filter=TemporalFilter(alpha=0.4),
    )
    return runtime, layers


def test_calibration_gives_hysteresis_gap():
    layers = _actor(12, [32], 12, seed=2)
    nominal = policy_features(np.random.default_rng(0).standard_normal((2000, 12)), layers)["latent"]
    scorer = MahalanobisScorer.fit(nominal)
    trip, clear = calibrate_arbiter_thresholds(scorer, nominal, trip_fpr=0.01, clear_fpr=0.2)
    assert trip > clear > 0


def test_calibration_rejects_bad_fpr_ordering():
    layers = _actor(8, [16], 8)
    nominal = policy_features(np.zeros((500, 8)), layers)["latent"]
    scorer = MahalanobisScorer.fit(nominal + np.random.default_rng(0).standard_normal((500, 1)) * 0.1)
    with pytest.raises(ValueError):
        calibrate_arbiter_thresholds(scorer, nominal, trip_fpr=0.3, clear_fpr=0.1)


def test_nominal_stream_keeps_learned_policy_in_control():
    runtime, layers = _build_runtime(seed=5)
    rng = np.random.default_rng(99)
    max_blend = 0.0
    for _ in range(400):
        obs = rng.standard_normal((1, 16))  # in-distribution
        dec = runtime.step(policy_features(obs, layers))
        max_blend = max(max_blend, dec.blend)
    # Calibrated at 1% trip FPR with persistence 3: nominal must not engage.
    assert max_blend == 0.0


def test_sustained_ood_hands_off_to_fallback():
    runtime, layers = _build_runtime(seed=5)
    rng = np.random.default_rng(7)
    final = None
    for _ in range(60):
        obs = rng.standard_normal((1, 16)) + 8.0  # far out-of-distribution
        final = runtime.step(policy_features(obs, layers))
    assert final.state is ShieldState.FALLBACK
    assert final.blend == pytest.approx(1.0)
    assert final.engaged


def test_decision_reports_scores():
    runtime, layers = _build_runtime(seed=3)
    dec = runtime.step(policy_features(np.zeros((1, 16)), layers))
    assert np.isfinite(dec.raw_score)
    assert np.isfinite(dec.filtered_score)
    assert dec.state is ShieldState.NOMINAL


def test_step_rejects_multi_row_batch():
    runtime, layers = _build_runtime(seed=1)
    feats = policy_features(np.zeros((4, 16)), layers)
    with pytest.raises(ValueError):
        runtime.step(feats)


def test_reset_clears_state():
    runtime, layers = _build_runtime(seed=5)
    rng = np.random.default_rng(7)
    for _ in range(60):
        runtime.step(policy_features(rng.standard_normal((1, 16)) + 8.0, layers))
    assert runtime.arbiter.state is ShieldState.FALLBACK
    runtime.reset()
    assert runtime.arbiter.state is ShieldState.NOMINAL
    dec = runtime.step(policy_features(np.zeros((1, 16)), layers))
    assert dec.blend == 0.0
