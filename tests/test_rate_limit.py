"""Tests for the in-training per-step rate limiter (CI-safe, numpy only).

The Isaac Lab ActionTerm wrapper (``phoenix.sim_env.rate_limited_action``) is
sim-gated and not exercised here; what matters for correctness is that the pure
clamp it calls is identical to the deploy-side limiter so sim and hardware agree.
"""

from __future__ import annotations

import numpy as np

from phoenix.sim2real.safety import MAX_DELTA_PER_STEP_RAD, per_step_clip_array
from phoenix.sim_env.rate_limit import MAX_DELTA_PER_STEP_RAD as SIM_MAX_DELTA
from phoenix.sim_env.rate_limit import rate_limit_targets


def test_constant_is_shared_with_deploy():
    """Sim and deploy must use the exact same cap, or the whole fix is moot."""
    assert SIM_MAX_DELTA == MAX_DELTA_PER_STEP_RAD


def test_matches_deploy_clip_elementwise():
    """rate_limit_targets must equal the deploy per_step_clip_array everywhere."""
    rng = np.random.default_rng(0)
    q = rng.uniform(-1.5, 1.5, size=(8, 12))
    # Targets that range from within the cap to far outside it.
    target = q + rng.uniform(-0.5, 0.5, size=(8, 12))
    got = rate_limit_targets(target, q, MAX_DELTA_PER_STEP_RAD)
    want = per_step_clip_array(target, q, MAX_DELTA_PER_STEP_RAD)
    np.testing.assert_allclose(got, want)


def test_clips_upward_delta():
    q = np.array([0.0, 0.0, 0.0])
    target = np.array([0.5, 0.1, -0.5])  # 0.5 and -0.5 exceed the 0.175 cap
    out = rate_limit_targets(target, q, 0.175)
    np.testing.assert_allclose(out, [0.175, 0.1, -0.175])


def test_within_cap_is_passthrough():
    q = np.array([0.63, 1.0, -1.5])
    target = np.array([0.70, 0.95, -1.55])  # all deltas <= 0.175
    out = rate_limit_targets(target, q, 0.175)
    np.testing.assert_allclose(out, target)


def test_rear_thigh_startup_jump_is_capped():
    """The 0.37 rad default-vs-settled gap must be limited to one cap step."""
    settled_q = 0.63
    default_target = 1.0  # use_default_offset puts the first target at the default
    out = float(rate_limit_targets(np.array([default_target]), np.array([settled_q]), 0.175)[0])
    assert abs(out - settled_q) <= 0.175 + 1e-9
    assert out == settled_q + 0.175  # clipped, not the full 0.37 jump
