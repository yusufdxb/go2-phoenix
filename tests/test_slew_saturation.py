"""Tests for the slew metric in ``phoenix.training.slew``.

Two definitions live in that module:

* :func:`slew_clip_activation_rate`, the deploy-equivalent one. Deployment
  builds ``target = default_q + action_scale * action`` and clips it against
  the MEASURED joint position with
  :func:`phoenix.sim2real.safety.per_step_clip_array`. This metric asks
  whether that clip would actually alter the target.
* :func:`legacy_raw_action_delta_saturation_rate`, the original definition
  that compared RAW ACTION deltas against the same 0.175 rad bound. It is a
  different quantity in different units and is retained only to reproduce
  numbers recorded before 2026-09-11.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real.safety import MAX_DELTA_PER_STEP_RAD, per_step_clip_array
from phoenix.training.slew import (
    legacy_raw_action_delta_saturation_rate,
    slew_clip_activation_rate,
)

DEFAULT_Q = np.array([0.1, -0.3, 0.8, 0.0], dtype=np.float32)
SCALE = 0.25


# --------------------------------------------------------------------------
# Deploy-equivalent metric
# --------------------------------------------------------------------------
def test_target_on_top_of_measured_q_never_clips() -> None:
    actions = np.zeros((4, 4), dtype=np.float32)
    measured_q = np.broadcast_to(DEFAULT_Q, (4, 4)).astype(np.float32)
    assert (
        slew_clip_activation_rate(
            actions=actions,
            measured_q=measured_q,
            default_q=DEFAULT_Q,
            action_scale=SCALE,
        )
        == 0.0
    )


def test_every_target_beyond_the_cap_clips() -> None:
    # action = 1.0, scale = 0.25 puts every target 0.25 rad from default_q,
    # and measured_q sits at default_q, so every motor exceeds 0.175.
    actions = np.ones((3, 4), dtype=np.float32)
    measured_q = np.broadcast_to(DEFAULT_Q, (3, 4)).astype(np.float32)
    assert (
        slew_clip_activation_rate(
            actions=actions,
            measured_q=measured_q,
            default_q=DEFAULT_Q,
            action_scale=SCALE,
        )
        == 1.0
    )


def test_exactly_at_the_cap_is_not_counted() -> None:
    """per_step_clip_array returns the target unchanged at exactly max_delta.

    The legacy metric counted ``>= threshold`` as saturated. The deploy clip
    does not act there, so the deploy-equivalent metric must not count it.
    """
    default_q = np.zeros(1, dtype=np.float32)
    actions = np.array([[MAX_DELTA_PER_STEP_RAD]], dtype=np.float32)
    measured_q = np.zeros((1, 1), dtype=np.float32)
    rate = slew_clip_activation_rate(
        actions=actions,
        measured_q=measured_q,
        default_q=default_q,
        action_scale=1.0,
    )
    assert rate == 0.0
    # ... and one ULP past it is counted.
    actions_over = np.array([[np.nextafter(MAX_DELTA_PER_STEP_RAD, 1.0)]], dtype=np.float64)
    assert (
        slew_clip_activation_rate(
            actions=actions_over,
            measured_q=measured_q.astype(np.float64),
            default_q=default_q.astype(np.float64),
            action_scale=1.0,
        )
        == 1.0
    )


def test_raw_action_delta_is_not_the_deploy_metric() -> None:
    """Regression for the bug this replaced.

    A 0.2 raw-action step saturates 100% under the legacy definition, while
    the deploy clip never fires, because 0.2 * action_scale = 0.05 rad of
    joint target and the robot tracks its target.
    """
    prev_actions = np.zeros((2, 4), dtype=np.float32)
    actions = np.full((2, 4), 0.2, dtype=np.float32)
    measured_q = (DEFAULT_Q + SCALE * actions).astype(np.float32)
    legacy = legacy_raw_action_delta_saturation_rate(
        prev_actions, actions, threshold=MAX_DELTA_PER_STEP_RAD
    )
    deploy = slew_clip_activation_rate(
        actions=actions,
        measured_q=measured_q,
        default_q=DEFAULT_Q,
        action_scale=SCALE,
    )
    assert legacy == 1.0
    assert deploy == 0.0


def test_mixed_fraction_counts_only_the_clipped_motors() -> None:
    default_q = np.zeros(4, dtype=np.float32)
    measured_q = np.zeros((2, 4), dtype=np.float32)
    actions = np.zeros((2, 4), dtype=np.float32)
    actions[0, :2] = 1.0  # 2 of 8 samples move 0.25 rad from q
    rate = slew_clip_activation_rate(
        actions=actions,
        measured_q=measured_q,
        default_q=default_q,
        action_scale=SCALE,
    )
    assert rate == pytest.approx(2 / 8)


def test_matches_the_shared_deploy_helper_on_random_data() -> None:
    """The metric must agree with per_step_clip_array itself, not a copy."""
    rng = np.random.default_rng(20260911)
    actions = rng.normal(0.0, 1.5, size=(16, 12)).astype(np.float32)
    measured_q = rng.normal(0.0, 0.5, size=(16, 12)).astype(np.float32)
    default_q = rng.normal(0.0, 0.5, size=12).astype(np.float32)
    target = default_q + SCALE * actions
    expected = float(
        np.mean(per_step_clip_array(target, measured_q, MAX_DELTA_PER_STEP_RAD) != target)
    )
    assert 0.0 < expected < 1.0  # the fixture must actually exercise both branches
    assert (
        slew_clip_activation_rate(
            actions=actions,
            measured_q=measured_q,
            default_q=default_q,
            action_scale=SCALE,
        )
        == expected
    )


def test_per_env_default_q_is_accepted() -> None:
    default_q = np.tile(DEFAULT_Q, (3, 1))
    actions = np.zeros((3, 4), dtype=np.float32)
    measured_q = default_q.astype(np.float32)
    assert (
        slew_clip_activation_rate(
            actions=actions,
            measured_q=measured_q,
            default_q=default_q,
            action_scale=SCALE,
        )
        == 0.0
    )


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        slew_clip_activation_rate(
            actions=np.zeros((4, 12), dtype=np.float32),
            measured_q=np.zeros((3, 12), dtype=np.float32),
            default_q=np.zeros(12, dtype=np.float32),
            action_scale=SCALE,
        )


def test_bad_default_q_shape_raises() -> None:
    with pytest.raises(ValueError, match="default_q shape"):
        slew_clip_activation_rate(
            actions=np.zeros((4, 12), dtype=np.float32),
            measured_q=np.zeros((4, 12), dtype=np.float32),
            default_q=np.zeros(11, dtype=np.float32),
            action_scale=SCALE,
        )


def test_non_finite_raises() -> None:
    actions = np.zeros((1, 2), dtype=np.float32)
    actions[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        slew_clip_activation_rate(
            actions=actions,
            measured_q=np.zeros((1, 2), dtype=np.float32),
            default_q=np.zeros(2, dtype=np.float32),
            action_scale=SCALE,
        )


def test_max_delta_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_delta"):
        slew_clip_activation_rate(
            actions=np.zeros((1, 1), dtype=np.float32),
            measured_q=np.zeros((1, 1), dtype=np.float32),
            default_q=np.zeros(1, dtype=np.float32),
            action_scale=SCALE,
            max_delta=0.0,
        )


def test_agrees_with_the_reliability_ablation_definition() -> None:
    """Two deploy-equivalent implementations exist in this repo; pin them together.

    ``phoenix.reliability.deploy_ablation.deploy_slew_saturation`` open-codes
    ``|target - q| > max_delta`` instead of calling the shared clip helper. The
    two must not disagree, so this test fails if either one drifts.
    """
    from phoenix.reliability.deploy_ablation import deploy_slew_saturation

    rng = np.random.default_rng(4)
    actions = rng.normal(0.0, 1.5, size=(64, 12))
    measured_q = rng.normal(0.0, 0.5, size=(64, 12))
    default_q = rng.normal(0.0, 0.5, size=12)
    theirs = deploy_slew_saturation(
        actions, measured_q, default_q, SCALE, MAX_DELTA_PER_STEP_RAD
    )
    mine = slew_clip_activation_rate(
        actions=actions,
        measured_q=measured_q,
        default_q=default_q,
        action_scale=SCALE,
        max_delta=MAX_DELTA_PER_STEP_RAD,
    )
    assert 0.0 < theirs < 1.0
    assert mine == theirs


# --------------------------------------------------------------------------
# Legacy metric: still reproducible, but no longer reachable under a name
# that suggests it is deploy-equivalent.
# --------------------------------------------------------------------------
def test_legacy_name_is_gone() -> None:
    import phoenix.training.slew as slew_mod

    assert not hasattr(slew_mod, "slew_saturation_rate")


def test_legacy_zero_delta_returns_zero() -> None:
    prev = np.zeros((4, 12), dtype=np.float32)
    curr = np.zeros((4, 12), dtype=np.float32)
    assert legacy_raw_action_delta_saturation_rate(prev, curr, threshold=0.175) == 0.0


def test_legacy_exact_threshold_counts_as_saturated() -> None:
    prev = np.zeros((1, 1), dtype=np.float32)
    curr = np.full((1, 1), 0.175, dtype=np.float32)
    # Historical spec: >= threshold is saturated. Preserved so old numbers
    # reproduce bit-for-bit.
    assert legacy_raw_action_delta_saturation_rate(prev, curr, threshold=0.175) == 1.0


def test_legacy_mixed_fraction_matches_expectation() -> None:
    prev = np.zeros((4, 12), dtype=np.float32)
    curr = np.zeros((4, 12), dtype=np.float32)
    curr[:3, :4] = 0.2  # 12 of 48 samples saturate
    rate = legacy_raw_action_delta_saturation_rate(prev, curr, threshold=0.175)
    assert rate == pytest.approx(12 / 48)


def test_legacy_negative_deltas_counted_by_magnitude() -> None:
    prev = np.zeros((1, 2), dtype=np.float32)
    curr = np.array([[-0.2, 0.1]], dtype=np.float32)
    assert legacy_raw_action_delta_saturation_rate(prev, curr, threshold=0.175) == 0.5


def test_legacy_shape_mismatch_raises() -> None:
    prev = np.zeros((4, 12), dtype=np.float32)
    curr = np.zeros((3, 12), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        legacy_raw_action_delta_saturation_rate(prev, curr, threshold=0.175)


def test_legacy_threshold_must_be_positive() -> None:
    prev = np.zeros((1, 1), dtype=np.float32)
    curr = np.zeros((1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="threshold"):
        legacy_raw_action_delta_saturation_rate(prev, curr, threshold=0.0)
