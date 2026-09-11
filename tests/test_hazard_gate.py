"""Tests for the H0 gate v2 decision layer.

Pure logic, no simulator. The rollout driver is exercised separately; what is
pinned here is that the three criteria stay separated and that the specific
methodological defect in gate v1 cannot reappear.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phoenix.reliability.hazard_gate import (
    GATE_VERSION,
    HazardEstimate,
    assess_delivery,
    classify_seed,
    newcombe_difference,
    tilt_deg_from_quat_xyzw,
    wilson_interval,
)

IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]  # xyzw


def quat_from_pitch(pitch_rad: float) -> list[float]:
    return [0.0, math.sin(pitch_rad / 2), 0.0, math.cos(pitch_rad / 2)]


def nominal_delivery(**overrides):
    """A state identical to an ordinary reset unless overridden."""
    kwargs = {
        "base_height_m": 0.4,
        "base_quat_xyzw": IDENTITY_QUAT,
        "joint_vel": np.zeros(12),
        "base_lin_vel_body": np.zeros(3),
        "base_ang_vel_body": np.zeros(3),
    }
    kwargs.update(overrides)
    return assess_delivery(**kwargs)


# --------------------------------------------------------------------------
# Wilson and Newcombe intervals
# --------------------------------------------------------------------------


def test_wilson_interval_brackets_the_point_estimate() -> None:
    lo, hi = wilson_interval(30, 100)
    assert lo < 0.30 < hi


@pytest.mark.parametrize("k,n", [(0, 20), (20, 20), (0, 1), (1, 1)])
def test_wilson_interval_stays_inside_the_unit_interval_at_the_boundaries(k, n) -> None:
    """The normal approximation leaves [0, 1] here; Wilson must not."""
    lo, hi = wilson_interval(k, n)
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_interval_narrows_as_n_grows() -> None:
    wide = wilson_interval(5, 10)
    narrow = wilson_interval(500, 1000)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_interval_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        wilson_interval(5, 3)


def test_newcombe_difference_excludes_zero_for_a_clear_separation() -> None:
    diff, lo, hi = newcombe_difference(HazardEstimate(100, 80), HazardEstimate(100, 20))
    assert diff == pytest.approx(0.60)
    assert lo > 0.0 and hi > lo


def test_newcombe_difference_includes_zero_for_identical_arms() -> None:
    _, lo, hi = newcombe_difference(HazardEstimate(50, 25), HazardEstimate(50, 25))
    assert lo <= 0.0 <= hi


def test_hazard_estimate_rejects_more_failures_than_trials() -> None:
    with pytest.raises(ValueError):
        HazardEstimate(n=10, n_fail=11)


# --------------------------------------------------------------------------
# Criterion A: delivery, against the NOMINAL RESET, not the trajectory history
# --------------------------------------------------------------------------


def test_a_state_equal_to_an_ordinary_reset_is_not_delivered() -> None:
    assert nominal_delivery().delivered is False


def test_root_velocity_alone_is_not_delivery() -> None:
    """Nominal reset already randomizes root velocity over +/-0.5 per axis."""
    verdict = nominal_delivery(
        base_lin_vel_body=np.array([0.4, -0.3, 0.2]),
        base_ang_vel_body=np.array([0.1, 0.4, -0.2]),
    )
    assert verdict.delivered is False
    assert verdict.root_vel_norm > 0.0


def test_lowered_height_is_delivery() -> None:
    verdict = nominal_delivery(base_height_m=0.30)
    assert verdict.delivered is True
    assert verdict.height_delta_m == pytest.approx(-0.10)


def test_tilt_is_delivery() -> None:
    verdict = nominal_delivery(base_quat_xyzw=quat_from_pitch(0.5))
    assert verdict.delivered is True
    assert verdict.tilt_deg == pytest.approx(math.degrees(0.5), abs=1e-6)


def test_nonzero_joint_velocity_is_delivery() -> None:
    assert nominal_delivery(joint_vel=np.full(12, 2.0)).delivered is True


def test_tilt_is_invariant_to_yaw() -> None:
    yaws = [tilt_deg_from_quat_xyzw([0.0, 0.0, math.sin(y / 2), math.cos(y / 2)]) for y in
            np.linspace(-math.pi, math.pi, 25)]
    assert max(yaws) - min(yaws) < 1e-9


def test_tilt_rejects_a_degenerate_quaternion() -> None:
    with pytest.raises(ValueError):
        tilt_deg_from_quat_xyzw([0.0, 0.0, 0.0, 0.0])


# --------------------------------------------------------------------------
# Criteria B and C, and the v1 defect that must not reappear
# --------------------------------------------------------------------------


def test_an_undelivered_state_is_rejected_regardless_of_its_rollouts() -> None:
    """Fail-closed ordering: no rollout result can rescue a non-treatment."""
    v = classify_seed(
        delivery=nominal_delivery(),
        seeded=HazardEstimate(100, 90),
        baseline=HazardEstimate(100, 5),
    )
    assert v.verdict == "NOT_DELIVERED"
    assert v.usable_seed is False


def test_an_already_doomed_state_is_rejected_even_with_huge_elevation() -> None:
    """A near-certain failure has no counterfactual left to learn from."""
    v = classify_seed(
        delivery=nominal_delivery(base_height_m=0.12),
        seeded=HazardEstimate(100, 100),
        baseline=HazardEstimate(100, 5),
    )
    assert v.verdict == "ALREADY_DOOMED"
    assert v.usable_seed is False


def test_a_delivered_state_with_no_elevated_hazard_is_not_a_precursor() -> None:
    v = classify_seed(
        delivery=nominal_delivery(base_height_m=0.30),
        seeded=HazardEstimate(200, 42),
        baseline=HazardEstimate(200, 40),
    )
    assert v.verdict == "NOT_ELEVATED"


def test_the_recoverable_hazard_band_accepts_elevated_but_escapable() -> None:
    v = classify_seed(
        delivery=nominal_delivery(base_height_m=0.30),
        seeded=HazardEstimate(200, 120),
        baseline=HazardEstimate(200, 20),
    )
    assert v.verdict == "RECOVERABLE_HAZARD"
    assert v.usable_seed is True
    assert v.elevation == pytest.approx(0.50)
    assert v.elevation_ci[0] > 0.0


def test_gate_v2_does_not_penalise_seeding_earlier_and_healthier() -> None:
    """The exact defect in gate v1.

    v1 anchored on the trajectory's own history, so a seed taken further before
    onset looked "wrong direction" precisely because the robot was healthier
    there. Under v2 a healthier-looking precursor is judged on whether it raises
    future failure probability, so an earlier, less extreme, still-hazardous
    state is accepted rather than scored as a failure.
    """
    early = classify_seed(
        # Mildly departed from an ordinary reset: healthier looking.
        delivery=nominal_delivery(base_height_m=0.35),
        seeded=HazardEstimate(200, 90),
        baseline=HazardEstimate(200, 20),
    )
    late = classify_seed(
        # Deep into the failure.
        delivery=nominal_delivery(base_height_m=0.12),
        seeded=HazardEstimate(200, 180),
        baseline=HazardEstimate(200, 20),
    )
    assert early.verdict == "RECOVERABLE_HAZARD"
    assert early.usable_seed is True
    assert late.verdict == "RECOVERABLE_HAZARD"
    # And the pathological end of that same axis is still rejected.
    doomed = classify_seed(
        delivery=nominal_delivery(base_height_m=0.08),
        seeded=HazardEstimate(200, 199),
        baseline=HazardEstimate(200, 20),
    )
    assert doomed.verdict == "ALREADY_DOOMED"


def test_missing_rollouts_are_reported_not_silently_passed() -> None:
    v = classify_seed(
        delivery=nominal_delivery(base_height_m=0.30),
        seeded=HazardEstimate(0, 0),
        baseline=HazardEstimate(0, 0),
    )
    assert v.verdict == "INSUFFICIENT_SAMPLES"
    assert v.usable_seed is False


def test_every_verdict_carries_the_gate_version() -> None:
    """A recorded result must never be mistakable for a v1 result."""
    v = classify_seed(
        delivery=nominal_delivery(base_height_m=0.30),
        seeded=HazardEstimate(100, 60),
        baseline=HazardEstimate(100, 10),
    )
    assert v.gate_version == GATE_VERSION == "h0-gate-v2"
