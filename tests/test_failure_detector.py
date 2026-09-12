"""Tests for the rule-based real-robot failure detector.

Covers each mode's trip condition, its DETECTION LATENCY (the interval from
true onset to the emitted event), and the fail-loud behaviour on missing or
non-finite sensor input. Latency matters because these events are what the
replay pipeline seeds from: a mode that reports the wrong onset seeds the
wrong state.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.real_world.failure_detector import (
    MODE_DEFINITIONS,
    FailureDetector,
    FailureMode,
    FailureThresholds,
)

DT = 0.02  # 50 Hz control period


def _nominal(**overrides) -> dict:
    sample = dict(
        timestamp_s=0.0,
        pitch_rad=0.0,
        roll_rad=0.0,
        base_height_m=0.3,
        cmd_lin_vel=np.zeros(2),
        actual_lin_vel=np.zeros(2),
    )
    sample.update(overrides)
    return sample


# --------------------------------------------------------------------------
# Trip conditions
# --------------------------------------------------------------------------


def test_attitude_failure_fires_immediately() -> None:
    det = FailureDetector()
    ev = det.step(**_nominal(pitch_rad=1.0))  # > 0.8
    assert ev is not None and ev.mode == FailureMode.ATTITUDE


def test_attitude_roll_threshold_is_tighter_than_pitch() -> None:
    det = FailureDetector()
    # 0.7 rad trips roll (0.6) but would not trip pitch (0.8).
    assert det.step(**_nominal(roll_rad=0.7)) is not None
    assert FailureDetector().step(**_nominal(pitch_rad=0.7)) is None


def test_collapse_failure() -> None:
    det = FailureDetector()
    ev = det.step(**_nominal(base_height_m=0.10))  # < 0.15
    assert ev is not None and ev.mode == FailureMode.COLLAPSE


def test_collapse_is_unavailable_when_no_height_source_exists() -> None:
    """``base_height_m=None`` is the real-robot case: no ground-relative height.

    Regression for the hardware-labelling bug: ``/utlidar/robot_odom``'s z is
    boot-pose relative, so it was feeding the collapse threshold a number that
    is not a height. The detector must report nothing rather than guess.
    """
    det = FailureDetector()
    for i in range(200):
        ev = det.step(**_nominal(timestamp_s=i * DT, base_height_m=None))
        assert ev is None


def test_slip_requires_sustained_discrepancy() -> None:
    det = FailureDetector(FailureThresholds(slip_min_duration_s=0.5, min_event_gap_s=0.0))
    stalled = dict(cmd_lin_vel=np.asarray([0.5, 0.0]), actual_lin_vel=np.zeros(2))
    assert det.step(**_nominal(timestamp_s=0.0, **stalled)) is None
    assert det.step(**_nominal(timestamp_s=0.2, **stalled)) is None  # only 0.2s in
    ev = det.step(**_nominal(timestamp_s=0.6, **stalled))
    assert ev is not None and ev.mode == FailureMode.SLIP


def test_slip_resets_when_commanded_speed_drops() -> None:
    det = FailureDetector(FailureThresholds(slip_min_duration_s=0.5, min_event_gap_s=0.0))
    det.step(**_nominal(timestamp_s=0.0, cmd_lin_vel=np.asarray([0.5, 0.0])))
    det.step(**_nominal(timestamp_s=0.3))  # command drops to zero, timer resets
    ev = det.step(**_nominal(timestamp_s=0.6, cmd_lin_vel=np.asarray([0.5, 0.0])))
    assert ev is None  # the new stall episode is only 0.0s old


def test_min_event_gap_suppresses_duplicates() -> None:
    det = FailureDetector(FailureThresholds(min_event_gap_s=1.0))
    assert det.step(**_nominal(timestamp_s=0.0, pitch_rad=1.0)) is not None
    assert det.step(**_nominal(timestamp_s=0.5, pitch_rad=1.0)) is None


def test_min_event_gap_is_global_not_per_mode() -> None:
    """Locks the documented limitation: the gap suppresses OTHER modes too."""
    det = FailureDetector(FailureThresholds(min_event_gap_s=1.0))
    assert det.step(**_nominal(timestamp_s=0.0, pitch_rad=1.0)) is not None
    # A genuine collapse 0.5s later is dropped, by design; replay seeds from
    # the first event of an episode either way.
    assert det.step(**_nominal(timestamp_s=0.5, base_height_m=0.05)) is None
    assert det.step(**_nominal(timestamp_s=1.5, base_height_m=0.05)) is not None


# --------------------------------------------------------------------------
# Detection latency
# --------------------------------------------------------------------------


def test_attitude_latency_is_one_sample() -> None:
    """Sweep pitch across the threshold at 50 Hz and measure onset -> event."""
    det = FailureDetector()
    thresholds = FailureThresholds()
    onset_t: float | None = None
    fired_t: float | None = None
    for i in range(100):
        t = i * DT
        pitch = 0.02 * i  # crosses 0.8 rad at i = 40
        if onset_t is None and abs(pitch) > thresholds.pitch_rad:
            onset_t = t
        ev = det.step(**_nominal(timestamp_s=t, pitch_rad=pitch))
        if ev is not None:
            fired_t = t
            assert ev.detail["detection_latency_s"] == pytest.approx(0.0)
            break
    assert onset_t is not None and fired_t is not None
    assert fired_t - onset_t == pytest.approx(0.0, abs=1e-9)


def test_collapse_latency_is_one_sample() -> None:
    det = FailureDetector()
    thresholds = FailureThresholds()
    onset_t = fired_t = None
    for i in range(100):
        t = i * DT
        height = 0.35 - 0.01 * i  # crosses 0.15 m at i = 21
        if onset_t is None and height < thresholds.base_height_min_m:
            onset_t = t
        ev = det.step(**_nominal(timestamp_s=t, base_height_m=height))
        if ev is not None:
            fired_t = t
            break
    assert onset_t is not None and fired_t is not None
    assert fired_t - onset_t == pytest.approx(0.0, abs=1e-9)


def test_slip_latency_equals_the_required_duration() -> None:
    """A stall is reported ``slip_min_duration_s`` after it truly began."""
    det = FailureDetector(FailureThresholds(slip_min_duration_s=0.5, min_event_gap_s=0.0))
    onset_t = 0.4  # the stall starts here
    fired_t = None
    for i in range(100):
        t = i * DT
        stalling = t >= onset_t
        ev = det.step(
            **_nominal(
                timestamp_s=t,
                cmd_lin_vel=np.asarray([0.5, 0.0]) if stalling else np.zeros(2),
                actual_lin_vel=np.zeros(2),
            )
        )
        if ev is not None:
            fired_t = t
            assert ev.mode == FailureMode.SLIP
            assert ev.detail["onset_timestamp_s"] == pytest.approx(onset_t)
            assert ev.detail["detection_latency_s"] == pytest.approx(0.5, abs=DT)
            break
    assert fired_t is not None
    # One sample period of slack: the check happens on control ticks.
    assert fired_t - onset_t == pytest.approx(0.5, abs=DT)


def test_suppressed_ticks_do_not_freeze_the_slip_onset() -> None:
    """Regression: the duplicate-event gate used to skip the slip bookkeeping.

    A stall beginning inside the suppression window inherited the stale onset
    and fired instantly, reporting a detection latency it had not earned.
    """
    det = FailureDetector(FailureThresholds(slip_min_duration_s=0.5, min_event_gap_s=1.0))
    stalled = dict(cmd_lin_vel=np.asarray([0.5, 0.0]), actual_lin_vel=np.zeros(2))

    # A stall starts at t=0 and an attitude event fires at t=0.1, opening a
    # 1.0s suppression window.
    det.step(**_nominal(timestamp_s=0.0, **stalled))
    assert det.step(**_nominal(timestamp_s=0.1, pitch_rad=1.0, **stalled)) is not None
    # The robot recovers: no stall during the window.
    for i in range(2, 60):
        det.step(**_nominal(timestamp_s=i * DT))
    # A NEW stall starts just after the window closes.
    ev = det.step(**_nominal(timestamp_s=1.2, **stalled))
    assert ev is None, "a freshly started stall must not fire on its first sample"
    ev = det.step(**_nominal(timestamp_s=1.8, **stalled))
    assert ev is not None and ev.mode == FailureMode.SLIP
    assert ev.detail["onset_timestamp_s"] == pytest.approx(1.2)
    assert ev.detail["detection_latency_s"] == pytest.approx(0.6)


# --------------------------------------------------------------------------
# Missing / non-finite sensor inputs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["timestamp_s", "pitch_rad", "roll_rad", "base_height_m"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_scalar_input_raises(field, bad) -> None:
    """A NaN satisfies no comparison, so it would read as 'no failure'."""
    det = FailureDetector()
    with pytest.raises(ValueError, match=field):
        det.step(**_nominal(**{field: bad}))


@pytest.mark.parametrize("field", ["cmd_lin_vel", "actual_lin_vel"])
def test_non_finite_velocity_input_raises(field) -> None:
    det = FailureDetector()
    with pytest.raises(ValueError, match=field):
        det.step(**_nominal(**{field: np.asarray([float("nan"), 0.0])}))


@pytest.mark.parametrize("field", ["cmd_lin_vel", "actual_lin_vel"])
def test_missing_velocity_components_raise(field) -> None:
    det = FailureDetector()
    with pytest.raises(ValueError, match="at least 2 components"):
        det.step(**_nominal(**{field: np.asarray([0.5])}))
    with pytest.raises(ValueError, match="at least 2 components"):
        det.step(**_nominal(**{field: None}))


def test_base_height_none_is_allowed_but_nan_is_not() -> None:
    det = FailureDetector()
    assert det.step(**_nominal(base_height_m=None)) is None
    with pytest.raises(ValueError):
        det.step(**_nominal(base_height_m=float("nan")))


# --------------------------------------------------------------------------
# Documented contract
# --------------------------------------------------------------------------


def test_every_mode_has_a_written_definition() -> None:
    assert set(MODE_DEFINITIONS) == set(FailureMode)
    for mode, text in MODE_DEFINITIONS.items():
        assert len(text) > 20, mode


def test_slip_definition_does_not_claim_a_contact_signal() -> None:
    """The docstring used to claim slip needed unstable foot contact.

    ``step()`` takes no contact argument and never has. Lock the corrected
    definition so the claim cannot come back without the implementation.
    """
    import inspect

    from phoenix.real_world import failure_detector as mod

    sig = inspect.signature(FailureDetector.step)
    assert "contact" not in " ".join(sig.parameters)
    doc = mod.__doc__ or ""
    assert "NO CONTACT SIGNAL IS CONSULTED" in doc
    assert "raw_counts" not in MODE_DEFINITIONS[FailureMode.SLIP]
    assert "no contact signal is consulted" in MODE_DEFINITIONS[FailureMode.SLIP]


def test_module_documents_the_simulator_ontology_mismatch() -> None:
    from phoenix.real_world import failure_detector as mod

    doc = mod.__doc__ or ""
    assert "base_contact" in doc
    assert "time_out" in doc
    assert "NOT THE SIMULATOR'S TERMINATION ONTOLOGY" in doc
