"""Tests for the rule-based real-robot failure detector."""

from __future__ import annotations

import numpy as np

from phoenix.real_world.failure_detector import (
    FailureDetector,
    FailureMode,
    FailureThresholds,
)


def test_attitude_failure_fires_immediately() -> None:
    det = FailureDetector()
    ev = det.step(
        timestamp_s=0.0,
        pitch_rad=1.0,  # > 0.8
        roll_rad=0.0,
        base_height_m=0.3,
        cmd_lin_vel=np.zeros(2),
        actual_lin_vel=np.zeros(2),
    )
    assert ev is not None and ev.mode == FailureMode.ATTITUDE


def test_collapse_failure() -> None:
    det = FailureDetector()
    ev = det.step(
        timestamp_s=0.0,
        pitch_rad=0.0,
        roll_rad=0.0,
        base_height_m=0.10,  # < 0.15
        cmd_lin_vel=np.zeros(2),
        actual_lin_vel=np.zeros(2),
    )
    assert ev is not None and ev.mode == FailureMode.COLLAPSE


def test_slip_requires_sustained_discrepancy() -> None:
    det = FailureDetector(FailureThresholds(slip_min_duration_s=0.5, min_event_gap_s=0.0))
    # First step: slipping starts — no event yet.
    ev = det.step(
        timestamp_s=0.0,
        pitch_rad=0.0,
        roll_rad=0.0,
        base_height_m=0.3,
        cmd_lin_vel=np.asarray([0.5, 0.0]),
        actual_lin_vel=np.asarray([0.0, 0.0]),
    )
    assert ev is None

    # Still slipping, but only 0.2s in — still no event.
    ev = det.step(
        timestamp_s=0.2,
        pitch_rad=0.0,
        roll_rad=0.0,
        base_height_m=0.3,
        cmd_lin_vel=np.asarray([0.5, 0.0]),
        actual_lin_vel=np.asarray([0.0, 0.0]),
    )
    assert ev is None

    # 0.6s → above min_duration — event fires.
    ev = det.step(
        timestamp_s=0.6,
        pitch_rad=0.0,
        roll_rad=0.0,
        base_height_m=0.3,
        cmd_lin_vel=np.asarray([0.5, 0.0]),
        actual_lin_vel=np.asarray([0.0, 0.0]),
    )
    assert ev is not None and ev.mode == FailureMode.SLIP


def test_slip_resets_when_commanded_speed_drops() -> None:
    det = FailureDetector(FailureThresholds(slip_min_duration_s=0.5, min_event_gap_s=0.0))
    det.step(
        timestamp_s=0.0,
        pitch_rad=0.0,
        roll_rad=0.0,
        base_height_m=0.3,
        cmd_lin_vel=np.asarray([0.5, 0.0]),
        actual_lin_vel=np.asarray([0.0, 0.0]),
    )
    # Command drops to zero → slip timer must reset.
    det.step(
        timestamp_s=0.3,
        pitch_rad=0.0,
        roll_rad=0.0,
        base_height_m=0.3,
        cmd_lin_vel=np.zeros(2),
        actual_lin_vel=np.zeros(2),
    )
    # Even if the robot slips again at 0.6s, it has been less than 0.5s since
    # slip resumed, so no event should fire.
    ev = det.step(
        timestamp_s=0.6,
        pitch_rad=0.0,
        roll_rad=0.0,
        base_height_m=0.3,
        cmd_lin_vel=np.asarray([0.5, 0.0]),
        actual_lin_vel=np.asarray([0.0, 0.0]),
    )
    assert ev is None


def test_min_event_gap_suppresses_duplicates() -> None:
    det = FailureDetector(FailureThresholds(min_event_gap_s=1.0))
    ev1 = det.step(
        timestamp_s=0.0,
        pitch_rad=1.0,
        roll_rad=0.0,
        base_height_m=0.3,
        cmd_lin_vel=np.zeros(2),
        actual_lin_vel=np.zeros(2),
    )
    ev2 = det.step(
        timestamp_s=0.5,
        pitch_rad=1.0,
        roll_rad=0.0,
        base_height_m=0.3,
        cmd_lin_vel=np.zeros(2),
        actual_lin_vel=np.zeros(2),
    )
    assert ev1 is not None
    assert ev2 is None  # suppressed by gap
