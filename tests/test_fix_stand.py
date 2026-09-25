"""FIX_STAND: deterministic rise from the measured posture, with tracking and abort.

Everything here runs against ``tests/test_fix_stand_plant.Plant`` (lag, load,
latency, frozen joints), not ideal tracking. These are sequencing tests of the
controller logic, NOT hardware validation: the plant is not the GO2.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real.fix_stand import (
    PHASE_ABORTED,
    PHASE_COMPLETE,
    FixStand,
    FixStandError,
    FixStandParams,
    params_from_mapping,
    smoothstep,
    stand_target,
)
from phoenix.sim2real.go2_model import JOINT_POSITION_LIMITS_RAD, limits_in_order

from .test_fix_stand_plant import FOLDED, ORDER, Plant

LO, HI = limits_in_order(ORDER)
STAND = stand_target(ORDER)


def _run(fs: FixStand, plant: Plant, ticks: int):
    out = []
    for _ in range(ticks):
        q, _dq = plant.measured()
        tick = fs.step(q)
        out.append(tick)
        plant.step(tick.target, tick.kp, tick.kd)
    return out


def test_rise_from_the_measured_folded_pose_completes_under_lag_and_load() -> None:
    plant = Plant(FOLDED)
    fs = FixStand(ORDER)
    fs.start(plant.measured()[0])
    ticks = _run(fs, plant, 400)
    assert fs.complete, fs.summary()
    targets = np.asarray([t.target for t in ticks])
    # Starts from the measured posture (clipped onto the limit), not from a plan.
    assert np.allclose(fs.q0, np.clip(FOLDED, LO, HI))
    # Never leaves the hard limits, never steps more than the cap.
    assert np.all(targets >= LO - 1e-12) and np.all(targets <= HI + 1e-12)
    steps = np.abs(np.diff(np.vstack([fs.q0, targets]), axis=0))
    assert steps.max() <= fs.params.max_step_rad + 1e-12
    # Tracking recorded on every tick, and it is genuinely non-ideal.
    assert len(fs.history) == len(ticks)
    errs = np.asarray([h["tracking_error"] for h in fs.history])
    assert np.abs(errs).max() > 0.01
    assert fs.summary()["max_abs_tracking_error_rad"] < fs.params.track_pause_rad
    # The plant ends near the stance with its steady-state load error, not exactly on it.
    q_end, _ = plant.measured()
    assert np.abs(q_end - STAND).max() < 0.10
    assert np.abs(q_end - STAND).max() > 0.01


def test_duration_respects_the_reference_speed_cap() -> None:
    fs = FixStand(ORDER)
    fs.start(FOLDED)
    excursion = np.abs(STAND - fs.q0).max()
    assert fs.duration_s == pytest.approx(max(2.0, 1.5 * excursion / 0.8))
    refs = np.asarray([fs.reference(s) for s in np.linspace(0, 1, 1001)])
    speed = np.abs(np.diff(refs, axis=0)).max() * 1000 / fs.duration_s
    assert speed <= 0.8 + 1e-3


def test_a_frozen_leg_pauses_the_reference_and_then_aborts_to_damping() -> None:
    calf = ORDER.index("FR_calf_joint")
    plant = Plant(FOLDED, frozen=[calf])
    fs = FixStand(ORDER)
    fs.start(plant.measured()[0])
    ticks = _run(fs, plant, 400)
    assert fs.aborted and fs.fault == "fix_stand_tracking_lost", fs.fault
    # Progress stopped while the leg could not follow: the timer did not finish it.
    assert fs.progress < 1.0
    assert any(t.paused for t in ticks)
    after = [t for t in ticks if t.phase == PHASE_ABORTED]
    assert after and all(t.kp == 0.0 for t in after)
    assert all(t.kd == fs.params.damp_kd for t in after)


def test_huge_following_error_aborts_immediately() -> None:
    fs = FixStand(ORDER)
    fs.start(STAND)
    fs.step(STAND)
    tick = fs.step(STAND + 0.7)  # a joint 0.7 rad from where it was told to be
    assert tick.phase == PHASE_ABORTED and tick.kp == 0.0
    assert fs.fault.startswith("fix_stand_tracking_error")


def test_abort_is_immediate_and_latched() -> None:
    plant = Plant(FOLDED)
    fs = FixStand(ORDER)
    fs.start(plant.measured()[0])
    _run(fs, plant, 20)
    fs.abort("deadman_released")
    ticks = _run(fs, plant, 5)
    assert all(t.phase == PHASE_ABORTED and t.kp == 0.0 for t in ticks)
    assert fs.fault == "deadman_released"
    fs.abort("second")
    assert fs.fault == "deadman_released"


def test_measured_q_beyond_the_band_refuses_to_start() -> None:
    q = STAND.copy()
    j = ORDER.index("RR_thigh_joint")
    q[j] = JOINT_POSITION_LIMITS_RAD["RR_thigh_joint"][0] - 0.3
    with pytest.raises(FixStandError):
        FixStand(ORDER).start(q)


def test_non_finite_measurement_refuses_start_and_aborts_mid_rise() -> None:
    q = STAND.copy()
    q[3] = np.nan
    with pytest.raises(FixStandError):
        FixStand(ORDER).start(q)
    fs = FixStand(ORDER)
    fs.start(FOLDED)
    tick = fs.step(q)
    assert tick.phase == PHASE_ABORTED and fs.fault == "fix_stand_measured_q_invalid"


def test_start_twice_and_step_before_start_are_refused() -> None:
    fs = FixStand(ORDER)
    with pytest.raises(FixStandError):
        fs.step(STAND)
    fs.start(STAND)
    with pytest.raises(FixStandError):
        fs.start(STAND)


def test_starting_at_the_stance_completes_and_holds_it() -> None:
    plant = Plant(STAND)
    fs = FixStand(ORDER)
    fs.start(plant.measured()[0])
    ticks = _run(fs, plant, 200)
    assert ticks[-1].phase == PHASE_COMPLETE
    assert np.allclose(ticks[-1].target, STAND)


def test_timeout_aborts_a_rise_that_keeps_pausing() -> None:
    params = FixStandParams(max_pause_s=100.0, timeout_factor=1.0)
    calf = ORDER.index("RL_calf_joint")
    plant = Plant(FOLDED, frozen=[calf])
    fs = FixStand(ORDER, params)
    fs.start(plant.measured()[0])
    _run(fs, plant, 400)
    assert fs.fault == "fix_stand_timeout"


def test_params_are_validated() -> None:
    with pytest.raises(ValueError):
        FixStandParams(track_pause_rad=0.5, track_abort_rad=0.4)
    with pytest.raises(ValueError):
        FixStandParams(max_step_rad=0.001)  # would silently cap the planned speed
    with pytest.raises(ValueError):
        params_from_mapping({"kp": 60, "surprise": 1})
    assert params_from_mapping({"kp": 50}).kp == 50.0


def test_smoothstep_endpoints() -> None:
    assert smoothstep(-1) == 0.0 and smoothstep(0) == 0.0
    assert smoothstep(1) == 1.0 and smoothstep(2) == 1.0
    assert smoothstep(0.5) == pytest.approx(0.5)
