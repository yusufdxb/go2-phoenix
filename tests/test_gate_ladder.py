"""Exhaustive tests for the deploy-path gate ladder.

Every individual predicate in :mod:`phoenix.sim2real.safety` was already
tested; what was never tested was the *composition* (which gate wins when
several could fire, and what each one publishes). That gap is why L1-L3 and L5
(see ``docs/NATIVE_RUNTIME_PLAN.md``) coexisted undetected. These tests are
also the parity oracle for the native C++ port, so they assert the output
contract, not just the abort reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real.gate import (
    GateConfig,
    Outcome,
    SensorSnapshot,
    evaluate_gates,
)

SEC = 1_000_000_000
NOW = 10 * SEC

CONFIG = GateConfig(
    max_runtime_s=120.0,
    estop_timeout_s=0.5,
    sensor_timeout_s=0.2,
    first_message_timeout_s=15.0,
    pitch_rad=0.8,
    roll_rad=0.6,
)


def snapshot(**overrides) -> SensorSnapshot:
    """A nominal, everything-healthy snapshot; override one field per test."""
    base = dict(
        now_ns=NOW,
        elapsed_s=10.0,
        node_started_ns=0,
        seen_estop=True,
        seen_imu=True,
        seen_joint_state=True,
        estop_last_ns=NOW,
        estop_value=False,
        imu_last_ns=NOW,
        joint_state_last_ns=NOW,
        joint_pos=np.zeros(12, dtype=np.float32),
        joint_vel=np.zeros(12, dtype=np.float32),
        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        ang_vel=(0.0, 0.0, 0.0),
        roll=0.0,
        pitch=0.0,
    )
    base.update(overrides)
    return SensorSnapshot(**base)


def decide(snap: SensorSnapshot, *, already_latched: bool = False):
    return evaluate_gates(snap, CONFIG, already_latched=already_latched)


# --------------------------------------------------------------------------
# Nominal and latched
# --------------------------------------------------------------------------


def test_nominal_runs_policy() -> None:
    d = decide(snapshot())
    assert d.outcome is Outcome.RUN_POLICY
    assert d.reason is None
    assert not d.latches
    assert not d.publishes_default


def test_already_latched_is_permanently_silent() -> None:
    d = decide(snapshot(), already_latched=True)
    assert d.outcome is Outcome.SILENT
    assert not d.publishes_default, "post-abort rebroadcast caused a Jetson brownout"


def test_already_latched_wins_over_every_other_fault() -> None:
    # A latched node stays silent no matter what else is wrong; it must never
    # be talked back into publishing.
    d = decide(
        snapshot(
            estop_value=True,
            joint_pos=np.full(12, np.nan, dtype=np.float32),
            pitch=3.0,
        ),
        already_latched=True,
    )
    assert d.outcome is Outcome.SILENT


# --------------------------------------------------------------------------
# Runtime watchdog
# --------------------------------------------------------------------------


def test_max_runtime_latches_without_publishing() -> None:
    # Deliberately asymmetric with every other abort: max_runtime emits
    # nothing, because the original code latched and then hit the
    # already-latched early return on the same tick.
    d = decide(snapshot(elapsed_s=120.1))
    assert d.outcome is Outcome.LATCH_SILENT
    assert d.reason == "max_runtime"
    assert d.latches
    assert not d.publishes_default


def test_max_runtime_boundary_is_strict_greater_than() -> None:
    assert decide(snapshot(elapsed_s=120.0)).outcome is Outcome.RUN_POLICY
    assert decide(snapshot(elapsed_s=120.001)).outcome is Outcome.LATCH_SILENT


def test_max_runtime_does_not_relatch_when_already_latched() -> None:
    d = decide(snapshot(elapsed_s=999.0), already_latched=True)
    assert d.outcome is Outcome.SILENT
    assert d.reason is None, "first cause must be preserved, not overwritten"


def test_max_runtime_outranks_all_gates_below_it() -> None:
    d = decide(snapshot(elapsed_s=999.0, estop_value=True, pitch=3.0))
    assert d.reason == "max_runtime"


# --------------------------------------------------------------------------
# Startup gate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["seen_estop", "seen_imu", "seen_joint_state"])
def test_startup_waiting_publishes_default_without_latching(missing: str) -> None:
    d = decide(snapshot(**{missing: False}))
    assert d.outcome is Outcome.PUBLISH_DEFAULT
    assert not d.latches, "waiting is the one repeatedly-publishing, non-latching state"
    assert d.publishes_default


def test_startup_timeout_latches_with_specific_reason() -> None:
    d = decide(snapshot(seen_imu=False, now_ns=NOW, node_started_ns=NOW - 16 * SEC))
    assert d.outcome is Outcome.LATCH_AND_PUBLISH_DEFAULT
    assert d.reason is not None and "imu" in d.reason.lower()


def test_startup_outranks_sensor_and_attitude_gates() -> None:
    # A half-discovered graph must not run inference, and must not be
    # misattributed to a sensor or attitude fault.
    d = decide(snapshot(seen_imu=False, estop_value=True, pitch=3.0))
    assert d.outcome is Outcome.PUBLISH_DEFAULT


# --------------------------------------------------------------------------
# Estop chain and sensor freshness
# --------------------------------------------------------------------------


def test_external_estop_latches() -> None:
    d = decide(snapshot(estop_value=True))
    assert d.outcome is Outcome.LATCH_AND_PUBLISH_DEFAULT
    assert d.reason == "external_estop"


def test_stale_estop_heartbeat_latches() -> None:
    d = decide(snapshot(estop_last_ns=NOW - int(0.6 * SEC)))
    assert d.reason == "estop_heartbeat_stale"


def test_missing_estop_publisher_latches() -> None:
    d = decide(snapshot(estop_last_ns=None, estop_value=None))
    assert d.reason == "estop_publisher_missing"


@pytest.mark.parametrize("field", ["imu_last_ns", "joint_state_last_ns"])
def test_stale_sensor_latches(field: str) -> None:
    d = decide(snapshot(**{field: NOW - int(0.3 * SEC)}))
    assert d.outcome is Outcome.LATCH_AND_PUBLISH_DEFAULT
    assert d.reason == "sensor_stale"


def test_estop_outranks_sensor_staleness() -> None:
    d = decide(snapshot(estop_value=True, imu_last_ns=NOW - int(5 * SEC)))
    assert d.reason == "external_estop"


# --------------------------------------------------------------------------
# Non-finite sensor data
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["joint_pos", "joint_vel"])
def test_non_finite_joint_state_latches(field: str, bad: float) -> None:
    arr = np.zeros(12, dtype=np.float32)
    arr[7] = bad
    d = decide(snapshot(**{field: arr}))
    assert d.outcome is Outcome.LATCH_AND_PUBLISH_DEFAULT
    assert d.reason == "nan_in_joint_state"


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("index", range(4))
def test_non_finite_quaternion_latches(index: int, bad: float) -> None:
    # Regression for L2. Before the IMU finiteness gate existed, a corrupt
    # quaternion produced NaN roll/pitch, sailed through the attitude gate
    # (abs(NaN) > 0.8 is False), and poisoned the observation with NaN
    # projected gravity, and the policy then acted on garbage.
    quat = [0.0, 0.0, 0.0, 1.0]
    quat[index] = bad
    d = decide(snapshot(quat_xyzw=tuple(quat), roll=np.nan, pitch=np.nan))
    assert d.outcome is Outcome.LATCH_AND_PUBLISH_DEFAULT
    assert d.reason == "nan_in_imu"


@pytest.mark.parametrize("index", range(3))
def test_non_finite_angular_velocity_latches(index: int) -> None:
    ang = [0.0, 0.0, 0.0]
    ang[index] = np.nan
    d = decide(snapshot(ang_vel=tuple(ang)))
    assert d.reason == "nan_in_imu"


def test_nan_attitude_cannot_silently_pass_the_attitude_gate() -> None:
    # The precise L2 failure: NaN roll/pitch with a finite-looking comparison.
    # Without the IMU gate this returned RUN_POLICY.
    d = decide(snapshot(quat_xyzw=(np.nan, 0.0, 0.0, 1.0), roll=np.nan, pitch=np.nan))
    assert d.outcome is not Outcome.RUN_POLICY


def test_joint_state_validity_outranks_imu_validity() -> None:
    d = decide(
        snapshot(
            joint_pos=np.full(12, np.nan, dtype=np.float32),
            quat_xyzw=(np.nan, 0.0, 0.0, 1.0),
        )
    )
    assert d.reason == "nan_in_joint_state"


# --------------------------------------------------------------------------
# Attitude
# --------------------------------------------------------------------------


def test_pitch_abort() -> None:
    d = decide(snapshot(pitch=0.81))
    assert d.outcome is Outcome.LATCH_AND_PUBLISH_DEFAULT
    assert d.reason is not None and d.reason.startswith("attitude")


def test_roll_abort() -> None:
    d = decide(snapshot(roll=-0.61))
    assert d.reason is not None and d.reason.startswith("attitude")


def test_attitude_thresholds_are_asymmetric_and_exclusive() -> None:
    # roll 0.6 and pitch 0.8 are deliberately different; a symmetric port
    # would silently loosen roll or tighten pitch.
    assert decide(snapshot(pitch=0.8)).outcome is Outcome.RUN_POLICY
    assert decide(snapshot(roll=0.6)).outcome is Outcome.RUN_POLICY
    assert decide(snapshot(roll=0.61)).outcome is Outcome.LATCH_AND_PUBLISH_DEFAULT
    assert decide(snapshot(pitch=0.79, roll=0.59)).outcome is Outcome.RUN_POLICY


def test_imu_validity_outranks_attitude() -> None:
    d = decide(snapshot(ang_vel=(np.nan, 0.0, 0.0), pitch=3.0))
    assert d.reason == "nan_in_imu"


# --------------------------------------------------------------------------
# Full-ladder precedence
# --------------------------------------------------------------------------


def test_full_precedence_order() -> None:
    """Every gate firing at once: they must resolve in ladder order."""
    everything_wrong = dict(
        elapsed_s=999.0,
        seen_imu=False,
        estop_value=True,
        imu_last_ns=NOW - 5 * SEC,
        joint_pos=np.full(12, np.nan, dtype=np.float32),
        quat_xyzw=(np.nan, np.nan, np.nan, np.nan),
        pitch=3.0,
        roll=3.0,
    )
    # 1. max_runtime wins outright.
    assert decide(snapshot(**everything_wrong)).reason == "max_runtime"

    # 2. Drop it: startup gate wins, and does not latch.
    everything_wrong.pop("elapsed_s")
    assert decide(snapshot(**everything_wrong)).outcome is Outcome.PUBLISH_DEFAULT

    # 3. Drop it: the estop wins over sensors, NaN and attitude.
    everything_wrong.pop("seen_imu")
    assert decide(snapshot(**everything_wrong)).reason == "external_estop"

    # 4. Drop it: sensor staleness wins over NaN and attitude.
    everything_wrong["estop_value"] = False
    assert decide(snapshot(**everything_wrong)).reason == "sensor_stale"

    # 5. Drop it: joint NaN wins over IMU NaN and attitude.
    everything_wrong["imu_last_ns"] = NOW
    assert decide(snapshot(**everything_wrong)).reason == "nan_in_joint_state"

    # 6. Drop it: IMU NaN wins over attitude.
    everything_wrong["joint_pos"] = np.zeros(12, dtype=np.float32)
    assert decide(snapshot(**everything_wrong)).reason == "nan_in_imu"

    # 7. Drop it: attitude is last.
    everything_wrong["quat_xyzw"] = (0.0, 0.0, 0.0, 1.0)
    assert decide(snapshot(**everything_wrong)).reason.startswith("attitude")

    # 8. Drop everything: the policy finally gets to command motion.
    everything_wrong["pitch"] = 0.0
    everything_wrong["roll"] = 0.0
    assert decide(snapshot(**everything_wrong)).outcome is Outcome.RUN_POLICY


def test_only_latching_outcomes_carry_a_reason() -> None:
    for snap, latched in [
        (snapshot(), False),
        (snapshot(), True),
        (snapshot(seen_imu=False), False),
    ]:
        d = decide(snap, already_latched=latched)
        assert (d.reason is not None) == d.latches
