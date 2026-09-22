"""Bridge stand-up ramp (2026-09-21).

Stage F on the GO2 from a folded start: the loaded legs could not rise under the
measured-q slew clip and the unloaded diagonal flailed out of distribution until
RR_thigh requested a target beyond its limit. The bridge now ramps to the training
stance first, under the same deadman and limit guards, and follows the policy only
once the stance is reached.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real import preflight_eval as pe
from phoenix.sim2real.actuator_gate import ActuatorGate, GateParams
from phoenix.sim2real.bridge_telemetry import summarize
from phoenix.sim2real.command_wire import KIND_ABORT, KIND_POLICY, encode
from phoenix.sim2real.go2_model import (
    POLICY_JOINT_ORDER,
    TRAINING_DEFAULT_JOINT_POS,
    UNITREE_MOTOR_ORDER,
)

ORDER = POLICY_JOINT_ORDER
DEFAULT_P = np.asarray([TRAINING_DEFAULT_JOINT_POS[n] for n in ORDER])
STAND_U = np.asarray([TRAINING_DEFAULT_JOINT_POS[n] for n in UNITREE_MOTOR_ORDER])
# Measured on the GO2 at the start of stage F, 2026-09-21 (calves past the -2.72 limit).
FOLDED_U = np.asarray([-0.07, 1.25, -2.76, 0.05, 1.25, -2.8, -0.36, 1.28, -2.8, 0.37, 1.26, -2.78])
T0 = 10_000_000_000
PERIOD_NS = 20_000_000
REAL = ("phoenix_wireless_estop",)


def _gate(standup_s: float = 2.0) -> ActuatorGate:
    return ActuatorGate(
        GateParams(
            live=True,
            kp=25.0,
            kd=0.5,
            hold_kp=20.0,
            hold_kd=1.0,
            watchdog_s=0.2,
            estop_timeout_s=0.5,
            lowstate_timeout_s=0.2,
            stale_hold_s=0.2,
            first_message_timeout_s=15.0,
            standup_s=standup_s,
        ),
        T0,
    )


def _policy(gate: ActuatorGate, t: int, seq: int) -> None:
    label, data = encode(
        ORDER,
        seq=seq,
        kind=KIND_POLICY,
        target=DEFAULT_P,
        requested_target=DEFAULT_P,
        q_policy=DEFAULT_P,
        obs_source_code=0.0,
        stand_only=1.0,
        base_lin_vel_fed=[0.0] * 3,
        velocity_command_fed=[0.0] * 3,
        cmd_vel_received=[0.0] * 3,
        roll_rad=0.0,
        pitch_rad=0.0,
    )
    gate.on_command(t, label, data)


def _step(gate, i, q, *, released=False, policy_seq=None):
    t = T0 + i * PERIOD_NS
    gate.on_lowstate(t, q, np.zeros(12))
    gate.on_estop(t, released)
    gate.on_estop_publishers(list(REAL))
    if policy_seq is not None:
        _policy(gate, t + 1, policy_seq)
    return gate.tick(t + 2_000_000)


def test_ramp_goes_from_measured_folded_pose_to_training_stance_at_standup_gains() -> None:
    gate = _gate(standup_s=2.0)
    recs = [_step(gate, i, FOLDED_U) for i in range(110)]
    ramp = [r for r in recs if r["mode"] == "standup"]
    assert ramp, [r["mode"] for r in recs[:5]]
    first = np.asarray(ramp[0]["final_target_unitree"])
    lo_clipped = np.clip(FOLDED_U, -2.7227, None)
    assert np.allclose(first[[2, 5, 8, 11]], lo_clipped[[2, 5, 8, 11]], atol=0.03)
    assert all(r["kp"] == 60.0 and r["kd"] == 5.0 for r in ramp)
    steps = np.diff([np.asarray(r["final_target_unitree"]) for r in ramp], axis=0)
    assert np.abs(steps).max() < 0.03  # 1.3 rad over 100 ticks, never a jump
    done = [r for r in ramp if r["standup_done"]]
    assert done and np.allclose(done[-1]["final_target_unitree"], STAND_U)
    assert gate.fault is None


def test_policy_is_not_followed_until_the_stance_is_reached() -> None:
    gate = _gate(standup_s=1.0)
    modes = [_step(gate, i, FOLDED_U, policy_seq=i + 1)["mode"] for i in range(80)]
    first_policy = modes.index("policy")
    assert set(modes[:first_policy]) <= {"hold", "standup"}
    assert modes[first_policy - 1] == "standup"
    assert first_policy >= 50  # 1 s at 50 Hz
    assert set(modes[first_policy:]) == {"policy"}


def test_deadman_release_mid_ramp_leaves_standup_and_rearm_restarts_from_measured() -> None:
    gate = _gate(standup_s=2.0)
    for i in range(30):
        _step(gate, i, FOLDED_U)
    half = FOLDED_U + 0.3 * (STAND_U - FOLDED_U)
    rec = _step(gate, 30, half, released=True)
    assert rec["mode"] == "hold" and rec["kp"] == 20.0
    assert rec["final_target_unitree"] == pytest.approx(list(np.clip(half, -2.7227, None)))


def test_standup_disabled_keeps_the_previous_hold_behaviour() -> None:
    gate = _gate(standup_s=0.0)
    recs = [_step(gate, i, FOLDED_U) for i in range(60)]
    assert {r["mode"] for r in recs} <= {"hold"}
    assert all(r["kp"] == 20.0 for r in recs if r["mode"] == "hold")


def test_negative_standup_duration_is_refused() -> None:
    with pytest.raises(ValueError):
        _gate(standup_s=-1.0)


def test_stand_stage_with_standup_evaluates_go() -> None:
    gate = _gate(standup_s=1.0)
    ticks = []
    q = FOLDED_U.copy()
    n_policy = 0
    aborted = False
    for i in range(250):
        t = T0 + i * PERIOD_NS
        gate.on_lowstate(t, q, np.zeros(12))
        gate.on_estop(t, False)
        gate.on_estop_publishers(list(REAL))
        if gate._standup_done or n_policy:  # the harness launches the policy after the stand
            if n_policy < 100:
                _policy(gate, t + 1, n_policy + 1)
                n_policy += 1
            elif not aborted:
                label, data = encode(
                    ORDER,
                    seq=10_000,
                    kind=KIND_ABORT,
                    target=DEFAULT_P,
                    abort_reason="authority_window_complete",
                )
                gate.on_command(t + 1, label, data)
                aborted = True
        rec = gate.tick(t + 2_000_000)
        ticks.append(rec)
        q = np.asarray(rec["final_target_unitree"])  # an ideal robot tracks its target
    gate.request_shutdown()
    for k in range(10):
        ticks.append(gate.tick(T0 + (250 + k) * PERIOD_NS))
    checks = pe.stand_checks(
        stage="F",
        manifest={},
        ticks=ticks,
        summary=summarize({}, ticks),
        cfg={
            "control": {"rate_hz": 50},
            "safety": {"sensor_timeout_s": 0.2, "estop_timeout_s": 0.5},
        },
        lock={},
        expected_sha=None,
        authority_s=2.0,
        watchdog_s=0.2,
        operator_confirmed_stand=True,
    )
    failed = [c for c in checks if not c.ok and c.gating and not c.name.startswith("bridge ")]
    assert not failed, failed
