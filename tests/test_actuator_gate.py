"""The final actuator gate, exhaustively: every joint, both limits, every fault path.

This is the last code between a policy output and the GO2 motors, so the tests
are written against observable outputs (mode, target, gains, latched fault)
rather than internals.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real.actuator_gate import (
    REAL_DEADMAN_NODE_NAMES,
    ActuatorGate,
    GateParams,
    Mode,
)
from phoenix.sim2real.command_wire import (
    KIND_ABORT,
    KIND_POLICY,
    KIND_STARTUP_DEFAULT,
    encode,
    wire_label,
)
from phoenix.sim2real.go2_model import (
    JOINT_POSITION_LIMITS_RAD,
    POLICY_JOINT_ORDER,
    TRAINING_DEFAULT_JOINT_POS,
    UNITREE_EXAMPLE_FOLDED_POSE,
    UNITREE_MOTOR_ORDER,
    limits_in_order,
)
from phoenix.sim2real.motor_crc import phoenix_to_unitree
from phoenix.sim2real.safety import MAX_DELTA_PER_STEP_RAD

ORDER = POLICY_JOINT_ORDER
T0 = 5_000_000_000
DEFAULT_P = np.asarray([TRAINING_DEFAULT_JOINT_POS[n] for n in ORDER])
DEFAULT_U = np.asarray(phoenix_to_unitree(DEFAULT_P))
LO_U, HI_U = limits_in_order(UNITREE_MOTOR_ORDER)
ZERO12 = np.zeros(12)


def ns(seconds: float) -> int:
    return T0 + int(round(seconds * 1e9))


def params(**over) -> GateParams:
    base = dict(
        live=False,
        kp=25.0,
        kd=0.5,
        hold_kp=20.0,
        hold_kd=1.0,
        watchdog_s=0.2,
        estop_timeout_s=0.5,
        lowstate_timeout_s=0.2,
        stale_hold_s=0.2,
        first_message_timeout_s=15.0,
    )
    base.update(over)
    return GateParams(**base)


def policy_cmd(target_p, seq=1, **over):
    fields = dict(
        seq=seq,
        kind=KIND_POLICY,
        target=target_p,
        requested_target=target_p,
        q_policy=DEFAULT_P,
        obs_source_code=0.0,
        stand_only=1.0,
        velocity_command_fed=[0.0, 0.0, 0.0],
        cmd_vel_received=[0.0, 0.0, 0.0],
    )
    fields.update(over)
    return encode(ORDER, **fields)


def armed(live=False, q_u=DEFAULT_U, t=0.0, **p_over) -> ActuatorGate:
    gate = ActuatorGate(params(live=live, **p_over), ns(0.0))
    gate.on_lowstate(ns(t), q_u, ZERO12)
    gate.on_estop(ns(t), False)
    if live:
        gate.on_estop_publishers(["phoenix_wireless_estop"])
    return gate


def send(gate, t, target_p, seq=1, **over):
    label, data = policy_cmd(target_p, seq=seq, **over)
    gate.on_command(ns(t), label, data)


def u_index(phoenix_index: int) -> int:
    return UNITREE_MOTOR_ORDER.index(ORDER[phoenix_index])


# ------------------------------------------------------------------ basics
def test_silent_until_first_lowstate() -> None:
    gate = ActuatorGate(params(), ns(0.0))
    gate.on_estop(ns(0.0), False)
    rec = gate.tick(ns(0.01))
    assert rec["mode"] == Mode.SILENT.value and rec["publish"] is False


def test_hold_at_measured_posture_with_no_command() -> None:
    gate = armed()
    rec = gate.tick(ns(0.01))
    assert rec["mode"] == "hold" and rec["hold_cause"] == "no_policy_command"
    assert np.allclose(rec["final_target_unitree"], DEFAULT_U)
    assert (rec["kp"], rec["kd"]) == (20.0, 1.0)
    assert rec["fault"] is None


def test_startup_default_pose_is_not_followed() -> None:
    folded = np.asarray(UNITREE_EXAMPLE_FOLDED_POSE)
    gate = armed(q_u=folded)
    label, data = encode(ORDER, seq=1, kind=KIND_STARTUP_DEFAULT, target=DEFAULT_P)
    gate.on_command(ns(0.0), label, data)
    rec = gate.tick(ns(0.01))
    assert rec["mode"] == "hold" and rec["hold_cause"] == "startup_default"
    assert np.allclose(rec["final_target_unitree"], folded), "must hold folded, not drive to stand"
    assert rec["fault"] is None


def test_policy_passthrough_inside_slew_and_limits() -> None:
    gate = armed()
    target = DEFAULT_P + 0.05
    send(gate, 0.0, target)
    rec = gate.tick(ns(0.01))
    assert rec["mode"] == "policy"
    assert np.allclose(rec["final_target_unitree"], phoenix_to_unitree(target))
    assert rec["slew_clip"] == [False] * 12 and rec["limit_clip"] == [False] * 12
    assert (rec["kp"], rec["kd"]) == (25.0, 0.5)
    assert rec["cmd_is_new"] is True
    again = gate.tick(ns(0.02))
    assert again["mode"] == "policy" and again["cmd_is_new"] is False


def test_permutation_is_applied_by_name() -> None:
    gate = armed()
    target = DEFAULT_P + np.arange(12) * 0.001
    send(gate, 0.0, target)
    rec = gate.tick(ns(0.01))
    for k, name in enumerate(UNITREE_MOTOR_ORDER):
        assert rec["final_target_unitree"][k] == pytest.approx(target[ORDER.index(name)])


# ------------------------------------------------------------- slew clip
@pytest.mark.parametrize("sign", [+1.0, -1.0])
@pytest.mark.parametrize("j", range(12))
def test_slew_clip_is_per_joint_and_only_that_joint(j: int, sign: float) -> None:
    gate = armed()
    target = DEFAULT_P.copy()
    target[j] += sign * 0.1  # small, inside limits
    target_big = DEFAULT_P.copy()
    target_big[j] += sign * 0.4
    send(gate, 0.0, target_big)
    rec = gate.tick(ns(0.01))
    k = u_index(j)
    assert rec["mode"] == "policy"
    lo, hi = JOINT_POSITION_LIMITS_RAD[ORDER[j]]
    expected = np.clip(DEFAULT_U[k] + sign * MAX_DELTA_PER_STEP_RAD, lo, hi)
    assert rec["final_target_unitree"][k] == pytest.approx(expected)
    assert rec["slew_clip"] == [i == k for i in range(12)]
    assert rec["slew_margin"][k] == pytest.approx(MAX_DELTA_PER_STEP_RAD - 0.4)
    assert rec["fault"] is None


# ------------------------------------------------------------ joint limits
@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("j", range(12))
def test_small_overshoot_is_clipped_to_the_hard_limit(j: int, side: str) -> None:
    name = ORDER[j]
    lo, hi = JOINT_POSITION_LIMITS_RAD[name]
    k = u_index(j)
    q = DEFAULT_U.copy()
    q[k] = lo + 0.05 if side == "lower" else hi - 0.05
    gate = armed(q_u=q)
    target = np.asarray([q[UNITREE_MOTOR_ORDER.index(n)] for n in ORDER])
    target[j] = lo - 0.1 if side == "lower" else hi + 0.1  # inside the abort band
    send(gate, 0.0, target)
    rec = gate.tick(ns(0.01))
    assert rec["mode"] == "policy", rec["fault"]
    assert rec["final_target_unitree"][k] == pytest.approx(lo if side == "lower" else hi)
    assert rec["limit_clip"] == [i == k for i in range(12)]
    assert rec["limit_margin"][k] == pytest.approx(0.0)
    assert rec["fault"] is None


@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("j", range(12))
def test_target_beyond_abort_band_latches_hold(j: int, side: str) -> None:
    name = ORDER[j]
    lo, hi = JOINT_POSITION_LIMITS_RAD[name]
    gate = armed()
    target = DEFAULT_P.copy()
    target[j] = lo - 0.2 if side == "lower" else hi + 0.2
    send(gate, 0.0, target)
    rec = gate.tick(ns(0.01))
    assert rec["mode"] == "hold"
    assert rec["fault"] == f"target_beyond_limit:{name}"
    assert np.allclose(rec["final_target_unitree"], DEFAULT_U)
    # Latched: a perfectly valid command afterwards does not restore authority.
    send(gate, 0.02, DEFAULT_P, seq=2)
    assert gate.tick(ns(0.03))["mode"] == "hold"


def test_final_target_always_within_limits_and_one_slew_step() -> None:
    rng = np.random.default_rng(0)
    for trial in range(400):
        q = rng.uniform(LO_U, HI_U)
        gate = armed(q_u=q)
        req_u = rng.uniform(LO_U - 0.17, HI_U + 0.17)
        req_p = np.asarray([req_u[UNITREE_MOTOR_ORDER.index(n)] for n in ORDER])
        send(gate, 0.0, req_p, seq=trial)
        rec = gate.tick(ns(0.01))
        assert rec["mode"] == "policy"
        final = np.asarray(rec["final_target_unitree"])
        assert np.all(final >= LO_U) and np.all(final <= HI_U)
        assert np.all(np.abs(final - q) <= MAX_DELTA_PER_STEP_RAD + 1e-12)


# ----------------------------------------------------- malformed commands
def test_nan_command_latches_hold() -> None:
    gate = armed()
    label, data = policy_cmd(DEFAULT_P)
    data[4] = float("nan")  # first target joint
    gate.on_command(ns(0.0), label, data)
    rec = gate.tick(ns(0.01))
    assert rec["mode"] == "hold" and rec["fault"] == "command_rejected:non_finite_target"


def test_joint_order_mismatch_latches_hold() -> None:
    gate = armed()
    swapped = list(ORDER)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    label, data = encode(
        swapped, seq=1, kind=KIND_POLICY, target=DEFAULT_P, obs_source_code=0.0, stand_only=1.0
    )
    gate.on_command(ns(0.0), label, data)
    rec = gate.tick(ns(0.01))
    assert rec["fault"] == "command_rejected:label_mismatch" and rec["mode"] == "hold"


def test_legacy_twelve_float_command_latches_hold() -> None:
    gate = armed()
    gate.on_command(ns(0.0), "", list(DEFAULT_P))
    assert gate.tick(ns(0.01))["fault"] == "command_rejected:label_mismatch"


def test_policy_abort_holds_measured_posture_not_default_pose() -> None:
    folded = np.asarray(UNITREE_EXAMPLE_FOLDED_POSE)
    gate = armed(q_u=folded)
    send(gate, 0.0, np.asarray([folded[UNITREE_MOTOR_ORDER.index(n)] for n in ORDER]))
    assert gate.tick(ns(0.01))["mode"] == "policy"
    label, data = encode(
        ORDER, seq=2, kind=KIND_ABORT, target=DEFAULT_P, abort_reason="attitude pitch=0.9"
    )
    gate.on_command(ns(0.015), label, data)
    rec = gate.tick(ns(0.02))
    assert rec["mode"] == "hold" and rec["fault"] == "policy_abort:attitude"
    assert np.allclose(rec["final_target_unitree"], folded)


# ------------------------------------------------------ walking is blocked
def test_nonzero_velocity_command_in_stand_only_latches() -> None:
    gate = armed()
    send(gate, 0.0, DEFAULT_P, velocity_command_fed=[0.3, 0.0, 0.0])
    assert gate.tick(ns(0.01))["fault"] == "walking_blocked_stand_only"


def test_nonzero_cmd_vel_received_in_stand_only_latches() -> None:
    gate = armed()
    send(gate, 0.0, DEFAULT_P, cmd_vel_received=[0.0, 0.0, 0.5])
    assert gate.tick(ns(0.01))["fault"] == "walking_blocked_stand_only"


def test_zeros_source_without_stand_only_latches() -> None:
    gate = armed()
    send(gate, 0.0, DEFAULT_P, stand_only=0.0)
    assert gate.tick(ns(0.01))["fault"] == "walking_blocked_zero_base_lin_vel"


def test_odom_walking_is_still_blocked_while_walking_disabled() -> None:
    gate = armed()
    send(
        gate,
        0.0,
        DEFAULT_P,
        stand_only=0.0,
        obs_source_code=1.0,
        velocity_command_fed=[0.3, 0.0, 0.0],
    )
    assert gate.tick(ns(0.01))["fault"] == "walking_blocked_not_enabled"


def test_command_without_observation_provenance_latches() -> None:
    gate = armed()
    label, data = encode(ORDER, seq=1, kind=KIND_POLICY, target=DEFAULT_P)
    gate.on_command(ns(0.0), label, data)
    assert gate.tick(ns(0.01))["fault"] == "command_rejected:missing_observation_provenance"


# ------------------------------------------------------ estop and watchdog
def test_estop_not_yet_seen_holds_without_latching_then_arms() -> None:
    gate = ActuatorGate(params(), ns(0.0))
    gate.on_lowstate(ns(1.0), DEFAULT_U, ZERO12)
    send(gate, 1.0, DEFAULT_P)
    rec = gate.tick(ns(1.0))
    assert (
        rec["mode"] == "hold" and rec["hold_cause"] == "estop_not_yet_seen" and rec["fault"] is None
    )
    gate.on_lowstate(ns(2.0), DEFAULT_U, ZERO12)
    gate.on_estop(ns(2.0), False)
    send(gate, 2.0, DEFAULT_P, seq=2)
    assert gate.tick(ns(2.01))["mode"] == "policy"


def test_estop_never_seen_past_first_message_timeout_latches() -> None:
    gate = ActuatorGate(params(first_message_timeout_s=1.0), ns(0.0))
    gate.on_lowstate(ns(1.5), DEFAULT_U, ZERO12)
    rec = gate.tick(ns(1.51))
    assert rec["fault"] == "estop_publisher_missing"


def test_estop_true_latches_and_false_does_not_release() -> None:
    gate = armed()
    send(gate, 0.0, DEFAULT_P)
    assert gate.tick(ns(0.01))["mode"] == "policy"
    gate.on_estop(ns(0.02), True)
    assert gate.tick(ns(0.03))["fault"] == "estop_asserted"
    gate.on_estop(ns(0.04), False)
    gate.on_lowstate(ns(0.04), DEFAULT_U, ZERO12)
    send(gate, 0.04, DEFAULT_P, seq=2)
    rec = gate.tick(ns(0.05))
    assert rec["mode"] == "hold" and rec["fault"] == "estop_asserted"


def test_stale_estop_heartbeat_latches() -> None:
    gate = armed()
    gate.on_lowstate(ns(0.6), DEFAULT_U, ZERO12)
    send(gate, 0.6, DEFAULT_P)
    rec = gate.tick(ns(0.61))
    assert rec["fault"] == "estop_heartbeat_stale" and rec["mode"] == "hold"


def test_command_watchdog_holds_without_latching_and_resumes() -> None:
    gate = armed()
    send(gate, 0.0, DEFAULT_P)
    assert gate.tick(ns(0.1))["mode"] == "policy"
    gate.on_lowstate(ns(0.25), DEFAULT_U, ZERO12)
    gate.on_estop(ns(0.25), False)
    rec = gate.tick(ns(0.25))
    assert rec["mode"] == "hold" and rec["hold_cause"] == "command_stale" and rec["fault"] is None
    send(gate, 0.26, DEFAULT_P, seq=2)
    assert gate.tick(ns(0.27))["mode"] == "policy"


# ------------------------------------------------------ LowState freshness
def test_stale_lowstate_revokes_authority_holds_then_damps() -> None:
    gate = armed()
    send(gate, 0.0, DEFAULT_P)
    assert gate.tick(ns(0.1))["mode"] == "policy"
    gate.on_estop(ns(0.25), False)
    send(gate, 0.25, DEFAULT_P, seq=2)
    rec = gate.tick(ns(0.25))  # lowstate is 0.25 s old > 0.2 s
    assert rec["lowstate_fresh"] is False
    assert rec["mode"] == "hold" and rec["fault"] == "lowstate_stale"
    assert np.allclose(rec["final_target_unitree"], DEFAULT_U)
    gate.on_estop(ns(0.4), False)
    rec = gate.tick(ns(0.46))  # stale for 0.21 s > stale_hold_s
    assert rec["mode"] == "damp" and rec["kp"] == 0.0 and rec["kd"] == 1.0
    # Latched: LowState coming back does not re-stiffen or restore authority.
    gate.on_lowstate(ns(0.5), DEFAULT_U, ZERO12)
    gate.on_estop(ns(0.5), False)
    send(gate, 0.5, DEFAULT_P, seq=3)
    assert gate.tick(ns(0.51))["mode"] == "damp"


def test_lowstate_recovering_inside_stale_window_holds_but_never_resumes_policy() -> None:
    gate = armed()
    gate.on_estop(ns(0.25), False)
    assert gate.tick(ns(0.25))["fault"] == "lowstate_stale"
    gate.on_lowstate(ns(0.3), DEFAULT_U + 0.01, ZERO12)
    gate.on_estop(ns(0.3), False)
    send(gate, 0.3, DEFAULT_P)
    rec = gate.tick(ns(0.31))
    assert rec["mode"] == "hold" and rec["lowstate_fresh"] is True
    assert np.allclose(rec["final_target_unitree"], DEFAULT_U + 0.01)


def test_non_finite_lowstate_damps_immediately() -> None:
    gate = armed()
    bad = DEFAULT_U.copy()
    bad[3] = float("nan")
    gate.on_lowstate(ns(0.01), bad, ZERO12)
    rec = gate.tick(ns(0.02))
    assert rec["mode"] == "damp" and rec["fault"] == "lowstate_non_finite"


@pytest.mark.parametrize("j", range(12))
def test_impossible_measured_position_damps(j: int) -> None:
    q = DEFAULT_U.copy()
    q[j] = HI_U[j] + 0.3
    gate = armed(q_u=q)
    rec = gate.tick(ns(0.01))
    assert rec["mode"] == "damp"
    assert rec["fault"] == f"measured_q_impossible:{UNITREE_MOTOR_ORDER[j]}"


@pytest.mark.parametrize("j", range(12))
def test_measured_position_just_past_limit_is_held_clamped(j: int) -> None:
    q = DEFAULT_U.copy()
    q[j] = LO_U[j] - 0.01
    gate = armed(q_u=q)
    rec = gate.tick(ns(0.01))
    assert rec["mode"] == "hold" and rec["fault"] is None
    assert rec["final_target_unitree"][j] == pytest.approx(LO_U[j])
    assert rec["limit_clip"] == [i == j for i in range(12)]


# ------------------------------------------------------------ real deadman
def test_live_gate_waits_for_deadman_discovery_without_latching() -> None:
    gate = ActuatorGate(params(live=True), ns(0.0))
    gate.on_lowstate(ns(0.0), DEFAULT_U, ZERO12)
    gate.on_estop(ns(0.0), False)
    send(gate, 0.0, DEFAULT_P)
    rec = gate.tick(ns(0.01))
    assert rec["mode"] == "hold" and rec["hold_cause"] == "deadman_source_not_yet_discovered"
    assert rec["fault"] is None
    gate.on_estop_publishers(["phoenix_wireless_estop"])
    assert gate.tick(ns(0.02))["mode"] == "policy"


@pytest.mark.parametrize(
    "pubs",
    [
        ["_ros2cli_4242"],
        ["phoenix_wireless_estop", "_ros2cli_4242"],
        ["phoenix_wireless_estop", "phoenix_deadman"],
    ],
)
def test_live_gate_refuses_anything_but_one_real_deadman(pubs) -> None:
    gate = armed(live=True)
    gate.on_estop_publishers(pubs)
    send(gate, 0.0, DEFAULT_P)
    rec = gate.tick(ns(0.01))
    assert rec["mode"] == "hold"
    assert rec["fault"].startswith("estop_source_not_a_real_deadman:")


def test_live_gate_latches_when_the_deadman_publisher_disappears() -> None:
    gate = armed(live=True)
    send(gate, 0.0, DEFAULT_P)
    assert gate.tick(ns(0.01))["mode"] == "policy"
    gate.on_estop_publishers([])
    assert gate.tick(ns(0.02))["fault"] == "deadman_publisher_lost"


def test_dry_gate_accepts_synthetic_heartbeat_but_records_it() -> None:
    gate = armed(live=False)
    gate.on_estop_publishers(["_ros2cli_4242"])
    send(gate, 0.0, DEFAULT_P)
    rec = gate.tick(ns(0.01))
    assert rec["mode"] == "policy" and rec["deadman_source_ok"] is False


def test_live_gate_cannot_disable_deadman_requirement() -> None:
    with pytest.raises(ValueError):
        params(live=True, require_real_deadman=False)
    assert REAL_DEADMAN_NODE_NAMES == {"phoenix_wireless_estop", "phoenix_deadman"}


def test_gate_refuses_a_wrong_joint_order_at_construction() -> None:
    swapped = list(ORDER)
    swapped[2], swapped[3] = swapped[3], swapped[2]
    with pytest.raises(ValueError):
        ActuatorGate(params(joint_order=tuple(swapped)), ns(0.0))
    assert ActuatorGate(params(), ns(0.0)).expected_label == wire_label(ORDER)
