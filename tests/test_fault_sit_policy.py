"""Fault policy without zero stiffness: every fault class, driven through the gate and the FSM.

Per fault class: kp is never 0; kp is below the stand/hold gains only in FALL_SOFT or the
folded shutdown hold; the sequence ends at the sit pose; per-step target deltas are
rate-limited and the implied PD torque is clipped. The gate is driven at 500 Hz with
ideal tracking (measured q = last commanded target).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from phoenix.sim2real.actuator_gate import ActuatorGate, GateParams, Mode
from phoenix.sim2real.command_wire import KIND_ABORT, encode
from phoenix.sim2real.go2_model import UNITREE_MOTOR_ORDER, limits_in_order, torque_limits_in_order
from phoenix.sim2real.sit_policy import SIT_POSE_UNITREE, SitParams, fall_tilt_rad
from tests.test_actuator_gate import DEFAULT_P, DEFAULT_U, ORDER, ZERO12, ns, params, policy_cmd

DT = 0.002
SIT = SitParams()
LO, HI = limits_in_order(UNITREE_MOTOR_ORDER)
TAU = torque_limits_in_order(UNITREE_MOTOR_ORDER)
FALL_QUAT = [math.cos(math.radians(35)), math.sin(math.radians(35)), 0.0, 0.0]  # 70 deg tilt
CALF = next(i for i, n in enumerate(ORDER) if "calf" in n)


class Feed:
    def __init__(self) -> None:
        self.lowstate = True
        self.cmd = True
        self.estop = False
        self.q_override: np.ndarray | None = None
        self.target_p = DEFAULT_P


def run(apply_fault, t_fault=0.5, t_end=5.2, freeze_q=False, **p_over):
    gate = ActuatorGate(params(**p_over), ns(0.0))
    gate.on_lowstate(ns(0.0), DEFAULT_U, ZERO12)
    feed, q, recs, seq, fired = Feed(), DEFAULT_U.copy(), [], 0, False
    for k in range(1, int(round(t_end / DT)) + 1):
        t = k * DT
        if not fired and t >= t_fault:
            apply_fault(gate, feed, t)
            fired = True
        qs = feed.q_override if feed.q_override is not None else (DEFAULT_U if freeze_q else q)
        if feed.lowstate:
            gate.on_lowstate(ns(t), qs, ZERO12)
        gate.on_estop(ns(t), feed.estop)
        if feed.cmd and k % 10 == 0:
            seq += 1
            label, data = policy_cmd(feed.target_p, seq=seq)
            gate.on_command(ns(t), label, data)
        rec = gate.tick(ns(t))
        rec["t"] = t
        if rec["publish"]:
            q = np.asarray(rec["final_target_unitree"])
        recs.append(rec)
    return gate, recs


def _stale(_g, f, _t):
    f.lowstate = False


def _nonfinite(_g, f, _t):
    f.q_override = np.full(12, np.nan)


def _impossible(_g, f, _t):
    f.q_override = HI + 0.3


def _nan_cmd(_g, f, _t):
    f.target_p = np.where(np.arange(12) == 0, np.nan, DEFAULT_P)


def _abort(g, f, t):
    f.cmd = False
    label, data = encode(ORDER, seq=999, kind=KIND_ABORT, target=DEFAULT_P, abort_reason="test")
    g.on_command(ns(t), label, data)


def _estop(_g, f, _t):
    f.estop = True


def _stale_cmd(_g, f, _t):
    f.cmd = False


def _over_limit(_g, f, _t):
    tgt = DEFAULT_P.copy()
    tgt[CALF] = 0.0
    f.target_p = tgt


#: name -> (fault injector, sensor-class fault: sits open loop with no 1 s hold)
FAULTS = {
    "stale_lowstate_beyond_hold": (_stale, True),
    "non_finite_lowstate": (_nonfinite, True),
    "impossible_q": (_impossible, True),
    "policy_nan_command": (_nan_cmd, False),
    "policy_abort": (_abort, False),
    "estop": (_estop, False),
    "stale_command": (_stale_cmd, False),
    "sustained_over_limit": (_over_limit, False),
}


@pytest.mark.parametrize("name", FAULTS)
def test_fault_class_never_zero_stiffness_rate_limited_and_ends_at_sit_pose(name: str) -> None:
    inject, sensor_class = FAULTS[name]
    t_fault = 0.5
    _, recs = run(inject, t_fault=t_fault, t_end=t_fault + 5.2)
    pub = [r for r in recs if r["publish"]]
    post = [r for r in pub if r["t"] > t_fault and r["mode"] != "policy"]
    assert post and post[0]["mode"] in ("hold", "sit")
    p = params()
    for r in pub:
        assert r["kp"] > 0.0 and r["kd"] > 0.0, (name, r["mode"], r["t"])
        assert r["kp"] >= min(p.hold_kp, p.kp) or r["mode"] in ("fall_soft", "folded")
        tgt = np.asarray(r["final_target_unitree"])
        assert np.all(tgt >= LO - 1e-9) and np.all(tgt <= HI + 1e-9)
    sits = [r for r in post if r["mode"] == "sit"]
    assert sits, name
    hold_s = sits[0]["t"] - post[0]["t"]
    assert hold_s <= (0.45 if sensor_class else SIT.fault_hold_s + 0.01), (name, hold_s)
    assert all(r["mode"] != "hold" for r in post[post.index(sits[0]) :])
    assert sits[0]["sit_open_loop"] is sensor_class or name == "stale_command"
    for a, b in zip(sits, sits[1:], strict=False):
        step = np.abs(np.asarray(b["final_target_unitree"]) - np.asarray(a["final_target_unitree"]))
        assert step.max() <= SIT.max_rate_rad_s * DT + 1e-9, (name, step.max())
    last = pub[-1]
    assert last["mode"] == "sit" and last["kp"] == SIT.sit_kp
    assert np.allclose(last["final_target_unitree"], SIT_POSE_UNITREE, atol=1e-6), name
    assert Mode.SIT.value == "sit"


def test_sit_target_is_torque_clipped_when_the_leg_cannot_follow() -> None:
    # Measured q frozen at the stand pose (a blocked leg): the commanded target may
    # only lead it by what the torque limit allows, so kp (target - q) <= limit.
    _, recs = run(_estop, t_fault=0.1, t_end=4.0, freeze_q=True)
    sits = [r for r in recs if r["mode"] == "sit"]
    assert sits
    for r in sits:
        tau = SIT.sit_kp * (np.asarray(r["final_target_unitree"]) - DEFAULT_U)
        assert np.all(np.abs(tau) <= TAU + 1e-6)
    assert max(np.abs(np.asarray(r["final_target_unitree"]) - DEFAULT_U).max() for r in sits) > 0.3


def test_fall_is_low_stiffness_never_zero_and_latched() -> None:
    def fall(g, _f, t):
        g.on_imu(ns(t), FALL_QUAT)

    assert fall_tilt_rad(FALL_QUAT) == pytest.approx(math.radians(70))
    _, recs = run(fall, t_end=2.0)
    post = [r for r in recs if r["t"] > 0.5]
    assert all(r["mode"] == "fall_soft" for r in post)
    assert all(r["kp"] == SIT.fall_kp and 0.0 < r["kp"] < params().hold_kp for r in post)
    assert all(r["kd"] == SIT.fall_kd for r in post)


def test_tilt_below_threshold_is_not_a_fall() -> None:
    def tilt30(g, _f, t):
        g.on_imu(ns(t), [math.cos(math.radians(15)), math.sin(math.radians(15)), 0.0, 0.0])

    _, recs = run(tilt30, t_end=1.0)
    assert all(r["mode"] == "policy" for r in recs if r["t"] > 0.3)


def test_shutdown_sits_settles_then_holds_folded_low_kp_within_the_bound() -> None:
    t_req = 0.5

    def shutdown(g, _f, t):
        g.request_shutdown(ns(t))

    _, recs = run(shutdown, t_fault=t_req, t_end=t_req + 3.5)
    post = [r for r in recs if r["t"] >= t_req and r["publish"]]
    assert all(r["kp"] > 0.0 for r in post)
    folded = [r for r in post if r["mode"] == "folded"]
    assert folded, "never reached the folded hold"
    t_folded = folded[0]["t"] - t_req
    assert t_folded <= SIT.sit_s + SIT.folded_settle_s + 2 * DT
    assert t_folded + SIT.folded_hold_s <= SIT.shutdown_bound_s + 2 * DT
    assert SIT.shutdown_bound_s == pytest.approx(2.8)
    assert all(r["kp"] == SIT.folded_kp and r["kd"] == SIT.folded_kd for r in folded)
    assert np.allclose(folded[0]["final_target_unitree"], SIT_POSE_UNITREE, atol=1e-6)
    assert post[0]["mode"] == "sit" and post[0]["kp"] == SIT.sit_kp


def test_zero_stiffness_gains_are_refused_at_construction() -> None:
    base = dict(
        live=False, kp=25.0, kd=0.5, hold_kp=20.0, hold_kd=1.0, watchdog_s=0.2,
        estop_timeout_s=0.5, lowstate_timeout_s=0.2, stale_hold_s=0.2, first_message_timeout_s=15.0,
    )  # fmt: skip
    for bad in ("kp", "hold_kp"):
        with pytest.raises(ValueError, match="kp=0 is banned"):
            GateParams(**{**base, bad: 0.0})
    with pytest.raises(ValueError, match="standup_kp"):
        GateParams(**{**base, "standup_s": 1.0, "standup_kp": 0.0})
    for name in ("sit_kp", "fall_kp", "folded_kp"):
        with pytest.raises(ValueError):
            SitParams(**{name: 0.0})


# ------------------------------------------------------------------- FSM
def test_fsm_faults_hold_then_sit_and_stop_sits_never_kp_zero() -> None:
    from phoenix.sim2real.controller_fsm import State
    from tests.test_controller_fsm import (
        DT_NS,
        make_fsm,
        nominal_input,
        run_to_velocity_ready,
        zero_policy,
    )

    expected_p = np.empty(12)
    perm = [list(ORDER).index(n) for n in UNITREE_MOTOR_ORDER]
    expected_p[perm] = SIT_POSE_UNITREE
    for fault in ("abort", "estop", "deadman"):
        fsm = make_fsm(policy=zero_policy)
        t = run_to_velocity_ready(fsm)
        fsm.request_policy()
        fsm.tick(nominal_input(t))
        t += DT_NS
        outs = []
        for k in range(int(5.0e9 / DT_NS)):
            kw = {"estop": dict(estop_ok=False), "deadman": dict(deadman_ok=False)}.get(fault, {})
            if fault == "abort" and k == 0:
                fsm.abort("test_abort")
            outs.append(fsm.tick(nominal_input(t, **kw)))
            t += DT_NS
        assert fsm.state is State.FAULT
        assert all(o.publish and o.kp > 0.0 and o.kd > 0.0 for o in outs), fault
        assert outs[0].reason == "fault_hold" and outs[0].kp == fsm.params.hold_kp
        first_sit = next(i for i, o in enumerate(outs) if o.kp == fsm.params.sit.sit_kp)
        assert first_sit * DT_NS / 1e9 <= fsm.params.sit.fault_hold_s + 2 * DT_NS / 1e9
        tg = np.asarray([o.target for o in outs[first_sit:]])
        assert np.abs(np.diff(tg, axis=0)).max() <= SIT.max_rate_rad_s * DT_NS / 1e9 + 1e-9
        assert np.allclose(tg[-1], expected_p, atol=1e-6), fault


def test_fsm_fall_is_low_stiffness_not_zero() -> None:
    from tests.test_controller_fsm import (
        DT_NS,
        make_fsm,
        nominal_input,
        run_to_velocity_ready,
        zero_policy,
    )

    fsm = make_fsm(policy=zero_policy)
    t = run_to_velocity_ready(fsm)
    fsm.abort("test_abort")
    out = fsm.tick(dataclasses.replace(nominal_input(t), quat_wxyz=tuple(FALL_QUAT)))
    assert out.reason == "fall_soft" and out.kp == fsm.params.sit.fall_kp > 0.0
    assert DT_NS > 0
