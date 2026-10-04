"""Hardening of the Damp-free fault policy (reviewer findings 1-4).

* IMU: debounced + rpy cross-checked fall latch, wxyz order, arm only with a valid IMU,
  stale IMU is a sensor fault.
* Bridge: telemetry / a raising gate / a publish error never interrupt control or the
  shutdown sit; the shutdown sit spins callbacks (stays closed-loop); last-mile kp check.
"""

from __future__ import annotations

import math
import time
import types

import numpy as np
import pytest

import tests.test_lowcmd_bridge as _bridge_tests
from phoenix.sim2real.actuator_gate import ActuatorGate
from phoenix.sim2real.sit_policy import SIT_POSE_UNITREE, SitParams, fall_tilt_rad, valid_unit_quat
from tests.test_actuator_gate import DEFAULT_P, DEFAULT_U, ZERO12, armed, ns, params, send

bridge_module = _bridge_tests.bridge_module  # re-export the stubbed-rclpy fixture


def quat_roll(deg: float) -> list[float]:
    h = math.radians(deg) / 2
    return [math.cos(h), math.sin(h), 0.0, 0.0]  # (w, x, y, z)


def quat_pitch(deg: float) -> list[float]:
    h = math.radians(deg) / 2
    return [math.cos(h), 0.0, math.sin(h), 0.0]


# --------------------------------------------------------------- 2b quaternion order
@pytest.mark.parametrize("deg", [0.0, 30.0, 70.0, 90.0])
def test_known_roll_and_pitch_give_that_tilt_in_wxyz(deg) -> None:
    assert math.degrees(fall_tilt_rad(quat_roll(deg))) == pytest.approx(deg, abs=1e-6)
    assert math.degrees(fall_tilt_rad(quat_pitch(deg))) == pytest.approx(deg, abs=1e-6)


def test_the_same_roll_read_as_xyzw_is_wrong_so_the_order_matters() -> None:
    w, x, y, z = quat_roll(90.0)
    assert math.degrees(fall_tilt_rad([x, y, z, w])) == pytest.approx(0.0, abs=1e-6)  # a miss


# --------------------------------------------------------------- 2a debounce + cross-check
def _feed_tilt(gate, quat, t0_s, n, dt_s=0.002, rpy=None):
    for k in range(n):
        gate.on_imu(ns(t0_s + k * dt_s), quat, rpy)


def test_single_spurious_sample_never_latches() -> None:
    g = armed()
    g.on_imu(ns(0.00), quat_roll(0.0))
    g.on_imu(ns(0.002), quat_roll(61.0))
    g.on_imu(ns(0.004), quat_roll(0.0))
    g.on_lowstate(ns(0.004), DEFAULT_U, ZERO12)
    rec = g.tick(ns(0.006))
    assert rec["mode"] != "fall_soft" and not g.faults


def test_tilt_run_must_span_the_debounce_time_and_sample_count() -> None:
    g = armed()
    _feed_tilt(g, quat_roll(70.0), 0.0, 10)  # 10 samples over 18 ms: count ok, time short
    assert not g.faults
    g.on_imu(ns(0.0205), quat_roll(70.0))  # run now spans 20.5 ms
    assert any(f.startswith("fall_detected") for f in g.faults)
    g2 = armed()
    g2.on_imu(ns(0.0), quat_roll(70.0))
    g2.on_imu(ns(0.5), quat_roll(70.0))  # long span but only 2 samples (< 3)
    assert not g2.faults
    g2.on_imu(ns(0.502), quat_roll(70.0))
    assert g2.faults


def test_a_dip_below_threshold_resets_the_run() -> None:
    g = armed()
    _feed_tilt(g, quat_roll(70.0), 0.0, 8)  # 14 ms
    g.on_imu(ns(0.016), quat_roll(10.0))
    _feed_tilt(g, quat_roll(70.0), 0.018, 8)  # another 14 ms
    assert not g.faults


def test_rpy_must_agree_to_latch() -> None:
    g = armed()
    _feed_tilt(g, quat_roll(70.0), 0.0, 20, rpy=[0.05, 0.02, 0.0])  # rpy says upright
    assert not g.faults and g.counters["imu_disagree"] == 20
    g = armed()
    _feed_tilt(g, quat_roll(70.0), 0.0, 20, rpy=[math.radians(70), 0.0, 1.0])  # agrees
    assert any(f.startswith("fall_detected") for f in g.faults)
    g = armed()  # an all-zero rpy is an unfilled message, not a second opinion
    _feed_tilt(g, quat_roll(70.0), 0.0, 20, rpy=[0.0, 0.0, 0.0])
    assert any(f.startswith("fall_detected") for f in g.faults)


# --------------------------------------------------------------- 2c IMU presence
def _tick_after_cmd(gate, t):
    gate.on_lowstate(ns(t), DEFAULT_U, ZERO12)
    gate.on_estop(ns(t), False)
    send(gate, t, DEFAULT_P, seq=int(t * 1000) + 1)
    return gate.tick(ns(t))


@pytest.mark.parametrize(
    "bad", [[0, 0, 0, 0], [float("nan"), 0, 0, 1], [2.0, 0, 0, 0], [1, 0, 0], None]
)
def test_policy_will_not_arm_until_a_valid_unit_quaternion_arrives(bad) -> None:
    g = armed(require_imu=True)
    rec = _tick_after_cmd(g, 0.01)
    assert rec["mode"] == "hold" and rec["hold_cause"] == "imu_not_valid_yet"
    assert rec["imu_seen_valid"] is False and rec["imu_age_s"] is None
    if bad is not None:
        g.on_imu(ns(0.02), bad)
    rec = _tick_after_cmd(g, 0.03)
    assert rec["mode"] == "hold" and rec["hold_cause"] == "imu_not_valid_yet"
    assert valid_unit_quat(bad if bad is not None else [0, 0, 0, 0]) is None
    g.on_imu(ns(0.04), quat_roll(5.0))
    rec = _tick_after_cmd(g, 0.05)
    assert rec["mode"] == "policy"
    assert rec["imu_valid"] is True and rec["imu_age_s"] == pytest.approx(0.01)
    assert rec["tilt_deg"] == pytest.approx(5.0, abs=1e-6)


def test_stale_imu_while_armed_is_a_sensor_fault_hold_then_closed_loop_sit() -> None:
    g = armed(require_imu=True)
    g.on_imu(ns(0.0), quat_roll(0.0))
    modes = {}
    for k in range(1, 301):  # 3 s at 100 Hz, IMU never refreshed
        t = k * 0.01
        rec = _tick_after_cmd(g, t)
        modes.setdefault(rec["mode"], t)
    assert "imu_stale" in g.faults
    assert modes["policy"] < 0.1 and 0.5 < modes["hold"] < 0.7
    assert 1.5 < modes["sit"] < 1.8  # imu_stale at ~0.5 s, hold 1.0 s, then the sit
    assert rec["sit_open_loop"] is False  # fresh LowState: closed loop


# --------------------------------------------------------------- bridge fixtures
class _Log:
    def info(self, *a):
        pass

    warn = error = info


def _msg(quat=(1.0, 0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)):
    ms = [types.SimpleNamespace(q=float(DEFAULT_U[i]), dq=0.0) for i in range(12)]
    return types.SimpleNamespace(
        motor_state=ms, imu_state=types.SimpleNamespace(quaternion=list(quat), rpy=list(rpy))
    )


FAST = SitParams(sit_s=0.4, folded_settle_s=0.1, folded_hold_s=0.2)  # bound 0.7 s (+0.2 margin)


def make_bridge(bm, monkeypatch, *, telemetry=None, imu=True):
    class _Cmd:
        def __init__(self):
            self.head = [0, 0]
            self.level_flag = 0
            self.crc = 0
            self.motor_cmd = [
                types.SimpleNamespace(mode=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0)
                for _ in range(20)
            ]

    monkeypatch.setattr(bm, "LowCmd", _Cmd)
    gate = ActuatorGate(params(sit=FAST, require_imu=False), time.monotonic_ns())
    b = object.__new__(bm.LowCmdBridge)
    b._gate, b._telemetry = gate, telemetry
    b._cfg = types.SimpleNamespace(publish_rate_hz=500.0)
    b._last_mode = b._last_fault = None
    b._init_guard()
    b.sent = []
    b._pub = types.SimpleNamespace(publish=b.sent.append)
    b.get_logger = lambda: _Log()
    b._timer = b._graph_timer = types.SimpleNamespace(cancel=lambda: None)
    b._on_lowstate(_msg())
    gate.on_estop(time.monotonic_ns(), False)
    return b


class _FullDisk:
    def __init__(self):
        self.writes = 0

    def write_tick(self, rec):
        self.writes += 1
        raise OSError(28, "No space left on device")


def _kps(b):
    return [m.motor_cmd[0].kp for m in b.sent]


# --------------------------------------------------------------- 1 never interrupt control
def test_telemetry_enospc_during_tick_does_not_stop_control_and_telemetry_is_disabled(
    bridge_module, monkeypatch
) -> None:
    disk = _FullDisk()
    b = make_bridge(bridge_module, monkeypatch, telemetry=disk)
    for _ in range(30):
        b._tick()
    assert len(b.sent) == 30 and all(k > 0 for k in _kps(b))
    assert b.guard_counts["telemetry_failures"] == bridge_module.TELEMETRY_MAX_FAILURES
    assert b._telemetry is None and disk.writes == bridge_module.TELEMETRY_MAX_FAILURES


def test_telemetry_enospc_during_shutdown_sit_runs_the_whole_sit_and_ends_folded(
    bridge_module, monkeypatch
) -> None:
    b = make_bridge(bridge_module, monkeypatch, telemetry=_FullDisk())
    t0 = time.monotonic()
    b.shutdown_sit()
    wall = time.monotonic() - t0
    kps = _kps(b)
    assert len(kps) > 100 and all(k > 0 for k in kps)
    assert kps[0] == FAST.sit_kp and kps[-1] == FAST.folded_kp
    assert wall <= FAST.shutdown_bound_s + 0.2 + 0.1


def test_gate_tick_raising_in_tick_continues_a_fallback_sit_at_sit_gains(
    bridge_module, monkeypatch
) -> None:
    b = make_bridge(bridge_module, monkeypatch)
    b._tick()
    b._tick()
    n_ok = len(b.sent)

    def boom(_now):
        raise RuntimeError("gate bug")

    b._gate.tick = boom
    for _ in range(40):
        b._tick()
        time.sleep(0.002)
    assert len(b.sent) == n_ok + 40 and b.guard_counts["gate_failures"] == 40
    fb = b.sent[n_ok:]
    assert all(m.motor_cmd[0].kp == FAST.sit_kp for m in fb)
    first = np.array([m.q for m in fb[0].motor_cmd[:12]])
    last = np.array([m.q for m in fb[-1].motor_cmd[:12]])
    assert np.abs(last - SIT_POSE_UNITREE).sum() < np.abs(first - SIT_POSE_UNITREE).sum()


def test_gate_tick_raising_in_shutdown_sit_still_sits_and_ends_folded(
    bridge_module, monkeypatch
) -> None:
    b = make_bridge(bridge_module, monkeypatch)
    b._tick()

    def boom(_now):
        raise RuntimeError("gate bug")

    b._gate.tick = boom
    t0 = time.monotonic()
    b.shutdown_sit()
    assert time.monotonic() - t0 <= FAST.shutdown_bound_s + 0.2 + 0.1
    kps = _kps(b)
    assert len(kps) > 100 and all(k > 0 for k in kps)
    assert kps[1] == FAST.sit_kp and kps[-1] == FAST.folded_kp


def test_fallback_sit_cannot_raise_on_garbage_input() -> None:
    from phoenix.sim2real.sit_policy import FallbackSit

    fb = FallbackSit(SitParams(), [float("nan")] * 3, 0.0)
    q, kp, kd, _mode = fb.step(float("nan"))
    assert kp > 0 and len(q) == 12


def test_publish_error_in_tick_is_counted_not_raised(bridge_module, monkeypatch) -> None:
    b = make_bridge(bridge_module, monkeypatch)
    b._pub = types.SimpleNamespace(publish=lambda m: (_ for _ in ()).throw(RuntimeError("ctx")))
    b._tick()
    assert b.guard_counts["publish_failures"] == 1


# --------------------------------------------------------------- 3 shutdown spins
def test_shutdown_sit_spins_callbacks_and_stays_closed_loop(bridge_module, monkeypatch) -> None:
    b = make_bridge(bridge_module, monkeypatch)
    calls = {"n": 0}

    def spin_once(node, timeout_sec=None):
        assert node is b and timeout_sec == 0
        calls["n"] += 1
        b._on_lowstate(_msg())

    monkeypatch.setattr(bridge_module.rclpy, "spin_once", spin_once, raising=False)
    recs = []
    b._report = recs.append
    b.shutdown_sit()
    assert calls["n"] > 100
    assert recs and all(r["sit_open_loop"] is False for r in recs if r["mode"] == "sit")
    assert not any("lowstate_stale" in r["faults"] for r in recs)
    assert recs[-1]["mode"] == "folded"


def test_without_spin_the_sit_goes_open_loop_after_the_lowstate_timeout(
    bridge_module, monkeypatch
) -> None:
    # Contrast: documents why the spin is needed (the old behaviour).
    b = make_bridge(bridge_module, monkeypatch)
    monkeypatch.setattr(bridge_module.rclpy, "spin_once", lambda *a, **k: None, raising=False)
    b._gate = ActuatorGate(
        params(sit=SitParams(sit_s=1.0, folded_settle_s=0.1, folded_hold_s=0.2), require_imu=False),
        time.monotonic_ns(),
    )
    b._on_lowstate(_msg())
    b._gate.on_estop(time.monotonic_ns(), False)
    recs = []
    b._report = recs.append
    b.shutdown_sit()
    assert any(r.get("sit_open_loop") for r in recs)


# --------------------------------------------------------------- 4 last-mile kp check
@pytest.mark.parametrize(
    ("kp", "kd", "q0"),
    [
        (0.0, 1.0, 0.0), (-0.0, 1.0, 0.0), (-3.0, 1.0, 0.0), (float("nan"), 1.0, 0.0),
        (float("inf"), 1.0, 0.0), (5.0, float("nan"), 0.0), (5.0, 1.0, float("nan")),
        (5.0, 1.0, float("inf")),
    ],
)  # fmt: skip
def test_unsendable_lowcmd_is_replaced_by_folded_hold(bridge_module, kp, kd, q0) -> None:
    target = [q0] * 12
    q, kp_o, kd_o, sub = bridge_module.sanitize_lowcmd(target, kp, kd)
    assert sub and kp_o == 5.0 and kd_o == 1.0 and np.allclose(q, SIT_POSE_UNITREE)
    q2, crc = bridge_module.lowcmd_fields(target, kp, kd)
    assert q2 == q and crc == bridge_module.lowcmd_fields(list(SIT_POSE_UNITREE), 5.0, 1.0)[1]


def test_sendable_lowcmd_passes_unchanged_and_wrong_length_is_substituted(bridge_module) -> None:
    good = [0.1] * 12
    assert bridge_module.sanitize_lowcmd(good, 25.0, 0.5) == (good, 25.0, 0.5, False)
    assert bridge_module.sanitize_lowcmd([0.1] * 11, 25.0, 0.5)[3] is True


def test_publish_never_sends_kp_zero_and_counts_the_fault(bridge_module, monkeypatch) -> None:
    b = make_bridge(bridge_module, monkeypatch)
    b._publish([0.0] * 12, 0.0, 0.0)
    assert b.sent[-1].motor_cmd[0].kp == 5.0 and b.sent[-1].motor_cmd[0].kd == 1.0
    assert b.guard_counts["lowcmd_substituted"] == 1
