"""Tests for the active joint-friction probe (pure numpy; no ROS, no MuJoCo)."""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real import friction_probe as fp
from phoenix.sim2real.go2_model import UNITREE_EXAMPLE_FOLDED_POSE

DT = 1.0 / fp.TICK_HZ
N = fp.N_JOINTS
STAND = np.array([0.0, 0.67, -1.3] * 4)


def run_ideal(
    q0: np.ndarray,
    fs: np.ndarray,
    g: np.ndarray,
    *,
    stuck_forever: bool = False,
) -> tuple[fp.ProbeTrace, fp.ProbeSequencer]:
    """Quasi-static Coulomb joint: stays put while |kp*(target-q) - g| <= Fs, else sits at
    the friction boundary. Returns the per-tick trace the robot node would record."""
    seq = fp.ProbeSequencer(q0)
    q = q0.copy()
    rows: dict[str, list] = {"t": [], "q": [], "target": [], "dq": []}
    prev = q.copy()
    k = 0
    while not seq.finished and k * DT < 400.0:
        t = k * DT
        target = seq.step(t, q)
        rows["t"].append(t)
        rows["q"].append(q.copy())
        rows["target"].append(target)
        rows["dq"].append((q - prev) / DT)
        prev = q.copy()
        if not stuck_forever:
            net = fp.KP * (target - q) - g
            moving = np.abs(net) > fs
            q = np.where(moving, target - (g + fs * np.sign(net)) / fp.KP, q)
        k += 1
    for _ in range(int(1.0 * fp.TICK_HZ)):  # trailing second, like the sim script
        t = k * DT
        target = seq.step(t, q)
        rows["t"].append(t)
        rows["q"].append(q.copy())
        rows["target"].append(target)
        rows["dq"].append(np.zeros(N))
        k += 1
    trace = fp.ProbeTrace(**{n: np.asarray(v) for n, v in rows.items()})
    return trace, seq


def test_worst_case_duration_is_documented_value():
    assert fp.worst_case_duration_s() == pytest.approx(2.0 + 12 * 25.5)
    assert fp.PROBE_JOINT_ORDER[0] == "FR_hip_joint" and fp.N_JOINTS == 12
    assert (fp.RATE_RAD_S, fp.BREAK_DELTA_RAD, fp.MAX_OFFSET_RAD) == (0.05, 0.02, 0.30)


def test_clamp_target_respects_limits_minus_margin():
    lo, hi = fp.probe_limits()
    wild = np.where(np.arange(N) % 2 == 0, lo - 1.0, hi + 1.0)
    out = fp.clamp_target(wild)
    assert np.all(out >= lo + fp.LIMIT_MARGIN_RAD - 1e-12)
    assert np.all(out <= hi - fp.LIMIT_MARGIN_RAD + 1e-12)
    inside = (lo + hi) / 2
    assert np.array_equal(fp.clamp_target(inside), inside)
    assert np.all(fp.clamp_target(wild, margin=0.0) == np.clip(wild, lo, hi))


def test_excursion_offset_shape():
    f = fp.excursion_offset
    assert f(-1.0, 0.1) == 0.0 and f(0.0, 0.1) == 0.0 and f(1.0, 0.0) == 0.0
    assert f(1.0, 0.1) == pytest.approx(0.05)  # rise at RATE
    assert f(2.0, 0.1) == pytest.approx(0.1)  # peak
    assert f(3.0, 0.1) == pytest.approx(0.05)  # fall at RATE
    assert f(4.0, 0.1) == 0.0 and f(99.0, 0.1) == 0.0


def test_sequencer_free_tracking_shape_and_order():
    """Joint that follows its target: breakaway at 0.02 rad, +0.5 s of ramp, one joint at a time."""
    seq = fp.ProbeSequencer(STAND)
    q = STAND.copy()
    lo, hi = fp.probe_limits()
    peaks: dict[tuple[int, int], float] = {}
    first_active: list[int] = []
    k = 0
    while not seq.finished:
        target = seq.step(k * DT, q)
        off = target - STAND
        active = np.nonzero(np.abs(off) > 1e-12)[0]
        assert len(active) <= 1
        assert np.all(target >= lo + fp.LIMIT_MARGIN_RAD - 1e-9)
        assert np.all(target <= hi - fp.LIMIT_MARGIN_RAD + 1e-9)
        if len(active):
            j = int(active[0])
            assert abs(off[j]) <= fp.MAX_OFFSET_RAD + 1e-9
            d = 1 if off[j] > 0 else -1
            peaks[(j, d)] = max(peaks.get((j, d), 0.0), abs(off[j]))
            if j not in first_active:
                first_active.append(j)
        q = target.copy()
        k += 1
        assert k * DT < fp.worst_case_duration_s()
    assert first_active == list(range(N))
    assert len(peaks) == 2 * N
    expected = fp.BREAK_DELTA_RAD + fp.POST_BREAK_S * fp.RATE_RAD_S
    for peak in peaks.values():
        assert peak == pytest.approx(expected, abs=4 * fp.RATE_RAD_S * DT)  # tick lag


def test_sequencer_ramp_rate_and_stuck_joint_reaches_cap():
    trace, seq = run_ideal(STAND, np.full(N, 1e3), np.zeros(N), stuck_forever=True)
    off0 = trace.target[:, 0] - STAND[0]
    i = np.nonzero(off0 > 1e-12)[0]
    slope = np.diff(off0[i[0] : i[0] + 50]) / DT
    assert np.allclose(slope, fp.RATE_RAD_S, atol=1e-9)
    assert off0.max() == pytest.approx(fp.MAX_OFFSET_RAD, abs=1e-9)
    assert off0.min() == pytest.approx(-fp.MAX_OFFSET_RAD, abs=1e-9)
    assert seq.finished
    assert trace.t[-1] <= fp.worst_case_duration_s() + 1.0 + 1e-6
    assert trace.t[-1] >= 0.95 * fp.worst_case_duration_s()


def test_sequencer_never_commands_past_limit_room():
    folded = np.asarray(UNITREE_EXAMPLE_FOLDED_POSE)
    lo, hi = fp.probe_limits()
    trace, _ = run_ideal(folded, np.full(N, 1e3), np.zeros(N), stuck_forever=True)
    assert np.all(trace.target >= lo + fp.LIMIT_MARGIN_RAD - 1e-9)
    assert np.all(trace.target <= hi - fp.LIMIT_MARGIN_RAD + 1e-9)
    calf = fp.PROBE_JOINT_ORDER.index("FR_calf_joint")
    room = folded[calf] - (lo[calf] + fp.LIMIT_MARGIN_RAD)
    assert trace.target[:, calf].min() == pytest.approx(folded[calf] - room, abs=1e-9)
    assert room < fp.MAX_OFFSET_RAD


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_estimator_recovers_fs_bias_kinetic_both_gravity_signs(sign):
    fs = np.linspace(0.2, 1.4, N)
    g = sign * np.linspace(0.3, 2.5, N) * np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    trace, _ = run_ideal(STAND, fs, g)
    trace.tau_est = trace.motor_torque() + 0.0  # perfect torque sensor: cross-check path
    est = fp.estimate(trace)
    for j, je in enumerate(est.joints):
        assert je.two_sided, (je.name, je.flags)
        assert je.fs_nm == pytest.approx(fs[j], abs=0.03), je.name
        assert je.bias_nm == pytest.approx(g[j], abs=0.03), je.name
        assert je.kinetic_nm is not None
        assert je.kinetic_nm == pytest.approx(fs[j], abs=0.06), je.name
        assert je.fs_tau_est_nm == pytest.approx(je.fs_nm, abs=1e-9)
        assert je.bias_tau_est_nm == pytest.approx(je.bias_nm, abs=1e-9)
        assert je.fs_lower_bound_nm is None
        assert fp.FLAG_NO_BREAK_PLUS not in je.flags


def test_estimator_zero_friction():
    trace, _ = run_ideal(STAND, np.zeros(N), np.full(N, 1.0))
    for je in fp.estimate(trace).joints:
        assert je.fs_nm is not None and abs(je.fs_nm) < 0.05


def test_estimator_no_breakaway_gives_lower_bound():
    g = np.full(N, 0.5)
    trace, _ = run_ideal(STAND, np.full(N, 50.0), g, stuck_forever=True)
    for je in fp.estimate(trace).joints:
        assert not je.two_sided and je.fs_nm is None and je.bias_nm is None
        assert fp.FLAG_NO_BREAK_PLUS in je.flags and fp.FLAG_NO_BREAK_MINUS in je.flags
        assert je.fs_lower_bound_nm == pytest.approx(fp.KP * fp.MAX_OFFSET_RAD, abs=0.2)


def test_estimator_one_sided_breakaway_is_lower_bound_only():
    """Friction between the + and - capability: only one direction can break away."""
    fs, g = np.full(N, 0.8), np.zeros(N)
    # Joint 0 gets a large gravity bias the model cannot hold: + direction breaks, - cannot
    # reach Fs - g within the offset cap because it is pinned by a huge friction proxy.
    trace, _ = run_ideal(STAND, fs, g)
    j = 1
    # Make joint j never move in the - direction by freezing q after the + run.
    off = trace.target[:, j] - STAND[j]
    first_minus = int(np.argmax(off < -1e-9))
    trace.q[first_minus:, j] = trace.q[first_minus - 1, j]
    trace.dq[first_minus:, j] = 0.0
    je = fp.estimate(trace).joints[j]
    assert fp.FLAG_NO_BREAK_MINUS in je.flags and fp.FLAG_NO_BREAK_PLUS not in je.flags
    assert je.fs_nm is None and je.fs_lower_bound_nm is not None
    assert je.fs_lower_bound_nm > 0.8


def test_estimator_flags_moved_before_ramp():
    fs, g = np.full(N, 0.6), np.full(N, 0.4)
    trace, _ = run_ideal(STAND, fs, g)
    pre = (trace.t >= fp.SETTLE_S - 0.3) & (trace.t < fp.SETTLE_S)
    drift = 0.02 * (trace.t[pre] - (fp.SETTLE_S - 0.3)) / 0.3
    trace.q[pre, 0] += drift  # joint 0 still creeping when its ramp starts
    est = fp.estimate(trace)
    assert fp.FLAG_MOVED_BEFORE_RAMP in est.joints[0].flags
    assert all(fp.FLAG_MOVED_BEFORE_RAMP not in je.flags for je in est.joints[1:])


def test_estimator_noisy_flag_and_trace_validation():
    rng = np.random.default_rng(0)
    trace, _ = run_ideal(STAND, np.full(N, 0.6), np.zeros(N))
    trace.q = trace.q + rng.normal(0.0, 0.05, trace.q.shape)  # 0.05 rad * kp = 1.25 N m noise
    assert any(fp.FLAG_NOISY in je.flags for je in fp.estimate(trace).joints)
    bad = fp.ProbeTrace(t=trace.t, q=trace.q[:, :11], target=trace.target, dq=trace.dq)
    with pytest.raises(ValueError):
        fp.estimate(bad)
