"""probe_metrics, including the registered P2 positive control for the slip detector."""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real import probe_metrics as pm

KP, KD, INERTIA, FC, DT = 25.0, 0.5, 0.02, 0.3, 0.001


def _plant(fs: float, rate: float = 0.05, ramp_s: float = 4.0, seconds: float = 6.0,
           inertia: float = INERTIA):
    """1-DOF joint under the v2 PD on a ramp, Karnopp friction (stick below ``fs``,
    slide at FC), 1 ms steps, targets held at 50 Hz. Returns 50 Hz (t, q)."""
    q = v = 0.0
    ts, qs = [], []
    for k in range(int(seconds / DT)):
        tk = k * DT
        tick = np.floor(tk * 50.0) / 50.0
        target = rate * min(tick, ramp_s)
        f = KP * (target - q) - KD * v
        if abs(v) < 1e-6 and abs(f) <= fs:
            v = 0.0
        else:
            fr = -FC * np.sign(v) if abs(v) >= 1e-6 else -FC * np.sign(f)
            v_new = v + (f + fr) / inertia * DT
            v = 0.0 if v != 0.0 and np.sign(v_new) != np.sign(v) and abs(f) <= fs else v_new
            q += v * DT
        if k % 20 == 0:
            ts.append(tk)
            qs.append(q)
    return np.asarray(ts), np.asarray(qs)


def _slips(t, q):
    return pm.micro_slips(pm.slip_events(q, t, moving=5e-4, still=3), 0.002, 0.020)


def test_p2_detector_recalls_planted_slips_at_hw_noise():
    # Registered P2 (detector not blind): planted stick-slip steps on a stuck joint with
    # the 10-07 settle noise (~3e-5 rad per 50 Hz tick); all found, nothing else.
    rng = np.random.default_rng(1)
    t = np.arange(1000) / 50.0
    q = np.full(len(t), 0.8)
    planted = [(100, 0.003, 3), (220, -0.005, 4), (400, 0.010, 4), (600, -0.015, 5), (800, 0.008, 2)]
    for k, disp, n in planted:
        q[k + 1 : k + 1 + n] += disp * np.arange(1, n + 1) / n
        q[k + 1 + n :] += disp
    ev = _slips(t, q + rng.normal(0.0, 3e-5, len(t)))
    assert pm.matched([(t[k], d) for k, d, _ in planted], ev) == len(planted)
    assert len(ev) == len(planted)


def test_p2_pure_coulomb_plant_yields_none():
    t, q = _plant(fs=FC)
    assert _slips(t, q) == []


def test_p2_noise_only_trace_yields_none():
    rng = np.random.default_rng(0)
    t = np.arange(500) / 50.0
    for sigma in (5e-5, 3e-4):  # hw settle noise is ~3e-5 rad per tick; 3e-4 is 10x worse
        assert _slips(t, 0.8 + rng.normal(0.0, sigma, len(t))) == []


# Karnopp characterisation under the v2 PD (amendment 1, 2026-10-30). As registered, P2
# expected micro-slips at Fs = 1.5 Fc; with the GO2 hip/thigh inertia (~0.022 kg m^2 at q0)
# and kd 0.5 that plant breaks away once and then slides. Kept as assertions so the
# interpretation of a micro-slip null on hardware stays tied to what Karnopp can produce.


def test_karnopp_1p5_breaks_once_then_slides():
    t, q = _plant(fs=1.5 * FC)
    assert len(_slips(t, q)) <= 1


def test_karnopp_2x_hip_thigh_slow_ramp_yields_micro_slips():
    t, q = _plant(fs=2.0 * FC, ramp_s=5.0, seconds=6.0)
    ev = _slips(t, q)
    assert len(ev) >= 5
    assert all(e[1] > 0 for e in ev)


def test_karnopp_calf_inertia_never_stick_slips():
    t, q = _plant(fs=2.0 * FC, ramp_s=5.0, seconds=6.0, inertia=0.005)
    assert _slips(t, q) == []


def test_slip_event_needs_still_ticks_on_both_sides():
    t = np.arange(12) / 50.0
    q = np.array([0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3], dtype=float) * 1e-3
    assert pm.slip_events(q, t) == [(pytest.approx(t[3]), pytest.approx(3e-3))]
    assert pm.slip_events(q[2:], t[2:]) == []  # only 1 still tick before the run


def test_matched_needs_same_sign_and_window():
    hw = [(1.0, 0.005), (2.0, -0.004)]
    assert pm.matched(hw, [(1.05, 0.006), (2.02, 0.004)]) == 1
    assert pm.matched(hw, [(1.2, 0.005)]) == 0


def test_sign_flip_floor():
    assert pm.sign_flip_p([1.0] * 12) == pytest.approx(2 / 4096)
    assert pm.sign_flip_p([1.0] * 4) == pytest.approx(0.125)
    assert pm.sign_flip_p([1.0, -1.0, 1.0, -1.0]) == 1.0


def _probe_rows():
    order = ["FR_hip_joint", "FR_thigh_joint"]
    q0 = np.array([0.0, 0.8])
    ph, act, tgt, rate = [], [], [], []

    def add(p, a, j_off, n, r=0.05):
        for _ in range(n):
            x = q0.copy()
            x[order.index(a)] += j_off
            ph.append(p), act.append(a), tgt.append(x), rate.append(r)

    for a in order:
        add("settle", a, 0.0, 5)
        for d in (1, -1):
            add("rise", a, 0.01 * d, 4)
            add("dwell_peak", a, 0.02 * d, 2)
            add("fall", a, 0.01 * d, 4)
            add("dwell_zero", a, 0.0, 3)
    add("rise", "FR_thigh_joint", 0.01, 3)  # aborted tail: no fall
    add("abort", "FR_thigh_joint", 0.0, 1)
    return order, q0, np.asarray(ph), np.asarray(act), np.asarray(tgt), np.asarray(rate)


def test_excursions_split_and_drop_incomplete():
    order, q0, ph, act, tgt, rate = _probe_rows()
    ex = pm.excursions(act, ph, rate, tgt, q0, order)
    assert [(e.joint, e.k, e.direction) for e in ex] == [(0, 0, 1), (0, 1, -1), (1, 0, 1), (1, 1, -1)]
    assert all(len(e.rows) == 13 for e in ex)
    w = pm.windows(ex[1], ph)
    assert [(n, d, len(r)) for n, d, r in w] == [("rise", -1, 6), ("fall", 1, 7)]
