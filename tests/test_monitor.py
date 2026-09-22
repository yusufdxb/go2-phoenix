"""Actuator monitor: layer accounting, residuals, calibration, persistence, fidelity.

All telemetry here is SYNTHETIC, produced by :func:`records` from a static PD
model (``q = sent - load / (s * kp) + noise``). It exercises the code paths; it is
not evidence about the robot.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.monitor.fidelity import PREREGISTERED, fidelity_report
from phoenix.monitor.health import (
    HealthMonitor,
    JointState,
    PersistenceConfig,
    degraded_joints,
    format_report,
)
from phoenix.monitor.layers import from_records, tracking_pairs
from phoenix.monitor.residual import (
    MIN_BASELINE_WINDOWS,
    WindowConfig,
    authority_ratio,
    calibrate,
    window_stats,
)
from phoenix.sim2real.go2_model import UNITREE_MOTOR_ORDER
from phoenix.sim2real.motor_crc import unitree_to_phoenix

J = UNITREE_MOTOR_ORDER.index("RR_thigh_joint")
KP = 25.0
LOAD = np.linspace(1.5, 4.0, 12)  # N m, per joint, a static stance load
BASE = np.linspace(-1.0, 1.0, 12)


def records(
    n: int,
    s=None,
    seed: int = 0,
    clip_every: int = 0,
    policy_clip_joint: int | None = None,
    mode: str = "policy",
    noise: float = 0.001,
):
    """Synthetic bridge tick records with a known authority vector ``s``."""
    rng = np.random.default_rng(seed)
    s = np.ones(12) if s is None else np.asarray(s, dtype=float)
    out = [{"record": "manifest"}]
    sent_prev = BASE.copy()
    for k in range(n):
        q = sent_prev - LOAD / (s * KP) + rng.normal(0.0, noise, 12)
        requested = BASE + 0.02 * np.sin(0.3 * k + np.arange(12))
        node_target = requested.copy()
        if policy_clip_joint is not None:
            node_target[policy_clip_joint] += 0.1  # the policy node moved it
        sent = node_target.copy()
        slew = [False] * 12
        if clip_every and k % clip_every == 0:
            slew[0] = True
        out.append(
            {
                "record": "tick",
                "t_mono_ns": int(k * 20_000_000),
                "tick": k,
                "mode": mode,
                "publish": True,
                "cmd_is_new": True,
                "q_unitree": list(q),
                "dq_unitree": [0.0] * 12,
                "requested_target_unitree": list(node_target),
                "final_target_unitree": list(sent),
                "kp": KP,
                "kd": 0.5,
                "slew_clip": slew,
                "limit_clip": [False] * 12,
                "policy": {
                    "raw_action": unitree_to_phoenix(np.zeros(12)),
                    "requested_target": unitree_to_phoenix(requested),
                    "target": unitree_to_phoenix(node_target),
                },
            }
        )
        sent_prev = sent
    return out


def stats_of(recs, cfg=None):
    return window_stats(tracking_pairs(from_records(recs)), cfg or WindowConfig())


@pytest.fixture(scope="module")
def baseline():
    runs = [stats_of(records(1500, seed=i)) for i in range(3)]
    return calibrate(runs, regime="stand", source=["synthetic"])


# ------------------------------------------------------------------ layers
def test_requested_layer_is_the_policy_request_not_the_bridge_input():
    recs = records(10, policy_clip_joint=J)
    lay = from_records(recs)
    assert np.allclose(lay.sent[:, J] - lay.requested[:, J], 0.1)
    assert np.allclose(lay.policy_node_target, lay.sent)


def test_inconsistent_command_path_is_refused():
    recs = records(3)
    recs[2]["requested_target_unitree"] = [9.0] * 12
    with pytest.raises(ValueError, match="one command path"):
        from_records(recs)


def test_policy_node_clip_makes_samples_invalid():
    pairs = tracking_pairs(from_records(records(50, policy_clip_joint=J)))
    assert not pairs.valid[:, J].any()
    assert pairs.valid[:, (J + 1) % 12].all()


def test_bridge_slew_flag_makes_samples_invalid():
    pairs = tracking_pairs(from_records(records(40, clip_every=2)))
    assert pairs.valid[:, 0].mean() == pytest.approx(0.5, abs=0.05)


def test_non_policy_ticks_never_pair():
    pairs = tracking_pairs(from_records(records(20, mode="hold")))
    assert not pairs.valid.any()


# ---------------------------------------------------------------- residual
def test_calibration_needs_enough_windows():
    with pytest.raises(ValueError, match="usable nominal windows"):
        calibrate([stats_of(records(10 * 50))], regime="stand")
    assert MIN_BASELINE_WINDOWS >= 20


def test_nominal_authority_is_one(baseline):
    s = authority_ratio(stats_of(records(1000, seed=9)), baseline)
    assert np.nanmedian(s, axis=0) == pytest.approx(np.ones(12), abs=0.03)


@pytest.mark.parametrize("true_s", [0.5, 0.6, 0.8])
def test_authority_ratio_recovers_injected_scale(baseline, true_s):
    sv = np.ones(12)
    sv[J] = true_s
    s = authority_ratio(stats_of(records(1000, s=sv, seed=11)), baseline)
    med = np.nanmedian(s, axis=0)
    assert med[J] == pytest.approx(true_s, rel=0.05)
    assert np.delete(med, J) == pytest.approx(np.ones(11), abs=0.03)


def test_threshold_is_never_closer_to_one_than_min_effect(baseline):
    assert max(baseline.threshold) <= 1.0 - baseline.min_effect + 1e-12


def test_windows_with_too_few_valid_samples_are_nan():
    st = stats_of(records(200, clip_every=1))  # joint 0 clipped every tick
    assert np.all(np.isnan(st.rms_error[:, 0]))
    assert np.all(np.isfinite(st.rms_error[:, 1]))


def test_torque_gain_ratio_from_tau_est():
    recs = records(200)
    sv = np.ones(12)
    sv[J] = 0.6
    for r in recs[1:]:
        e = np.asarray(r["final_target_unitree"]) - np.asarray(r["q_unitree"])
        r["tau_est_unitree"] = list(sv * KP * e)  # what a scaled gain delivers
    st = stats_of(recs)
    g = np.nanmedian(st.torque_gain_ratio, axis=0)
    # e here uses the SAME tick's q, the monitor pairs k with k+1; the static model
    # makes those equal up to noise
    assert g[J] == pytest.approx(0.6, rel=0.1)


# -------------------------------------------------------------- persistence
def _feed(mon, s_rows):
    rep = None
    for row in s_rows:
        rep = mon.update(row)
    return rep


def test_single_bad_window_is_only_suspect(baseline):
    mon = HealthMonitor(baseline)
    rows = [np.ones(12)] * 9 + [np.where(np.arange(12) == J, 0.5, 1.0)]
    rep = _feed(mon, rows)
    assert rep[J].state == JointState.SUSPECT.value
    assert degraded_joints(rep) == []


def test_persistent_drop_is_degraded_and_localised(baseline):
    mon = HealthMonitor(baseline)
    row = np.where(np.arange(12) == J, 0.6, 1.0)
    rep = _feed(mon, [np.ones(12)] * 2 + [row] * 8)
    assert rep[J].state == JointState.DEGRADED.value
    assert [h.joint for h in degraded_joints(rep)] == ["RR_thigh_joint"]
    assert rep[J].s_hat == pytest.approx(0.6)
    assert all(h.state == "NOMINAL" for i, h in enumerate(rep) if i != J)


def test_recovery_needs_hysteresis(baseline):
    mon = HealthMonitor(baseline, PersistenceConfig(n=10, k_of_n=8, recover_max=2))
    bad = np.where(np.arange(12) == J, 0.6, 1.0)
    _feed(mon, [bad] * 10)
    rep = _feed(mon, [np.ones(12)] * 5)  # 5 of last 10 still bad
    assert rep[J].state == JointState.DEGRADED.value
    rep = _feed(mon, [np.ones(12)] * 3)  # 2 of last 10 bad
    assert rep[J].state != JointState.DEGRADED.value


def test_many_joints_at_once_is_a_global_shift_not_a_fault(baseline):
    mon = HealthMonitor(baseline)
    rep = _feed(mon, [np.full(12, 0.7)] * 10)
    assert all(h.state == JointState.GLOBAL_SHIFT.value for h in rep)
    assert degraded_joints(rep) == []


def test_unusable_windows_are_insufficient_not_nominal(baseline):
    mon = HealthMonitor(baseline)
    row = np.ones(12)
    row[J] = np.nan
    rep = _feed(mon, [row] * 10)
    assert rep[J].state == JointState.INSUFFICIENT_DATA.value
    assert rep[J].s_hat is None


def test_no_false_positive_on_long_nominal_run(baseline):
    """Nominal synthetic run of 120 windows: no joint may reach DEGRADED."""
    s = authority_ratio(stats_of(records(120 * 50, seed=77)), baseline)
    mon = HealthMonitor(baseline)
    for row in s:
        rep = mon.update(row)
        assert degraded_joints(rep) == []


def test_format_report_has_one_line_per_joint(baseline):
    mon = HealthMonitor(baseline)
    txt = format_report(_feed(mon, [np.ones(12)] * 10))
    assert len(txt.splitlines()) == 13 and "RR_thigh" in txt


@pytest.mark.parametrize(
    "kw", [dict(k_of_n=11), dict(recover_max=8), dict(min_usable=0), dict(max_localised=0)]
)
def test_persistence_config_validation(kw):
    with pytest.raises(ValueError):
        PersistenceConfig(**kw)


# ----------------------------------------------------------------- fidelity
def test_fidelity_passes_clean_long_run():
    rep = fidelity_report(from_records(records(600)))
    assert rep["verdict"] == "PASS", rep["reasons"]
    assert rep["altered_fraction"] == 0.0


def test_fidelity_fails_short_run():
    rep = fidelity_report(from_records(records(100)))
    assert rep["verdict"] == "FAIL" and any("authority_s" in r for r in rep["reasons"])


def test_fidelity_counts_policy_node_alterations():
    rep = fidelity_report(from_records(records(600, policy_clip_joint=J)))
    assert rep["altered_fraction"] == pytest.approx(1 / 12)
    assert rep["attribution"]["policy_node_slew_clip_fraction"] == pytest.approx(1 / 12)
    assert rep["per_joint"]["RR_thigh_joint"]["rms_distortion_rad"] == pytest.approx(0.1)
    assert rep["verdict"] == "FAIL"


def test_fidelity_catches_one_bad_joint_that_the_average_hides():
    recs = records(600)
    for r in recs[1:]:  # one joint altered by 2 mrad on every tick: 1/12 = 8 % overall
        r["policy"]["requested_target"][4] -= 0.002
    rep = fidelity_report(from_records(recs), PREREGISTERED)
    assert rep["verdict"] == "FAIL"
    assert any(r.startswith(tuple(UNITREE_MOTOR_ORDER)) for r in rep["reasons"])


def test_fidelity_with_no_policy_ticks_fails():
    rep = fidelity_report(from_records(records(50, mode="hold")))
    assert rep["verdict"] == "FAIL" and rep["policy_ticks"] == 0


def test_preregistered_thresholds_are_frozen_values():
    assert (PREREGISTERED.max_altered_fraction, PREREGISTERED.max_rms_distortion_rad) == (
        0.05,
        0.01,
    )
    assert PREREGISTERED.min_authority_s == 10.0
