"""Synthetic-data tests for the pure E2 actuator-effectiveness functions."""

from __future__ import annotations

import json

import numpy as np
import pytest

from phoenix.sim2real import actuator_check as ac
from phoenix.sim2real.go2_model import POLICY_JOINT_ORDER, UNITREE_MOTOR_ORDER

NS = 1_000_000_000


def _tick(t_s, mode="policy", kp=25.0, q=None, tgt=None, dq=None, kd=0.5):
    z = [0.0] * 12
    return {
        "record": "tick",
        "mode": mode,
        "t_mono_ns": int(round(t_s * NS)),
        "q_unitree": list(q if q is not None else z),
        "final_target_unitree": list(tgt if tgt is not None else z),
        "dq_unitree": list(dq if dq is not None else z),
        "kp_eff": kp,
        "kd_eff": kd,
    }


def _summary(err_by_name, kp=25.0):
    names = list(POLICY_JOINT_ORDER)
    return {
        "err": {n: float(err_by_name[n]) for n in names},
        "kp": {n: float(kp) for n in names},
        "kd": {n: 0.5 for n in names},
        "n": 10,
    }


def _errs(hip=0.04, thigh=0.05, calf=0.2):
    return {
        n: {"hip": hip, "thigh": thigh, "calf": calf}[ac.joint_group(n)] for n in POLICY_JOINT_ORDER
    }


# ------------------------------------------------------------------ window selection


def test_window_skips_first_1p5s_and_non_policy_and_handover_kp():
    ticks = [_tick(0.0, mode="hold"), _tick(1.0, mode="standup")]
    # policy starts at t=10; kp blends 60 -> 25 then settles
    ticks += [_tick(10.0 + 0.1 * i, kp=60.0 - 35.0 * min(1.0, 0.1 * i)) for i in range(0, 40)]
    ticks += [_tick(14.0, mode="hold")]  # non-policy tick after authority must be ignored
    win = ac.select_settled_window(ticks)
    ts = [t["t_mono_ns"] / NS for t in win]
    assert min(ts) >= 11.5 - 1e-9  # 1.5 s after first policy tick
    assert all(t["kp_eff"] == 25.0 for t in win)
    assert all(t["mode"] == "policy" for t in win)
    assert max(ts) == pytest.approx(13.9)


def test_window_drops_ticks_with_kp_not_equal_to_final():
    ticks = [_tick(0.0), _tick(2.0, kp=30.0), _tick(3.0, kp=25.0), _tick(4.0, kp=25.0)]
    win = ac.select_settled_window(ticks)
    assert [t["t_mono_ns"] // NS for t in win] == [3, 4]


def test_window_empty_without_policy_ticks_and_summarize_raises():
    ticks = [_tick(0.0, mode="hold")]
    assert ac.select_settled_window(ticks) == []
    with pytest.raises(ValueError):
        ac.summarize_hw([ticks])


# ------------------------------------------------------------------ joint order mapping


def test_unitree_to_policy_maps_by_name_not_index():
    vals = list(range(12))  # value == Unitree index
    m = ac.unitree_to_policy(vals)
    for i, name in enumerate(UNITREE_MOTOR_ORDER):
        assert m[name] == i
    # The orders genuinely differ, so an assumed index mapping would be wrong.
    assert m["FL_hip_joint"] == 3 and POLICY_JOINT_ORDER.index("FL_hip_joint") == 0
    assert m["RL_calf_joint"] == 11 and POLICY_JOINT_ORDER.index("RL_calf_joint") == 10
    with pytest.raises(ValueError):
        ac.unitree_to_policy([0.0] * 11)


def test_summarize_hw_maps_target_minus_q_to_the_right_joint():
    err_u = [0.0] * 12
    err_u[UNITREE_MOTOR_ORDER.index("RR_calf_joint")] = 0.3
    ticks = [_tick(10.0 + 0.1 * i, tgt=err_u) for i in range(40)]
    s = ac.summarize_hw([ticks])
    assert s["err"]["RR_calf_joint"] == pytest.approx(0.3)
    assert all(abs(v) < 1e-12 for n, v in s["err"].items() if n != "RR_calf_joint")
    assert s["kp"]["FL_hip_joint"] == 25.0 and s["n"] == 25  # t >= 11.5 .. 13.9


def test_summarize_hw_pools_files_and_uses_median():
    def f(e):
        tgt = [e] * 12
        return [_tick(10.0 + 0.1 * i, tgt=tgt) for i in range(40)]

    s = ac.summarize_hw([f(0.1), f(0.2), f(0.9)])
    assert s["err"]["FL_thigh_joint"] == pytest.approx(0.2)


def test_summarize_trace_window_and_name_mapping():
    names = list(reversed(POLICY_JOINT_ORDER))  # trace columns in a different order
    t = np.arange(1, 501) * 0.02
    q = np.zeros((500, 12))
    tg = np.zeros((500, 12))
    tg[:, names.index("FL_calf_joint")] = 0.5
    tg[t < 2.0, :] = 9.0  # transient in the skipped part must not matter
    kp = dict.fromkeys(POLICY_JOINT_ORDER, 25.0)
    s = ac.summarize_trace(t, q, tg, names, kp, kp)
    assert s["err"]["FL_calf_joint"] == pytest.approx(0.5)
    assert s["err"]["FR_calf_joint"] == pytest.approx(0.0)
    assert s["n"] == int((t >= 2.0).sum())
    with pytest.raises(ValueError):
        ac.summarize_trace(t[:50], q[:50], tg[:50], names, kp, kp)


# ------------------------------------------------------------------ ratio + exclusion


def test_ratio_recovers_g_and_normalises_kp():
    ref = _summary(_errs())
    hw = _summary({n: v / 0.75 if ac.joint_group(n) == "calf" else v for n, v in _errs().items()})
    est = ac.estimate_effectiveness(ref, hw)
    assert est["g"]["FL_calf_joint"] == pytest.approx(0.75)
    assert est["g"]["FL_thigh_joint"] == pytest.approx(1.0)
    assert est["calf_mean"] == pytest.approx(0.75)
    assert est["kp_normalised"] is False
    # Hardware commanded kp 12.5 (half): same err means g = 2 (kp_ref*err / kp_hw*err)
    hw2 = _summary(_errs(), kp=12.5)
    est2 = ac.estimate_effectiveness(ref, hw2)
    assert est2["g"]["FL_calf_joint"] == pytest.approx(2.0)
    assert est2["kp_normalised"] is True


def test_low_snr_joint_excluded_and_hips_not_in_rule():
    errs = _errs(hip=0.01, thigh=0.05, calf=0.2)
    errs["RL_thigh_joint"] = 0.019  # below 0.02 rad -> excluded
    ref = _summary(errs)
    hw_err = dict(errs)
    hw_err["RL_thigh_joint"] = 0.019 / 5.0  # would give g=5 and wreck the mean if included
    est = ac.estimate_effectiveness(ref, _summary(hw_err))
    assert "RL_thigh_joint" in est["excluded"] and "low SNR" in est["excluded"]["RL_thigh_joint"]
    assert all(h in est["excluded"] for h in POLICY_JOINT_ORDER if ac.joint_group(h) == "hip")
    assert est["n_rule_used"] == 7 and est["n_calf_used"] == 4
    assert est["global_mean"] == pytest.approx(1.0)


def test_hip_with_snr_is_reported_but_still_outside_the_rule():
    ref = _summary(_errs(hip=0.05))
    hw = _summary(
        {n: v / 0.5 if ac.joint_group(n) == "hip" else v for n, v in _errs(hip=0.05).items()}
    )
    est = ac.estimate_effectiveness(ref, hw)
    assert est["g"]["FL_hip_joint"] == pytest.approx(0.5)
    assert (
        est["global_mean"] == pytest.approx(1.0)
        and est["ratio_verdict_uncalibrated"] == ac.VERDICT_GO
    )


def test_zero_hw_error_gives_undefined_g_and_exclusion():
    ref = _summary(_errs())
    e = _errs()
    e["FL_calf_joint"] = 0.0
    est = ac.estimate_effectiveness(ref, _summary(e))
    assert est["g"]["FL_calf_joint"] is None and "FL_calf_joint" in est["excluded"]
    assert est["n_calf_used"] == 3


# ------------------------------------------------------------------ verdict boundaries


@pytest.mark.parametrize(
    ("calf", "glob", "want"),
    [
        (0.80, 0.90, ac.VERDICT_STOP),  # calf boundary is strict
        (0.8001, 0.90, ac.VERDICT_GO),
        (0.90, 0.75, ac.VERDICT_STOP),  # global boundary is strict
        (0.90, 0.7501, ac.VERDICT_GO),
        (0.79, 0.99, ac.VERDICT_STOP),
        (0.99, 0.74, ac.VERDICT_STOP),
        (None, 0.99, ac.VERDICT_STOP),
        (0.99, None, ac.VERDICT_STOP),
    ],
)
def test_verdict_boundaries(calf, glob, want):
    assert ac.verdict(calf, glob) == want


def test_estimate_verdict_end_to_end_and_g_true_passthrough():
    ref = _summary(_errs())
    hw = _summary(_errs())
    hw["g_true"] = {n: 1.0 for n in POLICY_JOINT_ORDER}
    est = ac.estimate_effectiveness(ref, hw)
    assert (
        est["ratio_verdict_uncalibrated"] == ac.VERDICT_GO and est["g_true"]["FL_calf_joint"] == 1.0
    )
    weak = _summary({n: v / 0.75 if ac.joint_group(n) == "calf" else v for n, v in _errs().items()})
    assert ac.estimate_effectiveness(ref, weak)["ratio_verdict_uncalibrated"] == ac.VERDICT_STOP


def test_load_ticks_keeps_only_tick_records(tmp_path):
    p = tmp_path / "b.jsonl"
    p.write_text(
        json.dumps({"record": "manifest"})
        + "\n\n"
        + json.dumps(_tick(1.0))
        + "\n"
        + json.dumps({"record": "end"})
        + "\n"
    )
    (ticks,) = ac.load_ticks([p])
    assert len(ticks) == 1 and ticks[0]["record"] == "tick"


# ------------------------------------------------------------------ --factor parsing


def test_parse_factor_valid_and_invalid():
    assert ac.parse_factor("calf=0.85") == ("calf", 0.85)
    assert ac.parse_factor("all=1") == ("all", 1.0)
    for bad in ("calf", "knee=0.5", "calf=", "calf=abc", "calf=0", "calf=-1", "calf=nan"):
        with pytest.raises(ValueError):
            ac.parse_factor(bad)


def test_apply_factors_later_wins_per_joint():
    f = ac.apply_factors([1.0] * 12, POLICY_JOINT_ORDER, ["all=0.7", "calf=0.9"])
    for n, v in zip(POLICY_JOINT_ORDER, f, strict=True):
        assert v == (0.9 if ac.joint_group(n) == "calf" else 0.7)
    f2 = ac.apply_factors([1.0] * 12, POLICY_JOINT_ORDER, ["calf=0.9", "all=0.7"])
    assert set(f2) == {0.7}
    base = [0.75 if ac.joint_group(n) == "calf" else 1.0 for n in POLICY_JOINT_ORDER]
    f3 = ac.apply_factors(base, POLICY_JOINT_ORDER, ["thigh=0.5"])
    assert f3[POLICY_JOINT_ORDER.index("FL_thigh_joint")] == 0.5
    assert f3[POLICY_JOINT_ORDER.index("FL_calf_joint")] == 0.75


# ------------------------------------------------------------------ primary decision


def _calf_ref(calf_err, thigh_err=0.05):
    return _summary(_errs(thigh=thigh_err, calf=calf_err))


NOM, B_CALF, B_ALL = _calf_ref(0.20), _calf_ref(0.26), _calf_ref(0.30)


def test_decide_go_needs_both_tests_strict():
    ok = ac.decide(_calf_ref(0.25), B_CALF, B_ALL, NOM)
    assert ok["verdict"] == ac.VERDICT_GO and ok["calf_test_go"] and ok["global_test_go"]
    # Exactly on the calf boundary: strict, so STOP.
    on = ac.decide(_calf_ref(0.26), B_CALF, B_ALL, NOM)
    assert on["verdict"] == ac.VERDICT_STOP and not on["calf_test_go"] and on["global_test_go"]
    # Above the calf boundary: STOP.
    assert ac.decide(_calf_ref(0.27), B_CALF, B_ALL, NOM)["verdict"] == ac.VERDICT_STOP


def test_decide_global_test_can_be_the_binding_one():
    # all-0.75 plant sags LESS on calves than calf-0.80 plant: global test binds.
    b_calf, b_all = _calf_ref(0.30), _calf_ref(0.26)
    d = ac.decide(_calf_ref(0.28), b_calf, b_all, NOM)
    assert d["calf_test_go"] and not d["global_test_go"] and d["verdict"] == ac.VERDICT_STOP
    d2 = ac.decide(_calf_ref(0.26), b_calf, b_all, NOM)  # equal to the global boundary
    assert not d2["global_test_go"]


def test_decide_sign_and_monotonicity_check():
    with pytest.raises(ValueError):
        ac.decide(NOM, _calf_ref(0.15), B_ALL, NOM)  # boundary stronger than nominal
    with pytest.raises(ValueError):
        ac.decide(NOM, B_CALF, B_ALL, _calf_ref(-0.2))  # wrong sign convention


def test_decide_reports_thigh_informationally_only():
    d = ac.decide(_calf_ref(0.25, thigh_err=9.0), B_CALF, B_ALL, NOM)
    assert d["verdict"] == ac.VERDICT_GO  # absurd thigh err does not change the verdict
    assert d["thigh_informational"]["thigh_err_hw"] == pytest.approx(9.0)


# ------------------------------------------------------------------ calibrated inverse


PTS = [(0.6, 0.40), (0.8, 0.30), (1.0, 0.20), (1.2, 0.10)]


def test_invert_sweep_interpolates_and_clips():
    g, status = ac.invert_sweep(PTS, 0.25)
    assert g == pytest.approx(0.9) and status == "in sweep"
    g, status = ac.invert_sweep(list(reversed(PTS)), 0.30)
    assert g == pytest.approx(0.8)
    g, status = ac.invert_sweep(PTS, 0.50)
    assert g == 0.6 and "outside sweep" in status
    g, status = ac.invert_sweep(PTS, 0.05)
    assert g == 1.2 and "outside sweep" in status
    with pytest.raises(ValueError):
        ac.invert_sweep([(0.6, 0.2), (0.8, 0.3)], 0.25)  # err must fall with g


def test_sweep_points_requires_calf_only_refs():
    def ref(gc, gt=1.0):
        r = _calf_ref(0.2 / gc)
        r["g_true"] = {n: (gc if ac.joint_group(n) == "calf" else gt) for n in POLICY_JOINT_ORDER}
        return r

    pts = ac.sweep_points([ref(0.8), ref(0.6)])
    assert [p[0] for p in pts] == [0.6, 0.8] and pts[0][1] == pytest.approx(0.2 / 0.6)
    with pytest.raises(ValueError):
        ac.sweep_points([ref(0.8, gt=0.8)])
