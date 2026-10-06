"""Synthetic-data tests for the pure E2 actuator-effectiveness functions."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from phoenix.sim2real import actuator_check as ac
from phoenix.sim2real.go2_model import POLICY_JOINT_ORDER, UNITREE_MOTOR_ORDER

NS = 1_000_000_000
REPO = Path(__file__).resolve().parents[1]
SHIPPED = REPO / "configs/sim2real/e2/walk_iter2750"
LOCK = REPO / "configs/sim2real/locks/deploy_walk_iter2750_stand.lock.yaml"
IDENT = {k: c * 64 for k, c in zip(ac.IDENTITY_KEYS, "abcd", strict=True)}


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


# ------------------------------------------------------------------ synthetic telemetry


def _u(by_name, default=0.0):
    """Per-joint values by policy name -> Unitree-order list."""
    return [by_name.get(n, default) for n in UNITREE_MOTOR_ORDER]


def _episode(start=10.0, dur=6.0, dt=0.02, kp_final=25.0, err=None, dq=0.0, **over):
    """One policy episode: 1 s kp blend, then a still, healthy stand at zero command."""
    tgt = _u(err or {})
    out = []
    for k in range(int(round(dur / dt))):
        a = min(1.0, k / 50)
        t = _tick(start + k * dt, kp=60.0 - (60.0 - kp_final) * a, tgt=tgt, dq=[dq] * 12)
        t.update(
            handover_alpha=a,
            lowstate_fresh=True,
            fault=None,
            faults=[],
            tilt_deg=1.0,
            policy={"velocity_command_fed": [0.0, 0.0, 0.0], "cmd_vel_received": [0.0] * 3},
        )
        t.update(over)
        out.append(t)
    return out


def _manifest(ident=IDENT, live=True):
    return {
        "record": "manifest",
        "live": live,
        "stage": "G",
        "artifacts": {r: {"sha256": ident[r]} for r in ac.ARTIFACT_ROLES},
        "deploy_config": {"semantic_sha256": ident["config_semantic_sha256"]},
        "code_identity": {"sha": "abc1234", "dirty": False},
        "lock": {"name": "walk-iter2750-stand"},
    }


def _file(ticks, manifest=None, path="bridge.jsonl"):
    return {
        "path": path,
        "manifest": _manifest() if manifest is None else manifest,
        "n_manifests": 1,
        "ticks": ticks,
    }


def _with(ticks, i, **kw):
    ticks = copy.deepcopy(ticks)
    ticks[i].update(kw)
    return ticks


# ------------------------------------------------------------------ gate 4: window quality


def test_window_starts_1s_after_blend_end_not_after_first_policy_tick():
    ticks = [_tick(0.0, mode="hold"), _tick(1.0, mode="standup")]
    ticks += _episode(start=10.0)
    win, rep = ac.settle_window(ticks)
    ts = [t["t_mono_ns"] / NS for t in win]
    assert min(ts) == pytest.approx(12.0)  # blend ends at 11.0, window 1.0 s later
    assert max(ts) == pytest.approx(15.98)
    assert all(t["kp_eff"] == 25.0 and t["mode"] == "policy" for t in win)
    assert rep["passed"] and rep["window_s"] == pytest.approx(3.98) and rep["episodes"] == 1


def test_window_1_tick_refused():  # reviewer probe 2
    with pytest.raises(ac.Refusal, match="settled window|no policy ticks"):
        ac.summarize_hw([_episode(dur=2.04)])  # blend end + 1 s lands on the last tick
    only = _episode(dur=0.02)
    with pytest.raises(ac.Refusal):
        ac.summarize_hw([only])


@pytest.mark.parametrize("dur", [3.0, 4.0])
def test_short_window_refused_stage_f_style(dur):
    # 2 s of policy, or a 4 s run whose settled part is 1.98 s: both below 2.0 s.
    with pytest.raises(ac.Refusal, match=r"need >= 2\.0 s"):
        ac.summarize_hw([_episode(dur=dur)])


def test_window_accepts_stage_g_length():
    assert ac.summarize_hw([_episode(dur=4.5)])["n"] >= 100


def test_two_policy_episodes_refused():
    ticks = _episode(start=10.0, dur=6.0) + [_tick(17.0, mode="hold")]
    ticks += _episode(start=18.0, dur=6.0)
    with pytest.raises(ac.Refusal, match="2 separate policy episodes"):
        ac.summarize_hw([ticks])


def test_moving_robot_refused_on_dq_rms():
    with pytest.raises(ac.Refusal, match="robot is moving"):
        ac.summarize_hw([_episode(dq=0.06)])
    assert ac.summarize_hw([_episode(dq=0.04)])["dq_rms"]["FL_calf_joint"] == pytest.approx(0.04)


def test_nonzero_velocity_command_refused():
    bad = {"velocity_command_fed": [0.05, 0.0, 0.0], "cmd_vel_received": [0.0] * 3}
    with pytest.raises(ac.Refusal, match="commanded velocity"):
        ac.summarize_hw([_with(_episode(), 200, policy=bad)])


def test_fault_stale_lowstate_and_tilt_ticks_refused():
    with pytest.raises(ac.Refusal, match="fault"):
        ac.summarize_hw([_with(_episode(), 150, fault="estop")])
    with pytest.raises(ac.Refusal, match="fault"):
        ac.summarize_hw([_with(_episode(), 150, faults=["imu"])])
    with pytest.raises(ac.Refusal, match="stale lowstate"):
        ac.summarize_hw([_with(_episode(), 150, lowstate_fresh=False)])
    with pytest.raises(ac.Refusal, match="tilt"):
        ac.summarize_hw([_with(_episode(), 150, tilt_deg=12.0)])


def test_fault_in_the_blend_is_not_in_the_window():
    # A tick before the window (still blending) is outside the gates: window-only checks.
    assert ac.summarize_hw([_with(_episode(), 10, lowstate_fresh=False)])["n"] > 0


def test_kp_change_inside_window_refused_and_gap_refused():
    t = _episode()
    t[200]["kp_eff"] = 30.0
    with pytest.raises(ac.Refusal, match="gains change"):
        ac.summarize_hw([t])
    gap = _episode()[:150] + _episode(start=10.0 + 150 * 0.02 + 0.5)[:150]
    with pytest.raises(ac.Refusal, match="tick gap"):
        ac.summarize_hw([gap])


def test_window_empty_without_policy_ticks_and_summarize_raises():
    ticks = [_tick(0.0, mode="hold")]
    with pytest.raises(ac.Refusal):
        ac.select_settled_window(ticks)
    with pytest.raises(ac.Refusal):
        ac.summarize_hw([ticks])
    with pytest.raises(ac.Refusal):
        ac.summarize_hw([])


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
    s = ac.summarize_hw([_episode(err={"RR_calf_joint": 0.3})])
    assert s["err"]["RR_calf_joint"] == pytest.approx(0.3)
    assert all(abs(v) < 1e-12 for n, v in s["err"].items() if n != "RR_calf_joint")
    assert s["kp"]["FL_hip_joint"] == 25.0 and s["n"] == 200  # k = 100..299


def test_summarize_hw_pools_files_and_uses_median():
    def f(e):
        return _episode(err=dict.fromkeys(POLICY_JOINT_ORDER, e))

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


def test_load_telemetry_keeps_manifest_and_flags_missing_or_garbled(tmp_path):
    p = tmp_path / "b.jsonl"
    p.write_text(json.dumps(_manifest()) + "\n" + json.dumps(_tick(1.0)) + "\n")
    (f,) = ac.load_telemetry([p])
    assert f["manifest"]["live"] is True and f["n_manifests"] == 1 and len(f["ticks"]) == 1
    q = tmp_path / "nomanifest.jsonl"
    q.write_text(json.dumps(_tick(1.0)) + "\n")
    (g,) = ac.load_telemetry([q])
    assert g["manifest"] is None
    with pytest.raises(ac.Refusal, match="no manifest|not a manifest"):
        ac.check_telemetry_manifest(g, IDENT)
    r = tmp_path / "bad.jsonl"
    r.write_text("{not json\n")
    with pytest.raises(ac.Refusal, match="unparsable"):
        ac.load_telemetry([r])


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


def _ref(calf=1.0, all_=None, calf_err=0.2, thigh_err=0.05, ident=None, **over):
    """A simulator reference with g_true per group (``all_`` overrides every group)."""
    r = _summary(_errs(thigh=thigh_err, calf=calf_err))
    g = {
        n: (all_ if all_ is not None else calf if ac.joint_group(n) == "calf" else 1.0)
        for n in POLICY_JOINT_ORDER
    }
    r.update(schema=ac.SCHEMA, source="sim", fell=False, g_true=g, identity=dict(ident or IDENT))
    r.update(over)
    return r


def _calf_ref(calf_err, thigh_err=0.05):
    return _ref(calf_err=calf_err, thigh_err=thigh_err)


NOM = _ref(calf_err=0.20)
B_CALF = _ref(calf=0.8, calf_err=0.26)
B_ALL = _ref(all_=0.75, calf_err=0.30)


def test_decide_go_needs_both_tests_strict():
    ok = ac.decide(_calf_ref(0.25), B_CALF, B_ALL, NOM)
    assert ok["verdict"] == ac.VERDICT_GO and ok["calf_test_go"] and ok["global_test_go"]
    assert ok["calf_boundary_label"] == "calf=0.80" and ok["global_boundary_label"] == "all=0.75"
    # Exactly on the calf boundary: strict, so STOP.
    on = ac.decide(_calf_ref(0.26), B_CALF, B_ALL, NOM)
    assert on["verdict"] == ac.VERDICT_STOP and not on["calf_test_go"] and on["global_test_go"]
    # Above the calf boundary: STOP.
    assert ac.decide(_calf_ref(0.27), B_CALF, B_ALL, NOM)["verdict"] == ac.VERDICT_STOP


def test_decide_global_test_can_be_the_binding_one():
    # all-0.75 plant sags LESS on calves than calf-0.80 plant: global test binds.
    b_calf, b_all = _ref(calf=0.8, calf_err=0.30), _ref(all_=0.75, calf_err=0.26)
    d = ac.decide(_calf_ref(0.28), b_calf, b_all, NOM)
    assert d["calf_test_go"] and not d["global_test_go"] and d["verdict"] == ac.VERDICT_STOP
    d2 = ac.decide(_calf_ref(0.26), b_calf, b_all, NOM)  # equal to the global boundary
    assert not d2["global_test_go"]


def test_decide_sign_and_monotonicity_check():
    with pytest.raises(ac.Refusal):
        ac.decide(NOM, _ref(calf=0.8, calf_err=0.15), B_ALL, NOM)  # boundary stronger than nominal
    with pytest.raises(ac.Refusal):
        ac.decide(NOM, B_CALF, B_ALL, _ref(calf_err=-0.2))  # wrong sign convention
    with pytest.raises(ac.Refusal, match="nominal reference is required"):
        ac.decide(NOM, B_CALF, B_ALL)


def test_decide_reports_thigh_informationally_only():
    d = ac.decide(_calf_ref(0.25, thigh_err=9.0), B_CALF, B_ALL, NOM)
    assert d["verdict"] == ac.VERDICT_GO  # absurd thigh err does not change the verdict
    assert d["thigh_informational"]["thigh_err_hw"] == pytest.approx(9.0)


# ------------------------------------------------------------------ gate 2: boundary identity


def _shipped(name):
    return json.loads((SHIPPED / name).read_text())


def test_probe_wrong_boundaries_calf070_all070_refused():  # reviewer probe 1
    nom, c70 = _shipped("ref_nominal.json"), _shipped("ref_calf_0.70.json")
    hw = _shipped("ref_calf_1.00.json")
    with pytest.raises(ac.Refusal, match="boundary-calf is not the required plant"):
        ac.decide(hw, c70, _shipped("ref_all_0.75.json"), nom)
    with pytest.raises(ac.Refusal, match="boundary-all is not the required plant"):
        ac.decide(hw, _shipped("ref_calf_0.80.json"), c70, nom)  # calf-0.70 as 'all' boundary
    all70 = _ref(all_=0.70, calf_err=0.28)  # a genuine all=0.70 plant is not all=0.75 either
    with pytest.raises(ac.Refusal, match="boundary-all"):
        ac.decide(NOM, B_CALF, all70, NOM)


def test_boundary_swapped_or_wrong_nominal_refused():
    with pytest.raises(ac.Refusal, match="boundary-calf"):
        ac.decide(NOM, B_ALL, B_CALF, NOM)
    with pytest.raises(ac.Refusal, match="nominal is not the required plant"):
        ac.decide(NOM, B_CALF, B_ALL, _ref(calf=0.9, calf_err=0.20))
    with pytest.raises(ac.Refusal, match="boundary-calf"):  # calf 0.80 but thighs also scaled
        ac.decide(NOM, _ref(all_=0.8, calf_err=0.26), B_ALL, NOM)


def test_refs_must_share_schema_fell_artifacts_and_gains():
    with pytest.raises(ac.Refusal, match="fell"):
        ac.decide(NOM, _ref(calf=0.8, calf_err=0.26, fell=True), B_ALL, NOM)
    with pytest.raises(ac.Refusal, match="fell"):
        ac.decide(NOM, B_CALF, {k: v for k, v in B_ALL.items() if k != "fell"}, NOM)
    with pytest.raises(ac.Refusal, match="not a phoenix-actuator-ref"):
        ac.decide(NOM, _ref(calf=0.8, calf_err=0.26, schema="other"), B_ALL, NOM)
    other = {**IDENT, "policy.onnx": "e" * 64}
    with pytest.raises(ac.Refusal, match="identity mismatch on policy.onnx"):
        ac.decide(NOM, _ref(calf=0.8, calf_err=0.26, ident=other), B_ALL, NOM)
    no_id = _ref(calf=0.8, calf_err=0.26)
    del no_id["identity"]
    with pytest.raises(ac.Refusal, match="no complete artifact identity"):
        ac.decide(NOM, no_id, B_ALL, NOM)
    kp = _ref(all_=0.75, calf_err=0.30)
    kp["kp"] = dict.fromkeys(POLICY_JOINT_ORDER, 20.0)
    with pytest.raises(ac.Refusal, match="kp mismatch"):
        ac.decide(NOM, B_CALF, kp, NOM)


def test_describe_factors_reads_labels_from_the_file():
    assert ac.describe_factors(_ref(all_=0.75)) == "all=0.75"
    assert ac.describe_factors(_ref(calf=0.8)) == "calf=0.80"
    assert ac.describe_factors(NOM) == "all=1.00"


def test_shipped_refs_carry_lock_identity_and_validate():
    lock = yaml.safe_load(LOCK.read_text())
    want = {r: lock["artifacts"][r]["sha256"] for r in ac.ARTIFACT_ROLES}
    want["config_semantic_sha256"] = lock["deploy_config"]["semantic_sha256"]
    for p in sorted(SHIPPED.glob("ref_*.json")):
        assert ac.ref_identity(json.loads(p.read_text()), p.name) == want, p.name
    labels = ac.validate_boundaries(
        _shipped("ref_nominal.json"), _shipped("ref_calf_0.80.json"), _shipped("ref_all_0.75.json")
    )
    assert labels == {
        "nominal": "all=1.00",
        "calf_boundary": "calf=0.80",
        "global_boundary": "all=0.75",
    }


# ------------------------------------------------------------------ gates 1 and 3: hardware


def _hw(files=None, ident=IDENT, ledger=True, **kw):
    hw = ac.summarize_telemetry(files or [_file(_episode())], ident, **kw)
    if ledger:  # what the CLI attaches after ledger_telemetry() accepted the G/H records
        hw["ledger"] = [{"stage": "G", "record": "stage_G.json", "files": ["bridge.jsonl"]}]
    return hw


def test_hw_summary_not_bound_to_ledger_refused():
    with pytest.raises(ac.Refusal, match="ledger"):
        ac.decide(_hw(ledger=False), B_CALF, B_ALL, NOM)


@pytest.mark.parametrize("key", ac.IDENTITY_KEYS)
def test_telemetry_identity_mismatch_refused(key):
    bad = {**IDENT, key: "f" * 64}
    with pytest.raises(ac.Refusal, match=f"identity mismatch on {key}"):
        _hw([_file(_episode(), manifest=_manifest(bad))])


def test_rehearsal_telemetry_needs_the_flag():
    reh = _manifest(live=False)
    with pytest.raises(ac.Refusal, match="not live"):
        _hw([_file(_episode(), manifest=reh)])
    hw = _hw([_file(_episode(), manifest=reh)], allow_rehearsal=True)
    assert hw["live"] is False and hw["allow_rehearsal"] is True
    # decide() independently refuses a non-live summary without the flag.
    hw2 = dict(hw, allow_rehearsal=False)
    with pytest.raises(ac.Refusal, match="not from live telemetry"):
        ac.decide(hw2, B_CALF, B_ALL, NOM)


def test_every_file_manifest_is_checked():
    good, bad = _file(_episode()), _file(_episode(), manifest=_manifest(live=False), path="b2")
    with pytest.raises(ac.Refusal, match="b2"):
        _hw([good, bad])


def test_hw_kp_mismatch_refused_in_decide():
    hw = _hw([_file(_episode(kp_final=30.0, err={"FL_calf_joint": 0.1}))])
    with pytest.raises(ac.Refusal, match="kp mismatch"):
        ac.decide(hw, B_CALF, B_ALL, NOM)
    ok = _hw([_file(_episode(kp_final=25.0004))])  # within 1e-3: accepted
    assert ac.decide(ok, B_CALF, B_ALL, NOM)["verdict"] == ac.VERDICT_GO


def test_hw_summary_identity_must_equal_nominal():
    hw = _hw(ident=IDENT)
    other = {**IDENT, "policy.onnx.data": "9" * 64}
    nom = _ref(calf_err=0.20, ident=other)
    with pytest.raises(ac.Refusal, match="identity mismatch"):
        ac.decide(
            hw,
            _ref(calf=0.8, calf_err=0.26, ident=other),
            _ref(all_=0.75, calf_err=0.3, ident=other),
            nom,
        )


def test_hw_summary_without_quality_record_refused():
    hw = _hw()
    for bad in ({**hw, "window_gates": []}, {k: v for k, v in hw.items() if k != "window_gates"}):
        with pytest.raises(ac.Refusal, match="window-quality"):
            ac.decide(bad, B_CALF, B_ALL, NOM)


def test_hw_go_and_stop_end_to_end_from_telemetry():
    def hw(calf):
        return _hw([_file(_episode(err=_errs(calf=calf)))])

    d = ac.decide(hw(0.25), B_CALF, B_ALL, NOM)
    assert d["verdict"] == ac.VERDICT_GO and d["input_gates"]["window_gates"][0]["passed"]
    assert ac.decide(hw(0.27), B_CALF, B_ALL, NOM)["verdict"] == ac.VERDICT_STOP


# ------------------------------------------------------------------ CLI


def _cli():
    spec = importlib.util.spec_from_file_location(
        "actuator_check_cli", REPO / "scripts/actuator_check.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_bridge(path, ticks, manifest):
    path.write_text("\n".join(json.dumps(r) for r in [manifest, *ticks]) + "\n")


def _sha(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_session(tmp_path, ticks, manifest, cfg_hash, stages=("G", "H"), **rec_over):
    """A harness session: one bridge file per stage and a GO, non-rehearsal ledger record."""
    session = tmp_path / "session"
    for st in stages:
        run = session / f"{st}1_run"
        run.mkdir(parents=True, exist_ok=True)
        b = run / "bridge.jsonl"
        _write_bridge(b, ticks, {**manifest, "stage": st})
        rec = {
            "stage": st, "verdict": "GO", "rehearsal": False, "motors": "live",
            "lock_semantic_sha256": cfg_hash,
            "evidence_files": [{"path": f"{st}1_run/bridge.jsonl", "sha256": _sha(b)}],
            **rec_over,
        }  # fmt: skip
        (session / f"stage_{st}.json").write_text(json.dumps(rec))
    return session


def _estimate(tmp_path, *extra, calf="ref_calf_0.80.json", allb="ref_all_0.75.json"):
    out = tmp_path / "out.json"
    argv = [
        "estimate", "--reference", str(SHIPPED / "ref_nominal.json"),
        "--boundary-calf", str(SHIPPED / calf), "--boundary-all", str(SHIPPED / allb),
        "--out", str(out), *extra,
    ]  # fmt: skip
    return _cli().main(argv), json.loads(out.read_text())


def test_cli_probe_wrong_boundaries_exits_2_without_a_verdict(tmp_path, capsys):
    rc, out = _estimate(
        tmp_path, "--sim-as-hw", str(SHIPPED / "ref_calf_1.00.json"),
        calf="ref_calf_0.70.json", allb="ref_calf_0.70.json",
    )  # fmt: skip
    assert rc == 2 and out["verdict"] == "REFUSED" and "GO" not in out["verdict"]
    assert "REFUSED" in capsys.readouterr().out


def test_cli_probe_one_tick_window_exits_2(tmp_path):
    ident = ac.ref_identity(_shipped("ref_nominal.json"))
    s = _write_session(
        tmp_path, _episode(dur=0.02), _manifest(ident), ident["config_semantic_sha256"]
    )
    rc, out = _estimate(tmp_path, "--session", str(s))
    assert rc == 2 and out["verdict"] == "REFUSED" and "ledger" not in out["reason"]


def test_cli_session_go_stop_and_dry_refusal(tmp_path, capsys):
    nom = _shipped("ref_nominal.json")
    ident = ac.ref_identity(nom)
    cfg = ident["config_semantic_sha256"]
    s = _write_session(tmp_path / "a", _episode(err=nom["err"]), _manifest(ident), cfg)
    rc, out = _estimate(tmp_path, "--session", str(s))
    assert rc == 0 and out["verdict"] == ac.VERDICT_GO
    text = capsys.readouterr().out
    assert "calf boundary (calf=0.80)" in text and "global boundary (all=0.75)" in text
    weak = {n: v * 1.6 if ac.joint_group(n) == "calf" else v for n, v in nom["err"].items()}
    s = _write_session(tmp_path / "b", _episode(err=weak), _manifest(ident), cfg)
    rc, out = _estimate(tmp_path, "--session", str(s))
    assert rc == 0 and out["verdict"] == ac.VERDICT_STOP
    s = _write_session(tmp_path / "c", _episode(err=nom["err"]), _manifest(ident, live=False), cfg)
    rc, out = _estimate(tmp_path, "--session", str(s))
    assert rc == 2 and "not live" in out["reason"]


def test_cli_bridge_needs_allow_rehearsal(tmp_path):
    nom = _shipped("ref_nominal.json")
    b = tmp_path / "bridge.jsonl"
    _write_bridge(b, _episode(err=nom["err"]), _manifest(ac.ref_identity(nom)))
    rc, out = _estimate(tmp_path, "--bridge", str(b))
    assert rc == 2 and "ledger" in out["reason"]
    rc, out = _estimate(tmp_path, "--bridge", str(b), "--allow-rehearsal")
    assert rc == 0 and out["verdict"] == ac.VERDICT_GO


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"rehearsal": True}, "rehearsal is True"),
        ({"verdict": "NO-GO"}, "not GO"),
        ({"motors": "off"}, "not live"),
        ({"lock_semantic_sha256": "0" * 64}, "different deploy config"),
    ],
)
def test_cli_ledger_record_must_be_go_live_hardware_for_this_config(tmp_path, over, match):
    nom = _shipped("ref_nominal.json")
    ident = ac.ref_identity(nom)
    s = _write_session(
        tmp_path,
        _episode(err=nom["err"]),
        _manifest(ident),
        ident["config_semantic_sha256"],
        **over,
    )
    rc, out = _estimate(tmp_path, "--session", str(s))
    assert rc == 2 and match in out["reason"]


def test_cli_tampered_or_missing_ledger_evidence_refused(tmp_path):
    nom = _shipped("ref_nominal.json")
    ident = ac.ref_identity(nom)
    s = _write_session(
        tmp_path, _episode(err=nom["err"]), _manifest(ident), ident["config_semantic_sha256"]
    )
    (s / "G1_run" / "bridge.jsonl").write_text("tampered\n")
    rc, out = _estimate(tmp_path, "--session", str(s))
    assert rc == 2 and "hash differs" in out["reason"]
    (s / "stage_H.json").unlink()
    rc, out = _estimate(tmp_path, "--session", str(s))
    assert rc == 2


def test_cli_identity_mismatch_exits_2(tmp_path):
    nom = _shipped("ref_nominal.json")
    cfg = ac.ref_identity(nom)["config_semantic_sha256"]
    s = _write_session(tmp_path, _episode(err=nom["err"]), _manifest({**IDENT}), cfg)
    rc, out = _estimate(tmp_path, "--session", str(s))
    assert rc == 2 and "identity mismatch" in out["reason"]


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


def test_cli_sim_as_hw_rejects_a_hand_made_hardware_summary(tmp_path):
    """Re-review probe D1: a self-declared 'hardware' file must not reach a verdict."""
    nom = _shipped("ref_nominal.json")
    fake = {
        "schema": nom["schema"], "source": "hardware", "identity": nom["identity"],
        "err": nom["err"], "kp": nom["kp"], "kd": nom["kd"], "allow_rehearsal": True,
        "window_gates": [{"passed": True}],
    }  # fmt: skip
    f = tmp_path / "fake.json"
    f.write_text(json.dumps(fake))
    rc, out = _estimate(tmp_path, "--sim-as-hw", str(f))
    assert rc == 2 and out["verdict"] == "REFUSED" and "simulator reference" in out["reason"]


def test_cli_output_says_whether_it_is_hardware_evidence(tmp_path):
    rc, out = _estimate(tmp_path, "--sim-as-hw", str(SHIPPED / "ref_calf_1.00.json"))
    assert rc == 0 and out["input_mode"] == "sim-as-hw" and out["valid_for_walk_decision"] is False
    nom = _shipped("ref_nominal.json")
    ident = ac.ref_identity(nom)
    s = _write_session(
        tmp_path, _episode(err=nom["err"]), _manifest(ident), ident["config_semantic_sha256"]
    )
    rc, out = _estimate(tmp_path, "--session", str(s))
    assert rc == 0 and out["input_mode"] == "session" and out["valid_for_walk_decision"] is True
    assert [r["stage"] for r in out["ledger"]] == ["G", "H"]


def test_hardware_ticks_must_carry_tilt_and_velocity_fields():
    """Re-review B10: missing tilt/velocity fields must refuse for hardware, not skip."""
    ticks = _episode()
    stripped = [{k: v for k, v in t.items() if k != "tilt_deg"} for t in ticks]
    with pytest.raises(ac.Refusal, match="tilt_deg"):
        ac.settle_window(stripped, strict_fields=True)
    ac.settle_window(stripped, strict_fields=False)  # desk/sim callers keep the lenient path


def test_stand_run_never_received_cmd_vel_is_accepted_but_fed_command_is_required():
    """Real stand telemetry (stage B, 2026-10-06) logs cmd_vel_received [None]*3 and
    velocity_command_fed [0]*3; the first is 'never received', not a nonzero command."""
    ticks = _episode()
    for t in ticks:
        t.setdefault("policy", {})
        t["policy"]["cmd_vel_received"] = [None, None, None]
        t["policy"]["velocity_command_fed"] = [0.0, 0.0, 0.0]
    ac.settle_window(ticks, strict_fields=True)
    for t in ticks:
        del t["policy"]["velocity_command_fed"]
        t.pop("velocity_command_fed", None)
    with pytest.raises(ac.Refusal, match="velocity_command_fed"):
        ac.settle_window(ticks, strict_fields=True)
