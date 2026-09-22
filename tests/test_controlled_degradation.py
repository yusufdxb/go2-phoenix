"""Controlled actuator degradation: bounded, one joint, policy mode only, triple-locked.

The degradation lives inside the final actuator gate, so these tests check that
it cannot raise a gain, cannot move a target, cannot survive into hold / damp /
standup, and cannot be armed without every lock.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real.degradation import (
    ARM_ENV,
    ARM_VALUE,
    DEGRADATION_PIN_BAND_RAD,
    DEGRADATION_PIN_BAND_WALK_RAD,
    JOINT_GROUPS,
    MIN_SCALE,
    MIN_SCALE_MULTI,
    RAMP_S,
    SATURATION_LATCH_S,
    DegradationSpec,
    activation_problems,
    parse_spec,
)
from phoenix.sim2real.go2_model import UNITREE_MOTOR_ORDER
from phoenix.sim2real.motor_crc import build_raw_from_motor_values, compute_crc

from .test_actuator_gate import DEFAULT_P, DEFAULT_U, ZERO12, armed, ns, send
from .test_lowcmd_bridge import bridge_module  # noqa: F401  (pytest fixture)

RR_THIGH = "RR_thigh_joint"
J = UNITREE_MOTOR_ORDER.index(RR_THIGH)
SPEC = DegradationSpec(RR_THIGH, 0.6, 0.6)
ARMED_ENV = {ARM_ENV: ARM_VALUE}


# ------------------------------------------------------------------- the spec
@pytest.mark.parametrize("scale", [1.01, 1.5, MIN_SCALE - 0.01, 0.0, -1.0, float("nan")])
def test_spec_rejects_out_of_range_scales(scale):
    with pytest.raises(ValueError):
        DegradationSpec(RR_THIGH, scale, 1.0)
    with pytest.raises(ValueError):
        DegradationSpec(RR_THIGH, 1.0, scale)


def test_spec_rejects_unknown_joint():
    with pytest.raises(ValueError):
        DegradationSpec("RR_knee_joint", 0.7, 0.7)


def test_spec_vectors_touch_one_motor_only():
    kp = SPEC.kp_scale_vector()
    assert kp[J] == 0.6
    assert np.all(np.delete(kp, J) == 1.0)


def test_parse_spec_forms():
    assert parse_spec("RR_thigh:0.7") == DegradationSpec(RR_THIGH, 0.7, 0.7)
    assert parse_spec("RR_thigh_joint:0.7:0.9") == DegradationSpec(RR_THIGH, 0.7, 0.9)
    with pytest.raises(ValueError):
        parse_spec("RR_thigh")
    with pytest.raises(ValueError):
        parse_spec("RR_thigh:1.2")


# ------------------------------------------------------------------ the locks
def test_no_spec_needs_no_locks():
    assert activation_problems(None, "F", environ={}) == []


def test_every_lock_is_required():
    assert activation_problems(SPEC, "X1", environ=ARMED_ENV) == []
    assert len(activation_problems(SPEC, "X1", environ={})) == 1
    assert len(activation_problems(SPEC, "F", environ=ARMED_ENV)) == 1
    assert len(activation_problems(SPEC, "F", environ={ARM_ENV: "yes"})) == 2


def test_bridge_refuses_to_start_without_locks(bridge_module, monkeypatch, tmp_path):  # noqa: F811
    bridge = bridge_module
    monkeypatch.delenv(ARM_ENV, raising=False)
    args = bridge._parse_args(
        ["--config", str(tmp_path / "none.yaml"), "--experiment-degradation", "RR_thigh:0.6"]
    )
    cfg = bridge._build_config(args)
    problems, manifest = bridge.startup_problems(cfg)
    assert any("controlled degradation" in p for p in problems)
    assert manifest["controlled_degradation"]["joint"] == RR_THIGH


# ------------------------------------------------------------- in the gate
DT = 0.02


def run_policy(gate, t0, t1, q_u=DEFAULT_U, seq0=1):
    """Feed fresh LowState and policy commands every tick from t0 to t1; return records."""
    recs, seq, t = [], seq0, t0
    while t <= t1 + 1e-9:
        gate.on_lowstate(ns(t), q_u, ZERO12)
        gate.on_estop(ns(t), False)
        send(gate, t, DEFAULT_P, seq=seq)
        recs.append(gate.tick(ns(t + 0.001)))
        seq += 1
        t += DT
    return recs


def test_scale_ramps_in_and_touches_only_the_named_motor_never_the_target():
    ref = armed()
    r0 = run_policy(ref, 0.01, RAMP_S + 0.2)
    recs = run_policy(armed(degradation=SPEC), 0.01, RAMP_S + 0.2)
    assert all(r["mode"] == "policy" for r in recs)
    assert [r["final_target_unitree"] for r in recs] == [r["final_target_unitree"] for r in r0]
    first, mid, last = recs[0], recs[len(recs) // 2], recs[-1]
    assert first["kp_unitree"][J] == pytest.approx(25.0)  # ramp starts at nominal
    assert 15.0 < mid["kp_unitree"][J] < 25.0
    assert last["kp_unitree"][J] == pytest.approx(15.0) and last["kd_unitree"][J] == pytest.approx(
        0.3
    )
    assert last["degradation"]["ramp"] == 1.0 and last["degradation"]["applied"] is True
    for r in recs:
        kp, kd = np.asarray(r["kp_unitree"]), np.asarray(r["kd_unitree"])
        assert (r["kp"], r["kd"]) == (25.0, 0.5)
        assert np.all(np.delete(kp, J) == 25.0) and np.all(np.delete(kd, J) == 0.5)
        assert kp.max() <= 25.0 and kd.max() <= 0.5  # never above nominal, any tick
        assert r["degradation"]["kp_scale_unitree"][J] == pytest.approx(kp[J] / 25.0)


def test_ramp_restarts_after_leaving_policy_mode():
    gate = armed(degradation=SPEC)
    run_policy(gate, 0.01, RAMP_S + 0.2)
    gate.on_lowstate(ns(3.0), DEFAULT_U, ZERO12)
    gate.on_estop(ns(3.0), False)
    stale = gate.tick(ns(3.0))  # last command is 0.8 s old -> watchdog hold
    assert stale["mode"] == "hold" and stale["hold_cause"] == "command_stale"
    assert stale["kp_unitree"] == [20.0] * 12 and stale["degradation"]["applied"] is False
    again = run_policy(gate, 3.02, 3.1, seq0=500)
    assert again[0]["mode"] == "policy" and again[0]["kp_unitree"][J] == pytest.approx(25.0)


def test_pinned_degraded_joint_latches_hold():
    sagged = DEFAULT_U.copy()
    sagged[J] -= 0.3  # the joint lags its target by more than one slew cap
    recs = run_policy(armed(degradation=SPEC), 0.01, SATURATION_LATCH_S + 0.2, q_u=sagged)
    faults = [r["fault"] for r in recs]
    assert "degradation_joint_saturated:RR_thigh_joint" in faults
    after = recs[faults.index("degradation_joint_saturated:RR_thigh_joint")]
    assert after["mode"] == "hold" and after["kp_unitree"] == [20.0] * 12
    assert recs[-1]["mode"] == "hold"  # latched


def test_pinned_joint_without_degradation_does_not_latch():
    sagged = DEFAULT_U.copy()
    sagged[J] -= 0.3
    recs = run_policy(armed(), 0.01, SATURATION_LATCH_S + 0.2, q_u=sagged)
    assert all(r["mode"] == "policy" for r in recs)


def test_brief_pin_does_not_latch():
    gate = armed(degradation=SPEC)
    sagged = DEFAULT_U.copy()
    sagged[J] -= 0.3
    run_policy(gate, 0.01, SATURATION_LATCH_S / 2, q_u=sagged)
    recs = run_policy(gate, SATURATION_LATCH_S / 2 + DT, 1.5, seq0=200)
    assert all(r["mode"] == "policy" for r in recs)


def test_hold_keeps_nominal_gains():
    gate = armed(degradation=SPEC)  # no policy command -> hold
    rec = gate.tick(ns(0.02))
    assert rec["mode"] == "hold"
    assert rec["kp_unitree"] == [20.0] * 12 and rec["kd_unitree"] == [1.0] * 12
    assert rec["degradation"]["applied"] is False


def test_estop_drops_to_hold_with_nominal_gains():
    gate = armed(degradation=SPEC)
    assert run_policy(gate, 0.01, RAMP_S + 0.1)[-1]["mode"] == "policy"
    t = RAMP_S + 0.2
    gate.on_estop(ns(t), True)
    rec = gate.tick(ns(t + 0.001))
    assert rec["mode"] == "hold" and rec["fault"] == "estop_asserted"
    assert rec["kp_unitree"] == [20.0] * 12


def test_damp_keeps_zero_kp_on_every_motor():
    gate = armed(degradation=SPEC)
    run_policy(gate, 0.01, RAMP_S + 0.1)
    gate.request_shutdown()
    rec = gate.tick(ns(RAMP_S + 0.2))
    assert rec["mode"] == "damp"
    assert rec["kp_unitree"] == [0.0] * 12 and rec["kd_unitree"] == [1.0] * 12


def test_standup_keeps_nominal_standup_gains():
    gate = armed(degradation=SPEC, standup_s=1.0)
    rec = gate.tick(ns(0.02))
    assert rec["mode"] == "standup"
    assert rec["kp_unitree"] == [60.0] * 12


def test_without_spec_records_are_unchanged_apart_from_vectors():
    gate = armed()
    send(gate, 0.01, DEFAULT_P)
    rec = gate.tick(ns(0.02))
    assert rec["degradation"] is None
    assert rec["kp_unitree"] == [25.0] * 12


def test_tau_est_is_recorded_and_malformed_tau_is_dropped():
    gate = armed()
    gate.on_lowstate(ns(0.01), DEFAULT_U, ZERO12, np.arange(12.0))
    assert gate.tick(ns(0.02))["tau_est_unitree"] == list(np.arange(12.0))
    gate.on_lowstate(ns(0.03), DEFAULT_U, ZERO12, [1.0, 2.0])
    rec = gate.tick(ns(0.04))
    assert rec["tau_est_unitree"] is None and rec["fault"] is None


def test_lowcmd_fields_accepts_per_motor_gains(bridge_module):  # noqa: F811
    bridge = bridge_module
    kp = [25.0] * 12
    kp[J] = 15.0
    q, crc = bridge.lowcmd_fields(DEFAULT_U, kp, [0.5] * 12)
    _, crc_nominal = bridge.lowcmd_fields(DEFAULT_U, 25.0, 0.5)
    assert len(q) == 12 and crc != crc_nominal
    with pytest.raises(ValueError):
        bridge.lowcmd_fields(DEFAULT_U, [25.0] * 11, 0.5)


def test_stale_lowstate_goes_to_damp_with_nominal_gains():
    gate = armed(degradation=SPEC)
    run_policy(gate, 0.01, RAMP_S + 0.1)
    t = RAMP_S + 0.1
    for k in range(1, 30):  # LowState stops arriving
        rec = gate.tick(ns(t + k * DT))
    assert rec["mode"] == "damp" and rec["kp_unitree"] == [0.0] * 12


# ------------------------------------------------------------- in the node
def test_bridge_requires_telemetry_with_degradation(
    bridge_module, monkeypatch, tmp_path  # noqa: F811
):
    monkeypatch.setenv(ARM_ENV, ARM_VALUE)
    args = bridge_module._parse_args(
        [
            "--config",
            str(tmp_path / "none.yaml"),
            "--stage",
            "X1",
            "--experiment-degradation",
            "RR_thigh:0.6",
        ]
    )
    problems, _ = bridge_module.startup_problems(bridge_module._build_config(args))
    assert any("requires --telemetry" in p for p in problems)
    args = bridge_module._parse_args(
        [
            "--config",
            str(tmp_path / "none.yaml"),
            "--stage",
            "X1",
            "--telemetry",
            str(tmp_path / "t.jsonl"),
            "--experiment-degradation",
            "RR_thigh:0.6",
        ]
    )
    problems, _ = bridge_module.startup_problems(bridge_module._build_config(args))
    assert not any("controlled degradation" in p for p in problems)


def test_node_publishes_exactly_the_recorded_per_motor_gains(bridge_module):  # noqa: F811
    import types

    sent, reported = [], []
    gate = armed(degradation=SPEC)
    run_policy(gate, 0.01, RAMP_S + 0.1)
    gate.on_lowstate(ns(2.3), DEFAULT_U, ZERO12)
    gate.on_estop(ns(2.3), False)
    send(gate, 2.3, DEFAULT_P, seq=900)
    fake = types.SimpleNamespace(
        _gate=types.SimpleNamespace(tick=lambda _now: gate.tick(ns(2.301))),
        _publish=lambda target, kp, kd: sent.append((target, kp, kd)),
        _report=reported.append,
    )
    bridge_module.LowCmdBridge._tick(fake)
    rec = reported[0]
    assert rec["mode"] == "policy"
    assert sent == [(rec["final_target_unitree"], rec["kp_unitree"], rec["kd_unitree"])]
    assert sent[0][1][J] == pytest.approx(15.0)


def test_message_gains_are_the_checksummed_gains(bridge_module):  # noqa: F811
    import types

    out = []
    fake = types.SimpleNamespace(_pub=types.SimpleNamespace(publish=out.append))
    kp = [25.0] * 12
    kp[J] = 15.0
    kd = [0.5] * 12
    kd[J] = 0.3
    bridge_module.LowCmdBridge._publish(fake, list(DEFAULT_U), kp, kd)
    msg = out[0]
    m = msg.motor_cmd[:12]
    assert [c.kp for c in m] == kp and [c.kd for c in m] == kd
    raw = build_raw_from_motor_values([c.q for c in m], [c.kp for c in m], [c.kd for c in m])
    assert msg.crc == compute_crc(raw)


# ------------------------------------------------- the multi-joint form (amendment 13)
LEG_RR = tuple(j for j in UNITREE_MOTOR_ORDER if j.startswith("RR_"))
LEG_IDX = [UNITREE_MOTOR_ORDER.index(j) for j in LEG_RR]
LEG_SPEC = DegradationSpec(LEG_RR, 0.8, 0.8)


def test_multi_joint_floor_is_higher_than_the_single_joint_floor():
    assert MIN_SCALE_MULTI > MIN_SCALE
    # 0.5 is legal for one joint and illegal for a set, which is the whole point.
    assert DegradationSpec(RR_THIGH, MIN_SCALE, MIN_SCALE).min_scale == MIN_SCALE
    with pytest.raises(ValueError):
        DegradationSpec(LEG_RR, MIN_SCALE, MIN_SCALE)
    with pytest.raises(ValueError):
        DegradationSpec(LEG_RR, MIN_SCALE_MULTI - 0.01, 1.0)
    assert DegradationSpec(LEG_RR, MIN_SCALE_MULTI, MIN_SCALE_MULTI).min_scale == MIN_SCALE_MULTI


def test_multi_joint_spec_scales_exactly_its_set_and_nothing_else():
    kp, kd = LEG_SPEC.kp_scale_vector(), LEG_SPEC.kd_scale_vector()
    assert set(LEG_SPEC.joints) == set(LEG_RR)
    assert sorted(LEG_SPEC.motor_indices) == sorted(LEG_IDX)
    assert np.all(kp[LEG_IDX] == 0.8) and np.all(kd[LEG_IDX] == 0.8)
    assert np.all(np.delete(kp, LEG_IDX) == 1.0) and np.all(np.delete(kd, LEG_IDX) == 1.0)
    assert kp.max() <= 1.0  # a reduction only, never a boost


def test_single_element_set_normalises_to_the_historical_single_joint_spec():
    assert DegradationSpec((RR_THIGH,), 0.6, 0.6) == SPEC
    assert DegradationSpec((RR_THIGH,), 0.6, 0.6).min_scale == MIN_SCALE


def test_motor_index_refuses_a_multi_joint_spec():
    assert SPEC.motor_index == J
    with pytest.raises(ValueError):
        _ = LEG_SPEC.motor_index


@pytest.mark.parametrize(
    ("text", "n", "scale"),
    [
        ("leg_RR:0.8", 3, 0.8),
        ("rear:0.75", 6, 0.75),
        ("all:0.75", 12, 0.75),
        ("thighs:0.9", 4, 0.9),
        ("RR_hip+RR_thigh:0.8", 2, 0.8),
    ],
)
def test_parse_spec_group_forms(text, n, scale):
    spec = parse_spec(text)
    assert len(spec.joints) == n
    assert spec.kp_scale == scale == spec.kd_scale
    assert spec.min_scale == MIN_SCALE_MULTI


def test_parse_spec_rejects_a_group_below_the_multi_joint_floor():
    with pytest.raises(ValueError):
        parse_spec("all:0.5")
    with pytest.raises(ValueError):
        parse_spec("leg_RR:0.69")


def test_parse_spec_rejects_duplicates_and_unknown_names():
    with pytest.raises(ValueError):
        parse_spec("RR_thigh+RR_thigh:0.8")
    with pytest.raises(ValueError):
        parse_spec("RR_thigh+bogus:0.8")


def test_every_named_group_is_a_real_physical_grouping():
    for name, joints in JOINT_GROUPS.items():
        assert len(joints) == len(set(joints)), name
        assert set(joints) <= set(UNITREE_MOTOR_ORDER), name
        assert len(joints) >= 2, name
    assert set(JOINT_GROUPS["all"]) == set(UNITREE_MOTOR_ORDER)
    assert set(JOINT_GROUPS["rear"]) == set(JOINT_GROUPS["leg_RR"]) | set(JOINT_GROUPS["leg_RL"])
    assert set(JOINT_GROUPS["thighs"]) == {j for j in UNITREE_MOTOR_ORDER if "_thigh_" in j}


def test_multi_joint_scale_ramps_in_on_its_whole_set_and_never_moves_a_target():
    ref = armed()
    r0 = run_policy(ref, 0.01, RAMP_S + 0.2)
    recs = run_policy(armed(degradation=LEG_SPEC), 0.01, RAMP_S + 0.2)
    assert all(r["mode"] == "policy" for r in recs)
    assert [r["final_target_unitree"] for r in recs] == [r["final_target_unitree"] for r in r0]
    assert recs[0]["kp_unitree"] == [25.0] * 12  # ramp starts at nominal
    last = recs[-1]
    assert last["degradation"]["ramp"] == 1.0 and last["degradation"]["applied"] is True
    assert sorted(last["degradation"]["joints"]) == sorted(LEG_RR)
    for r in recs:
        kp, kd = np.asarray(r["kp_unitree"]), np.asarray(r["kd_unitree"])
        assert kp.max() <= 25.0 and kd.max() <= 0.5  # never above nominal, any tick
        assert np.all(np.delete(kp, LEG_IDX) == 25.0)
        assert np.all(np.delete(kd, LEG_IDX) == 0.5)
        assert len(set(np.round(kp[LEG_IDX], 9))) == 1  # one scale over the whole set
    assert np.allclose(np.asarray(last["kp_unitree"])[LEG_IDX], 20.0)
    assert np.allclose(np.asarray(last["kd_unitree"])[LEG_IDX], 0.4)


@pytest.mark.parametrize("joint", LEG_RR)
def test_any_single_pinned_joint_in_the_set_latches_hold(joint):
    """The watch is per joint: a wider set must not dilute it."""
    idx = UNITREE_MOTOR_ORDER.index(joint)
    sagged = DEFAULT_U.copy()
    sagged[idx] -= 0.3
    recs = run_policy(armed(degradation=LEG_SPEC), 0.01, SATURATION_LATCH_S + 0.2, q_u=sagged)
    faults = [r["fault"] for r in recs]
    assert f"degradation_joint_saturated:{joint}" in faults
    assert recs[-1]["mode"] == "hold" and recs[-1]["kp_unitree"] == [20.0] * 12


def test_multi_joint_degradation_keeps_nominal_gains_outside_policy_mode():
    for mode_setup, expect_kp in (
        (dict(), 20.0),  # no policy command -> hold
        (dict(standup_s=1.0), 60.0),  # standup ramp
    ):
        gate = armed(degradation=LEG_SPEC, **mode_setup)
        rec = gate.tick(ns(0.02))
        assert rec["kp_unitree"] == [expect_kp] * 12
        assert rec["degradation"]["applied"] is False


def test_multi_joint_spec_serialises_with_its_set_and_floor():
    d = LEG_SPEC.to_dict()
    assert sorted(d["joints"]) == sorted(LEG_RR)
    assert d["kp_scale"] == d["kd_scale"] == 0.8
    assert d["min_scale"] == MIN_SCALE_MULTI
    import json

    assert json.loads(json.dumps(d))["min_scale"] == MIN_SCALE_MULTI


# ------------------------------ the walking pin band (amendment 14)
def test_walking_pin_band_is_wider_than_the_standing_one():
    """The standing bound fires on a HEALTHY bang-bang walking policy, so it is not a
    sag detector there. The walking bound is derived from nominal walking telemetry."""
    assert DEGRADATION_PIN_BAND_WALK_RAD > DEGRADATION_PIN_BAND_RAD
    # Still inside the general catastrophic-tracking watchdog, so it remains the
    # stricter, degradation-specific guard rather than a replacement for it.
    assert DEGRADATION_PIN_BAND_WALK_RAD < 1.25


def test_pin_band_defaults_to_the_standing_bound():
    """Standing configs must be bit-identical to before this parameter existed."""
    gate = armed(degradation=SPEC)
    assert gate.params.degradation_pin_band == DEGRADATION_PIN_BAND_RAD


def test_a_gap_between_the_two_bands_latches_on_standing_and_not_on_walking():
    """The exact behaviour change, pinned in both directions."""
    sagged = DEFAULT_U.copy()
    sagged[J] -= 0.5  # beyond the standing band, inside the walking band

    standing = run_policy(armed(degradation=SPEC), 0.01, SATURATION_LATCH_S + 0.2, q_u=sagged)
    assert any(
        r["fault"] == f"degradation_joint_saturated:{RR_THIGH}" for r in standing
    ), "the standing bound must still latch"

    walking = run_policy(
        armed(degradation=SPEC, degradation_pin_band=DEGRADATION_PIN_BAND_WALK_RAD),
        0.01,
        SATURATION_LATCH_S + 0.2,
        q_u=sagged,
    )
    assert all(r["mode"] == "policy" for r in walking)
    assert all(r["fault"] is None for r in walking)


def test_the_walking_band_still_latches_a_genuinely_pinned_joint():
    """Widening it must not disable it."""
    sagged = DEFAULT_U.copy()
    sagged[J] -= DEGRADATION_PIN_BAND_WALK_RAD + 0.1
    recs = run_policy(
        armed(degradation=SPEC, degradation_pin_band=DEGRADATION_PIN_BAND_WALK_RAD),
        0.01,
        SATURATION_LATCH_S + 0.2,
        q_u=sagged,
    )
    assert any(r["fault"] == f"degradation_joint_saturated:{RR_THIGH}" for r in recs)
    assert recs[-1]["mode"] == "hold"


def test_the_applied_pin_band_is_recorded_in_the_tick_record():
    """A run must be readable back without guessing which bound was in force."""
    gate = armed(degradation=SPEC, degradation_pin_band=DEGRADATION_PIN_BAND_WALK_RAD)
    rec = run_policy(gate, 0.01, 0.1)[-1]
    assert rec["degradation"]["pin_band"] == DEGRADATION_PIN_BAND_WALK_RAD
