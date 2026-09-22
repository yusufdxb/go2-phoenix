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
    MIN_SCALE,
    DegradationSpec,
    activation_problems,
    parse_spec,
)
from phoenix.sim2real.go2_model import UNITREE_MOTOR_ORDER

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
def _policy_gate(**over):
    gate = armed(degradation=SPEC, **over)
    send(gate, 0.01, DEFAULT_P)
    return gate


def test_policy_mode_scales_only_the_named_motor_and_never_the_target():
    ref = armed()
    send(ref, 0.01, DEFAULT_P)
    r0 = ref.tick(ns(0.02))
    gate = _policy_gate()
    rec = gate.tick(ns(0.02))
    assert rec["mode"] == "policy"
    assert rec["final_target_unitree"] == r0["final_target_unitree"]
    assert (rec["kp"], rec["kd"]) == (25.0, 0.5)  # scalar mode gains unchanged
    kp = np.asarray(rec["kp_unitree"])
    kd = np.asarray(rec["kd_unitree"])
    assert kp[J] == pytest.approx(15.0) and kd[J] == pytest.approx(0.3)
    assert np.all(np.delete(kp, J) == 25.0) and np.all(np.delete(kd, J) == 0.5)
    assert rec["degradation"]["applied"] is True
    assert rec["degradation"]["kp_scale_unitree"][J] == 0.6


def test_no_gain_ever_exceeds_nominal():
    gate = _policy_gate()
    rec = gate.tick(ns(0.02))
    assert max(rec["kp_unitree"]) <= rec["kp"] and max(rec["kd_unitree"]) <= rec["kd"]


def test_hold_keeps_nominal_gains():
    gate = armed(degradation=SPEC)  # no policy command -> hold
    rec = gate.tick(ns(0.02))
    assert rec["mode"] == "hold"
    assert rec["kp_unitree"] == [20.0] * 12 and rec["kd_unitree"] == [1.0] * 12
    assert rec["degradation"]["applied"] is False


def test_estop_drops_to_hold_with_nominal_gains():
    gate = _policy_gate()
    assert gate.tick(ns(0.02))["mode"] == "policy"
    gate.on_estop(ns(0.03), True)
    rec = gate.tick(ns(0.04))
    assert rec["mode"] == "hold" and rec["fault"] == "estop_asserted"
    assert rec["kp_unitree"] == [20.0] * 12


def test_damp_keeps_zero_kp_on_every_motor():
    gate = _policy_gate()
    gate.request_shutdown()
    rec = gate.tick(ns(0.02))
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
