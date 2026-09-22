"""CONDITION (health -> targeted distribution) and VERIFY (candidate gate, promotion)."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from phoenix.condition.distribution import (
    S_MIN,
    ConditionError,
    TargetedActuatorSpec,
    build_targeted_spec,
    condition,
    targeted_scale_factors,
)
from phoenix.monitor.health import JointHealth
from phoenix.sim2real.go2_model import UNITREE_MOTOR_ORDER
from phoenix.sim_env.config_loader import load_layered_config
from phoenix.sim_env.go2_env_cfg import _unwired_sections_present
from phoenix.validate.candidate_gate import (
    PREREGISTERED,
    bootstrap_diff_ci,
    candidate_gate,
    compare_arms,
)
from phoenix.validate.promotion import promotion_problems


def health(state="NOMINAL", joint="RR_thigh_joint", s=(1.0, 0.95, 1.05)):
    return JointHealth(
        joint=joint,
        state=state,
        s_hat=s[0],
        s_lo=s[1],
        s_hi=s[2],
        threshold=0.9,
        below=0,
        usable=10,
        n=10,
        torque_gain_ratio=None,
    )


def report(**states):
    out = []
    for name in UNITREE_MOTOR_ORDER:
        st = states.get(name)
        out.append(health(*st) if st else health(joint=name))
    return out


DEG = ("DEGRADED", "RR_thigh_joint", (0.61, 0.57, 0.66))


# --------------------------------------------------------------- condition
def test_degraded_joint_becomes_a_widened_capped_range():
    spec = build_targeted_spec(report(RR_thigh_joint=DEG), margin=0.05)
    assert list(spec.joints) == ["RR_thigh_joint"]
    lo, hi = spec.joints["RR_thigh_joint"]
    assert lo == pytest.approx(0.52) and hi == pytest.approx(0.71)
    assert spec.nominal_fraction == 0.5


def test_range_never_exceeds_nominal_or_goes_below_floor():
    spec = build_targeted_spec(
        report(RR_thigh_joint=("DEGRADED", "RR_thigh_joint", (0.2, 0.1, 0.98)))
    )
    lo, hi = spec.joints["RR_thigh_joint"]
    assert lo == S_MIN and hi == 1.0


def test_no_degraded_joint_refuses():
    with pytest.raises(ConditionError, match="no joint"):
        build_targeted_spec(report())


def test_global_shift_refuses():
    r = report(RR_thigh_joint=("GLOBAL_SHIFT", "RR_thigh_joint", (0.7, 0.6, 0.8)))
    with pytest.raises(ConditionError, match="global"):
        build_targeted_spec(r)


def test_too_many_degraded_joints_refuses():
    states = {
        n: ("DEGRADED", n, (0.7, 0.6, 0.8))
        for n in ("RR_thigh_joint", "RL_thigh_joint", "FR_calf_joint")
    }
    with pytest.raises(ConditionError, match="more than"):
        build_targeted_spec(report(**states))


@pytest.mark.parametrize("bad", [(0.2, 0.5), (0.5, 1.2), (0.8, 0.6)])
def test_spec_validation(bad):
    with pytest.raises(ValueError):
        TargetedActuatorSpec(joints={"RR_thigh_joint": bad})


def test_scale_factors_touch_only_targeted_joint_in_targeted_envs():
    spec = TargetedActuatorSpec(joints={"RR_thigh_joint": (0.5, 0.7)}, nominal_fraction=0.25)
    stiff, damp = targeted_scale_factors(8, UNITREE_MOTOR_ORDER, spec, np.random.default_rng(0))
    j = UNITREE_MOTOR_ORDER.index("RR_thigh_joint")
    assert np.all(stiff[:2] == 1.0)
    assert np.all((stiff[2:, j] >= 0.5) & (stiff[2:, j] <= 0.7))
    assert np.all(np.delete(stiff, j, axis=1) == 1.0)
    assert np.array_equal(stiff, damp)


def test_scale_factors_reject_unknown_joint():
    spec = TargetedActuatorSpec(joints={"RR_thigh_joint": (0.5, 0.7)})
    with pytest.raises(KeyError):
        targeted_scale_factors(4, ["FR_hip_joint"], spec, np.random.default_rng(0))


def test_overlay_loads_through_the_real_config_loader(tmp_path):
    env_dir = tmp_path / "configs" / "env"
    (env_dir / "conditioned").mkdir(parents=True)
    (env_dir / "parent.yaml").write_text(
        "env: {task_name: X}\ndomain_randomization: {motor_strength_scale: [0.85, 1.15]}\n"
    )
    tele = tmp_path / "bridge.jsonl"
    tele.write_text('{"record": "manifest"}\n')
    res = condition(report(RR_thigh_joint=DEG), parent_env="../parent", telemetry_paths=[tele])
    out = env_dir / "conditioned" / "rr.yaml"
    out.write_text(yaml.safe_dump(res.overlay(), sort_keys=False))
    data = load_layered_config(out).to_container()
    dr = data["domain_randomization"]
    assert dr["motor_strength_scale"] == [0.85, 1.15]  # parent recipe kept
    assert dr["targeted_actuator"]["joints"]["RR_thigh_joint"] == pytest.approx([0.52, 0.71])
    assert data["phoenix_condition"]["telemetry"][0]["sha256"]
    assert _unwired_sections_present(data) == []  # the factory consumes every key


# -------------------------------------------------------------------- gate
def test_bootstrap_ci_brackets_true_difference():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.8, 0.05, 40), rng.normal(0.7, 0.05, 40)
    d, lo, hi = bootstrap_diff_ci(a, b, 4000, 0.95, 0)
    assert lo < 0.1 < hi and lo < d < hi


def test_bootstrap_rejects_tiny_or_bad_samples():
    with pytest.raises(ValueError):
        bootstrap_diff_ci([1.0], [1.0, 2.0], 100, 0.95, 0)
    with pytest.raises(ValueError):
        bootstrap_diff_ci([1.0, float("nan")], [1.0, 2.0], 100, 0.95, 0)


SHA = "a" * 64
PARITY = {"checkpoint_sha256": SHA, "max_abs": 2e-6}
EVAL = {c: SHA for c in ("degraded", "nominal", "held_out")}


def evals(deg, nom, held, n=30, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "degraded": np.clip(rng.normal(deg, 0.03, n), 0, 1),
        "nominal": np.clip(rng.normal(nom, 0.03, n), 0, 1),
        "held_out": np.clip(rng.normal(held, 0.03, n), 0, 1),
    }


def test_gate_promotes_a_clear_improvement_that_keeps_nominal():
    dec = candidate_gate(evals(0.8, 0.95, 0.7), evals(0.5, 0.95, 0.5, seed=1), SHA, EVAL, PARITY)
    assert dec["decision"] == "PROMOTE", dec["reasons"]


def test_gate_rejects_nominal_regression():
    dec = candidate_gate(evals(0.8, 0.85, 0.7), evals(0.5, 0.95, 0.5, seed=1), SHA, EVAL, PARITY)
    assert dec["decision"] == "REJECT"
    assert any("nominal" in r for r in dec["reasons"])


def test_gate_rejects_small_improvement():
    dec = candidate_gate(evals(0.52, 0.95, 0.5), evals(0.5, 0.95, 0.5, seed=1), SHA, EVAL, PARITY)
    assert any("degraded improvement" in r for r in dec["reasons"])


def test_gate_rejects_held_out_overfit():
    dec = candidate_gate(evals(0.8, 0.95, 0.3), evals(0.5, 0.95, 0.5, seed=1), SHA, EVAL, PARITY)
    assert any("held-out" in r for r in dec["reasons"])


@pytest.mark.parametrize(
    "parity",
    [
        None,
        {"checkpoint_sha256": "b" * 64, "max_abs": 1e-7},
        {"checkpoint_sha256": SHA, "max_abs": 1e-3},
    ],
)
def test_gate_rejects_bad_parity(parity):
    dec = candidate_gate(evals(0.8, 0.95, 0.7), evals(0.5, 0.95, 0.5, seed=1), SHA, EVAL, parity)
    assert dec["decision"] == "REJECT"


def test_gate_rejects_evaluation_of_a_different_checkpoint():
    ev = dict(EVAL, nominal="c" * 64)
    dec = candidate_gate(evals(0.8, 0.95, 0.7), evals(0.5, 0.95, 0.5, seed=1), SHA, ev, PARITY)
    assert any("nominal evaluation names" in r for r in dec["reasons"])


def test_gate_rejects_missing_condition():
    cand = evals(0.8, 0.95, 0.7)
    del cand["held_out"]
    dec = candidate_gate(cand, evals(0.5, 0.95, 0.5, seed=1), SHA, EVAL, PARITY)
    assert any("missing evaluation" in r for r in dec["reasons"])


def test_compare_arms_reports_ci_and_effect_size():
    per_seed = {
        "broad": {"degraded": [0.6, 0.62, 0.58, 0.61, 0.6]},
        "phoenix": {"degraded": [0.7, 0.72, 0.69, 0.71, 0.7]},
    }
    out = compare_arms(per_seed, reference="broad")
    r = out["phoenix"]["degraded"]
    assert r["diff"] == pytest.approx(0.102, abs=1e-9)
    assert r["ci"][0] > 0 and r["effect_size"] > 1


def test_preregistered_rules_are_frozen():
    assert (PREREGISTERED.min_improvement, PREREGISTERED.nominal_margin) == (0.05, 0.02)


# --------------------------------------------------------------- promotion
def test_promotion_requires_promote_for_the_locked_checkpoint():
    dec = {"schema": "phoenix-candidate-gate/v1", "decision": "PROMOTE", "candidate_sha256": SHA}
    lock = {"artifacts": {"checkpoint": {"sha256": SHA}}}
    assert promotion_problems(dec, lock) == []
    assert promotion_problems({**dec, "decision": "REJECT"}, lock)
    assert promotion_problems(dec, {"artifacts": {"checkpoint": {"sha256": "d" * 64}}})
    assert promotion_problems(dec, {"artifacts": {}})
