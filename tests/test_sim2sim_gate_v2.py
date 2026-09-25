"""Gate v2: SAFETY tier blocks, PERFORMANCE tier is reported; saturation relative to the deploy clip."""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2sim.deploy_spec import DeploySpecError, spec_from_phoenix_manifest
from phoenix.sim2sim.gate import (
    evaluate_scenario,
    gate_verdict,
    load_gate_config,
    performance_table,
    saturation_mask,
    tag_spec_checks,
)
from phoenix.velocity.contract import CommandRanges, build_manifest

ENV = {"lin_vel_x": 0.5, "lin_vel_y": 0.3, "ang_vel_z": 0.8}


def manifest(deploy=None, ang=(-1.0, 1.0)):
    m = build_manifest(
        checkpoint_sha256="0" * 64,
        commands=CommandRanges((-1.0, 1.0), (-0.5, 0.5), ang, 0.1),
        git_sha="a" * 40, git_dirty=False, seed=1, task="t", simulator="isaaclab",
        reward_scales={}, domain_randomization={}, curriculum={},
    )
    if deploy is not None:
        m["deploy"] = deploy
    return m


def metrics(**over):
    m = {
        "fell": False, "nonfinite_action": False, "pre_clip_saturation_rate": 0.0,
        "hard_limit_violation_steps": 0,
        "torque_saturation_fraction": {"hip": 0.0, "thigh": 0.0, "calf": 0.0},
        "near_limit_fraction": {"hip": 0.0, "thigh": 0.0, "calf": 0.0},
        "lin_vel_rmse_mps": 0.05, "yaw_rate_rmse_radps": 0.05, "mean_base_height_m": 0.3,
    }
    m.update(over)
    return m


def report(cfg, per_scenario, spec_checks=()):
    rep = {"spec_checks": tag_spec_checks([dict(c) for c in spec_checks], cfg), "scenarios": {}}
    for name, m in per_scenario.items():
        rep["scenarios"][name] = {"checks": evaluate_scenario(m, cfg.scenario(name), cfg)}
    return rep


# ------------------------------------------------------------------ config


def test_v2_is_default_and_keeps_every_v1_number():
    v1, v2 = load_gate_config("v1"), load_gate_config("v2")
    assert v2.version == 2 and v1.version == 1
    assert v2.raw["name"] == "gate_v2" and v2.path.endswith("gate_v2.yaml")
    for key in ("physics", "actuator", "fall", "tracking", "scenarios", "video"):
        assert v2.raw[key] == v1.raw[key], key
    t1, t2 = v1.thresholds, v2.thresholds
    for tier in ("nominal", "stress", "stand"):
        assert t2[tier] == t1[tier], tier
    for k, v in t1["all"].items():
        assert t2["all"][k] == v, k
    assert v2.required_envelope == ENV
    assert v1.required_envelope == {"lin_vel_x": 0.5, "lin_vel_y": 0.3, "ang_vel_z": 0.6}


# ------------------------------------------------------------------ clip-relative saturation


def test_v2_saturation_is_abs_raw_ge_deploy_clip():
    cfg = load_gate_config("v2")
    raw = np.array([99.9, -99.99, 100.0, -100.0, 250.0] + [0.0] * 7)
    assert saturation_mask(raw, 100.0, cfg).tolist() == [False, False, True, True, True] + [False] * 7
    # Relative to the clip: 7.0 is not saturated at clip 100, but is at clip 1.
    assert not saturation_mask(np.full(12, 7.0), 100.0, cfg).any()
    assert saturation_mask(np.full(12, 7.0), 1.0, cfg).all()
    assert saturation_mask(np.full(12, 1.0), 1.0, cfg).all()  # exactly at the clip counts


def test_v2_missing_clip_is_undefined_and_fails_safety():
    cfg = load_gate_config("v2")
    assert saturation_mask(np.zeros(12), None, cfg) is None
    checks = evaluate_scenario(metrics(pre_clip_saturation_rate=None), cfg.scenario("forward_0p3"), cfg)
    sat = [c for c in checks if c["check"] == "pre_clip_saturation_rate"][0]
    assert sat["tier"] == "safety" and not sat["pass"]


def test_v1_saturation_semantics_unchanged():
    cfg = load_gate_config("v1")
    assert saturation_mask(np.full(12, 1.0), 1.0, cfg).sum() == 0  # strict >
    assert saturation_mask(np.full(12, 1.01), 1.0, cfg).all()
    assert saturation_mask(np.full(12, 5.0), None, cfg).sum() == 0


# ------------------------------------------------------------------ tiers


def test_v2_tracking_failures_are_performance_only():
    cfg = load_gate_config("v2")
    rep = report(cfg, {
        "stand_20s": metrics(yaw_rate_rmse_radps=0.204),   # robot_lab-like stand drift
        "yaw_0p6": metrics(yaw_rate_rmse_radps=0.597),     # himloco-like no turn in place
        "forward_0p5": metrics(lin_vel_rmse_mps=0.5),
    })
    v = gate_verdict(rep)
    assert v["verdict"] == "PASS" and v["failures"] == []
    assert set(v["performance_failures"]) == {"stand_20s:yaw_rate_rmse_radps", "yaw_0p6:yaw_rate_rmse_radps",
                                              "forward_0p5:lin_vel_rmse_mps"}
    perf = performance_table(rep, cfg)
    assert perf["verdict"] == "FAIL"
    assert perf["named"]["stand_yaw_drift"]["value"] == 0.204 and not perf["named"]["stand_yaw_drift"]["pass"]
    assert perf["named"]["turn_in_place"]["value"] == 0.597


def test_v1_same_tracking_failures_block():
    cfg = load_gate_config("v1")
    rep = report(cfg, {"stand_20s": metrics(yaw_rate_rmse_radps=0.204)})
    v = gate_verdict(rep)
    assert v["verdict"] == "FAIL" and v["failures"] == ["stand_20s:yaw_rate_rmse_radps"]


@pytest.mark.parametrize("scenario,over,failed", [
    ("forward_0p3", {"fell": True}, "no_fall"),
    ("forward_0p3", {"nonfinite_action": True}, "finite_actions"),
    ("forward_0p3", {"pre_clip_saturation_rate": 0.06}, "pre_clip_saturation_rate"),
    ("forward_0p3", {"hard_limit_violation_steps": 1}, "hard_limit_violation_steps"),
    ("forward_0p3", {"torque_saturation_fraction": {"hip": 0.0, "thigh": 0.0, "calf": 0.06}},
     "torque_saturation_fraction.calf"),
    ("forward_0p3", {"near_limit_fraction": {"hip": 0.03, "thigh": 0.0, "calf": 0.0}}, "near_limit_fraction.hip"),
    ("stand_20s", {"mean_base_height_m": 0.229}, "mean_base_height_m"),
])
def test_v2_each_safety_check_blocks(scenario, over, failed):
    cfg = load_gate_config("v2")
    v = gate_verdict(report(cfg, {scenario: metrics(**over)}))
    assert v["verdict"] == "FAIL"
    assert f"{scenario}:{failed}" in v["failures"]


@pytest.mark.parametrize("check", ["manifest_valid", "envelope.ang_vel_z", "joint_sign_audit"])
def test_v2_spec_checks_are_safety(check):
    cfg = load_gate_config("v2")
    v = gate_verdict(report(cfg, {"forward_0p3": metrics()}, [{"check": check, "pass": False}]))
    assert v["verdict"] == "FAIL" and v["failures"] == [f"spec:{check}"]


def test_v2_envelope_needs_yaw_0p8_both_signs():
    from phoenix.sim2sim.gate import envelope_checks

    cfg = load_gate_config("v2")
    ok = spec_from_phoenix_manifest(manifest(ang=(-0.8, 0.8)), name="m", required_envelope=cfg.required_envelope)
    assert all(c["pass"] for c in envelope_checks(ok, cfg.required_envelope))
    short = spec_from_phoenix_manifest(manifest(ang=(-0.6, 1.0)), name="m", required_envelope=cfg.required_envelope)
    bad = {c["check"] for c in envelope_checks(short, cfg.required_envelope) if not c["pass"]}
    assert bad == {"envelope.ang_vel_z"}
    # The contract validator compares max |range| (1.0 >= 0.8) and accepts it; only
    # the gate's both-signs check catches a one-sided yaw envelope.
    assert short.manifest_problems == ()


# ------------------------------------------------------------------ manifest deploy block


def test_manifest_deploy_block_clip_100_and_gains_by_group():
    spec = spec_from_phoenix_manifest(
        manifest({"action_clip": 100.0, "kp": {"hip": 25.0, "thigh": 25.0, "calf": 40.0}, "kd": [0.5] * 12}),
        name="m", required_envelope=ENV)
    assert spec.action_clip == 100.0 and spec.assumptions == ()
    assert [spec.kp[i] for i, n in enumerate(spec.joint_order) if "calf" in n] == [40.0] * 4
    assert [spec.kp[i] for i, n in enumerate(spec.joint_order) if "calf" not in n] == [25.0] * 8
    action, targets, _ = spec.postprocess(np.full(12, 150.0))
    assert np.all(action == 100.0)  # clamp at the deploy clip, then scale 0.25


def test_manifest_deploy_block_by_joint_name_and_null_clip():
    names = manifest()["joint_order"]
    spec = spec_from_phoenix_manifest(
        manifest({"action_clip": None, "kp": {n: 20.0 + i for i, n in enumerate(names)}, "kd": 0.6}),
        name="m", required_envelope=ENV)
    assert spec.action_clip is None
    assert spec.kp.tolist() == [20.0 + i for i in range(12)] and np.all(spec.kd == 0.6)


@pytest.mark.parametrize("deploy", [{"kp": 25.0}, {"kd": 0.5}, {"kp": {"hip": 25.0}, "kd": 0.5},
                                    {"kp": 25.0, "kd": 0.5, "action_clip": 0.0}])
def test_manifest_deploy_block_rejects_incomplete(deploy):
    with pytest.raises(DeploySpecError):
        spec_from_phoenix_manifest(manifest(deploy), name="m", required_envelope=ENV)


def test_missing_deploy_block_is_recorded_as_assumptions():
    spec = spec_from_phoenix_manifest(manifest(), name="m", required_envelope=ENV)
    assert spec.action_clip == 1.0
    assert any("kp" in a.lower() for a in spec.assumptions)
    assert any("action_clip" in a for a in spec.assumptions)


# ------------------------------------------------------------------ MuJoCo


def test_v2_end_to_end_unclamped_deploy_fails_safety():
    pytest.importorskip("mujoco")
    from phoenix.sim2sim.gate_runner import run_gate

    cfg = load_gate_config("v2")
    spec = spec_from_phoenix_manifest(manifest({"action_clip": None, "kp": 25.0, "kd": 0.5}), name="z",
                                      required_envelope=cfg.required_envelope)
    rep = run_gate(spec, lambda obs: np.zeros(12), cfg, policy_info={}, scenario_names=["stand_20s"])
    m = rep["scenarios"]["stand_20s"]["metrics"]
    assert m["fell"] is False and m["pre_clip_saturation_rate"] is None
    assert rep["would_be_verdict"] == "FAIL"
    assert rep["failures"] == ["stand_20s:pre_clip_saturation_rate"]
    assert rep["tiers"]["safety"]["blocking"] is True and rep["tiers"]["performance"]["blocking"] is False

    spec100 = spec.with_action_clip(100.0)
    rep = run_gate(spec100, lambda obs: np.zeros(12), cfg, policy_info={}, scenario_names=["stand_20s"])
    assert rep["would_be_verdict"] == "PASS"
    assert rep["scenarios"]["stand_20s"]["metrics"]["pre_clip_saturation_rate"] == 0.0
