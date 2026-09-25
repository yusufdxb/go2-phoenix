"""Gate v3: v2 plus a blocking responsiveness check."""

from __future__ import annotations

import copy

import pytest

from phoenix.sim2sim.gate import GateConfigError, evaluate_scenario, gate_verdict, load_gate_config


def metrics(**over):
    m = {
        "fell": False, "nonfinite_action": False, "pre_clip_saturation_rate": 0.0,
        "hard_limit_violation_steps": 0,
        "torque_saturation_fraction": {"hip": 0.0, "thigh": 0.0, "calf": 0.0},
        "near_limit_fraction": {"hip": 0.0, "thigh": 0.0, "calf": 0.0},
        "lin_vel_rmse_mps": 0.05, "yaw_rate_rmse_radps": 0.05, "mean_base_height_m": 0.3,
        "tracked_mean_vx": 0.0, "tracked_mean_vy": 0.0, "tracked_mean_wz": 0.0,
    }
    m.update(over)
    return m


def verdict(cfg, name, m):
    return gate_verdict({"spec_checks": [], "scenarios": {name: {"checks": evaluate_scenario(m, cfg.scenario(name), cfg)}}})


def test_v3_is_default_and_equals_v2_plus_responsiveness():
    v2, v3 = load_gate_config("v2"), load_gate_config()
    assert v3.version == 3 and v3.raw["name"] == "gate_v3"
    for key in ("physics", "actuator", "fall", "tracking", "scenarios", "video", "saturation",
                "required_command_envelope", "performance"):
        assert v3.raw[key] == v2.raw[key], key
    s3 = copy.deepcopy(v3.raw["safety"])
    resp = s3.pop("responsiveness")
    assert s3 == v2.raw["safety"]
    assert resp == {"min_fraction": 0.5, "scenarios": {"forward_0p3": "lin_vel_x", "forward_0p5": "lin_vel_x",
                                                       "yaw_0p6": "ang_vel_z"}}


@pytest.mark.parametrize("name,key,cmd", [("forward_0p3", "tracked_mean_vx", 0.3),
                                          ("forward_0p5", "tracked_mean_vx", 0.5),
                                          ("yaw_0p6", "tracked_mean_wz", 0.6)])
def test_frozen_policy_fails_and_responsive_policy_passes(name, key, cmd):
    cfg = load_gate_config("v3")
    frozen = verdict(cfg, name, metrics(**{key: 0.0}))
    assert frozen["verdict"] == "FAIL"
    assert frozen["failures"] == [f"{name}:responsiveness.{'ang_vel_z' if 'wz' in key else 'lin_vel_x'}"]
    assert verdict(cfg, name, metrics(**{key: 0.5 * cmd}))["verdict"] == "PASS"   # boundary inclusive
    assert verdict(cfg, name, metrics(**{key: 0.49 * cmd}))["verdict"] == "FAIL"
    assert verdict(cfg, name, metrics(**{key: -cmd}))["verdict"] == "FAIL"       # wrong sign


def test_responsiveness_is_safety_tier_and_fall_fails_it():
    cfg = load_gate_config("v3")
    checks = evaluate_scenario(metrics(fell=True, tracked_mean_vx=None), cfg.scenario("forward_0p3"), cfg)
    r = [c for c in checks if c["check"] == "responsiveness.lin_vel_x"][0]
    assert r["tier"] == "safety" and not r["pass"] and r["value"] is None


def test_unlisted_scenarios_and_older_gates_have_no_responsiveness():
    v3, v2 = load_gate_config("v3"), load_gate_config("v2")
    assert verdict(v3, "stand_20s", metrics())["verdict"] == "PASS"
    assert verdict(v3, "lateral_0p2", metrics())["verdict"] == "PASS"
    assert verdict(v2, "forward_0p3", metrics(tracked_mean_vx=0.0))["verdict"] == "PASS"


def test_responsiveness_on_zero_command_is_a_config_error():
    cfg = load_gate_config("v3")
    raw = copy.deepcopy(cfg.raw)
    raw["safety"]["responsiveness"]["scenarios"]["stand_20s"] = "lin_vel_x"
    bad = type(cfg)(raw=raw, path=cfg.path, scenarios=cfg.scenarios)
    with pytest.raises(GateConfigError):
        evaluate_scenario(metrics(), bad.scenario("stand_20s"), bad)


def test_trace_hook_records_every_control_step_without_changing_metrics():
    pytest.importorskip("mujoco")
    import numpy as np

    from phoenix.sim2sim.deploy_spec import spec_from_phoenix_manifest
    from phoenix.sim2sim.gate import GateScenario
    from phoenix.sim2sim.gate_runner import RunOptions, run_gate_scenario
    from phoenix.velocity.contract import CommandRanges, build_manifest

    cfg = load_gate_config("v3")
    m = build_manifest(checkpoint_sha256="0" * 64, commands=CommandRanges((-1, 1), (-0.5, 0.5), (-1, 1), 0.1),
                       git_sha="a" * 40, git_dirty=False, seed=1, task="t", simulator="isaaclab",
                       reward_scales={}, domain_randomization={}, curriculum={})
    spec = spec_from_phoenix_manifest(m, name="z", required_envelope=cfg.required_envelope)
    s = GateScenario("short", "nominal", 1.0, ((0.0, (0.0, 0.0, 0.0)),))
    tr = []
    a = run_gate_scenario(spec, lambda o: np.zeros(12), s, cfg, RunOptions(latency_ms=0.0, trace=tr))["metrics"]
    b = run_gate_scenario(spec, lambda o: np.zeros(12), s, cfg, RunOptions(latency_ms=0.0))["metrics"]
    assert len(tr) == 50 and set(tr[0]) == {"t", "cmd", "q", "target", "raw", "base_height", "wz"}
    assert a == b
