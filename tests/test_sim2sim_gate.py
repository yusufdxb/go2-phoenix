"""Sim2sim gate: deploy spec, obs/action path, metric math, verdict, and a MuJoCo smoke run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phoenix.sim2sim.deploy_spec import (
    OBS_LEGACY_48,
    OBS_PHOENIX_45,
    DeploySpecError,
    ObsHistory,
    load_deploy_spec,
    spec_from_phoenix_manifest,
)
from phoenix.sim2sim.gate import (
    envelope_checks,
    evaluate_scenario,
    gate_verdict,
    load_gate_config,
    roll_pitch_from_quat_wxyz,
    tracking_rmse,
    trailing_mean,
)
from phoenix.velocity.contract import JOINT_ORDER, CommandRanges, build_manifest
from phoenix.velocity.observation import actions_to_joint_targets, build_actor_observation

REPO = Path(__file__).resolve().parents[1]
PC = REPO / "configs" / "sim2sim" / "positive_control"
H25 = REPO / "configs" / "sim2real" / "manifests" / "h25_stand_only.json"
ENV = {"lin_vel_x": 0.5, "lin_vel_y": 0.3, "ang_vel_z": 0.6}


def manifest(**over):
    m = build_manifest(
        checkpoint_sha256="0" * 64,
        commands=CommandRanges((-1.0, 1.0), (-0.5, 0.5), (-1.0, 1.0), 0.1),
        git_sha="a" * 40, git_dirty=False, seed=1, task="t", simulator="isaaclab",
        reward_scales={}, domain_randomization={}, curriculum={},
    )
    m.update(over)
    return m


def state(rng):
    q = rng.normal(0, 0.2, 12) + actions_to_joint_targets(np.zeros(12), 0.25)
    quat = rng.normal(0, 0.1, 4) + np.array([1.0, 0, 0, 0])
    return dict(gyro_body=rng.normal(0, 0.5, 3), quat_wxyz=quat / np.linalg.norm(quat),
                command=(0.3, -0.1, 0.4), joint_pos=q, joint_vel=rng.normal(0, 1, 12),
                last_action=rng.normal(0, 0.5, 12))


# ------------------------------------------------------------------ gate config


def test_gate_config_is_the_preregistered_file_and_parses():
    cfg = load_gate_config("v1")
    assert Path(cfg.path).name == "gate_v1.yaml"
    names = [s.name for s in cfg.scenarios]
    assert names == ["stand_20s", "forward_0p3", "forward_0p5", "lateral_0p2", "yaw_0p6",
                     "lateral_step", "friction_0p4", "friction_1p0", "payload_2kg"]
    assert cfg.physics["physics_hz"] in (500, 1000)
    assert cfg.actuator["calf"] == {"effort_limit": 45.43, "velocity_limit": 15.70}
    assert cfg.actuator["hip"]["effort_limit"] == 23.5 and cfg.actuator["thigh"]["effort_limit"] == 23.5
    assert cfg.fall == {"base_height_min_m": 0.20, "max_abs_roll_pitch_rad": 0.8}
    assert cfg.scenario("stand_20s").duration_s == 20.0
    assert cfg.scenario("payload_2kg").payload_kg == 2.0
    assert cfg.scenario("friction_0p4").foot_friction == 0.4
    step = cfg.scenario("lateral_step")
    assert step.command_at(6.0) == (0.0, 0.3, 0.0) and step.command_at(9.0) == (0.0, -0.3, 0.0)


# ------------------------------------------------------------------ deploy spec


def test_phoenix_manifest_uses_the_deploy_obs_builder_and_clamp():
    spec = spec_from_phoenix_manifest(manifest(), name="m", required_envelope=ENV)
    assert spec.obs_builder == OBS_PHOENIX_45 and spec.obs_dim == 45
    assert spec.manifest_problems == ()
    assert spec.action_clip == 1.0 and np.all(spec.kp == 25.0) and np.all(spec.kd == 0.5)
    assert any("Kp 25" in a for a in spec.assumptions)
    s = state(np.random.default_rng(0))
    assert np.array_equal(spec.build_obs(**s), build_actor_observation(**s))
    raw = np.linspace(-3, 3, 12)
    action, targets, sat = spec.postprocess(raw)
    assert np.array_equal(action, np.clip(raw, -1, 1))
    assert np.allclose(targets, actions_to_joint_targets(np.clip(raw, -1, 1), 0.25))
    assert sat.tolist() == (np.abs(raw) > 1).tolist()


def test_phoenix_manifest_deploy_block_overrides_gains():
    spec = spec_from_phoenix_manifest(manifest(deploy={"kp": 20.0, "kd": [0.6] * 12, "action_clip": None}),
                                      name="m", required_envelope=ENV)
    assert np.all(spec.kp == 20.0) and np.all(spec.kd == 0.6) and spec.action_clip is None
    assert spec.assumptions == ()


def test_stand_only_phoenix_manifest_is_refused_for_the_gate():
    m = build_manifest(
        checkpoint_sha256=None, commands=CommandRanges((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 1.0),
        git_sha="a" * 40, git_dirty=False, seed=1, task="t", simulator="isaaclab",
        reward_scales={}, domain_randomization={}, curriculum={},
    )
    spec = spec_from_phoenix_manifest(m, name="m", required_envelope=ENV)
    assert spec.manifest_problems
    checks = envelope_checks(spec, ENV)
    assert not any(c["pass"] for c in checks)


def test_h25_legacy_obs_is_zeros_then_the_45d_builder():
    spec = load_deploy_spec(H25)
    assert spec.obs_builder == OBS_LEGACY_48 and spec.obs_dim == 48
    assert spec.manifest_problems  # never valid for velocity mode
    s = state(np.random.default_rng(1))
    o = spec.build_obs(**s)
    assert o.shape == (48,) and np.all(o[:3] == 0.0)
    assert np.array_equal(o[3:], build_actor_observation(**s))
    assert spec.action_clip == 1.0  # the gate applies the patched deploy clamp


def test_rl_sar_terms_builder_matches_hand_computed_obs():
    spec = load_deploy_spec(PC / "rl_sar_go2_robot_lab.json")
    assert spec.obs_dim == 45 and spec.history_length == 1
    s = state(np.random.default_rng(2))
    o = spec.build_obs(**s)
    from phoenix.velocity.observation import projected_gravity_wxyz

    exp = np.concatenate([s["gyro_body"] * 0.25, projected_gravity_wxyz(s["quat_wxyz"]), s["command"],
                          s["joint_pos"] - spec.default_joint_pos, s["joint_vel"] * 0.05, s["last_action"]])
    assert np.allclose(o, exp.astype(np.float32))
    _a, targets, _s = spec.postprocess(np.ones(12))
    assert np.allclose(targets - spec.default_joint_pos, [0.125, 0.25, 0.25] * 4)


def test_himloco_history_is_newest_first_zero_filled():
    spec = load_deploy_spec(PC / "rl_sar_go2_himloco.json")
    assert spec.history_length == 6 and spec.obs_dim == 270
    h = ObsHistory(3, 2)
    a = h.push(np.array([1.0, 1.0]))
    assert a.tolist() == [1, 1, 0, 0, 0, 0]
    b = h.push(np.array([2.0, 2.0]))
    assert b.tolist() == [2, 2, 1, 1, 0, 0]
    h.reset()
    assert h.push(np.array([3.0, 3.0])).tolist() == [3, 3, 0, 0, 0, 0]


def test_explicit_spec_rejects_unmeasurable_terms(tmp_path):
    data = json.loads((PC / "rl_sar_go2_robot_lab.json").read_text())
    data["observation"]["terms"].insert(0, {"name": "lin_vel", "scale": 2.0})
    p = tmp_path / "s.json"
    p.write_text(json.dumps(data))
    with pytest.raises(DeploySpecError):
        load_deploy_spec(p)


def test_unknown_schema_rejected(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"schema": "nope"}))
    with pytest.raises(DeploySpecError):
        load_deploy_spec(p)


# ------------------------------------------------------------------ metric math


def test_trailing_mean():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert trailing_mean(x, 2).tolist() == [1.0, 1.5, 2.5, 3.5]
    assert trailing_mean(x, 1).tolist() == x.tolist()


def test_tracking_rmse_excludes_transient_and_smooths_gait_oscillation():
    hz = 50
    t = np.arange(1, 501) / hz
    cmd = np.where(t[:, None] >= 1.0, [0.3, 0.0, 0.0], [0.0, 0.0, 0.0])
    # Perfect mean tracking plus a 2 Hz gait oscillation (period 0.5 s == window).
    vx = np.where(t >= 1.0, 0.3, 0.0) + 0.2 * np.sin(2 * np.pi * 2.0 * t)
    v = np.stack([vx, np.zeros_like(t)], axis=1)
    r = tracking_rmse(t, cmd, v, np.zeros_like(t), [0.0, 1.0], control_hz=hz,
                      smoothing_window_s=0.5, exclude_after_change_s=1.5)
    assert r["n_samples"] == int(np.sum(t >= 2.5 - 1e-9))
    assert r["lin_vel_rmse_mps"] < 0.01
    r_raw = tracking_rmse(t, cmd, v, np.zeros_like(t), [0.0, 1.0], control_hz=hz,
                          smoothing_window_s=0.0, exclude_after_change_s=1.5)
    assert r_raw["lin_vel_rmse_mps"] > 0.1


def test_tracking_rmse_constant_bias():
    t = np.arange(1, 301) / 50
    cmd = np.tile([0.0, 0.0, 0.6], (300, 1))
    r = tracking_rmse(t, cmd, np.zeros((300, 2)), np.full(300, 0.4), [0.0], control_hz=50,
                      smoothing_window_s=0.5, exclude_after_change_s=1.0)
    assert r["yaw_rate_rmse_radps"] == pytest.approx(0.2)
    assert r["mean_wz"] == pytest.approx(0.4)


def test_roll_pitch():
    a = 0.3
    q = [np.cos(a / 2), np.sin(a / 2), 0, 0]
    assert roll_pitch_from_quat_wxyz(q) == pytest.approx((a, 0.0))
    q = [np.cos(a / 2), 0, np.sin(a / 2), 0]
    assert roll_pitch_from_quat_wxyz(q) == pytest.approx((0.0, a))


def _metrics(**over):
    m = {
        "fell": False, "nonfinite_action": False, "pre_clip_saturation_rate": 0.0,
        "hard_limit_violation_steps": 0,
        "torque_saturation_fraction": {"hip": 0.0, "thigh": 0.0, "calf": 0.0},
        "near_limit_fraction": {"hip": 0.0, "thigh": 0.0, "calf": 0.0},
        "lin_vel_rmse_mps": 0.05, "yaw_rate_rmse_radps": 0.05, "mean_base_height_m": 0.3,
    }
    m.update(over)
    return m


def test_evaluate_scenario_reads_thresholds_from_config():
    cfg = load_gate_config("v1")
    stand = cfg.scenario("stand_20s")
    assert all(c["pass"] for c in evaluate_scenario(_metrics(), stand, cfg))
    th = cfg.thresholds
    fails = {c["check"] for c in evaluate_scenario(
        _metrics(mean_base_height_m=th["stand"]["min_mean_base_height_m"] - 0.01), stand, cfg) if not c["pass"]}
    assert fails == {"mean_base_height_m"}
    fails = {c["check"] for c in evaluate_scenario(
        _metrics(pre_clip_saturation_rate=th["all"]["max_pre_clip_saturation_rate"] + 0.01), stand, cfg)
        if not c["pass"]}
    assert fails == {"pre_clip_saturation_rate"}
    stress = cfg.scenario("friction_0p4")
    lin = th["nominal"]["max_lin_vel_rmse_mps"] + 0.01
    assert all(c["pass"] for c in evaluate_scenario(_metrics(lin_vel_rmse_mps=lin), stress, cfg))
    assert not all(c["pass"] for c in evaluate_scenario(_metrics(lin_vel_rmse_mps=lin), cfg.scenario("forward_0p3"), cfg))
    assert not all(c["pass"] for c in evaluate_scenario(_metrics(fell=True), stress, cfg))
    assert not all(c["pass"] for c in evaluate_scenario(_metrics(lin_vel_rmse_mps=None), stress, cfg))


def test_gate_verdict():
    ok = {"spec_checks": [{"check": "a", "pass": True}],
          "scenarios": {"s": {"checks": [{"check": "b", "pass": True}]}}}
    assert gate_verdict(ok) == {"verdict": "PASS", "failures": [], "performance_failures": []}
    # Untagged checks default to the v1 "gate" tier: all blocking.
    bad = {"spec_checks": [{"check": "a", "pass": False}],
           "scenarios": {"s": {"checks": [{"check": "b", "pass": False}]}}}
    assert gate_verdict(bad) == {"verdict": "FAIL", "failures": ["spec:a", "s:b"], "performance_failures": []}


# ------------------------------------------------------------------ MuJoCo


def test_real_profile_and_payload_and_timestep():
    pytest.importorskip("mujoco")
    from phoenix.sim2sim.model import REAL_ARMATURE, load_go2_model

    m0, idx, info0 = load_go2_model("real_go2", timestep=0.001)
    m2, _, info2 = load_go2_model("real_go2", timestep=0.001, payload_kg=2.0)
    assert m0.opt.timestep == 0.001
    assert info2["total_mass_kg"] == pytest.approx(info0["total_mass_kg"] + 2.0)
    arm = m0.dof_armature[idx.dof_adr]
    for name, a in zip(JOINT_ORDER, arm, strict=True):
        assert a == pytest.approx(REAL_ARMATURE[name.split("_")[1]])
    assert np.all(m0.dof_damping[idx.dof_adr] == 0.0)


def test_real_motor_envelope_is_per_group():
    from phoenix.sim2sim.gate_runner import pd_torque, real_motor_params

    motor = real_motor_params(load_gate_config())
    big = np.full(12, 1000.0)
    _c, applied = pd_torque(big, np.zeros(12), np.zeros(12), np.ones(12), np.zeros(12), motor)
    for name, tau in zip(JOINT_ORDER, applied, strict=True):
        assert tau == pytest.approx(45.43 if "calf" in name else 23.5)
    # At the calf velocity limit no further torque is available in that direction.
    qd = np.array([15.70 if "calf" in n else 0.0 for n in JOINT_ORDER])
    _c, applied = pd_torque(big, np.zeros(12), qd, np.ones(12), np.zeros(12), motor)
    for name, tau in zip(JOINT_ORDER, applied, strict=True):
        if "calf" in name:
            assert tau == pytest.approx(0.0, abs=1e-9)


def test_gate_runs_end_to_end_with_a_zero_policy(tmp_path):
    pytest.importorskip("mujoco")
    from phoenix.sim2sim.gate_runner import run_gate

    cfg = load_gate_config()
    spec = spec_from_phoenix_manifest(manifest(), name="zero", required_envelope=cfg.required_envelope)
    rep = run_gate(spec, lambda obs: np.zeros(12), cfg, policy_info={"name": "zero"},
                   scenario_names=["stand_20s"])
    assert rep["verdict"] == "DIAGNOSTIC"  # scenario subset
    r = rep["scenarios"]["stand_20s"]
    m = r["metrics"]
    # PD holding the default pose on the real plant: stands, no saturation.
    assert m["fell"] is False and m["simulated_s"] == pytest.approx(20.0)
    assert m["pre_clip_saturation_rate"] == 0.0
    assert 0.24 < m["mean_base_height_m"] < 0.34
    assert rep["joint_audit"]["pass"] is True
    assert {c["check"] for c in rep["spec_checks"]} >= {"manifest_valid", "joint_sign_audit"}
    json.dumps(rep, default=float)
