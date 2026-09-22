"""Phoenix v2 amendment 1: command-rate limiter, trained action clamp, telemetry v2.

Covers the deploy side (actuator gate, policy-node action map, deploy contract) and
the offline replay used for Phase A. The hard envelope is asserted unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.monitor.layers import CommandLayers, from_records
from phoenix.monitor.limiter_replay import (
    ReplayInput,
    apply_limiter,
    limiter_metrics,
    with_trained_action_clip,
)
from phoenix.sim2real import go2_model
from phoenix.sim2real.action_map import node_soft_limit, policy_action_map
from phoenix.sim2real.actuator_gate import ActuatorGate, GateParams
from phoenix.sim2real.command_wire import KIND_POLICY, encode
from phoenix.sim2real.deploy_contract import _limiter_problems
from phoenix.sim2real.go2_model import (
    LIMIT_ABORT_BAND_RAD,
    POLICY_JOINT_ORDER,
    TRAINING_DEFAULT_JOINT_POS,
    UNITREE_MOTOR_ORDER,
    limits_in_order,
)
from phoenix.sim2real.motor_crc import phoenix_to_unitree
from phoenix.sim2real.safety import TRAINED_ACTION_CLIP, rate_limit_array

ORDER = POLICY_JOINT_ORDER
T0 = 5_000_000_000
DEFAULT_P = np.asarray([TRAINING_DEFAULT_JOINT_POS[n] for n in ORDER])
DEFAULT_U = np.asarray(phoenix_to_unitree(DEFAULT_P))
LO_U, HI_U = limits_in_order(UNITREE_MOTOR_ORDER)
ZERO12 = np.zeros(12)


def ns(seconds: float) -> int:
    return T0 + int(round(seconds * 1e9))


def gate(q_u=DEFAULT_U, **over) -> ActuatorGate:
    base = dict(
        live=False,
        kp=25.0,
        kd=0.5,
        hold_kp=20.0,
        hold_kd=1.0,
        watchdog_s=0.2,
        estop_timeout_s=0.5,
        lowstate_timeout_s=0.2,
        stale_hold_s=0.2,
        first_message_timeout_s=15.0,
        limiter_mode="prev_command",
        max_delta=0.05,
        tracking_abort_rad=0.6,
        tracking_abort_s=0.2,
    )
    base.update(over)
    g = ActuatorGate(GateParams(**base), ns(0.0))
    g.on_lowstate(ns(0.0), q_u, ZERO12)
    g.on_estop(ns(0.0), False)
    return g


def send(g, t, target_p, seq):
    label, data = encode(
        ORDER,
        seq=seq,
        kind=KIND_POLICY,
        target=target_p,
        requested_target=target_p,
        q_policy=DEFAULT_P,
        obs_source_code=0.0,
        stand_only=1.0,
        velocity_command_fed=[0.0, 0.0, 0.0],
        cmd_vel_received=[0.0, 0.0, 0.0],
    )
    g.on_command(ns(t), label, data)


def step(g, t, target_p, seq, q_u=None):
    if q_u is not None:
        g.on_lowstate(ns(t), q_u, ZERO12)
    else:
        g.on_lowstate(ns(t), g._q, ZERO12)
    send(g, t, target_p, seq)
    g.on_estop(ns(t), False)
    return g.tick(ns(t))


# ------------------------------------------------------------ hard envelope
def test_abort_band_is_its_own_constant_and_unchanged() -> None:
    assert LIMIT_ABORT_BAND_RAD == 0.175
    assert "MAX_DELTA_PER_STEP_RAD" not in vars(go2_model)


def test_rate_limit_array_bounds_the_step_from_the_previous_command() -> None:
    out = rate_limit_array([1.0, -1.0, 0.01], [0.0, 0.0, 0.0], 0.05)
    assert np.allclose(out, [0.05, -0.05, 0.01])


# -------------------------------------------------------------- the gate
def test_prev_command_rate_limits_from_the_last_sent_target() -> None:
    g = gate()
    target = DEFAULT_P.copy()
    target[0] += 0.3  # FL_hip, far from the default
    j = UNITREE_MOTOR_ORDER.index(ORDER[0])
    sent = []
    for k in range(8):
        rec = step(g, 0.02 * (k + 1), target, seq=k + 1)
        assert rec["mode"] == "policy"
        sent.append(rec["final_target_unitree"][j])
    steps = np.diff([DEFAULT_U[j], *sent])
    assert np.all(np.abs(steps) <= 0.05 + 1e-12)
    assert np.isclose(sent[5], DEFAULT_U[j] + 0.3)  # reached after 6 ticks, then held
    assert np.isclose(sent[-1], DEFAULT_U[j] + 0.3)


def test_prev_command_does_not_touch_a_request_within_the_bound() -> None:
    g = gate()
    target = DEFAULT_P + 0.03
    rec = step(g, 0.02, target, seq=1)
    assert rec["slew_clip"] == [False] * 12
    assert np.allclose(rec["final_target_unitree"], phoenix_to_unitree(target))


def test_measured_q_mode_is_the_incumbent_clip() -> None:
    q = DEFAULT_U.copy()
    g = gate(q_u=q, limiter_mode="measured_q", max_delta=0.175, tracking_abort_rad=0.0)
    target = DEFAULT_P + 0.5
    rec = step(g, 0.02, target, seq=1)
    assert np.allclose(rec["final_target_unitree"], np.clip(q + 0.175, LO_U, HI_U))


def test_hard_limits_still_apply_after_the_rate_limit() -> None:
    lo_p, hi_p = limits_in_order(ORDER)
    q_p = DEFAULT_P.copy()
    q_p[4] = hi_p[4] - 0.01  # FL_thigh near its upper limit
    g = gate(q_u=np.asarray(phoenix_to_unitree(q_p)), max_delta=0.1)
    target = q_p.copy()
    target[4] = hi_p[4] + 0.1  # inside the abort band: clip, do not abort
    # The previous sent target is the hold target (measured q); one tick moves 0.1.
    rec = step(g, 0.02, target, seq=1)
    j = UNITREE_MOTOR_ORDER.index(ORDER[4])
    assert rec["mode"] == "policy"
    assert rec["final_target_unitree"][j] == pytest.approx(HI_U[j])
    assert rec["limit_clip"][j] is True


def test_abort_band_still_latches_on_the_policy_request() -> None:
    lo_p, _ = limits_in_order(ORDER)
    g = gate()
    target = DEFAULT_P.copy()
    target[8] = lo_p[8] - LIMIT_ABORT_BAND_RAD - 0.01  # FL_calf far below its limit
    rec = step(g, 0.02, target, seq=1)
    assert rec["mode"] == "hold"
    assert rec["fault"].startswith("target_beyond_limit")


def test_tracking_abort_replaces_the_effort_cap_without_rewriting_commands() -> None:
    q = DEFAULT_U.copy()
    g = gate(q_u=q, max_delta=0.175, tracking_abort_rad=0.3, tracking_abort_s=0.1)
    target = DEFAULT_P.copy()
    target[0] += 0.5
    faults = []
    for k in range(12):
        rec = step(g, 0.02 * (k + 1), target, seq=k + 1, q_u=q)  # the joint never moves
        faults.append(rec["fault"])
    assert faults[0] is None
    assert any(f and f.startswith("catastrophic_tracking_error") for f in faults)
    assert rec["mode"] == "hold"


def test_record_uses_unambiguous_v2_names() -> None:
    g = gate()
    rec = step(g, 0.02, DEFAULT_P, seq=1)
    assert "requested_target_unitree" not in rec
    assert rec["node_target_unitree"] is not None
    assert rec["soft_target_unitree"] is not None
    assert rec["limiter_mode"] == "prev_command"
    layers = from_records([{**rec, "record": "tick"}])
    assert isinstance(layers, CommandLayers)


def test_unknown_limiter_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="limiter_mode"):
        gate(limiter_mode="nope")


# --------------------------------------------------------- policy node map
def test_action_map_clamps_target_and_fed_back_action() -> None:
    raw = np.array([3.0, -7.2] + [0.5] * 10, dtype=np.float32)
    fed, req = policy_action_map(raw, DEFAULT_P, 0.25, TRAINED_ACTION_CLIP)
    assert fed[0] == 1.0 and fed[1] == -1.0 and fed[2] == 0.5
    assert np.allclose(req, DEFAULT_P + 0.25 * fed)


def test_action_map_legacy_has_no_clamp() -> None:
    raw = np.array([3.0] * 12, dtype=np.float32)
    fed, req = policy_action_map(raw, DEFAULT_P, 0.25, None)
    assert np.allclose(fed, raw) and np.allclose(req, DEFAULT_P + 0.75)


def test_node_applies_no_soft_limit_in_rate_mode() -> None:
    req = DEFAULT_P + 0.5
    assert np.allclose(node_soft_limit(req, DEFAULT_P, "prev_command", 0.05), req)
    assert np.allclose(node_soft_limit(req, DEFAULT_P, "measured_q", 0.175), DEFAULT_P + 0.175)


# ------------------------------------------------------------ deploy contract
def _cfg(**limiter):
    return {"control": {"action_clip": 1.0}, "limiter": limiter}


def test_contract_accepts_a_complete_rate_limiter_block() -> None:
    assert _limiter_problems(
        _cfg(mode="prev_command", max_delta_per_step=0.05, tracking_abort_rad=0.6, tracking_abort_s=0.2)
    ) == []


@pytest.mark.parametrize(
    "limiter",
    [
        {"mode": "prev_command", "max_delta_per_step": 0.05},  # no tracking abort
        {"mode": "prev_command", "max_delta_per_step": 0.3, "tracking_abort_rad": 0.6, "tracking_abort_s": 0.2},
        {"mode": "bogus", "max_delta_per_step": 0.05},
    ],
)
def test_contract_refuses_incomplete_or_wide_limiters(limiter) -> None:
    assert _limiter_problems(_cfg(**limiter))


def test_contract_refuses_rate_limiter_without_action_clip() -> None:
    cfg = {
        "control": {},
        "limiter": {
            "mode": "prev_command",
            "max_delta_per_step": 0.05,
            "tracking_abort_rad": 0.6,
            "tracking_abort_s": 0.2,
        },
    }
    assert any("action_clip" in p for p in _limiter_problems(cfg))


def test_contract_refuses_a_different_action_clip() -> None:
    assert _limiter_problems({"control": {"action_clip": 2.0}})


# ------------------------------------------------------------ offline replay
def _inp(req, q, anchor, raw=None):
    T = req.shape[0]
    return ReplayInput(
        t_s=np.arange(T) * 0.02,
        requested=req,
        q_node=q.copy(),
        q_bridge=q.copy(),
        sent_recorded=np.full_like(req, np.nan),
        anchor=anchor,
        kp=np.full_like(req, 25.0),
        raw_action=raw,
    )


def test_replay_limiters_on_a_step_request() -> None:
    T = 10
    default_u = DEFAULT_U
    req = np.tile(default_u + 0.2, (T, 1))
    q = np.tile(default_u, (T, 1))
    inp = _inp(req, q, default_u.copy())
    lo, hi = limits_in_order(UNITREE_MOTOR_ORDER)
    sent_mq, soft_mq, _ = apply_limiter(inp, "measured_q", 0.175, lo, hi)
    sent_rc, soft_rc, _ = apply_limiter(inp, "prev_command", 0.05, lo, hi)
    sent_h, soft_h, _ = apply_limiter(inp, "hard_only", 0.0, lo, hi)
    assert soft_mq.all()  # q never moves in this open-loop replay, so it binds forever
    assert soft_rc[:3].all() and not soft_rc[4:].any()  # 4 ticks of 0.05 reach 0.2
    assert not soft_h.any()
    m = limiter_metrics(inp, "prev_command", 0.05)
    assert m["max_abs_target_rate_rad_s"] == pytest.approx(0.05 * 50)
    assert m["hard_limit_violations"] == 0


def test_trained_clip_replay_checks_the_action_map() -> None:
    raw = np.tile(np.array([2.0] * 12), (3, 1))
    req = np.tile(DEFAULT_U + 0.25 * 2.0, (3, 1))
    inp = _inp(req, np.tile(DEFAULT_U, (3, 1)), DEFAULT_U.copy(), raw=raw)
    clipped = with_trained_action_clip(inp)
    assert np.allclose(clipped.requested, DEFAULT_U + 0.25)
    bad = _inp(req + 0.1, np.tile(DEFAULT_U, (3, 1)), DEFAULT_U.copy(), raw=raw)
    with pytest.raises(ValueError, match="action map"):
        with_trained_action_clip(bad)


# ------------------------------------------------- the frozen v2 deploy config
def test_v2_deploy_config_carries_the_frozen_amendment_2_values() -> None:
    from pathlib import Path

    import yaml

    from phoenix.sim2real.actuator_gate import limiter_params_from_config
    from phoenix.sim2real.deploy_contract import load_lock, validate_deploy_contract, verify_lock

    path = Path("configs/sim2real/deploy_stand_h25_v2.yaml")
    cfg = yaml.safe_load(path.read_text())
    assert validate_deploy_contract(cfg) == []
    assert cfg["control"]["action_clip"] == TRAINED_ACTION_CLIP
    assert limiter_params_from_config(cfg) == {
        "limiter_mode": "prev_command",
        "max_delta": 0.075,
        "tracking_abort_rad": 1.55,
        "tracking_abort_s": 0.2,
    }
    lock = load_lock(Path("configs/sim2real/locks/deploy_stand_h25_v2.lock.yaml"))
    problems, _ = verify_lock(lock, cfg, path)
    # Artifact files are gitignored; only the config hash is checkable everywhere.
    assert not [p for p in problems if "deploy config" in p or "semantic" in p]


def test_incumbent_config_keeps_the_measured_q_limiter() -> None:
    from pathlib import Path

    import yaml

    from phoenix.sim2real.actuator_gate import limiter_params_from_config

    cfg = yaml.safe_load(Path("configs/sim2real/deploy_stand_h25.yaml").read_text())
    assert limiter_params_from_config(cfg)["limiter_mode"] == "measured_q"
    assert cfg["control"].get("action_clip") is None


def test_telemetry_reader_accepts_v1_and_v2(tmp_path) -> None:
    import json

    from phoenix.sim2real.bridge_telemetry import (
        READABLE_TELEMETRY_SCHEMAS,
        TELEMETRY_SCHEMA,
        read_telemetry,
    )

    assert TELEMETRY_SCHEMA == "phoenix-bridge-telemetry/v2"
    for schema in READABLE_TELEMETRY_SCHEMAS:
        p = tmp_path / f"{schema.rsplit('/', 1)[1]}.jsonl"
        p.write_text(json.dumps({"record": "manifest", "schema": schema}) + "\n")
        manifest, ticks, end = read_telemetry(p)
        assert manifest["schema"] == schema and ticks == [] and end is None
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"record": "manifest", "schema": "other/v9"}) + "\n")
    with pytest.raises(ValueError):
        read_telemetry(bad)


def test_walk_scoring_excludes_settle_and_checks_tracking() -> None:
    from phoenix.monitor.stand_metrics import score_walk_episodes

    T, N, dt = 200, 2, 0.02
    cmd = np.zeros((T, N, 3))
    cmd[:, :, 0] = 0.5
    cmd[100:, :, 0] = 1.0  # resample at 2 s
    linv = cmd.copy()
    linv[:, 1, 0] += 0.4  # env 1 tracks badly
    linv[100:140, 0, 0] = 0.0  # env 0 lags only inside the 1 s settle window
    eps = [{"success": True}, {"success": True}]
    out = score_walk_episodes(eps, linv=linv, angv=np.zeros((T, N, 3)), cmd=cmd,
                              valid=np.ones((T, N), bool), dt=dt)
    assert eps[0]["walk_success"] and not eps[1]["walk_success"]
    assert out["walk_success_rate"] == 0.5
