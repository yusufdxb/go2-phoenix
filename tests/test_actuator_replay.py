"""Synthetic-data tests for the actuator replay loader and metrics (no MuJoCo)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from phoenix.sim2real import actuator_replay as ar
from phoenix.sim2real.go2_model import UNITREE_MOTOR_ORDER

REPO = Path(__file__).resolve().parents[1]


def _tick(i: int, q: list[float], target: list[float], publish: bool = True) -> dict:
    return {"record": "tick", "t_mono_ns": 1_000_000_000 + i * 3_000_000, "mode": "policy",
            "publish": publish, "lowstate_fresh": True, "q_unitree": q, "dq_unitree": [0.0] * 12,
            "final_target_unitree": target, "kp_eff": 25.0, "kd_eff": 0.5}


def test_load_session_keeps_published_ticks_and_probe(tmp_path: Path) -> None:
    lines = [{"record": "manifest", "stage": "FP", "code_identity": {"sha": "abc"},
              "artifacts": {"checkpoint": {"sha256": "c0ffee"}}}]
    lines += [_tick(i, [0.1 * i] * 12, [0.2] * 12, publish=(i != 2)) for i in range(5)]
    (tmp_path / "bridge.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n{truncated")
    n = 4
    np.savez(tmp_path / "fp_trace.npz", t_mono_ns=np.arange(n) * 20_000_000 + 1_000_000_000,
             q=np.zeros((n, 12)), dq=np.zeros((n, 12)), tau_est=np.zeros((n, 12)),
             kp=np.full(n, 25.0), kd=np.full(n, 0.5), active=np.array(["FR_hip_joint"] * n),
             phase=np.array(["rise"] * n), joint_order=np.array(UNITREE_MOTOR_ORDER))
    s = ar.load_session(tmp_path)
    assert len(s.t) == 4 and s.t[0] == 0.0
    assert s.kp.shape == (4, 12) and np.all(s.kp == 25.0)
    assert s.identity["code_sha"] == "abc" and s.identity["checkpoint_sha256"] == "c0ffee"
    assert np.allclose(s.probe["t"], [0.0, 0.02, 0.04, 0.06])
    ar.save_cache(s, tmp_path / "c.npz")
    c = ar.load_cache(tmp_path / "c.npz")
    assert np.array_equal(c.q, s.q) and c.identity == s.identity and set(c.probe) == set(s.probe)


def test_best_lag_recovers_shift() -> None:
    dt = 0.002
    rng = np.random.default_rng(0)
    x = np.convolve(rng.standard_normal(5000), np.ones(25) / 25, mode="same")
    y = np.roll(x, 20)  # y trails x by 40 ms
    lag, corr = ar.best_lag(x, y, dt, 0.15)
    assert abs(lag - 0.040) < 1e-9 and corr > 0.95


def test_steady_windows_and_overshoot() -> None:
    t = np.arange(0, 3, 0.01)
    target = np.where(t < 1.0, 0.0, 0.2)[:, None].repeat(12, axis=1)
    q = target.copy()
    q[(t >= 1.0) & (t < 1.1), 0] = 0.25  # 25 % overshoot on joint 0
    dq = np.zeros_like(q)
    m = ar.steady_windows(target, dq, t, hold_s=0.5)
    assert not m[30, 0] and m[60, 0] and not m[120, 0] and m[200, 0]
    ev = ar.move_events(target, t, 0, min_move=0.05, settle_s=0.5)
    assert len(ev) == 1
    assert abs(ar.overshoot(q, target, ev[0], 0) - 0.25) < 1e-9


def test_probe_torque_check_recovers_gain_and_excludes() -> None:
    n = 200
    t = np.arange(n) * 0.02
    rng = np.random.default_rng(1)
    q = rng.uniform(-0.1, 0.1, (n, 12))
    target = np.zeros((n, 12))
    tau_pd = 25.0 * (target - q)
    s = ar.Session(name="x", t=t, mode=np.array(["policy"] * n), q=q, dq=np.zeros((n, 12)),
                   target=target, kp=np.full((n, 12), 25.0), kd=np.full((n, 12), 0.5),
                   probe={"t": t, "q": q, "dq": np.zeros((n, 12)), "tau_est": 1.03 * tau_pd + 0.05,
                          "kp": np.full(n, 25.0), "kd": np.full(n, 0.5),
                          "active": np.array(["FR_hip_joint"] * n), "phase": np.array(["rise"] * n)})
    out = ar.probe_torque_check(s, exclude={"FR_hip_joint": "test"})
    row = out["FR_hip_joint"]
    assert abs(row["gain_scale"] - 1.03) < 1e-9 and abs(row["offset_nm"] - 0.05) < 1e-9
    assert row["excluded"] == "test" and list(out) == ["FR_hip_joint"]


def test_split_is_disjoint_and_covers_every_joint_type() -> None:
    split = yaml.safe_load((REPO / "configs/sim2real/actuator_replay_split.yaml").read_text())
    assert not set(split["train"]) & set(split["heldout"])
    names = {Path(p).name for p in split["train"] + split["heldout"]}
    assert set(split["exclusions"]) <= names
