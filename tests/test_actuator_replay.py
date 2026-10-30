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


def _ramp_session(stuck_until: float, n: int = 1000, dt: float = 0.002) -> tuple[ar.Session, np.ndarray]:
    """FR calf ramps up then down at 0.05 rad/s; hw q lags until ``stuck_until`` s into each ramp."""
    t = np.arange(n) * dt
    half = n // 2
    off = np.where(np.arange(n) < half, t * 0.05, (t[half] - (t - t[half])) * 0.05)
    target = np.zeros((n, 12))
    target[:, 2] = off
    q = target.copy()
    for a, b in ((0, half), (half, n)):
        held = (t[a:b] - t[a]) < stuck_until
        q[a:b][held, 2] = off[a]
        q[a:b][~held, 2] -= np.sign(off[b - 1] - off[a]) * stuck_until * 0.05  # keeps the lag built while stuck
    phase = np.where(np.arange(n) < half, "rise", "fall")
    probe = {"t": t, "phase": phase, "active": np.array(["FR_calf_joint"] * n)}
    s = ar.Session(name="x", t=t, mode=np.array(["probe"] * n), q=q, dq=np.zeros((n, 12)), target=target,
                   kp=np.full((n, 12), 25.0), kd=np.full((n, 12), 0.5), probe=probe)
    return s, q


def test_probe_ramps_and_breakaway_on_synthetic_stick() -> None:
    s, q = _ramp_session(stuck_until=0.3)
    r = ar.probe_ramps(s)
    assert [(j, ph, d) for j, ph, _, _, d in r] == [(2, "rise", 1.0), (2, "fall", -1.0)]
    for j, _ph, t0, t1, d in r:
        tb, err = ar.breakaway(q[:, j], s.target[:, j], s.t, t0, t1, d)
        assert 0.3 < tb < 0.3 + 0.003 / 0.05 + 0.01  # released at 0.3 s, then needs 3 mrad of travel
        assert err > 0.3 * 0.05 * 0.9  # lag built up while stuck, signed along the ramp
    still = np.zeros_like(q)
    assert np.isnan(ar.breakaway(still[:, 2], s.target[:, 2], s.t, *r[0][2:4], r[0][4])[0])


def test_stick_slip_metrics_self_agreement_and_miss_filter() -> None:
    s, q = _ramp_session(stuck_until=0.3)
    m = ar.stick_slip_metrics(s, q.copy())["FR_calf_joint"]
    assert m["stuck_agree"] == 1.0 and abs(m["dir_resid_split"]) < 1e-12
    assert m["n_ramps"] == 2 and m["n_ramps_both_broke"] == 2 and m["breakaway_abs_err_s"] == 0.0
    # a sim joint that never moves is not scored on timing: only n_ramps_both_broke reveals the miss
    m0 = ar.stick_slip_metrics(s, np.zeros_like(q))["FR_calf_joint"]
    assert m0["n_ramps_both_broke"] == 0 and "breakaway_abs_err_s" not in m0


def test_smooth_velocity_constant_rate() -> None:
    t = np.arange(500) * 0.002
    q = np.outer(0.05 * t, np.ones(12))
    v = ar.smooth_velocity(q, t)
    assert np.allclose(v, 0.05)


def _replay_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("replay_script", REPO / "scripts/actuator_replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tilt_gravity_and_set_joint_friction() -> None:
    from types import SimpleNamespace

    rs = _replay_script()
    m = SimpleNamespace(opt=SimpleNamespace(gravity=np.zeros(3)), dof_frictionloss=np.zeros(18),
                        dof_solref=np.zeros((18, 2)), dof_solimp=np.zeros((18, 5)))
    rs.tilt_gravity(m)
    assert np.allclose(m.opt.gravity, [0, 0, -9.81])
    rs.tilt_gravity(m, roll_deg=3.0, pitch_deg=-2.0)
    assert np.isclose(np.linalg.norm(m.opt.gravity), 9.81) and m.opt.gravity[2] < -9.79
    dadr = np.arange(6, 18)
    rs.set_joint_friction(m, dadr, {"hip": 0.5, "thigh": 0.3, "calf": 1.2}, [0.002, 1.0], [0.999, 0.999, 0.001])
    assert np.allclose(m.dof_frictionloss[:6], 0) and np.allclose(m.dof_frictionloss[6:9], [0.5, 0.3, 1.2])
    assert np.allclose(m.dof_solref[6:], [0.002, 1.0]) and np.allclose(m.dof_solimp[6:, :3], [0.999, 0.999, 0.001])
    rs.set_joint_friction(m, dadr, {"hip": 0.5, "thigh": 0.3, "calf": 1.2, "RL_calf_joint": 0.607})
    assert np.isclose(m.dof_frictionloss[17], 0.607) and np.isclose(m.dof_frictionloss[14], 1.2)
    per_joint = {n: 0.1 * (k + 1) for k, n in enumerate(UNITREE_MOTOR_ORDER)}
    rs.set_joint_friction(m, dadr, per_joint)
    assert np.allclose(m.dof_frictionloss[6:], 0.1 * np.arange(1, 13))


def test_frictionloss_holds_below_threshold_and_slides_above() -> None:
    """The model B mechanism on a one-hinge MuJoCo model, gravity off: a constant torque
    below F leaves the joint at rest (stiction, not damping); above F it accelerates."""
    import pytest

    mujoco = pytest.importorskip("mujoco")
    xml = """<mujoco><option timestep="0.001" gravity="0 0 0"/><worldbody><body>
      <joint name="j" type="hinge" axis="0 0 1" frictionloss="1.0" solreffriction="0.002 1"
             solimpfriction="0.999 0.999 0.001"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.2 0 0" mass="1"/></body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    drift = {}
    for frac in (0.5, 0.9, 1.1):
        d = mujoco.MjData(model)
        for _ in range(2000):
            d.qfrc_applied[0] = frac
            mujoco.mj_step(model, d)
        drift[frac] = abs(float(d.qpos[0]))
    assert drift[0.5] < 1e-3 and drift[0.9] < 1e-3
    assert drift[1.1] > 0.1
