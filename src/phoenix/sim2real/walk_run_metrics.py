"""Per-run metrics, synchronized Parquet and plots for one GO2 walking run.

Offline. Reads what a walking run (harness stage ``W`` / FREE_FLOOR) already wrote:

* ``bridge.jsonl`` -- one record per bridge tick (``phoenix.sim2real.bridge_telemetry``),
  motor (Unitree) order, plus the policy-node record it acted on (policy order).
* ``walk_cmd.jsonl`` -- the /cmd_vel profile the harness published (phase, vx, vy, wz).
* ``policy.parquet`` -- optional, the 50 Hz policy-node log (IMU quaternion and gyro).

Error is split into the four places it can come from, each from recorded fields only:

* ``saturation``: requested target != final target (slew / limit clip), PD torque demand
  over the effort limit, raw action beyond the clip.
* ``motor_tracking``: final target sent to the motor vs measured q (and, when the bridge
  recorded ``tau_est``, PD torque demand vs the motor's own torque estimate).
* ``state_estimation``: what the policy was fed vs what was measured (base_lin_vel zeros,
  gyro vs quaternion-derived yaw rate).
* ``body_tracking``: commanded vs achieved body motion. Yaw comes from the IMU (quaternion
  and gyro integral, independent of the policy and of odometry). Forward displacement has
  no on-robot ground truth while sport mode is released: it is UNAVAILABLE unless an
  external measurement (tape, video) is passed in, and it is never taken from odometry.

Model prediction error (sim vs real) is not computed here: compare these metrics with the
same metrics from ``scripts/eval_physics_hypotheses.py`` on the same command profile.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .go2_model import POLICY_JOINT_ORDER
from .motor_crc import PHOENIX_FOR_MOTOR

MOTOR_JOINT_ORDER: tuple[str, ...] = tuple(POLICY_JOINT_ORDER[i] for i in PHOENIX_FOR_MOTOR)
#: Real GO2 effort limits (N m), motor order; calf 45.43 per env_cfg.CALF_EFFORT_LIMIT_NM.
EFFORT_LIMIT_MOTOR = np.array([23.5, 23.5, 45.43] * 4)
METRICS_SCHEMA = "phoenix-walk-run-metrics/v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a truncated last line from a killed writer
    return out


def _arr(rows: list[dict[str, Any]], key: str, n: int) -> np.ndarray:
    return np.array(
        [r[key] if isinstance(r.get(key), list) and len(r[key]) == n else [np.nan] * n for r in rows],
        dtype=float,
    )


def _pol(rows: list[dict[str, Any]], key: str, n: int) -> np.ndarray:
    vals = []
    for r in rows:
        p = r.get("policy") or {}
        v = p.get(key)
        vals.append(v if isinstance(v, list) and len(v) == n else [np.nan] * n)
    return np.array(vals, dtype=float)


def _sensor(rows: list[dict[str, Any]], key: str, n: int) -> np.ndarray:
    vals = []
    for r in rows:
        s = r.get("sensors") or {}
        v = s.get(key)
        vals.append(v if isinstance(v, list) and len(v) == n else [np.nan] * n)
    return np.array(vals, dtype=float)


def load_run(run_dir: Path) -> dict[str, Any]:
    """Bridge ticks as aligned numpy arrays (motor order), plus the command profile."""
    recs = _read_jsonl(run_dir / "bridge.jsonl")
    manifest = next((r for r in recs if r.get("record") == "manifest"), {})
    ticks = [r for r in recs if r.get("record") == "tick"]
    if not ticks:
        raise ValueError(f"{run_dir}: no bridge ticks")
    t0 = ticks[0]["t_mono_ns"]
    d: dict[str, Any] = {
        "manifest": manifest,
        "t": (np.array([r["t_mono_ns"] for r in ticks], dtype=np.int64) - t0) / 1e9,
        "t0_ns": t0,
        "mode": np.array([r.get("mode") or "" for r in ticks]),
        "publish": np.array([bool(r.get("publish")) for r in ticks]),
        "q": _arr(ticks, "q_unitree", 12),
        "dq": _arr(ticks, "dq_unitree", 12),
        "req": _arr(ticks, "requested_target_unitree", 12),
        "fin": _arr(ticks, "final_target_unitree", 12),
        "kp": np.array([r.get("kp") if r.get("kp") is not None else np.nan for r in ticks], float),
        "kd": np.array([r.get("kd") if r.get("kd") is not None else np.nan for r in ticks], float),
        "slew_clip": _arr(ticks, "slew_clip", 12),
        "limit_clip": _arr(ticks, "limit_clip", 12),
        "lowstate_age": np.array([r.get("lowstate_age_s") or np.nan for r in ticks], float),
        "cmd_age": np.array([r.get("cmd_age_s") or np.nan for r in ticks], float),
        "policy_seq": np.array([(r.get("policy") or {}).get("seq", -1) for r in ticks]),
        "raw_action": _pol(ticks, "raw_action", 12),
        "policy_requested": _pol(ticks, "requested_target", 12),
        "cmd_fed": _pol(ticks, "velocity_command_fed", 3),
        "base_lin_vel_fed": _pol(ticks, "base_lin_vel_fed", 3),
        "roll": np.array([(r.get("policy") or {}).get("roll_rad", np.nan) for r in ticks], float),
        "pitch": np.array([(r.get("policy") or {}).get("pitch_rad", np.nan) for r in ticks], float),
        "tau_est": _sensor(ticks, "tau_est", 12),
        "gyro": _sensor(ticks, "gyroscope", 3),
        "acc": _sensor(ticks, "accelerometer", 3),
        "rpy": _sensor(ticks, "rpy", 3),
        "quat": _sensor(ticks, "quaternion", 4),
        "foot_force": _sensor(ticks, "foot_force", 4),
    }
    wc_path = run_dir / "walk_cmd.jsonl"
    cmd = [r for r in _read_jsonl(wc_path) if "t_mono_ns" in r] if wc_path.exists() else []
    d["cmd_t"] = np.array([(r["t_mono_ns"] - t0) / 1e9 for r in cmd])
    d["cmd_v"] = np.array([[r.get("vx", 0.0), r.get("vy", 0.0), r.get("wz", 0.0)] for r in cmd])
    d["cmd_phase"] = [r.get("phase", "") for r in cmd]
    pq_path = run_dir / "policy.parquet"
    d["policy_df"] = None
    if pq_path.exists():
        import pandas as pd

        d["policy_df"] = pd.read_parquet(pq_path)
    return d


def _yaw_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.T
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _swing(x: np.ndarray) -> np.ndarray:
    return np.nanpercentile(x, 95, axis=0) - np.nanpercentile(x, 5, axis=0)


def _lag_s(target: np.ndarray, q: np.ndarray, dt: float, max_lag: int = 50) -> float:
    a = target - np.nanmean(target)
    b = q - np.nanmean(q)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or np.std(a) < 1e-6:
        return float("nan")
    best, best_k = -np.inf, 0
    for k in range(0, max_lag + 1):
        c = float(np.dot(a[: len(a) - k], b[k:]))
        if c > best:
            best, best_k = c, k
    return best_k * dt


def _stats(x: np.ndarray) -> dict[str, float | None]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "max": float(x.max()),
    }


def compute_metrics(
    d: dict[str, Any],
    measured_forward_m: float | None = None,
    measured_yaw_deg: float | None = None,
    measurement_method: str | None = None,
) -> dict[str, Any]:
    t = d["t"]
    pol = d["mode"] == "policy"
    dt = float(np.median(np.diff(t))) if len(t) > 1 else float("nan")
    out: dict[str, Any] = {"schema": METRICS_SCHEMA, "joint_order_motor": list(MOTOR_JOINT_ORDER)}
    man = d["manifest"]
    out["provenance"] = {
        "code_sha": (man.get("code_identity") or {}).get("sha"),
        "deploy_config": man.get("deploy_config"),
        "artifacts": {k: (v or {}).get("sha256") for k, v in (man.get("artifacts") or {}).items()},
        "stage": man.get("stage"),
        "start_utc": man.get("start_utc"),
        "live": man.get("live"),
    }

    # --- command windows -------------------------------------------------
    ct, cv = d["cmd_t"], d["cmd_v"]
    moving_cmd = np.abs(cv).sum(axis=1) > 1e-6 if len(cv) else np.zeros(0, bool)
    if moving_cmd.any():
        m0, m1 = float(ct[moving_cmd][0]), float(ct[moving_cmd][-1])
        cmd_int = np.zeros(3)
        for k in range(len(ct) - 1):
            cmd_int += cv[k] * (ct[k + 1] - ct[k])
        out["command"] = {
            "motion_start_s": m0,
            "motion_end_s": m1,
            "peak": np.abs(cv).max(axis=0).tolist(),
            "integral_vx_m": float(cmd_int[0]),
            "integral_vy_m": float(cmd_int[1]),
            "integral_yaw_deg": math.degrees(cmd_int[2]),
        }
    else:
        m0 = m1 = None
        out["command"] = {"motion_start_s": None, "note": "no nonzero command in walk_cmd.jsonl"}
    out["policy_window_s"] = [float(t[pol][0]), float(t[pol][-1])] if pol.any() else None

    # motion window on bridge ticks (policy mode, command active + 0.5 s settle tail)
    win = pol.copy()
    if m0 is not None:
        win &= (t >= m0) & (t <= m1 + 0.5)
    out["motion_ticks"] = int(win.sum())

    # --- body tracking (IMU) ---------------------------------------------
    body: dict[str, Any] = {}
    df = d["policy_df"]
    if df is not None and m0 is not None and len(df):
        ts = df["timestamp_s"].to_numpy()
        quat_xyzw = np.stack(df["base_quat"].to_numpy())
        q_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
        yaw = _yaw_from_quat_wxyz(q_wxyz)
        wz = np.stack(df["base_ang_vel_body"].to_numpy())[:, 2]
        c = np.stack(df["command_vel"].to_numpy())
        nz = np.flatnonzero(np.abs(c).sum(axis=1) > 1e-6)
        if nz.size:
            a, b = max(nz[0] - 10, 0), min(nz[-1] + 50, len(c) - 1)
            cmd_yaw = math.degrees(float(np.sum(c[a:b, 2] * np.diff(ts[a : b + 1]))))
            imu_yaw = math.degrees(float(yaw[b] - yaw[a]))
            gyro_yaw = math.degrees(float(np.sum(wz[a:b] * np.diff(ts[a : b + 1]))))
            body.update(
                yaw_commanded_deg=cmd_yaw,
                yaw_imu_quat_deg=imu_yaw,
                yaw_gyro_integral_deg=gyro_yaw,
                yaw_tracking_frac=(imu_yaw / cmd_yaw) if abs(cmd_yaw) > 1.0 else None,
                yaw_rate_rmse_radps=float(np.sqrt(np.mean((wz[nz] - c[nz, 2]) ** 2))),
                yaw_quat_vs_gyro_deg=imu_yaw - gyro_yaw,
            )
    body["forward_commanded_m"] = out["command"].get("integral_vx_m")
    if measured_forward_m is not None:
        body["forward_measured_m"] = measured_forward_m
        body["forward_measurement_method"] = measurement_method or "external (unspecified)"
        ci = body["forward_commanded_m"]
        body["forward_tracking_frac"] = measured_forward_m / ci if ci and abs(ci) > 1e-3 else None
    else:
        body["forward_measured_m"] = None
        body["forward_note"] = (
            "UNAVAILABLE: no independent displacement measurement passed in; "
            "robot odometry is not used as ground truth"
        )
    if measured_yaw_deg is not None:
        body["yaw_external_deg"] = measured_yaw_deg
    out["body_tracking"] = body

    # --- attitude ----------------------------------------------------------
    roll, pitch = d["roll"], d["pitch"]
    out["attitude"] = {
        "roll_deg_motion": _stats(np.degrees(np.abs(roll[win]))),
        "pitch_deg_motion": _stats(np.degrees(pitch[win])),
    }

    # --- joint (motor) tracking ---------------------------------------------
    q, fin, req, dq = d["q"][win], d["fin"][win], d["req"][win], d["dq"][win]
    joints: dict[str, Any] = {}
    if len(q) > 10:
        sw_q, sw_t = _swing(q), _swing(fin)
        rms = np.sqrt(np.nanmean((fin - q) ** 2, axis=0))
        for i, n in enumerate(MOTOR_JOINT_ORDER):
            joints[n] = {
                "target_swing_rad": float(sw_t[i]),
                "q_swing_rad": float(sw_q[i]),
                "swing_ratio": float(sw_q[i] / sw_t[i]) if sw_t[i] > 1e-6 else None,
                "rms_target_minus_q_rad": float(rms[i]),
                "lag_s": _lag_s(fin[:, i], q[:, i], dt),
            }
    out["motor_tracking"] = {"joints": joints}
    kp, kd = d["kp"][win], d["kd"][win]
    tau_pd = kp[:, None] * (fin - q) - kd[:, None] * dq
    tau_est = d["tau_est"][win]
    mt = out["motor_tracking"]
    mt["pd_torque_demand_abs_nm"] = {n: _stats(np.abs(tau_pd[:, i])) for i, n in enumerate(MOTOR_JOINT_ORDER)}
    if np.isfinite(tau_est).any():
        err = tau_est - tau_pd
        mt["tau_est_minus_pd_demand_nm"] = {
            n: {"mean": float(np.nanmean(err[:, i])), "rms": float(np.sqrt(np.nanmean(err[:, i] ** 2)))}
            for i, n in enumerate(MOTOR_JOINT_ORDER)
        }
    else:
        mt["tau_est_minus_pd_demand_nm"] = "UNAVAILABLE: bridge did not record tau_est in this run"

    # --- saturation -------------------------------------------------------
    clipped = np.any(np.abs(req - fin) > 1e-6, axis=1)
    over = np.abs(tau_pd) > EFFORT_LIMIT_MOTOR[None, :]
    raw = d["raw_action"][win]
    out["saturation"] = {
        "target_altered_tick_frac": float(np.mean(clipped)) if len(clipped) else None,
        "slew_clip_tick_frac": float(np.mean(np.nan_to_num(d["slew_clip"][win]).any(axis=1))) if win.any() else None,
        "limit_clip_tick_frac": float(np.mean(np.nan_to_num(d["limit_clip"][win]).any(axis=1))) if win.any() else None,
        "pd_demand_over_effort_limit_frac": float(np.mean(over.any(axis=1))) if len(over) else None,
        "raw_action_abs_p99": float(np.nanpercentile(np.abs(raw), 99)) if np.isfinite(raw).any() else None,
    }

    # --- state estimation ---------------------------------------------------
    blv = d["base_lin_vel_fed"][win]
    se: dict[str, Any] = {
        "base_lin_vel_fed_all_zero": bool(np.isfinite(blv).any() and np.nanmax(np.abs(blv)) == 0.0),
    }
    if "yaw_quat_vs_gyro_deg" in body:
        se["yaw_quat_minus_gyro_integral_deg"] = body["yaw_quat_vs_gyro_deg"]
    out["state_estimation"] = se

    # --- timing ---------------------------------------------------------------
    tick_dt = np.diff(t[pol]) if pol.sum() > 1 else np.zeros(0)
    seq = d["policy_seq"][pol]
    new_pol = np.flatnonzero(np.diff(seq) != 0) if len(seq) > 1 else np.zeros(0, int)
    pol_dt = np.diff(t[pol][new_pol]) if len(new_pol) > 1 else np.zeros(0)
    out["timing"] = {
        "bridge_tick_dt_s": _stats(tick_dt),
        "bridge_rate_hz": float(1.0 / np.median(tick_dt)) if len(tick_dt) else None,
        "policy_update_dt_s": _stats(pol_dt),
        "lowstate_age_s": _stats(d["lowstate_age"][pol]),
        "policy_cmd_age_s": _stats(d["cmd_age"][pol]),
    }
    return out


def synchronized_frame(d: dict[str, Any]):
    """One row per bridge tick: every logged channel, flattened with named columns."""
    import pandas as pd

    cols: dict[str, Any] = {"t_s": d["t"], "mode": d["mode"], "publish": d["publish"], "kp": d["kp"], "kd": d["kd"]}
    for key in ("q", "dq", "req", "fin", "tau_est", "slew_clip", "limit_clip"):
        for i, n in enumerate(MOTOR_JOINT_ORDER):
            cols[f"{key}.{n}"] = d[key][:, i]
    for i, n in enumerate(POLICY_JOINT_ORDER):
        cols[f"raw_action.{n}"] = d["raw_action"][:, i]
    for key, names in (
        ("cmd_fed", ("vx", "vy", "wz")),
        ("gyro", ("x", "y", "z")),
        ("acc", ("x", "y", "z")),
        ("rpy", ("r", "p", "y")),
        ("quat", ("w", "x", "y", "z")),
        ("foot_force", ("FR", "FL", "RR", "RL")),
    ):
        for i, n in enumerate(names):
            cols[f"{key}.{n}"] = d[key][:, i]
    cols["roll"] = d["roll"]
    cols["pitch"] = d["pitch"]
    cols["policy_seq"] = d["policy_seq"]
    cols["lowstate_age_s"] = d["lowstate_age"]
    cols["cmd_age_s"] = d["cmd_age"]
    # requested command (harness) at each tick: last walk_cmd sample at or before t
    ct, cv = d["cmd_t"], d["cmd_v"]
    if len(ct):
        k = np.searchsorted(ct, d["t"], side="right") - 1
        req_cmd = np.where(k[:, None] >= 0, cv[np.clip(k, 0, None)], 0.0)
    else:
        req_cmd = np.zeros((len(d["t"]), 3))
    for i, n in enumerate(("vx", "vy", "wz")):
        cols[f"cmd_requested.{n}"] = req_cmd[:, i]
    return pd.DataFrame(cols)


def make_plots(d: dict[str, Any], out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    t, pol = d["t"], d["mode"] == "policy"
    df = d["policy_df"]
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    if len(d["cmd_t"]):
        ax[0].step(d["cmd_t"], d["cmd_v"][:, 0], where="post", label="vx cmd")
        ax[1].step(d["cmd_t"], d["cmd_v"][:, 2], where="post", label="wz cmd")
    if df is not None and len(df):
        ts = df["timestamp_s"].to_numpy()
        # policy.parquet timestamps are node-relative: align by its first row to the first policy tick
        ts = ts - ts[0] + (t[pol][0] if pol.any() else 0.0)
        wz = np.stack(df["base_ang_vel_body"].to_numpy())[:, 2]
        ax[1].plot(ts, wz, lw=0.8, label="wz gyro (policy log)")
    ax[2].plot(t[pol], np.degrees(d["roll"][pol]), lw=0.8, label="roll")
    ax[2].plot(t[pol], np.degrees(d["pitch"][pol]), lw=0.8, label="pitch")
    ax[0].set_ylabel("m/s")
    ax[1].set_ylabel("rad/s")
    ax[2].set_ylabel("deg")
    ax[2].set_xlabel("s since first bridge tick")
    for a in ax:
        a.legend(loc="upper right", fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "body.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(str(p))

    fig, ax = plt.subplots(4, 3, figsize=(13, 10), sharex=True)
    for i, n in enumerate(MOTOR_JOINT_ORDER):
        a = ax[i // 3, i % 3]
        a.plot(t[pol], d["fin"][pol, i], lw=0.8, label="target sent")
        a.plot(t[pol], d["q"][pol, i], lw=0.8, label="q measured")
        a.set_title(n, fontsize=9)
        a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    p = out_dir / "joints.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(str(p))

    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    clipped = np.any(np.abs(d["req"] - d["fin"]) > 1e-6, axis=1)
    ax[0].plot(t[pol], clipped[pol].astype(float), lw=0.8, label="target altered (clip)")
    tau_pd = d["kp"][:, None] * (d["fin"] - d["q"]) - d["kd"][:, None] * d["dq"]
    ax[1].plot(t[pol], np.nanmax(np.abs(tau_pd[pol]) / EFFORT_LIMIT_MOTOR, axis=1), lw=0.8, label="max |PD tau| / limit")
    if np.isfinite(d["tau_est"]).any():
        ax[1].plot(t[pol], np.nanmax(np.abs(d["tau_est"][pol]) / EFFORT_LIMIT_MOTOR, axis=1), lw=0.8, label="max |tau_est| / limit")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "saturation.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(str(p))
    return written


def analyze_run(
    run_dir: Path,
    out_dir: Path | None = None,
    measured_forward_m: float | None = None,
    measured_yaw_deg: float | None = None,
    measurement_method: str | None = None,
    plots: bool = True,
) -> dict[str, Any]:
    out_dir = out_dir or run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = load_run(run_dir)
    m = compute_metrics(d, measured_forward_m, measured_yaw_deg, measurement_method)
    m["run_dir"] = str(run_dir)
    frame = synchronized_frame(d)
    frame.to_parquet(out_dir / "ticks.parquet", index=False)
    m["ticks_parquet"] = str(out_dir / "ticks.parquet")
    if plots:
        m["plots"] = make_plots(d, out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2, default=float))
    return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=Path, nargs="+")
    ap.add_argument("--measured-forward-m", type=float, default=None)
    ap.add_argument("--measured-yaw-deg", type=float, default=None)
    ap.add_argument("--measurement-method", default=None)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    for rd in args.run_dir:
        m = analyze_run(
            rd,
            measured_forward_m=args.measured_forward_m,
            measured_yaw_deg=args.measured_yaw_deg,
            measurement_method=args.measurement_method,
            plots=not args.no_plots,
        )
        b = m["body_tracking"]
        print(
            f"{rd.name}: yaw cmd {b.get('yaw_commanded_deg')} imu {b.get('yaw_imu_quat_deg')} "
            f"frac {b.get('yaw_tracking_frac')} | fwd cmd {b.get('forward_commanded_m')} "
            f"meas {b.get('forward_measured_m')} | sat {m['saturation']} | rate {m['timing']['bridge_rate_hz']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
