"""Actuator-level sim-to-real replay: recorded GO2 motor commands vs simulated joints.

Data (hung robot, feet off the ground): each ``FP_*`` session directory holds the
bridge telemetry ``bridge.jsonl`` (~320 Hz ticks: measured ``q``/``dq``, the
``final_target`` and effective ``kp``/``kd`` actually published) and, when the
friction probe ran, ``fp_trace.npz`` (50 Hz, adds the motor's ``tau_est``). Both
are in :data:`UNITREE_MOTOR_ORDER` and share the Jetson monotonic clock.

Signal status (what each array is, never inferred from a zero):

* measured: ``q``, ``dq`` (lowstate), ``tau_est`` (motor current estimate, probe only)
* commanded (logged as sent): ``target``, ``kp``, ``kd``, ``mode``
* unavailable: motor temperature, battery voltage, base pose (assumed level, fixed)

Replay rule: the simulator is driven open-loop by the logged commands (zero-order
hold per tick) and is reset to the measured state every ``reset_s`` seconds. No
simulator quantity is fed back from hardware between resets.

This module is numpy only (no MuJoCo, no ROS); ``scripts/actuator_replay.py`` runs
the simulator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .go2_model import UNITREE_MOTOR_ORDER, torque_limits_in_order

JOINT_TYPES = ("hip", "thigh", "calf")
SIGNAL_STATUS = {
    "q": "measured",
    "dq": "measured",
    "tau_est": "measured (motor current estimate, fp_trace only, 50 Hz)",
    "target": "commanded (bridge final_target)",
    "kp/kd": "commanded (bridge kp_eff/kd_eff)",
    "mode": "commanded (bridge mode)",
    "temperature": "unavailable",
    "battery": "unavailable",
    "base_pose": "unavailable (hung; assumed level and fixed)",
    "contact": "not applicable (feet off ground)",
}


def joint_type(name: str) -> str:
    return name.split("_")[1]


@dataclass
class Session:
    """One recording, all arrays in UNITREE_MOTOR_ORDER, time in seconds."""

    name: str
    t: np.ndarray  # (N,)
    mode: np.ndarray  # (N,) str
    q: np.ndarray  # (N, 12)
    dq: np.ndarray
    target: np.ndarray
    kp: np.ndarray  # (N, 12)
    kd: np.ndarray
    identity: dict[str, Any] = field(default_factory=dict)
    probe: dict[str, np.ndarray] | None = None  # t, q, dq, tau_est, kp, kd, active, phase


def _per_joint(v: Any) -> list[float]:
    return list(v) if isinstance(v, list) else [float(v)] * 12


def load_session(session_dir: str | Path) -> Session:
    """Parse ``bridge.jsonl`` (+ ``fp_trace.npz`` if present) into a :class:`Session`.

    Keeps only published ticks with a fresh state, a target and gains.
    """
    d = Path(session_dir)
    t, mode, q, dq, tgt, kp, kd = [], [], [], [], [], [], []
    ident: dict[str, Any] = {}
    with (d / "bridge.jsonl").open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = r.get("record")
            if rec == "manifest":
                code = r.get("code_identity") or {}
                arts = r.get("artifacts") or {}
                ident = {
                    "stage": r.get("stage"),
                    "code_sha": code.get("sha"),
                    "branch": code.get("branch"),
                    "deploy_config": (r.get("deploy_config") or {}).get("path"),
                    "checkpoint_sha256": (arts.get("checkpoint") or {}).get("sha256"),
                    "onnx_sha256": (arts.get("policy.onnx") or {}).get("sha256"),
                }
                continue
            if rec != "tick" or not r.get("publish") or not r.get("lowstate_fresh"):
                continue
            if r.get("final_target_unitree") is None or r.get("kp_eff") is None:
                continue
            t.append(r["t_mono_ns"])
            mode.append(r["mode"])
            q.append(r["q_unitree"])
            dq.append(r["dq_unitree"])
            tgt.append(r["final_target_unitree"])
            kp.append(_per_joint(r["kp_eff"]))
            kd.append(_per_joint(r["kd_eff"]))
    if not t:
        raise ValueError(f"{d}: no usable published ticks")
    t_ns = np.asarray(t, dtype=np.int64)
    t0 = int(t_ns[0])
    probe = None
    fp = d / "fp_trace.npz"
    if fp.exists():
        z = np.load(fp)
        if tuple(str(n) for n in z["joint_order"]) != UNITREE_MOTOR_ORDER:
            raise ValueError(f"{fp}: joint_order is not the Unitree motor order")
        probe = {
            "t": (z["t_mono_ns"].astype(np.int64) - t0) * 1e-9,
            "q": z["q"],
            "dq": z["dq"],
            "tau_est": z["tau_est"],
            "kp": z["kp"],
            "kd": z["kd"],
            "active": z["active"].astype(str),
            "phase": z["phase"].astype(str),
        }
    return Session(
        name=d.name,
        t=(t_ns - t0) * 1e-9,
        mode=np.asarray(mode),
        q=np.asarray(q, dtype=np.float64),
        dq=np.asarray(dq, dtype=np.float64),
        target=np.asarray(tgt, dtype=np.float64),
        kp=np.asarray(kp, dtype=np.float64),
        kd=np.asarray(kd, dtype=np.float64),
        identity=ident,
        probe=probe,
    )


def save_cache(s: Session, path: Path) -> None:
    extra = {f"probe_{k}": v for k, v in (s.probe or {}).items()}
    np.savez_compressed(
        path, name=s.name, t=s.t, mode=s.mode, q=s.q, dq=s.dq, target=s.target,
        kp=s.kp, kd=s.kd, identity=json.dumps(s.identity), **extra,
    )


def load_cache(path: Path) -> Session:
    z = np.load(path)
    probe = {k[6:]: z[k] for k in z.files if k.startswith("probe_")} or None
    return Session(
        name=str(z["name"]), t=z["t"], mode=z["mode"].astype(str), q=z["q"], dq=z["dq"],
        target=z["target"], kp=z["kp"], kd=z["kd"], identity=json.loads(str(z["identity"])),
        probe=probe,
    )


# ----------------------------------------------------------------- metrics


def best_lag(x: np.ndarray, y: np.ndarray, dt: float, max_lag_s: float) -> tuple[float, float]:
    """Lag (s, >0 means ``y`` trails ``x``) maximizing normalized correlation; and the peak."""
    x = x - x.mean()
    y = y - y.mean()
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx < 1e-12 or ny < 1e-12:
        return float("nan"), float("nan")
    m = int(round(max_lag_s / dt))
    best, best_k = -np.inf, 0
    for k in range(-m, m + 1):
        c = np.dot(x[: len(x) - k], y[k:]) if k >= 0 else np.dot(x[-k:], y[: len(y) + k])
        if c > best:
            best, best_k = c, k
    return best_k * dt, float(best / (nx * ny))


def steady_windows(target: np.ndarray, dq: np.ndarray, t: np.ndarray, hold_s: float = 0.5,
                   tol: float = 1e-4, still: float = 0.02) -> np.ndarray:
    """Mask (N, 12): the target has been constant for ``hold_s`` and the joint is still."""
    n = len(t)
    mask = np.zeros(target.shape, dtype=bool)
    for j in range(target.shape[1]):
        start = 0
        for i in range(1, n + 1):
            if i == n or abs(target[i, j] - target[start, j]) > tol:
                sel = (t[start:i] - t[start]) >= hold_s
                mask[start:i, j] = sel
                start = i
    return mask & (np.abs(dq) < still)


def move_events(target: np.ndarray, t: np.ndarray, j: int, min_move: float = 0.05,
                settle_s: float = 0.5, tol: float = 1e-4) -> list[tuple[int, int, int]]:
    """``(i_start, i_end, i_settle_end)`` for target moves >= ``min_move`` followed by a hold."""
    x = target[:, j]
    const = np.abs(np.diff(x, prepend=x[0])) <= tol
    events, i, n = [], 0, len(x)
    while i < n:
        if const[i]:
            i += 1
            continue
        s = i - 1 if i > 0 else 0
        while i < n and not const[i]:
            i += 1
        e = i
        k = e
        while k < n and const[k] and t[k] - t[e if e < n else n - 1] < settle_s:
            k += 1
        if k < n and t[min(k, n - 1)] - t[min(e, n - 1)] >= settle_s * 0.99 and abs(x[min(e, n - 1)] - x[s]) >= min_move:
            events.append((s, e, k))
        i = max(k, e + 1)
    return events


def overshoot(q: np.ndarray, target: np.ndarray, ev: tuple[int, int, int], j: int) -> float:
    """Peak excursion past the settled position, as a fraction of the move size."""
    s, e, k = ev
    move = target[e, j] - target[s, j]
    settled = np.mean(q[max(k - 20, e):k, j])
    exc = np.sign(move) * (q[s:k, j] - settled)
    return float(max(exc.max(), 0.0) / abs(move))


def pd_torque(s: Session) -> np.ndarray:
    return s.kp * (s.target - s.q) - s.kd * s.dq


def joint_metrics(s: Session, q_sim: np.ndarray, dq_sim: np.ndarray, sim_clip: np.ndarray,
                  dt_grid: float = 0.002, exclude: dict[str, str] | None = None) -> dict[str, Any]:
    """Per-joint, per-mode metrics of the hardware vs the replayed simulator.

    Joints in ``exclude`` (name -> reason) keep their per-joint row, tagged, and are
    left out of the per-mode aggregates.
    """
    exclude = exclude or {}
    names = UNITREE_MOTOR_ORDER
    lim = torque_limits_in_order(names)
    tau_hw = pd_torque(s)
    steady = steady_windows(s.target, s.dq, s.t)
    grid = np.arange(s.t[0], s.t[-1], dt_grid)
    out: dict[str, Any] = {"per_joint": {}, "per_mode": {}}
    for j, n in enumerate(names):
        qh, qs = s.q[:, j], q_sim[:, j]
        m = {
            "pos_rmse": float(np.sqrt(np.mean((qs - qh) ** 2))),
            "vel_rmse": float(np.sqrt(np.mean((dq_sim[:, j] - s.dq[:, j]) ** 2))),
            "tau_cmd_hw_abs_max": float(np.abs(tau_hw[:, j]).max()),
            "hw_frac_over_limit": float(np.mean(np.abs(tau_hw[:, j]) > lim[j])),
            "sim_frac_clipped": float(np.mean(sim_clip[:, j])),
            "hw_frac_dq_over_limit": float(np.mean(np.abs(s.dq[:, j]) > (15.7 if "calf" in n else 30.0))),
        }
        if steady[:, j].sum() > 50:
            w = steady[:, j]
            m["sse_hw"] = float(np.mean(s.target[w, j] - qh[w]))
            m["sse_sim"] = float(np.mean(s.target[w, j] - qs[w]))
            m["sse_diff"] = m["sse_sim"] - m["sse_hw"]
            m["steady_n"] = int(w.sum())
        dqh = np.interp(grid, s.t, s.dq[:, j])
        dqs = np.interp(grid, s.t, dq_sim[:, j])
        tg = np.gradient(np.interp(grid, s.t, s.target[:, j]), dt_grid)
        m["lag_hw_s"], m["lag_hw_corr"] = best_lag(tg, dqh, dt_grid, 0.15)
        m["lag_sim_s"], m["lag_sim_corr"] = best_lag(tg, dqs, dt_grid, 0.15)
        m["lag_hw_vs_sim_s"], m["lag_hw_vs_sim_corr"] = best_lag(dqs, dqh, dt_grid, 0.15)
        ovs = [(overshoot(qh[:, None], s.target[:, [j]], ev, 0), overshoot(qs[:, None], s.target[:, [j]], ev, 0))
               for ev in move_events(s.target, s.t, j)]
        if ovs:
            a = np.asarray(ovs)
            m["overshoot_hw"], m["overshoot_sim"], m["n_moves"] = float(a[:, 0].mean()), float(a[:, 1].mean()), len(ovs)
        if n in exclude:
            m["excluded"] = exclude[n]
        out["per_joint"][n] = m
    for mode in sorted(set(s.mode.tolist())):
        sel = s.mode == mode
        if sel.sum() < 50:
            continue
        err = q_sim[sel] - s.q[sel]
        row = {"n": int(sel.sum()), "kp": float(np.median(s.kp[sel])), "kd": float(np.median(s.kd[sel]))}
        for jt in JOINT_TYPES:
            cols = [i for i, n in enumerate(names) if joint_type(n) == jt and n not in exclude]
            row[f"pos_rmse_{jt}"] = float(np.sqrt(np.mean(err[:, cols] ** 2)))
            row[f"vel_rmse_{jt}"] = float(np.sqrt(np.mean((dq_sim[sel][:, cols] - s.dq[sel][:, cols]) ** 2)))
        out["per_mode"][mode] = row
    return out


def probe_torque_check(s: Session, tau_sim_at: np.ndarray | None = None,
                       exclude: dict[str, str] | None = None) -> dict[str, Any]:
    """Motor ``tau_est`` vs the PD law on the bridge's published target, active joint only.

    Fits ``tau_est = a * tau_pd + b`` per joint; ``a`` is the effective gain scale.
    """
    if s.probe is None or len(s.probe["t"]) < 50:
        return {}
    p = s.probe
    idx = np.clip(np.searchsorted(s.t, p["t"], side="right") - 1, 0, len(s.t) - 1)
    tgt, kp, kd = s.target[idx], s.kp[idx], s.kd[idx]
    tau_pd = kp * (tgt - p["q"]) - kd * p["dq"]
    out = {}
    for j, n in enumerate(UNITREE_MOTOR_ORDER):
        sel = (p["active"] == n) & np.isin(p["phase"], ["rise", "fall"])
        if sel.sum() < 30:
            continue
        x, y = tau_pd[sel, j], p["tau_est"][sel, j]
        a, b = np.polyfit(x, y, 1)
        row = {"n": int(sel.sum()), "gain_scale": float(a), "offset_nm": float(b),
               "resid_rmse_nm": float(np.sqrt(np.mean((y - (a * x + b)) ** 2))),
               "tau_est_minus_pd_rmse_nm": float(np.sqrt(np.mean((y - x) ** 2)))}
        if tau_sim_at is not None:
            row["tau_sim_minus_tau_est_rmse_nm"] = float(np.sqrt(np.mean((tau_sim_at[sel, j] - y) ** 2)))
        if exclude and n in exclude:
            row["excluded"] = exclude[n]
        out[n] = row
    return out


def smooth_velocity(q: np.ndarray, t: np.ndarray, half_window_s: float = 0.05) -> np.ndarray:
    """Central difference over +-``half_window_s``; the encoder ``dq`` is too noisy
    (std ~0.03 rad/s at rest) to tell stuck from slipping."""
    lo = np.clip(np.searchsorted(t, t - half_window_s), 0, len(t) - 1)
    hi = np.clip(np.searchsorted(t, t + half_window_s, side="right") - 1, 0, len(t) - 1)
    dt = np.maximum(t[hi] - t[lo], 1e-6)[:, None]
    return (q[hi] - q[lo]) / dt


def probe_ramps(s: Session) -> list[tuple[int, str, float, float, float]]:
    """(joint index, phase, t0, t1, direction) for each probe rise/fall ramp."""
    if s.probe is None or len(s.probe["t"]) < 50:
        return []
    ph, act, tp = s.probe["phase"], s.probe["active"], s.probe["t"]
    ch = np.flatnonzero((ph[1:] != ph[:-1]) | (act[1:] != act[:-1])) + 1
    out = []
    for a, b in zip(np.r_[0, ch], np.r_[ch, len(tp)], strict=True):
        n = str(act[a])
        if ph[a] not in ("rise", "fall") or n not in UNITREE_MOTOR_ORDER or b - a < 5:
            continue
        j = UNITREE_MOTOR_ORDER.index(n)
        t0, t1 = float(tp[a]), float(tp[b - 1])
        i0, i1 = np.searchsorted(s.t, [t0, t1])
        i1 = min(i1, len(s.t) - 1)
        direction = float(np.sign(s.target[i1, j] - s.target[i0, j]))
        if direction:
            out.append((j, str(ph[a]), t0, t1, direction))
    return out


def breakaway(q: np.ndarray, target: np.ndarray, t: np.ndarray, t0: float, t1: float, direction: float,
              thr: float = 0.003) -> tuple[float, float]:
    """Time after ramp start until the joint has moved ``thr`` rad in the ramp
    direction, and the tracking error (direction * (target - q)) at that moment."""
    w = (t >= t0) & (t <= t1)
    d = (q[w] - q[w][0]) * direction
    k = np.flatnonzero(d > thr)
    if not len(k):
        return float("nan"), float("nan")
    return float(t[w][k[0]] - t0), float(direction * (target[w][k[0]] - q[w][k[0]]))


def stick_slip_metrics(s: Session, q_sim: np.ndarray, exclude: dict[str, str] | None = None,
                       still: float = 0.01, moving_cmd: float = 0.01) -> dict[str, dict[str, float]]:
    """Per-joint friction signatures, hardware vs sim, keyed like ``joint_metrics``.

    * breakaway: per probe ramp, time to move 3 mrad and the tracking error then
    * stuck while commanded: among ticks whose target moves (> ``moving_cmd`` rad/s),
      the fraction with smoothed |v| < ``still``, and hw/sim agreement on that flag
    * direction residual: mean q_sim - q_hw over moving-target ticks, split by the
      sign of the target velocity (Coulomb mismatch shows as a split of opposite sign)
    """
    exclude = exclude or {}
    vt = smooth_velocity(s.target, s.t)
    vh, vs = smooth_velocity(s.q, s.t), smooth_velocity(q_sim, s.t)
    out: dict[str, dict[str, float]] = {n: {} for n in UNITREE_MOTOR_ORDER}
    for j, n in enumerate(UNITREE_MOTOR_ORDER):
        m = out[n]
        mov = np.abs(vt[:, j]) > moving_cmd
        if mov.sum() > 50:
            sh, ss = np.abs(vh[mov, j]) < still, np.abs(vs[mov, j]) < still
            m["stuck_frac_hw"], m["stuck_frac_sim"] = float(sh.mean()), float(ss.mean())
            m["stuck_agree"] = float((sh == ss).mean())
            r = q_sim[:, j] - s.q[:, j]
            up, dn = mov & (vt[:, j] > 0), mov & (vt[:, j] < 0)
            if up.sum() > 20 and dn.sum() > 20:
                m["dir_resid_up"], m["dir_resid_down"] = float(r[up].mean()), float(r[dn].mean())
                m["dir_resid_split"] = m["dir_resid_up"] - m["dir_resid_down"]
    rows: dict[int, list[tuple[float, ...]]] = {}
    for j, _ph, t0, t1, dr in probe_ramps(s):
        th, eh = breakaway(s.q[:, j], s.target[:, j], s.t, t0, t1, dr)
        ts, es = breakaway(q_sim[:, j], s.target[:, j], s.t, t0, t1, dr)
        rows.setdefault(j, []).append((th, ts, eh, es))
    for j, rr in rows.items():
        a = np.asarray(rr)
        ok = np.isfinite(a[:, 0]) & np.isfinite(a[:, 1])
        m = out[UNITREE_MOTOR_ORDER[j]]
        m["n_ramps"], m["n_ramps_both_broke"] = len(rr), int(ok.sum())
        if ok.any():
            m["breakaway_hw_s"], m["breakaway_sim_s"] = float(a[ok, 0].mean()), float(a[ok, 1].mean())
            m["breakaway_abs_err_s"] = float(np.abs(a[ok, 1] - a[ok, 0]).mean())
            m["err_at_break_hw"], m["err_at_break_sim"] = float(a[ok, 2].mean()), float(a[ok, 3].mean())
    for n in exclude:
        if n in out:
            out[n]["excluded"] = exclude[n]
    return out


def session_tilt_deg(session_dir: Path) -> float:
    """Median IMU tilt magnitude [deg] over valid bridge ticks (direction not logged)."""
    vals = []
    with (Path(session_dir) / "bridge.jsonl").open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("record") == "tick" and r.get("imu_valid") and r.get("tilt_deg") is not None:
                vals.append(float(r["tilt_deg"]))
    return float(np.median(vals)) if vals else float("nan")


__all__ = [
    "JOINT_TYPES", "SIGNAL_STATUS", "Session", "best_lag", "joint_metrics", "joint_type",
    "load_cache", "load_session", "move_events", "overshoot", "pd_torque",
    "breakaway", "probe_ramps", "probe_torque_check", "save_cache", "session_tilt_deg",
    "smooth_velocity", "steady_windows", "stick_slip_metrics",
]
