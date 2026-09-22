"""Directional walking diagnostics (Stage W, diagnostic only, never a gate).

Aggregate walking success can hide a policy that solves one direction of travel and
ignores the other (W1-full: backward tracked, forward not). This module splits every
episode into command SEGMENTS (at command jumps, as ``score_walk_v2`` does), drops the
first ``settle_s`` of each, and bins segments by the commanded forward velocity:

    strong_backward  vx_cmd < -0.4
    mild_backward    -0.4 <= vx_cmd < 0
    mild_forward     0 < vx_cmd <= 0.4
    strong_forward   vx_cmd > 0.4

Pure numpy over the ``steps.npz`` arrays written by ``phoenix_v2_sim_stand.py
--save-steps``. A segment "succeeds" when its own settled planar and yaw-rate errors
are within the walking thresholds and its episode met the stand criteria (so the
per-segment success is comparable across bins; it is not the per-episode gate).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .stand_metrics import WALK_V2_PROVISIONAL, attitude_from_gravity

BINS: tuple[tuple[str, float, float], ...] = (
    ("strong_backward", -np.inf, -0.4),
    ("mild_backward", -0.4, 0.0),
    ("mild_forward", 0.0, 0.4),
    ("strong_forward", 0.4, np.inf),
)


def command_segments(cmd: np.ndarray, valid: np.ndarray, settle: int, jump_tol: float = 0.05):
    """Yield ``(env, start, stop)`` settled segments of constant command, ``stop`` exclusive."""
    n_t, n_e = valid.shape
    for e in range(n_e):
        v = valid[:, e]
        if not v.any():
            continue
        end = int(np.flatnonzero(v)[-1]) + 1
        jumps = [0] + [
            k for k in range(1, end) if np.any(np.abs(cmd[k, e] - cmd[k - 1, e]) > jump_tol)
        ] + [end]
        for a, b in zip(jumps[:-1], jumps[1:], strict=True):
            if b - (a + settle) >= settle:  # at least settle_s of settled data
                yield e, a + settle, b


def bin_of(vx: float) -> str | None:
    """Bin name for a commanded forward velocity; ``None`` for exactly zero."""
    if vx < -0.4:
        return "strong_backward"
    if vx < 0.0:
        return "mild_backward"
    if vx == 0.0:
        return None
    if vx <= 0.4:
        return "mild_forward"
    return "strong_forward"


def directional_report(
    z: dict[str, np.ndarray],
    episodes: list[dict[str, Any]],
    *,
    dt: float,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Per-bin statistics. ``z`` = the loaded ``steps.npz``; ``episodes`` = episodes.jsonl rows."""
    th = dict(WALK_V2_PROVISIONAL)
    th.update(thresholds or {})
    settle = int(round(th["settle_s"] / dt))
    cmd, linv, angv, valid = z["cmd"], z["linv"], z["angv"], z["valid"].astype(bool)
    roll, pitch, _ = attitude_from_gravity(z["grav"])
    att_bad = (np.abs(roll) > 0.40) | (np.abs(pitch) > 0.40)
    clipped = np.clip(z["raw"], -1.0, 1.0)
    act = np.abs(clipped)
    # the training penalties' own quantities, per step (action_rate_l2, dof_torques_l2)
    act_rate = np.sum(np.diff(clipped, axis=0, prepend=clipped[:1]) ** 2, axis=-1)
    torque_l2 = np.sum(z["tau_a"] ** 2, axis=-1)
    raw_oob = np.abs(z["raw"]) > 1.0
    qd = np.abs(z["qd1"]) if "qd1" in z else None
    sat = np.abs(z["tau_c"] - z["tau_a"]) > 1e-3
    foot = z.get("foot_contact")
    thigh = z.get("thigh_contact")
    foot_names = [str(n) for n in z["foot_names"]] if "foot_names" in z else None
    contact_term = np.array([ep["trunk_contact"] for ep in episodes])
    stand_ok = np.array([ep["success"] for ep in episodes])

    rows: dict[str, list[dict[str, float]]] = {name: [] for name, _, _ in BINS}
    for e, a, b in command_segments(cmd, valid, settle):
        c = cmd[a, e]
        name = bin_of(float(c[0]))
        if name is None:
            continue
        v = linv[a:b, e]
        w = angv[a:b, e, 2]
        lin_err = float(np.linalg.norm(v[:, :2] - c[:2], axis=-1).mean())
        yaw_err = float(np.abs(w - c[2]).mean())
        r = {
            "vx_cmd": float(c[0]),
            "vx_achieved": float(v[:, 0].mean()),
            "abs_vx_err": float(np.abs(v[:, 0] - c[0]).mean()),
            "planar_err": lin_err,
            "yaw_err": yaw_err,
            "success": float(
                stand_ok[e] and lin_err <= th["max_lin_err_m_s"] and yaw_err <= th["max_yaw_err_rad_s"]
            ),
            "trunk_contact_episode": float(contact_term[e]),
            "attitude_violation": float(att_bad[a:b, e].any()),
            "mean_abs_action": float(act[a:b, e].mean()),
            "raw_out_of_range": float(raw_oob[a:b, e].mean()),
            "torque_saturation": float(sat[a:b, e].mean()),
            "action_rate_l2": float(act_rate[a:b, e].mean()),
            "torque_l2": float(torque_l2[a:b, e].mean()),
            "mean_abs_pitch": float(np.abs(pitch[a:b, e]).mean()),
            "mean_pitch": float(pitch[a:b, e].mean()),
        }
        if "height" in z:
            r["min_height"] = float(z["height"][a:b, e].min())
        if qd is not None:
            r["p95_joint_speed"] = float(np.percentile(qd[a:b, e], 95))
        if foot is not None:
            duty = foot[a:b, e].mean(axis=0)  # per foot fraction of time in contact
            for i, fname in enumerate(foot_names or range(duty.size)):
                r[f"duty_{fname}"] = float(duty[i])
            if foot_names:
                front = [i for i, f in enumerate(foot_names) if f.startswith("F")]
                rear = [i for i, f in enumerate(foot_names) if f.startswith("R")]
                r["front_duty_minus_rear"] = float(duty[front].mean() - duty[rear].mean())
        if thigh is not None:
            r["thigh_contact"] = float(thigh[a:b, e].any())
        r["env"] = e
        rows[name].append(r)

    out: dict[str, Any] = {"settle_s": th["settle_s"], "bins": {}}
    for name, lo, hi in BINS:
        rs = rows[name]
        if not rs:
            out["bins"][name] = {"n_segments": 0}
            continue
        keys = [k for k in rs[0] if k != "env"]
        agg = {k: float(np.mean([r[k] for r in rs])) for k in keys}
        agg["n_segments"] = len(rs)
        agg["n_episodes"] = len({r["env"] for r in rs})
        agg["range"] = [lo, hi]
        out["bins"][name] = agg
    return out


__all__ = ["BINS", "bin_of", "command_segments", "directional_report"]
