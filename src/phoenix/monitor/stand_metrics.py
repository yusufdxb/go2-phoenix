"""Physical scoring of simulated stand rollouts (Phoenix v2 amendment 1).

Pure numpy, shared by the Isaac Lab action-term harness and the simulated
deployment-path harness so both score an episode the same way. Inputs are per-step
arrays shaped ``(T, N, ...)`` (steps, episodes); joint arrays are in POLICY order.

Layer names: ``raw`` actor output, ``req`` layer 2 (the trained plant's request),
``sent`` the target actually applied, ``q0`` measured position at step start,
``q1`` after the step. Attitude comes from projected gravity, never a quaternion.
"""

from __future__ import annotations

from typing import Any

import numpy as np

ROLL_PITCH_LIMIT_RAD = 0.40
FIDELITY_TOL_RAD = 1e-3
FIDELITY_MAX_ALTERED = 0.05
FIDELITY_MAX_RMS_RAD = 0.01


def attitude_from_gravity(grav: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(roll, pitch, tilt)`` from body-frame projected gravity ``(..., 3)``."""
    roll = np.arctan2(-grav[..., 1], -grav[..., 2])
    pitch = np.arctan2(grav[..., 0], -grav[..., 2])
    tilt = np.arccos(np.clip(-grav[..., 2], -1.0, 1.0))
    return roll, pitch, tilt


def score_stand_rollout(
    *,
    raw: np.ndarray,
    req: np.ndarray,
    sent: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
    tau_c: np.ndarray,
    tau_a: np.ndarray,
    grav: np.ndarray,
    height: np.ndarray,
    linv: np.ndarray,
    angv: np.ndarray,
    cmd: np.ndarray,
    alive: np.ndarray,
    contact_term: np.ndarray,
    ended: np.ndarray,
    default: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    dt: float,
    abort_band: float,
    joint_names: list[str],
    safety_hold: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return ``(aggregate metrics, per-episode records)``.

    ``safety_hold`` (``(T, N)`` bool, optional) marks steps on which a deploy-path
    safety layer held or aborted; such an episode cannot be a success.
    """
    T, N = alive.shape
    LIMIT_ABORT_BAND_RAD = abort_band
    roll, pitch, tilt = attitude_from_gravity(grav)
    att_ok = (np.abs(roll) <= ROLL_PITCH_LIMIT_RAD) & (np.abs(pitch) <= ROLL_PITCH_LIMIT_RAD)
    # An episode's steps after its first termination are not part of it.
    first_end = np.where((~alive).any(axis=0), np.argmax(~alive, axis=0), T)
    valid = np.arange(T)[:, None] < np.minimum(first_end, T)[None, :]
    # alive[k] is False on the step AFTER an env ended, so the step that ended it counts.
    step_ok = att_ok & valid
    d = sent - req
    altered = (np.abs(d) > FIDELITY_TOL_RAD) & valid[..., None]
    beyond = ((req < lo - LIMIT_ABORT_BAND_RAD) | (req > hi + LIMIT_ABORT_BAND_RAD)) & valid[..., None]
    at_limit = ((sent <= lo + 1e-6) | (sent >= hi - 1e-6)) & valid[..., None]
    sat = (np.abs(tau_c - tau_a) > 1e-3) & valid[..., None]
    rate = np.abs(np.diff(sent, axis=0, prepend=sent[:1])) / dt
    raw_oob = (np.abs(raw) > 1.0) & valid[..., None]

    episodes = []
    for e in range(N):
        v = valid[:, e]
        nv = int(v.sum())
        de = d[v, e]
        alt_j = altered[v, e].mean(axis=0) if nv else np.ones(12)
        rms = float(np.sqrt(np.mean(de**2))) if nv else float("nan")
        fid_ok = bool(nv and altered[v, e].mean() <= FIDELITY_MAX_ALTERED
                      and alt_j.max() <= FIDELITY_MAX_ALTERED and rms <= FIDELITY_MAX_RMS_RAD)
        score = float(step_ok[:, e].sum() / T)  # continuous primary endpoint
        survived = not contact_term[e]
        att_all = bool(att_ok[v, e].all()) if nv else False
        held = bool(safety_hold[v, e].any()) if safety_hold is not None else False
        success = bool(
            survived and att_all and not beyond[:, e].any() and fid_ok and nv == T and not held
        )
        episodes.append({
            "env": e,
            "steps": nv,
            "trunk_contact": bool(contact_term[e]),
            "primary_score": score,
            "max_abs_roll_rad": float(np.abs(roll[v, e]).max()) if nv else None,
            "max_abs_pitch_rad": float(np.abs(pitch[v, e]).max()) if nv else None,
            "max_tilt_rad": float(tilt[v, e].max()) if nv else None,
            "attitude_violation_steps": int((~att_ok[v, e]).sum()),
            "abort_band_requests": int(beyond[:, e].any(axis=1).sum()),
            "altered_fraction": float(altered[v, e].mean()) if nv else 1.0,
            "worst_joint_altered_fraction": float(alt_j.max()),
            "rms_distortion_rad": rms,
            "fidelity_pass": fid_ok,
            "safety_hold": held,
            "success": success,
        })
    jn = list(joint_names)
    per_joint = {
        jn[j]: {
            "altered_fraction": float(altered[..., j].sum() / valid.sum()),
            "rms_modification_rad": float(np.sqrt((d[..., j][valid] ** 2).mean())),
            "max_modification_rad": float(np.abs(d[..., j][valid]).max()),
            "raw_out_of_range_fraction": float(raw_oob[..., j].sum() / valid.sum()),
            "p99_abs_target_rate_rad_s": float(np.percentile(rate[..., j][valid], 99)),
            "max_abs_target_rate_rad_s": float(rate[..., j][valid].max()),
            "p99_abs_applied_torque_nm": float(np.percentile(np.abs(tau_a[..., j][valid]), 99)),
            "max_abs_computed_torque_nm": float(np.abs(tau_c[..., j][valid]).max()),
            "effort_saturation_fraction": float(sat[..., j].sum() / valid.sum()),
            "at_hard_limit_fraction": float(at_limit[..., j].sum() / valid.sum()),
            "min_distance_to_limit_rad": float(np.minimum(sent[..., j] - lo[j], hi[j] - sent[..., j])[valid].min()),
            "p99_abs_tracking_gap_rad": float(np.percentile(np.abs(req[..., j] - q0[..., j])[valid], 99)),
            "mean_abs_posture_offset_rad": float(np.abs(q1[..., j] - default[j])[valid].mean()),
        }
        for j in range(12)
    }
    succ = np.array([ep["success"] for ep in episodes])
    scores = np.array([ep["primary_score"] for ep in episodes])
    fid = np.array([ep["fidelity_pass"] for ep in episodes])
    lin_err = np.linalg.norm((linv[..., :2] - cmd[..., :2]), axis=-1)[valid]
    yaw_err = np.abs(angv[..., 2] - cmd[..., 2])[valid]
    succ = np.array([ep["success"] for ep in episodes])
    scores = np.array([ep["primary_score"] for ep in episodes])
    fid = np.array([ep["fidelity_pass"] for ep in episodes])
    lin_err = np.linalg.norm((linv[..., :2] - cmd[..., :2]), axis=-1)[valid]
    yaw_err = np.abs(angv[..., 2] - cmd[..., 2])[valid]
    metrics = {
        "thresholds": {
            "roll_pitch_limit_rad": ROLL_PITCH_LIMIT_RAD,
            "fidelity_tol_rad": FIDELITY_TOL_RAD,
            "fidelity_max_altered": FIDELITY_MAX_ALTERED,
            "fidelity_max_rms_rad": FIDELITY_MAX_RMS_RAD,
            "abort_band_rad": abort_band,
        },
        "success_rate": float(succ.mean()),
        "survival_rate": float(1.0 - contact_term.mean()),
        "timeout_rate_legacy": float((~ended | (ended & ~contact_term)).mean()),
        "mean_primary_score": float(scores.mean()),
        "fidelity_pass_rate": float(fid.mean()),
        "safety_hold_episode_rate": float(np.mean([ep["safety_hold"] for ep in episodes])),
        "attitude_violation_episode_rate": float(
            np.mean([ep["attitude_violation_steps"] > 0 for ep in episodes])
        ),
        "abort_band_episode_rate": float(np.mean([ep["abort_band_requests"] > 0 for ep in episodes])),
        "altered_fraction": float(altered.sum() / (valid.sum() * 12)),
        "rms_modification_rad": float(np.sqrt((d[valid] ** 2).mean())),
        "distortion_D": float(
            np.linalg.norm(d, axis=-1)[valid].sum()
            / (np.linalg.norm(req - default, axis=-1)[valid].sum() + 1e-9)
        ),
        "raw_out_of_range_fraction": float(raw_oob.sum() / (valid.sum() * 12)),
        "effort_saturation_fraction": float(sat.sum() / (valid.sum() * 12)),
        "at_hard_limit_fraction": float(at_limit.sum() / (valid.sum() * 12)),
        "max_abs_roll_rad": float(np.abs(roll[valid]).max()),
        "max_abs_pitch_rad": float(np.abs(pitch[valid]).max()),
        "p99_tilt_rad": float(np.percentile(tilt[valid], 99)),
        "mean_base_height_m": float(height[valid].mean()),
        "mean_lin_vel_error_m_s": float(lin_err.mean()),
        "mean_yaw_rate_error_rad_s": float(yaw_err.mean()),
        "per_joint": per_joint,
    }
    return metrics, episodes


WALK_MAX_LIN_ERR_M_S = 0.25
WALK_MAX_YAW_ERR_RAD_S = 0.30
WALK_SETTLE_S = 1.0


def score_walk_episodes(
    episodes: list[dict[str, Any]],
    *,
    linv: np.ndarray,
    angv: np.ndarray,
    cmd: np.ndarray,
    valid: np.ndarray,
    dt: float,
) -> dict[str, Any]:
    """Add Amendment 3 walking success to stand-scored episodes (in place) and summarise.

    Tracking errors exclude the first ``WALK_SETTLE_S`` after episode start and after
    every command change. Walking success = stand success (no trunk contact, attitude,
    abort band, fidelity) AND mean planar error <= 0.25 m/s AND mean yaw-rate error
    <= 0.30 rad/s.
    """
    T, N = valid.shape
    settle = int(round(WALK_SETTLE_S / dt))
    changed = np.zeros((T, N), bool)
    changed[0] = True
    changed[1:] = np.any(np.abs(np.diff(cmd, axis=0)) > 1e-6, axis=-1)
    since = np.zeros((T, N), int)
    for k in range(T):
        since[k] = np.where(changed[k], 0, since[k - 1] + 1 if k else 0)
    use = valid & (since >= settle)
    lin_err = np.linalg.norm(linv[..., :2] - cmd[..., :2], axis=-1)
    yaw_err = np.abs(angv[..., 2] - cmd[..., 2])
    speed = np.linalg.norm(cmd[..., :2], axis=-1)
    for e, ep in enumerate(episodes):
        u = use[:, e]
        le = float(lin_err[u, e].mean()) if u.any() else float("inf")
        ye = float(yaw_err[u, e].mean()) if u.any() else float("inf")
        ep["mean_lin_vel_error_m_s"] = le
        ep["mean_yaw_rate_error_rad_s"] = ye
        ep["mean_cmd_speed_m_s"] = float(speed[u, e].mean()) if u.any() else 0.0
        ep["walk_success"] = bool(
            ep["success"] and le <= WALK_MAX_LIN_ERR_M_S and ye <= WALK_MAX_YAW_ERR_RAD_S
        )
    return {
        "walk_thresholds": {
            "max_lin_err_m_s": WALK_MAX_LIN_ERR_M_S,
            "max_yaw_err_rad_s": WALK_MAX_YAW_ERR_RAD_S,
            "settle_s": WALK_SETTLE_S,
        },
        "walk_success_rate": float(np.mean([ep["walk_success"] for ep in episodes])),
        "walk_mean_lin_vel_error_m_s": float(lin_err[use].mean()),
        "walk_mean_yaw_rate_error_rad_s": float(yaw_err[use].mean()),
        "walk_mean_cmd_speed_m_s": float(speed[use].mean()),
    }


__all__ = [
    "score_walk_episodes",
    "FIDELITY_MAX_ALTERED",
    "FIDELITY_MAX_RMS_RAD",
    "FIDELITY_TOL_RAD",
    "ROLL_PITCH_LIMIT_RAD",
    "attitude_from_gravity",
    "score_stand_rollout",
]
