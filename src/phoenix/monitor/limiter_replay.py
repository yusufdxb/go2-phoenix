"""Offline limiter forensics: replay recorded policy requests through candidate limiters.

Phase A of the Phoenix v2 program (``docs/research/EXPERIMENT.md``, amendment 1). The
question is how much each soft limiter rewrites the policy it is supposed to protect,
measured on real GO2 telemetry, before any limiter runs in simulation or on the robot.

Three limiters, all followed by the SAME absolute hard envelope (URDF joint limits and
the abort band, :mod:`phoenix.sim2real.go2_model`), which is not a candidate and is
never weakened:

* ``measured_q`` (A1, the incumbent): the policy node clips its request to its own
  measured ``q +/- 0.175``, then the bridge clips again against a fresher ``q``.
  Re-derived from the recorded request and ``q`` and checked against the recorded
  ``sent`` target, so the replay is proven to reproduce what the robot was told.
* ``prev_command`` (A2, the candidate): ``|sent[k] - sent[k-1]| <= max_delta``. The
  anchor at the first policy tick is the target the bridge sent on the tick before
  (its stand-up or hold target), so a handover jump is rate-limited too.
* ``hard_only`` (A3, OFFLINE REFERENCE ONLY): no soft limiter, hard envelope only.
  Never a deploy option; it shows what the envelope alone would have allowed.

What this replay is NOT. It is open loop: the recorded ``q`` was produced by the
incumbent limiter, so a different limiter would have led to a different ``q`` and
different policy requests afterwards. The replay measures how each limiter treats the
same request stream, not what the robot would have done. Torque demand
``kp * (sent - q)`` uses the recorded ``q`` for the same reason and is an estimate of
the first-tick PD demand only. Every report says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from phoenix.sim2real.go2_model import (
    LIMIT_ABORT_BAND_RAD,
    TRAINING_DEFAULT_JOINT_POS,
    UNITREE_MOTOR_ORDER,
    limits_in_order,
)

from .layers import CommandLayers

SCHEMA = "phoenix-limiter-replay/v1"
CONTROL_HZ = 50.0
LIMITERS = ("measured_q", "prev_command", "hard_only")
#: A change smaller than this is below what the fidelity gate calls material.
TOL_RAD = 1e-3


@dataclass(frozen=True)
class ReplayInput:
    """Policy ticks of one run, Unitree motor order, shape ``(T, 12)`` unless noted."""

    t_s: np.ndarray  # (T,)
    requested: np.ndarray  # the policy's own request, default_q + scale * raw
    q_node: np.ndarray  # q the policy node clipped against (NaN if not logged)
    q_bridge: np.ndarray  # q the bridge clipped against
    sent_recorded: np.ndarray  # what the bridge actually sent
    anchor: np.ndarray  # (12,) target sent on the tick before the first policy tick
    kp: np.ndarray  # (T, 12) gains the targets were sent with
    raw_action: np.ndarray | None = None  # (T, 12) ONNX output, motor order


#: Isaac Lab's ``RslRlVecEnvWrapper(clip_actions=1.0)`` clamps every raw action to
#: [-1, 1] before the action term, in training and in evaluation. That clamp is part
#: of the plant the policy was trained on, not a safety layer.
TRAINED_ACTION_CLIP = 1.0


def with_trained_action_clip(inp: ReplayInput, action_scale: float = 0.25) -> ReplayInput:
    """The request the TRAINED plant would have executed: ``default + scale * clip(raw)``."""
    if inp.raw_action is None:
        raise ValueError("raw_action not available")
    default = np.asarray([TRAINING_DEFAULT_JOINT_POS[n] for n in UNITREE_MOTOR_ORDER])
    clipped = np.clip(inp.raw_action, -TRAINED_ACTION_CLIP, TRAINED_ACTION_CLIP)
    req = default + action_scale * clipped
    # The recorded request must equal default + scale * raw, or the layer map is wrong.
    recon = default + action_scale * inp.raw_action
    if not np.allclose(recon, inp.requested, atol=1e-5):
        raise ValueError("requested != default + scale * raw_action; wrong action map")
    return ReplayInput(
        t_s=inp.t_s,
        requested=req,
        q_node=inp.q_node,
        q_bridge=inp.q_bridge,
        sent_recorded=inp.sent_recorded,
        anchor=inp.anchor,
        kp=inp.kp,
        raw_action=inp.raw_action,
    )


def replay_input(layers: CommandLayers, q_node: np.ndarray | None = None) -> ReplayInput:
    """Select the new-command policy ticks of a run and the handover anchor."""
    idx = np.flatnonzero(layers.policy_mask & layers.cmd_is_new)
    if idx.size == 0:
        raise ValueError("run has no policy ticks")
    first = int(idx[0])
    anchor = layers.sent[first - 1] if first > 0 else layers.q[first]
    if not np.all(np.isfinite(anchor)):
        anchor = layers.q[first]
    qn = np.full((idx.size, 12), np.nan) if q_node is None else np.asarray(q_node)[idx]
    return ReplayInput(
        t_s=layers.t_s[idx],
        requested=layers.requested[idx],
        q_node=qn,
        q_bridge=layers.q[idx],
        sent_recorded=layers.sent[idx],
        anchor=np.asarray(anchor, dtype=np.float64),
        kp=layers.kp[idx],
        raw_action=layers.raw_action[idx],
    )


def apply_limiter(
    inp: ReplayInput,
    limiter: str,
    max_delta: float,
    lo: np.ndarray,
    hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(sent, soft_limited, hard_clipped)``: the target after the soft limiter
    and the hard envelope, a mask of samples the SOFT limiter changed, and a mask of
    samples the hard envelope then changed."""
    req = inp.requested
    if limiter == "measured_q":
        node = req
        if np.all(np.isfinite(inp.q_node)):
            node = np.clip(req, inp.q_node - max_delta, inp.q_node + max_delta)
        soft = np.clip(node, inp.q_bridge - max_delta, inp.q_bridge + max_delta)
    elif limiter == "prev_command":
        soft = np.empty_like(req)
        prev = inp.anchor.copy()
        for k in range(req.shape[0]):
            prev = np.clip(req[k], prev - max_delta, prev + max_delta)
            soft[k] = prev
    elif limiter == "hard_only":
        soft = req.copy()
    else:
        raise ValueError(f"unknown limiter {limiter!r}")
    sent = np.clip(soft, lo, hi)
    return sent, np.abs(soft - req) > 0.0, sent != soft


def _runs(mask: np.ndarray) -> list[int]:
    out, n = [], 0
    for m in mask:
        if m:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def limiter_metrics(
    inp: ReplayInput,
    limiter: str,
    max_delta: float,
) -> dict[str, Any]:
    """Execution-distortion metrics for one limiter on one run."""
    lo, hi = limits_in_order(UNITREE_MOTOR_ORDER)
    default = np.asarray([TRAINING_DEFAULT_JOINT_POS[n] for n in UNITREE_MOTOR_ORDER])
    sent, soft, hard = apply_limiter(inp, limiter, max_delta, lo, hi)
    req = inp.requested
    d = sent - req
    ad = np.abs(d)
    altered = ad > TOL_RAD
    # Rate of the target actually sent, rad/s, including the handover step.
    prev = np.vstack([inp.anchor[None, :], sent[:-1]])
    rate = (sent - prev) * CONTROL_HZ
    req_prev = np.vstack([inp.anchor[None, :], req[:-1]])
    d_req = req - req_prev
    d_sent = sent - prev
    reversal = (np.sign(d_req) * np.sign(d_sent) < 0) & (np.abs(d_req) > TOL_RAD)
    reversal &= np.abs(d_sent) > TOL_RAD
    # Abort band: a request further out than the band latches the bridge (unchanged by
    # the choice of soft limiter; the check runs on the policy's request).
    beyond = (req < lo - LIMIT_ABORT_BAND_RAD) | (req > hi + LIMIT_ABORT_BAND_RAD)
    abort_tick = int(np.flatnonzero(beyond.any(axis=1))[0]) if beyond.any() else None
    margin = np.minimum(sent - lo, hi - sent)
    tau = inp.kp * (sent - inp.q_bridge)
    req_norm = np.linalg.norm(req - default, axis=1)
    d_norm = np.linalg.norm(d, axis=1)
    distortion_d = float(d_norm.sum() / (req_norm.sum() + 1e-9))
    # Self-reinforcing lag (measured-q limiter only has a mechanism for it): inside a
    # run of consecutive soft clips on a joint, does the gap between the request and
    # the measured position grow? Measured on recorded q, so only meaningful for the
    # limiter that produced that q (A1).
    gap = np.abs(req - inp.q_bridge)
    growing, total_runs = 0, 0
    for j in range(12):
        k = 0
        while k < soft.shape[0]:
            if soft[k, j]:
                s = k
                while k < soft.shape[0] and soft[k, j]:
                    k += 1
                if k - s >= 3:
                    total_runs += 1
                    if gap[k - 1, j] > gap[s, j] + TOL_RAD:
                        growing += 1
            else:
                k += 1
    per_joint = {}
    for j, name in enumerate(UNITREE_MOTOR_ORDER):
        runs = _runs(soft[:, j])
        per_joint[name] = {
            "altered_fraction": float(altered[:, j].mean()),
            "soft_limited_fraction": float(soft[:, j].mean()),
            "rms_modification_rad": float(np.sqrt(np.mean(d[:, j] ** 2))),
            "max_modification_rad": float(ad[:, j].max()),
            "cumulative_modification_rad": float(ad[:, j].sum()),
            "limiter_reversals": int(reversal[:, j].sum()),
            "min_distance_to_hard_limit_rad": float(margin[:, j].min()),
            "max_abs_target_rate_rad_s": float(np.abs(rate[:, j]).max()),
            "p99_abs_target_rate_rad_s": float(np.percentile(np.abs(rate[:, j]), 99)),
            "max_abs_pd_torque_demand_nm": float(np.nanmax(np.abs(tau[:, j]))),
            "longest_soft_limit_run_ticks": int(max(runs) if runs else 0),
        }
    return {
        "limiter": limiter,
        "max_delta_rad": None if limiter == "hard_only" else float(max_delta),
        "ticks": int(req.shape[0]),
        "duration_s": float(inp.t_s[-1] - inp.t_s[0]) if req.shape[0] > 1 else 0.0,
        "altered_fraction": float(altered.mean()),
        "soft_limited_fraction": float(soft.mean()),
        "hard_envelope_clip_fraction": float(hard.mean()),
        "ticks_with_any_altered_fraction": float(altered.any(axis=1).mean()),
        "rms_modification_rad": float(np.sqrt(np.mean(d**2))),
        "max_modification_rad": float(ad.max()),
        "cumulative_modification_rad": float(ad.sum()),
        "distortion_D": distortion_d,
        "limiter_reversals": int(reversal.sum()),
        "min_distance_to_hard_limit_rad": float(margin.min()),
        "hard_limit_violations": int(((sent < lo - 1e-12) | (sent > hi + 1e-12)).sum()),
        "abort_band_first_tick": abort_tick,
        "max_abs_target_rate_rad_s": float(np.abs(rate).max()),
        "max_abs_pd_torque_demand_nm": float(np.nanmax(np.abs(tau))),
        "soft_clip_runs_ge3": total_runs,
        "soft_clip_runs_with_growing_gap": growing,
        "affected_joints": [n for n, v in per_joint.items() if v["altered_fraction"] > 0.05],
        "per_joint": per_joint,
        "sent": sent,
    }


def verify_incumbent_replay(inp: ReplayInput, max_delta: float) -> dict[str, Any]:
    """Prove the measured-q replay reproduces what the bridge actually sent."""
    lo, hi = limits_in_order(UNITREE_MOTOR_ORDER)
    sent, _, _ = apply_limiter(inp, "measured_q", max_delta, lo, hi)
    ok = np.isfinite(inp.sent_recorded)
    err = np.abs(sent - inp.sent_recorded)[ok]
    return {
        "max_abs_error_rad": float(err.max()) if err.size else None,
        "samples": int(ok.sum()),
        "reproduces_recorded": bool(err.size and err.max() < 1e-5),
        "node_q_available": bool(np.all(np.isfinite(inp.q_node))),
    }


__all__ = [
    "CONTROL_HZ",
    "LIMITERS",
    "SCHEMA",
    "ReplayInput",
    "apply_limiter",
    "limiter_metrics",
    "replay_input",
    "verify_incumbent_replay",
    "with_trained_action_clip",
    "TRAINED_ACTION_CLIP",
]
