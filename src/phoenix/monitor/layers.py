"""The four layers of every deployed joint command, as arrays.

Phoenix must never blame the hardware for a change its own safety stack made, so
every analysis starts from the same four layers, per bridge tick and per joint, in
Unitree motor order:

1. ``raw_action``      the ONNX policy output (unitless, policy scale).
2. ``requested``       ``default_q + action_scale * raw_action``: what the policy
                       asked the joint to do, in radians (wire ``requested_target``).
   ``policy_node_target``  the policy node's output (wire ``target``): after its
                       slew clip in v1 files, equal to ``requested`` when the node
                       applies no soft limiter (v2). The bridge logs it as
                       ``node_target_unitree`` (v2) or, in v1 files, under the
                       misleading name ``requested_target_unitree``.
3. ``sent``            the target the bridge actually wrote to ``LowCmd`` after the
                       policy node's slew clip, the bridge's own slew clip and the
                       hard-limit clip (``final_target_unitree``), together with the
                       ``kp``/``kd`` it was sent with.
4. ``q`` / ``dq`` / ``tau_est``  the measured response from ``LowState``.

The source is the bridge telemetry JSONL (``phoenix.sim2real.bridge_telemetry``),
which already records layers 2 to 4 per tick and the policy node's wire record
(layer 1, plus the policy-node target) nested under ``policy``. ``tau_est`` and
per-motor gains were added to the bridge record with this module; older files
simply lack them and the arrays come back NaN.

Timing: the bridge ticks at 50 Hz and records the freshest LowState it has at the
start of the tick, then publishes that tick's target. The response to the target
sent at tick ``k`` is therefore first visible in the ``q`` recorded at tick
``k + 1``; :func:`tracking_pairs` pairs them that way.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phoenix.sim2real.motor_crc import PHOENIX_FOR_MOTOR

N_JOINTS = 12
_PERM = np.asarray(PHOENIX_FOR_MOTOR, dtype=np.int64)


def _arr(value: Any, width: int = N_JOINTS) -> np.ndarray:
    if value is None:
        return np.full(width, np.nan)
    if np.isscalar(value):
        return np.full(width, float(value))  # type: ignore[arg-type]
    out = np.asarray([np.nan if v is None else float(v) for v in value], dtype=np.float64)
    if out.shape != (width,):
        raise ValueError(f"expected {width} values, got shape {out.shape}")
    return out


def _bools(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(N_JOINTS, dtype=bool)
    out = np.asarray([bool(v) for v in value], dtype=bool)
    if out.shape != (N_JOINTS,):
        raise ValueError(f"expected {N_JOINTS} flags, got shape {out.shape}")
    return out


def _policy_to_motor(value: Any) -> np.ndarray:
    """A policy-order wire array to Unitree motor order (NaN if absent)."""
    arr = _arr(value)
    return arr[_PERM]


@dataclass(frozen=True)
class CommandLayers:
    """Per-tick arrays, shape ``(T, 12)`` unless noted, Unitree motor order."""

    t_s: np.ndarray  # (T,) monotonic seconds
    mode: np.ndarray  # (T,) str
    cmd_is_new: np.ndarray  # (T,) bool
    raw_action: np.ndarray
    requested: np.ndarray
    policy_node_target: np.ndarray
    sent: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    tau_est: np.ndarray
    bridge_slew_clip: np.ndarray  # bool
    limit_clip: np.ndarray  # bool
    degradation_kp_scale: np.ndarray  # 1.0 where no controlled degradation was applied

    def __len__(self) -> int:
        return int(self.t_s.shape[0])

    @property
    def policy_mask(self) -> np.ndarray:
        mask: np.ndarray = self.mode == "policy"
        return mask


def from_records(records: Iterable[Mapping[str, Any]]) -> CommandLayers:
    """Build :class:`CommandLayers` from bridge telemetry tick records.

    Non-tick records (manifest, end) are skipped. Ticks that published nothing
    (``silent``) are kept with NaN targets so indices stay aligned with time.
    """
    cols: dict[str, list[Any]] = {
        k: []
        for k in (
            "t_s",
            "mode",
            "cmd_is_new",
            "raw_action",
            "requested",
            "policy_node_target",
            "sent",
            "kp",
            "kd",
            "q",
            "dq",
            "tau_est",
            "bridge_slew_clip",
            "limit_clip",
            "degradation_kp_scale",
        )
    }
    for rec in records:
        if rec.get("record", "tick") != "tick":
            continue
        pol = rec.get("policy") or {}
        mode = str(rec.get("mode"))
        cols["t_s"].append(float(rec["t_mono_ns"]) / 1e9)
        cols["mode"].append(mode)
        cols["cmd_is_new"].append(bool(rec.get("cmd_is_new", False)))
        is_policy = mode == "policy"
        # The policy-side layers only describe what was actuated on a policy tick.
        cols["raw_action"].append(
            _policy_to_motor(pol.get("raw_action")) if is_policy else _arr(None)
        )
        # NOT the bridge's ``requested_target_unitree``: despite its name that field
        # is the policy node's target AFTER the node's slew clip (``cmd.target``),
        # so comparing against it hides the policy-node layer entirely. The policy's
        # own request is the wire field ``requested_target``.
        cols["requested"].append(
            _policy_to_motor(pol.get("requested_target")) if is_policy else _arr(None)
        )
        node_target = _policy_to_motor(pol.get("target")) if is_policy else _arr(None)
        # Schema v2 names it ``node_target_unitree``; v1 files carry the same value
        # under the misleading ``requested_target_unitree``.
        bridge_in = (
            rec.get("node_target_unitree", rec.get("requested_target_unitree"))
            if is_policy
            else None
        )
        if bridge_in is not None and np.all(np.isfinite(node_target)):
            if not np.allclose(_arr(bridge_in), node_target, atol=1e-9, rtol=0.0):
                raise ValueError(
                    f"tick {rec.get('tick')}: bridge input differs from the policy-node "
                    "target; the telemetry does not describe one command path"
                )
        cols["policy_node_target"].append(node_target)
        cols["sent"].append(_arr(rec.get("final_target_unitree") if rec.get("publish") else None))
        kp = rec.get("kp_unitree", rec.get("kp"))
        kd = rec.get("kd_unitree", rec.get("kd"))
        cols["kp"].append(_arr(kp if rec.get("publish") else None))
        cols["kd"].append(_arr(kd if rec.get("publish") else None))
        cols["q"].append(_arr(rec.get("q_unitree")))
        cols["dq"].append(_arr(rec.get("dq_unitree")))
        cols["tau_est"].append(_arr(rec.get("tau_est_unitree")))
        cols["bridge_slew_clip"].append(_bools(rec.get("slew_clip")))
        cols["limit_clip"].append(_bools(rec.get("limit_clip")))
        deg = rec.get("degradation") or {}
        cols["degradation_kp_scale"].append(_arr(deg.get("kp_scale_unitree", 1.0)))
    if not cols["t_s"]:
        raise ValueError("no tick records")
    return CommandLayers(
        t_s=np.asarray(cols["t_s"]),
        mode=np.asarray(cols["mode"]),
        cmd_is_new=np.asarray(cols["cmd_is_new"], dtype=bool),
        **{k: np.vstack(cols[k]) for k in cols if k not in ("t_s", "mode", "cmd_is_new")},
    )


def read_bridge_telemetry(path: str | Path) -> CommandLayers:
    """Read a bridge ``bridge.jsonl``. A truncated last line (killed process) is skipped."""
    records = []
    with Path(path).open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return from_records(records)


@dataclass(frozen=True)
class TrackingPairs:
    """Target sent at tick k paired with the response measured at tick k+1."""

    t_s: np.ndarray  # (N,)
    sent: np.ndarray  # (N, 12)
    q_next: np.ndarray  # (N, 12)
    dq_next: np.ndarray  # (N, 12)
    tau_next: np.ndarray  # (N, 12)
    kp: np.ndarray  # (N, 12)
    kd: np.ndarray  # (N, 12)
    #: True where the target at tick k was NOT altered by any safety layer relative
    #: to the policy request, and both ticks were policy ticks. Only these samples say
    #: anything about the actuator; a clipped target pins the error to the clip.
    valid: np.ndarray  # (N, 12) bool
    safety_altered: np.ndarray  # (N, 12) bool

    @property
    def error(self) -> np.ndarray:
        """``sent[k] - q[k+1]``, radians."""
        err: np.ndarray = self.sent - self.q_next
        return err


def tracking_pairs(layers: CommandLayers, alter_tol_rad: float = 1e-6) -> TrackingPairs:
    """Pair each policy tick's sent target with the next tick's measured response."""
    if len(layers) < 2:
        raise ValueError("need at least two ticks")
    k = np.arange(len(layers) - 1)
    both_policy = layers.policy_mask[k] & layers.policy_mask[k + 1]
    altered = np.abs(layers.sent[k] - layers.requested[k]) > alter_tol_rad
    altered |= layers.bridge_slew_clip[k] | layers.limit_clip[k]
    finite = (
        np.isfinite(layers.sent[k])
        & np.isfinite(layers.requested[k])
        & np.isfinite(layers.q[k + 1])
    )
    valid = both_policy[:, None] & finite & ~altered
    return TrackingPairs(
        t_s=layers.t_s[k],
        sent=layers.sent[k],
        q_next=layers.q[k + 1],
        dq_next=layers.dq[k + 1],
        tau_next=layers.tau_est[k + 1],
        kp=layers.kp[k],
        kd=layers.kd[k],
        valid=valid,
        safety_altered=altered & both_policy[:, None],
    )


__all__ = [
    "N_JOINTS",
    "CommandLayers",
    "TrackingPairs",
    "from_records",
    "read_bridge_telemetry",
    "tracking_pairs",
]
