"""One crash-safe telemetry file per bridge run, and the hardware slew metric.

Format: JSON Lines. Line 1 is ``{"record": "manifest", ...}`` (code identity,
config and artifact hashes, gate parameters, joint orders and limits). Every
following line is ``{"record": "tick", ...}``, exactly the dict
:meth:`phoenix.sim2real.actuator_gate.ActuatorGate.tick` returned plus the
wall-clock time. An optional last line is ``{"record": "end", ...}``.

JSON Lines rather than Parquet on purpose: a Parquet file with no footer is
unreadable, and this project has already lost three hardware captures that way
(policy node shutdown ordering, 2026-04-18). Every line here is flushed as it is
written, so a killed process loses at most the line in flight.

Hardware slew metric
--------------------
:data:`HARDWARE_SLEW_METRIC` is defined on the FINAL bridge's clip, the one that
actually shapes the motor command:

    over tick rows with ``mode == "policy"`` and ``cmd_is_new``, the percentage of
    (row, joint) samples whose ``slew_clip`` is true, i.e. whose requested target
    was altered by ``per_step_clip_array(requested, measured_q, 0.175)``.

``cmd_is_new`` restricts it to the first tick that processed each policy command,
so a command the bridge re-applied on a second tick (its 50 Hz timer is not
synchronised with the policy's) is counted once. The all-policy-ticks variant is
reported next to it. ``policy_node_slew_clip_pct`` recomputes the policy node's
own clip from the wire (``requested_target`` against ``q_policy``), which is the
quantity the corrected sim metric ``slew_clip_activation_rate`` measures, so the
two layers can be compared on the same run.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .go2_model import UNITREE_MOTOR_ORDER
from .safety import MAX_DELTA_PER_STEP_RAD, per_step_clip_array

TELEMETRY_SCHEMA = "phoenix-bridge-telemetry/v1"
HARDWARE_SLEW_METRIC = "bridge_final_slew_clip_activation_v1"
HARDWARE_SLEW_METRIC_DEFINITION = (
    "100 * mean over tick rows with mode=='policy' and cmd_is_new, and over the 12 joints, "
    "of slew_clip (the final LowCmd bridge's per_step_clip_array altered the requested target "
    "against the fresh measured joint position, cap 0.175 rad)"
)


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_sanitize(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class TelemetryWriter:
    """Append-only JSONL writer. Refuses to overwrite an existing file."""

    def __init__(self, path: str | Path, manifest: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("x", buffering=1)
        self._write(
            {
                "record": "manifest",
                "schema": TELEMETRY_SCHEMA,
                "written_utc": utc_now_iso(),
                **manifest,
            }
        )

    def _write(self, obj: Mapping[str, Any]) -> None:
        self._fh.write(json.dumps(_sanitize(obj), allow_nan=False, separators=(",", ":")) + "\n")
        self._fh.flush()

    def write_tick(self, record: Mapping[str, Any]) -> None:
        self._write({"record": "tick", "utc": utc_now_iso(), **record})

    def close(self, end: Mapping[str, Any] | None = None) -> None:
        if self._fh.closed:
            return
        try:
            self._write({"record": "end", "utc": utc_now_iso(), **(end or {})})
        finally:
            self._fh.close()


def read_telemetry(
    path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    """Return ``(manifest, ticks, end)``. A truncated last line is ignored, loudly counted."""
    manifest: dict[str, Any] | None = None
    ticks: list[dict[str, Any]] = []
    end: dict[str, Any] | None = None
    bad = 0
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            kind = obj.get("record")
            if kind == "manifest":
                manifest = obj
            elif kind == "tick":
                ticks.append(obj)
            elif kind == "end":
                end = obj
    if manifest is None:
        raise ValueError(f"{path}: no manifest record")
    if manifest.get("schema") != TELEMETRY_SCHEMA:
        raise ValueError(f"{path}: schema {manifest.get('schema')!r} != {TELEMETRY_SCHEMA!r}")
    manifest["_unparseable_lines"] = bad
    return manifest, ticks, end


def _pct(rows: list[dict[str, Any]], key: str) -> float | None:
    samples = [bool(v) for r in rows for v in (r.get(key) or [])]
    return 100.0 * sum(samples) / len(samples) if samples else None


def _policy_node_clip_pct(rows: Iterable[dict[str, Any]], max_delta: float) -> float | None:
    hits = 0
    total = 0
    for r in rows:
        pol = r.get("policy") or {}
        req = pol.get("requested_target")
        qp = pol.get("q_policy")
        if not req or not qp or any(v is None for v in (*req, *qp)):
            continue
        req_arr = np.asarray(req, dtype=np.float64)
        clipped = per_step_clip_array(req_arr, np.asarray(qp, dtype=np.float64), max_delta)
        hits += int(np.sum(clipped != req_arr))
        total += req_arr.size
    return 100.0 * hits / total if total else None


def summarize(manifest: Mapping[str, Any], ticks: list[dict[str, Any]]) -> dict[str, Any]:
    """Everything a stage verdict needs, recomputed from the rows alone."""
    max_delta = float((manifest.get("gate_params") or {}).get("max_delta", MAX_DELTA_PER_STEP_RAD))
    modes = Counter(t.get("mode") for t in ticks)
    policy_rows = [t for t in ticks if t.get("mode") == "policy"]
    new_rows = [t for t in policy_rows if t.get("cmd_is_new")]

    per_joint: dict[str, float | None] = {}
    for j, name in enumerate(UNITREE_MOTOR_ORDER):
        vals = [bool((t.get("slew_clip") or [False] * 12)[j]) for t in new_rows]
        per_joint[name] = 100.0 * sum(vals) / len(vals) if vals else None

    limit_clip_by_joint: Counter[str] = Counter()
    for t in ticks:
        for j, hit in enumerate(t.get("limit_clip") or []):
            if hit:
                limit_clip_by_joint[UNITREE_MOTOR_ORDER[j]] += 1
    margins = [m for t in ticks for m in (t.get("limit_margin") or []) if m is not None]

    faults: list[str] = []
    for t in ticks:
        for f in t.get("faults") or []:
            if f not in faults:
                faults.append(f)

    abort_reasons = sorted(
        {
            str((t.get("policy") or {}).get("abort_reason"))
            for t in ticks
            if (t.get("policy") or {}).get("kind") == 3
            and (t.get("policy") or {}).get("abort_reason") is not None
        }
    )

    t_ns = [int(t["t_mono_ns"]) for t in ticks if "t_mono_ns" in t]
    gaps = np.diff(np.asarray(t_ns, dtype=np.int64)) / 1e9 if len(t_ns) > 1 else np.asarray([])
    policy_t = [int(t["t_mono_ns"]) for t in policy_rows]

    def _max(key: str, rows: list[dict[str, Any]]) -> float | None:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return max(vals) if vals else None

    def _max_abs_policy(field: str) -> float | None:
        vals: list[float] = []
        for r in policy_rows:
            value = (r.get("policy") or {}).get(field)
            if value is not None:
                vals.append(abs(float(value)))
        return max(vals) if vals else None

    first_q = next((t["q_unitree"] for t in ticks if t.get("q_unitree")), None)
    q_excursion = None
    if first_q is not None:
        q0 = np.asarray(first_q, dtype=np.float64)
        qs = [np.asarray(t["q_unitree"], dtype=np.float64) for t in ticks if t.get("q_unitree")]
        q_excursion = {
            name: float(v)
            for name, v in zip(
                UNITREE_MOTOR_ORDER, np.max(np.abs(np.stack(qs) - q0), axis=0), strict=True
            )
        }

    return {
        "metric": HARDWARE_SLEW_METRIC,
        "metric_definition": HARDWARE_SLEW_METRIC_DEFINITION,
        "n_ticks": len(ticks),
        "mode_counts": dict(modes),
        "policy_ticks": len(policy_rows),
        "policy_new_command_ticks": len(new_rows),
        "bridge_slew_clip_pct": _pct(new_rows, "slew_clip"),
        "bridge_slew_clip_pct_all_policy_ticks": _pct(policy_rows, "slew_clip"),
        "bridge_slew_clip_pct_per_joint": per_joint,
        "policy_node_slew_clip_pct": _policy_node_clip_pct(new_rows, max_delta),
        "limit_clip_counts": dict(limit_clip_by_joint),
        "min_limit_margin_rad": min(margins) if margins else None,
        "faults": faults,
        "first_fault": faults[0] if faults else None,
        "policy_abort_reasons": abort_reasons,
        "max_tick_gap_s": float(gaps.max()) if gaps.size else None,
        "max_lowstate_age_s": _max("lowstate_age_s", ticks),
        "max_cmd_age_s_in_policy": _max("cmd_age_s", policy_rows),
        "max_estop_age_s": _max("estop_age_s", ticks),
        "policy_window_s": (policy_t[-1] - policy_t[0]) / 1e9 if len(policy_t) > 1 else 0.0,
        "max_abs_roll_rad": _max_abs_policy("roll_rad"),
        "max_abs_pitch_rad": _max_abs_policy("pitch_rad"),
        "q_max_excursion_rad": q_excursion,
        "live": bool(manifest.get("live")),
    }


__all__ = [
    "HARDWARE_SLEW_METRIC",
    "HARDWARE_SLEW_METRIC_DEFINITION",
    "TELEMETRY_SCHEMA",
    "TelemetryWriter",
    "read_telemetry",
    "summarize",
    "utc_now_iso",
]
