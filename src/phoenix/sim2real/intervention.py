"""Safety-layer transparency: what the policy asked for vs what the motors were told.

The 2026-09-21 F run looks, from the bridge's own slew counter, like a policy whose
targets were clipped on 28 % of joint-ticks. End to end it was worse: on 88.8 % of
joint-ticks the target the motors received differed from the target the policy
asked for (``default + 0.25 * action``), by up to 2.23 rad and 0.80 rad RMS, over the
0.56 s before an illegal rear-thigh target latched a fault. Two slew clips in two
processes (policy node, then bridge) each looked moderate; together they meant
the robot was not executing the policy at all. The run was still reported as a
stand attempt of "the policy".

This module makes that impossible to miss:

* :class:`SafetyFilter` is the single-process safety layer used by the controller
  FSM: slew cap against measured ``q``, hard-limit clip, illegal-target abort, and
  a reason code for every joint it touches.
* :func:`tick_record` / :func:`records_from_bridge_ticks` produce one record per
  tick with the raw action, the scaled target, the safety-modified target, the
  final transmitted target, per-joint reason codes and magnitudes, from either the
  FSM or the existing bridge telemetry (``bridge.jsonl``).
* :func:`summarize` computes the metrics and :func:`deploy_gate` turns them into a
  PASS / FAIL that a stage evaluation must include.

Deploy gate thresholds (:class:`InterventionThresholds`) and why
---------------------------------------------------------------
A stage that claims "the policy did X" must show the policy's commands reached the
motors. Training applies ``default + scale * action`` with no clip, so any
modification is a departure from the trained closed loop.

* ``max_joint_tick_fraction = 0.02``: at most 2 % of (tick, joint) samples modified.
  A clip on one joint about every fourth tick is already visible behaviour; 2 % is
  the level at which the run can still be attributed to the policy. F 2026-09-21:
  88.8 %, 44x over.
* ``max_tick_fraction = 0.10``: at most 10 % of ticks with any joint modified.
  F: 100 %.
* ``max_rms_rad = 0.02``: RMS modification over all joint-ticks. 0.02 rad is 0.08
  action units, about the size of observation noise on ``joint_pos``. F: 0.80 rad.
* ``max_abs_rad = 0.10``: no single modification larger than 0.10 rad (0.4 action
  units, below one slew cap of 0.175). F: 2.23 rad.
* ``max_illegal_target_aborts = 0``: any target beyond the hard-limit abort band
  fails the stage. F: 1 (``target_beyond_limit:RR_thigh_joint``).
* ``min_policy_ticks = 50``: a verdict over fewer than 1 s of authority at 50 Hz is
  not a measurement; fewer ticks FAILS (fail closed), it does not pass vacuously.
* ``max_divergence_rad = 0.05``: mean over ticks of the L2 norm (12 joints) of
  ``final - scaled``; the policy-execution divergence.

These are proposals from first principles, not calibrated on a passing hardware
run (none exists). They are deliberately strict: a real policy that fails them
should be investigated, not waved through by loosening them.

Reason codes
------------
``slew_clip``          target moved more than the per-tick cap from measured q
``limit_clip``         target clipped onto a hard joint limit (within the band)
``illegal_target``     target beyond a hard limit by more than the abort band
``policy_node_slew``   (bridge telemetry) the policy node's own slew clip
``bridge_slew``        (bridge telemetry) the bridge's slew clip
``not_executed``       (bridge telemetry) the policy had authority but the tick
                       sent hold or damp instead of its target
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .go2_model import (
    LIMIT_ABORT_BAND_RAD,
    TRAINING_DEFAULT_JOINT_POS,
    limits_in_order,
    torque_limits_in_order,
)
from .safety import (
    DEFAULT_POSITION_CLIP_MARGIN_RAD,
    DEFAULT_SUSTAINED_CLIP_TICKS,
    MAX_DELTA_PER_STEP_RAD,
    SustainedClipTracker,
    clip_to_hard_limits_array,
    measured_position_illegal,
    torque_limited_target_array,
)

INTERVENTION_SCHEMA = "phoenix-intervention/v1"

SLEW_CLIP = "slew_clip"
LIMIT_CLIP = "limit_clip"
ILLEGAL_TARGET = "illegal_target"
POLICY_NODE_SLEW = "policy_node_slew"
BRIDGE_SLEW = "bridge_slew"
NOT_EXECUTED = "not_executed"
REASONS = (SLEW_CLIP, LIMIT_CLIP, ILLEGAL_TARGET, POLICY_NODE_SLEW, BRIDGE_SLEW, NOT_EXECUTED)

#: Below this a difference is float noise from permutation / float32 round trips.
MOD_EPS_RAD = 1e-6


@dataclass(frozen=True)
class InterventionThresholds:
    max_joint_tick_fraction: float = 0.02
    max_tick_fraction: float = 0.10
    max_rms_rad: float = 0.02
    max_abs_rad: float = 0.10
    max_illegal_target_aborts: int = 0
    min_policy_ticks: int = 50
    max_divergence_rad: float = 0.05

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


@dataclass
class FilterResult:
    final_target: np.ndarray
    reasons: list[list[str]]
    illegal: bool
    #: Specific cause when ``illegal`` is True: "command_non_finite",
    #: "measured_position_illegal", or "sustained_clip_exceeded". ``None``
    #: when not illegal. Callers may fall back to a generic label when this
    #: is unset for backward compatibility.
    fault_reason: str | None = None


class SafetyFilter:
    """Torque cap, hard-limit clip (with a small inward margin) and
    illegal-condition detection, with reasons.

    Order of operations matches :class:`phoenix.sim2real.actuator_gate.ActuatorGate`:

    1. A non-finite (NaN/Inf) requested target is illegal outright: it cannot
       come from any legal policy output, clamped or not.
    2. A MEASURED position beyond a hard limit by more than ``limit_abort_band``,
       or non-finite, is illegal: a sensor fault or a robot that has actually
       exceeded its mechanical stop, neither of which a command-side clip can
       fix.
    3. Otherwise: a torque cap against the measured ``q``/``dq`` at the
       commanded gains (:func:`phoenix.sim2real.safety.torque_limited_target_array`),
       then a position clip into the hard range with a small inward margin
       (:func:`phoenix.sim2real.safety.clip_to_hard_limits_array`). A target
       beyond a hard limit is CLIPPED here, not illegal: a PD controller
       commanding past a mechanical stop to press against it is normal
       (2026-09-25 sim2sim gate finding: a good walking policy requests calf
       targets 0.25-0.31 rad past the calf's hard upper limit during
       counter-clockwise yaw; proven stacks do not abort on that). Sustained
       clipping on any one joint for ``sustained_clip_ticks`` CONSECUTIVE
       ticks IS illegal: that signals a policy that cannot be steered off the
       stop, not a brief press.
    """

    def __init__(
        self,
        order: Sequence[str],
        max_delta: float = MAX_DELTA_PER_STEP_RAD,
        limit_abort_band: float = LIMIT_ABORT_BAND_RAD,
        kp: float = 25.0,
        kd: float = 0.5,
        torque_limit: Sequence[float] | None = None,
        position_clip_margin: float = DEFAULT_POSITION_CLIP_MARGIN_RAD,
        sustained_clip_ticks: int = DEFAULT_SUSTAINED_CLIP_TICKS,
    ) -> None:
        self.order = tuple(order)
        self.lo, self.hi = limits_in_order(self.order)
        self.max_delta = float(max_delta)
        self.band = float(limit_abort_band)
        self.kp = float(kp)
        self.kd = float(kd)
        self.torque_limit = (
            torque_limits_in_order(self.order)
            if torque_limit is None
            else np.asarray(torque_limit, dtype=np.float64)
        )
        self.position_clip_margin = float(position_clip_margin)
        self._clip_tracker = SustainedClipTracker(len(self.order), sustained_clip_ticks)

    def apply(
        self,
        requested: Sequence[float],
        q_measured: Sequence[float],
        dq_measured: Sequence[float] | None = None,
    ) -> FilterResult:
        req = np.asarray(requested, dtype=np.float64)
        q = np.asarray(q_measured, dtype=np.float64)
        n = len(self.order)
        dq = (
            np.zeros(n, dtype=np.float64)
            if dq_measured is None
            else np.asarray(dq_measured, dtype=np.float64)
        )
        reasons: list[list[str]] = [[] for _ in range(n)]

        non_finite = ~np.isfinite(req)
        if non_finite.any():
            for j in np.flatnonzero(non_finite):
                reasons[int(j)].append(ILLEGAL_TARGET)
            return FilterResult(
                np.clip(np.nan_to_num(q), self.lo, self.hi),
                reasons,
                True,
                "command_non_finite",
            )

        measured_illegal = measured_position_illegal(q, self.lo, self.hi, self.band)
        if measured_illegal.any():
            for j in np.flatnonzero(measured_illegal):
                reasons[int(j)].append(ILLEGAL_TARGET)
            return FilterResult(
                np.clip(np.nan_to_num(q), self.lo, self.hi),
                reasons,
                True,
                "measured_position_illegal",
            )

        torque_clipped = np.asarray(
            torque_limited_target_array(req, q, dq, self.kp, self.kd, self.torque_limit),
            dtype=np.float64,
        )
        final, position_clipped = clip_to_hard_limits_array(
            torque_clipped, self.lo, self.hi, self.position_clip_margin
        )
        for j in range(n):
            if abs(torque_clipped[j] - req[j]) > 0.0:
                reasons[j].append(SLEW_CLIP)
            if position_clipped[j]:
                reasons[j].append(LIMIT_CLIP)

        sustained_now = self._clip_tracker.update(position_clipped)
        if sustained_now.any():
            for j in np.flatnonzero(sustained_now):
                reasons[int(j)].append(ILLEGAL_TARGET)
            return FilterResult(
                np.clip(q, self.lo, self.hi), reasons, True, "sustained_clip_exceeded"
            )

        return FilterResult(final, reasons, False)


def tick_record(
    *,
    t_ns: int,
    order: Sequence[str],
    raw_action: Sequence[float] | None,
    scaled_target: Sequence[float],
    safety_target: Sequence[float],
    final_target: Sequence[float],
    reasons: Sequence[Sequence[str]],
    source: str,
) -> dict[str, Any]:
    """One per-tick intervention record. All arrays in ``order``. JSON-ready."""
    scaled = np.asarray(scaled_target, dtype=np.float64)
    final = np.asarray(final_target, dtype=np.float64)
    mod = np.abs(final - scaled)
    modified = [bool(m > MOD_EPS_RAD) for m in mod]
    return {
        "schema": INTERVENTION_SCHEMA,
        "source": source,
        "t_ns": int(t_ns),
        "order": list(order),
        "raw_action": None if raw_action is None else [float(v) for v in raw_action],
        "scaled_target": [float(v) for v in scaled],
        "safety_target": [float(v) for v in safety_target],
        "final_target": [float(v) for v in final],
        "modified": modified,
        "mod_rad": [float(v) for v in mod],
        "reasons": [list(r) for r in reasons],
    }


class InterventionRecorder:
    """Accumulates tick records in memory (the FSM path) and a live abort check."""

    def __init__(self, live_window: int = 25, live_max_fraction: float = 0.5) -> None:
        self.records: list[dict[str, Any]] = []
        self.illegal_aborts = 0
        self._window = int(live_window)
        self._live_max_fraction = float(live_max_fraction)

    def add(self, record: Mapping[str, Any], *, illegal: bool = False) -> None:
        self.records.append(dict(record))
        if illegal:
            self.illegal_aborts += 1

    def live_budget_exceeded(self) -> bool:
        """True when the last ``live_window`` ticks were modified on more than
        ``live_max_fraction`` of joint-ticks. A runtime brake, looser than the deploy
        gate, so a run that has stopped executing the policy is ended while it happens."""
        if len(self.records) < self._window:
            return False
        recent = self.records[-self._window :]
        hits = sum(sum(bool(m) for m in r["modified"]) for r in recent)
        total = sum(len(r["modified"]) for r in recent)
        return total > 0 and hits / total > self._live_max_fraction

    def summary(self) -> dict[str, Any]:
        return summarize(self.records, illegal_target_aborts=self.illegal_aborts)


def summarize(
    records: Sequence[Mapping[str, Any]], *, illegal_target_aborts: int = 0
) -> dict[str, Any]:
    """Metrics over tick records. Empty input gives ``n_ticks = 0`` and ``None`` metrics."""
    n = len(records)
    out: dict[str, Any] = {
        "schema": INTERVENTION_SCHEMA,
        "n_ticks": n,
        "illegal_target_aborts": int(illegal_target_aborts),
        "tick_fraction_modified": None,
        "joint_tick_fraction_modified": None,
        "max_mod_rad": None,
        "rms_mod_rad": None,
        "divergence_rad": None,
        "reason_rate": {},
        "per_joint_fraction_modified": {},
    }
    if n == 0:
        return out
    mods = np.asarray([r["mod_rad"] for r in records], dtype=np.float64)
    modified = np.asarray([r["modified"] for r in records], dtype=bool)
    joint_ticks = modified.size
    reason_counts: Counter[str] = Counter()
    for r in records:
        for joint_reasons in r["reasons"]:
            for reason in set(joint_reasons):
                reason_counts[reason] += 1
    order = list(records[0].get("order") or range(mods.shape[1]))
    out.update(
        {
            "tick_fraction_modified": float(modified.any(axis=1).mean()),
            "joint_tick_fraction_modified": float(modified.sum() / joint_ticks),
            "max_mod_rad": float(mods.max()),
            "rms_mod_rad": float(np.sqrt(np.mean(mods**2))),
            "divergence_rad": float(np.mean(np.linalg.norm(mods, axis=1))),
            "reason_rate": {k: reason_counts[k] / joint_ticks for k in sorted(reason_counts)},
            "per_joint_fraction_modified": {
                str(name): float(v) for name, v in zip(order, modified.mean(axis=0), strict=True)
            },
        }
    )
    return out


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deploy_gate(
    summary: Mapping[str, Any], thresholds: InterventionThresholds | None = None
) -> GateResult:
    """PASS only if every metric exists and is within its threshold. Fails closed."""
    th = thresholds or InterventionThresholds()
    failures: list[str] = []
    n = int(summary.get("n_ticks") or 0)
    if n < th.min_policy_ticks:
        failures.append(f"only {n} policy ticks (< {th.min_policy_ticks}): no verdict possible")

    def _metric(key: str, limit: float, label: str) -> None:
        value = summary.get(key)
        if value is None or not math.isfinite(float(value)):
            failures.append(f"{label}: metric missing")
        elif float(value) > limit:
            failures.append(f"{label} {float(value):.4g} > {limit:g}")

    _metric("joint_tick_fraction_modified", th.max_joint_tick_fraction, "joint-tick fraction")
    _metric("tick_fraction_modified", th.max_tick_fraction, "tick fraction")
    _metric("rms_mod_rad", th.max_rms_rad, "RMS modification rad")
    _metric("max_mod_rad", th.max_abs_rad, "max modification rad")
    _metric("divergence_rad", th.max_divergence_rad, "policy-execution divergence rad")
    aborts = int(summary.get("illegal_target_aborts") or 0)
    if aborts > th.max_illegal_target_aborts:
        failures.append(f"{aborts} illegal-target abort(s)")
    return GateResult(
        passed=not failures,
        failures=failures,
        summary=dict(summary),
        thresholds=th.to_dict(),
    )


# ------------------------------------------------------ bridge telemetry path
def bridge_tick_intervention(
    rec: Mapping[str, Any], perm: Sequence[int], motor_order: Sequence[str]
) -> dict[str, Any] | None:
    """Intervention record for one bridge tick in which the policy held authority.

    ``rec`` is an :class:`~phoenix.sim2real.actuator_gate.ActuatorGate` tick record.
    Everything is converted to Unitree motor order (``motor_order``) using
    ``perm`` (``PHOENIX_FOR_MOTOR``). Returns ``None`` for ticks with no policy
    command payload (nothing to compare).
    """
    pol = rec.get("policy") or {}
    requested = pol.get("requested_target")
    if not requested or any(v is None for v in requested):
        return None
    p = np.asarray(list(perm), dtype=np.int64)
    scaled = np.asarray(requested, dtype=np.float64)[p]
    node_target = pol.get("target")
    safety = (
        np.asarray(node_target, dtype=np.float64)[p]
        if node_target and all(v is not None for v in node_target)
        else scaled
    )
    final_list = rec.get("final_target_unitree")
    executed = rec.get("mode") == "policy" and final_list is not None
    final = np.asarray(final_list, dtype=np.float64) if final_list is not None else safety
    raw = pol.get("raw_action")
    raw_motor = (
        [float(v) for v in np.asarray(raw, dtype=np.float64)[p]]
        if raw and all(v is not None for v in raw)
        else None
    )
    reasons: list[list[str]] = [[] for _ in range(len(motor_order))]
    slew = rec.get("slew_clip") or [False] * len(motor_order)
    limit = rec.get("limit_clip") or [False] * len(motor_order)
    faults = [str(f) for f in rec.get("faults") or []]
    illegal_now = (
        any(
            f.startswith(("target_beyond_limit", "command_non_finite", "sustained_clip_exceeded"))
            for f in faults
        )
        and not executed
    )
    for j in range(len(motor_order)):
        if abs(safety[j] - scaled[j]) > MOD_EPS_RAD:
            reasons[j].append(POLICY_NODE_SLEW)
        if executed and slew[j]:
            reasons[j].append(BRIDGE_SLEW)
        if executed and limit[j]:
            reasons[j].append(LIMIT_CLIP)
        if not executed:
            reasons[j].append(ILLEGAL_TARGET if illegal_now else NOT_EXECUTED)
    return tick_record(
        t_ns=int(rec.get("t_mono_ns") or 0),
        order=motor_order,
        raw_action=raw_motor,
        scaled_target=scaled,
        safety_target=safety,
        final_target=final,
        reasons=reasons,
        source="bridge",
    )


def records_from_bridge_ticks(
    ticks: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Intervention records for the policy's authority window in a bridge run.

    The window is from the first ``mode == "policy"`` tick to the tick where the
    policy's authority ended (its own abort notice, e.g. ``authority_window_complete``)
    or the first fault after it. Ticks inside the window that are not ``policy`` mode
    are counted as ``not_executed``: the policy had authority and its target did not
    reach the motors. Only the first tick processing each command (``cmd_is_new``)
    is counted for executed ticks, matching :mod:`phoenix.sim2real.bridge_telemetry`.
    Returns ``(records, illegal_target_aborts)``.
    """
    from .go2_model import UNITREE_MOTOR_ORDER
    from .motor_crc import PHOENIX_FOR_MOTOR

    rows = list(ticks)
    first = next((i for i, t in enumerate(rows) if t.get("mode") == "policy"), None)
    if first is None:
        return [], 0
    records: list[dict[str, Any]] = []
    illegal = 0
    seen_faults = set(rows[first - 1].get("faults") or []) if first > 0 else set()
    for t in rows[first:]:
        faults = [str(f) for f in t.get("faults") or []]
        new_faults = [f for f in faults if f not in seen_faults]
        seen_faults.update(faults)
        if t.get("mode") == "policy":
            if not t.get("cmd_is_new", True):
                continue
            rec = bridge_tick_intervention(t, PHOENIX_FOR_MOTOR, UNITREE_MOTOR_ORDER)
            if rec is not None:
                records.append(rec)
            continue
        # Authority ended on this tick. An illegal target is the policy's command not
        # being executed; the window's own completion notice is not an intervention.
        if any(
            f.startswith(("target_beyond_limit", "command_non_finite", "sustained_clip_exceeded"))
            for f in new_faults
        ):
            illegal += 1
            rec = bridge_tick_intervention(t, PHOENIX_FOR_MOTOR, UNITREE_MOTOR_ORDER)
            if rec is not None:
                records.append(rec)
        break
    return records, illegal


def evaluate_bridge_run(
    ticks: Iterable[Mapping[str, Any]], thresholds: InterventionThresholds | None = None
) -> GateResult:
    records, illegal = records_from_bridge_ticks(ticks)
    return deploy_gate(summarize(records, illegal_target_aborts=illegal), thresholds)


def default_pose(order: Sequence[str]) -> np.ndarray:
    return np.asarray([TRAINING_DEFAULT_JOINT_POS[n] for n in order], dtype=np.float64)


__all__ = [
    "BRIDGE_SLEW",
    "ILLEGAL_TARGET",
    "INTERVENTION_SCHEMA",
    "LIMIT_CLIP",
    "NOT_EXECUTED",
    "POLICY_NODE_SLEW",
    "REASONS",
    "SLEW_CLIP",
    "FilterResult",
    "GateResult",
    "InterventionRecorder",
    "InterventionThresholds",
    "SafetyFilter",
    "bridge_tick_intervention",
    "default_pose",
    "deploy_gate",
    "evaluate_bridge_run",
    "records_from_bridge_ticks",
    "summarize",
    "tick_record",
]
