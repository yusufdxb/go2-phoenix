"""FIX_STAND: a deterministic, tracking-aware rise from the MEASURED posture to the stance.

No learned component runs here. This replaces the 2026-09-21 bridge ramp
(``ActuatorGate`` ``standup`` mode, commit ``9df76d7``) as the way a GO2 gets from
lying on the mat to the training stance, because that ramp had three properties
that made its outcome unobservable:

1. **It was open loop in time.** The reference was ``(1 - a) q0 + a q_stand`` with
   ``a = elapsed / standup_s``. If a leg could not follow (the F run on 2026-09-21
   showed loaded calves that did not move for 29 policy ticks), the reference ran
   away from the robot and still reached ``a = 1``.
2. **Completion was the timer.** ``standup_done`` became true when ``a`` reached 1,
   whatever the joints were doing, and the harness launched the policy on that flag.
   Nothing compared the measured posture with the stance before handing over.
3. **Tracking was not recorded.** The telemetry held the target and the measured
   ``q`` but no following error, so "did it rise?" was an operator impression.

What this module does instead
-----------------------------
* The start posture is the measured ``q`` (hard-limit checked: beyond a limit by
  more than :data:`~phoenix.sim2real.go2_model.LIMIT_ABORT_BAND_RAD` refuses to
  start, within the band is clipped onto the limit, exactly as the gate treats it).
* The reference follows a smoothstep ``3s^2 - 2s^3`` in a PROGRESS variable ``s``,
  not in time. ``s`` advances by ``1 / (duration * rate)`` per tick only while every
  joint is within ``track_pause_rad`` of the reference; otherwise the reference
  pauses where it is. A leg that cannot follow therefore stops the rise instead of
  being out-run, and a pause longer than ``max_pause_s`` aborts to damping.
* The duration is chosen from the largest joint excursion so the smoothstep's
  peak speed (1.5 x average) stays under ``max_ref_speed_rad_s``, with a floor of
  ``min_duration_s``. Every commanded step is also capped at
  ``max_step_rad`` per tick as a second, independent bound.
* A following error beyond ``track_abort_rad`` on any joint aborts immediately.
* :meth:`FixStand.abort` switches to damping (``kp = 0``, ``kd = damp_kd``) at once
  and is latched; every later tick is damping.
* Every tick returns the reference, the measured posture and the per-joint error.

"Ramp complete" (``s == 1``) is reported, but it is NOT a handoff criterion:
:mod:`phoenix.sim2real.handoff` decides that from measured state only.

Numbers and where they come from
--------------------------------
``kp = 60, kd = 5``: Unitree's ``go2_stand_example.cpp`` gains, the same values the
2026-09-21 ramp used (so F1 changes the reference, not the stiffness).
``max_ref_speed_rad_s = 0.8``: folded (calf about -2.78 rad, measured 2026-09-21) to
stance (-1.5 rad) is about 1.3 rad, so the rise takes about 2.4 s; the previous
ramp did it in 2.0 s with a linear profile (0.65 rad/s average, a velocity step at
both ends). ``max_step_rad = 0.03``: about twice the smoothstep's largest per-tick
step at 0.8 rad/s and 50 Hz (0.016 rad), so it only binds if the planner is wrong.
``track_pause_rad = 0.25`` and ``track_abort_rad = 0.60``: engineering choices, NOT
measurements. At kp 60 a 0.25 rad error is 15 N m of spring torque, above what a
standing GO2 leg joint needs (hardware-unverified, see docs/hardware/), so a
healthy rise should not pause; 0.60 rad is well beyond anything a following leg
shows and well inside what a blocked or unpowered one would. F1 on hardware exists
to measure the real following error and revise these.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .go2_model import LIMIT_ABORT_BAND_RAD, TRAINING_DEFAULT_JOINT_POS, limits_in_order

PHASE_IDLE = "idle"
PHASE_RAMP = "ramp"
PHASE_COMPLETE = "complete"
PHASE_ABORTED = "aborted"


class FixStandError(ValueError):
    """The rise cannot start (bad measured state or parameters)."""


@dataclass(frozen=True)
class FixStandParams:
    rate_hz: float = 50.0
    kp: float = 60.0
    kd: float = 5.0
    damp_kd: float = 1.0
    max_ref_speed_rad_s: float = 0.8
    min_duration_s: float = 2.0
    max_step_rad: float = 0.03
    track_pause_rad: float = 0.25
    track_abort_rad: float = 0.60
    max_pause_s: float = 1.0
    #: The rise must finish within this multiple of its planned duration.
    timeout_factor: float = 3.0
    start_limit_band_rad: float = LIMIT_ABORT_BAND_RAD

    def __post_init__(self) -> None:
        for name in (
            "rate_hz",
            "kp",
            "max_ref_speed_rad_s",
            "min_duration_s",
            "max_step_rad",
            "track_pause_rad",
            "track_abort_rad",
            "max_pause_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"FixStandParams.{name} must be positive and finite, got {value}")
        for name in ("kd", "damp_kd", "start_limit_band_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"FixStandParams.{name} must be >= 0 and finite, got {value}")
        if self.track_abort_rad <= self.track_pause_rad:
            raise ValueError("track_abort_rad must exceed track_pause_rad")
        if self.timeout_factor < 1.0:
            raise ValueError("timeout_factor must be >= 1")
        if self.max_step_rad * self.rate_hz < self.max_ref_speed_rad_s:
            raise ValueError(
                "max_step_rad * rate_hz must be >= max_ref_speed_rad_s, or the step cap "
                "would silently slow every rise"
            )

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


def stand_target(order: Sequence[str]) -> np.ndarray:
    """The training default stance in ``order`` (a KeyError for an unknown joint name)."""
    return np.asarray([TRAINING_DEFAULT_JOINT_POS[n] for n in order], dtype=np.float64)


def smoothstep(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s * s * (3.0 - 2.0 * s)


@dataclass
class FixStandTick:
    """One tick's command and its tracking record. Arrays are in the planner's order."""

    phase: str
    publish: bool
    target: np.ndarray
    kp: float
    kd: float
    progress: float
    paused: bool
    q_measured: np.ndarray
    tracking_error: np.ndarray  # previous commanded target minus measured q
    max_abs_error: float
    fault: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "publish": self.publish,
            "target": [float(v) for v in self.target],
            "kp": float(self.kp),
            "kd": float(self.kd),
            "progress": float(self.progress),
            "paused": bool(self.paused),
            "q_measured": [float(v) for v in self.q_measured],
            "tracking_error": [float(v) for v in self.tracking_error],
            "max_abs_error": float(self.max_abs_error),
            "fault": self.fault,
            "notes": list(self.notes),
        }


class FixStand:
    """Tick-driven deterministic rise. See the module docstring.

    Usage::

        fs = FixStand(order, params)
        fs.start(q_measured)              # raises FixStandError if it must not start
        tick = fs.step(q_measured)        # every control tick
        fs.abort("deadman_released")      # any time; latched damping
    """

    def __init__(
        self,
        order: Sequence[str],
        params: FixStandParams | None = None,
        target: Sequence[float] | None = None,
    ) -> None:
        self.order = tuple(order)
        self.params = params or FixStandParams()
        self.lo, self.hi = limits_in_order(self.order)
        self.target = stand_target(self.order) if target is None else np.asarray(target, float)
        if self.target.shape != (len(self.order),):
            raise FixStandError(f"target shape {self.target.shape} != ({len(self.order)},)")
        if np.any(self.target < self.lo) or np.any(self.target > self.hi):
            raise FixStandError("stand target is outside the hard joint limits")
        self.phase = PHASE_IDLE
        self.fault: str | None = None
        self.q0: np.ndarray | None = None
        self.duration_s: float | None = None
        self._s = 0.0
        self._ticks = 0
        self._paused_ticks = 0
        self._last_cmd: np.ndarray | None = None
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------ lifecycle
    @property
    def progress(self) -> float:
        return self._s

    @property
    def complete(self) -> bool:
        return self.phase == PHASE_COMPLETE

    @property
    def aborted(self) -> bool:
        return self.phase == PHASE_ABORTED

    def _check_q(self, q: np.ndarray, what: str) -> None:
        if q.shape != (len(self.order),):
            raise FixStandError(f"{what} shape {q.shape} != ({len(self.order)},)")
        if not np.all(np.isfinite(q)):
            raise FixStandError(f"{what} is not finite: {q.tolist()}")

    def plan_duration(self, q0: np.ndarray) -> float:
        p = self.params
        excursion = float(np.max(np.abs(self.target - q0)))
        return max(p.min_duration_s, 1.5 * excursion / p.max_ref_speed_rad_s)

    def start(self, q_measured: Sequence[float]) -> None:
        if self.phase != PHASE_IDLE:
            raise FixStandError(f"start() in phase {self.phase}; a FixStand runs once")
        q = np.asarray(q_measured, dtype=np.float64).reshape(-1)
        self._check_q(q, "measured q")
        band = self.params.start_limit_band_rad
        beyond = (q < self.lo - band) | (q > self.hi + band)
        if beyond.any():
            names = [n for n, b in zip(self.order, beyond, strict=True) if b]
            raise FixStandError(f"measured q beyond hard limits by more than {band} rad: {names}")
        self.q0 = np.clip(q, self.lo, self.hi)
        self.duration_s = self.plan_duration(self.q0)
        self._last_cmd = self.q0.copy()
        self.phase = PHASE_RAMP

    def abort(self, reason: str) -> None:
        """Latch damping. Idempotent; the first reason is kept."""
        if self.phase != PHASE_ABORTED:
            self.fault = reason
            self.phase = PHASE_ABORTED

    # ----------------------------------------------------------------- tick
    def reference(self, s: float) -> np.ndarray:
        assert self.q0 is not None
        a = smoothstep(s)
        return (1.0 - a) * self.q0 + a * self.target

    def _damp(self, q: np.ndarray, err: np.ndarray, notes: list[str]) -> FixStandTick:
        safe_q = np.clip(np.nan_to_num(q, nan=0.0), self.lo, self.hi)
        tick = FixStandTick(
            phase=PHASE_ABORTED,
            publish=True,
            target=safe_q,
            kp=0.0,
            kd=self.params.damp_kd,
            progress=self._s,
            paused=False,
            q_measured=q,
            tracking_error=err,
            max_abs_error=float(np.max(np.abs(err))) if err.size else 0.0,
            fault=self.fault,
            notes=notes,
        )
        self.history.append(tick.to_record())
        return tick

    def step(self, q_measured: Sequence[float]) -> FixStandTick:
        """One control tick. Must be called after :meth:`start` (or after an abort)."""
        p = self.params
        q = np.asarray(q_measured, dtype=np.float64).reshape(-1)
        notes: list[str] = []
        if self.phase == PHASE_IDLE:
            raise FixStandError("step() before start()")
        if q.shape != (len(self.order),) or not np.all(np.isfinite(q)):
            self.abort("fix_stand_measured_q_invalid")
            return self._damp(
                q if q.shape == (len(self.order),) else np.zeros(len(self.order)),
                np.zeros(len(self.order)),
                ["measured q invalid"],
            )
        assert self._last_cmd is not None and self.duration_s is not None
        err = self._last_cmd - q
        max_err = float(np.max(np.abs(err)))
        if self.phase == PHASE_ABORTED:
            return self._damp(q, err, notes)

        band = p.start_limit_band_rad
        if np.any((q < self.lo - band) | (q > self.hi + band)):
            self.abort("fix_stand_measured_q_beyond_limit")
            return self._damp(q, err, ["measured q beyond hard limit band"])
        if max_err > p.track_abort_rad:
            self.abort(f"fix_stand_tracking_error:{max_err:.3f}rad")
            return self._damp(q, err, notes)

        self._ticks += 1
        paused = max_err > p.track_pause_rad
        if paused:
            self._paused_ticks += 1
            notes.append(f"reference paused: max following error {max_err:.3f} rad")
            if self._paused_ticks / p.rate_hz > p.max_pause_s:
                self.abort("fix_stand_tracking_lost")
                return self._damp(q, err, notes)
        else:
            self._paused_ticks = 0
            self._s = min(1.0, self._s + 1.0 / (self.duration_s * p.rate_hz))

        if self.phase == PHASE_RAMP and self._ticks / p.rate_hz > p.timeout_factor * (
            self.duration_s
        ):
            self.abort("fix_stand_timeout")
            return self._damp(q, err, notes)

        ref = self.reference(self._s)
        step = np.clip(ref - self._last_cmd, -p.max_step_rad, p.max_step_rad)
        if np.any(step != ref - self._last_cmd):
            notes.append("per-step cap bound")
        cmd = np.clip(self._last_cmd + step, self.lo, self.hi)
        self._last_cmd = cmd
        if self._s >= 1.0 and np.allclose(cmd, self.target, atol=1e-12, rtol=0.0):
            self.phase = PHASE_COMPLETE
        tick = FixStandTick(
            phase=self.phase,
            publish=True,
            target=cmd.copy(),
            kp=p.kp,
            kd=p.kd,
            progress=self._s,
            paused=paused,
            q_measured=q,
            tracking_error=err,
            max_abs_error=max_err,
            notes=notes,
        )
        self.history.append(tick.to_record())
        return tick

    def summary(self) -> dict[str, Any]:
        """Tracking statistics over every recorded tick (for the stage evidence)."""
        errs = np.asarray([h["tracking_error"] for h in self.history], dtype=np.float64)
        out: dict[str, Any] = {
            "phase": self.phase,
            "fault": self.fault,
            "duration_planned_s": self.duration_s,
            "ticks": len(self.history),
            "paused_ticks": int(sum(1 for h in self.history if h["paused"])),
            "params": self.params.to_dict(),
        }
        if errs.size:
            out["max_abs_tracking_error_rad"] = float(np.max(np.abs(errs)))
            out["rms_tracking_error_rad"] = float(np.sqrt(np.mean(errs**2)))
            out["max_abs_tracking_error_per_joint"] = {
                n: float(v) for n, v in zip(self.order, np.max(np.abs(errs), axis=0), strict=True)
            }
        return out


def params_from_mapping(data: Mapping[str, Any] | None) -> FixStandParams:
    """Build params from a config mapping; unknown keys are refused, not ignored."""
    data = dict(data or {})
    known = set(FixStandParams.__dataclass_fields__)
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown fix_stand keys {unknown}")
    return FixStandParams(**{k: float(v) for k, v in data.items()})


__all__ = [
    "PHASE_ABORTED",
    "PHASE_COMPLETE",
    "PHASE_IDLE",
    "PHASE_RAMP",
    "FixStand",
    "FixStandError",
    "FixStandParams",
    "FixStandTick",
    "params_from_mapping",
    "smoothstep",
    "stand_target",
]
