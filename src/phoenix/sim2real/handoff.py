"""Handoff criteria: is the MEASURED stance stable enough to grant the policy authority?

``FIX_STAND`` (:mod:`phoenix.sim2real.fix_stand`) drives a deterministic rise to the
training stance and reports when its own reference reached the target ("ramp
complete"). That is not a handoff criterion: a rise can finish on time while the
measured posture never actually settled (a leg still oscillating, a following
error the ramp's own abort band didn't quite trip). This module is the thing that
decides handoff from MEASURED state only, independent of what FIX_STAND thinks it
commanded.

:class:`HandoffMonitor` is a small stateful accumulator: every tick it scores one
:class:`HandoffSample` against :class:`HandoffCriteria` and returns a
:class:`HandoffStatus`. ``ready`` is True only after the sample has been
FAILING-FREE for ``criteria.sustained_ticks`` CONSECUTIVE ticks in a row; one good
sample after a bad one restarts the count at 1, not at the sustained count. This
is the "measured stance stable for its whole interval" the controller FSM's
module docstring describes: a stance that is momentarily fine but not sustained
must not hand over.

Where the default numbers come from
------------------------------------
None of these are hardware measurements (no robot has completed a FIX_STAND rise
yet); they are engineering choices, deliberately conservative for a first live
session, and every stage evaluation records what actually happened so they can be
revised from evidence.

* ``max_q_error_rad = 0.05``: five times tighter than FIX_STAND's own
  ``track_pause_rad`` (0.25), because "the ramp is still tracking" and "the stance
  is settled enough to hand over" are different bars.
* ``max_dq_rad_s = 0.5``: a joint still moving at a meaningful fraction of the
  policy's own action-derived rate (action_scale 0.25 * 50 Hz ~= 12.5 rad/s peak,
  so 0.5 rad/s is deep in "essentially still") is not a settled stance.
* ``max_gyro_rad_s = 0.3``: below the attitude-abort angular rate that would
  suggest the base itself is still rocking.
* ``max_tilt_rad = 0.15``: half of ``ControllerFSM``'s own ``attitude_abort_rad``
  default (0.40), so the handoff bar is stricter than the running abort bar.
* ``max_lowstate_age_s`` / ``max_imu_age_s = 0.1``: tighter than
  :class:`phoenix.sim2real.controller_fsm.FsmParams.sensor_timeout_s` default
  (0.1s exactly matches it here; a handoff must not grant on the last legal tick
  before a sensor would be declared stale).
* ``sustained_ticks = 25``: 0.5 s at 50 Hz. Long enough that a single settling
  transient does not look like a stable stance, short enough that a genuinely
  settled robot is not kept waiting.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HandoffCriteria:
    max_q_error_rad: float = 0.05
    max_dq_rad_s: float = 0.5
    max_gyro_rad_s: float = 0.3
    max_tilt_rad: float = 0.15
    max_lowstate_age_s: float = 0.1
    max_imu_age_s: float = 0.1
    sustained_ticks: int = 25

    def __post_init__(self) -> None:
        for name in (
            "max_q_error_rad",
            "max_dq_rad_s",
            "max_gyro_rad_s",
            "max_tilt_rad",
            "max_lowstate_age_s",
            "max_imu_age_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"HandoffCriteria.{name} must be positive and finite")
        if self.sustained_ticks < 1:
            raise ValueError("HandoffCriteria.sustained_ticks must be >= 1")


@dataclass(frozen=True)
class HandoffSample:
    """One tick's worth of measured state, in the FSM's joint order."""

    t_ns: int
    q: Sequence[float]
    dq: Sequence[float]
    gyro_body: Sequence[float] | None
    projected_gravity: np.ndarray | None
    lowstate_age_s: float | None
    imu_age_s: float | None
    estop_ok: bool
    deadman_ok: bool
    faults: tuple[str, ...] = ()
    fix_stand_complete: bool = False
    fix_stand_aborted: bool = False
    manifest_problems: tuple[str, ...] = ()


@dataclass
class HandoffStatus:
    ready: bool
    failing: list[str]
    consecutive_good_ticks: int

    def to_record(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "failing": list(self.failing),
            "consecutive_good_ticks": self.consecutive_good_ticks,
        }


class HandoffMonitor:
    """See the module docstring. Stateful: call :meth:`update` once per tick."""

    def __init__(
        self,
        order: Sequence[str],
        target: Sequence[float],
        criteria: HandoffCriteria | None = None,
    ) -> None:
        self.order = tuple(order)
        self.target = np.asarray(target, dtype=np.float64).reshape(-1)
        if self.target.shape != (len(self.order),):
            raise ValueError(f"target shape {self.target.shape} != ({len(self.order)},)")
        self.criteria = criteria or HandoffCriteria()
        self._consecutive_good = 0

    def reset(self) -> None:
        """Zero the consecutive-good counter (e.g. on re-entering FIX_STAND_HOLD)."""
        self._consecutive_good = 0

    def update(self, sample: HandoffSample) -> HandoffStatus:
        c = self.criteria
        failing: list[str] = []

        if not sample.estop_ok:
            failing.append("estop_not_ok")
        if not sample.deadman_ok:
            failing.append("deadman_not_ok")
        if sample.faults:
            failing.append("faults:" + ",".join(sample.faults))
        if sample.manifest_problems:
            failing.append("manifest_problems")
        if sample.fix_stand_aborted:
            failing.append("fix_stand_aborted")
        if not sample.fix_stand_complete:
            failing.append("fix_stand_not_complete")
        if sample.lowstate_age_s is None or not math.isfinite(sample.lowstate_age_s):
            failing.append("lowstate_age_missing")
        elif sample.lowstate_age_s > c.max_lowstate_age_s:
            failing.append(f"lowstate_stale_{sample.lowstate_age_s:.3f}s")
        if sample.imu_age_s is None or not math.isfinite(sample.imu_age_s):
            failing.append("imu_age_missing")
        elif sample.imu_age_s > c.max_imu_age_s:
            failing.append(f"imu_stale_{sample.imu_age_s:.3f}s")

        q = np.asarray(sample.q, dtype=np.float64).reshape(-1)
        if q.shape != self.target.shape or not np.all(np.isfinite(q)):
            failing.append("q_invalid")
        else:
            err = float(np.max(np.abs(q - self.target)))
            if err > c.max_q_error_rad:
                failing.append(f"q_error_{err:.3f}rad")

        dq = np.asarray(sample.dq, dtype=np.float64).reshape(-1)
        if dq.shape != self.target.shape or not np.all(np.isfinite(dq)):
            failing.append("dq_invalid")
        else:
            max_dq = float(np.max(np.abs(dq)))
            if max_dq > c.max_dq_rad_s:
                failing.append(f"dq_{max_dq:.3f}rad_s")

        if sample.gyro_body is None:
            failing.append("gyro_missing")
        else:
            gyro = np.asarray(sample.gyro_body, dtype=np.float64).reshape(-1)
            if gyro.shape != (3,) or not np.all(np.isfinite(gyro)):
                failing.append("gyro_invalid")
            else:
                mag = float(np.linalg.norm(gyro))
                if mag > c.max_gyro_rad_s:
                    failing.append(f"gyro_{mag:.3f}rad_s")

        if sample.projected_gravity is None:
            failing.append("attitude_missing")
        else:
            g = np.asarray(sample.projected_gravity, dtype=np.float64).reshape(-1)
            if g.shape != (3,) or not np.all(np.isfinite(g)):
                failing.append("attitude_invalid")
            else:
                tilt = float(np.arccos(np.clip(-g[2], -1.0, 1.0)))
                if tilt > c.max_tilt_rad:
                    failing.append(f"tilt_{tilt:.3f}rad")

        good = not failing
        self._consecutive_good = self._consecutive_good + 1 if good else 0
        ready = good and self._consecutive_good >= c.sustained_ticks
        return HandoffStatus(
            ready=ready, failing=failing, consecutive_good_ticks=self._consecutive_good
        )


__all__ = [
    "HandoffCriteria",
    "HandoffMonitor",
    "HandoffSample",
    "HandoffStatus",
]
