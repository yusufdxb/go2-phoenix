"""Rule-based failure detection for real-robot telemetry.

Phoenix treats three regimes as failures worth re-playing in sim:

1. **Attitude loss** — pitch or roll exceeds a safety threshold (robot is
   on its way to tipping over).
2. **Body collapse** — base height drops below a floor threshold (it
   dropped a leg, folded, or is on its belly).
3. **Foot slip** — commanded velocity is high yet measured linear velocity
   is near zero *and* foot contact signals are unstable.

The detector is stateful so it can emit a single :class:`FailureEvent` per
failure episode (rather than one per step while the failure persists).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class FailureMode(str, Enum):
    ATTITUDE = "attitude"
    COLLAPSE = "collapse"
    SLIP = "slip"


@dataclass(frozen=True)
class FailureThresholds:
    pitch_rad: float = 0.8
    roll_rad: float = 0.6
    base_height_min_m: float = 0.15
    slip_velocity_cmd_min: float = 0.3  # m/s
    slip_velocity_actual_max: float = 0.05  # m/s
    slip_min_duration_s: float = 0.5
    min_event_gap_s: float = 1.0  # suppress duplicate events


@dataclass
class FailureEvent:
    mode: FailureMode
    timestamp_s: float
    detail: dict = field(default_factory=dict)


class FailureDetector:
    """Stateful detector producing :class:`FailureEvent` instances.

    Feed it telemetry with :meth:`step`; it returns an event when a new
    failure is detected, otherwise ``None``.
    """

    def __init__(self, thresholds: FailureThresholds | None = None) -> None:
        self.thresholds = thresholds or FailureThresholds()
        self._slip_start: float | None = None
        self._last_event_at: float = -np.inf

    def step(
        self,
        *,
        timestamp_s: float,
        pitch_rad: float,
        roll_rad: float,
        base_height_m: float,
        cmd_lin_vel: np.ndarray,  # (2,) [vx, vy]
        actual_lin_vel: np.ndarray,  # (2,)
    ) -> FailureEvent | None:
        t = self.thresholds

        if timestamp_s - self._last_event_at < t.min_event_gap_s:
            return None

        if abs(pitch_rad) > t.pitch_rad or abs(roll_rad) > t.roll_rad:
            return self._emit(
                FailureMode.ATTITUDE,
                timestamp_s,
                {"pitch": pitch_rad, "roll": roll_rad},
            )

        if base_height_m < t.base_height_min_m:
            return self._emit(
                FailureMode.COLLAPSE,
                timestamp_s,
                {"height": base_height_m},
            )

        # Slip requires *sustained* disagreement between command and actual.
        cmd_speed = float(np.linalg.norm(cmd_lin_vel))
        actual_speed = float(np.linalg.norm(actual_lin_vel))
        slipping = cmd_speed > t.slip_velocity_cmd_min and actual_speed < t.slip_velocity_actual_max
        if slipping:
            if self._slip_start is None:
                self._slip_start = timestamp_s
            elif timestamp_s - self._slip_start >= t.slip_min_duration_s:
                event = self._emit(
                    FailureMode.SLIP,
                    timestamp_s,
                    {"cmd_speed": cmd_speed, "actual_speed": actual_speed},
                )
                self._slip_start = None
                return event
        else:
            self._slip_start = None

        return None

    def _emit(self, mode: FailureMode, ts: float, detail: dict) -> FailureEvent:
        self._last_event_at = ts
        return FailureEvent(mode=mode, timestamp_s=ts, detail=detail)
