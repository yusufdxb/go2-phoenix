"""Rule-based failure detection for real-robot telemetry.

WHAT THIS DETECTOR ACTUALLY MEASURES
------------------------------------

Phoenix flags three regimes as failures worth re-playing in sim. The
definitions below are the implementation, not an aspiration; :meth:`step`
consults exactly the arguments in its signature and nothing else.

1. **Attitude loss** (:attr:`FailureMode.ATTITUDE`): ``|pitch|`` or ``|roll|``
   exceeds a threshold. Source: IMU orientation. Fires on the first sample
   past the threshold, so detection latency is one sample period.

2. **Body collapse** (:attr:`FailureMode.COLLAPSE`): a *validated
   ground-relative* base height drops below a floor. Source: whatever the
   caller passes as ``base_height_m``, which must be measured from the ground
   plane. Pass ``None`` when no such source exists and the mode is reported
   as unavailable rather than silently never firing.

   On the real GO2 there is currently NO validated ground-relative height
   source. ``/utlidar/robot_odom`` is origin-at-boot-pose
   (docs/go2_field_notes.md section 3), so ``pose.position.z`` is displacement
   from wherever the robot booted, not height above the floor: a robot that
   boots standing reads z ~ 0 while standing, and a robot carried up a step
   reads a large z while lying down. Hardware captures therefore pass
   ``None`` here and never claim a collapse. Simulator captures pass the true
   env-origin-relative base height and do.

3. **Commanded-velocity tracking stall** (:attr:`FailureMode.SLIP`, wire value
   ``"slip"``): commanded planar speed is high while measured planar speed
   stays near zero, sustained for ``slip_min_duration_s``.

   NO CONTACT SIGNAL IS CONSULTED, deliberately. An earlier version of this
   docstring claimed slip required "foot contact signals are unstable"; that
   was never implemented and there is no honest way to implement it today.
   The only per-foot signal the platform offers is
   ``unitree_go/msg/LowState.foot_force``, ``int16[4]``, with no documented
   Newton calibration and no documented per-index leg ordering (see
   :func:`phoenix.sim2real.telemetry.foot_force_to_array`). A contact
   threshold on uncalibrated counts would be an unfalsifiable number, and a
   slip criterion built on it would not be reproducible by anyone else.

   So the mode is kinematic, and it is a SUPERSET of slip: a sustained
   failure to track the commanded velocity is consistent with foot slip, but
   also with being blocked by an obstacle, a stuck or saturated actuator, a
   command the gait cannot deliver, or a controller that is simply ignoring
   the command. Consumers must not report a ``"slip"`` row as a measured
   foot-slip event. Upgrading this to a real slip definition requires
   calibrated per-foot normal force plus per-foot velocity, neither of which
   the stock robot publishes; that is a hardware-instrumentation task, not a
   threshold change.

PHOENIX MODES ARE NOT THE SIMULATOR'S TERMINATION ONTOLOGY
----------------------------------------------------------

These are two different vocabularies and they do not map one-to-one.

============================  ==========================================
Phoenix detector mode          Nearest simulator termination
============================  ==========================================
``attitude``                   none. Isaac Lab's Go2 velocity task has no
                               pitch/roll termination term; a tipping
                               robot ends the episode only once its trunk
                               actually touches something
                               (``base_contact``).
``collapse``                   none directly. A collapsed robot usually
                               reaches ``base_contact``, but by trunk
                               contact force, not by height.
``slip``                       none. A stalled robot accrues a poor
                               ``track_lin_vel_xy`` reward and keeps
                               running to ``time_out``.
(no Phoenix mode)              ``base_contact``: illegal contact force on
                               the trunk body. This is the sim's actual
                               failure event and the detector cannot see
                               it, because the real robot publishes no
                               trunk contact sensor.
(no Phoenix mode)              ``time_out``: episode length reached. Not a
                               failure at all.
============================  ==========================================

Practical consequences, both of which have bitten this repo before:

* A sim rollout that terminated on ``base_contact`` may carry zero Phoenix
  failure rows, and a Phoenix ``slip`` row may sit inside an episode the sim
  considered perfectly nominal. "Failure count" is only comparable when both
  sides name which ontology they used.
* ``configs/env/base.yaml``'s ``termination:`` block (``base_contact``,
  ``pitch_threshold_rad``, ``roll_threshold_rad``, ``base_height_min``) is
  UNWIRED, it is listed in ``go2_env_cfg._UNWIRED_TOP_LEVEL``. Its numbers
  happen to equal :class:`FailureThresholds`' defaults because they were
  hand-copied, not because anything loads one into the other. Editing that
  YAML block changes neither the simulator nor this detector.

The detector is stateful so it can emit a single :class:`FailureEvent` per
failure episode (rather than one per step while the failure persists).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class FailureMode(str, Enum):
    #: Pitch or roll past threshold. See module docstring.
    ATTITUDE = "attitude"
    #: Validated ground-relative base height below threshold.
    COLLAPSE = "collapse"
    #: Sustained commanded-velocity tracking stall. NOT a measured foot slip.
    SLIP = "slip"


#: One-line definition per mode, kept next to the enum so a consumer never has
#: to guess. ``tests/test_failure_detector.py`` asserts every mode has one.
MODE_DEFINITIONS: dict[FailureMode, str] = {
    FailureMode.ATTITUDE: "abs(pitch) or abs(roll) above threshold, from IMU orientation",
    FailureMode.COLLAPSE: (
        "validated ground-relative base height below threshold; unavailable when the caller "
        "passes base_height_m=None (no such source exists on the real GO2)"
    ),
    FailureMode.SLIP: (
        "commanded planar speed high while measured planar speed is near zero, sustained; "
        "kinematic only, no contact signal is consulted, so it is a superset of foot slip"
    ),
}


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

    Every event's ``detail`` carries ``onset_timestamp_s`` and
    ``detection_latency_s``, so a consumer can measure how late the detector
    was without re-deriving onset from the raw trace.
    """

    def __init__(self, thresholds: FailureThresholds | None = None) -> None:
        self.thresholds = thresholds or FailureThresholds()
        self._slip_start: float | None = None
        self._last_event_at: float = -math.inf

    def step(
        self,
        *,
        timestamp_s: float,
        pitch_rad: float,
        roll_rad: float,
        base_height_m: float | None,
        cmd_lin_vel,  # (2,) [vx, vy]
        actual_lin_vel,  # (2,)
    ) -> FailureEvent | None:
        """Classify one telemetry sample.

        ``base_height_m=None`` means "no validated ground-relative height
        source this step": collapse detection is skipped rather than being
        fed a number that does not mean height. Any other non-finite input
        raises, because a NaN silently satisfies no comparison and would turn
        a broken sensor into a clean "no failure" row.
        """
        t = self.thresholds

        timestamp_s = _require_finite("timestamp_s", timestamp_s)
        pitch_rad = _require_finite("pitch_rad", pitch_rad)
        roll_rad = _require_finite("roll_rad", roll_rad)
        if base_height_m is not None:
            base_height_m = _require_finite("base_height_m", base_height_m)
        cmd_speed = _planar_speed("cmd_lin_vel", cmd_lin_vel)
        actual_speed = _planar_speed("actual_lin_vel", actual_lin_vel)

        # Slip bookkeeping runs BEFORE the duplicate-event gate. It used to
        # run after, behind an early ``return``, which meant a suppressed tick
        # froze ``_slip_start``: a slip episode that began after an unrelated
        # event could inherit a stale onset and fire instantly, reporting a
        # detection latency it had not earned.
        slipping = cmd_speed > t.slip_velocity_cmd_min and actual_speed < t.slip_velocity_actual_max
        if slipping:
            if self._slip_start is None:
                self._slip_start = timestamp_s
        else:
            self._slip_start = None

        # Duplicate suppression. NOTE: this is a global gate, not a per-mode
        # one, so a genuine collapse within min_event_gap_s of an attitude
        # event is dropped. Deliberate and locked by a test; a rollout is
        # replayed from its first event, so the second label would not change
        # what gets replayed.
        if timestamp_s - self._last_event_at < t.min_event_gap_s:
            return None

        if abs(pitch_rad) > t.pitch_rad or abs(roll_rad) > t.roll_rad:
            return self._emit(
                FailureMode.ATTITUDE,
                timestamp_s,
                onset_s=timestamp_s,
                detail={"pitch": pitch_rad, "roll": roll_rad},
            )

        if base_height_m is not None and base_height_m < t.base_height_min_m:
            return self._emit(
                FailureMode.COLLAPSE,
                timestamp_s,
                onset_s=timestamp_s,
                detail={"height": base_height_m},
            )

        if (
            slipping
            and self._slip_start is not None
            and timestamp_s - self._slip_start >= t.slip_min_duration_s
        ):
            onset_s = self._slip_start
            event = self._emit(
                FailureMode.SLIP,
                timestamp_s,
                onset_s=onset_s,
                detail={"cmd_speed": cmd_speed, "actual_speed": actual_speed},
            )
            self._slip_start = None
            return event

        return None

    def _emit(
        self, mode: FailureMode, ts: float, *, onset_s: float, detail: dict
    ) -> FailureEvent:
        self._last_event_at = ts
        full = dict(detail)
        full["onset_timestamp_s"] = onset_s
        full["detection_latency_s"] = ts - onset_s
        return FailureEvent(mode=mode, timestamp_s=ts, detail=full)


def _require_finite(name: str, value) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(
            f"{name} is {out!r}; a non-finite telemetry sample satisfies no threshold "
            "comparison and would be silently reported as 'no failure'"
        )
    return out


def _planar_speed(name: str, vec) -> float:
    try:
        vx = float(vec[0])
        vy = float(vec[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError(f"{name} must be indexable with at least 2 components") from exc
    _require_finite(f"{name}[0]", vx)
    _require_finite(f"{name}[1]", vy)
    return math.hypot(vx, vy)
