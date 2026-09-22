"""The final actuator gate: the last code that decides what the GO2 motors are told.

``lowcmd_bridge_node`` is a thin ROS shell around :class:`ActuatorGate`. Every
decision lives here, as a pure state machine over explicit monotonic
nanosecond timestamps, so the whole safety boundary is exercised by the
no-hardware test suite. The ROS node only feeds messages in, publishes what
:meth:`ActuatorGate.tick` returns, and writes the returned record to telemetry.

Modes, in order of precedence
-----------------------------
``silent``  no LowState has ever arrived, so there is no posture to hold. Publish
            nothing (unchanged from the previous bridge).
``damp``    the measured posture cannot be trusted: LowState went non-finite,
            reported a physically impossible joint position, or stayed stale for
            longer than ``stale_hold_s``. Publish ``kp=0, kd=hold_kd`` so the
            robot sinks onto the mat under damping instead of being stiffened
            toward a posture nobody has measured. Latched.
``hold``    hold the last measured posture with the hold gains. Entered for any
            latched fault (estop, ordering mismatch, NaN command, target beyond
            the abort band, policy abort, stale LowState while still inside the
            stale window) and, NOT latched, for a stale or absent policy command.
            The non-latching command watchdog is the pre-existing behaviour.
``standup`` (only when ``standup_s > 0``) the bridge is armed, the real deadman is
            held, nothing is latched and the stand has not finished: ramp
            linearly from the posture measured at the first such tick to
            ``TRAINING_DEFAULT_JOINT_POS`` over ``standup_s`` at the standup
            gains, then hold that stance until a policy command arrives. A
            policy command during the ramp is not followed. Any fault, estop or
            deadman release drops out of it by the precedence above and resets
            it, so a re-arm ramps again from wherever the robot then is. Added
            2026-09-21: stage F showed a folded start cannot rise under the
            measured-q slew clip (kp x 0.175 rad caps each joint near 4.4 N m)
            and is out of the policy's training distribution.
``policy``  follow the policy node's command, permuted to Unitree motor order,
            slew-clipped against the fresh measured position, then clipped to
            the hard joint limits.

What changed relative to the previous bridge, and why
-----------------------------------------------------
* **Absolute joint limits.** There were none. Targets are now held to the hard
  URDF limits in :mod:`phoenix.sim2real.go2_model`; a slew-clipped target within
  one slew cap of a limit is clipped to it (counted per joint), and a requested
  target further out than that aborts to hold. See ``LIMIT_ABORT_BAND_RAD``.
* **LowState freshness.** The previous bridge clipped and held against the last
  LowState forever, however old. Now a LowState older than
  ``lowstate_timeout_s`` latches a fault (policy authority revoked), holds the
  last posture for at most ``stale_hold_s``, then damps.
* **Monotonic time.** The previous bridge timed the estop heartbeat with the ROS
  wall clock. The payload has no RTC and its clock is set by hand mid-session
  (``docs/go2_field_notes.md`` section 6); a backwards ``date -s`` made a dead
  estop publisher look fresh. The node now passes ``time.monotonic_ns()``.
* **NaN / malformed commands fail closed.** They used to be dropped with a
  warning, which left the previous target in force until the watchdog expired.
  Now they latch hold.
* **Default-pose messages are not followed.** The policy node publishes its
  nominal pose while waiting for its first messages and once on abort. The old
  bridge drove the robot toward it (up to 0.175 rad per tick at kp=25), an
  unpoliced stand-up or a lurch at abort. The bridge now holds measured posture.
* **Estop discovery race.** An estop that has not been heard from yet, inside
  ``first_message_timeout_s``, holds without latching; before, DDS discovery
  order alone could latch the bridge forever at startup. A heard ``True``
  still latches, exactly as before.
* **Real deadman required when live.** In live mode the ``/phoenix/estop``
  publisher set must be exactly one node from :data:`REAL_DEADMAN_NODE_NAMES`.
  ``scripts/estop_publisher.sh`` (``ros2 topic pub``) can never arm a live bridge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .command_wire import (
    KIND_ABORT,
    KIND_STARTUP_DEFAULT,
    OBS_SOURCE_CODES,
    DecodedCommand,
    WireError,
    decode,
    wire_label,
)
from .deploy_contract import WALKING_ENABLED
from .go2_model import (
    LIMIT_ABORT_BAND_RAD,
    POLICY_JOINT_ORDER,
    TRAINING_DEFAULT_JOINT_POS,
    UNITREE_MOTOR_ORDER,
    limits_in_order,
    verify_joint_model,
)
from .motor_crc import PHOENIX_FOR_MOTOR
from .safety import MAX_DELTA_PER_STEP_RAD, estop_is_active, per_step_clip_array

#: Node names of the two real deadman adapters in this package
#: (``wireless_estop_node`` and ``deadman_joy_node``). Nothing else may arm a
#: live bridge.
REAL_DEADMAN_NODE_NAMES: frozenset[str] = frozenset({"phoenix_wireless_estop", "phoenix_deadman"})

_ZEROS_CODE = float(OBS_SOURCE_CODES["zeros"])


class Mode(str, Enum):
    SILENT = "silent"
    HOLD = "hold"
    DAMP = "damp"
    POLICY = "policy"
    STANDUP = "standup"


@dataclass(frozen=True)
class GateParams:
    live: bool
    kp: float
    kd: float
    hold_kp: float
    hold_kd: float
    watchdog_s: float
    estop_timeout_s: float
    lowstate_timeout_s: float
    stale_hold_s: float
    first_message_timeout_s: float
    joint_order: tuple[str, ...] = POLICY_JOINT_ORDER
    max_delta: float = MAX_DELTA_PER_STEP_RAD
    limit_abort_band: float = LIMIT_ABORT_BAND_RAD
    #: ``None`` means "required exactly when live". A live gate cannot opt out.
    require_real_deadman: bool | None = None
    #: 0 disables the standup ramp (the pre-2026-09-21 behaviour).
    standup_s: float = 0.0
    standup_kp: float = 60.0
    standup_kd: float = 5.0

    def __post_init__(self) -> None:
        for name in (
            "watchdog_s",
            "estop_timeout_s",
            "lowstate_timeout_s",
            "first_message_timeout_s",
            "max_delta",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite, got {value}")
        for name in (
            "kp",
            "kd",
            "hold_kp",
            "hold_kd",
            "stale_hold_s",
            "limit_abort_band",
            "standup_s",
            "standup_kp",
            "standup_kd",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite, got {value}")
        if self.live and self.require_real_deadman is False:
            raise ValueError("a live gate cannot disable the real-deadman requirement")

    @property
    def deadman_required(self) -> bool:
        return self.live if self.require_real_deadman is None else bool(self.require_real_deadman)


def _names(mask: np.ndarray) -> str:
    return ",".join(n for n, m in zip(UNITREE_MOTOR_ORDER, mask, strict=True) if m)


def _floats(arr: np.ndarray | None) -> list[float] | None:
    return None if arr is None else [float(v) for v in arr]


class ActuatorGate:
    """See the module docstring. All timestamps are monotonic nanoseconds."""

    def __init__(self, params: GateParams, started_ns: int) -> None:
        problems = verify_joint_model(params.joint_order, PHOENIX_FOR_MOTOR)
        if problems:
            raise ValueError("joint model check failed: " + "; ".join(problems))
        self.params = params
        self._perm = np.asarray(PHOENIX_FOR_MOTOR, dtype=np.int64)
        self._lo, self._hi = limits_in_order(UNITREE_MOTOR_ORDER)
        self._label = wire_label(params.joint_order)
        self._started_ns = int(started_ns)

        self._q: np.ndarray | None = None
        self._dq: np.ndarray | None = None
        self._lowstate_ns: int | None = None
        self._stale_since_ns: int | None = None

        self._estop_value: bool | None = None
        self._estop_ns: int | None = None
        self._publishers: tuple[str, ...] | None = None
        self._deadman_ok_once = False

        self._cmd: DecodedCommand | None = None
        self._cmd_ns: int | None = None
        self._last_decoded: DecodedCommand | None = None
        self._last_processed_seq: int | None = None
        self._startup_default_seen = False

        self._stand = np.asarray(
            [TRAINING_DEFAULT_JOINT_POS[n] for n in UNITREE_MOTOR_ORDER], dtype=np.float64
        )
        if np.any(self._stand < self._lo) or np.any(self._stand > self._hi):
            raise ValueError("training default stance is outside the hard joint limits")
        self._standup_q0: np.ndarray | None = None
        self._standup_start_ns: int | None = None
        self._standup_done = False

        self._faults: list[str] = []
        self._damp_latched = False
        self._tick = 0
        self.counters = {"commands_rejected": 0, "lowstate_rejected": 0, "commands_accepted": 0}

    # ---------------------------------------------------------------- state
    @property
    def expected_label(self) -> str:
        return self._label

    @property
    def fault(self) -> str | None:
        return self._faults[0] if self._faults else None

    @property
    def faults(self) -> tuple[str, ...]:
        return tuple(self._faults)

    @property
    def damp_latched(self) -> bool:
        return self._damp_latched

    def _latch(self, reason: str, *, damp: bool = False) -> None:
        if reason not in self._faults:
            self._faults.append(reason)
        if damp:
            self._damp_latched = True

    # --------------------------------------------------------------- inputs
    def on_lowstate(
        self, now_ns: int, q_unitree: Sequence[float], dq_unitree: Sequence[float]
    ) -> None:
        q = np.asarray(q_unitree, dtype=np.float64).reshape(-1)
        dq = np.asarray(dq_unitree, dtype=np.float64).reshape(-1)
        if q.shape != (12,) or dq.shape != (12,):
            self.counters["lowstate_rejected"] += 1
            self._latch(f"lowstate_bad_shape:{q.shape}", damp=True)
            return
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq))):
            self.counters["lowstate_rejected"] += 1
            self._latch("lowstate_non_finite", damp=True)
            return
        self._q = q
        self._dq = dq
        self._lowstate_ns = int(now_ns)
        band = self.params.limit_abort_band
        impossible = (q < self._lo - band) | (q > self._hi + band)
        if impossible.any():
            self._latch(f"measured_q_impossible:{_names(impossible)}", damp=True)

    def request_shutdown(self) -> None:
        """Latch damping. The bridge calls this on SIGINT so that the last command
        the motors received is ``kp=0`` damping, not a stiff hold of a stale pose."""
        self._latch("bridge_shutdown", damp=True)

    def on_estop(self, now_ns: int, value: bool) -> None:
        self._estop_value = bool(value)
        self._estop_ns = int(now_ns)

    def on_estop_publishers(self, names: Sequence[str]) -> None:
        self._publishers = tuple(sorted(str(n) for n in names))

    def on_command(self, now_ns: int, label: str, data: Sequence[float]) -> None:
        try:
            cmd = decode(label, data, self._label)
        except WireError as exc:
            self.counters["commands_rejected"] += 1
            self._latch(f"command_rejected:{exc.code}")
            return
        self._last_decoded = cmd
        if cmd.kind == KIND_ABORT:
            self._latch(f"policy_abort:{cmd.abort_reason}")
            return
        if cmd.kind == KIND_STARTUP_DEFAULT:
            self._startup_default_seen = True
            return
        f = cmd.fields
        obs_code = f["obs_source_code"][0]
        stand_only = f["stand_only"][0]
        if not np.isfinite(obs_code) or stand_only not in (0.0, 1.0):
            self.counters["commands_rejected"] += 1
            self._latch("command_rejected:missing_observation_provenance")
            return
        # Walking is blocked here too, not only in the deploy contract: a policy
        # node started from a config that skipped validation must still be
        # unable to walk the robot through this bridge.
        if stand_only != 1.0:
            if obs_code == _ZEROS_CODE:
                self._latch("walking_blocked_zero_base_lin_vel")
                return
            if not WALKING_ENABLED:
                self._latch("walking_blocked_not_enabled")
                return
        else:
            moving = [
                v
                for v in (*f["velocity_command_fed"], *f["cmd_vel_received"])
                if np.isfinite(v) and v != 0.0
            ]
            if moving:
                self._latch("walking_blocked_stand_only")
                return
        self._cmd = cmd
        self._cmd_ns = int(now_ns)
        self.counters["commands_accepted"] += 1

    # ----------------------------------------------------------------- tick
    def _elapsed_s(self, now_ns: int) -> float:
        return (now_ns - self._started_ns) / 1e9

    def _estop_state(self, now_ns: int) -> tuple[str, str | None]:
        if self._estop_ns is None:
            if self._elapsed_s(now_ns) <= self.params.first_message_timeout_s:
                return "waiting", "estop_not_yet_seen"
            self._latch("estop_publisher_missing")
            return "latched", "estop_publisher_missing"
        if estop_is_active(
            last_msg_received_ns=self._estop_ns,
            latest_value=self._estop_value,
            now_ns=now_ns,
            timeout_s=self.params.estop_timeout_s,
        ):
            reason = "estop_asserted" if self._estop_value else "estop_heartbeat_stale"
            self._latch(reason)
            return "latched", reason
        return "armed", None

    def _deadman_source_ok(self) -> bool:
        pubs = self._publishers
        return pubs is not None and len(pubs) == 1 and pubs[0] in REAL_DEADMAN_NODE_NAMES

    def _deadman_state(self, now_ns: int) -> tuple[bool, str | None]:
        if not self.params.deadman_required:
            return True, None
        pubs = self._publishers
        if not pubs:
            if self._deadman_ok_once:
                self._latch("deadman_publisher_lost")
                return False, "deadman_publisher_lost"
            if self._elapsed_s(now_ns) <= self.params.first_message_timeout_s:
                return False, "deadman_source_not_yet_discovered"
            self._latch("deadman_source_unverified")
            return False, "deadman_source_unverified"
        if self._deadman_source_ok():
            self._deadman_ok_once = True
            return True, None
        reason = "estop_source_not_a_real_deadman:" + ",".join(pubs)
        self._latch(reason)
        return False, reason

    def tick(self, now_ns: int) -> dict[str, Any]:
        """Decide this tick's motor command. Returns a JSON-ready record.

        ``record["publish"]`` says whether to send a LowCmd; when it is true,
        ``final_target_unitree``, ``kp`` and ``kd`` are that command.
        """
        now_ns = int(now_ns)
        p = self.params
        self._tick += 1
        rec: dict[str, Any] = {
            "t_mono_ns": now_ns,
            "tick": self._tick,
            "live": p.live,
            "mode": Mode.SILENT.value,
            "publish": False,
            "hold_cause": None,
            "fault": None,
            "faults": [],
            "lowstate_age_s": None,
            "lowstate_fresh": False,
            "q_unitree": None,
            "dq_unitree": None,
            "estop_value": self._estop_value,
            "estop_age_s": None if self._estop_ns is None else (now_ns - self._estop_ns) / 1e9,
            "estop_state": None,
            "estop_publishers": None if self._publishers is None else list(self._publishers),
            "deadman_source_ok": self._deadman_source_ok(),
            "cmd_seq": None,
            "cmd_kind": None,
            "cmd_is_new": False,
            "cmd_age_s": None if self._cmd_ns is None else (now_ns - self._cmd_ns) / 1e9,
            "requested_target_unitree": None,
            "final_target_unitree": None,
            "kp": None,
            "kd": None,
            "slew_clip": None,
            "slew_margin": None,
            "limit_clip": None,
            "limit_margin": None,
            "policy": None if self._last_decoded is None else self._last_decoded.telemetry(),
            "standup_alpha": None,
            "standup_done": self._standup_done,
            "counters": dict(self.counters),
        }
        if self._last_decoded is not None:
            rec["cmd_seq"] = self._last_decoded.seq
            rec["cmd_kind"] = self._last_decoded.kind

        if self._q is None:
            rec["faults"] = list(self._faults)
            rec["fault"] = self.fault
            return rec

        q = self._q
        age_s = (
            (now_ns - self._lowstate_ns) / 1e9 if self._lowstate_ns is not None else float("inf")
        )
        fresh = age_s <= p.lowstate_timeout_s
        rec["lowstate_age_s"] = age_s
        rec["lowstate_fresh"] = fresh
        rec["q_unitree"] = _floats(q)
        rec["dq_unitree"] = _floats(self._dq)
        if fresh:
            self._stale_since_ns = None
        else:
            if self._stale_since_ns is None:
                self._stale_since_ns = now_ns
            self._latch("lowstate_stale")
            if (now_ns - self._stale_since_ns) / 1e9 > p.stale_hold_s:
                self._damp_latched = True

        estop_state, estop_cause = self._estop_state(now_ns)
        rec["estop_state"] = estop_state
        deadman_ok, deadman_cause = self._deadman_state(now_ns)

        mode = Mode.HOLD
        cause: str | None = None
        if self._damp_latched:
            mode = Mode.DAMP
            cause = self.fault
        elif self._faults:
            cause = self.fault
        elif estop_state != "armed":
            cause = estop_cause
        elif not deadman_ok:
            cause = deadman_cause
        elif self._cmd is None or self._cmd_ns is None:
            cause = "startup_default" if self._startup_default_seen else "no_policy_command"
        elif (now_ns - self._cmd_ns) / 1e9 > p.watchdog_s:
            cause = "command_stale"
        elif p.standup_s > 0 and not self._standup_done:
            cause = "standup_in_progress"
        else:
            mode = Mode.POLICY

        standup_ok = (
            p.standup_s > 0
            and mode is Mode.HOLD
            and cause in ("no_policy_command", "startup_default", "standup_in_progress")
        )
        if not standup_ok:
            self._standup_q0 = None
            self._standup_start_ns = None
            if mode is not Mode.POLICY:
                self._standup_done = False
        else:
            if self._standup_q0 is None or self._standup_start_ns is None:
                self._standup_q0 = np.clip(q, self._lo, self._hi)
                self._standup_start_ns = now_ns
            alpha = min(1.0, (now_ns - self._standup_start_ns) / 1e9 / p.standup_s)
            if alpha >= 1.0:
                self._standup_done = True
            mode = Mode.STANDUP
            cause = None
            standup_target = (1.0 - alpha) * self._standup_q0 + alpha * self._stand
            rec["standup_alpha"] = float(alpha)

        if mode is Mode.STANDUP:
            final = np.clip(standup_target, self._lo, self._hi)
            rec["limit_clip"] = [bool(v) for v in final != standup_target]
            kp, kd = p.standup_kp, p.standup_kd

        if mode is Mode.POLICY:
            cmd = self._cmd
            assert cmd is not None  # POLICY is only reachable with a command
            requested = cmd.target[self._perm]
            rec["requested_target_unitree"] = _floats(requested)
            beyond = (requested < self._lo - p.limit_abort_band) | (
                requested > self._hi + p.limit_abort_band
            )
            if beyond.any():
                self._latch(f"target_beyond_limit:{_names(beyond)}")
                mode = Mode.HOLD
                cause = self.fault
            else:
                slewed = np.asarray(
                    per_step_clip_array(requested, q, p.max_delta), dtype=np.float64
                )
                final = np.clip(slewed, self._lo, self._hi)
                rec["slew_clip"] = [bool(v) for v in slewed != requested]
                rec["slew_margin"] = _floats(p.max_delta - np.abs(requested - q))
                rec["limit_clip"] = [bool(v) for v in final != slewed]
                rec["cmd_is_new"] = cmd.seq != self._last_processed_seq
                self._last_processed_seq = cmd.seq
                rec["cmd_seq"] = cmd.seq
                rec["cmd_kind"] = cmd.kind
                rec["policy"] = cmd.telemetry()
                kp, kd = p.kp, p.kd

        if mode not in (Mode.POLICY, Mode.STANDUP):
            final = np.clip(q, self._lo, self._hi)
            rec["limit_clip"] = [bool(v) for v in final != q]
            kp, kd = (0.0, p.hold_kd) if mode is Mode.DAMP else (p.hold_kp, p.hold_kd)

        if not np.all(np.isfinite(final)):  # unreachable by construction; fail closed anyway
            self._latch("non_finite_final_target", damp=True)
            final = np.clip(np.nan_to_num(q), self._lo, self._hi)
            mode, kp, kd = Mode.DAMP, 0.0, p.hold_kd

        rec["mode"] = mode.value
        rec["publish"] = True
        rec["hold_cause"] = cause if mode is Mode.HOLD or mode is Mode.DAMP else None
        rec["standup_done"] = self._standup_done
        rec["final_target_unitree"] = _floats(final)
        rec["limit_margin"] = _floats(np.minimum(final - self._lo, self._hi - final))
        rec["kp"] = float(kp)
        rec["kd"] = float(kd)
        rec["faults"] = list(self._faults)
        rec["fault"] = self.fault
        rec["counters"] = dict(self.counters)
        return rec


__all__ = ["REAL_DEADMAN_NODE_NAMES", "ActuatorGate", "GateParams", "Mode"]
