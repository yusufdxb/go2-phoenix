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
``fall_soft`` the IMU reports tilt beyond ``sit.fall_tilt_rad`` (60 deg): the robot
            is down. Low stiffness (``sit.fall_kp`` / ``fall_kd``, never zero) at the
            measured posture so it does not flail. Latched. Highest precedence.
``sit``     the controlled sit (:mod:`phoenix.sim2real.sit_policy`), latched. Open
            loop from the last good q when the measured posture cannot be trusted
            (LowState non-finite, physically impossible q, or stale beyond
            ``stale_hold_s``); closed loop after ``sit.fault_hold_s`` of hold for
            every other latched fault (estop, policy abort, NaN or over-limit
            command, stale command, SIGINT/SIGTERM). Gains ``sit.sit_kp`` /
            ``sit_kd``; targets rate-limited and torque-clipped; ends holding the
            folded pose. ``kp=0`` is never commanded in any mode.
``folded``  shutdown only: the sit has finished and settled; hold it at the low
            ``sit.folded_kp`` until the process exits.
``hold``    hold the last measured posture with the hold gains. Entered for any
            latched fault (estop, ordering mismatch, NaN command, target beyond
            the abort band, policy abort, stale LowState while still inside the
            stale window) for at most ``sit.fault_hold_s``, after which the
            closed-loop sit takes over; and, NOT latched, for a stale or absent
            policy command (stale longer than ``watchdog_s + sit.fault_hold_s``
            after a command had been followed latches the sit as well).
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
  last posture for at most ``stale_hold_s``, then sits open loop.
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
from typing import Any, cast

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
from .go2_model import (
    LIMIT_ABORT_BAND_RAD,
    POLICY_JOINT_ORDER,
    TRAINING_DEFAULT_JOINT_POS,
    UNITREE_MOTOR_ORDER,
    limits_in_order,
    torque_limits_in_order,
    verify_joint_model,
)
from .motor_crc import PHOENIX_FOR_MOTOR
from .safety import (
    DEFAULT_POSITION_CLIP_MARGIN_RAD,
    DEFAULT_SUSTAINED_CLIP_TICKS,
    MAX_DELTA_PER_STEP_RAD,
    SustainedClipTracker,
    clip_to_hard_limits_array,
    estop_is_active,
    expected_torque,
    torque_limited_target_array,
)
from .sit_policy import (
    SIT_POSE_UNITREE,
    SitParams,
    SitTrajectory,
    fall_tilt_rad,
    rpy_tilt_rad,
    valid_unit_quat,
)

#: Node names of the two real deadman adapters in this package
#: (``wireless_estop_node`` and ``deadman_joy_node``). Nothing else may arm a
#: live bridge.
REAL_DEADMAN_NODE_NAMES: frozenset[str] = frozenset({"phoenix_wireless_estop", "phoenix_deadman"})

_ZEROS_CODE = float(OBS_SOURCE_CODES["zeros"])


class Mode(str, Enum):
    SILENT = "silent"
    HOLD = "hold"
    SIT = "sit"
    FALL_SOFT = "fall_soft"
    FOLDED = "folded"
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
    #: Kept for config back-compat and validated for finiteness/positivity, but
    #: POLICY-mode clipping no longer uses it: :meth:`ActuatorGate.tick` clips
    #: on torque via ``go2_model.JOINT_TORQUE_LIMITS_NM``
    #: (see ``safety.torque_limited_target_array``), not on this position delta.
    max_delta: float = MAX_DELTA_PER_STEP_RAD
    limit_abort_band: float = LIMIT_ABORT_BAND_RAD
    #: ``None`` means "required exactly when live". A live gate cannot opt out.
    require_real_deadman: bool | None = None
    #: 0 disables the standup ramp (the pre-2026-09-21 behaviour).
    standup_s: float = 0.0
    standup_kp: float = 60.0
    standup_kd: float = 5.0
    #: Length of the linear kp blend from the stand-up kp to the policy kp, starting on
    #: the first POLICY tick after the stand-up. Without it kp steps 60 -> 25 in one
    #: 20 ms tick, a torque step of about (kp_standup - kp_policy) x tracking error.
    #: 0 restores the step. Only meaningful when ``standup_s > 0``. kp stays strictly
    #: positive across the whole blend. kd is NOT blended: it steps from standup_kd to
    #: the policy kd on the first POLICY tick (2026-10-04 desk gate, frozen for hardware
    #: A-G). At the end of the stand-up ramp dq is about 0, so the kd step changes torque
    #: by about (kd_standup - kd_policy) x dq, i.e. almost nothing, while holding the
    #: policy under kd up to 5 is what aborted: m2750 in sim, 5 seeds x 2000 envs on the
    #: training DR, the linear 60/5 -> 25/0.5 blend aborted 2.9 % (falls 0.25 %), a
    #: staged 60/5 -> 25/5 -> 25/0.5 blend 15.3 % (falls 2.4 %), this kd step 0 % (0
    #: falls), equal to the linear blend on both actuator corners
    #: (prehardware results exploratory/handover_staged/pooled.json).
    handover_s: float = 1.0
    #: 2026-09-25: a PD controller commanding a target PAST a hard stop is
    #: normal (pressing against it); proven stacks do not abort on it. The
    #: commanded target is clipped into the hard range with this small inward
    #: margin instead of latching a fault. See safety.DEFAULT_POSITION_CLIP_MARGIN_RAD.
    position_clip_margin: float = DEFAULT_POSITION_CLIP_MARGIN_RAD
    #: Consecutive ticks a joint may be clipped before that is treated as a
    #: genuinely broken policy rather than a normal brief press. See
    #: safety.DEFAULT_SUSTAINED_CLIP_TICKS.
    sustained_clip_ticks: int = DEFAULT_SUSTAINED_CLIP_TICKS
    #: The zero-stiffness-free fault policy (controlled sit, FALL_SOFT, folded hold).
    sit: SitParams = SitParams()
    #: The policy may not arm until a valid IMU quaternion has arrived (fall detection
    #: is otherwise blind). Only tests of unrelated behaviour opt out; BridgeConfig
    #: never exposes this, so no CLI/config path can disable it.
    require_imu: bool = True
    #: Walking commands are refused unless this is True. Nothing in the bridge CLI sets
    #: it: a caller must derive it from :func:`deploy_contract.walking_evidence_problems`
    #: (stages F and G on hardware for the same artifacts) before constructing the gate.
    walking_allowed: bool = False
    #: The deploy manifest declares ``phoenix-velocity-obs/v2`` (45-D, no base_lin_vel
    #: term), so a ``zeros`` observation source is an inert placeholder rather than a
    #: blinded trained input. Derived by the bridge from the manifest; never a CLI flag.
    obs_schema_v2: bool = False

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
        # Zero stiffness is never a legal gain (owner rule 2026-10-03).
        for name in ("kp", "hold_kp"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be positive and finite (kp=0 is banned), got {value}"
                )
        if self.standup_s > 0 and self.standup_kp <= 0:
            raise ValueError("standup_kp must be positive when standup_s > 0 (kp=0 is banned)")
        for name in (
            "kd",
            "hold_kd",
            "stale_hold_s",
            "limit_abort_band",
            "standup_s",
            "standup_kp",
            "standup_kd",
            "handover_s",
            "position_clip_margin",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite, got {value}")
        if self.sustained_clip_ticks < 1:
            raise ValueError(f"sustained_clip_ticks must be >= 1, got {self.sustained_clip_ticks}")
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
        self._torque_limit = torque_limits_in_order(UNITREE_MOTOR_ORDER)
        self._clip_tracker = SustainedClipTracker(12, params.sustained_clip_ticks)
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
        # Handover (stand-up gains -> policy gains) bookkeeping.
        self._handover_start_ns: int | None = None
        #: Gains published on the previous tick, and the (kp, kd) a controlled sit is
        #: ramping down from when it starts above the sit gains (fault hold at 60/5).
        self._prev_gains: tuple[float, float] | None = None
        self._sit_ramp: tuple[float, float] | None = None
        #: (kp, kd) in effect on the last STANDUP tick or still-blending POLICY tick;
        #: a fault latched while this is set holds at those gains (never below hold gains).
        self._entry_gains: tuple[float, float] | None = None
        self._fault_hold_gains: tuple[float, float] | None = None

        self._faults: list[str] = []
        self._sit_open = False  # measured posture untrusted: open-loop sit
        self._sit_closed = False  # trusted posture: closed-loop sit
        self._fall = False
        self._imu_valid = False
        self._imu_valid_ns: int | None = None
        self._tilt: float | None = None
        self._tilt_run = 0
        self._tilt_run_start_ns: int | None = None
        self._shutdown = False
        self._fault_since_ns: int | None = None
        self._q_good: np.ndarray | None = None
        self._fault_hold_q: np.ndarray | None = None  # posture latched at hold entry
        self._last_final: np.ndarray | None = None
        self._traj = SitTrajectory(params.sit)
        self._tick = 0
        self.counters = {
            "commands_rejected": 0,
            "lowstate_rejected": 0,
            "commands_accepted": 0,
            "imu_rejected": 0,
            "imu_disagree": 0,
        }

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
    def sit_latched(self) -> bool:
        return self._sit_open or self._sit_closed

    def _latch(self, reason: str, *, sensor_bad: bool = False) -> None:
        if reason not in self._faults:
            self._faults.append(reason)
        if sensor_bad:
            self._sit_open = True

    # --------------------------------------------------------------- inputs
    def on_lowstate(
        self, now_ns: int, q_unitree: Sequence[float], dq_unitree: Sequence[float]
    ) -> None:
        q = np.asarray(q_unitree, dtype=np.float64).reshape(-1)
        dq = np.asarray(dq_unitree, dtype=np.float64).reshape(-1)
        if q.shape != (12,) or dq.shape != (12,):
            self.counters["lowstate_rejected"] += 1
            self._latch(f"lowstate_bad_shape:{q.shape}", sensor_bad=True)
            return
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq))):
            self.counters["lowstate_rejected"] += 1
            self._latch("lowstate_non_finite", sensor_bad=True)
            return
        self._q = q
        self._dq = dq
        self._lowstate_ns = int(now_ns)
        band = self.params.limit_abort_band
        impossible = (q < self._lo - band) | (q > self._hi + band)
        if not impossible.any():
            self._q_good = q
        if impossible.any():
            self._latch(f"measured_q_impossible:{_names(impossible)}", sensor_bad=True)

    def request_shutdown(self, now_ns: int | None = None) -> None:
        """Start the shutdown sit (no hold first). The bridge calls this on
        SIGINT/SIGTERM; the sequence is sit, settle, folded low-kp hold, bounded
        by ``params.sit.shutdown_bound_s``. ``kp=0`` is never sent."""
        self._latch("bridge_shutdown")
        self._shutdown = True
        self._sit_closed = True

    def on_imu(
        self, now_ns: int, quat_wxyz: Sequence[float], rpy: Sequence[float] | None = None
    ) -> None:
        """Feed the IMU orientation, quaternion order (w, x, y, z) (Unitree
        ``imu_state.quaternion``; see ``read_low_state.cpp``: qw, qx, qy, qz).

        An invalid quaternion (malformed, non-finite, zero or not unit-norm) is
        counted and ignored: it is neither a fall nor proof of a working IMU. A fall
        latches FALL_SOFT only when the quaternion tilt exceeds ``sit.fall_tilt_rad``
        for ``sit.fall_debounce_samples`` consecutive samples spanning at least
        ``sit.fall_debounce_s``, and the ``rpy`` roll/pitch tilt (when available)
        agrees. A single spurious sample never latches.
        """
        sit = self.params.sit
        q = valid_unit_quat(quat_wxyz, sit.imu_unit_tol)
        if q is None:
            self.counters["imu_rejected"] += 1
            self._imu_valid = False
            return
        now_ns = int(now_ns)
        self._imu_valid = True
        self._imu_valid_ns = now_ns
        tilt = fall_tilt_rad(q)
        self._tilt = tilt
        over = tilt is not None and tilt > sit.fall_tilt_rad
        if over:
            tilt_rpy = rpy_tilt_rad(rpy)
            if tilt_rpy is not None and tilt_rpy <= sit.fall_tilt_rad:
                self.counters["imu_disagree"] += 1
                over = False
        if not over:
            self._tilt_run = 0
            self._tilt_run_start_ns = None
            return
        if self._tilt_run == 0:
            self._tilt_run_start_ns = now_ns
        self._tilt_run += 1
        start = self._tilt_run_start_ns if self._tilt_run_start_ns is not None else now_ns
        if (
            self._tilt_run >= sit.fall_debounce_samples
            and (now_ns - start) / 1e9 >= sit.fall_debounce_s
        ):
            self._latch(f"fall_detected:tilt_{tilt:.2f}rad")
            self._fall = True

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
            if obs_code == _ZEROS_CODE and not self.params.obs_schema_v2:
                self._latch("walking_blocked_zero_base_lin_vel")
                return
            if not self.params.walking_allowed:
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
            "torque_margin_nm": None,
            "kp_eff": None,
            "kd_eff": None,
            "limit_clip": None,
            "limit_margin": None,
            "policy": None if self._last_decoded is None else self._last_decoded.telemetry(),
            "standup_alpha": None,
            "handover_alpha": None,
            "standup_done": self._standup_done,
            "counters": dict(self.counters),
            "tilt_deg": None if self._tilt is None else float(np.degrees(self._tilt)),
            "imu_valid": self._imu_valid,
            "imu_seen_valid": self._imu_valid_ns is not None,
            "imu_age_s": (
                None if self._imu_valid_ns is None else (now_ns - self._imu_valid_ns) / 1e9
            ),
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
                self._sit_open = True

        if (
            p.require_imu
            and self._imu_valid_ns is not None
            and (now_ns - self._imu_valid_ns) / 1e9 > p.sit.imu_stale_s
        ):
            self._latch("imu_stale")  # hold, then the closed-loop sit (not an open-loop one)

        estop_state, estop_cause = self._estop_state(now_ns)
        rec["estop_state"] = estop_state
        deadman_ok, deadman_cause = self._deadman_state(now_ns)

        # A command that had been followed and then went silent for longer than the
        # watchdog plus the hold window ends in the closed-loop sit.
        if (
            self._cmd_ns is not None
            and not self._sit_closed
            and (now_ns - self._cmd_ns) / 1e9 > p.watchdog_s + p.sit.fault_hold_s
        ):
            self._latch("command_stale_timeout")
            self._sit_closed = True
        # Any other latched fault holds for at most sit.fault_hold_s, then sits.
        if self._faults and not self._sit_closed and not self._sit_open:
            if self._fault_since_ns is None:
                self._fault_since_ns = now_ns
            if (now_ns - self._fault_since_ns) / 1e9 >= p.sit.fault_hold_s:
                self._sit_closed = True

        mode = Mode.HOLD
        cause: str | None = None
        if self._fall:
            mode = Mode.FALL_SOFT
            cause = self.fault
        elif self._sit_open or self._sit_closed:
            mode = Mode.SIT
            cause = self.fault
        elif self._faults:
            cause = self.fault
        elif estop_state != "armed":
            cause = estop_cause
        elif not deadman_ok:
            cause = deadman_cause
        elif p.require_imu and self._imu_valid_ns is None:
            cause = "imu_not_valid_yet"  # refuse to arm blind: no fall detection
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

        handover_alpha: float | None = None
        gain_kp, gain_kd = p.kp, p.kd
        if mode is Mode.POLICY:
            cmd = self._cmd
            assert cmd is not None  # POLICY is only reachable with a command
            if p.standup_s > 0 and p.handover_s > 0:
                if self._handover_start_ns is None:
                    self._handover_start_ns = now_ns
                handover_alpha = min(1.0, (now_ns - self._handover_start_ns) / 1e9 / p.handover_s)
                gain_kp = p.standup_kp + (p.kp - p.standup_kp) * handover_alpha
                gain_kd = p.kd  # kd steps on the first POLICY tick (see handover_s)
                rec["handover_alpha"] = float(handover_alpha)
            requested = cmd.target[self._perm]
            rec["requested_target_unitree"] = _floats(requested)
            non_finite = ~np.isfinite(requested)
            if non_finite.any():
                # A NaN/Inf commanded target is the one requested-target
                # condition that still latches: it cannot come from any
                # legal policy output, clamped or not.
                self._latch(f"command_non_finite:{_names(non_finite)}")
                mode = Mode.HOLD
                cause = self.fault
            else:
                dq = self._dq if self._dq is not None else np.zeros(12, dtype=np.float64)
                torque_clipped = np.asarray(
                    torque_limited_target_array(
                        requested, q, dq, gain_kp, gain_kd, self._torque_limit
                    ),
                    dtype=np.float64,
                )
                # Key kept as "slew_clip" for backward compatibility with
                # bridge_telemetry.py's hardware slew metric and existing
                # stage evidence, but the semantics changed 2026-09-24: this
                # now flags a TORQUE-based clip (go2_model.JOINT_TORQUE_LIMITS_NM),
                # not the old measured-q +-0.175 rad/step position slew.
                rec["slew_clip"] = [bool(v) for v in torque_clipped != requested]
                expected_tau = expected_torque(torque_clipped, q, dq, gain_kp, gain_kd)
                rec["torque_margin_nm"] = _floats(self._torque_limit - np.abs(expected_tau))
                # 2026-09-25: a target beyond the hard stop is now CLIPPED,
                # not an abort trigger (a PD controller pressing into a stop
                # is normal; proven stacks do not abort on it either -- see
                # safety.DEFAULT_POSITION_CLIP_MARGIN_RAD). Sustained clipping
                # on any one joint IS still an abort trigger: that signals a
                # policy that cannot be steered off the stop, not a brief press.
                final, position_clipped = clip_to_hard_limits_array(
                    torque_clipped, self._lo, self._hi, p.position_clip_margin
                )
                rec["limit_clip"] = [bool(v) for v in position_clipped]
                sustained_now = self._clip_tracker.update(position_clipped)
                rec["sustained_clip_counts"] = [int(v) for v in self._clip_tracker.counts]
                if sustained_now.any():
                    self._latch(f"sustained_clip_exceeded:{_names(sustained_now)}")
                    mode = Mode.HOLD
                    cause = self.fault
                    # final/kp/kd for this tick are set by the common
                    # "mode not in (POLICY, STANDUP)" branch below, same as
                    # every other latched-this-tick path.
                else:
                    rec["cmd_is_new"] = cmd.seq != self._last_processed_seq
                    self._last_processed_seq = cmd.seq
                    rec["cmd_seq"] = cmd.seq
                    rec["cmd_kind"] = cmd.kind
                    rec["policy"] = cmd.telemetry()
                    kp, kd = gain_kp, gain_kd

        if mode is Mode.SIT:
            sensors_ok = not self._sit_open and fresh
            if not self._traj.started:
                pg = self._prev_gains
                self._sit_ramp = (pg[0], pg[1]) if pg is not None and pg[0] > p.sit.sit_kp else None
            if not self._traj.started:
                # Open loop starts from the last COMMANDED target (no command jump);
                # closed loop from the measured posture.
                if sensors_ok or (self._last_final is None and self._q_good is None):
                    q_start = q
                else:
                    # Reached only when at least one of the two is set (see condition above).
                    q_start = cast(
                        np.ndarray,
                        self._last_final if self._last_final is not None else self._q_good,
                    )
                self._traj.start(q_start, now_ns)
            final_opt, _done = self._traj.step(
                now_ns, q if sensors_ok else None, self._dq if sensors_ok else None
            )
            final = cast(np.ndarray, final_opt)
            rec["limit_clip"] = [False] * 12
            rec["sit_open_loop"] = not sensors_ok
            kp, kd = p.sit.sit_kp, p.sit.sit_kd
            if self._sit_ramp is not None and self._prev_gains is not None:
                # A sit that starts above the sit gains (a fault hold at the 60/5 stand-up or
                # handover gains) blends down at no more than SIT_GAIN_RAMP_KP_PER_TICK per tick
                # instead of stepping 60/5 -> 30/1.5 in one 20 ms tick; kd follows kp linearly.
                kp0, kd0 = self._sit_ramp
                kp = max(p.sit.sit_kp, self._prev_gains[0] - SIT_GAIN_RAMP_KP_PER_TICK)
                kd = p.sit.sit_kd + (kp - p.sit.sit_kp) / (kp0 - p.sit.sit_kp) * (
                    kd0 - p.sit.sit_kd
                )
                if kp <= p.sit.sit_kp:
                    self._sit_ramp = None
                if sensors_ok:
                    dq_ramp = self._dq if self._dq is not None else np.zeros(12, dtype=np.float64)
                    final = np.asarray(
                        torque_limited_target_array(final, q, dq_ramp, kp, kd, self._torque_limit),
                        dtype=np.float64,
                    )
            done_ns = self._traj.done_ns
            if (
                self._shutdown
                and done_ns is not None
                and (now_ns - done_ns) / 1e9 >= p.sit.folded_settle_s
            ):
                mode = Mode.FOLDED
                kp, kd = p.sit.folded_kp, p.sit.folded_kd
        elif mode is Mode.FALL_SOFT:
            base = self._q_good if self._q_good is not None else q
            final = np.clip(base, self._lo, self._hi)
            rec["limit_clip"] = [bool(v) for v in final != base]
            kp, kd = p.sit.fall_kp, p.sit.fall_kd
        elif mode not in (Mode.POLICY, Mode.STANDUP):
            # A fault / stale-command hold targets the posture measured ONCE at entry.
            # Re-targeting the measured q every tick makes the hold a ratchet: the
            # restoring term vanishes and the robot sinks at about load / kd.
            if self._faults or cause == "command_stale":
                if self._fault_hold_q is None:
                    self._fault_hold_q = np.clip(q, self._lo, self._hi)
                    # A fault during the stand-up or the gain handover holds at the gains
                    # in effect at fault entry (never below the hold gains): dropping to
                    # the hold gains mid-rise is a torque step while the robot is least
                    # able to take one.
                    eg = self._entry_gains
                    self._fault_hold_gains = (
                        (max(eg[0], p.hold_kp), max(eg[1], p.hold_kd))
                        if eg is not None
                        else (p.hold_kp, p.hold_kd)
                    )
                base = self._fault_hold_q
                assert self._fault_hold_gains is not None
                hold_gains = self._fault_hold_gains
            else:
                self._fault_hold_q = None
                self._fault_hold_gains = None
                self._entry_gains = None
                base = q
                hold_gains = (p.hold_kp, p.hold_kd)
            final = np.clip(base, self._lo, self._hi)
            rec["limit_clip"] = [bool(v) for v in final != base]
            kp, kd = hold_gains

        if not np.all(np.isfinite(final)):  # unreachable by construction; fail closed anyway
            self._latch("non_finite_final_target")
            final = np.asarray(SIT_POSE_UNITREE, dtype=np.float64)
            mode, kp, kd = Mode.SIT, p.sit.sit_kp, p.sit.sit_kd

        if mode is Mode.STANDUP:
            self._entry_gains = (float(kp), float(kd))
        elif mode is Mode.POLICY:
            blending = handover_alpha is not None and handover_alpha < 1.0
            self._entry_gains = (float(kp), float(kd)) if blending else None
        if mode is not Mode.POLICY:
            self._handover_start_ns = None
        rec["mode"] = mode.value
        rec["publish"] = True
        rec["hold_cause"] = cause if mode not in (Mode.POLICY, Mode.STANDUP) else None
        rec["standup_done"] = self._standup_done
        rec["final_target_unitree"] = _floats(final)
        self._last_final = np.asarray(final, dtype=np.float64).copy()
        rec["limit_margin"] = _floats(np.minimum(final - self._lo, self._hi - final))
        rec["kp"] = float(kp)
        rec["kd"] = float(kd)
        # Stage F evidence: the gains actually published and the torque margin of the
        # published target (final, after every clip) on EVERY tick, in every mode.
        rec["kp_eff"] = float(kp)
        rec["kd_eff"] = float(kd)
        dq_pub = self._dq if self._dq is not None else np.zeros(12, dtype=np.float64)
        tau_pub = np.asarray(expected_torque(final, q, dq_pub, kp, kd), dtype=np.float64)
        if np.all(np.isfinite(tau_pub)):
            rec["torque_margin_nm"] = _floats(self._torque_limit - np.abs(tau_pub))
        self._prev_gains = (float(kp), float(kd))
        rec["faults"] = list(self._faults)
        rec["fault"] = self.fault
        rec["counters"] = dict(self.counters)
        return rec


#: Largest per-tick kp change while a controlled sit that starts above the sit gains
#: blends down to them: the handover bound (60 -> 25 over 1 s at 50 Hz = 0.7 per tick).
SIT_GAIN_RAMP_KP_PER_TICK: float = 0.7

__all__ = ["REAL_DEADMAN_NODE_NAMES", "ActuatorGate", "GateParams", "Mode"]
