"""The deployment state machine: rise deterministically, prove the stance, then hand over.

States::

    IDLE -> PRECHECK -> FIX_STAND -> FIX_STAND_HOLD -> VELOCITY_READY -> VELOCITY_ACTIVE
                                                                               |
                                                                           STOPPING
    FAULT is reachable from every state except IDLE (and is terminal).

Pure, deterministic and tick-driven: :meth:`ControllerFSM.tick` takes one
:class:`FsmInputs` (measured state and its ages, deadman / e-stop, the command) and
returns one :class:`FsmOutput` (what to send the motors and why). No clock is read
and nothing is published here; a ROS shell or a simulator loop owns time.

The one structural guarantee
----------------------------
The policy callable is invoked in exactly one place, :meth:`_tick_active`, which is
reachable only in ``VELOCITY_ACTIVE``. ``VELOCITY_ACTIVE`` is entered only from
``VELOCITY_READY`` on an explicit :meth:`request_policy`, and ``VELOCITY_READY`` is
entered only when :class:`phoenix.sim2real.handoff.HandoffMonitor` reports the
measured stance stable for its whole interval. A FIX_STAND that finished on time
but did not bring the robot into tolerance never reaches the policy. The tests
drive every state with a policy stub that raises if called.

Stage variants
--------------
* ``policy=None`` (stages F1 / F2): no learned component exists in the process;
  ``VELOCITY_READY`` is unreachable and :meth:`request_policy` is refused.
* ``stand_only=True`` (stage F3, H25; and G1, zero command): the command fed to
  the policy is forced to zero and any nonzero command in the inputs is a FAULT.
* velocity (G2+): the command must lie inside the deploy envelope, which the
  caller has already checked against the checkpoint manifest.

Outputs by state
----------------
IDLE, PRECHECK        publish nothing (the motors keep whatever owns them: nothing)
FIX_STAND             the FixStand reference at kp 60 / kd 5
FIX_STAND_HOLD,
VELOCITY_READY        the stand target at the FixStand gains
VELOCITY_ACTIVE       the policy target after the safety filter, at the policy gains
STOPPING              hold the posture measured on entry for ``stop_hold_s``, then damp
FAULT                 damp (kp 0, kd ``damp_kd``) at the measured posture, latched
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .fix_stand import FixStand, FixStandError, FixStandParams
from .handoff import HandoffCriteria, HandoffMonitor, HandoffSample, HandoffStatus
from .intervention import InterventionRecorder, SafetyFilter, tick_record


class State(str, Enum):
    IDLE = "IDLE"
    PRECHECK = "PRECHECK"
    FIX_STAND = "FIX_STAND"
    FIX_STAND_HOLD = "FIX_STAND_HOLD"
    VELOCITY_READY = "VELOCITY_READY"
    VELOCITY_ACTIVE = "VELOCITY_ACTIVE"
    STOPPING = "STOPPING"
    FAULT = "FAULT"


#: ``obs_builder(gyro_body, quat_wxyz, command, joint_pos, joint_vel, last_action)``
ObsBuilder = Callable[..., np.ndarray]
#: ``policy(obs) -> raw action``
Policy = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class FsmParams:
    rate_hz: float = 50.0
    action_scale: float = 0.25
    policy_kp: float = 25.0
    policy_kd: float = 0.5
    hold_kp: float = 20.0
    hold_kd: float = 1.0
    damp_kd: float = 1.0
    sensor_timeout_s: float = 0.1
    precheck_timeout_s: float = 5.0
    #: FIX_STAND_HOLD must reach the handoff criteria within this long, or FAULT.
    handoff_timeout_s: float = 10.0
    #: Policy authority length; 0 means until :meth:`ControllerFSM.request_stop`.
    authority_s: float = 0.0
    stop_hold_s: float = 1.0
    attitude_abort_rad: float = 0.40
    #: Runtime intervention brake (see ``InterventionRecorder.live_budget_exceeded``).
    live_intervention_window: int = 25
    live_intervention_max_fraction: float = 0.5
    #: Largest |command| per axis the deploy allows (vx, vy, wz). Velocity mode only.
    max_command: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name in ("rate_hz", "action_scale", "sensor_timeout_s", "precheck_timeout_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"FsmParams.{name} must be positive and finite")
        if len(self.max_command) != 3 or any(
            (not math.isfinite(float(v))) or float(v) < 0 for v in self.max_command
        ):
            raise ValueError("FsmParams.max_command must be three non-negative magnitudes")


@dataclass(frozen=True)
class FsmInputs:
    """One tick. Joint arrays in the contract order (``JOINT_ORDER``)."""

    t_ns: int
    q: Sequence[float] | None
    dq: Sequence[float] | None
    gyro_body: Sequence[float] | None
    quat_wxyz: Sequence[float] | None
    lowstate_age_s: float | None
    imu_age_s: float | None
    estop_ok: bool
    deadman_ok: bool
    command: Sequence[float] = (0.0, 0.0, 0.0)


@dataclass
class FsmOutput:
    state: State
    publish: bool
    target: list[float] | None
    kp: float | None
    kd: float | None
    reason: str | None = None
    fault: str | None = None
    fix_stand: dict[str, Any] | None = None
    handoff: dict[str, Any] | None = None
    intervention: dict[str, Any] | None = None
    events: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "publish": self.publish,
            "target": self.target,
            "kp": self.kp,
            "kd": self.kd,
            "reason": self.reason,
            "fault": self.fault,
            "fix_stand": self.fix_stand,
            "handoff": self.handoff,
            "intervention": self.intervention,
            "events": list(self.events),
        }


def _projected_gravity(quat_wxyz: Sequence[float] | None) -> np.ndarray | None:
    if quat_wxyz is None:
        return None
    from phoenix.velocity.observation import ObservationError, projected_gravity_wxyz

    try:
        return projected_gravity_wxyz(quat_wxyz)
    except ObservationError:
        return None


class ControllerFSM:
    """See the module docstring."""

    def __init__(
        self,
        order: Sequence[str],
        *,
        params: FsmParams | None = None,
        fix_stand_params: FixStandParams | None = None,
        handoff_criteria: HandoffCriteria | None = None,
        policy: Policy | None = None,
        obs_builder: ObsBuilder | None = None,
        stand_only: bool = True,
        manifest_problems: Sequence[str] = (),
    ) -> None:
        self.order = tuple(order)
        self.n = len(self.order)
        self.params = params or FsmParams()
        self._fs_params = fix_stand_params or FixStandParams(rate_hz=self.params.rate_hz)
        self.fix_stand = FixStand(self.order, self._fs_params)
        self.handoff = HandoffMonitor(self.order, self.fix_stand.target, handoff_criteria)
        self.safety = SafetyFilter(self.order)
        self.recorder = InterventionRecorder(
            self.params.live_intervention_window, self.params.live_intervention_max_fraction
        )
        self._policy = policy
        if policy is not None and obs_builder is None:
            from phoenix.velocity.observation import build_actor_observation

            obs_builder = build_actor_observation
        self._obs_builder = obs_builder
        self.stand_only = bool(stand_only)
        if not self.stand_only and not any(self.params.max_command):
            raise ValueError("a velocity-mode FSM needs a nonzero max_command envelope")
        self.manifest_problems = tuple(manifest_problems)

        self.state = State.IDLE
        self.fault: str | None = None
        self._state_since_ns: int | None = None
        self._start_requested = False
        self._policy_requested = False
        self._stop_requested = False
        self._last_action = np.zeros(self.n, dtype=np.float64)
        self._stop_q: np.ndarray | None = None
        self._stop_damp = False
        self._authority_start_ns: int | None = None
        self.policy_calls = 0
        self.transitions: list[tuple[int, str, str, str]] = []

    # ------------------------------------------------------------- operator
    def request_start(self) -> None:
        self._start_requested = True

    def request_policy(self) -> str | None:
        """Ask for the policy. Returns a refusal reason, or None if the request is queued.

        Queued only: the FSM grants it on a later tick, and only from VELOCITY_READY.
        """
        if self._policy is None:
            return "no_policy_configured"
        self._policy_requested = True
        return None

    def request_stop(self) -> None:
        self._stop_requested = True

    def abort(self, reason: str) -> None:
        """Immediate FAULT (damping) from any state. Idempotent."""
        self._fault(reason, self._state_since_ns or 0)

    # ------------------------------------------------------------ internals
    def _go(self, state: State, t_ns: int, why: str) -> None:
        if state is self.state:
            return
        self.transitions.append((int(t_ns), self.state.value, state.value, why))
        self.state = state
        self._state_since_ns = int(t_ns)

    def _fault(self, reason: str, t_ns: int) -> None:
        if self.state is State.FAULT:
            return
        self.fault = reason
        self.fix_stand.abort(reason)
        self._go(State.FAULT, t_ns, reason)

    def _elapsed(self, t_ns: int) -> float:
        return 0.0 if self._state_since_ns is None else (t_ns - self._state_since_ns) / 1e9

    def _q(self, inp: FsmInputs) -> np.ndarray | None:
        if inp.q is None:
            return None
        q = np.asarray(inp.q, dtype=np.float64).reshape(-1)
        return q if q.shape == (self.n,) and np.all(np.isfinite(q)) else None

    def _fresh(self, inp: FsmInputs) -> str | None:
        p = self.params
        for name, age in (("lowstate", inp.lowstate_age_s), ("imu", inp.imu_age_s)):
            if age is None or not math.isfinite(float(age)) or float(age) > p.sensor_timeout_s:
                return f"{name}_stale"
        return None

    def _damp_output(self, inp: FsmInputs, reason: str | None) -> FsmOutput:
        q = self._q(inp)
        if q is None:
            q = self.fix_stand.q0 if self.fix_stand.q0 is not None else self.fix_stand.target
        target = np.clip(q, self.fix_stand.lo, self.fix_stand.hi)
        return FsmOutput(
            state=self.state,
            publish=True,
            target=[float(v) for v in target],
            kp=0.0,
            kd=self.params.damp_kd,
            reason=reason,
            fault=self.fault,
        )

    def _handoff_sample(self, inp: FsmInputs) -> HandoffSample:
        faults = (self.fault,) if self.fault else ()
        return HandoffSample(
            t_ns=int(inp.t_ns),
            q=list(inp.q) if inp.q is not None else [float("nan")] * self.n,
            dq=list(inp.dq) if inp.dq is not None else [float("nan")] * self.n,
            gyro_body=inp.gyro_body,
            projected_gravity=_projected_gravity(inp.quat_wxyz),
            lowstate_age_s=inp.lowstate_age_s,
            imu_age_s=inp.imu_age_s,
            estop_ok=inp.estop_ok,
            deadman_ok=inp.deadman_ok,
            faults=faults,
            fix_stand_complete=self.fix_stand.complete,
            fix_stand_aborted=self.fix_stand.aborted,
            manifest_problems=self.manifest_problems,
        )

    def _stand_output(self, reason: str | None, status: HandoffStatus | None) -> FsmOutput:
        fs = self._fs_params
        return FsmOutput(
            state=self.state,
            publish=True,
            target=[float(v) for v in self.fix_stand.target],
            kp=fs.kp,
            kd=fs.kd,
            reason=reason,
            handoff=None if status is None else status.to_record(),
        )

    # ----------------------------------------------------------------- tick
    def tick(self, inp: FsmInputs) -> FsmOutput:
        t = int(inp.t_ns)
        if self._state_since_ns is None:
            self._state_since_ns = t
        # Safety inputs that end any engaged state, checked before the state logic.
        if self.state not in (State.IDLE, State.FAULT):
            if not inp.estop_ok:
                self._fault("estop", t)
            elif not inp.deadman_ok:
                self._fault("deadman_released", t)
            elif self.state is not State.PRECHECK and self._q(inp) is None:
                self._fault("measured_q_invalid", t)
            elif self.state is not State.PRECHECK and self._fresh(inp) is not None:
                self._fault(str(self._fresh(inp)), t)
        cmd = np.asarray(inp.command, dtype=np.float64).reshape(-1)
        if self.state not in (State.IDLE, State.FAULT, State.STOPPING):
            if cmd.shape != (3,) or not np.all(np.isfinite(cmd)):
                self._fault("command_invalid", t)
            elif self.stand_only and np.any(cmd != 0.0):
                self._fault("nonzero_command_in_stand_only", t)
            elif not self.stand_only and np.any(
                np.abs(cmd) > np.asarray(self.params.max_command) + 1e-12
            ):
                self._fault("command_outside_envelope", t)

        handler = {
            State.IDLE: self._tick_idle,
            State.PRECHECK: self._tick_precheck,
            State.FIX_STAND: self._tick_fix_stand,
            State.FIX_STAND_HOLD: self._tick_hold,
            State.VELOCITY_READY: self._tick_ready,
            State.VELOCITY_ACTIVE: self._tick_active,
            State.STOPPING: self._tick_stopping,
            State.FAULT: self._tick_fault,
        }[self.state]
        return handler(inp)

    def _tick_idle(self, inp: FsmInputs) -> FsmOutput:
        if self._start_requested:
            self._go(State.PRECHECK, inp.t_ns, "start_requested")
            return self._tick_precheck(inp)
        return FsmOutput(self.state, False, None, None, None, reason="idle")

    def precheck_problems(self, inp: FsmInputs) -> tuple[list[str], list[str]]:
        """``(permanent, transient)`` precheck failures for this tick."""
        permanent = [f"manifest: {p}" for p in self.manifest_problems]
        transient: list[str] = []
        q = self._q(inp)
        if q is None:
            transient.append("measured_q_missing_or_non_finite")
        else:
            band = self._fs_params.start_limit_band_rad
            lo, hi = self.fix_stand.lo, self.fix_stand.hi
            if np.any((q < lo - band) | (q > hi + band)):
                permanent.append("measured_q_beyond_hard_limits")
        fresh = self._fresh(inp)
        if fresh:
            transient.append(fresh)
        g = _projected_gravity(inp.quat_wxyz)
        if g is None:
            transient.append("imu_orientation_missing")
        elif (tilt := float(np.arccos(np.clip(-g[2], -1.0, 1.0)))) > self.params.attitude_abort_rad:
            permanent.append(f"base_tilt_{tilt:.2f}rad_beyond_{self.params.attitude_abort_rad}")
        if not inp.estop_ok:
            transient.append("estop_not_armed")
        if not inp.deadman_ok:
            transient.append("deadman_not_armed")
        return permanent, transient

    def _tick_precheck(self, inp: FsmInputs) -> FsmOutput:
        t = int(inp.t_ns)
        permanent, transient = self.precheck_problems(inp)
        if permanent:
            self._fault("precheck:" + permanent[0], t)
            return self._tick_fault(inp)
        if transient:
            if self._elapsed(t) > self.params.precheck_timeout_s:
                self._fault("precheck_timeout:" + transient[0], t)
                return self._tick_fault(inp)
            return FsmOutput(self.state, False, None, None, None, reason=";".join(transient))
        q = self._q(inp)
        assert q is not None
        try:
            self.fix_stand.start(q)
        except FixStandError as exc:
            self._fault(f"fix_stand_refused:{exc}", t)
            return self._tick_fault(inp)
        self._go(State.FIX_STAND, t, "precheck_passed")
        return self._tick_fix_stand(inp)

    def _tick_fix_stand(self, inp: FsmInputs) -> FsmOutput:
        t = int(inp.t_ns)
        if self._stop_requested:
            self._go(State.STOPPING, t, "stop_requested")
            return self._tick_stopping(inp)
        q = self._q(inp)
        assert q is not None
        fs_tick = self.fix_stand.step(q)
        if self.fix_stand.aborted:
            self._fault(self.fix_stand.fault or "fix_stand_aborted", t)
            out = self._tick_fault(inp)
            out.fix_stand = fs_tick.to_record()
            return out
        out = FsmOutput(
            state=self.state,
            publish=True,
            target=[float(v) for v in fs_tick.target],
            kp=fs_tick.kp,
            kd=fs_tick.kd,
            reason="fix_stand_ramp",
            fix_stand=fs_tick.to_record(),
        )
        if self.fix_stand.complete:
            # Ramp complete is NOT a handoff: it only moves to the measured hold.
            self._go(State.FIX_STAND_HOLD, t, "fix_stand_reference_complete")
        return out

    def _tick_hold_common(self, inp: FsmInputs) -> tuple[FsmOutput | None, HandoffStatus]:
        t = int(inp.t_ns)
        q = self._q(inp)
        assert q is not None
        fs_tick = self.fix_stand.step(q)  # keeps tracking the stance, can still abort
        status = self.handoff.update(self._handoff_sample(inp))
        if self.fix_stand.aborted:
            self._fault(self.fix_stand.fault or "fix_stand_aborted", t)
            out = self._tick_fault(inp)
            out.fix_stand = fs_tick.to_record()
            out.handoff = status.to_record()
            return out, status
        return None, status

    def _tick_hold(self, inp: FsmInputs) -> FsmOutput:
        t = int(inp.t_ns)
        if self._stop_requested:
            self._go(State.STOPPING, t, "stop_requested")
            return self._tick_stopping(inp)
        early, status = self._tick_hold_common(inp)
        if early is not None:
            return early
        if status.ready and self._policy is not None:
            self._go(State.VELOCITY_READY, t, "handoff_criteria_met")
            return self._stand_output("handoff_criteria_met", status)
        if (
            self._policy is not None
            and not status.ready
            and self._elapsed(t) > self.params.handoff_timeout_s
        ):
            self._fault("handoff_criteria_not_met:" + ";".join(status.failing[:2]), t)
            return self._tick_fault(inp)
        return self._stand_output(
            "stance_verified" if status.ready else "verifying_stance", status
        )

    def _tick_ready(self, inp: FsmInputs) -> FsmOutput:
        t = int(inp.t_ns)
        if self._stop_requested:
            self._go(State.STOPPING, t, "stop_requested")
            return self._tick_stopping(inp)
        early, status = self._tick_hold_common(inp)
        if early is not None:
            return early
        if not status.ready:
            # Lost the stance before the grant: back to verifying, never forward.
            self._policy_requested = False
            self._go(State.FIX_STAND_HOLD, t, "handoff_criteria_lost")
            return self._stand_output("handoff_criteria_lost", status)
        if self._policy_requested:
            self._policy_requested = False
            self._authority_start_ns = t
            self._last_action = np.zeros(self.n, dtype=np.float64)
            self._go(State.VELOCITY_ACTIVE, t, "policy_granted")
            out = self._tick_active(inp)
            out.handoff = status.to_record()
            return out
        return self._stand_output("velocity_ready", status)

    def _tick_active(self, inp: FsmInputs) -> FsmOutput:
        """The ONLY place the policy is called."""
        t = int(inp.t_ns)
        p = self.params
        if self.state is not State.VELOCITY_ACTIVE:  # structural guard, see module docstring
            raise RuntimeError(f"policy path entered in state {self.state.value}")
        if self._stop_requested:
            self._go(State.STOPPING, t, "stop_requested")
            return self._tick_stopping(inp)
        if (
            p.authority_s > 0
            and self._authority_start_ns is not None
            and (t - self._authority_start_ns) / 1e9 >= p.authority_s
        ):
            self._go(State.STOPPING, t, "authority_window_complete")
            return self._tick_stopping(inp)
        g = _projected_gravity(inp.quat_wxyz)
        if g is None:
            self._fault("imu_orientation_invalid", t)
            return self._tick_fault(inp)
        tilt = float(np.arccos(np.clip(-g[2], -1.0, 1.0)))
        if tilt > p.attitude_abort_rad:
            self._fault(f"attitude:{tilt:.3f}rad", t)
            return self._tick_fault(inp)
        q = self._q(inp)
        assert q is not None and inp.dq is not None and self._policy is not None
        assert self._obs_builder is not None
        command = np.zeros(3) if self.stand_only else np.asarray(inp.command, dtype=np.float64)
        try:
            obs = self._obs_builder(
                gyro_body=inp.gyro_body,
                quat_wxyz=inp.quat_wxyz,
                command=command,
                joint_pos=q,
                joint_vel=inp.dq,
                last_action=self._last_action,
            )
        except (ValueError, TypeError) as exc:  # ObservationError is a ValueError
            self._fault(f"observation_invalid:{exc}", t)
            return self._tick_fault(inp)
        self.policy_calls += 1
        action = np.asarray(self._policy(obs), dtype=np.float64).reshape(-1)
        if action.shape != (self.n,) or not np.all(np.isfinite(action)):
            self._fault("policy_output_invalid", t)
            return self._tick_fault(inp)
        scaled = self.fix_stand.target + p.action_scale * action
        result = self.safety.apply(scaled, q)
        record = tick_record(
            t_ns=t,
            order=self.order,
            raw_action=action,
            scaled_target=scaled,
            safety_target=result.final_target,
            final_target=result.final_target,
            reasons=result.reasons,
            source="fsm",
        )
        self.recorder.add(record, illegal=result.illegal)
        # last_action is the policy's own raw output, as in training, even when clipped.
        self._last_action = action
        if result.illegal:
            self._fault("illegal_target", t)
            out = self._tick_fault(inp)
            out.intervention = record
            return out
        if self.recorder.live_budget_exceeded():
            self._fault("intervention_budget_exceeded", t)
            out = self._tick_fault(inp)
            out.intervention = record
            return out
        return FsmOutput(
            state=self.state,
            publish=True,
            target=[float(v) for v in result.final_target],
            kp=p.policy_kp,
            kd=p.policy_kd,
            reason="policy",
            intervention=record,
        )

    def _tick_stopping(self, inp: FsmInputs) -> FsmOutput:
        t = int(inp.t_ns)
        q = self._q(inp)
        if self._stop_q is None:
            base = q if q is not None else self.fix_stand.target
            self._stop_q = np.clip(base, self.fix_stand.lo, self.fix_stand.hi)
        if self._stop_damp or self._elapsed(t) >= self.params.stop_hold_s:
            self._stop_damp = True
            return FsmOutput(
                state=self.state,
                publish=True,
                target=[float(v) for v in self._stop_q],
                kp=0.0,
                kd=self.params.damp_kd,
                reason="stopped_damping",
            )
        return FsmOutput(
            state=self.state,
            publish=True,
            target=[float(v) for v in self._stop_q],
            kp=self.params.hold_kp,
            kd=self.params.hold_kd,
            reason="stopping_hold",
        )

    def _tick_fault(self, inp: FsmInputs) -> FsmOutput:
        return self._damp_output(inp, self.fault)

    # --------------------------------------------------------------- report
    def summary(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "fault": self.fault,
            "stand_only": self.stand_only,
            "policy_configured": self._policy is not None,
            "policy_calls": self.policy_calls,
            "transitions": [
                {"t_ns": t, "from": a, "to": b, "why": why} for t, a, b, why in self.transitions
            ],
            "fix_stand": self.fix_stand.summary(),
            "intervention": self.recorder.summary(),
        }


def fsm_params_from_mapping(data: Mapping[str, Any] | None) -> FsmParams:
    data = dict(data or {})
    known = set(FsmParams.__dataclass_fields__)
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown fsm keys {unknown}")
    if "max_command" in data:
        data["max_command"] = tuple(float(v) for v in data["max_command"])
    return FsmParams(**data)


__all__ = [
    "ControllerFSM",
    "FsmInputs",
    "FsmOutput",
    "FsmParams",
    "State",
    "fsm_params_from_mapping",
]
