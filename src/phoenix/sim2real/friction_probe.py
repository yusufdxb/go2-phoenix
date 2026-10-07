"""Active joint-friction probe for a hung GO2: protocol, sequencer and estimator.

Pure numpy, no ROS and no MuJoCo. The robot-side node and the desk simulation both
import this module, so the protocol exists in exactly one place.

PROTOCOL (all arrays are in :data:`PROBE_JOINT_ORDER` = Unitree motor order, the order
of ``motor_cmd[k]`` / ``motor_state[k]``)

1. Every joint is held at its measured start pose ``q0`` under PD (:data:`KP`,
   :data:`KD`; the bridge's policy gains) for :data:`SETTLE_S` seconds.
2. One joint at a time, in :data:`PROBE_JOINT_ORDER`, the target ramps from ``q0`` at
   :data:`RATE_RAD_S` in the + direction until the joint has moved
   :data:`BREAK_DELTA_RAD` from where it sat when the ramp began (breakaway) or the
   offset reaches :data:`MAX_OFFSET_RAD` (or the joint-limit room, see below). The ramp
   continues for :data:`POST_BREAK_S` after breakaway so there is a short clean
   slow-motion segment for the kinetic estimate, then falls back to ``q0`` at the same
   rate. After :data:`REST_S` at ``q0`` the same is done in the - direction, and
   after :data:`HOLD_S` at ``q0`` the next joint starts. Every other target stays
   ``q0``.
3. Every commanded target passes through :func:`clamp_target`: never closer to a hard
   limit than :data:`LIMIT_MARGIN_RAD`. A direction with no room simply has a shorter
   (possibly zero) excursion, and the estimator reports a lower bound for it.

The ramp length depends on when breakaway happens, so the schedule is a small state
machine, :class:`ProbeSequencer`, fed the measured ``q`` each tick. The offset shape
alone is the pure function :func:`excursion_offset`. Worst-case total duration is
:func:`worst_case_duration_s`.

ESTIMATOR :func:`estimate` works on a per-tick :class:`ProbeTrace`. Motor torque is
``tau_m = kp * (target - q) - kd * dq``. In the + direction breakaway needs
``tau_m = g + Fs`` and in the - direction ``tau_m = g - Fs`` (``g`` is the torque
that statically holds the joint against gravity), so ``Fs = (tau_plus - tau_minus) / 2``
and ``g = (tau_plus + tau_minus) / 2``. A direction with no breakaway still bounds
``Fs`` from below by the same formula using the torque reached at the largest offset.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .go2_model import UNITREE_MOTOR_ORDER, limits_in_order

PROBE_JOINT_ORDER: tuple[str, ...] = UNITREE_MOTOR_ORDER
N_JOINTS = len(PROBE_JOINT_ORDER)

KP = 25.0
KD = 0.5
TICK_HZ = 50.0

SETTLE_S = 2.0
HANDOVER_S = 1.0  # robot side: gain handover before SETTLE_S starts; not simulated
RATE_RAD_S = 0.05
BREAK_DELTA_RAD = 0.02
MAX_OFFSET_RAD = 0.30
POST_BREAK_S = 0.5
REST_S = 0.5
HOLD_S = 1.0
LIMIT_MARGIN_RAD = 0.05

# Estimator settings.
WINDOW_HALF_S = 0.1  # median window around the breakaway instant
PRE_WINDOW_S = 0.2  # quiescent window before each ramp that defines q_ref
QUIET_P2P_RAD = 0.25 * BREAK_DELTA_RAD
V_MOVING_RAD_S = 0.01
MIN_KINETIC_SAMPLES = 5
NOISY_TAU_NM = 0.5
FS_NEGATIVE_TOL_NM = 0.1  # a slightly negative (tp - tm)/2 is noise at Fs ~ 0; below is a fault

FLAG_NO_BREAK_PLUS = "no_breakaway_plus"
FLAG_NO_BREAK_MINUS = "no_breakaway_minus"
FLAG_NO_EXCURSION_PLUS = "no_excursion_plus"
FLAG_NO_EXCURSION_MINUS = "no_excursion_minus"
FLAG_NOISY = "noisy"
FLAG_MOVED_BEFORE_RAMP = "moved_before_ramp"
FLAG_WRONG_DIRECTION = "wrong_direction"
FLAG_NO_PRE_WINDOW = "no_pre_window"
FLAG_KINETIC_UNCLEAN = "kinetic_unclean"


def worst_case_duration_s() -> float:
    """Upper bound on the probe duration: no breakaway anywhere, full ramps both ways."""
    per_joint = 2.0 * (2.0 * MAX_OFFSET_RAD / RATE_RAD_S) + REST_S + HOLD_S
    return SETTLE_S + N_JOINTS * per_joint


def probe_limits(order: Sequence[str] = PROBE_JOINT_ORDER) -> tuple[np.ndarray, np.ndarray]:
    """Hard position limits (lower, upper) in ``order`` from the go2_model table."""
    return limits_in_order(order)


def clamp_target(
    target: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    margin: float = LIMIT_MARGIN_RAD,
) -> np.ndarray:
    """Clip ``target`` into ``[lower + margin, upper - margin]`` (hard limits by default)."""
    if lower is None or upper is None:
        lower, upper = probe_limits()
    t = np.asarray(target, dtype=np.float64)
    return np.clip(t, np.asarray(lower) + margin, np.asarray(upper) - margin)


def excursion_offset(t_since_start: float, peak: float, rate: float = RATE_RAD_S) -> float:
    """Unsigned offset of one excursion: rise at ``rate`` to ``peak``, fall at ``rate`` to 0.

    Zero before the start and after the return. ``peak`` is the offset actually reached
    (breakaway plus :data:`POST_BREAK_S`, or the cap), known only at run time.
    """
    if t_since_start <= 0.0 or peak <= 0.0:
        return 0.0
    rise = peak / rate
    if t_since_start <= rise:
        return float(rate * t_since_start)
    return float(max(0.0, peak - rate * (t_since_start - rise)))


@dataclass(frozen=True)
class ProbeStatus:
    """Where the sequencer is. ``joint`` is an index into :data:`PROBE_JOINT_ORDER`."""

    phase: str  # settle | rise | fall | rest | done
    joint: int
    direction: int  # +1 or -1
    offset: float  # signed offset applied to the active joint
    broke: bool


class ProbeSequencer:
    """Feedback state machine that turns measured ``q`` into 12 clamped targets.

    Call :meth:`step` once per tick with the time since the probe started and the
    measured joint positions; it returns the 12 targets to command. Deterministic and
    side-effect free apart from its own state.
    """

    def __init__(
        self,
        q0: np.ndarray,
        *,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
        margin: float = LIMIT_MARGIN_RAD,
    ) -> None:
        q0 = np.asarray(q0, dtype=np.float64)
        if q0.shape != (N_JOINTS,):
            raise ValueError(f"q0 must have shape ({N_JOINTS},), got {q0.shape}")
        if lower is None or upper is None:
            lower, upper = probe_limits()
        self.q0 = q0.copy()
        self._lower = np.asarray(lower, dtype=np.float64)
        self._upper = np.asarray(upper, dtype=np.float64)
        self._margin = float(margin)
        self._origin: float | None = None
        self._phase = "settle"
        self._joint = 0
        self._dir = 1
        self._t_phase = 0.0
        self._rest_s = SETTLE_S
        self._peak = 0.0
        self._max_off = 0.0
        self._t_break: float | None = None
        self._q_ref = q0.copy()
        self._hist: deque[tuple[float, np.ndarray]] = deque()
        self._offset = 0.0

    @property
    def finished(self) -> bool:
        return self._phase == "done"

    @property
    def status(self) -> ProbeStatus:
        return ProbeStatus(
            self._phase, self._joint, self._dir, self._offset, self._t_break is not None
        )

    def _room(self, joint: int, direction: int) -> float:
        if direction > 0:
            room = self._upper[joint] - self._margin - self.q0[joint]
        else:
            room = self.q0[joint] - (self._lower[joint] + self._margin)
        return float(min(MAX_OFFSET_RAD, max(0.0, room)))

    def _begin_rise(self, te: float) -> None:
        self._phase = "rise"
        self._t_phase = te
        self._t_break = None
        self._max_off = self._room(self._joint, self._dir)
        self._q_ref = np.median(np.stack([q for _, q in self._hist]), axis=0)

    def _end_excursion(self, te: float) -> None:
        self._t_phase = te
        self._phase = "rest"
        if self._dir > 0:
            self._dir = -1
            self._rest_s = REST_S
        else:
            self._dir = 1
            self._joint += 1
            self._rest_s = HOLD_S
            if self._joint >= N_JOINTS:
                self._phase = "done"

    def step(self, t: float, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        if self._origin is None:
            self._origin = float(t)
        te = float(t) - self._origin
        self._hist.append((te, q.copy()))
        while self._hist and self._hist[0][0] < te - PRE_WINDOW_S:
            self._hist.popleft()

        if self._phase == "settle" and te >= SETTLE_S:
            self._begin_rise(te)
        elif self._phase == "rest" and te - self._t_phase >= self._rest_s:
            self._begin_rise(te)

        mag = 0.0
        if self._phase == "rise":
            mag = min(RATE_RAD_S * (te - self._t_phase), self._max_off)
            j = self._joint
            if self._t_break is None and self._dir * (q[j] - self._q_ref[j]) >= BREAK_DELTA_RAD:
                self._t_break = te
            capped = mag >= self._max_off
            post = self._t_break is not None and te - self._t_break >= POST_BREAK_S
            if capped or post:
                self._phase = "fall"
                self._t_phase = te
                self._peak = mag
        elif self._phase == "fall":
            mag = self._peak - RATE_RAD_S * (te - self._t_phase)
            if mag <= 0.0:
                mag = 0.0
                self._end_excursion(te)

        target = self.q0.copy()
        j = min(self._joint, N_JOINTS - 1)
        target[j] += self._dir * mag
        self._offset = self._dir * mag
        return clamp_target(target, self._lower, self._upper, self._margin)


@dataclass
class ProbeTrace:
    """Per-tick record. ``q``, ``target``, ``dq`` are ``(N, 12)`` in probe joint order.

    ``kp`` and ``kd`` may be scalars, ``(12,)`` or ``(N, 12)``. ``tau_est`` is the
    optional robot-reported joint torque estimate, ``(N, 12)``, same sign convention
    as ``tau_m``.
    """

    t: np.ndarray
    q: np.ndarray
    target: np.ndarray
    dq: np.ndarray
    kp: Any = KP
    kd: Any = KD
    tau_est: np.ndarray | None = None

    def validate(self) -> None:
        n = len(self.t)
        for name in ("q", "target", "dq"):
            arr = getattr(self, name)
            if arr.shape != (n, N_JOINTS):
                raise ValueError(f"{name} must have shape ({n}, {N_JOINTS}), got {arr.shape}")
        if self.tau_est is not None and self.tau_est.shape != (n, N_JOINTS):
            raise ValueError(f"tau_est must have shape ({n}, {N_JOINTS})")

    def motor_torque(self) -> np.ndarray:
        kp = np.broadcast_to(np.asarray(self.kp, dtype=np.float64), self.q.shape)
        kd = np.broadcast_to(np.asarray(self.kd, dtype=np.float64), self.q.shape)
        return np.asarray(kp * (self.target - self.q) - kd * self.dq)


@dataclass
class JointEstimate:
    name: str
    index: int
    fs_nm: float | None = None  # static friction, only when both directions broke away
    bias_nm: float | None = None  # gravity/bias holding torque
    kinetic_nm: float | None = None
    fs_lower_bound_nm: float | None = None  # when not two-sided: Fs > this
    tau_plus_nm: float | None = None  # torque at + breakaway (or at + peak if none)
    tau_minus_nm: float | None = None
    t_break_plus_s: float | None = None
    t_break_minus_s: float | None = None
    q_ref_plus: float | None = None
    q_ref_minus: float | None = None
    sag_rad: float | None = None  # q_ref_plus - q0
    fs_tau_est_nm: float | None = None
    bias_tau_est_nm: float | None = None
    flags: list[str] = field(default_factory=list)

    @property
    def two_sided(self) -> bool:
        return self.fs_nm is not None

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["two_sided"] = self.two_sided
        return d


@dataclass
class ProbeEstimate:
    joints: tuple[JointEstimate, ...]

    def by_name(self, name: str) -> JointEstimate:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {"joints": [j.to_dict() for j in self.joints]}


@dataclass
class _Run:
    i0: int  # first sample with a nonzero offset
    ipk: int  # sample of the largest |offset|
    direction: int


def _runs(off: np.ndarray) -> list[_Run]:
    nz = np.abs(off) > 1e-9
    out: list[_Run] = []
    n = len(off)
    i = 0
    while i < n:
        if not nz[i]:
            i += 1
            continue
        j = i
        while j < n and nz[j]:
            j += 1
        seg = off[i:j]
        ipk = i + int(np.argmax(np.abs(seg)))
        out.append(_Run(i, ipk, 1 if off[ipk] > 0 else -1))
        i = j
    return out


def _med(x: np.ndarray) -> float:
    return float(np.median(x))


def _robust_sigma_detrended(t: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 4:
        return 0.0
    fit = np.polyval(np.polyfit(t - t[0], y, 1), t - t[0])
    r = y - fit
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def estimate(trace: ProbeTrace) -> ProbeEstimate:
    """Estimate per-joint static friction, gravity bias and kinetic friction."""
    trace.validate()
    t = np.asarray(trace.t, dtype=np.float64)
    q = np.asarray(trace.q, dtype=np.float64)
    dq = np.asarray(trace.dq, dtype=np.float64)
    tau_m = trace.motor_torque()
    tau_e = None if trace.tau_est is None else np.asarray(trace.tau_est, dtype=np.float64)
    q0 = trace.target[0]
    out: list[JointEstimate] = []

    for j, name in enumerate(PROBE_JOINT_ORDER):
        est = JointEstimate(name=name, index=j)
        off = trace.target[:, j] - q0[j]
        tau_dir: dict[int, float] = {}
        tau_dir_e: dict[int, float] = {}
        kin: dict[int, float] = {}
        seen: set[int] = set()
        for run in _runs(off):
            d = run.direction
            if d in seen:
                continue
            seen.add(d)
            pre = np.nonzero((t >= t[run.i0] - PRE_WINDOW_S) & (np.arange(len(t)) < run.i0))[0]
            if len(pre) < 3:
                est.flags.append(FLAG_NO_PRE_WINDOW)
                pre = np.arange(max(0, run.i0 - 3), run.i0) if run.i0 > 0 else np.array([0])
            q_ref = _med(q[pre, j])
            if np.ptp(q[pre, j]) > QUIET_P2P_RAD:
                est.flags.append(FLAG_MOVED_BEFORE_RAMP)
            rise = np.arange(run.i0, run.ipk + 1)
            moved = d * (q[rise, j] - q_ref) >= BREAK_DELTA_RAD
            hit = np.nonzero(moved)[0]
            if d > 0:
                est.q_ref_plus = q_ref
                est.sag_rad = q_ref - float(q0[j])
            else:
                est.q_ref_minus = q_ref
            if len(hit) == 0:
                est.flags.append(FLAG_NO_BREAK_PLUS if d > 0 else FLAG_NO_BREAK_MINUS)
                w = rise[t[rise] >= t[run.ipk] - 2 * WINDOW_HALF_S]
                tau_dir[d] = _med(tau_m[w, j])
                if tau_e is not None:
                    tau_dir_e[d] = _med(tau_e[w, j])
                continue
            ib = int(rise[hit[0]])
            tb = float(t[ib])
            if d > 0:
                est.t_break_plus_s = tb
            else:
                est.t_break_minus_s = tb
            w = rise[np.abs(t[rise] - tb) <= WINDOW_HALF_S]
            tau_dir[d] = _med(tau_m[w, j])
            if tau_e is not None:
                tau_dir_e[d] = _med(tau_e[w, j])
            if _robust_sigma_detrended(t[w], tau_m[w, j]) > NOISY_TAU_NM:
                est.flags.append(FLAG_NOISY)
            after = rise[(rise >= ib) & (d * dq[rise, j] >= V_MOVING_RAD_S)]
            if len(after) >= MIN_KINETIC_SAMPLES:
                kin[d] = _med(tau_m[after, j])
            else:
                est.flags.append(FLAG_KINETIC_UNCLEAN)

        for d, flag in ((1, FLAG_NO_EXCURSION_PLUS), (-1, FLAG_NO_EXCURSION_MINUS)):
            if d not in seen:
                est.flags.append(flag)
        est.tau_plus_nm = tau_dir.get(1)
        est.tau_minus_nm = tau_dir.get(-1)
        if 1 in tau_dir and -1 in tau_dir:
            half = (tau_dir[1] - tau_dir[-1]) / 2.0
            both = est.t_break_plus_s is not None and est.t_break_minus_s is not None
            if both and half >= -FS_NEGATIVE_TOL_NM:
                est.fs_nm = half
                est.bias_nm = (tau_dir[1] + tau_dir[-1]) / 2.0
                if tau_e is not None and 1 in tau_dir_e and -1 in tau_dir_e:
                    est.fs_tau_est_nm = (tau_dir_e[1] - tau_dir_e[-1]) / 2.0
                    est.bias_tau_est_nm = (tau_dir_e[1] + tau_dir_e[-1]) / 2.0
            else:
                if both:
                    est.flags.append(FLAG_WRONG_DIRECTION)
                est.fs_lower_bound_nm = max(0.0, half)
        if est.fs_nm is not None and 1 in kin and -1 in kin:
            est.kinetic_nm = (kin[1] - kin[-1]) / 2.0
        elif est.fs_nm is not None and FLAG_KINETIC_UNCLEAN not in est.flags:
            est.flags.append(FLAG_KINETIC_UNCLEAN)
        est.flags = list(dict.fromkeys(est.flags))
        out.append(est)
    return ProbeEstimate(tuple(out))


__all__ = [
    "BREAK_DELTA_RAD",
    "FLAG_KINETIC_UNCLEAN",
    "FLAG_MOVED_BEFORE_RAMP",
    "FLAG_NOISY",
    "FLAG_NO_BREAK_MINUS",
    "FLAG_NO_BREAK_PLUS",
    "HANDOVER_S",
    "HOLD_S",
    "KD",
    "KP",
    "LIMIT_MARGIN_RAD",
    "MAX_OFFSET_RAD",
    "N_JOINTS",
    "POST_BREAK_S",
    "PROBE_JOINT_ORDER",
    "RATE_RAD_S",
    "REST_S",
    "SETTLE_S",
    "TICK_HZ",
    "JointEstimate",
    "ProbeEstimate",
    "ProbeSequencer",
    "ProbeStatus",
    "ProbeTrace",
    "clamp_target",
    "estimate",
    "excursion_offset",
    "probe_limits",
    "worst_case_duration_s",
]
