"""Fault and shutdown policy without zero stiffness: the controlled sit and its helpers.

Owner rule (2026-10-03): zero stiffness is never commanded while the robot may
be standing. Every fault path ends in a posed, powered, folded robot on the
ground, never in ``kp = 0``. Sport API 1001 is never sent.

Policy (implemented in :mod:`phoenix.sim2real.actuator_gate`,
:mod:`phoenix.sim2real.controller_fsm`, :mod:`phoenix.sim2real.fix_stand` and the
bridge shutdown path)::

    stale LowState inside stale_hold_s     hold the last target (hold gains)
    stale beyond / non-finite / impossible q   OPEN-LOOP sit from the last good q
    policy NaN / abort / stale command /
      sustained over-limit / estop / deadman   hold <= fault_hold_s, then CLOSED-LOOP sit
    fall (tilt > fall_tilt_rad)             FALL_SOFT: kp 5 / kd 1 at the measured q
    SIGINT / SIGTERM                        sit, settle, FOLDED low-kp hold, exit
    SIGKILL / power loss                    nothing can be sent (documented only)

The sit is a smoothstep from the start posture to :data:`SIT_POSE_UNITREE` over
``sit_s``, each joint's target rate-limited to ``max_rate_rad_s`` and clipped so
the implied PD torque ``kp (target - q) - kd dq`` stays inside the actuator limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .go2_model import (
    UNITREE_EXAMPLE_FOLDED_POSE,
    UNITREE_MOTOR_ORDER,
    limits_in_order,
    torque_limits_in_order,
)
from .safety import torque_limited_target_array

_LO, _HI = limits_in_order(UNITREE_MOTOR_ORDER)
#: Folded posture, Unitree motor order, held a small margin inside the hard limits.
SIT_POSE_UNITREE: np.ndarray = np.clip(
    np.asarray(UNITREE_EXAMPLE_FOLDED_POSE, dtype=np.float64), _LO + 0.02, _HI - 0.02
)


@dataclass(frozen=True)
class SitParams:
    sit_s: float = 2.0
    sit_kp: float = 30.0
    sit_kd: float = 1.5
    #: Per-joint bound on how fast the sit target may move.
    max_rate_rad_s: float = 2.0
    #: A tick gap longer than this is treated as this long (a stalled loop must
    #: not turn into one large target jump).
    max_dt_s: float = 0.1
    #: Closed-loop sit begins after this long in HOLD (fault classes 3 and 4).
    fault_hold_s: float = 1.0
    #: Low stiffness for a fallen robot (never zero).
    fall_kp: float = 5.0
    fall_kd: float = 1.0
    fall_tilt_rad: float = math.radians(60.0)
    #: Shutdown tail: hold the finished sit at sit gains, then folded low kp.
    folded_settle_s: float = 0.3
    folded_kp: float = 5.0
    folded_kd: float = 1.0
    folded_hold_s: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "sit_s", "sit_kp", "sit_kd", "max_rate_rad_s", "max_dt_s",
            "fall_kp", "fall_kd", "fall_tilt_rad", "folded_kp", "folded_kd",
        ):  # fmt: skip
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"SitParams.{name} must be positive and finite, got {v}")
        for name in ("fault_hold_s", "folded_settle_s", "folded_hold_s"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"SitParams.{name} must be non-negative and finite, got {v}")

    @property
    def shutdown_bound_s(self) -> float:
        """Longest a SIGINT/SIGTERM sequence runs: sit + settle + folded hold."""
        return self.sit_s + self.folded_settle_s + self.folded_hold_s


def fall_tilt_rad(quat_wxyz) -> float | None:
    """Angle between body z and world up from a (w, x, y, z) quaternion, or None."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        return None
    n = float(np.linalg.norm(q))
    if n < 1e-6:
        return None
    w, x, y, z = q / n
    return float(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


class SitTrajectory:
    """Time-parameterised sit. Closed-loop passes the measured q to :meth:`step`
    (torque clip is taken about it); open-loop passes ``None`` and the clip is
    taken about the previous command, so nothing measured is trusted."""

    def __init__(self, params: SitParams) -> None:
        self.params = params
        self._torque_limit = torque_limits_in_order(UNITREE_MOTOR_ORDER)
        self.q0: np.ndarray | None = None
        self.start_ns = 0
        self._last_ns = 0
        self._prev: np.ndarray | None = None
        self.done_ns: int | None = None

    @property
    def started(self) -> bool:
        return self.q0 is not None

    @property
    def target(self) -> np.ndarray:
        assert self._prev is not None
        return self._prev

    def start(self, q0, now_ns: int) -> None:
        self.q0 = np.clip(np.nan_to_num(np.asarray(q0, dtype=np.float64)), _LO, _HI)
        self.start_ns = self._last_ns = int(now_ns)
        self._prev = self.q0.copy()
        self.done_ns = None

    def step(self, now_ns: int, q_meas=None, dq_meas=None) -> tuple[np.ndarray, bool]:
        assert self.q0 is not None and self._prev is not None
        p = self.params
        dt = min(max((int(now_ns) - self._last_ns) / 1e9, 0.0), p.max_dt_s)
        self._last_ns = int(now_ns)
        s = min(max((int(now_ns) - self.start_ns) / 1e9 / p.sit_s, 0.0), 1.0)
        ideal = self.q0 + (s * s * (3.0 - 2.0 * s)) * (SIT_POSE_UNITREE - self.q0)
        step = p.max_rate_rad_s * dt
        t = np.clip(ideal, self._prev - step, self._prev + step)
        q_ref = self._prev if q_meas is None else np.asarray(q_meas, dtype=np.float64)
        dq_ref = np.zeros(12) if dq_meas is None or q_meas is None else np.asarray(dq_meas)
        t = np.asarray(
            torque_limited_target_array(t, q_ref, dq_ref, p.sit_kp, p.sit_kd, self._torque_limit),
            dtype=np.float64,
        )
        t = np.clip(t, _LO, _HI)
        self._prev = t
        done = s >= 1.0 and bool(np.allclose(t, SIT_POSE_UNITREE, atol=1e-9))
        if done and self.done_ns is None:
            self.done_ns = int(now_ns)
        return t, done
