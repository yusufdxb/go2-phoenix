"""The Phoenix health vector: per-joint state with persistence, confidence and localisation.

One entry per GO2 joint::

    RR_thigh   s_hat 0.61  [0.57, 0.66]   baseline thr 0.88   9/10 windows   DEGRADED

* ``s_hat``      median authority-ratio estimate over the persistence window
                 (:mod:`phoenix.monitor.residual`); 1.0 nominal.
* interval       2.5 / 97.5 percentiles of the window estimates in that span,
                 an honest spread, not a model-based confidence interval.
* ``threshold``  from the nominal baseline of the same command regime.
* persistence    how many of the last ``n`` usable windows were below threshold.
* state          NOMINAL / SUSPECT / DEGRADED / GLOBAL_SHIFT / INSUFFICIENT_DATA.

Rules
-----
* DEGRADED needs ``k_of_n`` below-threshold windows among the last ``n`` usable
  windows (default 8 of 10, i.e. about 10 s at 1 s windows). One bad window only
  makes a joint SUSPECT. This is the false-positive guard against transients.
* Recovery is hysteretic: a DEGRADED joint returns to NOMINAL only when at most
  ``recover_max`` of the last ``n`` windows are below threshold.
* If more than ``max_localised`` joints qualify at once, every qualifying joint is
  reported GLOBAL_SHIFT instead: a load, floor or regime change looks like that,
  one weak actuator does not. Phoenix never builds a targeted distribution from a
  global shift.
* A joint whose windows are mostly unusable (safety-altered, not under policy
  authority) is INSUFFICIENT_DATA, never NOMINAL. The monitor refuses to vouch for
  a joint it could not see.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np

from phoenix.sim2real.go2_model import UNITREE_MOTOR_ORDER

from .residual import Baseline


class JointState(str, Enum):
    NOMINAL = "NOMINAL"
    SUSPECT = "SUSPECT"
    DEGRADED = "DEGRADED"
    GLOBAL_SHIFT = "GLOBAL_SHIFT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class PersistenceConfig:
    n: int = 10
    k_of_n: int = 8
    recover_max: int = 2
    max_localised: int = 2
    #: fewer usable windows than this in the last ``n`` -> INSUFFICIENT_DATA
    min_usable: int = 6

    def __post_init__(self) -> None:
        if not 1 <= self.k_of_n <= self.n:
            raise ValueError("need 1 <= k_of_n <= n")
        if not 0 <= self.recover_max < self.k_of_n:
            raise ValueError("need 0 <= recover_max < k_of_n (hysteresis)")
        if not 1 <= self.min_usable <= self.n:
            raise ValueError("need 1 <= min_usable <= n")
        if self.max_localised < 1:
            raise ValueError("max_localised must be >= 1")


DEFAULT_PERSISTENCE = PersistenceConfig()


@dataclass(frozen=True)
class JointHealth:
    joint: str
    state: str
    s_hat: float | None
    s_lo: float | None
    s_hi: float | None
    threshold: float
    below: int
    usable: int
    n: int
    torque_gain_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _f(x: float) -> float | None:
    return None if not np.isfinite(x) else float(x)


class HealthMonitor:
    """Streaming state machine over window estimates. Feed :meth:`update` once per window."""

    def __init__(self, baseline: Baseline, cfg: PersistenceConfig = DEFAULT_PERSISTENCE) -> None:
        self.baseline = baseline
        self.cfg = cfg
        self._thr = np.asarray(baseline.threshold, dtype=np.float64)
        self._hist: list[deque[float]] = [deque(maxlen=cfg.n) for _ in range(12)]
        self._tq: list[deque[float]] = [deque(maxlen=cfg.n) for _ in range(12)]
        self._degraded = np.zeros(12, dtype=bool)
        self.windows_seen = 0

    def update(
        self, s_hat: np.ndarray, torque_gain_ratio: np.ndarray | None = None
    ) -> list[JointHealth]:
        s_hat = np.asarray(s_hat, dtype=np.float64).reshape(12)
        tg = (
            np.full(12, np.nan)
            if torque_gain_ratio is None
            else np.asarray(torque_gain_ratio, dtype=np.float64).reshape(12)
        )
        self.windows_seen += 1
        for j in range(12):
            self._hist[j].append(float(s_hat[j]))
            self._tq[j].append(float(tg[j]))
        return self.report()

    def report(self) -> list[JointHealth]:
        c = self.cfg
        below = np.zeros(12, dtype=int)
        usable = np.zeros(12, dtype=int)
        for j in range(12):
            h = np.asarray(self._hist[j], dtype=np.float64)
            ok = np.isfinite(h)
            usable[j] = int(ok.sum())
            below[j] = int(np.sum(h[ok] < self._thr[j]))
        enough = usable >= c.min_usable
        qualifies = enough & (below >= c.k_of_n)
        recovered = below <= c.recover_max
        # hysteresis: stay degraded until clearly recovered
        self._degraded = np.where(  # type: ignore[assignment]
            qualifies, True, np.where(recovered | ~enough, False, self._degraded)
        )
        global_shift = int(self._degraded.sum()) > c.max_localised
        out = []
        for j, name in enumerate(UNITREE_MOTOR_ORDER):
            h = np.asarray(self._hist[j], dtype=np.float64)
            h = h[np.isfinite(h)]
            t = np.asarray(self._tq[j], dtype=np.float64)
            t = t[np.isfinite(t)]
            if not enough[j]:
                state = JointState.INSUFFICIENT_DATA
            elif self._degraded[j]:
                state = JointState.GLOBAL_SHIFT if global_shift else JointState.DEGRADED
            elif below[j] > 0:
                state = JointState.SUSPECT
            else:
                state = JointState.NOMINAL
            out.append(
                JointHealth(
                    joint=name,
                    state=state.value,
                    s_hat=_f(float(np.median(h))) if h.size else None,
                    s_lo=_f(float(np.percentile(h, 2.5))) if h.size else None,
                    s_hi=_f(float(np.percentile(h, 97.5))) if h.size else None,
                    threshold=float(self._thr[j]),
                    below=int(below[j]),
                    usable=int(usable[j]),
                    n=len(self._hist[j]),
                    torque_gain_ratio=_f(float(np.median(t))) if t.size else None,
                )
            )
        return out


def degraded_joints(report: list[JointHealth]) -> list[JointHealth]:
    """Only localised degradations; a GLOBAL_SHIFT never qualifies."""
    return [h for h in report if h.state == JointState.DEGRADED.value]


def format_report(report: list[JointHealth]) -> str:
    """The fixed-width table the demo overlay and the CLI print."""
    lines = [f"{'joint':<10} {'s_hat':>6} {'interval':>15} {'thr':>5} {'persist':>8}  state"]
    for h in report:
        s = "  -  " if h.s_hat is None else f"{h.s_hat:6.2f}"
        iv = (
            "       -       "
            if h.s_lo is None or h.s_hi is None
            else f"[{h.s_lo:5.2f}, {h.s_hi:5.2f}]"
        )
        name = h.joint.replace("_joint", "")
        lines.append(
            f"{name:<10} {s:>6} {iv:>15} {h.threshold:5.2f} {h.below:>3}/{h.usable:<4}  {h.state}"
        )
    return "\n".join(lines)


__all__ = [
    "HealthMonitor",
    "JointHealth",
    "JointState",
    "PersistenceConfig",
    "degraded_joints",
    "format_report",
]
