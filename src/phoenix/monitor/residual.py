"""Per-joint actuator-response residuals, calibrated against the robot's own nominal runs.

What is measured
----------------
For each joint and each window of ``window_ticks`` bridge ticks (1 s at 50 Hz by
default), the RMS of the tracking error ``e = sent[k] - q[k+1]`` over the VALID
samples only (:func:`phoenix.monitor.layers.tracking_pairs`): both ticks under
policy authority, finite, and the target not altered by any safety layer. A target
the slew clip moved to ``q +/- 0.175`` pins the error to the clip, so those samples
say nothing about the actuator and are excluded, never "corrected".

Why RMS error and what it estimates
-----------------------------------
The GO2 motor driver runs ``tau = kp (q_sent - q) - kd dq``. Under the same task
(same command regime, same load) the steady tracking error of that loop scales as
``1 / s`` when the delivered authority changes by a factor ``s``. So with a
nominal reference RMS ``R_j`` (calibrated per joint, per command regime), the
window's authority-ratio estimate is::

    s_hat_j = R_j / rms_j(window)

``s_hat ~ 1`` nominal, ``< 1`` less authority than nominal. This is an estimate of
*response effectiveness under this task*, not a motor-health percentage: a heavier
payload, a softer floor or a different gait also raise the error. That is why the
estimate is only reported against a baseline recorded in the same regime, why a
change on many joints at once is reported as a GLOBAL shift and never localised to
one actuator (:mod:`phoenix.monitor.health`), and why nothing here names a
physical fault.

The approximation is checked on the one-joint model in
``phoenix.condition.toy_model`` (within 5% over s in [0.5, 1.0] in the linear
range). It is biased when the loop saturates (effort limit) and when the policy
compensates the very error it is measured by; both are documented limitations.

A second, independent estimate uses the motor's own torque estimate when the
bridge logged it: least squares of ``tau_est`` on ``kp e - kd dq`` gives the gain
the joint actually delivered relative to the gain that was sent. It needs no
baseline, but ``tau_est`` is computed by the firmware from motor current with a
nominal torque constant, so it sees a software gain change exactly and a weakened
motor not at all. It is reported alongside, never instead.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .layers import N_JOINTS, TrackingPairs


@dataclass(frozen=True)
class WindowConfig:
    window_ticks: int = 50  # 1 s at 50 Hz
    min_valid_fraction: float = 0.6  # of the window's samples, per joint
    #: Error below this RMS is treated as the noise floor: the ratio of two
    #: numbers at encoder resolution is meaningless (joint encoders on the GO2
    #: resolve well below 1 mrad; 2 mrad is a conservative floor).
    noise_floor_rad: float = 0.002

    def __post_init__(self) -> None:
        if self.window_ticks < 5:
            raise ValueError("window_ticks must be >= 5")
        if not 0.0 < self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in (0, 1]")
        if self.noise_floor_rad <= 0:
            raise ValueError("noise_floor_rad must be positive")


DEFAULT_WINDOW = WindowConfig()


@dataclass(frozen=True)
class WindowStats:
    """Per-window, per-joint statistics. Arrays have shape ``(W, 12)``."""

    t_start_s: np.ndarray  # (W,)
    rms_error: np.ndarray  # NaN where the window had too few valid samples
    mean_error: np.ndarray
    valid_fraction: np.ndarray
    safety_altered_fraction: np.ndarray
    torque_gain_ratio: np.ndarray  # NaN when tau_est was not logged

    @property
    def n_windows(self) -> int:
        return int(self.t_start_s.shape[0])


def _torque_gain_ratio(
    tau: np.ndarray, e: np.ndarray, dq: np.ndarray, kp: np.ndarray, kd: np.ndarray
):
    """Least-squares ``g`` in ``tau ~ g * (kp e - kd dq)``, per joint; NaN if unidentifiable."""
    drive = kp * e - kd * dq
    out = np.full(N_JOINTS, np.nan)
    for j in range(N_JOINTS):
        ok = np.isfinite(tau[:, j]) & np.isfinite(drive[:, j])
        if ok.sum() < 5:
            continue
        x, y = drive[ok, j], tau[ok, j]
        denom = float(np.dot(x, x))
        # Require the drive to vary enough to identify a slope (0.5 N m RMS).
        if denom <= 0.25 * ok.sum():
            continue
        out[j] = float(np.dot(x, y) / denom)
    return out


def window_stats(pairs: TrackingPairs, cfg: WindowConfig = DEFAULT_WINDOW) -> WindowStats:
    """Split the paired samples into consecutive non-overlapping windows."""
    n = pairs.t_s.shape[0]
    w = cfg.window_ticks
    n_win = n // w
    t0, rms, mean, vf, af, tg = [], [], [], [], [], []
    e_all = pairs.error
    for i in range(n_win):
        sl = slice(i * w, (i + 1) * w)
        valid = pairs.valid[sl]
        e = np.where(valid, e_all[sl], np.nan)
        frac = valid.mean(axis=0)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN joints -> NaN
            r = (
                np.sqrt(np.nanmean(np.square(e), axis=0))
                if valid.any()
                else np.full(N_JOINTS, np.nan)
            )
            m = np.nanmean(e, axis=0) if valid.any() else np.full(N_JOINTS, np.nan)
        enough = frac >= cfg.min_valid_fraction
        r = np.where(enough, r, np.nan)
        m = np.where(enough, m, np.nan)
        tau = np.where(valid, pairs.tau_next[sl], np.nan)
        tg.append(_torque_gain_ratio(tau, e_all[sl], pairs.dq_next[sl], pairs.kp[sl], pairs.kd[sl]))
        t0.append(float(pairs.t_s[sl][0]))
        rms.append(r)
        mean.append(m)
        vf.append(frac)
        af.append(pairs.safety_altered[sl].mean(axis=0))
    empty = np.zeros((0, N_JOINTS))
    return WindowStats(
        t_start_s=np.asarray(t0),
        rms_error=np.vstack(rms) if rms else empty,
        mean_error=np.vstack(mean) if mean else empty,
        valid_fraction=np.vstack(vf) if vf else empty,
        safety_altered_fraction=np.vstack(af) if af else empty,
        torque_gain_ratio=np.vstack(tg) if tg else empty,
    )


@dataclass(frozen=True)
class Baseline:
    """Nominal per-joint reference for ONE command regime, from nominal runs only.

    ``s_hat`` of each nominal window against the pooled reference gives the nominal
    spread of the estimator; the per-joint alarm threshold is a low quantile of that
    spread, but never closer to 1 than ``1 - min_effect`` so the monitor does not
    alarm on changes too small to matter for training.
    """

    regime: str
    reference_rms: list[float]  # R_j, pooled over nominal windows
    s_hat_quantile_lo: list[float]  # per joint, the ``alpha`` quantile of nominal s_hat
    threshold: list[float]  # per joint alarm threshold on s_hat
    n_windows: list[int]  # nominal windows that contributed, per joint
    alpha: float
    min_effect: float
    window: dict[str, Any] = field(default_factory=dict)
    source: list[str] = field(default_factory=list)  # provenance of the nominal runs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Baseline:
        return cls(**d)


MIN_BASELINE_WINDOWS = 20


def calibrate(
    stats: list[WindowStats],
    regime: str,
    cfg: WindowConfig = DEFAULT_WINDOW,
    alpha: float = 0.01,
    min_effect: float = 0.10,
    source: list[str] | None = None,
) -> Baseline:
    """Fit a :class:`Baseline` from nominal-run window statistics.

    Raises when any joint has fewer than :data:`MIN_BASELINE_WINDOWS` usable
    windows: a threshold from a handful of windows is not a calibration.
    """
    if not stats:
        raise ValueError("no nominal runs")
    rms = np.vstack([s.rms_error for s in stats if s.n_windows])
    if rms.size == 0:
        raise ValueError("nominal runs contain no windows")
    ref, qlo, thr, counts = [], [], [], []
    for j in range(N_JOINTS):
        r = rms[:, j]
        r = r[np.isfinite(r)]
        counts.append(int(r.size))
        if r.size < MIN_BASELINE_WINDOWS:
            raise ValueError(
                f"joint {j}: {r.size} usable nominal windows, need {MIN_BASELINE_WINDOWS}"
            )
        r = np.maximum(r, cfg.noise_floor_rad)
        rj = float(np.sqrt(np.mean(np.square(r))))
        s_nom = rj / r
        q = float(np.quantile(s_nom, alpha))
        ref.append(rj)
        qlo.append(q)
        thr.append(float(min(q, 1.0 - min_effect)))
    return Baseline(
        regime=regime,
        reference_rms=ref,
        s_hat_quantile_lo=qlo,
        threshold=thr,
        n_windows=counts,
        alpha=alpha,
        min_effect=min_effect,
        window=asdict(cfg),
        source=list(source or []),
    )


def authority_ratio(
    stats: WindowStats, baseline: Baseline, cfg: WindowConfig = DEFAULT_WINDOW
) -> np.ndarray:
    """``s_hat = R_j / rms_j`` per window and joint; NaN where the window was not usable."""
    ref = np.asarray(baseline.reference_rms, dtype=np.float64)
    rms = np.maximum(stats.rms_error, cfg.noise_floor_rad)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = ref[None, :] / rms
    return np.where(np.isfinite(stats.rms_error), s, np.nan)


__all__ = [
    "DEFAULT_WINDOW",
    "MIN_BASELINE_WINDOWS",
    "Baseline",
    "WindowConfig",
    "WindowStats",
    "authority_ratio",
    "calibrate",
    "window_stats",
]
