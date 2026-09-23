"""Detector v2: a hierarchical test for a persistent actuator-response shift.

EXPERIMENT.md amendment 17. One preregistered attempt. Detector v1
(:mod:`phoenix.monitor.response_shift`) failed its frozen gate at three of five criteria
(false-flag 0.125 against 0.05, detection 0.708 against 0.80, extent 0.375 against 0.70)
while its severity estimator passed both of its. This module changes the decision rule
and reuses the estimator unchanged.

What was wrong with v1
----------------------
v1 thresholded twelve groups independently, each against its own per-window quantile, and
reported the largest that flagged. Two consequences, both visible in its gate:

* **Multiple-comparison load.** Twelve independent tests per session. All three of its
  false alarms were groups the intervention never touched (``front``, ``hips``,
  ``leg_FL``).
* **Calibration at the wrong level.** Thresholds were quantiles of the pooled WINDOW
  distribution, but the gate measures a SESSION-level false-alarm rate. Windows within a
  session are not independent, so a 5 % window quantile did not give a 5 % session rate;
  the observed per-window alarm rate on held-out nominal was 0.105 to 0.351 against the
  0.05 the threshold was set for.

What v2 does instead
--------------------
**Stage 1, detection only.** One statistic: the median over all twelve joints of
``s_hat`` in each window, then the median of that over the session's windows. A session is
SHIFTED when that sits below a threshold calibrated as the ``alpha`` quantile of the same
statistic over NOMINAL DEVELOPMENT SESSIONS. Calibrating at session level is what makes
the false-alarm rate mean what the gate measures. Taking the median over windows IS the
persistence rule: a session flags only when more than half its windows are below the
bound, so a handful of bad windows cannot fire it.

**Stage 2, extent, only if stage 1 fired.** Five physically meaningful groups, fixed in
advance: the whole robot and its two complementary bisections. Each is standardised against its own nominal development
distribution, ``z_g = (mu_g - S_g) / sigma_g``, so groups of different sizes are
comparable. A single FAMILY-WISE threshold is the ``1 - alpha_fw`` quantile of
``max_g z_g`` over the nominal development sessions: the null distribution of the most
extreme group, which is what "pick the best group" actually tests. The reported group is
``argmax z_g`` subject to ``z_g > tau_fw``; if nothing clears it the extent is
UNRESOLVED and no group is named.

Note what argmax over standardised evidence does on its own, with no size preference
wired in: a group's statistic is a median over its members, so a whole-robot change moves
``all`` fully while a rear-only change moves ``all`` only halfway, and the larger group
also has the smaller ``sigma_g``. The right extent wins on evidence rather than by a
tie-break rule. **Nothing here privileges the rear legs**, and the specificity condition
of amendment 17 exists to check exactly that.

Declared limitation, stated before the gate
-------------------------------------------
Stage 1's statistic is a median over all twelve joints. A median only moves once at least
half the joints have moved, so **v2 detects an actuator-response shift affecting at least
half the robot, and cannot detect one confined to a single leg** (three of twelve), at any
severity. This is a property of the aggregate, not a tuning choice, and it is why the
stage-2 space contains no single-leg hypothesis. v2 is therefore not a general fault
detector. The intervention this study selected affects exactly six of twelve, as does the
specificity condition it is checked against.

**Severity is v1's estimator, unchanged**, read off the selected group: it passed both
halves of its criterion (median bias 0.0746 at an applied 0.70, interval covering truth in
76 % of detected sessions) and is not redesigned.

This module estimates and reports. It is not in any control path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..sim2real.go2_model import UNITREE_MOTOR_ORDER
from .residual import Baseline, WindowStats, authority_ratio

N_JOINTS = len(UNITREE_MOTOR_ORDER)
_IDX = {n: i for i, n in enumerate(UNITREE_MOTOR_ORDER)}


def _legs(*prefixes: str) -> tuple[int, ...]:
    return tuple(_IDX[j] for j in UNITREE_MOTOR_ORDER if j.split("_")[0] in prefixes)


#: The stage-2 hypothesis space, frozen. FIVE groups: the whole robot, and its two
#: complementary anatomical bisections. Every one of them is REACHABLE by stage 1, which
#: is the criterion for inclusion (see the declared limitation below).
#:
#: Deliberately excluded, to hold the multiple-comparison load down:
#:
#: * the four individual legs. A leg is three of twelve joints, and a median over twelve
#:   does not move when a quarter of them do, at ANY severity. Stage 1 can therefore never
#:   fire on a single-leg change, so a single-leg hypothesis is unreachable and would spend
#:   family-wise budget (raising ``tau_fw`` for every other group) while never being
#:   selectable. The screen also eliminated the one-leg family as an intervention: C2
#:   `leg_RR` reached a walking-success drop of 0.1250 against the 0.15 bar.
#: * the diagonal pairs and the per-joint classes (all hips, all thighs, all calves), which
#:   no screened intervention family produced.
#: * the twelve singletons, which the screen eliminated outright and which are the noisiest
#:   candidates in this regime.
V2_GROUPS: dict[str, tuple[int, ...]] = {
    "all": _legs("FR", "FL", "RR", "RL"),
    "front": _legs("FR", "FL"),
    "rear": _legs("RR", "RL"),
    "left": _legs("FL", "RL"),
    "right": _legs("FR", "RR"),
}

#: Stage 1 uses every joint.
ALL_JOINTS: tuple[int, ...] = tuple(range(N_JOINTS))


class V2State(str, Enum):
    NOMINAL = "nominal"
    SHIFTED = "shifted"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class V2Config:
    """Frozen before any validation session exists. Nothing is fitted to degraded data."""

    #: Stage-1 session-level false-alarm target: the quantile of the nominal development
    #: session statistic used as the detection threshold.
    alpha: float = 0.05
    #: Stage-1 threshold is never closer to 1 than this, so the detector cannot fire on a
    #: shift too small to be worth adapting to however tight the calibration looks.
    min_effect: float = 0.04
    #: Stage-2 family-wise error rate across the five groups, via the max statistic.
    alpha_fw: float = 0.05
    #: Quantile over a session's windows that forms the session statistic. 0.5 makes the
    #: persistence rule "more than half the session's windows are below the bound".
    window_quantile: float = 0.5
    #: Fewest usable windows before any verdict other than INSUFFICIENT_DATA.
    min_usable: int = 8
    #: Floor on a group's nominal spread, so a freakishly tight group cannot produce a
    #: huge z from a trivial difference.
    min_sigma: float = 0.01


DEFAULT_V2_CONFIG = V2Config()


@dataclass(frozen=True)
class V2Baseline:
    regime: str
    cfg: V2Config
    #: Stage 1.
    tau_global: float
    nominal_global_median: float
    nominal_global_quantile: float
    #: Stage 2, per group.
    mu: dict[str, float]
    sigma: dict[str, float]
    tau_fw: float
    groups: tuple[str, ...]
    sessions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "phoenix-response-shift-v2-baseline/v1",
            "regime": self.regime,
            "stage1": {
                "tau_global": self.tau_global,
                "nominal_median": self.nominal_global_median,
                "nominal_quantile": self.nominal_global_quantile,
            },
            "stage2": {
                "groups": list(self.groups),
                "mu": dict(self.mu),
                "sigma": dict(self.sigma),
                "tau_fw": self.tau_fw,
            },
            "config": {
                "alpha": self.cfg.alpha,
                "min_effect": self.cfg.min_effect,
                "alpha_fw": self.cfg.alpha_fw,
                "window_quantile": self.cfg.window_quantile,
                "min_usable": self.cfg.min_usable,
                "min_sigma": self.cfg.min_sigma,
            },
            "sessions": list(self.sessions),
        }


@dataclass(frozen=True)
class V2Report:
    state: str
    global_shift: float
    tau_global: float
    usable: int
    #: Stage 2. ``None`` when stage 1 did not fire, or when no group cleared ``tau_fw``.
    group: str | None
    group_z: float | None
    tau_fw: float
    #: Severity of the selected group, v1's estimator unchanged.
    severity: float | None
    lo: float | None
    hi: float | None
    z_by_group: dict[str, float] | None = None

    @property
    def shifted(self) -> bool:
        return self.state == V2State.SHIFTED.value

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "phoenix-response-shift-v2/v1",
            "state": self.state,
            "shifted": self.shifted,
            "global_shift": self.global_shift,
            "tau_global": self.tau_global,
            "usable_windows": self.usable,
            "group": self.group,
            "group_z": self.group_z,
            "tau_fw": self.tau_fw,
            "severity": self.severity,
            "range": None if self.lo is None else [self.lo, self.hi],
            "z_by_group": self.z_by_group,
        }


def _series(s_hat: np.ndarray, members: Sequence[int]) -> np.ndarray:
    """Per-window median of ``s_hat`` over a set of joints."""
    return np.nanmedian(s_hat[:, list(members)], axis=1)


def _session_stat(s_hat: np.ndarray, members: Sequence[int], q: float) -> float:
    v = _series(s_hat, members)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, q)) if v.size else float("nan")


def calibrate_v2(
    stats: Sequence[WindowStats],
    baseline: Baseline,
    *,
    regime: str,
    cfg: V2Config = DEFAULT_V2_CONFIG,
    groups: Mapping[str, tuple[int, ...]] | None = None,
    sessions: Sequence[str] = (),
) -> V2Baseline:
    """Both stages, from NOMINAL DEVELOPMENT sessions only.

    Stage 1's threshold is a session-level quantile, so it controls the quantity the gate
    measures. Stage 2's threshold is the family-wise max-statistic quantile, so "take the
    best group" is tested against the null distribution of the best group.
    """
    g = dict(groups or V2_GROUPS)
    if len(stats) < 4:
        raise ValueError(f"calibration needs at least 4 nominal sessions, got {len(stats)}")

    per_session_global: list[float] = []
    per_session_group: dict[str, list[float]] = {k: [] for k in g}
    for st in stats:
        s = authority_ratio(st, baseline)
        per_session_global.append(_session_stat(s, ALL_JOINTS, cfg.window_quantile))
        for name, members in g.items():
            per_session_group[name].append(_session_stat(s, members, cfg.window_quantile))

    gl = np.asarray([v for v in per_session_global if np.isfinite(v)], float)
    if gl.size < 4:
        raise ValueError("too few usable nominal sessions for stage 1")
    q = float(np.quantile(gl, cfg.alpha))
    tau_global = float(min(q, 1.0 - cfg.min_effect))

    mu, sigma = {}, {}
    for name, vals in per_session_group.items():
        arr = np.asarray([v for v in vals if np.isfinite(v)], float)
        if arr.size < 4:
            raise ValueError(f"too few usable nominal sessions for group {name!r}")
        mu[name] = float(arr.mean())
        sigma[name] = float(max(arr.std(ddof=1), cfg.min_sigma))

    # Family-wise null: the most extreme group in each nominal development session.
    maxima = []
    for i in range(len(stats)):
        zs = [
            (mu[name] - per_session_group[name][i]) / sigma[name]
            for name in g
            if np.isfinite(per_session_group[name][i])
        ]
        if zs:
            maxima.append(max(zs))
    if len(maxima) < 4:
        raise ValueError("too few usable nominal sessions for the family-wise threshold")
    tau_fw = float(np.quantile(np.asarray(maxima, float), 1.0 - cfg.alpha_fw))

    return V2Baseline(
        regime=regime,
        cfg=cfg,
        tau_global=tau_global,
        nominal_global_median=float(np.median(gl)),
        nominal_global_quantile=q,
        mu=mu,
        sigma=sigma,
        tau_fw=tau_fw,
        groups=tuple(g),
        sessions=tuple(str(x) for x in sessions),
    )


def assess_v2(
    st: WindowStats,
    baseline: Baseline,
    v2: V2Baseline,
    *,
    groups: Mapping[str, tuple[int, ...]] | None = None,
) -> V2Report:
    """Score one session. Stage 2 runs only if stage 1 fires."""
    g = dict(groups or V2_GROUPS)
    cfg = v2.cfg
    s = authority_ratio(st, baseline)
    usable = int(np.isfinite(_series(s, ALL_JOINTS)).sum())
    gshift = _session_stat(s, ALL_JOINTS, cfg.window_quantile)

    if usable < cfg.min_usable or not np.isfinite(gshift):
        return V2Report(
            state=V2State.INSUFFICIENT_DATA.value,
            global_shift=gshift,
            tau_global=v2.tau_global,
            usable=usable,
            group=None,
            group_z=None,
            tau_fw=v2.tau_fw,
            severity=None,
            lo=None,
            hi=None,
        )

    if gshift >= v2.tau_global:  # stage 1 says nominal; stage 2 never runs
        return V2Report(
            state=V2State.NOMINAL.value,
            global_shift=gshift,
            tau_global=v2.tau_global,
            usable=usable,
            group=None,
            group_z=None,
            tau_fw=v2.tau_fw,
            severity=None,
            lo=None,
            hi=None,
        )

    z_by = {}
    for name, members in g.items():
        stat = _session_stat(s, members, cfg.window_quantile)
        if np.isfinite(stat) and name in v2.mu:
            z_by[name] = float((v2.mu[name] - stat) / v2.sigma[name])
    eligible = {k: v for k, v in z_by.items() if v > v2.tau_fw}
    chosen = max(eligible, key=lambda k: eligible[k]) if eligible else None

    severity = lo = hi = None
    if chosen is not None:
        series = _series(s, g[chosen])
        series = series[np.isfinite(series)]
        severity = float(np.median(series))
        lo = float(np.percentile(series, 2.5))
        hi = float(np.percentile(series, 97.5))

    return V2Report(
        state=V2State.SHIFTED.value,
        global_shift=gshift,
        tau_global=v2.tau_global,
        usable=usable,
        group=chosen,
        group_z=None if chosen is None else z_by[chosen],
        tau_fw=v2.tau_fw,
        severity=severity,
        lo=lo,
        hi=hi,
        z_by_group=z_by,
    )


__all__ = [
    "ALL_JOINTS",
    "DEFAULT_V2_CONFIG",
    "V2_GROUPS",
    "V2Baseline",
    "V2Config",
    "V2Report",
    "V2State",
    "assess_v2",
    "calibrate_v2",
]
