"""Persistent actuator-response change, measured over joint GROUPS.

Phase H reframing (EXPERIMENT.md amendment 14). :mod:`phoenix.monitor.health` asks
"which single motor is bad" and answers per joint, with ``GLOBAL_SHIFT`` existing only
as a veto: when more than ``max_localised`` joints qualify it refuses to say anything,
and :func:`phoenix.condition.distribution.build_targeted_spec` then refuses outright.
That is the wrong question for this study, twice over.

**It is unanswerable in the walking regime.** Measured on nominal W2 walking telemetry,
the per-joint ratio ``s_hat`` has a nominal session-to-session spread far larger than the
effect being looked for: undegraded joints sit anywhere from 0.75 to 1.63, because
``s_hat`` assumes identical task and load and walking violates that (the commanded
velocity is resampled, gait phase varies, and W2 commands bang-bang joint targets, so
much of the residual is the plant low-passing a fast command rather than missing
authority). With a single joint truly at 0.80, it was the lowest-scoring joint in a
minority of sessions.

**And the intervention is not sparse.** The screening eliminated the single-joint family,
so what has to be detected is a uniform reduction over a group of joints.

What this module does instead
-----------------------------
For each group in a small, FIXED, physically meaningful hypothesis space (the same
:data:`phoenix.sim2real.degradation.JOINT_GROUPS` the intervention itself is drawn from,
plus the twelve singletons), it takes the median of ``s_hat`` over the group's joints in
each window, and the median of that over windows. A group of size ``m`` cuts the
per-joint noise roughly as ``1/sqrt(m)`` while leaving a uniform reduction untouched.

Because the group statistic is a MEDIAN over members, a group only flags when a MAJORITY
of its joints moved. That makes the **largest** flagged group the right answer for extent:

* a global reduction moves all twelve, so ``all`` flags (and so does every subgroup);
* a one-leg reduction moves three of twelve, so ``all`` (3 of 12) does not flag.

A median is only a majority test when the majority is strict, so a group exactly half of
whose members moved (``rear``, when one of the two rear legs is degraded) has its median
pulled to the midpoint and would flag on level alone. A second, threshold-free term
rejects that: a group represents ONE uniform reduction only if its members moved
TOGETHER, so the interquartile spread of the member joints' own medians must stay inside
its calibrated nominal range. Under a leg-only reduction ``rear`` splits into three moved
and three unmoved members and its spread blows up, while ``leg_RR`` stays tight.

How well that term discriminates on real walking telemetry is MEASURED in the Phase I
calibration, not assumed here: the nominal per-joint spread in this regime is itself
large. If it fails to separate, the monitor over-reports extent, which widens the
targeted distribution rather than pointing it somewhere false.

This module estimates and reports. It does not decide policy, does not touch a motor, and
is not in any control path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..sim2real.degradation import JOINT_GROUPS
from ..sim2real.go2_model import UNITREE_MOTOR_ORDER
from .residual import Baseline, WindowStats, authority_ratio

N_JOINTS = len(UNITREE_MOTOR_ORDER)


def candidate_groups(min_size: int = 1) -> dict[str, tuple[int, ...]]:
    """The fixed hypothesis space: named physical groups, plus singletons when allowed.

    Fixed in advance and small, so the monitor is not searching ``2**12`` subsets for
    whichever one happens to look worst. Every candidate is tested against the same
    calibrated threshold, so each one costs false-alarm budget: with two dozen candidates
    and a per-candidate alarm rate of ``alpha``, a session flags something far more often
    than ``alpha``.

    ``min_size`` drops candidates smaller than that. ``min_size=2`` leaves only the
    physical groups, which is the right space when the interventions under study are all
    multi-joint: the singletons are both the noisiest candidates in the walking regime
    and hypotheses no screened family can produce. Restricting the space is a change to
    the detector and belongs in a preregistration, not in a threshold sweep.
    """
    idx = {n: i for i, n in enumerate(UNITREE_MOTOR_ORDER)}
    groups = {name: tuple(idx[j] for j in js) for name, js in JOINT_GROUPS.items()}
    groups.update({n: (idx[n],) for n in UNITREE_MOTOR_ORDER})
    return {k: v for k, v in groups.items() if len(v) >= min_size}


CANDIDATE_GROUPS: dict[str, tuple[int, ...]] = candidate_groups()
#: The multi-joint-only space (no singletons); see :func:`candidate_groups`.
MULTI_JOINT_GROUPS: dict[str, tuple[int, ...]] = candidate_groups(min_size=2)


class ShiftState(str, Enum):
    NOMINAL = "nominal"
    SHIFTED = "shifted"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class ShiftConfig:
    """Frozen before the validation gate. Nothing here is fitted to degraded data."""

    #: Quantile of the NOMINAL group-shift distribution used as the alarm threshold.
    #: Lower is stricter (fewer false alarms, later detection).
    alpha: float = 0.05
    #: A threshold is never closer to 1 than this, so a group cannot be flagged for a
    #: change smaller than the smallest one worth adapting to, however tight its
    #: nominal spread happens to look on a finite calibration set.
    min_effect: float = 0.06
    #: Windows a group must be below its threshold, out of the last ``n``.
    n: int = 10
    k_of_n: int = 7
    #: Fewest usable windows before any verdict other than INSUFFICIENT_DATA.
    min_usable: int = 8
    #: Quantile of the NOMINAL member-spread distribution above which a group is judged
    #: incoherent (its members did not move together), so it cannot be the explanation
    #: however far its median has fallen.
    spread_quantile: float = 0.90
    #: Floor on that bound, so a group is never called incoherent for a spread smaller
    #: than the ordinary per-joint noise of this regime.
    min_spread_bound: float = 0.10


DEFAULT_SHIFT_CONFIG = ShiftConfig()


@dataclass(frozen=True)
class GroupBaseline:
    """Per-group nominal threshold, calibrated on nominal sessions only."""

    regime: str
    groups: tuple[str, ...]
    threshold: dict[str, float]
    nominal_median: dict[str, float]
    nominal_quantile: dict[str, float]
    #: Upper bound on the member-spread coherence term, per group.
    spread_bound: dict[str, float]
    cfg: ShiftConfig
    sessions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "phoenix-group-baseline/v1",
            "regime": self.regime,
            "groups": list(self.groups),
            "threshold": dict(self.threshold),
            "nominal_median": dict(self.nominal_median),
            "nominal_quantile": dict(self.nominal_quantile),
            "spread_bound": dict(self.spread_bound),
            "config": {
                "alpha": self.cfg.alpha,
                "min_effect": self.cfg.min_effect,
                "n": self.cfg.n,
                "k_of_n": self.cfg.k_of_n,
                "min_usable": self.cfg.min_usable,
                "spread_quantile": self.cfg.spread_quantile,
                "min_spread_bound": self.cfg.min_spread_bound,
            },
            "sessions": list(self.sessions),
        }


@dataclass(frozen=True)
class GroupShift:
    group: str
    joints: tuple[str, ...]
    state: str
    shift: float
    lo: float
    hi: float
    threshold: float
    below: int
    usable: int
    #: Interquartile spread of the member joints' own medians. Phase H's "consistency
    #: across joints": a genuine uniform reduction moves the members together, so this
    #: stays small. A group whose spread exceeds ``spread_bound`` is incoherent and
    #: cannot be the explanation, however far its median has fallen.
    member_spread: float
    spread_bound: float
    coherent: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "group": self.group,
            "joints": list(self.joints),
            "state": self.state,
            "shift": self.shift,
            "lo": self.lo,
            "hi": self.hi,
            "threshold": self.threshold,
            "below": self.below,
            "usable": self.usable,
            "member_spread": self.member_spread,
            "spread_bound": self.spread_bound,
            "coherent": self.coherent,
        }


@dataclass(frozen=True)
class ShiftReport:
    groups: tuple[GroupShift, ...]
    #: The largest flagged group, i.e. the widest extent the evidence supports. ``None``
    #: when nothing flagged.
    selected: GroupShift | None
    per_joint_median: tuple[float, ...] = field(default=())

    @property
    def shifted(self) -> bool:
        return self.selected is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "phoenix-response-shift/v1",
            "shifted": self.shifted,
            "selected": None if self.selected is None else self.selected.to_dict(),
            "groups": [g.to_dict() for g in self.groups],
            "per_joint_median": list(self.per_joint_median),
        }


def _group_series(s_hat: np.ndarray, members: Sequence[int]) -> np.ndarray:
    """Per-window median of ``s_hat`` over a group's joints; NaN where unusable."""
    return np.nanmedian(s_hat[:, list(members)], axis=1)


def _member_spread(per_joint_median: np.ndarray, members: Sequence[int]) -> float:
    """Interquartile spread of the member joints' own session medians.

    IQR rather than max-min so one wild joint (the walking regime produces them) cannot
    declare an otherwise coherent group incoherent. A singleton has no spread.
    """
    if len(members) < 2:
        return 0.0
    v = per_joint_median[list(members)]
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    return float(np.percentile(v, 75) - np.percentile(v, 25))


def calibrate_groups(
    stats: Sequence[WindowStats],
    baseline: Baseline,
    *,
    regime: str,
    cfg: ShiftConfig = DEFAULT_SHIFT_CONFIG,
    groups: Mapping[str, tuple[int, ...]] | None = None,
    sessions: Sequence[str] = (),
) -> GroupBaseline:
    """Per-group alarm thresholds from NOMINAL sessions only.

    The threshold is the ``alpha`` quantile of the nominal per-window group shift,
    floored so it is never closer to 1 than ``min_effect``.
    """
    g = dict(groups or CANDIDATE_GROUPS)
    if not stats:
        raise ValueError("calibration needs at least one nominal session")
    per_group: dict[str, list[float]] = {name: [] for name in g}
    spreads: dict[str, list[float]] = {name: [] for name in g}
    for st in stats:
        s = authority_ratio(st, baseline)
        pj = np.nanmedian(s, axis=0)
        for name, members in g.items():
            per_group[name].extend(v for v in _group_series(s, members) if np.isfinite(v))
            sp = _member_spread(pj, members)
            if np.isfinite(sp):
                spreads[name].append(sp)

    thr, med, qnt, sbound = {}, {}, {}, {}
    for name, vals in per_group.items():
        if not vals:
            raise ValueError(f"no usable nominal windows for group {name!r}")
        arr = np.asarray(vals, float)
        q = float(np.quantile(arr, cfg.alpha))
        qnt[name] = q
        med[name] = float(np.median(arr))
        thr[name] = float(min(q, 1.0 - cfg.min_effect))
        sp = spreads[name]
        sbound[name] = (
            max(float(np.quantile(sp, cfg.spread_quantile)), cfg.min_spread_bound)
            if sp
            else cfg.min_spread_bound
        )
    return GroupBaseline(
        regime=regime,
        groups=tuple(g),
        threshold=thr,
        nominal_median=med,
        nominal_quantile=qnt,
        spread_bound=sbound,
        cfg=cfg,
        sessions=tuple(str(s) for s in sessions),
    )


def assess_shift(
    st: WindowStats,
    baseline: Baseline,
    group_baseline: GroupBaseline,
    *,
    groups: Mapping[str, tuple[int, ...]] | None = None,
) -> ShiftReport:
    """Score one session against a calibrated group baseline.

    Selection rule, fixed: among flagged groups take the one with the most joints, and
    between equal sizes the one whose shift sits furthest below its threshold. Because
    each group statistic is a median over its members, a group flags only when a
    majority of its joints moved, so the largest flagged group is the widest extent the
    evidence supports rather than the most extreme-looking subset.
    """
    g = dict(groups or CANDIDATE_GROUPS)
    s = authority_ratio(st, baseline)
    cfg = group_baseline.cfg
    rows: list[GroupShift] = []
    per_joint = np.nanmedian(s, axis=0)

    for name, members in g.items():
        series = _group_series(s, members)
        usable_mask = np.isfinite(series)
        usable = int(usable_mask.sum())
        thr = group_baseline.threshold[name]
        vals = series[usable_mask]
        recent = vals[-cfg.n :] if usable else vals
        below = int((recent < thr).sum())
        spread = _member_spread(per_joint, members)
        bound = group_baseline.spread_bound.get(name, cfg.min_spread_bound)
        coherent = not np.isfinite(spread) or spread <= bound
        if usable < cfg.min_usable:
            state = ShiftState.INSUFFICIENT_DATA
        elif below >= min(cfg.k_of_n, len(recent)) and coherent:
            state = ShiftState.SHIFTED
        else:
            state = ShiftState.NOMINAL
        rows.append(
            GroupShift(
                group=name,
                joints=tuple(UNITREE_MOTOR_ORDER[i] for i in members),
                state=state.value,
                shift=float(np.nanmedian(vals)) if usable else float("nan"),
                lo=float(np.nanpercentile(vals, 2.5)) if usable else float("nan"),
                hi=float(np.nanpercentile(vals, 97.5)) if usable else float("nan"),
                threshold=thr,
                below=below,
                usable=usable,
                member_spread=spread,
                spread_bound=bound,
                coherent=bool(coherent),
            )
        )

    flagged = [r for r in rows if r.state == ShiftState.SHIFTED.value]
    selected = (
        max(flagged, key=lambda r: (len(r.joints), r.threshold - r.shift)) if flagged else None
    )
    rows.sort(key=lambda r: (-len(r.joints), r.group))
    return ShiftReport(
        groups=tuple(rows),
        selected=selected,
        per_joint_median=tuple(float(v) for v in per_joint),
    )


__all__ = [
    "CANDIDATE_GROUPS",
    "MULTI_JOINT_GROUPS",
    "DEFAULT_SHIFT_CONFIG",
    "GroupBaseline",
    "GroupShift",
    "ShiftConfig",
    "ShiftReport",
    "ShiftState",
    "assess_shift",
    "calibrate_groups",
    "candidate_groups",
]
