"""VERIFY and the redeploy decision: a candidate earns deployment or it does not.

Two entry points, one rule set.

:func:`compare_arms` (the scientific comparison, unit = independent training seed)
    For each condition, bootstrap the difference of arm means over training seeds.
    Used by the paper's matrix (docs/research/EXPERIMENT.md).

:func:`candidate_gate` (the deployment decision for ONE artifact)
    The candidate's and the incumbent's per-episode metric values on three
    evaluation conditions, plus the parity record of the exact candidate files.
    PROMOTE only if every rule holds:

    1. ``degraded``: the candidate improves the conditioned (degraded) condition by at
       least ``min_improvement`` and the bootstrap 95 % lower bound of the
       improvement is above zero.
    2. ``nominal``: non-inferiority, the 95 % lower bound of (candidate - incumbent)
       is at least ``-nominal_margin``. A policy that trades nominal locomotion for
       the degraded case is not deployed: the degradation may be repaired, or be a
       false positive.
    3. ``held_out``: at a nearby severity NOT used to build the training range, the
       point estimate is not worse than the incumbent by more than
       ``nominal_margin``. (Reported, and gating, so a narrow overfit is caught.)
    4. parity: the candidate's ONNX / TorchScript / checkpoint agree to
       ``parity_tol`` and the parity record names the same artifact hashes that
       were evaluated.
    5. provenance: the evaluation files name the candidate's checkpoint hash.

Every metric is "higher is better"; pass a negated value for costs. Thresholds
are frozen in :data:`PREREGISTERED`; changing them needs a dated note in
docs/research/EXPERIMENT.md. Nothing here runs a simulator: the inputs come from
``phoenix.training.evaluate`` runs, which need Isaac Lab.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

SCHEMA = "phoenix-candidate-gate/v1"
CONDITIONS = ("degraded", "nominal", "held_out")


@dataclass(frozen=True)
class GateRules:
    min_improvement: float = 0.05
    nominal_margin: float = 0.02
    parity_tol: float = 1e-5
    n_boot: int = 10_000
    ci: float = 0.95
    seed: int = 0


#: Frozen 2026-09-22 with docs/research/EXPERIMENT.md. They apply to the
#: preregistered primary endpoint, a per-episode score in [0, 1] (the fraction of the
#: episode spent inside the success envelope), so 0.05 = five points.
PREREGISTERED = GateRules()


def bootstrap_diff_ci(
    a: Sequence[float], b: Sequence[float], n_boot: int, ci: float, seed: int
) -> tuple[float, float, float]:
    """Mean(a) - mean(b) with a percentile bootstrap CI, resampling each group."""
    a_ = np.asarray(a, dtype=np.float64)
    b_ = np.asarray(b, dtype=np.float64)
    if a_.size < 2 or b_.size < 2:
        raise ValueError("need at least two values per group")
    if not (np.all(np.isfinite(a_)) and np.all(np.isfinite(b_))):
        raise ValueError("non-finite metric values")
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, a_.size, (n_boot, a_.size))
    ib = rng.integers(0, b_.size, (n_boot, b_.size))
    d = a_[ia].mean(axis=1) - b_[ib].mean(axis=1)
    lo, hi = np.quantile(d, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(a_.mean() - b_.mean()), float(lo), float(hi)


#: Two-sided 95 % Student t quantiles (standard table). Welch degrees of freedom are
#: floored to the next tabulated value, which only widens the interval.
_T975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    25: 2.060,
    30: 2.042,
    40: 2.021,
    60: 2.000,
    120: 1.980,
}


def _t975(df: float) -> float:
    keys = [k for k in sorted(_T975) if k <= df]
    return _T975[keys[-1]] if keys else _T975[1]


def welch_diff_ci(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    """Mean(a) - mean(b) with a 95 % Welch t interval; returns ``(diff, lo, hi, df)``.

    The primary interval for the seed-level comparison: with 5 to 10 seeds per arm a
    percentile bootstrap under-covers (it was measured at 0.85 to 0.89 for 5 seeds).
    """
    a_ = np.asarray(a, dtype=np.float64)
    b_ = np.asarray(b, dtype=np.float64)
    if a_.size < 2 or b_.size < 2:
        raise ValueError("need at least two values per group")
    va, vb = a_.var(ddof=1) / a_.size, b_.var(ddof=1) / b_.size
    se = float(np.sqrt(va + vb))
    d = float(a_.mean() - b_.mean())
    if se == 0.0:
        return d, d, d, float("inf")
    df = (va + vb) ** 2 / (va**2 / (a_.size - 1) + vb**2 / (b_.size - 1))
    h = _t975(df) * se
    return d, d - h, d + h, float(df)


def compare_arms(
    per_seed: Mapping[str, Mapping[str, Sequence[float]]],
    reference: str,
    rules: GateRules = PREREGISTERED,
) -> dict[str, Any]:
    """``per_seed[arm][condition]`` = one value per independent training seed.

    Returns, per non-reference arm and condition, the difference to ``reference``
    with its 95 % Welch t interval (primary), a percentile bootstrap interval
    (secondary) and a standardised effect size (difference / pooled SD).
    """
    if reference not in per_seed:
        raise KeyError(f"reference arm {reference!r} missing")
    out: dict[str, Any] = {
        "schema": SCHEMA + "/arms",
        "reference": reference,
        "rules": asdict(rules),
    }
    for arm, conds in per_seed.items():
        if arm == reference:
            continue
        out[arm] = {}
        for cond, vals in conds.items():
            ref_vals = per_seed[reference].get(cond)
            if ref_vals is None:
                continue
            d, lo, hi, df = welch_diff_ci(vals, ref_vals)
            _, blo, bhi = bootstrap_diff_ci(vals, ref_vals, rules.n_boot, rules.ci, rules.seed)
            sd = float(np.sqrt((np.var(vals, ddof=1) + np.var(ref_vals, ddof=1)) / 2))
            out[arm][cond] = {
                "diff": d,
                "ci": [lo, hi],
                "ci_method": "welch_t_95",
                "welch_df": df,
                "bootstrap_ci": [blo, bhi],
                "effect_size": d / sd if sd > 0 else None,
                "n": [len(vals), len(ref_vals)],
            }
    return out


def candidate_gate(
    candidate: Mapping[str, Sequence[float]],
    incumbent: Mapping[str, Sequence[float]],
    candidate_sha256: str,
    evaluated_sha256: Mapping[str, str],
    parity: Mapping[str, Any] | None,
    rules: GateRules = PREREGISTERED,
) -> dict[str, Any]:
    """PROMOTE / REJECT for one candidate artifact. See the module docstring."""
    reasons: list[str] = []
    results: dict[str, Any] = {}
    for cond in CONDITIONS:
        if cond not in candidate or cond not in incumbent:
            reasons.append(f"missing evaluation for condition {cond!r}")
            continue
        d, lo, hi = bootstrap_diff_ci(
            candidate[cond], incumbent[cond], rules.n_boot, rules.ci, rules.seed
        )
        results[cond] = {
            "diff": d,
            "ci": [lo, hi],
            "n": [len(candidate[cond]), len(incumbent[cond])],
        }
    if "degraded" in results:
        r = results["degraded"]
        if r["diff"] < rules.min_improvement:
            reasons.append(f"degraded improvement {r['diff']:.3f} < {rules.min_improvement}")
        if r["ci"][0] <= 0.0:
            reasons.append(f"degraded improvement CI lower bound {r['ci'][0]:.3f} <= 0")
    if "nominal" in results and results["nominal"]["ci"][0] < -rules.nominal_margin:
        reasons.append(
            f"nominal non-inferiority failed: CI lower bound {results['nominal']['ci'][0]:.3f} "
            f"< -{rules.nominal_margin}"
        )
    if "held_out" in results and results["held_out"]["diff"] < -rules.nominal_margin:
        reasons.append(
            f"held-out severity worse by {-results['held_out']['diff']:.3f} "
            f"(> {rules.nominal_margin})"
        )
    for cond in CONDITIONS:
        sha = evaluated_sha256.get(cond)
        if sha != candidate_sha256:
            reasons.append(f"{cond} evaluation names checkpoint {sha}, not the candidate")
    if parity is None:
        reasons.append("no parity record for the candidate")
    else:
        if parity.get("checkpoint_sha256") != candidate_sha256:
            reasons.append("parity record is for a different checkpoint")
        max_abs = parity.get("max_abs")
        if max_abs is None or not np.isfinite(max_abs) or max_abs > rules.parity_tol:
            reasons.append(f"parity max_abs {max_abs} exceeds {rules.parity_tol}")
    return {
        "schema": SCHEMA,
        "decision": "PROMOTE" if not reasons else "REJECT",
        "reasons": reasons,
        "candidate_sha256": candidate_sha256,
        "results": results,
        "rules": asdict(rules),
    }


__all__ = [
    "CONDITIONS",
    "PREREGISTERED",
    "SCHEMA",
    "GateRules",
    "bootstrap_diff_ci",
    "candidate_gate",
    "compare_arms",
    "welch_diff_ci",
]
