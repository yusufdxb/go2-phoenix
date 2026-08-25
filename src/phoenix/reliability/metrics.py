"""Threshold-free evaluation for the OOD monitor, plus lead time.

The reliability layer lives or dies on two questions a skeptical reviewer
will ask:

1. *Separability without a hand-picked threshold*, how well does the raw
   score rank out-of-distribution frames above nominal ones? Answered with
   AUROC, average precision (AUPR), and FPR@95TPR, each with a bootstrap
   confidence interval so a lucky split can't masquerade as a result.

2. *Does it warn in time?*, the headline claim is not "we detect OOD" but
   "we detect it early enough to act". :func:`lead_time_seconds` measures
   the gap between the monitor first firing and the **independent**
   observable-failure oracle (the rule-based
   :class:`phoenix.real_world.failure_detector.FailureDetector`). Because
   the monitor never sees the oracle's signal, a positive lead time is not
   circular.

Thresholds are calibrated on **nominal data only** (:func:`threshold_at_fpr`)
,  never on the OOD set, so the operating point can't be tuned to the very
perturbations we then report on.

Pure numpy: no sklearn/scipy, so this stays in the CI lane. Convention:
``labels`` are 1 for out-of-distribution / positive, 0 for nominal; higher
``scores`` mean more out-of-distribution.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _check(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(int)
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} must match")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be 0 (nominal) or 1 (OOD)")
    if y.min() == y.max():
        raise ValueError("need both classes present to score")
    return s, y


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the tie-corrected Mann-Whitney U.

    Equivalent to the probability that a random OOD frame outranks a random
    nominal frame; 1.0 is perfect, 0.5 is chance. Average ranks handle ties.
    """
    s, y = _check(scores, labels)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks within tied score groups.
    s_sorted = s[order]
    i = 0
    n = len(s_sorted)
    while i < n:
        j = i + 1
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0  # average of ranks (i+1 .. j)
            ranks[order[i:j]] = avg
        i = j
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve).

    Uses the step-sum definition ``sum_n (R_n - R_{n-1}) * P_n`` over
    thresholds swept from the highest score down.
    """
    s, y = _check(scores, labels)
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    cum_tp = np.cumsum(y_sorted)
    cum_fp = np.cumsum(1 - y_sorted)
    n_pos = int(y.sum())
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    recall = cum_tp / n_pos
    prev_recall = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - prev_recall) * precision))


def fpr_at_tpr(scores: np.ndarray, labels: np.ndarray, tpr_target: float = 0.95) -> float:
    """False-positive rate at the threshold that achieves ``tpr_target``.

    The threshold is the score below which only ``1 - tpr_target`` of OOD
    frames fall (so a fraction ``tpr_target`` are caught); FPR is the share
    of nominal frames at or above it. Lower is better.
    """
    s, y = _check(scores, labels)
    pos = s[y == 1]
    neg = s[y == 0]
    thr = float(np.quantile(pos, 1.0 - tpr_target, method="lower"))
    return float(np.mean(neg >= thr))


def threshold_at_fpr(nominal_scores: np.ndarray, target_fpr: float = 0.05) -> float:
    """Operating threshold calibrated on NOMINAL scores only.

    Returns the upper ``target_fpr`` quantile of the nominal score
    distribution: at this threshold, a fraction ``target_fpr`` of nominal
    frames raise a (false) alarm. No OOD data touches this calibration.
    """
    nom = np.asarray(nominal_scores, dtype=np.float64).ravel()
    nom = nom[np.isfinite(nom)]
    if nom.size == 0:
        raise ValueError("no finite nominal scores to calibrate on")
    return float(np.quantile(nom, 1.0 - target_fpr, method="higher"))


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Class-stratified bootstrap CI for a threshold-free metric.

    Resamples nominal and OOD frames with replacement *within class* (so
    the positive/negative counts stay fixed) and returns
    ``(point_estimate, lo, hi)`` at the ``1 - alpha`` level.
    """
    s, y = _check(scores, labels)
    point = metric_fn(s, y)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pi = rng.choice(pos_idx, size=pos_idx.size, replace=True)
        ni = rng.choice(neg_idx, size=neg_idx.size, replace=True)
        idx = np.concatenate([pi, ni])
        vals[b] = metric_fn(s[idx], y[idx])
    lo = float(np.quantile(vals, alpha / 2.0))
    hi = float(np.quantile(vals, 1.0 - alpha / 2.0))
    return point, lo, hi


def lead_time_seconds(
    step_scores: np.ndarray,
    threshold: float,
    failure_step: int | None,
    dt: float,
    *,
    min_consecutive: int = 1,
) -> float | None:
    """Seconds between the monitor first firing and observable failure.

    ``step_scores`` is the per-step OOD score for one episode; ``threshold``
    is the (nominal-calibrated) operating point; ``failure_step`` is the
    index at which the independent oracle declared failure (``None`` if the
    episode never failed). The monitor "fires" at the first index that
    begins a run of ``min_consecutive`` scores strictly above ``threshold``.

    Returns:
      * positive seconds of warning if it fires before failure,
      * ``<= 0`` if it fires at or after failure (a late alarm),
      * ``None`` if it never fires before failure (a miss), or if the
        episode never failed and the monitor stayed quiet (a true negative
        the caller scores separately).
    """
    scores = np.asarray(step_scores, dtype=np.float64).ravel()
    above = scores > threshold
    fire_step: int | None = None
    if min_consecutive <= 1:
        hits = np.flatnonzero(above)
        if hits.size:
            fire_step = int(hits[0])
    else:
        run = 0
        for i, a in enumerate(above):
            run = run + 1 if a else 0
            if run >= min_consecutive:
                fire_step = i - min_consecutive + 1
                break

    if failure_step is None:
        return None if fire_step is None else -(len(scores) - fire_step) * dt
    if fire_step is None or fire_step >= failure_step:
        # Never fired, or fired only at/after the failure onset.
        return None if fire_step is None else (failure_step - fire_step) * dt
    return (failure_step - fire_step) * dt
