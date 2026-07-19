"""Out-of-distribution scorers for the policy's own internal state.

The reliability layer asks a narrow question every control step: *how
far is what the policy is seeing / computing right now from the nominal
data it was trained and validated on?* A high answer, sustained, is the
early-warning signal that lets the Simplex arbiter (Phase 2) hand off to
a safe fallback before the robot visibly fails.

Two complementary scorers, both fit on **nominal rollouts only** (never
on failures — that would leak the label and make the lead-time claim
circular):

* :class:`MahalanobisScorer` — squared Mahalanobis distance to the
  nominal feature cloud under a shrinkage covariance. Cheap (one
  triangular solve), parametric, great when the nominal cloud is roughly
  one blob. Shrinkage (:func:`ledoit_wolf_shrinkage`) keeps the
  covariance invertible when feature dim approaches sample count, which
  is exactly the regime for penultimate-layer activations.

* :class:`KNNScorer` — distance to the k-th nearest nominal sample in a
  PCA-whitened subspace. Non-parametric, catches multi-modal nominal
  structure (different gaits / command bins) that a single Gaussian
  smears over.

Design rules baked in for the eventual Orin NX deployment (codex review,
2026-07-16):

* Everything is fit in float64 offline; deploy constants are whatever
  dtype the caller stores. The scorers never allocate per-call beyond a
  couple of temporaries.
* A NaN or inf anywhere in the feature vector returns ``+inf`` — the
  monitor must fail *toward* SAFE, never silently pass a garbage frame.
* No sklearn / scipy / torch dependency, so this stays in the pure-python
  CI lane.

These scorers are feature-agnostic: feed them policy activations, the
observation vector, or a concatenation. "Latents vs obs, compared" (the
Phase 1 study) is just two fitted instances over different feature
sources, compared by the eval harness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_2d(x: np.ndarray) -> np.ndarray:
    """Return ``x`` as a ``(n, d)`` float64 array (1-D promoted to a row)."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D features, got shape {arr.shape}")
    return arr


def _finite_rows(x: np.ndarray) -> np.ndarray:
    """Boolean mask of rows in ``x`` (n, d) that are entirely finite."""
    return np.all(np.isfinite(x), axis=1)


def ledoit_wolf_shrinkage(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage covariance toward a scaled identity.

    Returns ``(cov, shrinkage)`` where ``cov = (1 - a) * S + a * mu * I``,
    ``S`` is the empirical covariance, ``mu = trace(S) / d`` is the average
    eigenvalue, and ``a in [0, 1]`` is the analytically optimal shrinkage
    intensity from Ledoit & Wolf (2004). Shrinking toward ``mu * I`` keeps
    the matrix well-conditioned and invertible even when the number of
    samples is comparable to the feature dimension.

    ``x`` is ``(n, d)``; non-finite rows are dropped first.
    """
    arr = _as_2d(x)
    arr = arr[_finite_rows(arr)]
    n, d = arr.shape
    if n < 2:
        raise ValueError("need at least 2 finite samples to estimate covariance")

    mean = arr.mean(axis=0)
    centered = arr - mean
    # Empirical (MLE, 1/n) covariance — matches the Ledoit-Wolf derivation.
    emp = (centered.T @ centered) / n
    mu = np.trace(emp) / d
    target = mu * np.eye(d)

    # Optimal shrinkage intensity. b2 is the mean squared Frobenius error
    # of the per-sample covariance around the empirical estimate; d2 is the
    # squared distance from the empirical estimate to the identity target.
    d2 = np.sum((emp - target) ** 2)
    b_bar = 0.0
    for row in centered:
        outer = np.outer(row, row)
        b_bar += np.sum((outer - emp) ** 2)
    b_bar /= n * n
    b2 = min(b_bar, d2)  # clamp so shrinkage stays in [0, 1]
    shrinkage = 0.0 if d2 == 0.0 else float(b2 / d2)

    cov = (1.0 - shrinkage) * emp + shrinkage * target
    return cov, shrinkage


@dataclass
class MahalanobisScorer:
    """Squared Mahalanobis distance to a nominal feature cloud.

    Fit on nominal features only. ``score`` returns
    ``(x - mean)^T @ precision @ (x - mean)`` per row; larger means more
    out-of-distribution. Non-finite input rows score ``+inf``.
    """

    mean: np.ndarray
    _chol: np.ndarray  # lower Cholesky factor of the (shrunk) covariance
    shrinkage: float

    @classmethod
    def fit(cls, features: np.ndarray, *, jitter: float = 1e-6) -> MahalanobisScorer:
        arr = _as_2d(features)
        cov, shrinkage = ledoit_wolf_shrinkage(arr)
        mean = arr[_finite_rows(arr)].mean(axis=0)
        d = cov.shape[0]
        # Cholesky enables an O(d^2) triangular solve at score time instead
        # of forming and storing an explicit inverse. Jitter guards the rare
        # case where shrinkage still leaves a marginally non-PD matrix.
        chol = np.linalg.cholesky(cov + jitter * np.eye(d))
        return cls(mean=mean, _chol=chol, shrinkage=shrinkage)

    def score(self, features: np.ndarray) -> np.ndarray:
        arr = _as_2d(features)
        out = np.full(arr.shape[0], np.inf, dtype=np.float64)
        finite = _finite_rows(arr)
        if np.any(finite):
            z = arr[finite] - self.mean
            # Solve L y = z^T  =>  squared Mahalanobis = ||y||^2 column-wise.
            y = np.linalg.solve(self._chol, z.T)
            out[finite] = np.sum(y * y, axis=0)
        return out

    def score_one(self, feature: np.ndarray) -> float:
        return float(self.score(feature)[0])


@dataclass
class KNNScorer:
    """Distance to the k-th nearest nominal sample in a whitened subspace.

    Standardizes features, projects onto the top ``n_components`` PCA
    directions, and whitens them (unit variance per axis). ``score``
    returns the Euclidean distance to the k-th nearest nominal reference
    point in that space. Non-finite input rows score ``+inf``.
    """

    _feat_mean: np.ndarray
    _feat_std: np.ndarray
    _components: np.ndarray  # (n_components, d)
    _whiten: np.ndarray  # (n_components,) 1/sqrt(singular-value energy)
    _reference: np.ndarray  # (m, n_components) whitened nominal points
    k: int

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        *,
        n_components: int = 32,
        k: int = 5,
        max_reference: int = 4000,
        seed: int = 0,
    ) -> KNNScorer:
        arr = _as_2d(features)
        arr = arr[_finite_rows(arr)]
        n, d = arr.shape
        if n <= k:
            raise ValueError(f"need more than k={k} finite samples, got {n}")

        feat_mean = arr.mean(axis=0)
        feat_std = arr.std(axis=0)
        feat_std[feat_std < 1e-8] = 1.0  # freeze dead dims instead of exploding them
        std = (arr - feat_mean) / feat_std

        n_comp = int(min(n_components, d, n - 1))
        # PCA via SVD of the standardized matrix.
        _, s, vt = np.linalg.svd(std, full_matrices=False)
        components = vt[:n_comp]
        # Whitening scale: singular values relate to per-component std by
        # sigma_i = s_i / sqrt(n). Divide the projection by that to get unit
        # variance per whitened axis.
        comp_std = s[:n_comp] / np.sqrt(n)
        comp_std[comp_std < 1e-8] = 1.0
        whiten = 1.0 / comp_std

        projected = (std @ components.T) * whiten

        # Subsample the reference set to bound score-time cost (codex: full
        # KNN reference gets annoying). Deterministic under ``seed``.
        if projected.shape[0] > max_reference:
            rng = np.random.default_rng(seed)
            idx = rng.choice(projected.shape[0], size=max_reference, replace=False)
            projected = projected[idx]

        return cls(
            _feat_mean=feat_mean,
            _feat_std=feat_std,
            _components=components,
            _whiten=whiten,
            _reference=np.ascontiguousarray(projected),
            k=int(k),
        )

    def _project(self, arr: np.ndarray) -> np.ndarray:
        std = (arr - self._feat_mean) / self._feat_std
        return (std @ self._components.T) * self._whiten

    def score(self, features: np.ndarray) -> np.ndarray:
        arr = _as_2d(features)
        out = np.full(arr.shape[0], np.inf, dtype=np.float64)
        finite = _finite_rows(arr)
        if np.any(finite):
            q = self._project(arr[finite])
            # Pairwise squared distances to the reference cloud, then the
            # k-th smallest per query row.
            d2 = (
                np.sum(q * q, axis=1)[:, None]
                - 2.0 * (q @ self._reference.T)
                + np.sum(self._reference * self._reference, axis=1)[None, :]
            )
            np.maximum(d2, 0.0, out=d2)  # numerical floor
            kth = np.partition(d2, self.k - 1, axis=1)[:, self.k - 1]
            out[finite] = np.sqrt(kth)
        return out

    def score_one(self, feature: np.ndarray) -> float:
        return float(self.score(feature)[0])


class TemporalFilter:
    """EWMA smoother with an optional CUSUM change detector.

    A single noisy OOD sample should not trip a safety handoff; a
    *sustained* rise should. This filter turns the raw per-step score into
    (a) an exponentially-weighted moving average and (b) a one-sided CUSUM
    statistic that accumulates evidence above a reference ``drift`` level
    and resets to zero otherwise. Phase 2's arbiter thresholds these, with
    hysteresis and dwell, to decide when to hand off.

    ``alpha`` is the EWMA weight on the newest sample (higher = faster,
    noisier). ``drift`` is the CUSUM reference (scores below it bleed the
    statistic back toward zero). Non-finite scores are treated as maximal
    evidence: the EWMA jumps to the input (``inf``) and CUSUM saturates.
    """

    def __init__(self, alpha: float = 0.3, drift: float = 0.0) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.drift = float(drift)
        self._ewma: float | None = None
        self._cusum: float = 0.0

    def reset(self) -> None:
        self._ewma = None
        self._cusum = 0.0

    def update(self, score: float) -> tuple[float, float]:
        """Feed one raw score; return ``(ewma, cusum)``."""
        s = float(score)
        if not np.isfinite(s):
            self._ewma = s
            self._cusum = np.inf
            return self._ewma, self._cusum
        self._ewma = s if self._ewma is None else self.alpha * s + (1.0 - self.alpha) * self._ewma
        self._cusum = max(0.0, self._cusum + (s - self.drift))
        return self._ewma, self._cusum

    @property
    def ewma(self) -> float | None:
        return self._ewma

    @property
    def cusum(self) -> float:
        return self._cusum
