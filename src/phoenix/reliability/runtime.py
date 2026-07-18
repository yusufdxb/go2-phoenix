"""Deployable reliability runtime: monitor + filter + arbiter in one call.

This is the object the ROS 2 policy node holds. Every control tick it takes
the policy's extracted features (from :func:`phoenix.reliability.features`),
scores them against the nominal reference, smooths the score, and asks the
:class:`~phoenix.reliability.arbiter.SimplexArbiter` for a blend weight. The
node then mixes the learned and fallback controller outputs by that weight.

Keeping the whole reliability layer behind one pure-python object (no rclpy /
onnxruntime here, exactly like :mod:`phoenix.sim2real.mode_switch`) means the
end-to-end behavior — nominal stays on the learned policy, sustained OOD hands
off to the fallback — is unit-testable without Isaac or the robot. The ROS
node wiring is then a thin adapter, which is what makes the August hardware
bring-up plug-and-play.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phoenix.reliability.arbiter import ShieldState, SimplexArbiter
from phoenix.reliability.metrics import threshold_at_fpr
from phoenix.reliability.ood_monitor import TemporalFilter


@dataclass(frozen=True)
class ShieldDecision:
    blend: float  # weight on the fallback controller, [0, 1]
    state: ShieldState
    raw_score: float  # OOD score this tick, before smoothing
    filtered_score: float  # score actually fed to the arbiter

    @property
    def engaged(self) -> bool:
        return self.state is not ShieldState.NOMINAL


class ShieldRuntime:
    """Compose a fitted scorer, a temporal filter, and a Simplex arbiter.

    ``scorer`` is any fitted OOD scorer exposing ``score_one`` (Mahalanobis or
    KNN). ``arbiter`` is a constructed :class:`SimplexArbiter` whose trip /
    clear thresholds are on the same scale the filter emits (see
    :func:`calibrate_arbiter_thresholds`). ``feature_key`` selects which
    feature source to score when :meth:`step` is handed the ``{"latent",
    "obs"}`` dict from the feature extractor. The default temporal filter
    (``alpha=1``) is a pass-through, so the arbiter sees the raw score unless a
    smoothing filter is supplied.
    """

    def __init__(
        self,
        scorer,
        arbiter: SimplexArbiter,
        *,
        feature_key: str = "latent",
        temporal_filter: TemporalFilter | None = None,
    ) -> None:
        self.scorer = scorer
        self.arbiter = arbiter
        self.feature_key = feature_key
        self.filter = temporal_filter or TemporalFilter(alpha=1.0)

    def reset(self) -> None:
        self.arbiter.reset()
        self.filter.reset()

    def step(self, features) -> ShieldDecision:
        """Advance one control tick.

        ``features`` is either the ``{"latent", "obs"}`` dict for a single
        tick or a raw feature array (1-D, or 2-D with a single row).
        """
        feat = features[self.feature_key] if isinstance(features, dict) else features
        arr = np.asarray(feat, dtype=np.float64)
        if arr.ndim == 2:
            if arr.shape[0] != 1:
                raise ValueError("ShieldRuntime.step expects a single tick (one row)")
            row = arr[0]
        else:
            row = arr

        raw = self.scorer.score_one(row)
        filtered, _ = self.filter.update(raw)
        out = self.arbiter.update(filtered)
        return ShieldDecision(
            blend=out.blend,
            state=out.state,
            raw_score=raw,
            filtered_score=filtered,
        )


def calibrate_arbiter_thresholds(
    scorer,
    nominal_features: np.ndarray,
    *,
    trip_fpr: float = 0.01,
    clear_fpr: float = 0.2,
) -> tuple[float, float]:
    """Pick arbiter trip / clear thresholds from NOMINAL scores only.

    Returns ``(trip_threshold, clear_threshold)`` as upper quantiles of the
    nominal score distribution: ``trip`` is exceeded by a fraction
    ``trip_fpr`` of nominal frames (rare — engage conservatively) and
    ``clear`` by ``clear_fpr`` (looser — release generously). Requiring
    ``trip_fpr < clear_fpr`` guarantees ``trip > clear`` (the hysteresis gap
    the arbiter demands). No OOD data touches this calibration.
    """
    if not trip_fpr < clear_fpr:
        raise ValueError("trip_fpr must be < clear_fpr so trip_threshold > clear_threshold")
    scores = scorer.score(nominal_features)
    trip = threshold_at_fpr(scores, target_fpr=trip_fpr)
    clear = threshold_at_fpr(scores, target_fpr=clear_fpr)
    return trip, clear
