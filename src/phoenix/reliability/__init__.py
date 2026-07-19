"""Runtime reliability layer for a learned GO2 policy.

This package builds the *predictive* counterpart to the rule-based
:class:`phoenix.real_world.failure_detector.FailureDetector`. Where the
rule-based detector fires once a failure is already observable (attitude
loss, body collapse, foot slip), the reliability layer scores how
out-of-distribution the policy's own internal state is *before* the
output visibly degrades, so a Simplex arbiter can hand ``/cmd_vel`` to a
safe fallback with lead time to spare.

Phase 1 (this module set) is the OOD monitor core: pure-numpy scorers
(Mahalanobis, KNN) fit on nominal rollouts only, plus a temporal filter.
It has no torch / Isaac / ROS dependency so it stays unit-testable in CI.
The Simplex arbiter (hysteresis + dwell + bounded ramp) and the Isaac-twin
validation live in later phases.
"""

from __future__ import annotations

from phoenix.reliability.arbiter import (
    ArbiterOutput,
    ShieldState,
    SimplexArbiter,
    SimplexArbiterCfg,
)
from phoenix.reliability.features import (
    forward_hidden,
    policy_features,
)
from phoenix.reliability.metrics import (
    average_precision,
    bootstrap_ci,
    fpr_at_tpr,
    lead_time_seconds,
    roc_auc,
    threshold_at_fpr,
)
from phoenix.reliability.ood_monitor import (
    KNNScorer,
    MahalanobisScorer,
    TemporalFilter,
    ledoit_wolf_shrinkage,
)
from phoenix.reliability.runtime import (
    ShieldDecision,
    ShieldRuntime,
    calibrate_arbiter_thresholds,
)

__all__ = [
    "KNNScorer",
    "MahalanobisScorer",
    "TemporalFilter",
    "ledoit_wolf_shrinkage",
    "roc_auc",
    "average_precision",
    "fpr_at_tpr",
    "threshold_at_fpr",
    "bootstrap_ci",
    "lead_time_seconds",
    "policy_features",
    "forward_hidden",
    "SimplexArbiter",
    "SimplexArbiterCfg",
    "ShieldState",
    "ArbiterOutput",
    "ShieldRuntime",
    "ShieldDecision",
    "calibrate_arbiter_thresholds",
]
