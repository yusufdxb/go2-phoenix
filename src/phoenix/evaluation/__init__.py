"""Phoenix evaluation: outcome taxonomy, behavioral metrics, verdict thresholds.

Pure numpy, importable in CI. See ``docs/forensics/evaluation_repair.md`` for
why this replaced the old "time_out means success" rule.
"""

from .metrics import EpisodeMetrics, EpisodeTrace, compute_episode_metrics
from .orientation import (
    ISAACLAB_ROOT_QUAT_CONVENTION,
    Attitude,
    OrientationError,
    attitude_from_quat,
    check_against_projected_gravity,
    to_wxyz,
)
from .outcomes import EpisodeEvaluation, Outcome, evaluate_episode, summarize_evaluations
from .thresholds import FAIL, PASS, WARN

__all__ = [
    "FAIL",
    "ISAACLAB_ROOT_QUAT_CONVENTION",
    "PASS",
    "WARN",
    "Attitude",
    "EpisodeEvaluation",
    "EpisodeMetrics",
    "EpisodeTrace",
    "OrientationError",
    "Outcome",
    "attitude_from_quat",
    "check_against_projected_gravity",
    "compute_episode_metrics",
    "evaluate_episode",
    "summarize_evaluations",
    "to_wxyz",
]
