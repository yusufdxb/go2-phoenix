"""Episode outcome taxonomy and the per-episode verdict.

The legacy evaluator (``phoenix.training.evaluate`` before 2026-09-22) counted
an episode as a success when Isaac Lab's ``time_out`` term fired, and its V2
outcome wrote ``success = time_out and not terminated`` with the attitude
events and ``intervention_required`` sitting next to it, unused. So an episode
that "needed" an attitude intervention was still a success, and a stand-only
policy that merely survived 20 s was a locomotion success.

Here the two questions are separate:

* :class:`Outcome` says HOW the episode ended (one label, fixed priority);
* :func:`evaluate_episode` says whether the BEHAVIOR was acceptable
  (PASS / WARN / FAIL from :mod:`phoenix.evaluation.thresholds`).

``locomotion_success`` is True only when the episode survived to its planned
end (``timeout`` or ``completed``) AND the verdict is PASS. A timeout alone is
never success.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from . import thresholds as th
from .metrics import EpisodeMetrics


class Outcome(str, Enum):
    #: The harness ended the episode at its planned end (hardware authority
    #: window, scripted command profile). Survival only, not success.
    COMPLETED = "completed"
    #: Isaac Lab's ``time_out`` term fired. Survival only, not success.
    TIMEOUT = "timeout"
    TERMINATED_BAD_ORIENTATION = "terminated_bad_orientation"
    TERMINATED_CONTACT = "terminated_contact"
    TERMINATED_JOINT_LIMIT = "terminated_joint_limit"
    #: Tilted past the sim fall bar but the simulator did not terminate it
    #: (Isaac Lab's GO2 task has no orientation termination).
    FELL_WITHOUT_TERMINATION = "fell_without_termination"
    NUMERICAL_FAILURE = "numerical_failure"
    #: A safety layer took authority from the policy (hardware fault latch,
    #: illegal target abort, or a sim attitude past the 0.40 rad hardware
    #: intervention).
    SAFETY_INTERVENTION = "safety_intervention"
    #: The safety layer rewrote more than the FAIL fraction of commands.
    EXCESSIVE_INTERVENTION = "excessive_intervention"
    OTHER_TERMINATION = "other_termination"


SURVIVED = frozenset({Outcome.COMPLETED, Outcome.TIMEOUT})

#: Isaac Lab termination-term name fragments -> outcome. Matched lowercase,
#: first hit wins, so the more specific fragments come first.
_TERM_MAP: tuple[tuple[str, Outcome], ...] = (
    ("orientation", Outcome.TERMINATED_BAD_ORIENTATION),
    ("joint_pos", Outcome.TERMINATED_JOINT_LIMIT),
    ("joint_limit", Outcome.TERMINATED_JOINT_LIMIT),
    ("contact", Outcome.TERMINATED_CONTACT),
    ("height", Outcome.TERMINATED_BAD_ORIENTATION),
)


def classify_termination(termination_terms: Iterable[str], timed_out: bool) -> Outcome | None:
    """Outcome implied by simulator termination terms alone. ``None``: none fired."""
    terms = [t for t in termination_terms if t]
    failure_terms = [t for t in terms if t != "time_out"]
    for term in failure_terms:
        low = term.lower()
        for fragment, outcome in _TERM_MAP:
            if fragment in low:
                return outcome
    if failure_terms:
        return Outcome.OTHER_TERMINATION
    if timed_out or "time_out" in terms:
        return Outcome.TIMEOUT
    return None


@dataclass
class EpisodeEvaluation:
    outcome: Outcome
    verdict: str
    reasons: list[str]
    locomotion_success: bool
    survived: bool
    metrics: dict[str, Any]
    #: The pre-2026-09-22 fields, reproduced so old and new results sit side by
    #: side. ``legacy_success`` is exactly the old ``time_out`` rule.
    legacy: dict[str, Any] = field(default_factory=dict)
    evaluator_consistent: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d


def evaluate_episode(
    metrics: EpisodeMetrics,
    *,
    termination_terms: Sequence[str] = (),
    timed_out: bool = False,
    harness_completed: bool = False,
    safety_events: Sequence[str] = (),
    evaluator_consistent: bool = True,
    evaluator_problems: Sequence[str] = (),
    commanded_motion: bool | None = None,
    legacy: dict[str, Any] | None = None,
) -> EpisodeEvaluation:
    """Classify one episode and grade its behavior.

    Args:
        metrics: from :func:`phoenix.evaluation.metrics.compute_episode_metrics`.
        termination_terms: simulator termination terms that fired at the end.
        timed_out: the simulator's ``time_out`` flag for this episode.
        harness_completed: the harness (not the simulator) ended the episode at
            its planned end, e.g. a hardware authority window.
        safety_events: hardware safety-layer events (fault latches, aborts) that
            took authority from the policy. Any entry is a safety intervention.
        evaluator_consistent / evaluator_problems: the evaluator's self-checks.
        commanded_motion: whether this episode asked the robot to move. Tracking
            is graded either way (a zero command must be held too), progress
            ratios only when motion was commanded. ``None`` infers it from the
            integrated command.
    """
    reasons: list[str] = []
    levels: list[str] = []

    def fail(reason: str) -> None:
        reasons.append(f"FAIL: {reason}")
        levels.append(th.FAIL)

    def band(b: th.Band, value: float | None, label: str) -> None:
        level = b.level(value)
        if level != th.PASS:
            reasons.append(f"{level}: {label}={value!r} ({b.name} warn {b.warn} fail {b.fail})")
        levels.append(level)

    # ---- hard FAIL rules ----------------------------------------------------
    if not evaluator_consistent or evaluator_problems:
        fail("evaluator inconsistency: " + "; ".join(evaluator_problems or ["unspecified"]))
    if metrics.non_finite:
        fail(f"non-finite values in {metrics.non_finite_fields}")
    term_outcome = classify_termination(termination_terms, timed_out)
    fell = metrics.max_tilt_rad > th.SIM_FALL_TILT_RAD
    if term_outcome is not None and term_outcome not in SURVIVED:
        fail(f"terminated: {term_outcome.value} ({list(termination_terms)})")
    if fell:
        fail(f"fall: max tilt {metrics.max_tilt_rad:.3f} rad > {th.SIM_FALL_TILT_RAD}")
    illegal = metrics.raw.illegal_target_count
    if illegal:
        fail(f"{illegal} illegal joint target(s) beyond hard limit + {th.ILLEGAL_TARGET_BAND_RAD}")
    if safety_events:
        fail(f"safety layer took authority: {list(safety_events)}")
    survived_flag = term_outcome in SURVIVED or harness_completed
    planned = metrics.planned_steps
    if not survived_flag:
        fail("episode did not reach its planned end")
    elif planned is not None and metrics.n_steps < th.MIN_DURATION_FRACTION * planned:
        fail(f"ran {metrics.n_steps} of {planned} planned steps")

    # ---- banded metrics -----------------------------------------------------
    attitude_level = th.ATTITUDE.level(metrics.max_tilt_rad)
    band(th.ATTITUDE, metrics.max_tilt_rad, "max_tilt_rad")
    band(th.MODIFICATION_RATE, metrics.executed.modification_rate, "modification_rate")
    band(th.ACTION_SATURATION, metrics.raw.action_saturation_rate, "action_saturation_rate")
    band(
        th.DEPLOY_SLEW_ACTIVATION,
        metrics.raw.deploy_slew_activation_rate,
        "deploy_slew_activation_rate",
    )
    band(th.LIN_VEL_TRACKING, metrics.lin_vel_xy_rmse_mps, "lin_vel_xy_rmse_mps")
    band(th.YAW_RATE_TRACKING, metrics.yaw_rate_rmse_radps, "yaw_rate_rmse_radps")
    band(
        th.TRACKING_DEVIATION,
        metrics.executed.tracking_deviation_rms_rad,
        "executed_target_vs_measured_rms_rad",
    )
    if commanded_motion is None:
        commanded_motion = (
            metrics.commanded_distance_m >= th.MIN_COMMANDED_DISTANCE_M
            or abs(metrics.commanded_yaw_rad) >= th.MIN_COMMANDED_YAW_RAD
        )
    if commanded_motion:
        if metrics.commanded_distance_m >= th.MIN_COMMANDED_DISTANCE_M:
            ratio = metrics.distance_ratio
            band(
                th.PROGRESS_RATIO_ERROR,
                None if ratio is None else abs(ratio - 1.0),
                "distance_ratio_error",
            )
        if abs(metrics.commanded_yaw_rad) >= th.MIN_COMMANDED_YAW_RAD:
            ratio = metrics.yaw_ratio
            band(
                th.PROGRESS_RATIO_ERROR,
                None if ratio is None else abs(ratio - 1.0),
                "yaw_ratio_error",
            )

    verdict = th.worst(*levels)

    # ---- outcome label (fixed priority) ------------------------------------
    mod = metrics.executed.modification_rate
    if metrics.non_finite:
        outcome = Outcome.NUMERICAL_FAILURE
    elif term_outcome is not None and term_outcome not in SURVIVED:
        outcome = term_outcome
    elif fell:
        outcome = Outcome.FELL_WITHOUT_TERMINATION
    elif safety_events or illegal or attitude_level == th.FAIL:
        outcome = Outcome.SAFETY_INTERVENTION
    elif mod is not None and th.MODIFICATION_RATE.level(mod) == th.FAIL:
        outcome = Outcome.EXCESSIVE_INTERVENTION
    elif term_outcome == Outcome.TIMEOUT:
        outcome = Outcome.TIMEOUT
    elif harness_completed:
        outcome = Outcome.COMPLETED
    else:
        outcome = Outcome.OTHER_TERMINATION

    survived = outcome in SURVIVED
    return EpisodeEvaluation(
        outcome=outcome,
        verdict=verdict,
        reasons=reasons,
        locomotion_success=bool(survived and verdict == th.PASS),
        survived=survived,
        metrics=metrics.to_dict(),
        legacy=dict(legacy or {}),
        evaluator_consistent=bool(evaluator_consistent and not evaluator_problems),
    )


def summarize_evaluations(evals: Sequence[EpisodeEvaluation]) -> dict[str, Any]:
    """Run-level summary. Counts per outcome and verdict; nothing is collapsed."""
    n = len(evals)
    by_outcome: dict[str, int] = {o.value: 0 for o in Outcome}
    by_verdict: dict[str, int] = {lv: 0 for lv in th.LEVELS}
    for e in evals:
        by_outcome[e.outcome.value] += 1
        by_verdict[e.verdict] += 1
    run_verdict = th.worst(*(e.verdict for e in evals)) if evals else th.FAIL
    return {
        "num_episodes": n,
        "outcome_counts": by_outcome,
        "verdict_counts": by_verdict,
        "survival_rate": (sum(e.survived for e in evals) / n) if n else None,
        "locomotion_success_rate": (sum(e.locomotion_success for e in evals) / n) if n else None,
        "run_verdict": run_verdict,
        "evaluator_consistent": all(e.evaluator_consistent for e in evals),
        "note": (
            "survival_rate counts timeout/completed episodes and is NOT success. "
            "locomotion_success requires survival AND a PASS behavioral verdict."
        ),
    }


__all__ = [
    "EpisodeEvaluation",
    "Outcome",
    "SURVIVED",
    "classify_termination",
    "evaluate_episode",
    "summarize_evaluations",
]
