"""Fail-fast training-health guard for PhoenixVelocity PPO runs.

Pure python (no torch / Isaac at import), so every decision is unit-tested in CI.
``scripts/train_velocity.py`` feeds it one :class:`HealthSample` per rsl_rl
iteration (read from the runner's own logger call) and stops the run when it
returns ``ABORT``. Thresholds and their seed-46 provenance live in
:class:`phoenix.velocity.spec.HealthGuardSpec`.

Why: walk_v2 seed 46 ran 9000 iterations; its best checkpoints were 1000-1500
and everything after ~2000 was degradation, then collapse (episode length 843
-> 7). Nothing stopped it. The signals below move in that order on seed 46:
action p99 (onset ~2000), learning rate at the adaptive floor (~3200), std
(~3750), episode length (~6200).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .spec import HealthGuardSpec

OK = "OK"
WARN = "WARN"
ABORT = "ABORT"


class TrainingHealthAbortError(RuntimeError):
    """Raised by the training hook when the guard returns ABORT."""

    def __init__(self, verdict: HealthVerdict) -> None:
        super().__init__("; ".join(verdict.abort_reasons))
        self.verdict = verdict


@dataclass(frozen=True)
class HealthSample:
    """One iteration's health signals. ``None`` = not available this iteration."""

    iteration: int
    std_max: float | None = None  # max over action dims of the policy std
    std_mean: float | None = None
    action_p99: float | None = None  # Curriculum/telemetry/action_abs_p99
    episode_length: float | None = None  # rsl_rl mean episode length, policy steps
    kl: float | None = None  # mean KL over the update's minibatches
    learning_rate: float | None = None
    curriculum_level: float | None = None


@dataclass(frozen=True)
class HealthVerdict:
    status: str
    iteration: int
    abort_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    scalars: dict[str, float] = field(default_factory=dict)


@dataclass
class _Streak:
    count: int = 0
    first_iteration: int | None = None

    def step(self, breached: bool, iteration: int) -> int:
        if breached:
            if self.count == 0:
                self.first_iteration = iteration
            self.count += 1
        else:
            self.count = 0
            self.first_iteration = None
        return self.count


class TrainingHealthGuard:
    """Stateful per-iteration guard. Call :meth:`update` once per PPO iteration."""

    def __init__(self, spec: HealthGuardSpec, desired_kl: float | None) -> None:
        self.spec = spec
        self.desired_kl = desired_kl
        self._streaks = {k: _Streak() for k in ("std", "action_p99", "episode_length", "lr", "kl")}
        self.max_level_seen = 0.0
        self.history_tail: list[dict[str, Any]] = []

    def _abort_if_sustained(
        self,
        key: str,
        breached: bool,
        iteration: int,
        sustain: int,
        message: str,
        reasons: list[str],
    ) -> float:
        n = self._streaks[key].step(breached, iteration)
        if n >= sustain:
            first = self._streaks[key].first_iteration
            # ``reasons`` is the abort list for hard signals, the warning list for
            # signals the spec marks warn-only (``*_hard = False``).
            reasons.append(f"{message} for {n} consecutive iterations (since {first})")
        return float(n)

    def update(self, sample: HealthSample) -> HealthVerdict:
        s = self.spec
        it = int(sample.iteration)
        reasons: list[str] = []
        warnings: list[str] = []
        scalars: dict[str, float] = {}

        values = asdict(sample)
        values.pop("iteration")
        for name, v in values.items():
            if v is None:
                continue
            if not math.isfinite(float(v)):
                reasons.append(f"{name} is non-finite ({v})")
            else:
                scalars[name] = float(v)

        armed = it >= s.warmup_iterations

        std = sample.std_max
        if std is not None and math.isfinite(std):
            if armed and std > s.std_warn:
                warnings.append(f"std max {std:.4f} > warn {s.std_warn}")
            scalars["std_streak"] = self._abort_if_sustained(
                "std",
                armed and std > s.std_abort,
                it,
                s.sustain_iterations,
                f"policy std max > {s.std_abort} (now {std:.4f})",
                reasons,
            )

        p99 = sample.action_p99
        if p99 is not None and math.isfinite(p99):
            if armed and p99 > s.action_p99_warn:
                warnings.append(f"action p99 {p99:.3f} > warn {s.action_p99_warn}")
            scalars["action_p99_streak"] = self._abort_if_sustained(
                "action_p99",
                armed and p99 > s.action_p99_abort,
                it,
                s.sustain_iterations,
                f"action |a| p99 > {s.action_p99_abort} (now {p99:.3f})",
                reasons if s.action_p99_hard else warnings,
            )

        ep = sample.episode_length
        if ep is not None and math.isfinite(ep):
            if armed and ep < s.episode_length_warn:
                warnings.append(f"episode length {ep:.1f} < warn {s.episode_length_warn}")
            scalars["episode_length_streak"] = self._abort_if_sustained(
                "episode_length",
                armed and ep < s.episode_length_abort,
                it,
                s.sustain_iterations,
                f"mean episode length < {s.episode_length_abort} steps (now {ep:.1f})",
                reasons,
            )

        lr = sample.learning_rate
        if lr is not None and math.isfinite(lr):
            scalars["lr_streak"] = self._abort_if_sustained(
                "lr",
                armed and lr <= s.lr_abort,
                it,
                s.lr_sustain_iterations,
                f"learning rate <= {s.lr_abort:g} (now {lr:.3g}; KL repeatedly > 2 x desired)",
                reasons if s.lr_hard else warnings,
            )

        kl = sample.kl
        if kl is not None and math.isfinite(kl) and self.desired_kl:
            limit = s.kl_abort_factor * self.desired_kl
            scalars["kl_over_desired"] = kl / self.desired_kl
            scalars["kl_streak"] = self._abort_if_sustained(
                "kl",
                armed and kl > limit,
                it,
                s.kl_sustain_iterations,
                f"KL > {limit:g} (now {kl:.4g})",
                reasons if s.kl_hard else warnings,
            )

        level = sample.curriculum_level
        if level is not None and math.isfinite(level):
            self.max_level_seen = max(self.max_level_seen, level)
            if it >= s.curriculum_stall_warn_iteration and self.max_level_seen <= 0:
                warnings.append(
                    f"curriculum still at level 0 at iteration {it} "
                    f"(>= {s.curriculum_stall_warn_iteration}); envelope = level-0 ranges"
                )

        status = ABORT if reasons else (WARN if warnings else OK)
        scalars["status"] = {OK: 0.0, WARN: 1.0, ABORT: 2.0}[status]
        verdict = HealthVerdict(status, it, tuple(reasons), tuple(warnings), scalars)
        self.history_tail.append({"iteration": it, "status": status, **scalars})
        del self.history_tail[:-200]
        return verdict

    def report(self, verdict: HealthVerdict) -> dict[str, Any]:
        """JSON-able abort record (written as ``health_abort.json``)."""
        return {
            "status": verdict.status,
            "iteration": verdict.iteration,
            "abort_reasons": list(verdict.abort_reasons),
            "warnings": list(verdict.warnings),
            "thresholds": asdict(self.spec),
            "desired_kl": self.desired_kl,
            "last_iterations": list(self.history_tail),
        }


__all__ = [
    "ABORT",
    "OK",
    "WARN",
    "HealthSample",
    "HealthVerdict",
    "TrainingHealthAbortError",
    "TrainingHealthGuard",
]
