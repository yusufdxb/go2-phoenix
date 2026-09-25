"""Performance-gated command-range curriculum: the pure-python stepping rule.

The Isaac Lab curriculum term (:class:`phoenix.velocity.isaac_terms.VelocityCommandCurriculum`)
only gathers per-episode statistics from the env and calls :meth:`CommandCurriculum.observe`;
every decision is made here, so the rule is unit-tested in CI without Isaac.

Rule (all numbers in :class:`phoenix.velocity.spec.CurriculumSpec`):

* ranges(level) = initial + (final - initial) * level / num_levels, per bound;
* finished episodes are accumulated into a window; once the window holds
  ``window_episodes`` episodes it is evaluated and cleared;
* level += 1 iff lin score >= threshold AND yaw score >= threshold AND
  termination rate <= max; otherwise the level is held (never lowered);
* a score normalizes the mean tracking kernel against a robot that never moves
  on the CURRENT ranges: ``(k - k_still) / (1 - k_still)``.

The manifest records the initial and final ranges, the widest ranges reached
(as a contract :class:`CommandRanges`) and the rule parameters.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .contract import CommandRanges
from .spec import CommandSpec, CurriculumSpec

RULE_NAME = "phoenix-velocity-curriculum/v1: linear levels, windowed score gate, monotone"


def interpolate_ranges(
    initial: CommandRanges, final: CommandRanges, level: int, num_levels: int
) -> CommandRanges:
    """Ranges at ``level`` in ``[0, num_levels]`` (clamped). Level 0 = initial, max = final."""
    if num_levels < 1:
        raise ValueError("num_levels must be >= 1")
    frac = min(max(level, 0), num_levels) / num_levels

    def lerp(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
        lo = a[0] + (b[0] - a[0]) * frac
        hi = a[1] + (b[1] - a[1]) * frac
        # Exact endpoints at the ends: no float drift past the final range.
        if frac >= 1.0:
            return float(b[0]), float(b[1])
        return float(lo), float(hi)

    return CommandRanges(
        lerp(initial.lin_vel_x, final.lin_vel_x),
        lerp(initial.lin_vel_y, final.lin_vel_y),
        lerp(initial.ang_vel_z, final.ang_vel_z),
        float(final.rel_standing_envs),
    )


def uniform_exp_kernel_mean(lo: float, hi: float, std: float) -> float:
    """E[exp(-x^2 / std^2)] for x ~ U[lo, hi]; the tracking kernel of a zero output."""
    if std <= 0:
        raise ValueError("std must be positive")
    if hi < lo:
        raise ValueError("hi < lo")
    if hi - lo < 1e-12:
        return math.exp(-(lo * lo) / (std * std))
    return (std * math.sqrt(math.pi) / 2.0) * (math.erf(hi / std) - math.erf(lo / std)) / (hi - lo)


def standstill_tracking_baseline(ranges: CommandRanges, std: float) -> tuple[float, float]:
    """(lin, yaw) mean tracking kernel of a policy that never moves, on ``ranges``.

    Standing envs (probability ``rel_standing_envs``) have a zero command, so a
    still robot scores 1 on them. vx and vy are independent, and the lin kernel
    factorizes: exp(-(vx^2 + vy^2)/s^2) = exp(-vx^2/s^2) exp(-vy^2/s^2).
    """
    p = ranges.rel_standing_envs
    kx = uniform_exp_kernel_mean(*ranges.lin_vel_x, std)
    ky = uniform_exp_kernel_mean(*ranges.lin_vel_y, std)
    kz = uniform_exp_kernel_mean(*ranges.ang_vel_z, std)
    return p + (1.0 - p) * kx * ky, p + (1.0 - p) * kz


def normalized_score(measured: float, baseline: float) -> float:
    """0 = no better than standing still, 1 = perfect. Can go negative."""
    if baseline >= 1.0 - 1e-9:
        return 1.0 if measured >= baseline - 1e-9 else 0.0
    return (measured - baseline) / (1.0 - baseline)


@dataclass
class WindowStats:
    episodes: int = 0
    terminated: int = 0
    lin_kernel_sum: float = 0.0
    yaw_kernel_sum: float = 0.0

    def add(self, episodes: int, terminated: int, lin_sum: float, yaw_sum: float) -> None:
        if episodes < 0 or terminated < 0 or terminated > episodes:
            raise ValueError(f"bad episode counts: episodes={episodes} terminated={terminated}")
        for v in (lin_sum, yaw_sum):
            if not math.isfinite(v):
                raise ValueError(f"non-finite tracking sum {v}")
        self.episodes += int(episodes)
        self.terminated += int(terminated)
        self.lin_kernel_sum += float(lin_sum)
        self.yaw_kernel_sum += float(yaw_sum)

    @property
    def lin_kernel(self) -> float:
        return self.lin_kernel_sum / self.episodes if self.episodes else 0.0

    @property
    def yaw_kernel(self) -> float:
        return self.yaw_kernel_sum / self.episodes if self.episodes else 0.0

    @property
    def termination_rate(self) -> float:
        return self.terminated / self.episodes if self.episodes else 1.0


@dataclass(frozen=True)
class Decision:
    evaluated: bool
    expanded: bool
    level: int
    lin_score: float = float("nan")
    yaw_score: float = float("nan")
    termination_rate: float = float("nan")
    reason: str = ""


@dataclass
class CommandCurriculum:
    """Stateful, monotone, performance-gated command curriculum."""

    initial: CommandRanges
    final: CommandRanges
    spec: CurriculumSpec
    tracking_std: float
    window_episodes: int
    level: int = 0
    evaluations: int = 0
    total_episodes: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    window: WindowStats = field(default_factory=WindowStats)
    last_decision: Decision | None = None

    @classmethod
    def from_spec(
        cls, commands: CommandSpec, spec: CurriculumSpec, tracking_std: float, num_envs: int
    ) -> CommandCurriculum:
        cur = cls(
            initial=commands.initial_ranges(),
            final=commands.final_ranges(),
            spec=spec,
            tracking_std=float(tracking_std),
            window_episodes=spec.window_episodes(num_envs),
        )
        if not spec.enabled:
            cur.level = spec.num_levels  # no curriculum: train on the final ranges
        return cur

    @property
    def max_level(self) -> int:
        return self.spec.num_levels

    @property
    def ranges(self) -> CommandRanges:
        return interpolate_ranges(self.initial, self.final, self.level, self.max_level)

    def widest_ranges(self) -> CommandRanges:
        """Widest ranges ever sampled. Levels are monotone, so this is the current one."""
        return self.ranges

    def observe(
        self, episodes: int, terminated: int, lin_kernel_sum: float, yaw_kernel_sum: float
    ) -> Decision:
        """Add finished episodes to the window; evaluate the gate when the window is full.

        Args:
            episodes: number of episodes that just ended.
            terminated: how many of them ended by a failure termination (not time-out).
            lin_kernel_sum / yaw_kernel_sum: sum over those episodes of each episode's
                time-averaged tracking kernel (each in [0, 1]).
        """
        self.total_episodes += int(episodes)
        if self.level >= self.max_level or not self.spec.enabled:
            self.last_decision = Decision(False, False, self.level, reason="at final ranges")
            return self.last_decision
        self.window.add(episodes, terminated, lin_kernel_sum, yaw_kernel_sum)
        if self.window.episodes < self.window_episodes:
            self.last_decision = Decision(False, False, self.level, reason="window filling")
            return self.last_decision

        base_lin, base_yaw = standstill_tracking_baseline(self.ranges, self.tracking_std)
        lin_score = normalized_score(self.window.lin_kernel, base_lin)
        yaw_score = normalized_score(self.window.yaw_kernel, base_yaw)
        term_rate = self.window.termination_rate
        failures = []
        if lin_score < self.spec.lin_score_threshold:
            failures.append(f"lin score {lin_score:.3f} < {self.spec.lin_score_threshold}")
        if yaw_score < self.spec.yaw_score_threshold:
            failures.append(f"yaw score {yaw_score:.3f} < {self.spec.yaw_score_threshold}")
        if term_rate > self.spec.max_termination_rate:
            failures.append(f"termination rate {term_rate:.3f} > {self.spec.max_termination_rate}")
        expanded = not failures
        self.evaluations += 1
        if expanded:
            self.level += 1
            self.history.append(
                {
                    "level": self.level,
                    "at_episode": self.total_episodes,
                    "lin_score": lin_score,
                    "yaw_score": yaw_score,
                    "termination_rate": term_rate,
                    "ranges": self.ranges.to_dict(),
                }
            )
        self.window = WindowStats()
        self.last_decision = Decision(
            True,
            expanded,
            self.level,
            lin_score,
            yaw_score,
            term_rate,
            "expanded" if expanded else "; ".join(failures),
        )
        return self.last_decision

    def rule_dict(self) -> dict[str, Any]:
        return {
            "rule": RULE_NAME,
            "tracking_std": self.tracking_std,
            "window_episodes": self.window_episodes,
            **asdict(self.spec),
        }

    def manifest_dict(self) -> dict[str, Any]:
        """The ``curriculum`` block of the checkpoint manifest."""
        return {
            "initial_ranges": self.initial.to_dict(),
            "final_ranges": self.final.to_dict(),
            "widest_ranges_reached": self.widest_ranges().to_dict(),
            "level": self.level,
            "max_level": self.max_level,
            "evaluations": self.evaluations,
            "total_episodes": self.total_episodes,
            "level_history": list(self.history),
            "stepping_rule": self.rule_dict(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "evaluations": self.evaluations,
            "total_episodes": self.total_episodes,
            "history": list(self.history),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        level = int(state["level"])
        if not 0 <= level <= self.max_level:
            raise ValueError(f"curriculum level {level} outside [0, {self.max_level}]")
        self.level = level
        self.evaluations = int(state.get("evaluations", 0))
        self.total_episodes = int(state.get("total_episodes", 0))
        self.history = list(state.get("history", []))
        self.window = WindowStats()


__all__ = [
    "RULE_NAME",
    "CommandCurriculum",
    "Decision",
    "WindowStats",
    "interpolate_ranges",
    "normalized_score",
    "standstill_tracking_baseline",
    "uniform_exp_kernel_mean",
]
