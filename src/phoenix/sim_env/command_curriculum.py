"""Symmetric velocity-command curriculum whose advancement needs BOTH directions (recipe W3).

W1/W2 learned backward walking and ignored forward commands; aggregate reward kept
rising on backward alone. This curriculum scales every command range by a stage factor
(stage 1 = 0.2 x the full ranges, then 0.4, 0.6, 1.0) and advances only when forward
AND backward tracking each reach a ratio, measured separately, so one direction cannot
carry the curriculum.

This module is pure Python (no torch, no Isaac Lab): the stage machine is unit-tested in
CI. :mod:`phoenix.sim_env.curriculum_command` feeds it from the live command term.

Tracking ratio of one finished command segment: ``mean achieved vx / commanded vx`` over
the segment after its first ``settle_steps``; only segments with
``|vx_cmd| >= min_cmd_frac * stage_max_vx`` count, split by the sign of the command.
Per direction, the ratio is the mean over the last ``window`` such segments, each clipped
to [-1, 2] so one runaway segment cannot dominate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CurriculumSpec:
    factors: tuple[float, ...] = (0.2, 0.4, 0.6, 1.0)
    full_max_vx: float = 1.0
    min_cmd_frac: float = 0.5
    ratio_threshold: float = 0.7
    window: int = 2000
    min_segments_per_direction: int = 500
    check_every_steps: int = 1200  # 50 PPO iterations x 24 steps
    min_stage_steps: int = 2400  # 100 iterations
    max_stage_steps: int = 24000  # 1000 iterations
    settle_steps: int = 50  # 1 s at 50 Hz
    consecutive_checks: int = 2

    def __post_init__(self) -> None:
        if list(self.factors) != sorted(self.factors) or self.factors[-1] != 1.0:
            raise ValueError("factors must increase and end at 1.0")
        if not 0.0 < self.ratio_threshold <= 1.0:
            raise ValueError("ratio_threshold must be in (0, 1]")


@dataclass
class CurriculumState:
    spec: CurriculumSpec = field(default_factory=CurriculumSpec)
    stage: int = 0
    stage_start_step: int = 0
    last_check_step: int = 0
    passes_in_a_row: int = 0
    timed_out: bool = False
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._fwd: deque[float] = deque(maxlen=self.spec.window)
        self._bwd: deque[float] = deque(maxlen=self.spec.window)

    @property
    def factor(self) -> float:
        return self.spec.factors[self.stage]

    @property
    def final(self) -> bool:
        return self.stage == len(self.spec.factors) - 1

    def record_segment(self, vx_cmd: float, mean_vx: float, settled_steps: int) -> None:
        """One finished command segment (called at resample or episode end)."""
        if settled_steps <= 0:
            return
        if abs(vx_cmd) < self.spec.min_cmd_frac * self.factor * self.spec.full_max_vx:
            return
        ratio = min(max(mean_vx / vx_cmd, -1.0), 2.0)
        (self._fwd if vx_cmd > 0 else self._bwd).append(ratio)

    def ratios(self) -> tuple[float | None, float | None]:
        f = sum(self._fwd) / len(self._fwd) if len(self._fwd) >= self.spec.min_segments_per_direction else None
        b = sum(self._bwd) / len(self._bwd) if len(self._bwd) >= self.spec.min_segments_per_direction else None
        return f, b

    def maybe_advance(self, step: int) -> bool:
        """Check at most every ``check_every_steps``; return True when the stage advanced."""
        if self.final or step - self.last_check_step < self.spec.check_every_steps:
            return False
        self.last_check_step = step
        f, b = self.ratios()
        in_stage = step - self.stage_start_step
        ok = f is not None and b is not None and f >= self.spec.ratio_threshold and b >= self.spec.ratio_threshold
        self.passes_in_a_row = self.passes_in_a_row + 1 if ok else 0
        self.history.append({"step": step, "stage": self.stage, "fwd": f, "bwd": b, "pass": ok})
        if in_stage >= self.spec.max_stage_steps and not ok:
            # Frozen rule: a stage that never qualifies stays put; the run is reported as
            # failing to advance, it is NOT promoted on time.
            self.timed_out = True
            return False
        if ok and self.passes_in_a_row >= self.spec.consecutive_checks and in_stage >= self.spec.min_stage_steps:
            self.stage += 1
            self.stage_start_step = step
            self.passes_in_a_row = 0
            self._fwd.clear()
            self._bwd.clear()
            self.history[-1]["advanced_to"] = self.stage
            return True
        return False


__all__ = ["CurriculumSpec", "CurriculumState"]
