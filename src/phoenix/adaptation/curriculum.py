"""Curriculum scheduler for failure-seeded adaptation.

The scheduler decides, each PPO iteration, which subset of parallel envs
should be reset from a *real failure seed* versus the usual random
spawn. Keeping this logic stateless and numpy-only means we can unit-test
it without Isaac Lab.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TrajectoryPool:
    """List of Parquet trajectories available as failure seeds."""

    paths: Sequence[Path]

    @classmethod
    def from_directory(cls, directory: str | Path, pattern: str = "*.parquet") -> TrajectoryPool:
        p = Path(directory)
        if not p.exists():
            return cls(paths=[])
        return cls(paths=sorted(p.glob(pattern)))

    def __len__(self) -> int:
        return len(self.paths)

    def empty(self) -> bool:
        return len(self.paths) == 0


class FailureCurriculum:
    """Decides per-env reset sources across PPO iterations.

    Each call to :meth:`assign` returns an array of length ``num_envs``
    where each entry is either ``-1`` (use the standard sim spawn) or an
    index into :attr:`pool.paths` (reset from that failure trajectory).
    """

    def __init__(
        self,
        pool: TrajectoryPool,
        *,
        failure_fraction: float,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= failure_fraction <= 1.0:
            raise ValueError(f"failure_fraction must be in [0, 1], got {failure_fraction}")
        self.pool = pool
        self.failure_fraction = failure_fraction
        self._rng = np.random.default_rng(seed)

    def assign(self, num_envs: int) -> np.ndarray:
        """Return an ``int64[num_envs]`` assignment array."""
        assignment = np.full(num_envs, -1, dtype=np.int64)
        if self.pool.empty() or self.failure_fraction == 0.0:
            return assignment
        n_failure = int(round(num_envs * self.failure_fraction))
        if n_failure == 0:
            return assignment
        chosen = self._rng.choice(num_envs, size=n_failure, replace=False)
        pool_idx = self._rng.integers(0, len(self.pool), size=n_failure)
        assignment[chosen] = pool_idx
        return assignment
