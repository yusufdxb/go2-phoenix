"""Curriculum scheduler for failure-seeded adaptation.

The scheduler decides, each PPO iteration, which subset of parallel envs
should be reset from a *real failure seed* versus the usual random
spawn. Keeping this logic stateless and numpy-only means we can unit-test
it without Isaac Lab.

Sampling is a declared choice. ``uniform_legacy`` draws a pool FILE uniformly,
which makes the curriculum's mechanism mix a property of how many files each
mechanism happens to have produced: a harvest that yields fourteen slips and
one collapse teaches almost no collapse recovery, and nothing in the run
records that. ``stratified`` draws a stratum first and a file within it second,
so mechanism, severity, command bucket and scenario are represented evenly
regardless of dataset composition. The strata a pool actually resolves are
readable from :meth:`FailureCurriculum.describe_strata` so a run can report the
mix it trained on.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: Sampling modes. ``uniform_legacy`` is the historical file-uniform draw and
#: is kept only to reproduce runs recorded before stratification existed.
SAMPLING_MODES = ("uniform_legacy", "stratified")

#: Strata a pool entry can be grouped by, when its metadata declares them.
STRATUM_KEYS = ("failure_mode", "severity", "command_bucket", "scenario_id")

#: Default strata: the failure mechanism and the command regime it happened in.
DEFAULT_STRATA = ("failure_mode", "command_bucket")

#: Planar-speed edges (m/s) for the ``command_bucket`` stratum. Explicit rather
#: than learned, so the bucket a trajectory lands in is reproducible.
COMMAND_SPEED_EDGES = (0.2, 0.6)
_COMMAND_BUCKET_NAMES = ("stand", "slow", "fast")


@dataclass(frozen=True)
class TrajectoryMetadata:
    """What a pool file declares about itself, for filtering and stratification."""

    path: Path
    readable: bool = False
    failure_modes: frozenset[str] = frozenset()
    failure_mode: str | None = None
    severity: str | None = None
    command_bucket: str | None = None
    scenario_id: str | None = None

    def stratum(self, keys: Sequence[str]) -> tuple:
        return tuple(getattr(self, key) for key in keys)


def _command_bucket(command) -> str | None:
    if command is None:
        return None
    values = np.asarray(command, dtype=float).reshape(-1)
    if values.size < 2 or not np.isfinite(values[:2]).all():
        return None
    speed = float(np.hypot(values[0], values[1]))
    index = int(np.searchsorted(np.asarray(COMMAND_SPEED_EDGES, dtype=float), speed, side="right"))
    return _COMMAND_BUCKET_NAMES[min(index, len(_COMMAND_BUCKET_NAMES) - 1)]


def _onset_index(flags) -> int | None:
    for i, flag in enumerate(flags):
        if flag:
            return i
    return None


def _describe_parquet(path: Path) -> TrajectoryMetadata:
    try:
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - pyarrow is a hard dep
        return TrajectoryMetadata(path=path)
    try:
        available = set(pq.read_schema(path).names)
    except Exception:
        return TrajectoryMetadata(path=path)
    wanted = [c for c in ("failure_mode", "failure_flag", "command_vel") if c in available]
    try:
        table = pq.read_table(path, columns=wanted)
    except Exception:
        return TrajectoryMetadata(path=path)
    modes = (
        [v for v in table.column("failure_mode").to_pylist()]
        if "failure_mode" in wanted
        else []
    )
    flags = table.column("failure_flag").to_pylist() if "failure_flag" in wanted else []
    commands = table.column("command_vel").to_pylist() if "command_vel" in wanted else []
    onset = _onset_index(flags) if flags else None
    primary = None
    if modes:
        labelled = [m for m in modes[onset:] if m] if onset is not None else [m for m in modes if m]
        primary = labelled[0] if labelled else None
    command = None
    if commands:
        command = commands[onset] if onset is not None and onset < len(commands) else commands[-1]
    return TrajectoryMetadata(
        path=path,
        readable=True,
        failure_modes=frozenset(m for m in modes if m),
        failure_mode=primary,
        command_bucket=_command_bucket(command),
    )


def _describe_capsule(path: Path) -> TrajectoryMetadata:
    import json

    try:
        data = json.loads(path.read_text())
    except Exception:
        return TrajectoryMetadata(path=path)
    if not isinstance(data, dict):
        return TrajectoryMetadata(path=path)
    frames = data.get("frames") or []
    onset = data.get("failure_onset_index")
    modes = {m for m in (data.get("failure_modes") or []) if m}
    primary = data.get("failure_mode")
    if primary:
        modes.add(primary)
    elif modes:
        primary = sorted(modes)[0]
    command = None
    if frames:
        index = int(onset) if isinstance(onset, int) and 0 <= onset < len(frames) else -1
        command = frames[index].get("command_vel")
    severity = data.get("severity")
    return TrajectoryMetadata(
        path=path,
        readable=True,
        failure_modes=frozenset(modes),
        failure_mode=primary,
        severity=str(severity) if severity is not None else None,
        command_bucket=_command_bucket(command),
        scenario_id=data.get("scenario_id"),
    )


def describe_trajectory(path: str | Path) -> TrajectoryMetadata:
    """Read the strata a pool file declares, without raising on a bad file.

    An unreadable or non-trajectory file is reported as ``readable=False`` with
    empty strata rather than crashing a training launch; it then lands in the
    ``unlabelled`` stratum, which is visible in
    :meth:`FailureCurriculum.describe_strata`.
    """
    path = Path(path)
    return _describe_capsule(path) if path.suffix == ".json" else _describe_parquet(path)


@dataclass
class TrajectoryPool:
    """List of Parquet trajectories or JSON capsules available as failure seeds."""

    paths: Sequence[Path]
    _metadata: list[TrajectoryMetadata] | None = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        pattern: str = "*.parquet",
        *,
        failure_modes: Sequence[str] | None = None,
    ) -> TrajectoryPool:
        """Load parquets matching ``pattern`` under ``directory``.

        Parameters
        ----------
        directory
            Directory containing failure-trajectory parquets.
        pattern
            Glob pattern; defaults to ``*.parquet``.
        failure_modes
            Optional whitelist of failure-mode strings. If ``None`` or
            empty, every matching parquet is included (legacy behavior).
            Otherwise each parquet is opened and only kept when at least
            one of its non-null ``failure_mode`` rows is a member of the
            whitelist. Files without a ``failure_mode`` column are
            excluded under a non-empty filter.
        """
        p = Path(directory)
        if not p.exists():
            return cls(paths=[])
        candidates = sorted(p.glob(pattern))
        if not failure_modes:
            return cls(paths=candidates)
        wanted = set(failure_modes)
        kept: list[Path] = []
        metadata: list[TrajectoryMetadata] = []
        for path in candidates:
            described = describe_trajectory(path)
            if described.failure_modes & wanted:
                kept.append(path)
                metadata.append(described)
        pool = cls(paths=kept)
        pool._metadata = metadata
        return pool

    def __len__(self) -> int:
        return len(self.paths)

    def empty(self) -> bool:
        return len(self.paths) == 0

    def metadata(self) -> list[TrajectoryMetadata]:
        """Describe every pool entry once, caching the result."""
        if self._metadata is None or len(self._metadata) != len(self.paths):
            self._metadata = [describe_trajectory(path) for path in self.paths]
        return self._metadata


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
        failure_reset_fraction: float | None = None,
        failure_fraction: float | None = None,
        seed: int = 0,
        sampling: str = "uniform_legacy",
        strata: Sequence[str] = DEFAULT_STRATA,
    ) -> None:
        if failure_reset_fraction is not None and failure_fraction is not None:
            raise ValueError("Use failure_reset_fraction or legacy failure_fraction, not both")
        fraction = (
            failure_reset_fraction if failure_reset_fraction is not None else failure_fraction
        )
        if fraction is None:
            raise ValueError("failure_reset_fraction is required")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"failure_fraction must be in [0, 1], got {failure_fraction}")
        if sampling not in SAMPLING_MODES:
            raise ValueError(f"Unknown sampling={sampling!r}; expected {SAMPLING_MODES}")
        unknown = [key for key in strata if key not in STRATUM_KEYS]
        if unknown:
            raise ValueError(f"Unknown strata {unknown}; expected a subset of {STRATUM_KEYS}")
        if sampling == "stratified" and not tuple(strata):
            raise ValueError("Stratified sampling needs at least one stratum key")
        self.pool = pool
        self.failure_reset_fraction = float(fraction)
        self.sampling = sampling
        self.strata = tuple(strata)
        self._rng = np.random.default_rng(seed)
        self._groups: list[list[int]] | None = None
        if sampling == "uniform_legacy" and self.failure_reset_fraction > 0 and not pool.empty():
            logger.warning(
                "FailureCurriculum sampling='uniform_legacy' draws pool FILES uniformly and "
                "ignores dataset composition; sampling='stratified' balances %s",
                ", ".join(DEFAULT_STRATA),
            )

    @property
    def failure_fraction(self) -> float:
        """Compatibility alias for the historical name."""
        return self.failure_reset_fraction

    def describe_strata(self) -> dict[str, int]:
        """Return ``{stratum label: number of pool files}`` for the current strata."""
        counts: dict[str, int] = {}
        for described in self.pool.metadata():
            label = self._label(described.stratum(self.strata))
            counts[label] = counts.get(label, 0) + 1
        return counts

    @staticmethod
    def _label(stratum: tuple) -> str:
        return "|".join("unlabelled" if v is None else str(v) for v in stratum) or "unlabelled"

    def _stratum_groups(self) -> list[list[int]]:
        if self._groups is None or sum(len(g) for g in self._groups) != len(self.pool):
            buckets: dict[tuple, list[int]] = {}
            for index, described in enumerate(self.pool.metadata()):
                buckets.setdefault(described.stratum(self.strata), []).append(index)
            self._groups = [buckets[key] for key in sorted(buckets, key=self._label)]
        return self._groups

    def _draw_pool_indices(self, n_failure: int) -> np.ndarray:
        if self.sampling == "uniform_legacy":
            return self._rng.integers(0, len(self.pool), size=n_failure)
        groups = self._stratum_groups()
        chosen_groups = self._rng.integers(0, len(groups), size=n_failure)
        return np.asarray(
            [groups[g][int(self._rng.integers(0, len(groups[g])))] for g in chosen_groups],
            dtype=np.int64,
        )

    def assign(self, num_envs: int) -> np.ndarray:
        """Return an ``int64[num_envs]`` assignment array."""
        assignment = np.full(num_envs, -1, dtype=np.int64)
        if self.pool.empty() or self.failure_fraction == 0.0:
            return assignment
        n_failure = int(round(num_envs * self.failure_fraction))
        if n_failure == 0:
            return assignment
        chosen = self._rng.choice(num_envs, size=n_failure, replace=False)
        assignment[chosen] = self._draw_pool_indices(n_failure)
        return assignment
