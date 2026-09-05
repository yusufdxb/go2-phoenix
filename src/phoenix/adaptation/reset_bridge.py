"""Failure-seeded resets with explicit row resolution and complete kinematic writes.

This changes environment initialization only. PPO collects fresh on-policy data.
No saved transitions are inserted into its rollout buffer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from phoenix.replay.state_adapter import VelocityCommandAdapter, restore_state
from phoenix.replay.trajectory_reader import InitialState, TrajectoryReader, load_initial_state

from .curriculum import FailureCurriculum

logger = logging.getLogger(__name__)
_VALID_STRATEGIES = (
    "first",
    "failure_onset",
    "failure_onset_minus_k",
    "failure_onset_minus_steps",
    "failure_onset_minus_seconds",
)


def resolve_seed(path, strategy="failure_onset_minus_seconds", offset_steps=0, offset_seconds=0.5):
    """Resolve against event onset, with explicit provenance and no silent clamping.

    Legacy minus_k retains its historical clamp; canonical strategies reject an
    unavailable pre-onset interval. Seconds use actual logged timestamps when
    available, selecting the latest frame at or before the requested time.
    """
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"Unknown seed_row_strategy={strategy!r}; expected {_VALID_STRATEGIES}")
    reader = TrajectoryReader(path)
    if not len(reader):
        raise ValueError("Empty failure trajectory")
    try:
        indices = reader.failure_indices()
    except KeyError:
        indices = np.array([], dtype=int)
    onset = int(indices[0]) if len(indices) else None
    if onset is not None and not 0 <= onset < len(reader):
        raise ValueError("failure_onset_index out of bounds")
    if onset is None and strategy != "first":
        raise ValueError("Trajectory has no failure_flag=True row or capsule onset")
    timestamps = None
    if "timestamp_s" in reader._table.column_names:
        timestamps = reader.column("timestamp_s").astype(float)
    elif reader.metadata.get("control_dt") is not None:
        dt = float(reader.metadata["control_dt"])
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        timestamps = np.arange(len(reader)) * dt
    if timestamps is not None and (
        not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0)
    ):
        raise ValueError("Trajectory timestamps must be finite and strictly increasing")
    if strategy == "first":
        requested = resolved = 0
    elif strategy == "failure_onset":
        requested = resolved = onset
    elif strategy == "failure_onset_minus_seconds":
        if not np.isfinite(offset_seconds) or offset_seconds < 0:
            raise ValueError("offset_seconds must be finite and nonnegative")
        if timestamps is None:
            raise ValueError("Seconds reset needs timestamp_s or explicit control_dt")
        target = timestamps[onset] - offset_seconds
        requested = int(np.searchsorted(timestamps, target + 1e-10, side="right") - 1)
        resolved = requested
    else:
        if strategy == "failure_onset_minus_steps" and (
            offset_steps < 0 or int(offset_steps) != offset_steps
        ):
            raise ValueError("offset_steps must be a nonnegative integer")
        requested = onset - int(offset_steps)
        resolved = (
            max(0, min(onset, requested)) if strategy == "failure_onset_minus_k" else requested
        )
    minimum = int(reader.metadata.get("pre_failure_start_index", 0)) if strategy != "first" else 0
    if resolved < minimum or resolved >= len(reader):
        raise ValueError(
            f"Requested pre-onset row {requested} unavailable (minimum {minimum}); no generic reset fallback"
        )
    delta = (
        float(timestamps[onset] - timestamps[resolved])
        if onset is not None and timestamps is not None
        else None
    )
    return {
        "requested_seed_row": requested,
        "resolved_row": resolved,
        "failure_onset_row": onset,
        "time_before_onset_seconds": delta,
        "capsule_id": reader.metadata.get("capsule_id"),
        "seed_row_strategy": strategy,
        "source_format": "capsule" if reader.metadata else "legacy_parquet",
        "environment_context_restored": False,
    }


def _resolve_seed_row(path: Path, strategy: str, offset_k: int, offset_seconds=0.5) -> int:
    return resolve_seed(path, strategy, offset_k, offset_seconds)["resolved_row"]


class _InitialStateCache:
    def __init__(
        self,
        paths,
        *,
        seed_row_strategy="failure_onset_minus_seconds",
        seed_row_offset_k=0,
        seed_row_offset_seconds=0.5,
    ):
        self._paths = paths
        self._strategy = seed_row_strategy
        self._offset_k = seed_row_offset_k
        self._offset_seconds = seed_row_offset_seconds
        self._cache = {}
        self.telemetry = {}

    def get(self, pool_idx: int) -> InitialState:
        if pool_idx not in self._cache:
            record = resolve_seed(
                self._paths[pool_idx], self._strategy, self._offset_k, self._offset_seconds
            )
            self._cache[pool_idx] = load_initial_state(
                self._paths[pool_idx], record["resolved_row"]
            )
            self.telemetry[pool_idx] = record
        return self._cache[pool_idx]

    def resolved_row(self, pool_idx):
        return self.telemetry.get(pool_idx, {}).get("resolved_row", -1)


def install(
    env: Any,
    curriculum: FailureCurriculum,
    *,
    seed_row_strategy="failure_onset_minus_seconds",
    seed_row_offset_k=0,
    seed_row_offset_steps=None,
    seed_row_offset_seconds=0.5,
    write_velocity=True,
    command_name="base_velocity",
    command_hold_seconds=float("inf"),
    telemetry_path=None,
):
    """Install reset override; incompatible inputs fail before training starts."""
    if curriculum.failure_reset_fraction <= 0:
        return
    if curriculum.pool.empty():
        raise ValueError(
            "Active failure reset curriculum has an empty pool; refusing generic fallback"
        )
    if not write_velocity:
        raise ValueError(
            "Failure resets require velocity restoration; write_velocity=False is unsupported"
        )
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    cache = _InitialStateCache(
        list(curriculum.pool.paths),
        seed_row_strategy=seed_row_strategy,
        seed_row_offset_k=seed_row_offset_k
        if seed_row_offset_steps is None
        else seed_row_offset_steps,
        seed_row_offset_seconds=seed_row_offset_seconds,
    )
    for i in range(len(curriculum.pool)):
        cache.get(i)
    adapter = VelocityCommandAdapter(unwrapped, command_name)
    original = unwrapped._reset_idx
    unwrapped.phoenix_reset_telemetry = []

    def reset(env_ids):
        original(env_ids)
        if env_ids is None or len(env_ids) == 0:
            return
        assignment = curriculum.assign(len(env_ids))
        for local, pool_idx in enumerate(assignment):
            if pool_idx < 0:
                continue
            record = dict(cache.telemetry[int(pool_idx)])
            record.update(
                restore_state(
                    unwrapped,
                    cache.get(int(pool_idx)),
                    int(env_ids[local]),
                    command_adapter=adapter,
                    hold_seconds=command_hold_seconds,
                )
            )
            unwrapped.phoenix_reset_telemetry.append(record)
            # Bound in-memory logs while permitting a complete persisted stream.
            del unwrapped.phoenix_reset_telemetry[:-1000]
            if telemetry_path is not None:
                with Path(telemetry_path).open("a") as stream:
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            logger.info("failure_reset %s", json.dumps(record, sort_keys=True, allow_nan=False))

    unwrapped._reset_idx = reset
    unwrapped.phoenix_curriculum = curriculum


__all__ = ["install", "resolve_seed"]
