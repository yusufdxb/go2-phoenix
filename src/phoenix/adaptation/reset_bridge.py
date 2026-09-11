"""Failure-seeded resets with explicit row resolution and complete kinematic writes.

This changes environment initialization only. PPO collects fresh on-policy data.
No saved transitions are inserted into its rollout buffer.

Three things a seeded reset has to get right beyond the pose, all of which are
declared here rather than assumed:

* **Controller history.** ``last_action`` is a trained observation term and the
  actuator chain is delayed, so a single state row is not the system state.
  ``history_rows`` restores what is restorable and the reset telemetry names
  what the reset re-initialised; see
  :mod:`phoenix.replay.controller_history`. Exact replay is a claim the bridge
  refuses to make unless every component was reconstructed.
* **The cause.** A capsule that declares the physics it failed under
  (``environment_parameters``, ``disturbances``) must have them applied through
  a scenario adapter, otherwise the seeded environment gets a recorded state
  with fresh random physics attached and the causal continuation under study is
  gone. ``environment_policy`` decides whether that is an error or a recorded
  fact.
* **The command process.** ``command_policy`` replays the source command for a
  defined interval and then returns the environment to the NORMAL velocity
  command process, so a seeded env and a control env differ in state only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from phoenix.replay.state_adapter import (
    COMMAND_POLICIES,
    VelocityCommandAdapter,
    resolve_command_hold,
    restore_state,
)
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

#: How a seeded reset treats the environment parameters and disturbances a
#: source declares. ``require_declared`` refuses to seed a declared cause it
#: cannot apply, ``require_all`` additionally refuses sources that declare no
#: cause at all, ``record_only`` applies nothing and says so in telemetry.
ENVIRONMENT_POLICIES = ("require_declared", "require_all", "record_only")


def resolve_seed(
    path,
    strategy="failure_onset_minus_seconds",
    offset_steps=0,
    offset_seconds=0.5,
    *,
    position_frame=None,
):
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
    if "timestamp_s" in reader.column_names:
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
    frame, frame_source = reader.resolve_position_frame(position_frame)
    return {
        "requested_seed_row": requested,
        "resolved_row": resolved,
        "failure_onset_row": onset,
        "time_before_onset_seconds": delta,
        "capsule_id": reader.metadata.get("capsule_id"),
        "capsule_schema_version": reader.metadata.get("schema_version"),
        "seed_row_strategy": strategy,
        "source_format": "capsule" if reader.metadata else "legacy_parquet",
        "position_frame": frame,
        "position_frame_source": frame_source,
        "declared_environment_parameters": dict(reader.environment_parameters),
        "declared_disturbances": list(reader.disturbances),
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
        position_frame=None,
        history_rows=0,
    ):
        self._paths = paths
        self._strategy = seed_row_strategy
        self._offset_k = seed_row_offset_k
        self._offset_seconds = seed_row_offset_seconds
        self._position_frame = position_frame
        self._history_rows = history_rows
        self._cache = {}
        self.telemetry = {}

    def get(self, pool_idx: int) -> InitialState:
        if pool_idx not in self._cache:
            record = resolve_seed(
                self._paths[pool_idx],
                self._strategy,
                self._offset_k,
                self._offset_seconds,
                position_frame=self._position_frame,
            )
            self._cache[pool_idx] = load_initial_state(
                self._paths[pool_idx],
                record["resolved_row"],
                position_frame=self._position_frame,
                history_rows=self._history_rows,
            )
            self.telemetry[pool_idx] = record
        return self._cache[pool_idx]

    def resolved_row(self, pool_idx):
        return self.telemetry.get(pool_idx, {}).get("resolved_row", -1)


def _check_environment_context(record, state, *, policy, scenario_adapter, disturbance_applier):
    """Refuse, before training starts, to seed a cause the reset cannot apply."""
    if policy not in ENVIRONMENT_POLICIES:
        raise ValueError(f"Unknown environment_policy={policy!r}; expected {ENVIRONMENT_POLICIES}")
    declared = state.environment_parameters
    disturbances = record.get("declared_disturbances") or []
    if policy == "record_only":
        return
    if policy == "require_all" and not declared:
        raise ValueError(
            f"environment_policy='require_all' but {record.get('capsule_id') or 'the source'} "
            "declares no environment_parameters; the failure's physical cause is unrecorded, so "
            "a seeded environment would get fresh random physics"
        )
    if declared and scenario_adapter is None:
        raise ValueError(
            f"Seed declares environment parameters {sorted(declared)} but no scenario_adapter was "
            "given; seeding it would attach fresh random physics to a recorded state"
        )
    supported = getattr(scenario_adapter, "supported", None)
    if declared and supported is not None:
        unsupported = sorted(set(declared) - set(supported))
        if unsupported:
            raise ValueError(
                f"scenario_adapter cannot apply declared environment parameters {unsupported}"
            )
    if disturbances and disturbance_applier is None:
        raise ValueError(
            f"Seed declares {len(disturbances)} disturbance(s) but no disturbance_applier was "
            "given; the disturbance that caused the failure would not be reproduced"
        )


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
    command_policy="source_hold_to_onset",
    command_hold_seconds=None,
    position_frame=None,
    history_rows=2,
    require_exact_replay=False,
    scenario_adapter=None,
    disturbance_applier=None,
    environment_policy="require_declared",
    telemetry_path=None,
):
    """Install reset override; incompatible inputs fail before training starts.

    ``history_rows`` is how many recorded control steps ending at the seed row
    are used to rebuild controller state; 2 restores ``last_action`` and the
    previous action, a negative value uses the whole recorded window, and 0
    opts out and labels the result a state-only seed. ``require_exact_replay``
    turns any component that cannot be reconstructed into an exception.

    ``scenario_adapter`` is any object with ``apply(env_id, parameters)``,
    ``reset(env_ids)`` and an optional ``supported`` set, for example
    :class:`phoenix.adaptation.scenario_bridge.FrictionScenarioAdapter`. Its
    ``reset`` runs BEFORE the environment's own reset so per-env overrides are
    released before domain randomization redraws, matching
    :func:`phoenix.adaptation.scenario_bridge.install_scenario_reset`.
    """
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
    if command_policy not in COMMAND_POLICIES:
        raise ValueError(
            f"Unknown command_policy={command_policy!r}; expected {COMMAND_POLICIES}"
        )
    if command_hold_seconds is not None and command_policy != "fixed_hold":
        raise ValueError("command_hold_seconds only applies to command_policy='fixed_hold'")
    if require_exact_replay and history_rows == 0:
        raise ValueError("require_exact_replay needs controller history; history_rows=0 cannot")
    if environment_policy == "record_only" and (
        scenario_adapter is not None or disturbance_applier is not None
    ):
        raise ValueError(
            "environment_policy='record_only' applies nothing, so passing a scenario_adapter or "
            "disturbance_applier contradicts it; pick 'require_declared' to actually apply them"
        )
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    cache = _InitialStateCache(
        list(curriculum.pool.paths),
        seed_row_strategy=seed_row_strategy,
        seed_row_offset_k=seed_row_offset_k
        if seed_row_offset_steps is None
        else seed_row_offset_steps,
        seed_row_offset_seconds=seed_row_offset_seconds,
        position_frame=position_frame,
        history_rows=history_rows,
    )
    holds = {}
    for i in range(len(curriculum.pool)):
        state = cache.get(i)
        record = cache.telemetry[i]
        _check_environment_context(
            record,
            state,
            policy=environment_policy,
            scenario_adapter=scenario_adapter,
            disturbance_applier=disturbance_applier,
        )
        holds[i] = resolve_command_hold(
            command_policy,
            time_before_onset_seconds=record.get("time_before_onset_seconds"),
            hold_seconds=command_hold_seconds,
        )
    adapter = VelocityCommandAdapter(unwrapped, command_name)
    original = unwrapped._reset_idx
    unwrapped.phoenix_reset_telemetry = []

    def reset(env_ids):
        # Release per-env scenario overrides first, so the environment's own
        # reset (and its domain randomization) starts from the scene defaults.
        if scenario_adapter is not None and env_ids is not None and len(env_ids):
            scenario_adapter.reset(env_ids)
        original(env_ids)
        if env_ids is None or len(env_ids) == 0:
            return
        assignment = curriculum.assign(len(env_ids))
        for local, pool_idx in enumerate(assignment):
            if pool_idx < 0:
                continue
            pool_idx = int(pool_idx)
            env_id = int(env_ids[local])
            state = cache.get(pool_idx)
            record = dict(cache.telemetry[pool_idx])
            record["environment_policy"] = environment_policy
            if scenario_adapter is not None and state.environment_parameters:
                applied = scenario_adapter.apply(env_id, state.environment_parameters) or {}
                record.update(applied)
                record["environment_context_restored"] = True
            if disturbance_applier is not None and record.get("declared_disturbances"):
                record.update(
                    disturbance_applier(env_id, record["declared_disturbances"]) or {}
                )
                record["environment_context_restored"] = True
            hold, command_telemetry = holds[pool_idx]
            record.update(
                restore_state(
                    unwrapped,
                    state,
                    env_id,
                    command_adapter=adapter,
                    command_hold_seconds=hold,
                    command_telemetry=command_telemetry,
                    controller_history=state.controller_history,
                    require_exact_replay=require_exact_replay,
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


__all__ = ["ENVIRONMENT_POLICIES", "install", "resolve_seed"]
