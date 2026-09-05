"""Versioned research episode outcomes, independent of simulator imports.

V1 episode Parquets remain supported by ``episode_records``. V2 JSONL makes
unknown event, intervention and recovery observations explicitly nullable.
A scenario ID is valid for pairing only when a runner actually applied its
frozen manifest; ordinary randomized rollouts have ``scenario_id=None``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = "2.0.0"


@dataclass(frozen=True)
class EpisodeOutcome:
    policy_id: str
    evaluation_seed: int
    episode_id: int
    success: bool
    termination_reason: str
    episode_length_steps: int
    control_dt_s: float
    episode_return: float
    scenario_id: str | None = None
    training_seed: int | None = None
    scenario_seed: int | None = None
    parameter_sample_id: str | None = None
    command: list[float] | None = None  # mean command over observed steps
    tracking_error: float | None = None  # planar velocity RMSE in m/s
    angular_tracking_error: float | None = None  # yaw-rate MAE in rad/s
    failure_events: list[dict] | None = None
    failure_modes: list[str] | None = None
    failure_onset_s: float | None = None
    intervention_required: bool | None = None
    intervention_criterion: str | None = None
    recovery_outcome: str | None = None
    recovery_time_s: float | None = None
    environment_parameters: dict = field(default_factory=dict)
    observation_status: str = "terminal_pre_reset"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported episode outcome schema")
        if not self.policy_id or not self.termination_reason:
            raise ValueError("policy identity and termination reason required")
        if (
            self.episode_length_steps < 1
            or not math.isfinite(self.control_dt_s)
            or self.control_dt_s <= 0
            or not math.isfinite(self.episode_return)
        ):
            raise ValueError("invalid episode duration or return")
        for name in (
            "tracking_error",
            "angular_tracking_error",
            "failure_onset_s",
            "recovery_time_s",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"invalid {name}")
        if self.command is not None and (
            len(self.command) != 3 or not all(map(math.isfinite, self.command))
        ):
            raise ValueError("command must contain three finite values")
        if self.failure_modes is None and self.failure_events is not None:
            raise ValueError("observed events require observed mode labels")
        if self.scenario_id is not None and (
            self.scenario_seed is None or not self.parameter_sample_id
        ):
            raise ValueError("scenario identity requires independent seed and parameter sample ID")
        # Reject non-JSON and NaN environment/event metadata before writing.
        json.dumps(asdict(self), allow_nan=False)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["episode_length_s"] = self.episode_length_steps * self.control_dt_s
        return result


def write_outcomes(path: str | Path, outcomes: list[EpisodeOutcome]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(outcome.to_dict(), sort_keys=True, allow_nan=False) + "\n"
        for outcome in outcomes
    )
    if path.exists():
        if path.read_text() != payload:
            raise FileExistsError(f"Refusing to overwrite episode evidence: {path}")
    else:
        with path.open("x") as stream:
            stream.write(payload)
    return path


def load_outcomes(path: str | Path) -> list[EpisodeOutcome]:
    records = []
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        duration = record.pop("episode_length_s")
        outcome = EpisodeOutcome(**record)
        if not math.isclose(duration, outcome.episode_length_steps * outcome.control_dt_s):
            raise ValueError("episode duration disagrees with steps and control period")
        records.append(outcome)
    return records


class PreResetCapture:
    """Capture terminal signals immediately BEFORE Isaac Lab ``_reset_idx``.

    The adapter requires the manager environment reset method explicitly.
    The callback copies arrays, since simulator buffers mutate during reset.
    Call ``begin_step`` before every env.step and overlay snapshots afterwards.
    An unsupported simulator API raises, instead of logging reset state as failure.
    """

    def __init__(self, env, snapshot):
        if not callable(getattr(env, "_reset_idx", None)):
            raise RuntimeError("terminal capture requires Isaac Lab _reset_idx adapter")
        self.env = env
        self.snapshot = snapshot
        self.original = env._reset_idx
        self.terminal = {}
        self.installed = self._reset
        env._reset_idx = self.installed

    def _reset(self, env_ids, *args, **kwargs):
        ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
        if ids:
            values = self.snapshot()
            for index in ids:
                self.terminal[int(index)] = {
                    key: value[index].copy() if hasattr(value[index], "copy") else value[index]
                    for key, value in values.items()
                }
        return self.original(env_ids, *args, **kwargs)

    def begin_step(self):
        self.terminal.clear()

    def overlay(self, values):
        for index, snapshot in self.terminal.items():
            for key, value in snapshot.items():
                values[key][index] = value
        return values

    def close(self):
        if self.env._reset_idx == self.installed:
            self.env._reset_idx = self.original


def snapshot_manager_state(env, to_numpy):
    """Read copied, world-height corrected state using manager-based APIs."""
    import numpy as np

    root = env.scene["robot"].data
    position = to_numpy(root.root_pos_w).copy()
    position[:, 2] -= to_numpy(env.scene.env_origins)[:, 2]
    state = {
        "command": to_numpy(env.command_manager.get_command("base_velocity")).copy(),
        "linear": to_numpy(root.root_lin_vel_b).copy(),
        "angular": to_numpy(root.root_ang_vel_b).copy(),
        "quaternion": to_numpy(root.root_quat_w).copy(),
        "position": position,
        "joint_velocity": to_numpy(root.joint_vel).copy(),
    }
    # Contacts absent remain NaN, never indistinguishable from airborne feet.
    state["contacts"] = np.full((len(position), 4), np.nan)
    try:
        sensor = env.scene["contact_forces"]
        feet = [i for i, name in enumerate(sensor.body_names) if name.lower().endswith("foot")]
        if len(feet) == 4:
            state["contacts"] = np.linalg.norm(to_numpy(sensor.data.net_forces_w)[:, feet], axis=-1)
    except KeyError:
        pass
    manager = env.termination_manager
    state["terminated"] = to_numpy(manager.terminated).copy()
    for name in manager.active_terms:
        state[f"termination:{name}"] = to_numpy(manager.get_term(name)).copy()
    return state
