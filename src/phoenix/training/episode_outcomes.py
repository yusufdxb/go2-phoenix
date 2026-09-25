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

#: 2.1.0 (2026-09-22): ``success`` means LOCOMOTION success (survived AND a PASS
#: behavioral verdict from :mod:`phoenix.evaluation`). In 2.0.0 it meant "the
#: time_out term fired and no termination term did", which labeled every
#: surviving episode a success regardless of attitude events, interventions or
#: tracking; that value is preserved in ``legacy_success``.
SCHEMA_VERSION = "2.1.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({"2.0.0", SCHEMA_VERSION})
SUCCESS_DEFINITIONS = {
    "2.0.0": "legacy: time_out fired and no termination term fired (survival only)",
    "2.1.0": "survived to planned end AND phoenix.evaluation verdict PASS",
}


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
    # ---- 2.1.0 ----
    #: :class:`phoenix.evaluation.outcomes.Outcome` value.
    outcome_class: str | None = None
    #: PASS / WARN / FAIL from :mod:`phoenix.evaluation.thresholds`.
    verdict: str | None = None
    verdict_reasons: list[str] | None = None
    #: The 2.0.0 ``success`` value (time_out rule), kept for side-by-side comparison.
    legacy_success: bool | None = None

    def __post_init__(self):
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError("unsupported episode outcome schema")
        if self.schema_version != "2.0.0" and (
            self.outcome_class is None or self.verdict is None or self.legacy_success is None
        ):
            raise ValueError(
                "schema 2.1.0 requires outcome_class, verdict and legacy_success: success "
                "without a behavioral verdict is the defect 2.1.0 exists to remove"
            )
        if self.schema_version != "2.0.0" and self.success and self.verdict != "PASS":
            raise ValueError("success=True requires verdict PASS")
        if (
            self.schema_version != "2.0.0"
            and self.success
            and self.outcome_class
            not in (
                "timeout",
                "completed",
            )
        ):
            raise ValueError(f"success=True is impossible for outcome {self.outcome_class!r}")
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
        if self.schema_version == "2.0.0":
            for name in ("outcome_class", "verdict", "verdict_reasons", "legacy_success"):
                result.pop(name)
        result["episode_length_s"] = self.episode_length_steps * self.control_dt_s
        result["success_definition"] = SUCCESS_DEFINITIONS[self.schema_version]
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
        definition = record.pop("success_definition", None)
        version = record.get("schema_version")
        if definition is not None and definition != SUCCESS_DEFINITIONS.get(version):
            raise ValueError("success_definition disagrees with schema_version")
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
        # Episode generation per environment: how many times this capture has
        # seen that environment reset. Generation g is the episode that follows
        # the g-th observed reset; resets before the capture was installed are
        # not counted, so the first observed episode is generation 0.
        self.generation: dict[int, int] = {}
        # Environments reset during the current step, cleared by begin_step.
        self.reset_this_step: set[int] = set()
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
                self.generation[int(index)] = self.generation.get(int(index), 0) + 1
                self.reset_this_step.add(int(index))
        return self.original(env_ids, *args, **kwargs)

    def begin_step(self):
        self.terminal.clear()
        self.reset_this_step.clear()

    def overlay(self, values):
        for index, snapshot in self.terminal.items():
            for key, value in snapshot.items():
                values[key][index] = value
        return values

    def close(self):
        if self.env._reset_idx == self.installed:
            self.env._reset_idx = self.original


#: Frame of ``snapshot_manager_state``'s ``position``: measured from that
#: environment's origin in ALL THREE axes. Subtracting only z (as this did
#: until 2026-09-11) leaves world x and y in the record while
#: :func:`phoenix.replay.state_adapter.restore_state` adds the full origin
#: back, so a captured state restored into any environment whose origin has a
#: nonzero x or y, which is every environment in the usual grid layout, comes
#: back displaced by that origin. Capture and restore must be exact inverses.
SNAPSHOT_POSITION_FRAME = "env_local"


def snapshot_manager_state(env, to_numpy):
    """Read copied, environment-local state using manager-based APIs.

    ``position`` is in the :data:`SNAPSHOT_POSITION_FRAME` frame: the robot's
    root position minus its environment's origin on x, y and z. Height
    consumers read column 2, which is unchanged by making x and y local.
    """
    import numpy as np

    root = env.scene["robot"].data
    position = to_numpy(root.root_pos_w).copy()
    position -= to_numpy(env.scene.env_origins)[:, :3]
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
