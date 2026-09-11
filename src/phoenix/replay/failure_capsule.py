"""Write the FailureCapsule that :mod:`phoenix.replay.trajectory_reader` reads.

The capsule is the Parquet trajectory schema plus the three things a per-step
row cannot carry and a seeded reset needs:

* the FRAME of ``base_pos`` (``position_frame``), so restore is the exact
  inverse of capture instead of a guess;
* the CONTROLLER history (``action``, and ``joint_target`` where the producer
  can observe the applied target), so ``last_action`` and the rate limiter are
  reconstructed rather than zeroed;
* the CAUSE (``environment_parameters``, ``disturbances``) and
  ``episode_start_index``, so a seeded environment keeps the physics the
  failure happened under and a truncated ring-buffer window cannot be mistaken
  for a full episode.

Everything written here is observed by the producer. Nothing is defaulted into
existence: an unknown cause is an absent key, which the reset bridge reports as
an unrestored environment context rather than treating as "no perturbation".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .state_adapter import POSITION_FRAMES

#: Version this writer emits. The reader accepts this and the earlier 1.0.
CAPSULE_SCHEMA_VERSION = "1.1"

#: Per-frame fields every capsule must carry, and their widths.
REQUIRED_FRAME_FIELDS = {
    "base_pos": 3,
    "base_quat": 4,
    "base_lin_vel_body": 3,
    "base_ang_vel_body": 3,
    "joint_pos": 12,
    "joint_vel": 12,
    "command_vel": 3,
}

#: Per-frame fields that are optional, but all-or-nothing across the capsule.
OPTIONAL_FRAME_FIELDS = ("action", "joint_target")

__all__ = ["CAPSULE_SCHEMA_VERSION", "write_failure_capsule"]


def _vector(frame, name, index, width=None):
    value = frame.get(name)
    if value is None:
        raise ValueError(f"Frame {index} is missing {name}")
    array = np.asarray(value, dtype=float).reshape(-1)
    if width is not None and array.shape != (width,):
        raise ValueError(f"Frame {index} field {name} must have {width} values")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"Frame {index} field {name} must be non-empty and finite")
    return [float(v) for v in array]


def write_failure_capsule(
    path,
    frames,
    *,
    capsule_id,
    failure_onset_index,
    control_dt,
    position_frame="env_local",
    pre_failure_start_index=0,
    episode_start_index=None,
    environment_parameters=None,
    disturbances=None,
    failure_mode=None,
    failure_modes=None,
    severity=None,
    scenario_id=None,
    extra=None,
):
    """Validate and write one capsule; refuse to overwrite recorded evidence."""
    path = Path(path)
    frames = list(frames)
    if not frames:
        raise ValueError("A failure capsule needs at least one frame")
    if position_frame not in POSITION_FRAMES:
        raise ValueError(f"Unknown position_frame={position_frame!r}; expected {POSITION_FRAMES}")
    if not capsule_id:
        raise ValueError("capsule_id is required; an anonymous capsule cannot be cited")
    if not np.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be finite and positive")
    onset = int(failure_onset_index)
    start = int(pre_failure_start_index)
    if not 0 <= onset < len(frames):
        raise ValueError("failure_onset_index out of range")
    if not 0 <= start <= onset:
        raise ValueError("pre_failure_start_index must be within [0, failure_onset_index]")
    if episode_start_index is not None and not 0 <= int(episode_start_index) <= onset:
        raise ValueError("episode_start_index must be within [0, failure_onset_index]")

    present = {name for name in OPTIONAL_FRAME_FIELDS if frames[0].get(name) is not None}
    widths = {name: len(np.asarray(frames[0][name]).reshape(-1)) for name in present}
    written = []
    for index, frame in enumerate(frames):
        row = {
            name: _vector(frame, name, index, width)
            for name, width in REQUIRED_FRAME_FIELDS.items()
        }
        norm = float(np.linalg.norm(row["base_quat"]))
        if not np.isclose(norm, 1.0, atol=1e-3):
            raise ValueError(f"Frame {index} base_quat must be a normalized xyzw quaternion")
        for name in OPTIONAL_FRAME_FIELDS:
            has = frame.get(name) is not None
            if has != (name in present):
                raise ValueError(
                    f"Frame {index} {'adds' if has else 'omits'} {name}; a capsule's controller "
                    "history must be complete or absent, never partial"
                )
            if has:
                row[name] = _vector(frame, name, index, widths[name])
        written.append(row)

    data = {
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "capsule_id": str(capsule_id),
        "velocity_frame": "body",
        "position_frame": position_frame,
        "control_dt": float(control_dt),
        "failure_onset_index": onset,
        "pre_failure_start_index": start,
        "frames": written,
    }
    if episode_start_index is not None:
        data["episode_start_index"] = int(episode_start_index)
    if environment_parameters:
        data["environment_parameters"] = {
            str(k): float(v) for k, v in dict(environment_parameters).items()
        }
    if disturbances:
        data["disturbances"] = list(disturbances)
    for name, value in (
        ("failure_mode", failure_mode),
        ("severity", severity),
        ("scenario_id", scenario_id),
    ):
        if value is not None:
            data[name] = str(value)
    if failure_modes:
        data["failure_modes"] = [str(m) for m in failure_modes]
    if extra:
        overlap = sorted(set(extra) & set(data))
        if overlap:
            raise ValueError(f"extra may not redefine capsule fields {overlap}")
        data.update(extra)

    payload = json.dumps(data, sort_keys=True, allow_nan=False)
    if path.exists():
        if path.read_text() != payload:
            raise FileExistsError(f"Refusing to overwrite recorded failure evidence: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(payload)
    return path
