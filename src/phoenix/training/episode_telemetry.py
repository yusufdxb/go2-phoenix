"""Per-step, per-episode telemetry for Phoenix evaluation rollouts.

:mod:`phoenix.training.episode_records` answers "which episodes failed".
It cannot answer "how did they fail", because a failure mode is a property
of the trajectory, not of the episode summary. Diagnosing a mode needs the
per-step attitude, base height and velocity-tracking signals that the
episode summary averages away.

The pre-existing ``--telemetry-out`` CSV is not a substitute. It logs env
index 0 only, carries no episode identifier, and has neither attitude nor
base height, so it cannot be joined back to an episode record at all.

This module is that missing artifact: one parquet row per (episode, control
step), carrying the episode key ``(run_id, seed, episode_id)`` so a reader
can group rows into episodes and line them up with the matching episode
record.

Design rules, matching :mod:`phoenix.training.episode_records`:

* :data:`SCHEMA_VERSION` is stamped both in the Arrow schema metadata and
  in a per-row column, so a row read in isolation identifies its version.
* :data:`UNITS` is total, and is stamped into the schema metadata.
* The loader is fail-closed. A version mismatch, a missing column, a null
  in a required column or a zero-row artifact raises. Nothing defaults.

One convention decision is deliberate and load bearing. Attitude is stored
as ``pitch_rad`` and ``roll_rad`` scalars, computed here, rather than as a
raw quaternion. Isaac Lab reports ``root_quat_w`` as wxyz while Phoenix's
hardware trajectory logger stores xyzw, so shipping a quaternion across a
repo boundary would put a silent convention mismatch directly in the
contract. The raw quaternion is still written alongside, for provenance,
under an explicitly named wxyz column.

Signals that a given rollout cannot observe are written as zeros rather
than as nulls, because a consumer can detect an identically zero signal and
treat it as unobserved, which is what the downstream failure detector
already does for the GO2 hardware path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "1.0.0"

#: Arrow metadata key holding :data:`SCHEMA_VERSION`.
SCHEMA_VERSION_KEY = b"phoenix_episode_telemetry_schema_version"

#: Arrow metadata key holding a JSON dump of :data:`UNITS`.
UNITS_KEY = b"phoenix_episode_telemetry_units"

#: Number of actuated joints on the GO2.
N_JOINTS = 12

#: Number of feet, and so the width of the contact-force vector.
N_FEET = 4

#: Unit of every field. Total by construction: a consumer never has to
#: guess, including for the fields whose name already carries the unit.
UNITS: dict[str, str] = {
    "schema_version": "semver",
    "run_id": "identifier",
    "episode_id": "count",
    "seed": "identifier",
    "env_index": "index",
    "step_index": "env_steps",
    "t_s": "seconds",
    "control_dt_s": "seconds",
    "pitch_rad": "radians",
    "roll_rad": "radians",
    "yaw_rad": "radians",
    "base_height_m": "meters",
    "base_quat_wxyz": "unit_quaternion_wxyz",
    "cmd_lin_vel_x_mps": "m/s",
    "cmd_lin_vel_y_mps": "m/s",
    "cmd_ang_vel_z_radps": "rad/s",
    "actual_lin_vel_x_mps": "m/s",
    "actual_lin_vel_y_mps": "m/s",
    "actual_ang_vel_z_radps": "rad/s",
    "joint_vel_radps": "rad/s",
    "contact_forces_n": "newtons",
}

ARROW_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("run_id", pa.string()),
        ("episode_id", pa.int64()),
        ("seed", pa.int64()),
        ("env_index", pa.int64()),
        ("step_index", pa.int64()),
        ("t_s", pa.float64()),
        ("control_dt_s", pa.float64()),
        ("pitch_rad", pa.float64()),
        ("roll_rad", pa.float64()),
        ("yaw_rad", pa.float64()),
        ("base_height_m", pa.float64()),
        ("base_quat_wxyz", pa.list_(pa.float32(), 4)),
        ("cmd_lin_vel_x_mps", pa.float64()),
        ("cmd_lin_vel_y_mps", pa.float64()),
        ("cmd_ang_vel_z_radps", pa.float64()),
        ("actual_lin_vel_x_mps", pa.float64()),
        ("actual_lin_vel_y_mps", pa.float64()),
        ("actual_ang_vel_z_radps", pa.float64()),
        ("joint_vel_radps", pa.list_(pa.float32(), N_JOINTS)),
        ("contact_forces_n", pa.list_(pa.float32(), N_FEET)),
    ],
    metadata={
        SCHEMA_VERSION_KEY: SCHEMA_VERSION.encode(),
        UNITS_KEY: json.dumps(UNITS, sort_keys=True).encode(),
    },
)

REQUIRED_COLUMNS: tuple[str, ...] = tuple(ARROW_SCHEMA.names)

#: Columns that identify an episode. Grouping on these yields one episode.
KEY_COLUMNS: tuple[str, ...] = ("run_id", "seed", "episode_id")


class EpisodeTelemetrySchemaError(ValueError):
    """Raised when an episode-telemetry artifact fails validation."""


def quat_wxyz_to_euler(quat) -> tuple[float, float, float]:
    """Convert a wxyz quaternion to (roll, pitch, yaw) in radians.

    Isaac Lab's ``root_quat_w`` is wxyz. The pitch branch is clamped so a
    numerically slightly out of range sine cannot raise from arcsin.
    """
    q = np.asarray(quat, dtype=np.float64).ravel()
    if q.size != 4:
        raise ValueError(f"quaternion must have 4 components, got {q.size}")
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (w * y - z * x)
    pitch = float(np.arcsin(np.clip(sinp, -1.0, 1.0)))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


@dataclass
class TelemetryStep:
    """One control step of one episode."""

    run_id: str
    episode_id: int
    seed: int
    env_index: int
    step_index: int
    t_s: float
    control_dt_s: float
    pitch_rad: float
    roll_rad: float
    yaw_rad: float
    base_height_m: float
    base_quat_wxyz: np.ndarray
    cmd_lin_vel_x_mps: float
    cmd_lin_vel_y_mps: float
    cmd_ang_vel_z_radps: float
    actual_lin_vel_x_mps: float
    actual_lin_vel_y_mps: float
    actual_ang_vel_z_radps: float
    joint_vel_radps: np.ndarray
    contact_forces_n: np.ndarray
    schema_version: str = SCHEMA_VERSION

    def to_row(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "episode_id": int(self.episode_id),
            "seed": int(self.seed),
            "env_index": int(self.env_index),
            "step_index": int(self.step_index),
            "t_s": float(self.t_s),
            "control_dt_s": float(self.control_dt_s),
            "pitch_rad": float(self.pitch_rad),
            "roll_rad": float(self.roll_rad),
            "yaw_rad": float(self.yaw_rad),
            "base_height_m": float(self.base_height_m),
            "base_quat_wxyz": _fixed(self.base_quat_wxyz, 4),
            "cmd_lin_vel_x_mps": float(self.cmd_lin_vel_x_mps),
            "cmd_lin_vel_y_mps": float(self.cmd_lin_vel_y_mps),
            "cmd_ang_vel_z_radps": float(self.cmd_ang_vel_z_radps),
            "actual_lin_vel_x_mps": float(self.actual_lin_vel_x_mps),
            "actual_lin_vel_y_mps": float(self.actual_lin_vel_y_mps),
            "actual_ang_vel_z_radps": float(self.actual_ang_vel_z_radps),
            "joint_vel_radps": _fixed(self.joint_vel_radps, N_JOINTS),
            "contact_forces_n": _fixed(self.contact_forces_n, N_FEET),
        }


def _fixed(values, width: int) -> list[float]:
    """Coerce to a fixed-width float list, zero padding or truncating.

    The Arrow schema uses fixed size lists, so a ragged input has to be
    resolved before the write rather than raising inside pyarrow with a
    message that does not name the offending field.
    """
    arr = np.asarray(values, dtype=np.float32).ravel()
    if arr.size == width:
        return [float(v) for v in arr]
    out = np.zeros(width, dtype=np.float32)
    n = min(width, arr.size)
    out[:n] = arr[:n]
    return [float(v) for v in out]


class EpisodeTelemetryWriter:
    """Buffered parquet writer for per-step episode telemetry.

    A rollout produces one row per env per control step, so the row count
    grows as ``num_episodes * episode_length``. Rows are flushed in row
    groups to keep peak memory bounded on a long evaluation.

    Use as a context manager so the tail of the buffer is flushed even when
    the rollout raises::

        with EpisodeTelemetryWriter(path) as tel:
            tel.append(step)
    """

    def __init__(self, path: str | Path, row_group_size: int = 4096) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.row_group_size = int(row_group_size)
        self._writer: pq.ParquetWriter | None = None
        self._buffer: list[dict] = []
        self._rows_written = 0

    def __enter__(self) -> EpisodeTelemetryWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def rows_written(self) -> int:
        return self._rows_written + len(self._buffer)

    def append(self, step: TelemetryStep) -> None:
        self._buffer.append(step.to_row())
        if len(self._buffer) >= self.row_group_size:
            self._flush()

    def extend(self, steps) -> None:
        for step in steps:
            self.append(step)

    def close(self) -> None:
        if self._buffer:
            self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=ARROW_SCHEMA)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, ARROW_SCHEMA, compression="zstd")
        self._writer.write_table(table)
        self._rows_written += len(self._buffer)
        self._buffer.clear()


def write_episode_telemetry(path: str | Path, steps) -> Path:
    """Write every step in ``steps`` to one parquet, returning the path."""
    with EpisodeTelemetryWriter(path) as writer:
        writer.extend(steps)
    return Path(path)


def validate_table(table, *, source: str = "<table>") -> None:
    """Fail-closed validation of an episode-telemetry Arrow table."""
    missing = [name for name in REQUIRED_COLUMNS if name not in table.schema.names]
    if missing:
        raise EpisodeTelemetrySchemaError(
            f"{source}: missing required column(s): {', '.join(sorted(missing))}"
        )
    if table.num_rows == 0:
        raise EpisodeTelemetrySchemaError(
            f"{source}: artifact carries zero telemetry rows; an episode with no "
            "steps is not a readable trajectory"
        )

    versions: set[str] = set()
    meta = table.schema.metadata or {}
    if SCHEMA_VERSION_KEY in meta:
        versions.add(meta[SCHEMA_VERSION_KEY].decode())
    for value in table.column("schema_version").to_pylist():
        if value is None:
            raise EpisodeTelemetrySchemaError(f"{source}: null schema_version in a row")
        versions.add(str(value))
    if not versions:
        raise EpisodeTelemetrySchemaError(f"{source}: no schema version found")
    unexpected = sorted(v for v in versions if v != SCHEMA_VERSION)
    if unexpected:
        raise EpisodeTelemetrySchemaError(
            f"{source}: schema version mismatch, expected {SCHEMA_VERSION!r} but "
            f"artifact declares {unexpected!r}"
        )

    nulled = sorted(name for name in REQUIRED_COLUMNS if table.column(name).null_count > 0)
    if nulled:
        raise EpisodeTelemetrySchemaError(
            f"{source}: null value(s) in required column(s): {', '.join(nulled)}"
        )


def table_units(table, *, source: str = "<table>") -> dict[str, str]:
    """Return the units map stamped into the artifact, raising if absent."""
    meta = table.schema.metadata or {}
    if UNITS_KEY not in meta:
        raise EpisodeTelemetrySchemaError(f"{source}: units map missing from schema metadata")
    try:
        units = json.loads(meta[UNITS_KEY].decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EpisodeTelemetrySchemaError(
            f"{source}: units map is not valid JSON: {exc}"
        ) from exc
    if not isinstance(units, dict):
        raise EpisodeTelemetrySchemaError(
            f"{source}: units map is not a JSON object, got {type(units).__name__}"
        )
    if not units:
        raise EpisodeTelemetrySchemaError(f"{source}: units map is empty")
    return {str(k): str(v) for k, v in units.items()}


def load_episode_signals(path: str | Path) -> dict[tuple[str, int, int], dict]:
    """Load one telemetry parquet and group it into per-episode signal blocks.

    Returns a mapping from the episode key ``(run_id, seed, episode_id)`` to
    a dict of numpy arrays. The dict keys are chosen to match the field names
    of the downstream six-mode detector's telemetry container, so a consumer
    can construct it directly without a translation layer and without
    importing Phoenix.

    Rows are sorted by ``step_index`` within an episode, so a reader does not
    depend on parquet row order.
    """
    src = Path(path)
    if not src.exists():
        raise EpisodeTelemetrySchemaError(f"{src}: episode-telemetry artifact does not exist")
    if src.is_dir():
        raise EpisodeTelemetrySchemaError(
            f"{src}: is a directory, not an episode-telemetry parquet"
        )
    try:
        table = pq.read_table(src)
    except EpisodeTelemetrySchemaError:
        raise
    except Exception as exc:  # pyarrow raises several unrelated error types
        raise EpisodeTelemetrySchemaError(
            f"{src}: not a readable episode-telemetry parquet: {exc}"
        ) from exc
    validate_table(table, source=str(src))
    table_units(table, source=str(src))

    grouped: dict[tuple[str, int, int], list[dict]] = {}
    for row in table.to_pylist():
        key = (str(row["run_id"]), int(row["seed"]), int(row["episode_id"]))
        grouped.setdefault(key, []).append(row)

    out: dict[tuple[str, int, int], dict] = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda r: int(r["step_index"]))
        dt_values = {round(float(r["control_dt_s"]), 12) for r in rows}
        if len(dt_values) != 1:
            raise EpisodeTelemetrySchemaError(
                f"{src}: episode {key} carries more than one control_dt_s: "
                f"{sorted(dt_values)}"
            )
        out[key] = {
            "dt_s": float(rows[0]["control_dt_s"]),
            "pitch_rad": np.array([r["pitch_rad"] for r in rows], dtype=np.float64),
            "roll_rad": np.array([r["roll_rad"] for r in rows], dtype=np.float64),
            "base_height_m": np.array([r["base_height_m"] for r in rows], dtype=np.float64),
            "cmd_lin_vel": np.array(
                [[r["cmd_lin_vel_x_mps"], r["cmd_lin_vel_y_mps"]] for r in rows],
                dtype=np.float64,
            ),
            "actual_lin_vel": np.array(
                [[r["actual_lin_vel_x_mps"], r["actual_lin_vel_y_mps"]] for r in rows],
                dtype=np.float64,
            ),
            "joint_vel": np.array([r["joint_vel_radps"] for r in rows], dtype=np.float64),
            "contact_forces": np.array([r["contact_forces_n"] for r in rows], dtype=np.float64),
        }
    return out


__all__ = [
    "ARROW_SCHEMA",
    "KEY_COLUMNS",
    "N_FEET",
    "N_JOINTS",
    "REQUIRED_COLUMNS",
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_KEY",
    "UNITS",
    "UNITS_KEY",
    "EpisodeTelemetrySchemaError",
    "EpisodeTelemetryWriter",
    "TelemetryStep",
    "load_episode_signals",
    "quat_wxyz_to_euler",
    "table_units",
    "validate_table",
    "write_episode_telemetry",
]
