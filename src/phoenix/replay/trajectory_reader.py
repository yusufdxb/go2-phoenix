"""Read Phoenix trajectory Parquet files and failure capsules back into Python.

Paired with :class:`phoenix.real_world.TrajectoryLogger` for the Parquet schema.

The JSON FailureCapsule adds what a per-step Parquet row cannot carry:

``schema_version`` ``"1.0"``
    Frames plus ``failure_onset_index`` / ``pre_failure_start_index``.
``schema_version`` ``"1.1"``
    Adds ``position_frame`` (``env_local``, ``world`` or
    ``odom_boot_relative``, so a consumer never guesses the frame of
    ``base_pos``), ``environment_parameters`` and
    ``disturbances`` (the physics that CAUSED the failure, without which a
    seeded environment is a state with a fresh random cause attached), and
    ``episode_start_index`` (so a consumer can tell a full episode from a
    truncated ring-buffer window and refuse to call the truncated one exact).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .controller_history import ControllerHistory
from .state_adapter import (
    DEFAULT_POSITION_FRAME,
    POSITION_FRAME_ODOM_BOOT_RELATIVE,
    POSITION_FRAMES,
)

#: ``capture_source`` value that implies a boot-relative odometry frame. Kept
#: as a literal so the reader does not import the ROS-side logger module.
CAPTURE_SOURCE_HARDWARE = "hardware"

logger = logging.getLogger("phoenix.replay.trajectory_reader")

#: FailureCapsule schema versions this reader accepts.
CAPSULE_SCHEMA_VERSIONS = ("1.0", "1.1")

#: Parquet file-level key a writer may use to declare its position frame.
#: Mirrors ``phoenix.real_world.trajectory_logger.PARQUET_POSITION_FRAME_KEY``;
#: the two are pinned equal by ``tests/test_trajectory_reader.py``.
PARQUET_POSITION_FRAME_KEY = b"phoenix_position_frame"


@dataclass
class InitialState:
    """A validated trajectory snapshot; velocities are explicitly body-frame.

    ``position_frame`` says how ``base_pos`` must be mapped back into the
    simulator, and ``position_frame_source`` says where that answer came from,
    so a restore can never silently pick the wrong inverse.
    """

    base_pos: np.ndarray  # (3,)
    base_quat: np.ndarray  # (4,) xyzw
    base_lin_vel_body: np.ndarray  # (3,)
    base_ang_vel_body: np.ndarray  # (3,)
    joint_pos: np.ndarray  # (12,)
    joint_vel: np.ndarray  # (12,)
    command_vel: np.ndarray  # (3,)
    position_frame: str = DEFAULT_POSITION_FRAME
    position_frame_source: str = "phoenix_capture_default"
    controller_history: ControllerHistory | None = None
    environment_parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.position_frame not in POSITION_FRAMES:
            raise ValueError(
                f"Unknown position_frame={self.position_frame!r}; expected {POSITION_FRAMES}"
            )


def _validated_environment_parameters(raw) -> dict:
    if raw in (None, {}):
        return {}
    if not isinstance(raw, dict):
        raise ValueError("environment_parameters must be a mapping of name to number")
    parameters = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError("environment_parameters keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"environment_parameters[{name!r}] must be a number")
        if not np.isfinite(value):
            raise ValueError(f"environment_parameters[{name!r}] must be finite")
        parameters[name] = float(value)
    return parameters


class TrajectoryReader:
    """Thin reader that exposes the trajectory as numpy arrays.

    Loading is lazy per-column so large trajectories don't blow memory.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Trajectory not found: {self.path}")
        self.metadata = {}
        self.declared_position_frame = None
        if self.path.suffix == ".json":
            import pyarrow as pa

            data = json.loads(self.path.read_text())
            if data.get("schema_version") not in CAPSULE_SCHEMA_VERSIONS:
                raise ValueError("Unsupported FailureCapsule schema_version")
            if data.get("velocity_frame", "body") != "body":
                raise ValueError("FailureCapsule velocity_frame must be body")
            frame = data.get("position_frame")
            if frame is not None and frame not in POSITION_FRAMES:
                raise ValueError(
                    f"Unknown FailureCapsule position_frame={frame!r}; expected {POSITION_FRAMES}"
                )
            self.declared_position_frame = frame
            self.metadata = {k: v for k, v in data.items() if k != "frames"}
            self._table = pa.Table.from_pylist(data["frames"])
        else:
            self._table = pq.read_table(self.path)
            kv = (self._table.schema.metadata or {}).get(PARQUET_POSITION_FRAME_KEY)
            if kv is not None:
                frame = kv.decode()
                if frame not in POSITION_FRAMES:
                    raise ValueError(
                        f"Unknown parquet position frame {frame!r}; expected {POSITION_FRAMES}"
                    )
                self.declared_position_frame = frame
        self.inferred_position_frame = self._infer_position_frame()
        self.environment_parameters = _validated_environment_parameters(
            self.metadata.get("environment_parameters")
        )
        self.disturbances = list(self.metadata.get("disturbances") or [])

    @property
    def provenance_declared(self) -> bool:
        """Can this capture say where it came from?

        False for a capture written before the ``capture_source`` column
        existed (April 2026 and earlier) that also declares no frame. Those
        files carry a schema BYTE-FOR-BYTE IDENTICAL to a simulator capture's,
        so nothing in the file distinguishes a hardware recording from a sim
        rollout, and the old fallback silently called them ``env_local``.
        A caller that intends to seed a simulator from one must declare the
        frame itself rather than accept a guess.
        """
        return (
            self.declared_position_frame is not None
            or self.inferred_position_frame is not None
            or "capture_source" in self._table.column_names
        )

    def _infer_position_frame(self) -> str | None:
        """Infer the frame from ``capture_source`` when the writer declared none.

        Captures written before 2026-09-17 carry no frame declaration, and the
        old fallback was the simulator convention ``env_local``. For a hardware
        capture that is wrong in the worst way: ``base_pos`` is boot-relative
        odometry whose z is not ground clearance, so a standing robot reads
        z ~ 0 and the seeded sim trunk spawns in the floor. The rows themselves
        say where the capture came from, so use that rather than a default.
        """

        if "capture_source" not in self._table.column_names:
            return None
        import pyarrow.compute as pc

        sources = {
            value.as_py()
            for value in pc.unique(self._table.column("capture_source"))
            if value.as_py() is not None
        }
        if sources != {CAPTURE_SOURCE_HARDWARE}:
            # Inference is only sound for a single-provenance capture. A mixed
            # file has no one frame, so say nothing rather than guess; a real
            # hardware capture is uniform and, since 2026-09-17, also carries
            # an explicit declaration that outranks this path entirely.
            if CAPTURE_SOURCE_HARDWARE in sources:
                logger.warning(
                    "%s mixes capture_source values %s, so its position frame cannot be "
                    "inferred; declare it explicitly before using it as a seed.",
                    self.path,
                    sorted(sources),
                )
            return None
        return POSITION_FRAME_ODOM_BOOT_RELATIVE

    def __len__(self) -> int:
        return self._table.num_rows

    @property
    def column_names(self) -> list[str]:
        return list(self._table.column_names)

    def column(self, name: str) -> np.ndarray:
        """Return a column as a numpy array, stacking list-columns into 2D."""
        values = self._table.column(name).to_pylist()
        arr = np.asarray(values)
        return arr

    def failure_indices(self) -> np.ndarray:
        if "failure_onset_index" in self.metadata:
            return np.asarray([int(self.metadata["failure_onset_index"])], dtype=np.int64)
        flags = self._table.column("failure_flag").to_pylist()
        return np.asarray([i for i, f in enumerate(flags) if f], dtype=np.int64)

    def resolve_position_frame(self, requested: str | None = None) -> tuple[str, str]:
        """Return ``(frame, source)``, refusing to reconcile a real disagreement.

        Precedence is declaration, then the frame inferred from the rows'
        ``capture_source``, then the caller's request, then the simulator
        default. The inferred frame outranks a request on purpose: a caller
        asking for ``env_local`` on a hardware capture is making exactly the
        mistake this method exists to stop, and silently obeying it produces a
        seed that looks valid and is not.
        """
        if requested is not None and requested not in POSITION_FRAMES:
            raise ValueError(f"Unknown position_frame={requested!r}; expected {POSITION_FRAMES}")
        declared = self.declared_position_frame
        if declared is not None and requested is not None and declared != requested:
            raise ValueError(
                f"{self.path} declares position_frame={declared!r} but {requested!r} was "
                "requested; fix the caller or the capture, do not override a declaration"
            )
        if declared is not None:
            return declared, "declared_by_source"
        inferred = self.inferred_position_frame
        if inferred is not None:
            if requested is not None and requested != inferred:
                raise ValueError(
                    f"{self.path} has capture_source={CAPTURE_SOURCE_HARDWARE!r}, so its "
                    f"base_pos is in {inferred!r}, but {requested!r} was requested. A hardware "
                    "capture is not a simulator seed: its z is displacement from the boot pose, "
                    "not height above the floor. Fix the caller, do not override the capture."
                )
            return inferred, "inferred_from_capture_source"
        if requested is not None:
            return requested, "declared_by_caller"
        return DEFAULT_POSITION_FRAME, "phoenix_capture_default"

    def _history_matrix(self, name: str, first: int, last: int) -> np.ndarray | None:
        if name not in self._table.column_names:
            return None
        rows = self._table.column(name)[first : last + 1].to_pylist()
        if any(row is None for row in rows):
            raise ValueError(f"Missing {name} in controller history rows {first}..{last}")
        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != last - first + 1:
            raise ValueError(f"Column {name} is not a per-step vector column")
        if not np.isfinite(matrix).all():
            raise ValueError(f"Column {name} has non-finite values in rows {first}..{last}")
        return matrix

    def controller_history(self, row: int, history_rows: int) -> ControllerHistory | None:
        """Build the controller history ending at ``row``.

        ``history_rows`` of 0 returns ``None`` (a state-only seed). A negative
        value takes every recorded row up to and including ``row``.
        """
        if history_rows == 0:
            return None
        if history_rows < 0:
            first = 0
        else:
            first = max(0, row - history_rows + 1)
        actions = self._history_matrix("action", first, row)
        if actions is None:
            raise ValueError(
                f"{self.path} has no 'action' column, so last_action cannot be reconstructed; "
                "seed this trajectory with history_rows=0 and accept a state-only seed"
            )
        targets = self._history_matrix("joint_target", first, row)
        episode_start = self.metadata.get("episode_start_index")
        starts_at_episode_start = episode_start is not None and int(episode_start) == first
        control_dt = self.metadata.get("control_dt")
        if control_dt is None and "timestamp_s" in self._table.column_names and len(self) > 1:
            stamps = self.column("timestamp_s").astype(float)
            control_dt = float(np.median(np.diff(stamps)))
        return ControllerHistory(
            actions=actions,
            joint_targets=targets,
            starts_at_episode_start=bool(starts_at_episode_start),
            control_dt=float(control_dt) if control_dt else None,
            source_rows=(first, row),
        )


def undeclared_provenance_problem(path, position_frame: str | None = None) -> str | None:
    """Why this capture may not be seeded into a simulator, or ``None``.

    A capture written before the ``capture_source`` column existed carries a
    parquet schema BYTE-FOR-BYTE IDENTICAL to a simulator capture's, so nothing
    in the file says whether it is a sim rollout or a boot-relative robot
    recording. The old behaviour resolved it to the simulator convention
    ``env_local``, which for a hardware capture writes a trunk height that was
    never measured. Refusing and making the caller declare is the only honest
    answer; guessing is what produced fabricated seeds.

    Returns the operator-facing reason string when the capture must be refused,
    so the caller can log it and exit. ``position_frame`` being supplied by the
    caller IS the declaration, and clears the refusal.
    """

    if position_frame is not None:
        return None
    if TrajectoryReader(path).provenance_declared:
        return None
    return (
        f"{path} declares no position frame and has no capture_source column, so whether "
        "it is a simulator rollout or a boot-relative robot recording cannot be determined "
        "from the file: the two schemas are identical. Seeding it as env_local is a guess "
        "that spawns the trunk at a fabricated height when the guess is wrong. Pass "
        "--position-frame to declare what this capture actually is."
    )


def load_initial_state(
    path: str | Path,
    row: int = 0,
    *,
    position_frame: str | None = None,
    history_rows: int = 0,
) -> InitialState:
    """Return the :class:`InitialState` at ``row`` of the given trajectory."""
    reader = TrajectoryReader(path)
    if row < 0 or row >= len(reader):
        raise IndexError(f"Row {row} out of range (len={len(reader)})")

    def as_np(name: str) -> np.ndarray:
        if name not in reader._table.column_names:
            raise ValueError(f"Missing required seed state: {name}")
        value = reader._table.column(name)[row].as_py()
        if value is None:
            raise ValueError(f"Missing required seed state: {name} at row {row}")
        arr = np.asarray(value, dtype=np.float32)
        expected = 4 if name == "base_quat" else (12 if name.startswith("joint_") else 3)
        if arr.shape != (expected,) or not np.isfinite(arr).all():
            raise ValueError(f"Invalid seed state {name}: expected {expected} finite values")
        if name == "base_quat" and not np.isclose(np.linalg.norm(arr), 1.0, atol=1e-3):
            raise ValueError("base_quat must be a normalized xyzw quaternion")
        return arr

    frame, frame_source = reader.resolve_position_frame(position_frame)
    return InitialState(
        base_pos=as_np("base_pos"),
        base_quat=as_np("base_quat"),
        base_lin_vel_body=as_np("base_lin_vel_body"),
        base_ang_vel_body=as_np("base_ang_vel_body"),
        joint_pos=as_np("joint_pos"),
        joint_vel=as_np("joint_vel"),
        command_vel=as_np("command_vel"),
        position_frame=frame,
        position_frame_source=frame_source,
        controller_history=reader.controller_history(row, history_rows),
        environment_parameters=dict(reader.environment_parameters),
    )
