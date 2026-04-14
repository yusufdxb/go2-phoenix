"""Log real-robot rollouts to Apache Parquet for later sim replay.

Each row captures one control step (50 Hz). The schema is intentionally
stable: the :mod:`phoenix.replay` module reads it back into Isaac Sim.

.. code-block:: text

    step (int64)
    timestamp_s (float64)
    base_pos (list<float32>[3])
    base_quat (list<float32>[4])           # (x,y,z,w)
    base_lin_vel_body (list<float32>[3])
    base_ang_vel_body (list<float32>[3])
    joint_pos (list<float32>[12])
    joint_vel (list<float32>[12])
    command_vel (list<float32>[3])
    action (list<float32>[12])             # raw policy output, unscaled
    contact_forces (list<float32>[4])      # per-foot normal force
    failure_flag (bool)
    failure_mode (str, nullable)

Writer uses row-group buffering to keep memory bounded on long rollouts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class TrajectoryStep:
    step: int
    timestamp_s: float
    base_pos: np.ndarray  # (3,)
    base_quat: np.ndarray  # (4,) xyzw
    base_lin_vel_body: np.ndarray  # (3,)
    base_ang_vel_body: np.ndarray  # (3,)
    joint_pos: np.ndarray  # (12,)
    joint_vel: np.ndarray  # (12,)
    command_vel: np.ndarray  # (3,)
    action: np.ndarray  # (12,)
    contact_forces: np.ndarray  # (4,)
    failure_flag: bool = False
    failure_mode: str | None = None


_SCHEMA = pa.schema(
    [
        ("step", pa.int64()),
        ("timestamp_s", pa.float64()),
        ("base_pos", pa.list_(pa.float32(), 3)),
        ("base_quat", pa.list_(pa.float32(), 4)),
        ("base_lin_vel_body", pa.list_(pa.float32(), 3)),
        ("base_ang_vel_body", pa.list_(pa.float32(), 3)),
        ("joint_pos", pa.list_(pa.float32(), 12)),
        ("joint_vel", pa.list_(pa.float32(), 12)),
        ("command_vel", pa.list_(pa.float32(), 3)),
        ("action", pa.list_(pa.float32(), 12)),
        ("contact_forces", pa.list_(pa.float32(), 4)),
        ("failure_flag", pa.bool_()),
        ("failure_mode", pa.string()),
    ]
)


class TrajectoryLogger:
    """Buffered Parquet writer. Call :meth:`append` then :meth:`close`.

    Use as a context manager to guarantee flush on exceptions::

        with TrajectoryLogger("rollout.parquet") as log:
            for step in rollout():
                log.append(step)
    """

    def __init__(self, path: str | Path, row_group_size: int = 512) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.row_group_size = row_group_size
        self._writer: pq.ParquetWriter | None = None
        self._buffer: list[dict] = []
        self._rows_written = 0

    def __enter__(self) -> TrajectoryLogger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def rows_written(self) -> int:
        return self._rows_written + len(self._buffer)

    def append(self, step: TrajectoryStep) -> None:
        row = asdict(step)
        # dataclasses.asdict doesn't descend into numpy arrays cleanly, so we
        # convert array fields by name.
        for k in (
            "base_pos",
            "base_quat",
            "base_lin_vel_body",
            "base_ang_vel_body",
            "joint_pos",
            "joint_vel",
            "command_vel",
            "action",
            "contact_forces",
        ):
            row[k] = np.asarray(row[k], dtype=np.float32).tolist()
        self._buffer.append(row)
        if len(self._buffer) >= self.row_group_size:
            self._flush()

    def close(self) -> None:
        if self._buffer:
            self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=_SCHEMA)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, _SCHEMA, compression="zstd")
        self._writer.write_table(table)
        self._rows_written += len(self._buffer)
        self._buffer.clear()
