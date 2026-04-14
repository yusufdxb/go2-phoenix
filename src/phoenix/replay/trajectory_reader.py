"""Read Phoenix trajectory Parquet files back into Python.

Paired with :class:`phoenix.real_world.TrajectoryLogger` — the same schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


@dataclass
class InitialState:
    """The first-step snapshot needed to spawn a replay in sim."""

    base_pos: np.ndarray  # (3,)
    base_quat: np.ndarray  # (4,) xyzw
    base_lin_vel_body: np.ndarray  # (3,)
    base_ang_vel_body: np.ndarray  # (3,)
    joint_pos: np.ndarray  # (12,)
    joint_vel: np.ndarray  # (12,)
    command_vel: np.ndarray  # (3,)


class TrajectoryReader:
    """Thin reader that exposes the trajectory as numpy arrays.

    Loading is lazy per-column so large trajectories don't blow memory.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Trajectory not found: {self.path}")
        self._table = pq.read_table(self.path)

    def __len__(self) -> int:
        return self._table.num_rows

    def column(self, name: str) -> np.ndarray:
        """Return a column as a numpy array, stacking list-columns into 2D."""
        values = self._table.column(name).to_pylist()
        arr = np.asarray(values)
        return arr

    def failure_indices(self) -> np.ndarray:
        flags = self._table.column("failure_flag").to_pylist()
        return np.asarray([i for i, f in enumerate(flags) if f], dtype=np.int64)


def load_initial_state(path: str | Path, row: int = 0) -> InitialState:
    """Return the :class:`InitialState` at ``row`` of the given trajectory."""
    reader = TrajectoryReader(path)
    if row >= len(reader):
        raise IndexError(f"Row {row} out of range (len={len(reader)})")

    def as_np(name: str) -> np.ndarray:
        return np.asarray(reader._table.column(name)[row].as_py(), dtype=np.float32)

    return InitialState(
        base_pos=as_np("base_pos"),
        base_quat=as_np("base_quat"),
        base_lin_vel_body=as_np("base_lin_vel_body"),
        base_ang_vel_body=as_np("base_ang_vel_body"),
        joint_pos=as_np("joint_pos"),
        joint_vel=as_np("joint_vel"),
        command_vel=as_np("command_vel"),
    )
