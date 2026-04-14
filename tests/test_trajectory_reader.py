"""Round-trip test: logger writes a parquet, reader reads initial state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phoenix.real_world.trajectory_logger import TrajectoryLogger, TrajectoryStep
from phoenix.replay.trajectory_reader import TrajectoryReader, load_initial_state


def _write_tiny(path: Path, n: int = 5) -> None:
    with TrajectoryLogger(path) as log:
        for i in range(n):
            log.append(
                TrajectoryStep(
                    step=i,
                    timestamp_s=0.02 * i,
                    base_pos=np.asarray([float(i), 0.0, 0.4], dtype=np.float32),
                    base_quat=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                    base_lin_vel_body=np.zeros(3, dtype=np.float32),
                    base_ang_vel_body=np.zeros(3, dtype=np.float32),
                    joint_pos=np.arange(12, dtype=np.float32),
                    joint_vel=np.zeros(12, dtype=np.float32),
                    command_vel=np.asarray([0.5, 0.0, 0.0], dtype=np.float32),
                    action=np.zeros(12, dtype=np.float32),
                    contact_forces=np.ones(4, dtype=np.float32),
                    failure_flag=(i == 3),
                    failure_mode="attitude" if i == 3 else None,
                )
            )


def test_reader_counts_rows(tmp_path: Path) -> None:
    p = tmp_path / "t.parquet"
    _write_tiny(p, n=5)
    r = TrajectoryReader(p)
    assert len(r) == 5


def test_reader_exposes_failure_indices(tmp_path: Path) -> None:
    p = tmp_path / "t.parquet"
    _write_tiny(p, n=5)
    r = TrajectoryReader(p)
    assert r.failure_indices().tolist() == [3]


def test_load_initial_state_matches_write(tmp_path: Path) -> None:
    p = tmp_path / "t.parquet"
    _write_tiny(p, n=5)
    st = load_initial_state(p, row=0)
    assert np.allclose(st.base_pos, [0.0, 0.0, 0.4])
    assert np.allclose(st.joint_pos, np.arange(12))
    assert np.allclose(st.command_vel, [0.5, 0.0, 0.0])


def test_load_initial_state_out_of_range(tmp_path: Path) -> None:
    p = tmp_path / "t.parquet"
    _write_tiny(p, n=3)
    with pytest.raises(IndexError):
        load_initial_state(p, row=10)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TrajectoryReader(tmp_path / "nope.parquet")
