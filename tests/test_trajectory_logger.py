"""Round-trip test for the Parquet trajectory logger."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from phoenix.real_world.trajectory_logger import TrajectoryLogger, TrajectoryStep


def _make_step(i: int) -> TrajectoryStep:
    return TrajectoryStep(
        step=i,
        timestamp_s=i * 0.02,
        base_pos=np.asarray([0.1 * i, 0.0, 0.4], dtype=np.float32),
        base_quat=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        base_lin_vel_body=np.asarray([0.5, 0.0, 0.0], dtype=np.float32),
        base_ang_vel_body=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        joint_pos=np.arange(12, dtype=np.float32) * 0.1,
        joint_vel=np.arange(12, dtype=np.float32) * 0.01,
        command_vel=np.asarray([0.5, 0.0, 0.0], dtype=np.float32),
        action=np.arange(12, dtype=np.float32),
        contact_forces=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        failure_flag=(i == 7),
        failure_mode="attitude" if i == 7 else None,
    )


def test_roundtrip_small(tmp_path: Path) -> None:
    p = tmp_path / "traj.parquet"
    with TrajectoryLogger(p, row_group_size=4) as log:
        for i in range(10):
            log.append(_make_step(i))
    assert p.exists() and p.stat().st_size > 0

    table = pq.read_table(p)
    assert table.num_rows == 10
    # step column is monotonically increasing
    steps = table.column("step").to_pylist()
    assert steps == list(range(10))
    # failure_flag set only at step 7
    flags = table.column("failure_flag").to_pylist()
    assert flags[7] is True
    assert sum(flags) == 1
    # base_pos x-component equals 0.1 * step
    xs = [row[0] for row in table.column("base_pos").to_pylist()]
    assert np.allclose(xs, [0.1 * i for i in range(10)], atol=1e-6)


def test_rows_written_property(tmp_path: Path) -> None:
    log = TrajectoryLogger(tmp_path / "t.parquet", row_group_size=3)
    for i in range(5):
        log.append(_make_step(i))
    assert log.rows_written == 5
    log.close()
    assert log.rows_written == 5
