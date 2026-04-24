"""Unit tests for the ROS 2 policy node that don't need rclpy.

The policy node's runtime class (``_PhoenixPolicyNode``) imports rclpy
inside ``__init__``, so it can't be constructed in CI. What CAN be
constructed is a stub with the handful of attributes ``_latch_abort``
touches — that's enough to cover the parquet-footer-on-abort
regression surfaced in the 2026-04-20 lab findings (bug #1).
"""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest

from phoenix.real_world.trajectory_logger import TrajectoryLogger, TrajectoryStep
from phoenix.sim2real.ros2_policy_node import _PhoenixPolicyNode


def _make_step(step_idx: int) -> TrajectoryStep:
    return TrajectoryStep(
        step=step_idx,
        timestamp_s=float(step_idx) / 50.0,
        base_pos=np.zeros(3, dtype=np.float32),
        base_quat=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        base_lin_vel_body=np.zeros(3, dtype=np.float32),
        base_ang_vel_body=np.zeros(3, dtype=np.float32),
        joint_pos=np.zeros(12, dtype=np.float32),
        joint_vel=np.zeros(12, dtype=np.float32),
        command_vel=np.zeros(3, dtype=np.float32),
        action=np.zeros(12, dtype=np.float32),
        contact_forces=np.zeros(4, dtype=np.float32),
        failure_flag=False,
        failure_mode=None,
    )


class _StubNode:
    """Bare object standing in for ``_PhoenixPolicyNode`` state used by
    ``_latch_abort``. We bind the unbound method via ``__func__`` so we're
    calling the real implementation."""

    def __init__(self, logger: TrajectoryLogger | None) -> None:
        self._estopped = False
        self._abort_reason: str | None = None
        self._logger = logger


def _latch(stub: _StubNode, reason: str) -> None:
    _PhoenixPolicyNode._latch_abort(stub, reason)


def test_latch_abort_flushes_parquet_footer(tmp_path) -> None:
    path = tmp_path / "abort_before_close.parquet"
    logger = TrajectoryLogger(path, row_group_size=16)
    for i in range(10):
        logger.append(_make_step(i))

    stub = _StubNode(logger)
    _latch(stub, "max_runtime")

    # _latch_abort must leave a self-consistent, reader-parseable parquet —
    # even if the surrounding process is killed before shutdown() runs.
    assert stub._estopped is True
    assert stub._abort_reason == "max_runtime"
    assert stub._logger is None, "logger must be released after abort"

    table = pq.read_table(path)
    assert table.num_rows == 10


def test_latch_abort_is_idempotent_with_shutdown(tmp_path) -> None:
    path = tmp_path / "abort_then_shutdown.parquet"
    logger = TrajectoryLogger(path, row_group_size=16)
    logger.append(_make_step(0))

    stub = _StubNode(logger)
    _latch(stub, "external_estop")
    # Simulate the later shutdown() code path: it re-checks
    # ``self._logger is not None`` and so must not double-close.
    assert stub._logger is None
    # Second latch (defensive — control loop might re-enter) is a no-op.
    _latch(stub, "max_runtime")
    assert stub._abort_reason == "max_runtime"

    table = pq.read_table(path)
    assert table.num_rows == 1


def test_latch_abort_without_logger_is_noop() -> None:
    stub = _StubNode(None)
    _latch(stub, "external_estop")
    assert stub._estopped is True
    assert stub._abort_reason == "external_estop"
    assert stub._logger is None


@pytest.mark.parametrize(
    "rows_before_abort",
    [0, 1, 16, 17, 512, 1024],  # spans empty, sub-row-group, exact, multi
)
def test_latch_abort_flushes_any_row_count(tmp_path, rows_before_abort) -> None:
    path = tmp_path / f"abort_{rows_before_abort}.parquet"
    logger = TrajectoryLogger(path, row_group_size=16)
    for i in range(rows_before_abort):
        logger.append(_make_step(i))

    stub = _StubNode(logger)
    _latch(stub, "max_runtime")

    if rows_before_abort == 0:
        # ParquetWriter wasn't opened — no file was written. That's OK;
        # the logger still released cleanly.
        assert not path.exists() or pq.read_table(path).num_rows == 0
    else:
        table = pq.read_table(path)
        assert table.num_rows == rows_before_abort
