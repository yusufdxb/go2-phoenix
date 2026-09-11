"""Unit tests for the ROS 2 policy node that don't need rclpy.

The policy node's runtime class (``_PhoenixPolicyNode``) imports rclpy
inside ``__init__``, so it can't be constructed in CI. What CAN be
constructed is a stub with the handful of attributes ``_latch_abort``
touches — that's enough to cover the parquet-footer-on-abort
regression surfaced in the 2026-04-20 lab findings (bug #1).
"""

from __future__ import annotations

import time
import types

import numpy as np
import pyarrow.parquet as pq
import pytest

from phoenix.real_world.failure_detector import FailureDetector, FailureEvent, FailureMode
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


# ---------------------------------------------------------------------------
# Telemetry logging path: odometry -> base_pos/base_lin_vel_body, foot force
# -> contact_forces, and FailureDetector wiring. Same "bind the unbound
# method onto a bare stub" trick as _latch_abort above, so this exercises
# the real _log_step / _evaluate_failure implementations without rclpy.
# ---------------------------------------------------------------------------


def _fake_odom(position_xyz, linear_xyz):
    position = types.SimpleNamespace(x=position_xyz[0], y=position_xyz[1], z=position_xyz[2])
    linear = types.SimpleNamespace(x=linear_xyz[0], y=linear_xyz[1], z=linear_xyz[2])
    return types.SimpleNamespace(
        pose=types.SimpleNamespace(pose=types.SimpleNamespace(position=position)),
        twist=types.SimpleNamespace(twist=types.SimpleNamespace(linear=linear)),
    )


class _FakeDetector:
    """Stand-in for FailureDetector: records call kwargs, returns a canned event."""

    def __init__(self, event=None, raise_exc=None):
        self.event = event
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    def step(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.event


class _StubTelemetryNode:
    """Bare object standing in for the subset of ``_PhoenixPolicyNode`` state
    that ``_log_step`` / ``_evaluate_failure`` touch."""

    def __init__(
        self,
        *,
        logger=None,
        seen_odom=False,
        odom_ns=None,
        odom_msg=None,
        seen_foot_force=False,
        foot_force_ns=None,
        foot_force=None,
        velocity_command=None,
        telemetry_timeout_s=0.5,
        failure_detector=None,
        step_idx=0,
    ) -> None:
        self._logger = logger
        self._seen_odom = seen_odom
        self._latest_odom_ns = odom_ns
        self._latest_odom = odom_msg
        self._seen_foot_force = seen_foot_force
        self._latest_foot_force_ns = foot_force_ns
        self._latest_foot_force = (
            foot_force if foot_force is not None else np.zeros(4, dtype=np.float32)
        )
        self._velocity_command = (
            velocity_command if velocity_command is not None else np.zeros(3, dtype=np.float32)
        )
        self.telemetry_timeout_s = telemetry_timeout_s
        self._failure_detector = failure_detector if failure_detector is not None else FailureDetector()
        self._started_at = time.monotonic()
        self._step_idx = step_idx


def _log_step(stub: _StubTelemetryNode, **kwargs) -> None:
    _PhoenixPolicyNode._log_step(stub, **kwargs)


def _evaluate_failure(stub: _StubTelemetryNode, **kwargs):
    return _PhoenixPolicyNode._evaluate_failure(stub, **kwargs)


_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def test_log_step_odom_fresh_populates_base_pos_and_vel(tmp_path) -> None:
    path = tmp_path / "odom_fresh.parquet"
    logger = TrajectoryLogger(path, row_group_size=4)
    odom = _fake_odom(position_xyz=(1.0, 2.0, 0.4), linear_xyz=(0.5, 0.0, 0.0))
    stub = _StubTelemetryNode(
        logger=logger,
        seen_odom=True,
        odom_ns=time.monotonic_ns(),
        odom_msg=odom,
    )

    _log_step(
        stub,
        q=np.zeros(12, dtype=np.float32),
        qd=np.zeros(12, dtype=np.float32),
        action=np.zeros(12, dtype=np.float32),
        quat_xyzw=_IDENTITY_QUAT,
        ang_vel=np.zeros(3, dtype=np.float32),
    )
    logger.close()

    table = pq.read_table(path)
    assert table.column("odom_valid").to_pylist() == [True]
    base_pos = table.column("base_pos").to_pylist()[0]
    assert np.allclose(base_pos, [1.0, 2.0, 0.4])
    # Identity orientation: world_to_body rotation is the identity, so the
    # logged body velocity equals the raw odom twist.
    base_lin_vel = table.column("base_lin_vel_body").to_pylist()[0]
    assert np.allclose(base_lin_vel, [0.5, 0.0, 0.0])


def test_log_step_odom_never_seen_falls_back_to_zero_and_marks_invalid(tmp_path) -> None:
    path = tmp_path / "odom_absent.parquet"
    logger = TrajectoryLogger(path, row_group_size=4)
    stub = _StubTelemetryNode(logger=logger, seen_odom=False)

    _log_step(
        stub,
        q=np.zeros(12, dtype=np.float32),
        qd=np.zeros(12, dtype=np.float32),
        action=np.zeros(12, dtype=np.float32),
        quat_xyzw=_IDENTITY_QUAT,
        ang_vel=np.zeros(3, dtype=np.float32),
    )
    logger.close()

    table = pq.read_table(path)
    assert table.column("odom_valid").to_pylist() == [False]
    assert np.allclose(table.column("base_pos").to_pylist()[0], [0.0, 0.0, 0.0])
    assert np.allclose(table.column("base_lin_vel_body").to_pylist()[0], [0.0, 0.0, 0.0])


def test_log_step_odom_stale_treated_as_absent(tmp_path) -> None:
    path = tmp_path / "odom_stale.parquet"
    logger = TrajectoryLogger(path, row_group_size=4)
    odom = _fake_odom(position_xyz=(9.0, 9.0, 9.0), linear_xyz=(9.0, 9.0, 9.0))
    stub = _StubTelemetryNode(
        logger=logger,
        seen_odom=True,
        # 2 seconds old vs a 0.5s timeout.
        odom_ns=time.monotonic_ns() - int(2.0 * 1e9),
        odom_msg=odom,
        telemetry_timeout_s=0.5,
    )

    _log_step(
        stub,
        q=np.zeros(12, dtype=np.float32),
        qd=np.zeros(12, dtype=np.float32),
        action=np.zeros(12, dtype=np.float32),
        quat_xyzw=_IDENTITY_QUAT,
        ang_vel=np.zeros(3, dtype=np.float32),
    )
    logger.close()

    table = pq.read_table(path)
    assert table.column("odom_valid").to_pylist() == [False]
    assert np.allclose(table.column("base_pos").to_pylist()[0], [0.0, 0.0, 0.0])


def test_log_step_foot_force_fresh_vs_stale(tmp_path) -> None:
    path = tmp_path / "foot_force.parquet"
    logger = TrajectoryLogger(path, row_group_size=4)
    stub = _StubTelemetryNode(
        logger=logger,
        seen_foot_force=True,
        foot_force_ns=time.monotonic_ns(),
        foot_force=np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
    )

    _log_step(
        stub,
        q=np.zeros(12, dtype=np.float32),
        qd=np.zeros(12, dtype=np.float32),
        action=np.zeros(12, dtype=np.float32),
        quat_xyzw=_IDENTITY_QUAT,
        ang_vel=np.zeros(3, dtype=np.float32),
    )

    # Now go stale and log a second row.
    stub._latest_foot_force_ns = time.monotonic_ns() - int(2.0 * 1e9)
    stub._step_idx = 1
    _log_step(
        stub,
        q=np.zeros(12, dtype=np.float32),
        qd=np.zeros(12, dtype=np.float32),
        action=np.zeros(12, dtype=np.float32),
        quat_xyzw=_IDENTITY_QUAT,
        ang_vel=np.zeros(3, dtype=np.float32),
    )
    logger.close()

    table = pq.read_table(path)
    contact = table.column("contact_forces").to_pylist()
    assert np.allclose(contact[0], [10.0, 20.0, 30.0, 40.0])
    assert np.allclose(contact[1], [0.0, 0.0, 0.0, 0.0])


def test_evaluate_failure_forwards_none_event() -> None:
    stub = _StubTelemetryNode(failure_detector=_FakeDetector(event=None))
    flag, mode = _evaluate_failure(
        stub,
        quat_xyzw=_IDENTITY_QUAT,
        odom_valid=True,
        base_pos=np.asarray([0.0, 0.0, 0.4], dtype=np.float32),
        base_lin_vel_body=np.zeros(3, dtype=np.float32),
    )
    assert flag is False
    assert mode is None


def test_evaluate_failure_forwards_real_event() -> None:
    event = FailureEvent(mode=FailureMode.COLLAPSE, timestamp_s=1.0, detail={})
    stub = _StubTelemetryNode(failure_detector=_FakeDetector(event=event))
    flag, mode = _evaluate_failure(
        stub,
        quat_xyzw=_IDENTITY_QUAT,
        odom_valid=True,
        base_pos=np.asarray([0.0, 0.0, 0.05], dtype=np.float32),
        base_lin_vel_body=np.zeros(3, dtype=np.float32),
    )
    assert flag is True
    assert mode == "collapse"


def test_evaluate_failure_swallows_exceptions() -> None:
    stub = _StubTelemetryNode(failure_detector=_FakeDetector(raise_exc=RuntimeError("boom")))
    flag, mode = _evaluate_failure(
        stub,
        quat_xyzw=_IDENTITY_QUAT,
        odom_valid=True,
        base_pos=np.asarray([0.0, 0.0, 0.4], dtype=np.float32),
        base_lin_vel_body=np.zeros(3, dtype=np.float32),
    )
    assert flag is False
    assert mode is None


def test_evaluate_failure_odom_invalid_uses_inf_height_and_command_as_actual() -> None:
    detector = _FakeDetector(event=None)
    stub = _StubTelemetryNode(
        failure_detector=detector,
        velocity_command=np.asarray([0.5, 0.1, 0.0], dtype=np.float32),
    )
    _evaluate_failure(
        stub,
        quat_xyzw=_IDENTITY_QUAT,
        odom_valid=False,
        base_pos=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        base_lin_vel_body=np.zeros(3, dtype=np.float32),
    )
    assert len(detector.calls) == 1
    call = detector.calls[0]
    assert call["base_height_m"] == float("inf")
    # actual_lin_vel falls back to the commanded velocity so cmd == actual
    # and the slip condition (cmd_speed >> actual_speed) can never fire on
    # a step where we have no real measurement.
    assert np.allclose(call["actual_lin_vel"], [0.5, 0.1])
    assert np.allclose(call["cmd_lin_vel"], [0.5, 0.1])


def test_evaluate_failure_odom_valid_uses_real_height_and_velocity() -> None:
    detector = _FakeDetector(event=None)
    stub = _StubTelemetryNode(failure_detector=detector)
    _evaluate_failure(
        stub,
        quat_xyzw=_IDENTITY_QUAT,
        odom_valid=True,
        base_pos=np.asarray([1.0, 2.0, 0.33], dtype=np.float32),
        base_lin_vel_body=np.asarray([0.7, -0.2, 0.0], dtype=np.float32),
    )
    call = detector.calls[0]
    assert call["base_height_m"] == pytest.approx(0.33, abs=1e-6)
    assert np.allclose(call["actual_lin_vel"], [0.7, -0.2])


def test_evaluate_failure_real_detector_trips_on_attitude() -> None:
    # Pure-roll quaternion: (sin(theta/2), 0, 0, cos(theta/2)) gives
    # roll == theta exactly under _rpy_from_quat_xyzw (see module docstring).
    theta = 1.0  # > FailureThresholds.roll_rad default (0.6)
    quat = (float(np.sin(theta / 2)), 0.0, 0.0, float(np.cos(theta / 2)))
    stub = _StubTelemetryNode(failure_detector=FailureDetector())
    flag, mode = _evaluate_failure(
        stub,
        quat_xyzw=quat,
        odom_valid=True,
        base_pos=np.asarray([0.0, 0.0, 0.4], dtype=np.float32),
        base_lin_vel_body=np.zeros(3, dtype=np.float32),
    )
    assert flag is True
    assert mode == "attitude"
