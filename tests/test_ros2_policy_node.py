"""Unit tests for the ROS 2 policy node that don't need rclpy.

The policy node's runtime class (``_PhoenixPolicyNode``) imports rclpy
inside ``__init__``, so it can't be constructed in CI. What CAN be
constructed is a stub with the handful of attributes ``_latch_abort``
touches, which is enough to cover the parquet-footer-on-abort
regression surfaced in the 2026-04-20 lab findings (bug #1). The same
trick covers the telemetry and observation-provenance paths below.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import types
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from phoenix.real_world.failure_detector import FailureDetector, FailureEvent, FailureMode
from phoenix.real_world.trajectory_logger import TrajectoryLogger, TrajectoryStep
from phoenix.sim2real.command_wire import KIND_ABORT, KIND_STARTUP_DEFAULT, decode, wire_label
from phoenix.sim2real.deploy_contract import observed_artifact_hashes
from phoenix.sim2real.gate import SensorSnapshot
from phoenix.sim2real.go2_model import POLICY_JOINT_ORDER, TRAINING_DEFAULT_JOINT_POS
from phoenix.sim2real.observation import BaseLinVelSample, JointOrder
from phoenix.sim2real.ros2_policy_node import (
    _PhoenixPolicyNode,
    _require_base_lin_vel_source,
    resolve_startup,
)
from phoenix.sim2real.telemetry import OdomSample


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

    # _latch_abort must leave a self-consistent, reader-parseable parquet,
    # even if the surrounding process is killed before shutdown() runs.
    assert stub._estopped is True
    assert stub._abort_reason == "max_runtime"
    assert stub._logger is None, "logger must be released after abort"

    table = pq.read_table(path)
    assert table.num_rows == 10
    assert table.column("step").to_pylist() == list(range(10))
    assert pq.ParquetFile(path).metadata.num_rows == 10
    assert path.read_bytes()[-4:] == b"PAR1"


def test_latch_abort_is_idempotent_with_shutdown(tmp_path) -> None:
    path = tmp_path / "abort_then_shutdown.parquet"
    logger = TrajectoryLogger(path, row_group_size=16)
    logger.append(_make_step(0))

    stub = _StubNode(logger)
    _latch(stub, "external_estop")
    # Simulate the later shutdown() code path: it re-checks
    # ``self._logger is not None`` and so must not double-close.
    assert stub._logger is None
    # Second latch (defensive, the control loop might re-enter) is a no-op.
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
        # ParquetWriter wasn't opened, so no file was written. That's OK;
        # the logger still released cleanly.
        assert not path.exists() or pq.read_table(path).num_rows == 0
    else:
        table = pq.read_table(path)
        assert table.num_rows == rows_before_abort


# ---------------------------------------------------------------------------
# Observation provenance: base_lin_vel source selection and fail-closed abort.
#
# The regression these lock: ros2_policy_node used to feed the policy
# ``base_lin_vel = np.zeros(3)`` with no config key, no log line, and no
# record in the capture, while training fed that term a real body velocity.
# ---------------------------------------------------------------------------


def _cfg(**observation) -> dict:
    cfg = {"joint_order": list("abc")}
    if observation:
        cfg["observation"] = observation
    return cfg


def test_base_lin_vel_source_is_required() -> None:
    with pytest.raises(ValueError, match="observation.base_lin_vel_source"):
        _require_base_lin_vel_source(_cfg())


def test_base_lin_vel_source_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="not one of"):
        _require_base_lin_vel_source(_cfg(base_lin_vel_source="imu_integration"))


@pytest.mark.parametrize("source", ["zeros", "odom"])
def test_base_lin_vel_source_accepts_documented_values(source, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="phoenix.sim2real.ros2_policy_node"):
        assert _require_base_lin_vel_source(_cfg(base_lin_vel_source=source)) == source
    # Both modes must announce themselves at WARNING. A silent obs-source
    # choice is the defect; a loud one is the contract.
    assert any(source in rec.getMessage() for rec in caplog.records)


class _StubResolveNode:
    """Stub for ``_resolve_base_lin_vel_or_abort``, ``_on_cmd_vel`` and the wire publishers.

    Every method bound here is the REAL implementation, so the tests cover the
    actual latch and publish paths rather than mocks of them.
    """

    def __init__(self, source: str, stand_only: bool = True) -> None:
        self.base_lin_vel_source = source
        self.stand_only = stand_only
        self.joint_order = JointOrder(POLICY_JOINT_ORDER)
        self._estopped = False
        self._abort_reason: str | None = None
        self._logger = None
        self.default_q = np.asarray(
            [TRAINING_DEFAULT_JOINT_POS[n] for n in POLICY_JOINT_ORDER], dtype=np.float32
        )
        self.published: list[tuple[str, list[float]]] = []
        self._float_msg = _FakeMsg
        self._dim_msg = _FakeDim
        self.cmd_pub = _FakePub(self.published)
        self._seq = 0
        self._velocity_command = np.zeros(3, dtype=np.float32)
        self._cmd_vel_received = np.full(3, np.nan)
        self._cmd_vel_ns = None

    def _latch_abort(self, reason: str) -> None:
        _PhoenixPolicyNode._latch_abort(self, reason)

    def _publish_abort_notice(self, reason: str) -> None:
        _PhoenixPolicyNode._publish_abort_notice(self, reason)

    def _publish_startup_default(self) -> None:
        _PhoenixPolicyNode._publish_startup_default(self)

    def _publish_wire(self, *args, **kwargs) -> None:
        _PhoenixPolicyNode._publish_wire(self, *args, **kwargs)

    def _next_seq(self) -> int:
        return _PhoenixPolicyNode._next_seq(self)


class _FakeDim:
    def __init__(self) -> None:
        self.label = ""
        self.size = 0
        self.stride = 0


class _FakeMsg:
    def __init__(self) -> None:
        self.data: list[float] = []
        self.layout = types.SimpleNamespace(dim=[])


class _FakePub:
    def __init__(self, sink: list) -> None:
        self._sink = sink

    def publish(self, msg) -> None:
        label = msg.layout.dim[0].label if msg.layout.dim else ""
        self._sink.append((label, list(msg.data)))


_LABEL = wire_label(POLICY_JOINT_ORDER)


def _resolve(stub, odom):
    return _PhoenixPolicyNode._resolve_base_lin_vel_or_abort(stub, odom)


def _odom_sample(**kw) -> OdomSample:
    defaults = dict(
        fresh=True,
        position=np.asarray([1.0, 2.0, 0.4], dtype=np.float32),
        lin_vel_body=np.asarray([0.5, 0.0, 0.0], dtype=np.float32),
        twist_valid=True,
        provenance="body_passthrough",
    )
    defaults.update(kw)
    return OdomSample(**defaults)


def test_odom_source_uses_the_measurement() -> None:
    stub = _StubResolveNode("odom")
    sample = _resolve(stub, _odom_sample())
    assert sample is not None
    assert np.allclose(sample.value, [0.5, 0.0, 0.0])
    assert sample.measured is True
    assert sample.provenance == "odom:body_passthrough"
    assert stub._estopped is False


def test_odom_source_latches_abort_when_odom_is_unavailable() -> None:
    stub = _StubResolveNode("odom")
    sample = _resolve(stub, _odom_sample(fresh=False, twist_valid=False, provenance="stale"))
    # The whole point: no silent zeros. The node stops instead.
    assert sample is None
    assert stub._estopped is True
    assert "base_lin_vel_unavailable" in (stub._abort_reason or "")
    assert len(stub.published) == 1, "exactly one abort notice, then silence"
    cmd = decode(*stub.published[0], _LABEL)
    assert cmd.kind == KIND_ABORT and cmd.abort_reason == "base_lin_vel_unavailable"
    assert np.allclose(cmd.target, stub.default_q)


def test_zeros_source_never_aborts_and_records_the_substitution() -> None:
    stub = _StubResolveNode("zeros")
    sample = _resolve(stub, _odom_sample(fresh=False, twist_valid=False, provenance="absent"))
    assert sample is not None
    assert np.allclose(sample.value, 0.0)
    assert sample.measured is False
    assert sample.provenance == "zeros:operator_selected"
    # zeros mode keeps the historical control path exactly: no new abort.
    assert stub._estopped is False
    assert stub.published == []


# ---------------------------------------------------------------------------
# Telemetry logging path: odometry -> base_pos/base_lin_vel_body, foot force
# -> contact_forces, and FailureDetector wiring. Same "bind the unbound
# method onto a bare stub" trick as _latch_abort above, so this exercises
# the real _log_step / _sample_odom / _evaluate_failure implementations
# without rclpy.
# ---------------------------------------------------------------------------


def _fake_odom(position_xyz, linear_xyz, child_frame_id="base_link", frame_id="odom"):
    position = types.SimpleNamespace(x=position_xyz[0], y=position_xyz[1], z=position_xyz[2])
    linear = types.SimpleNamespace(x=linear_xyz[0], y=linear_xyz[1], z=linear_xyz[2])
    return types.SimpleNamespace(
        header=types.SimpleNamespace(frame_id=frame_id),
        child_frame_id=child_frame_id,
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
    that ``_log_step`` / ``_sample_odom`` / ``_evaluate_failure`` touch."""

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
        self._failure_detector = (
            failure_detector if failure_detector is not None else FailureDetector()
        )
        self._started_at = time.monotonic()
        self._step_idx = step_idx
        self._last_action = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
        self._attitude_terminal_logged = False
        self.base_lin_vel_source = "zeros"

    def _sample_odom(self, now_ns, quat_xyzw):
        return _PhoenixPolicyNode._sample_odom(self, now_ns, quat_xyzw)

    def _log_step(self, **kwargs) -> None:
        _PhoenixPolicyNode._log_step(self, **kwargs)


def _log_step(stub: _StubTelemetryNode, **kwargs) -> None:
    kwargs.setdefault("odom", _odom_sample())
    kwargs.setdefault(
        "base_lin_vel_sample",
        BaseLinVelSample(
            value=np.zeros(3, dtype=np.float32),
            provenance="zeros:operator_selected",
            measured=False,
        ),
    )
    _PhoenixPolicyNode._log_step(stub, **kwargs)


def _sample_odom_for(stub: _StubTelemetryNode, quat=(0.0, 0.0, 0.0, 1.0)) -> OdomSample:
    return _PhoenixPolicyNode._sample_odom(stub, time.monotonic_ns(), quat)


def _evaluate_failure(stub: _StubTelemetryNode, **kwargs):
    return _PhoenixPolicyNode._evaluate_failure(stub, **kwargs)


_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)
_LOG_KWARGS = dict(
    q=np.zeros(12, dtype=np.float32),
    qd=np.zeros(12, dtype=np.float32),
    action=np.zeros(12, dtype=np.float32),
    quat_xyzw=_IDENTITY_QUAT,
    ang_vel=np.zeros(3, dtype=np.float32),
)


def test_sample_odom_fresh_body_frame_twist_is_passed_through() -> None:
    # child_frame_id == base_link, so per nav_msgs/Odometry the twist is
    # ALREADY body frame. Rotating it here would be the bug.
    odom = _fake_odom(position_xyz=(1.0, 2.0, 0.4), linear_xyz=(0.5, -0.1, 0.0))
    stub = _StubTelemetryNode(seen_odom=True, odom_ns=time.monotonic_ns(), odom_msg=odom)
    # A 90 degree yaw: a spurious world->body rotation would turn
    # (0.5, -0.1) into (-0.1, -0.5) and this test would catch it.
    half = np.pi / 4.0
    quat = (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))
    sample = _sample_odom_for(stub, quat)
    assert sample.fresh is True
    assert sample.twist_valid is True
    assert sample.provenance == "body_passthrough"
    assert np.allclose(sample.lin_vel_body, [0.5, -0.1, 0.0], atol=1e-6)
    assert np.allclose(sample.position, [1.0, 2.0, 0.4])


def test_sample_odom_world_frame_child_is_rotated() -> None:
    odom = _fake_odom(
        position_xyz=(0.0, 0.0, 0.0), linear_xyz=(1.0, 0.0, 0.0), child_frame_id="odom"
    )
    stub = _StubTelemetryNode(seen_odom=True, odom_ns=time.monotonic_ns(), odom_msg=odom)
    half = np.pi / 4.0
    quat = (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))
    sample = _sample_odom_for(stub, quat)
    assert sample.provenance == "rotated_from_world"
    assert np.allclose(sample.lin_vel_body, [0.0, -1.0, 0.0], atol=1e-6)


def test_sample_odom_unknown_child_frame_is_invalid_not_guessed() -> None:
    odom = _fake_odom(
        position_xyz=(0.0, 0.0, 0.0), linear_xyz=(1.0, 0.0, 0.0), child_frame_id="mystery_link"
    )
    stub = _StubTelemetryNode(seen_odom=True, odom_ns=time.monotonic_ns(), odom_msg=odom)
    sample = _sample_odom_for(stub)
    assert sample.fresh is True
    assert sample.twist_valid is False
    assert "unrecognized_child_frame" in sample.provenance


def test_sample_odom_never_seen_is_absent() -> None:
    sample = _sample_odom_for(_StubTelemetryNode(seen_odom=False))
    assert sample.fresh is False
    assert sample.twist_valid is False
    assert sample.provenance == "absent"
    assert np.allclose(sample.position, 0.0)


def test_sample_odom_stale_is_not_used() -> None:
    odom = _fake_odom(position_xyz=(9.0, 9.0, 9.0), linear_xyz=(9.0, 9.0, 9.0))
    stub = _StubTelemetryNode(
        seen_odom=True,
        odom_ns=time.monotonic_ns() - int(2.0 * 1e9),  # 2s old vs a 0.5s timeout
        odom_msg=odom,
        telemetry_timeout_s=0.5,
    )
    sample = _sample_odom_for(stub)
    assert sample.fresh is False
    assert sample.provenance == "stale"
    assert np.allclose(sample.position, 0.0)


def test_log_step_writes_the_odom_sample_it_was_given(tmp_path) -> None:
    path = tmp_path / "odom_fresh.parquet"
    logger = TrajectoryLogger(path, row_group_size=4)
    stub = _StubTelemetryNode(logger=logger)

    _log_step(stub, odom=_odom_sample(), **_LOG_KWARGS)
    logger.close()

    table = pq.read_table(path)
    assert table.column("odom_valid").to_pylist() == [True]
    assert np.allclose(table.column("base_pos").to_pylist()[0], [1.0, 2.0, 0.4])
    assert np.allclose(table.column("base_lin_vel_body").to_pylist()[0], [0.5, 0.0, 0.0])


def test_log_step_odom_absent_marks_invalid_and_zeroes(tmp_path) -> None:
    path = tmp_path / "odom_absent.parquet"
    logger = TrajectoryLogger(path, row_group_size=4)
    stub = _StubTelemetryNode(logger=logger)

    _log_step(
        stub,
        odom=OdomSample(
            fresh=False,
            position=np.zeros(3, dtype=np.float32),
            lin_vel_body=np.zeros(3, dtype=np.float32),
            twist_valid=False,
            provenance="absent",
        ),
        **_LOG_KWARGS,
    )
    logger.close()

    table = pq.read_table(path)
    assert table.column("odom_valid").to_pylist() == [False]
    assert np.allclose(table.column("base_pos").to_pylist()[0], [0.0, 0.0, 0.0])
    assert np.allclose(table.column("base_lin_vel_body").to_pylist()[0], [0.0, 0.0, 0.0])


def test_log_step_records_hardware_provenance(tmp_path) -> None:
    """Item 6: a consumer must be able to tell where a row came from."""
    path = tmp_path / "provenance.parquet"
    logger = TrajectoryLogger(path, row_group_size=4)
    stub = _StubTelemetryNode(logger=logger)

    _log_step(
        stub,
        base_lin_vel_sample=BaseLinVelSample(
            value=np.zeros(3, dtype=np.float32),
            provenance="zeros:operator_selected",
            measured=False,
        ),
        **_LOG_KWARGS,
    )
    logger.close()

    table = pq.read_table(path)
    assert table.column("capture_source").to_pylist() == ["hardware"]
    # The logged velocity is the real odom measurement...
    assert table.column("base_lin_vel_source").to_pylist() == ["odom:body_passthrough"]
    # ...while the policy was fed zeros. A capture that recorded only one of
    # these would look as though the policy saw the logged value.
    assert table.column("obs_base_lin_vel_source").to_pylist() == ["zeros:operator_selected"]
    # foot_force is int16 with no documented calibration; never call it Newtons.
    assert table.column("contact_forces_units").to_pylist() == ["raw_counts_uncalibrated"]


def test_attitude_abort_enqueues_one_labelled_terminal_row_before_close(tmp_path) -> None:
    path = tmp_path / "attitude_terminal.parquet"
    logger = TrajectoryLogger(path, row_group_size=4)
    stub = _StubTelemetryNode(logger=logger, step_idx=25)
    theta = 0.41
    quat = (float(np.sin(theta / 2)), 0.0, 0.0, float(np.cos(theta / 2)))
    now_ns = time.monotonic_ns()
    snap = SensorSnapshot(
        now_ns=now_ns,
        elapsed_s=1.0,
        node_started_ns=0,
        seen_estop=True,
        seen_imu=True,
        seen_joint_state=True,
        estop_last_ns=now_ns,
        estop_value=False,
        imu_last_ns=now_ns,
        joint_state_last_ns=now_ns,
        joint_pos=np.arange(12, dtype=np.float32),
        joint_vel=np.arange(12, dtype=np.float32) * 0.1,
        quat_xyzw=quat,
        ang_vel=(0.1, 0.2, 0.3),
        roll=theta,
        pitch=0.0,
    )

    _PhoenixPolicyNode._log_attitude_terminal(stub, snap)
    _PhoenixPolicyNode._log_attitude_terminal(stub, snap)
    logger.close()

    table = pq.read_table(path)
    assert table.num_rows == 1
    assert table.column("step").to_pylist() == [25]
    assert table.column("failure_flag").to_pylist() == [True]
    assert table.column("failure_mode").to_pylist() == ["attitude"]
    assert table.column("failure_onset_source").to_pylist() == ["safety_gate"]
    assert table.column("action").to_pylist()[0] == pytest.approx(stub._last_action.tolist())
    assert stub._step_idx == 26


def test_log_step_foot_force_fresh_vs_stale(tmp_path) -> None:
    path = tmp_path / "foot_force.parquet"
    logger = TrajectoryLogger(path, row_group_size=4)
    stub = _StubTelemetryNode(
        logger=logger,
        seen_foot_force=True,
        foot_force_ns=time.monotonic_ns(),
        foot_force=np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
    )

    _log_step(stub, **_LOG_KWARGS)

    # Now go stale and log a second row.
    stub._latest_foot_force_ns = time.monotonic_ns() - int(2.0 * 1e9)
    stub._step_idx = 1
    _log_step(stub, **_LOG_KWARGS)
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
        base_lin_vel_body=np.zeros(3, dtype=np.float32),
    )
    assert flag is False
    assert mode is None


def test_evaluate_failure_forwards_real_event() -> None:
    event = FailureEvent(mode=FailureMode.ATTITUDE, timestamp_s=1.0, detail={})
    stub = _StubTelemetryNode(failure_detector=_FakeDetector(event=event))
    flag, mode = _evaluate_failure(
        stub,
        quat_xyzw=_IDENTITY_QUAT,
        odom_valid=True,
        base_lin_vel_body=np.zeros(3, dtype=np.float32),
    )
    assert flag is True
    assert mode == "attitude"


def test_evaluate_failure_swallows_exceptions() -> None:
    stub = _StubTelemetryNode(failure_detector=_FakeDetector(raise_exc=RuntimeError("boom")))
    flag, mode = _evaluate_failure(
        stub,
        quat_xyzw=_IDENTITY_QUAT,
        odom_valid=True,
        base_lin_vel_body=np.zeros(3, dtype=np.float32),
    )
    assert flag is False
    assert mode is None


def test_evaluate_failure_never_claims_a_height_on_hardware() -> None:
    """Item 6 regression: odom Z is boot-relative, not height above ground.

    This used to pass ``base_pos[2]`` straight into the collapse threshold,
    so a robot that booted on a step and walked down was labelled 'collapse'
    while standing normally.
    """
    detector = _FakeDetector(event=None)
    stub = _StubTelemetryNode(failure_detector=detector)
    _evaluate_failure(
        stub,
        quat_xyzw=_IDENTITY_QUAT,
        odom_valid=True,
        base_lin_vel_body=np.asarray([0.7, -0.2, 0.0], dtype=np.float32),
    )
    assert detector.calls[0]["base_height_m"] is None
    assert np.allclose(detector.calls[0]["actual_lin_vel"], [0.7, -0.2])


def test_evaluate_failure_odom_invalid_uses_command_as_actual() -> None:
    detector = _FakeDetector(event=None)
    stub = _StubTelemetryNode(
        failure_detector=detector,
        velocity_command=np.asarray([0.5, 0.1, 0.0], dtype=np.float32),
    )
    _evaluate_failure(
        stub,
        quat_xyzw=_IDENTITY_QUAT,
        odom_valid=False,
        base_lin_vel_body=np.zeros(3, dtype=np.float32),
    )
    call = detector.calls[0]
    assert call["base_height_m"] is None
    # actual_lin_vel falls back to the commanded velocity so cmd == actual
    # and the stall condition (cmd_speed >> actual_speed) can never fire on
    # a step where we have no real measurement.
    assert np.allclose(call["actual_lin_vel"], [0.5, 0.1])
    assert np.allclose(call["cmd_lin_vel"], [0.5, 0.1])


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
        base_lin_vel_body=np.zeros(3, dtype=np.float32),
    )
    assert flag is True
    assert mode == "attitude"


def test_evaluate_failure_real_detector_never_reports_collapse() -> None:
    """No validated ground-relative height source exists on the GO2."""
    stub = _StubTelemetryNode(failure_detector=FailureDetector())
    for _ in range(10):
        flag, mode = _evaluate_failure(
            stub,
            quat_xyzw=_IDENTITY_QUAT,
            odom_valid=True,
            base_lin_vel_body=np.zeros(3, dtype=np.float32),
        )
        assert mode != "collapse"
        assert flag is False


class _StubFootForceNode:
    """Stub for ``_on_foot_force``, which is the only writer of that state."""

    def __init__(self) -> None:
        self._latest_foot_force = np.zeros(4, dtype=np.float32)
        self._latest_foot_force_ns = None
        self._seen_foot_force = False


def test_on_foot_force_accepts_four_values() -> None:
    stub = _StubFootForceNode()
    _PhoenixPolicyNode._on_foot_force(stub, types.SimpleNamespace(data=[1, 2, 3, 4]))
    assert np.allclose(stub._latest_foot_force, [1.0, 2.0, 3.0, 4.0])
    assert stub._seen_foot_force is True
    assert stub._latest_foot_force_ns is not None


@pytest.mark.parametrize("data", [[], [1, 2, 3], [1, 2, 3, 4, 5]])
def test_on_foot_force_drops_malformed_messages(data) -> None:
    """A wrong-length message must not reach the fixed-size parquet column."""
    stub = _StubFootForceNode()
    _PhoenixPolicyNode._on_foot_force(stub, types.SimpleNamespace(data=data))
    assert stub._seen_foot_force is False
    assert stub._latest_foot_force_ns is None
    assert np.allclose(stub._latest_foot_force, 0.0)


def test_log_step_fresh_pose_with_unusable_twist_is_disambiguated(tmp_path) -> None:
    """odom_valid covers the POSE; base_lin_vel_source covers the twist."""
    path = tmp_path / "bad_frame.parquet"
    logger = TrajectoryLogger(path, row_group_size=4)
    stub = _StubTelemetryNode(logger=logger)

    _log_step(
        stub,
        odom=_odom_sample(
            twist_valid=False,
            lin_vel_body=np.zeros(3, dtype=np.float32),
            provenance="unrecognized_child_frame:mystery_link",
        ),
        **_LOG_KWARGS,
    )
    logger.close()

    table = pq.read_table(path)
    assert table.column("odom_valid").to_pylist() == [True]
    assert np.allclose(table.column("base_pos").to_pylist()[0], [1.0, 2.0, 0.4])
    assert np.allclose(table.column("base_lin_vel_body").to_pylist()[0], 0.0)
    assert table.column("base_lin_vel_source").to_pylist() == [
        "unrecognized_child_frame:mystery_link"
    ]


# ---------------------------------------------------------------------------
# STAND-ONLY guard, abort notices, startup refusal (hardware-readiness pass).
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
H25 = REPO_ROOT / "configs/sim2real/deploy_stand_h25.yaml"


def _twist(x, y, wz):
    return types.SimpleNamespace(
        linear=types.SimpleNamespace(x=x, y=y), angular=types.SimpleNamespace(z=wz)
    )


@pytest.mark.parametrize("cmd", [(0.2, 0.0, 0.0), (0.0, -0.1, 0.0), (0.0, 0.0, 0.5)])
def test_stand_only_nonzero_cmd_vel_latches_abort_and_is_never_fed(cmd) -> None:
    stub = _StubResolveNode("zeros")
    _PhoenixPolicyNode._on_cmd_vel(stub, _twist(*cmd))
    assert stub._estopped is True
    assert stub._abort_reason == "walking_command_blocked_stand_only"
    assert np.allclose(stub._velocity_command, 0.0)
    assert len(stub.published) == 1
    decoded = decode(*stub.published[0], _LABEL)
    assert decoded.kind == KIND_ABORT
    assert decoded.abort_reason == "walking_command_blocked_stand_only"
    # A second nonzero command does not produce a second notice.
    _PhoenixPolicyNode._on_cmd_vel(stub, _twist(*cmd))
    assert len(stub.published) == 1


def test_stand_only_zero_cmd_vel_is_recorded_not_fed_and_does_not_abort() -> None:
    stub = _StubResolveNode("zeros")
    _PhoenixPolicyNode._on_cmd_vel(stub, _twist(0.0, 0.0, 0.0))
    assert stub._estopped is False and stub.published == []
    assert np.allclose(stub._cmd_vel_received, 0.0)
    assert np.allclose(stub._velocity_command, 0.0)


def test_stand_only_nan_cmd_vel_aborts() -> None:
    stub = _StubResolveNode("zeros")
    _PhoenixPolicyNode._on_cmd_vel(stub, _twist(float("nan"), 0.0, 0.0))
    assert stub._abort_reason == "walking_command_blocked_stand_only"


def test_startup_default_is_its_own_kind_not_a_policy_target() -> None:
    stub = _StubResolveNode("zeros")
    _PhoenixPolicyNode._publish_startup_default(stub)
    decoded = decode(*stub.published[0], _LABEL)
    assert decoded.kind == KIND_STARTUP_DEFAULT
    assert decoded.fields["stand_only"][0] == 1.0
    assert decoded.fields["obs_source_code"][0] == 0.0


def test_repeated_latches_publish_exactly_one_notice() -> None:
    stub = _StubResolveNode("zeros")
    for reason in ("attitude pitch=0.9 roll=0.0", "max_runtime", "external_estop"):
        _PhoenixPolicyNode._latch_abort(stub, reason)
    assert len(stub.published) == 1
    assert decode(*stub.published[0], _LABEL).abort_reason == "attitude"


def _args(config, **over):
    base = dict(config=Path(config), onnx=None, lock=None, max_runtime_s=None, authority_s=None)
    base.update(over)
    return argparse.Namespace(**base)


def _fake_deploy(tmp_path, **safety_over):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    for name in ("policy.onnx", "policy.onnx.data", "policy.pt", "latest.pt"):
        (ckpt / name).write_bytes(name.encode())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"deploy": {"action_clip": 100.0, "kp": 25.0, "kd": 0.5}}))
    cfg = yaml.safe_load(H25.read_text())
    cfg["policy"]["onnx_path"] = str(ckpt / "policy.onnx")
    cfg["policy"]["torchscript_path"] = str(ckpt / "policy.pt")
    cfg["policy"]["manifest_path"] = str(manifest_path)
    cfg["safety"].update(safety_over)
    cfg_path = tmp_path / "deploy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg, cfg_path, ckpt


def _lock_for(cfg, cfg_path, tmp_path):
    observed = observed_artifact_hashes(cfg, cfg_path)
    lock = {
        "schema": "phoenix-deploy-lock/v1",
        "deploy_config": {"semantic_sha256": observed["deploy_config"]["semantic_sha256"]},
        "artifacts": {r: {"sha256": e["sha256"]} for r, e in observed["artifacts"].items()},
    }
    path = tmp_path / "lock.yaml"
    path.write_text(yaml.safe_dump(lock))
    return path


def test_resolve_startup_accepts_the_config_as_written(tmp_path) -> None:
    cfg, cfg_path, ckpt = _fake_deploy(tmp_path)
    onnx, problems = resolve_startup(cfg, _args(cfg_path))
    assert problems == [] and onnx == ckpt / "policy.onnx"


def test_resolve_startup_refuses_a_different_onnx(tmp_path) -> None:
    cfg, cfg_path, _ = _fake_deploy(tmp_path)
    _, problems = resolve_startup(cfg, _args(cfg_path, onnx=tmp_path / "other.onnx"))
    assert any("differs from the config" in p for p in problems)


def test_resolve_startup_refuses_a_walking_config(tmp_path) -> None:
    cfg, cfg_path, _ = _fake_deploy(tmp_path, stand_only=False)
    _, problems = resolve_startup(cfg, _args(cfg_path))
    assert any(p.startswith("deploy contract:") for p in problems)


def test_max_runtime_can_be_lowered(tmp_path) -> None:
    cfg, cfg_path, _ = _fake_deploy(tmp_path)
    _, problems = resolve_startup(cfg, _args(cfg_path, max_runtime_s=5.0))
    assert problems == [] and cfg["safety"]["max_runtime_s"] == 5.0


def test_max_runtime_can_never_be_raised(tmp_path) -> None:
    cfg, cfg_path, _ = _fake_deploy(tmp_path)
    _, problems = resolve_startup(cfg, _args(cfg_path, max_runtime_s=10_000.0))
    assert any("--max-runtime-s" in p for p in problems)
    assert cfg["safety"]["max_runtime_s"] == 120


def test_lock_is_checked_against_the_config_before_the_runtime_override(tmp_path) -> None:
    cfg, cfg_path, _ = _fake_deploy(tmp_path)
    lock = _lock_for(cfg, cfg_path, tmp_path)
    _, problems = resolve_startup(cfg, _args(cfg_path, lock=lock, max_runtime_s=2.0))
    assert problems == []


def test_lock_mismatch_refuses_startup(tmp_path) -> None:
    cfg, cfg_path, ckpt = _fake_deploy(tmp_path)
    lock = _lock_for(cfg, cfg_path, tmp_path)
    (ckpt / "policy.onnx.data").write_bytes(b"a different export")
    _, problems = resolve_startup(cfg, _args(cfg_path, lock=lock))
    assert any(p.startswith("lock:") and "policy.onnx.data" in p for p in problems)


def test_nonpositive_authority_window_is_refused(tmp_path) -> None:
    cfg, cfg_path, _ = _fake_deploy(tmp_path)
    _, problems = resolve_startup(cfg, _args(cfg_path, authority_s=0.0))
    assert any("--authority-s" in p for p in problems)


def test_resolve_startup_refuses_missing_manifest_path(tmp_path) -> None:
    cfg, cfg_path, _ = _fake_deploy(tmp_path)
    del cfg["policy"]["manifest_path"]
    cfg_path.write_text(yaml.safe_dump(cfg))
    _, problems = resolve_startup(cfg, _args(cfg_path))
    assert any("policy.manifest_path is required" in p for p in problems)


def test_resolve_startup_refuses_manifest_missing_action_clip(tmp_path) -> None:
    cfg, cfg_path, _ = _fake_deploy(tmp_path)
    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text(json.dumps({"deploy": {"kp": 25.0}}))
    cfg["policy"]["manifest_path"] = str(bad_manifest)
    cfg_path.write_text(yaml.safe_dump(cfg))
    _, problems = resolve_startup(cfg, _args(cfg_path))
    assert any("action_clip is missing" in p for p in problems)


# Note: _PhoenixPolicyNode.__init__ enforces the same manifest.action_clip
# requirement a second time (it needs rclpy + onnxruntime to construct, so
# it cannot run in this no-ROS suite; see the class docstring). The guard is
# the same load_deploy_manifest() call resolve_startup uses above, covered
# there; a full-construction regression test belongs in a rehearsal session.


# ---------------------------------------------------------------------------
# Action clamp (defect: unclamped raw action reached target-scaling math on
# 2026-09-21 stage F1) and velocity-command caps.
# ---------------------------------------------------------------------------


class _ActionClampStub:
    def __init__(self, action_clip: float = 1.0) -> None:
        self._action_clip = action_clip
        self._action_ticks = 0
        self._action_saturation_ticks = 0


def test_clip_action_and_report_clamps_and_counts_saturation() -> None:
    # action_clip=1.0: the legacy H25 contract, still a valid manifest value.
    stub = _ActionClampStub(action_clip=1.0)
    clipped = _PhoenixPolicyNode._clip_action_and_report(stub, np.asarray([-7.2, 0.1, 0.0]))
    assert np.allclose(clipped, [-1.0, 0.1, 0.0])
    assert stub._action_ticks == 1
    assert stub._action_saturation_ticks == 1


def test_clip_action_and_report_no_saturation_when_inside_range() -> None:
    stub = _ActionClampStub(action_clip=1.0)
    clipped = _PhoenixPolicyNode._clip_action_and_report(stub, np.asarray([0.2, -0.5, 0.9]))
    assert np.allclose(clipped, [0.2, -0.5, 0.9])
    assert stub._action_saturation_ticks == 0


def test_clip_action_and_report_clip_100_passes_historical_f1_magnitude_through() -> None:
    # 2026-09-24 contract: training moved from clip_actions=1.0 to 100.0. At
    # that clip, the historical F1 raw magnitude (-7.2) is NOT an outlier and
    # must pass through unmodified; the torque limit (a separate mechanism,
    # see test_clip_to_limits_bounds_by_torque_not_position_delta) is what
    # bounds the resulting target, not this clamp.
    stub = _ActionClampStub(action_clip=100.0)
    clipped = _PhoenixPolicyNode._clip_action_and_report(stub, np.asarray([-7.2, 0.1, 0.0]))
    assert np.allclose(clipped, [-7.2, 0.1, 0.0])
    assert stub._action_saturation_ticks == 0


def test_clip_action_and_report_clip_100_still_clamps_genuine_outliers() -> None:
    stub = _ActionClampStub(action_clip=100.0)
    clipped = _PhoenixPolicyNode._clip_action_and_report(stub, np.asarray([500.0, -0.1]))
    assert np.allclose(clipped, [100.0, -0.1])
    assert stub._action_saturation_ticks == 1


class _TorqueClipStub:
    def __init__(self, position_clip_margin: float = 0.01, sustained_clip_ticks: int = 10) -> None:
        from phoenix.sim2real.go2_model import limits_in_order, torque_limits_in_order
        from phoenix.sim2real.safety import SustainedClipTracker

        self._policy_kp = 25.0
        self._policy_kd = 0.5
        self._torque_limit = torque_limits_in_order(POLICY_JOINT_ORDER)
        self._lo, self._hi = limits_in_order(POLICY_JOINT_ORDER)
        self._position_clip_margin = position_clip_margin
        self._sustained_clip_ticks = sustained_clip_ticks
        self._clip_tracker = SustainedClipTracker(len(POLICY_JOINT_ORDER), sustained_clip_ticks)
        self._estopped = False
        self._abort_reason: str | None = None
        self.joint_order = JointOrder(POLICY_JOINT_ORDER)

    def _latch_abort(self, reason: str) -> None:
        self._estopped = True
        self._abort_reason = reason


def test_clip_to_limits_bounds_by_torque_not_position_delta() -> None:
    stub = _TorqueClipStub()
    q = np.asarray([TRAINING_DEFAULT_JOINT_POS[n] for n in POLICY_JOINT_ORDER], dtype=np.float32)
    qd = np.zeros(12, dtype=np.float32)
    target = q.copy()
    target[0] += 5.0  # absurd, must be clipped well below +5
    out = _PhoenixPolicyNode._clip_to_limits(stub, target, q, qd)
    assert out[0] < q[0] + 1.0
    assert stub._estopped is False


def test_clip_to_limits_clips_past_stop_target_without_aborting() -> None:
    """2026-09-25 sim2sim gate finding: a good walking policy requests calf
    targets past the calf's hard upper limit. A single such tick must be
    clipped, not treated as an abort condition."""
    from phoenix.sim2real.go2_model import JOINT_POSITION_LIMITS_RAD

    stub = _TorqueClipStub()
    j = POLICY_JOINT_ORDER.index("RR_calf_joint")
    lo, hi = JOINT_POSITION_LIMITS_RAD["RR_calf_joint"]
    q = np.asarray([TRAINING_DEFAULT_JOINT_POS[n] for n in POLICY_JOINT_ORDER], dtype=np.float32)
    qd = np.zeros(12, dtype=np.float32)
    target = q.copy()
    target[j] = hi + 0.28  # within the literal 0.25-0.31 rad sim2sim finding
    out = _PhoenixPolicyNode._clip_to_limits(stub, target, q, qd)
    assert lo <= out[j] <= hi
    assert stub._estopped is False


def test_clip_to_limits_sustained_clipping_latches_abort() -> None:
    from phoenix.sim2real.go2_model import JOINT_POSITION_LIMITS_RAD

    stub = _TorqueClipStub(sustained_clip_ticks=3)
    j = POLICY_JOINT_ORDER.index("RR_calf_joint")
    lo, hi = JOINT_POSITION_LIMITS_RAD["RR_calf_joint"]
    q = np.asarray([TRAINING_DEFAULT_JOINT_POS[n] for n in POLICY_JOINT_ORDER], dtype=np.float32)
    qd = np.zeros(12, dtype=np.float32)
    target = q.copy()
    target[j] = hi + 0.28
    for _ in range(2):
        _PhoenixPolicyNode._clip_to_limits(stub, target, q, qd)
        assert stub._estopped is False
    _PhoenixPolicyNode._clip_to_limits(stub, target, q, qd)
    assert stub._estopped is True
    assert stub._abort_reason.startswith("sustained_clip_exceeded:RR_calf_joint")


class _VelocityCapStub(_StubResolveNode):
    def __init__(self, max_command=(0.5, 0.3, 0.8)) -> None:
        super().__init__("odom", stand_only=False)
        self.max_command = np.asarray(max_command, dtype=np.float64)


def test_on_cmd_vel_clamps_to_configured_max_command() -> None:
    stub = _VelocityCapStub()
    _PhoenixPolicyNode._on_cmd_vel(stub, _twist(5.0, -5.0, 5.0))
    assert stub._estopped is False
    assert np.allclose(stub._velocity_command, [0.5, -0.3, 0.8])


def test_on_cmd_vel_passes_through_inside_cap() -> None:
    stub = _VelocityCapStub()
    _PhoenixPolicyNode._on_cmd_vel(stub, _twist(0.2, -0.1, 0.3))
    assert np.allclose(stub._velocity_command, [0.2, -0.1, 0.3])


def test_on_cmd_vel_nonfinite_becomes_zero_stand_not_propagated() -> None:
    stub = _VelocityCapStub()
    _PhoenixPolicyNode._on_cmd_vel(stub, _twist(float("nan"), 0.0, 0.0))
    assert stub._estopped is False
    assert np.allclose(stub._velocity_command, 0.0)
