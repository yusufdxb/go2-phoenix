"""Pure parts of the motors-off ROS probe: timing, content statistics, no ROS import at load."""

from __future__ import annotations

import ast
import math
import types
from pathlib import Path

import numpy as np
import pytest

from phoenix.sim2real import hw_probe

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_module_does_not_import_ros_at_load_time() -> None:
    tree = ast.parse((REPO_ROOT / "src/phoenix/sim2real/hw_probe.py").read_text())
    top_level = [
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    modules = [
        getattr(node, "module", None) for node in tree.body if isinstance(node, ast.ImportFrom)
    ]
    assert not any("rclpy" in (m or "") for m in modules + top_level)


def test_timing_stats() -> None:
    assert hw_probe.timing_stats([]) == {"count": 0, "rate_hz": None, "max_gap_s": None}
    assert hw_probe.timing_stats([1.0])["rate_hz"] is None
    times = [i * 0.002 for i in range(501)]
    times[300] += 0.0005
    stats = hw_probe.timing_stats(times)
    assert stats["count"] == 501
    assert stats["rate_hz"] == pytest.approx(500.0)
    assert stats["max_gap_s"] == pytest.approx(0.0025)


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.002
        return self.t


def _lowstate(q):
    return types.SimpleNamespace(
        motor_state=[types.SimpleNamespace(q=v, dq=0.0) for v in q]
        + [types.SimpleNamespace(q=0.0, dq=0.0)] * 8
    )


def test_lowstate_range_and_non_finite() -> None:
    rec = hw_probe.TopicRecorder("/lowstate", clock=_Clock())
    rec.on_message(_lowstate([0.1] * 12))
    rec.on_message(_lowstate([0.3] * 12))
    bad = [0.2] * 12
    bad[4] = float("nan")
    rec.on_message(_lowstate(bad))
    report = rec.report()
    assert report["count"] == 3
    assert report["q_min"] == pytest.approx([0.1] * 12)
    assert report["q_max"] == pytest.approx([0.3] * 12)
    assert report["non_finite"] == 1


def _imu(x, y, z, w):
    return types.SimpleNamespace(
        orientation=types.SimpleNamespace(x=x, y=y, z=z, w=w),
        angular_velocity=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
    )


def test_imu_attitude_extremes() -> None:
    rec = hw_probe.TopicRecorder("/imu/data", clock=_Clock())
    rec.on_message(_imu(0.0, 0.0, 0.0, 1.0))
    theta = 0.4
    rec.on_message(_imu(math.sin(theta / 2), 0.0, 0.0, math.cos(theta / 2)))
    rec.on_message(_imu(float("nan"), 0.0, 0.0, 1.0))
    report = rec.report()
    assert report["max_abs_roll_rad"] == pytest.approx(theta)
    assert report["max_abs_pitch_rad"] == pytest.approx(0.0, abs=1e-12)
    assert report["non_finite"] == 1


def test_joint_state_names_and_command_labels_and_estop_events() -> None:
    js = hw_probe.TopicRecorder("/joint_states", clock=_Clock())
    js.on_message(types.SimpleNamespace(name=["b", "a"]))
    js.on_message(types.SimpleNamespace(name=["c"]))
    assert js.report()["names"] == ["a", "b", "c"]

    cmd = hw_probe.TopicRecorder(hw_probe.COMMAND_TOPIC, clock=_Clock())
    dim = types.SimpleNamespace(label="phoenix_cmd/v2;len=72;order=x")
    cmd.on_message(types.SimpleNamespace(layout=types.SimpleNamespace(dim=[dim]), data=[0.0] * 72))
    cmd.on_message(types.SimpleNamespace(layout=types.SimpleNamespace(dim=[]), data=[0.0] * 12))
    report = cmd.report()
    assert report["labels"] == ["", "phoenix_cmd/v2;len=72;order=x"]
    assert report["lengths"] == [12, 72]

    estop = hw_probe.TopicRecorder("/phoenix/estop", clock=_Clock())
    estop.on_message(types.SimpleNamespace(data=True))
    estop.on_message(types.SimpleNamespace(data=False))
    assert [e["value"] for e in estop.events] == [True, False]
    assert estop.report()["values_seen"] == [False, True]


def test_roll_pitch_identity_and_pure_pitch() -> None:
    assert hw_probe.roll_pitch_from_quat_xyzw(0.0, 0.0, 0.0, 1.0) == pytest.approx((0.0, 0.0))
    theta = 0.3
    roll, pitch = hw_probe.roll_pitch_from_quat_xyzw(
        0.0, math.sin(theta / 2), 0.0, math.cos(theta / 2)
    )
    assert roll == pytest.approx(0.0, abs=1e-12) and pitch == pytest.approx(theta)


def test_parser_requires_outputs_and_timeout() -> None:
    parser = hw_probe.build_parser()
    args = parser.parse_args(["rates", "--out", "x.json", "--duration", "3"])
    assert args.duration == 3.0 and "/lowstate" in args.topics
    with pytest.raises(SystemExit):
        parser.parse_args(["deadman", "--out", "x.json"])  # --estop-timeout-s required
    assert np.isclose(
        parser.parse_args(["deadman", "--out", "t", "--estop-timeout-s", "0.5"]).phase_s, 3.0
    )
