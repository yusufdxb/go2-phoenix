"""Tests for the lowcmd bridge config builder + estop-timeout plumbing.

The ROS-side pieces of ``lowcmd_bridge_node`` need rclpy + the
``unitree_go`` messages, neither of which is in CI. The pieces that
*can* be tested without ROS are:

* ``_build_config`` — does it pick up topic + rate overrides from the
  deploy YAML, and does it carry the estop timeout through?
* The ``BridgeConfig`` dataclass defaults — the audit cares specifically
  about the new ``estop_timeout_s`` field actually being present.

Importing the bridge node module would normally pull in rclpy as a
side-effect; we monkey-patch ``sys.modules`` so the module-level
imports succeed in a CI environment that lacks ROS.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def bridge_module(monkeypatch):
    # Stub the rclpy + ROS-message namespaces required at import time so
    # the rest of the module loads cleanly. None of the stubbed objects
    # are exercised by the tests in this file.
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *a, **kw: None
    rclpy.shutdown = lambda *a, **kw: None
    rclpy.spin = lambda *a, **kw: None
    rclpy.ok = lambda: True

    rclpy_node = types.ModuleType("rclpy.node")

    class _Node:
        def __init__(self, name: str) -> None:
            self.name = name

        def create_subscription(self, *a, **kw):
            return None

        def create_publisher(self, *a, **kw):
            return None

        def create_timer(self, *a, **kw):
            return None

        def get_logger(self):
            class _L:
                def info(self, *a, **kw):
                    pass

                def warn(self, *a, **kw):
                    pass

            return _L()

        def get_clock(self):
            class _C:
                @property
                def now(self):
                    class _N:
                        nanoseconds = 0

                    return _N()

            return _C()

        def destroy_node(self):
            pass

    rclpy_node.Node = _Node

    rclpy_qos = types.ModuleType("rclpy.qos")

    class _QoS:
        def __init__(self, *a, **kw):
            pass

    class _Reliability:
        BEST_EFFORT = 1
        RELIABLE = 2

    class _History:
        KEEP_LAST = 1

    rclpy_qos.QoSProfile = _QoS
    rclpy_qos.ReliabilityPolicy = _Reliability
    rclpy_qos.HistoryPolicy = _History

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs.msg = std_msgs_msg

    class _Bool:
        data = False

    class _Float64MultiArray:
        data: list = []

    std_msgs_msg.Bool = _Bool
    std_msgs_msg.Float64MultiArray = _Float64MultiArray

    unitree = types.ModuleType("unitree_go")
    unitree_msg = types.ModuleType("unitree_go.msg")
    unitree.msg = unitree_msg

    class _LowState:
        motor_state: list = []

    class _MotorCmd:
        mode = 0
        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 0.0
        kd = 0.0

    class _LowCmd:
        head = [0, 0]
        level_flag = 0
        motor_cmd = [_MotorCmd() for _ in range(20)]
        crc = 0

    unitree_msg.LowState = _LowState
    unitree_msg.LowCmd = _LowCmd

    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.node", rclpy_node)
    monkeypatch.setitem(sys.modules, "rclpy.qos", rclpy_qos)
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msgs_msg)
    monkeypatch.setitem(sys.modules, "unitree_go", unitree)
    monkeypatch.setitem(sys.modules, "unitree_go.msg", unitree_msg)

    if "phoenix.sim2real.lowcmd_bridge_node" in sys.modules:
        importlib.reload(sys.modules["phoenix.sim2real.lowcmd_bridge_node"])
    return importlib.import_module("phoenix.sim2real.lowcmd_bridge_node")


def test_bridge_config_has_default_estop_timeout(bridge_module) -> None:
    cfg = bridge_module.BridgeConfig(
        rate_hz=50.0,
        watchdog_s=0.2,
        kp=25.0,
        kd=0.5,
        hold_kp=20.0,
        hold_kd=1.0,
        live=False,
        dry_topic="/lowcmd_dry",
        live_topic="/lowcmd",
        cmd_topic="/cmd",
        lowstate_topic="/lowstate",
        estop_topic="/phoenix/estop",
    )
    assert cfg.estop_timeout_s == 0.5  # documented default


def test_build_config_carries_cli_estop_timeout(bridge_module) -> None:
    args = argparse.Namespace(
        config=Path("configs/sim2real/deploy.yaml"),
        live=False,
        kp=25.0,
        kd=0.5,
        hold_kp=20.0,
        hold_kd=1.0,
        watchdog_s=0.2,
        estop_timeout_s=0.7,
    )
    cfg = bridge_module._build_config(args)
    assert cfg.estop_timeout_s == 0.7


def test_build_config_picks_up_topic_overrides(bridge_module, tmp_path) -> None:
    deploy_yaml = tmp_path / "deploy.yaml"
    deploy_yaml.write_text(
        "control:\n  rate_hz: 100\n"
        "topics:\n  joint_command: /custom/command\n"
        "safety:\n  emergency_stop_topic: /custom/estop\n"
    )
    args = argparse.Namespace(
        config=deploy_yaml,
        live=True,
        kp=30.0,
        kd=1.0,
        hold_kp=22.0,
        hold_kd=1.5,
        watchdog_s=0.1,
        estop_timeout_s=0.4,
    )
    cfg = bridge_module._build_config(args)
    assert cfg.rate_hz == 100.0
    assert cfg.cmd_topic == "/custom/command"
    assert cfg.estop_topic == "/custom/estop"
    assert cfg.live is True


def test_build_config_falls_back_to_defaults_on_missing_yaml(bridge_module, tmp_path) -> None:
    args = argparse.Namespace(
        config=tmp_path / "absent.yaml",
        live=False,
        kp=25.0,
        kd=0.5,
        hold_kp=20.0,
        hold_kd=1.0,
        watchdog_s=0.2,
        estop_timeout_s=None,
    )
    cfg = bridge_module._build_config(args)
    assert cfg.cmd_topic == "/joint_group_position_controller/command"
    assert cfg.estop_topic == "/phoenix/estop"
    assert cfg.lowstate_topic == "/lowstate"
    assert cfg.rate_hz == 50.0
    # No CLI override, no YAML — must fall back to the documented 0.5 s.
    assert cfg.estop_timeout_s == 0.5


def test_build_config_reads_estop_timeout_from_yaml(bridge_module, tmp_path) -> None:
    # The original implementation ignored safety.estop_timeout_s entirely,
    # silently leaving the bridge on the CLI default. Audit-driven fix.
    deploy_yaml = tmp_path / "deploy.yaml"
    deploy_yaml.write_text(
        "safety:\n" "  emergency_stop_topic: /phoenix/estop\n" "  estop_timeout_s: 0.3\n"
    )
    args = argparse.Namespace(
        config=deploy_yaml,
        live=False,
        kp=25.0,
        kd=0.5,
        hold_kp=20.0,
        hold_kd=1.0,
        watchdog_s=0.2,
        estop_timeout_s=None,  # CLI default → defer to YAML
    )
    cfg = bridge_module._build_config(args)
    assert cfg.estop_timeout_s == 0.3


def test_cli_estop_timeout_overrides_yaml(bridge_module, tmp_path) -> None:
    deploy_yaml = tmp_path / "deploy.yaml"
    deploy_yaml.write_text("safety:\n  estop_timeout_s: 0.3\n")
    args = argparse.Namespace(
        config=deploy_yaml,
        live=False,
        kp=25.0,
        kd=0.5,
        hold_kp=20.0,
        hold_kd=1.0,
        watchdog_s=0.2,
        estop_timeout_s=0.7,  # explicit CLI must win over YAML
    )
    cfg = bridge_module._build_config(args)
    assert cfg.estop_timeout_s == 0.7


def test_shipped_deploy_yaml_estop_timeout_is_loaded() -> None:
    # End-to-end check against the actually-shipped configs/sim2real/deploy.yaml.
    # If someone removes safety.estop_timeout_s from the file, this test
    # surfaces it immediately — keeps the docs honest.
    from pathlib import Path as _Path

    import yaml as _yaml

    cfg_path = _Path("configs/sim2real/deploy.yaml")
    if not cfg_path.exists():  # pragma: no cover - only when run from elsewhere
        pytest.skip("deploy.yaml not at expected path")
    cfg = _yaml.safe_load(cfg_path.read_text())
    assert "estop_timeout_s" in cfg.get(
        "safety", {}
    ), "configs/sim2real/deploy.yaml must declare safety.estop_timeout_s"


# ---------------------------------------------------------------------------
# Hardware-readiness pass: the bridge is a shell around ActuatorGate.
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent.parent


def test_lowcmd_fields_crc_matches_the_reference_packing(bridge_module) -> None:
    from phoenix.sim2real.motor_crc import build_raw_from_motor_values, compute_crc

    target = [0.1 * i for i in range(12)]
    q, crc = bridge_module.lowcmd_fields(target, 20.0, 1.0)
    assert q == target
    assert crc == compute_crc(build_raw_from_motor_values(target, [20.0] * 12, [1.0] * 12))
    _, crc_damp = bridge_module.lowcmd_fields(target, 0.0, 1.0)
    assert crc_damp != crc


def _ns(config, **over):
    base = dict(
        config=config,
        live=False,
        kp=25.0,
        kd=0.5,
        hold_kp=20.0,
        hold_kd=1.0,
        watchdog_s=0.2,
        estop_timeout_s=None,
        stale_hold_s=None,
        lock=None,
        telemetry=None,
        expect_sha=None,
        stage="test",
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_build_config_takes_freshness_and_order_from_the_deploy_config(bridge_module) -> None:
    cfg = bridge_module._build_config(_ns(_REPO / "configs/sim2real/deploy_stand_h25.yaml"))
    assert cfg.lowstate_timeout_s == 0.2  # safety.sensor_timeout_s
    assert cfg.first_message_timeout_s == 15.0
    assert cfg.stale_hold_s == cfg.watchdog_s
    assert cfg.joint_order[0] == "FL_hip_joint"
    params = cfg.gate_params()
    assert params.deadman_required is False


def test_live_bridge_refuses_without_lock_telemetry_and_expected_sha(bridge_module) -> None:
    cfg = bridge_module._build_config(
        _ns(_REPO / "configs/sim2real/deploy_stand_h25.yaml", live=True)
    )
    assert cfg.gate_params().deadman_required is True
    problems, manifest = bridge_module.startup_problems(cfg)
    joined = " | ".join(problems)
    assert "--live requires --lock" in joined
    assert "--live requires --expect-sha" in joined
    assert "--live requires --telemetry" in joined
    assert manifest["live"] is True and manifest["startup_problems"] == problems


def test_live_bridge_refuses_a_missing_config(bridge_module, tmp_path) -> None:
    cfg = bridge_module._build_config(_ns(tmp_path / "absent.yaml", live=True))
    problems, _ = bridge_module.startup_problems(cfg)
    assert any("requires an existing --config" in p for p in problems)


def test_bridge_refuses_a_lock_that_does_not_match(bridge_module, tmp_path) -> None:
    import yaml as _yaml

    lock = tmp_path / "lock.yaml"
    lock.write_text(
        _yaml.safe_dump(
            {
                "schema": "phoenix-deploy-lock/v1",
                "deploy_config": {"semantic_sha256": "0" * 64},
                "artifacts": {"policy.onnx": {"sha256": "1" * 64}},
            }
        )
    )
    cfg = bridge_module._build_config(
        _ns(_REPO / "configs/sim2real/deploy_stand_h25.yaml", lock=lock)
    )
    problems, _ = bridge_module.startup_problems(cfg)
    assert any(p.startswith("lock:") and "semantic sha256" in p for p in problems)


def test_manifest_records_limits_orders_and_metric(bridge_module) -> None:
    cfg = bridge_module._build_config(_ns(_REPO / "configs/sim2real/deploy_stand_h25.yaml"))
    _, manifest = bridge_module.startup_problems(cfg)
    assert manifest["joint_limits_rad"]["RL_thigh_joint"] == [-0.5236, 4.5379]
    assert manifest["motor_order_unitree"][0] == "FR_hip_joint"
    assert manifest["hardware_slew_metric"] == "final_target_vs_policy_request_clip_activation_v1"
    assert manifest["code_identity"]["source"] in ("git", "payload_sync", "unknown")


def test_bridge_times_freshness_with_the_monotonic_clock() -> None:
    """The ROS wall clock jumps when the payload clock is set by hand mid-session."""
    import ast

    src = (_REPO / "src/phoenix/sim2real/lowcmd_bridge_node.py").read_text()
    tree = ast.parse(src)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "get_clock" not in calls
    assert "monotonic_ns" in calls
