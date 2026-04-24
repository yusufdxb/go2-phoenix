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
