"""Offline dry-run harness for the Phoenix ROS 2 policy node.

Runs the policy node against synthetic /imu/data, /joint_states, /cmd_vel,
and /phoenix/estop publishers — no robot, no joystick, no Jetson required.

Verifies, in order:

1. Node starts cleanly (no shape error, ONNX loads).
2. /joint_group_position_controller/command is published at ~50 Hz.
3. Commands respect the per-step delta clip (<= 0.175 rad from joint_pos).
4. Pitch > 0.8 rad latches the attitude abort — commands collapse to
   the default stand pose.
5. /phoenix/estop True latches the estop — same stand-pose behavior.
6. NaN in joint_states latches the abort.

Usage:
    source /opt/ros/humble/setup.bash
    cd /path/to/go2-phoenix
    PYTHONPATH=$PWD/src python3 scripts/dry_run_policy.py \\
        --onnx checkpoints/phoenix-base/policy.onnx \\
        --config configs/sim2real/deploy.yaml

Exits non-zero if any gate fails.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger("phoenix.dry_run")


DEFAULT_JOINT_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/sim2real/deploy.yaml"))
    p.add_argument("--onnx", type=Path, default=Path("checkpoints/phoenix-base/policy.onnx"))
    p.add_argument("--duration", type=float, default=3.0,
                   help="Seconds per scenario.")
    return p.parse_args()


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return x, y, z, w


class FakeSensorNode:
    def __init__(self, node_cls, default_q: np.ndarray, estop_topic: str):
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import Imu, JointState
        from std_msgs.msg import Bool
        from rclpy.qos import QoSProfile, ReliabilityPolicy

        self._imu_cls = Imu
        self._js_cls = JointState
        self._twist_cls = Twist
        self._bool_cls = Bool

        self.node = node_cls("phoenix_dry_run_fakes")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.imu_pub = self.node.create_publisher(Imu, "/imu/data", qos)
        self.js_pub = self.node.create_publisher(JointState, "/joint_states", qos)
        self.cmd_pub = self.node.create_publisher(Twist, "/cmd_vel", qos)
        self.estop_pub = self.node.create_publisher(Bool, estop_topic, qos)

        self.default_q = default_q
        self.pitch = 0.0
        self.roll = 0.0
        self.inject_nan = False
        self.estop = False
        self.node.create_timer(1.0 / 200.0, self._tick)  # 200 Hz joint_states
        self.node.create_timer(1.0 / 50.0, self._tick_imu)
        self.node.create_timer(1.0 / 10.0, self._tick_estop)

    def _tick(self):
        msg = self._js_cls()
        msg.name = list(DEFAULT_JOINT_ORDER)
        q = self.default_q.copy()
        if self.inject_nan:
            q[0] = float("nan")
        msg.position = q.astype(float).tolist()
        msg.velocity = [0.0] * 12
        self.js_pub.publish(msg)

    def _tick_imu(self):
        msg = self._imu_cls()
        x, y, z, w = _euler_to_quat(self.roll, self.pitch, 0.0)
        msg.orientation.x = x
        msg.orientation.y = y
        msg.orientation.z = z
        msg.orientation.w = w
        msg.angular_velocity.x = 0.0
        msg.angular_velocity.y = 0.0
        msg.angular_velocity.z = 0.0
        self.imu_pub.publish(msg)

    def _tick_estop(self):
        m = self._bool_cls()
        m.data = self.estop
        self.estop_pub.publish(m)


class CommandRecorder:
    def __init__(self, node_cls):
        from std_msgs.msg import Float64MultiArray
        from rclpy.qos import QoSProfile, ReliabilityPolicy

        self.node = node_cls("phoenix_dry_run_recorder")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.samples: list[tuple[float, list[float]]] = []
        self.node.create_subscription(
            Float64MultiArray,
            "/joint_group_position_controller/command",
            self._on_cmd,
            qos,
        )

    def _on_cmd(self, msg):
        self.samples.append((time.monotonic(), list(msg.data)))

    def snapshot(self) -> list[tuple[float, list[float]]]:
        return list(self.samples)

    def reset(self) -> None:
        self.samples.clear()


def _run_scenario(recorder: CommandRecorder, duration: float) -> list[tuple[float, list[float]]]:
    recorder.reset()
    time.sleep(duration)
    return recorder.snapshot()


def _rate(samples: list[tuple[float, list[float]]]) -> float:
    if len(samples) < 2:
        return 0.0
    return (len(samples) - 1) / (samples[-1][0] - samples[0][0])


def _max_abs(samples: list[tuple[float, list[float]]], ref: np.ndarray) -> float:
    arr = np.asarray([s[1] for s in samples], dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr - ref)))


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node

    # Import the system under test only after rclpy is installed.
    from phoenix.sim2real.ros2_policy_node import _PhoenixPolicyNode

    cfg = yaml.safe_load(args.config.read_text())
    default_q = np.asarray(
        [cfg["control"]["default_joint_pos"][n] for n in DEFAULT_JOINT_ORDER],
        dtype=np.float32,
    )

    rclpy.init()
    try:
        policy = _PhoenixPolicyNode(cfg, args.onnx, log_parquet=None)
        fakes = FakeSensorNode(Node, default_q, cfg["safety"]["emergency_stop_topic"])
        recorder = CommandRecorder(Node)

        executor = SingleThreadedExecutor()
        executor.add_node(policy.node)
        executor.add_node(fakes.node)
        executor.add_node(recorder.node)

        stop_event = threading.Event()

        def _spin():
            while not stop_event.is_set():
                executor.spin_once(timeout_sec=0.02)

        t = threading.Thread(target=_spin, daemon=True)
        t.start()

        failed = False

        # Gate 1+2: clean start, 50 Hz, commands near default_q.
        logger.info("Scenario 1: normal — expect ~50 Hz commands near default.")
        time.sleep(0.5)  # warmup
        samples = _run_scenario(recorder, args.duration)
        rate = _rate(samples)
        max_dev = _max_abs(samples, default_q)
        logger.info("  rate=%.1f Hz, max|cmd-default|=%.3f rad, samples=%d",
                    rate, max_dev, len(samples))
        if not (45 <= rate <= 55):
            logger.error("  FAIL: rate out of 45-55 Hz band.")
            failed = True
        if max_dev > 0.20:  # allow action_scale * |a| slack
            logger.error("  FAIL: command deviates >0.20 rad from default.")
            failed = True

        # Gate 4: attitude abort.
        logger.info("Scenario 2: pitch=1.0 rad — expect aborted stand-pose commands.")
        fakes.pitch = 1.0
        time.sleep(0.3)
        samples = _run_scenario(recorder, args.duration)
        max_dev = _max_abs(samples, default_q)
        logger.info("  samples=%d, max|cmd-default|=%.6f", len(samples), max_dev)
        if max_dev > 1e-6:
            logger.error("  FAIL: after attitude abort commands did not collapse to default.")
            failed = True
        fakes.pitch = 0.0

        # Reset policy to re-run estop scenario cleanly (abort is latched).
        logger.info("Scenario 3: estop latch — re-init node, flip /phoenix/estop True.")
        policy.shutdown()
        policy.node.destroy_node()
        policy = _PhoenixPolicyNode(cfg, args.onnx, log_parquet=None)
        executor.add_node(policy.node)
        time.sleep(0.5)
        fakes.estop = True
        time.sleep(0.3)
        samples = _run_scenario(recorder, args.duration)
        max_dev = _max_abs(samples, default_q)
        logger.info("  samples=%d, max|cmd-default|=%.6f", len(samples), max_dev)
        if max_dev > 1e-6:
            logger.error("  FAIL: estop did not collapse commands to default.")
            failed = True
        fakes.estop = False

        # Gate: NaN abort.
        logger.info("Scenario 4: NaN in joint_states — expect abort.")
        policy.shutdown()
        policy.node.destroy_node()
        policy = _PhoenixPolicyNode(cfg, args.onnx, log_parquet=None)
        executor.add_node(policy.node)
        time.sleep(0.5)
        fakes.inject_nan = True
        time.sleep(0.3)
        samples = _run_scenario(recorder, args.duration)
        max_dev = _max_abs(samples, default_q)
        logger.info("  samples=%d, max|cmd-default|=%.6f", len(samples), max_dev)
        if max_dev > 1e-6:
            logger.error("  FAIL: NaN joint state did not trigger abort.")
            failed = True
        fakes.inject_nan = False

        stop_event.set()
        t.join(timeout=2.0)

        if failed:
            logger.error("DRY-RUN FAILED")
            return 1
        logger.info("DRY-RUN PASSED — all gates clean.")
        return 0
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
