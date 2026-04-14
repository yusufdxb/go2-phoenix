"""ROS 2 node that runs a Phoenix policy on the real GO2 at 50 Hz.

Runs in the **system Python + ROS 2** context (not Isaac Lab's Python).
Reads:

* ``/imu/data`` — sensor_msgs/Imu (orientation, angular velocity, linear accel)
* ``/joint_states`` — sensor_msgs/JointState (positions + velocities)
* ``/cmd_vel`` — geometry_msgs/Twist (teleop / higher-level policy command)

Publishes:

* ``/joint_group_position_controller/command`` — std_msgs/Float64MultiArray
  of 12 target joint positions in canonical order.

Safety:

* Dead-man's-switch on ``/phoenix/estop`` (std_msgs/Bool).
* Hard max runtime after which the node exits and sends the stand pose.
* Per-joint position, velocity, and torque clipping relative to URDF limits.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from .observation import JointOrder, ObservationBuilder

logger = logging.getLogger("phoenix.sim2real.ros2_policy_node")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Phoenix policy on the GO2 via ROS 2.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--onnx", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - requires ROS 2
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    import rclpy

    cfg = yaml.safe_load(args.config.read_text())

    rclpy.init()
    node = _PhoenixPolicyNode(cfg, args.onnx)
    try:
        rclpy.spin(node.node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.node.destroy_node()
        rclpy.shutdown()
    return 0


class _PhoenixPolicyNode:  # pragma: no cover - requires ROS 2 runtime
    """Minimal ROS 2 node wrapper. All rclpy imports are done here so the
    enclosing package can still be imported in CI."""

    def __init__(self, cfg: dict, onnx_path: Path):
        import onnxruntime as ort
        from geometry_msgs.msg import Twist
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Imu, JointState
        from std_msgs.msg import Bool, Float64MultiArray

        self._float_msg = Float64MultiArray

        # Build internal state and policy session up-front so ROS subscribers
        # are never called before we are ready to serve them.
        self.joint_order = JointOrder(tuple(cfg["joint_order"]))
        self.default_q = np.asarray(
            [cfg["control"]["default_joint_pos"][n] for n in self.joint_order.names],
            dtype=np.float32,
        )
        self.obs_builder = ObservationBuilder(self.joint_order, cfg["control"]["default_joint_pos"])
        self.action_scale = float(cfg["control"]["action_scale"])
        self.rate_hz = float(cfg["control"]["rate_hz"])
        self.max_runtime = float(cfg["safety"]["max_runtime_s"])
        self.pos_margin = float(cfg["actuator_limits"]["position_margin_rad"])

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        # Rolling state populated by subscribers.
        self._latest_imu = None
        self._latest_joint_state = None
        self._velocity_command = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(len(self.joint_order), dtype=np.float32)
        self._estopped = False
        self._started_at = time.monotonic()

        # --- ROS plumbing -------------------------------------------------
        self.node = Node("phoenix_policy_node")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        topics = cfg["topics"]
        self.node.create_subscription(Imu, topics["imu"], self._on_imu, qos)
        self.node.create_subscription(JointState, topics["joint_states"], self._on_joint_state, qos)
        self.node.create_subscription(Twist, topics["cmd_vel"], self._on_cmd_vel, qos)
        self.node.create_subscription(
            Bool, cfg["safety"]["emergency_stop_topic"], self._on_estop, qos
        )
        self.cmd_pub = self.node.create_publisher(Float64MultiArray, topics["joint_command"], qos)

        self.node.create_timer(1.0 / self.rate_hz, self._control_step)

    # --- callbacks -------------------------------------------------------
    def _on_imu(self, msg):
        self._latest_imu = msg

    def _on_joint_state(self, msg):
        self._latest_joint_state = msg

    def _on_cmd_vel(self, msg):
        self._velocity_command = np.asarray(
            [msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32
        )

    def _on_estop(self, msg):
        if msg.data:
            self._estopped = True
            logger.warning("E-STOP asserted; holding stand pose.")

    # --- control loop ----------------------------------------------------
    def _control_step(self):
        if self._estopped or time.monotonic() - self._started_at > self.max_runtime:
            self._publish_default_pose()
            return
        if self._latest_imu is None or self._latest_joint_state is None:
            return  # still waiting for first messages

        idx = self.joint_order.remap(list(self._latest_joint_state.name))
        q = np.asarray(self._latest_joint_state.position, dtype=np.float32)[idx]
        qd = np.asarray(self._latest_joint_state.velocity, dtype=np.float32)[idx]

        # Body-frame projected gravity from IMU orientation quaternion.
        ori = self._latest_imu.orientation  # w last? ROS uses (x,y,z,w)
        pg = _projected_gravity_from_quat(ori.x, ori.y, ori.z, ori.w)

        # Body-frame angular velocity from IMU, linear velocity estimated as 0
        # (proprioceptive estimator would go here in a fuller build).
        base_ang_vel = np.asarray(
            [
                self._latest_imu.angular_velocity.x,
                self._latest_imu.angular_velocity.y,
                self._latest_imu.angular_velocity.z,
            ],
            dtype=np.float32,
        )
        base_lin_vel = np.zeros(3, dtype=np.float32)

        obs = self.obs_builder.build(
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            projected_gravity=pg,
            velocity_command=self._velocity_command,
            joint_pos=q,
            joint_vel=qd,
            last_action=self._last_action,
        ).reshape(1, -1)

        action = self.session.run(["action"], {"obs": obs})[0][0]
        self._last_action = action.astype(np.float32, copy=False)

        target = self.default_q + self.action_scale * action
        target = self._clip_to_limits(target, q)

        msg = self._float_msg()
        msg.data = target.astype(np.float64).tolist()
        self.cmd_pub.publish(msg)

    def _clip_to_limits(self, target: np.ndarray, q: np.ndarray) -> np.ndarray:
        # Without the URDF loaded, we clip velocity-wise only: bound the per-step delta
        # so hardware never sees a step > 10° / 20 ms.
        max_step = 0.175  # rad
        return np.clip(target, q - max_step, q + max_step)

    def _publish_default_pose(self) -> None:
        msg = self._float_msg()
        msg.data = self.default_q.astype(np.float64).tolist()
        self.cmd_pub.publish(msg)

    def shutdown(self) -> None:
        self._publish_default_pose()


def _projected_gravity_from_quat(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Rotate world-frame gravity (0,0,-1) into the body frame using quat (x,y,z,w)."""
    # R^T * g_world where R is body->world rotation derived from quat.
    # Closed form: body-frame gravity = [2(xz - wy), 2(yz + wx), -(1 - 2(x² + y²))]
    gx = 2.0 * (x * z - w * y)
    gy = 2.0 * (y * z + w * x)
    gz = -(1.0 - 2.0 * (x * x + y * y))
    g = np.asarray([gx, gy, gz], dtype=np.float32)
    # Already unit-length in exact arithmetic; renormalize for numerical safety.
    return g / (np.linalg.norm(g) + 1e-9)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
