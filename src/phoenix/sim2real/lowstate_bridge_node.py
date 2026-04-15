"""Unitree /lowstate → standard ROS topics republisher.

The Unitree SDK publishes everything the Phoenix policy needs to see as
``unitree_go/msg/LowState`` on ``/lowstate``. The policy's ObservationBuilder
consumes ``sensor_msgs/JointState`` on ``/joint_states`` and
``sensor_msgs/Imu`` on ``/imu/data`` (deploy.yaml names). This node is the
~50-line translator.

Design notes:

* **Name-indexed, not positional.** Phoenix's ObservationBuilder looks up
  joints by name (see ``observation.py:JointOrder``), so we populate both
  ``msg.name`` and ``msg.position`` in any consistent order; Phoenix
  reorders as needed. We emit in the native Unitree motor order
  (FR, FL, RR, RL × hip/thigh/calf).
* **Quaternion convention.** Unitree ``imu_state.quaternion`` is
  ``[w, x, y, z]`` (confirmed against ``read_low_state.cpp``). We fan it
  into ``sensor_msgs/Imu.orientation`` which is ``(x, y, z, w)``.
* **Gyro / accel frames.** The raw IMU is body-frame. Phoenix's
  observation_builder consumes them as such. No rotation applied here.
"""

from __future__ import annotations

import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState
from unitree_go.msg import LowState


# Unitree motor indices 0..11 map to these joint names. Matches the
# convention used in the repo's URDF and ``deploy.yaml:joint_order``.
MOTOR_NAMES: tuple[str, ...] = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)


class LowStateBridge(Node):
    def __init__(self) -> None:
        super().__init__("phoenix_lowstate_bridge")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._js_pub = self.create_publisher(JointState, "/joint_states", qos)
        self._imu_pub = self.create_publisher(Imu, "/imu/data", qos)
        self._sub = self.create_subscription(LowState, "/lowstate", self._on_state, qos)

        self.get_logger().info(
            "lowstate bridge up: /lowstate → /joint_states + /imu/data"
        )

    def _on_state(self, msg: LowState) -> None:
        now = self.get_clock().now().to_msg()

        js = JointState()
        js.header.stamp = now
        js.header.frame_id = ""
        js.name = list(MOTOR_NAMES)
        js.position = [float(msg.motor_state[i].q) for i in range(12)]
        js.velocity = [float(msg.motor_state[i].dq) for i in range(12)]
        js.effort = [float(msg.motor_state[i].tau_est) for i in range(12)]
        self._js_pub.publish(js)

        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = "imu_link"
        # Unitree: [w, x, y, z]  →  ROS: (x, y, z, w)
        imu.orientation.w = float(msg.imu_state.quaternion[0])
        imu.orientation.x = float(msg.imu_state.quaternion[1])
        imu.orientation.y = float(msg.imu_state.quaternion[2])
        imu.orientation.z = float(msg.imu_state.quaternion[3])
        imu.angular_velocity.x = float(msg.imu_state.gyroscope[0])
        imu.angular_velocity.y = float(msg.imu_state.gyroscope[1])
        imu.angular_velocity.z = float(msg.imu_state.gyroscope[2])
        imu.linear_acceleration.x = float(msg.imu_state.accelerometer[0])
        imu.linear_acceleration.y = float(msg.imu_state.accelerometer[1])
        imu.linear_acceleration.z = float(msg.imu_state.accelerometer[2])
        self._imu_pub.publish(imu)


def main(argv: list[str] | None = None) -> int:
    rclpy.init()
    node = LowStateBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
