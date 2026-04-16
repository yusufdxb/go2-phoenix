"""Unitree wireless controller → /phoenix/estop adapter.

The Unitree wireless controller that ships with the GO2 pairs to the
robot's onboard Bluetooth, not the Jetson. Its button states reach ROS
on ``/wirelesscontroller`` (``unitree_go/msg/WirelessController``),
NOT the standard ``/joy``. So this node is the GO2-controller-flavoured
equivalent of ``deadman_joy_node`` — same external behaviour, different
input source.

Unitree wireless ``keys`` field is a uint16 bitmask. Standard mapping
on GO2 firmware:

  bit 0  R1       bit 8  A
  bit 1  L1       bit 9  B
  bit 2  Start    bit 10 X
  bit 3  Select   bit 11 Y
  bit 4  R2       bit 12 Up
  bit 5  L2       bit 13 Right
  bit 6  F1       bit 14 Down
  bit 7  F2       bit 15 Left

Default deadman: L1 (bit 1, mask 0x02). Hold to keep estop ``False``;
release or stop hearing /wirelesscontroller messages and estop flips
``True`` within one publish period.

Usage::

    python3 -m phoenix.sim2real.wireless_estop_node
    python3 -m phoenix.sim2real.wireless_estop_node --button-mask 0x01  # use R1 instead
"""

from __future__ import annotations

import argparse
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from unitree_go.msg import WirelessController

from phoenix.sim2real.safety import deadman_should_estop


class WirelessEstopNode(Node):
    def __init__(
        self,
        button_mask: int,
        joy_timeout_s: float,
        publish_rate_hz: float,
        estop_topic: str,
    ) -> None:
        super().__init__("phoenix_wireless_estop")
        self._button_mask = button_mask
        self._joy_timeout_s = joy_timeout_s

        self._last_msg_time_ns: int | None = None
        self._button_held: bool = False

        # /wirelesscontroller is published BEST_EFFORT by the Unitree SDK; match it.
        qos_in = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # /phoenix/estop subscribers (policy + lowcmd_bridge) are BEST_EFFORT.
        qos_out = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._sub = self.create_subscription(
            WirelessController, "/wirelesscontroller", self._on_msg, qos_in
        )
        self._pub = self.create_publisher(Bool, estop_topic, qos_out)
        self._timer = self.create_timer(1.0 / publish_rate_hz, self._tick)

        self.get_logger().info(
            f"wireless_estop up: hold mask=0x{button_mask:04x} on /wirelesscontroller "
            f"to keep {estop_topic}=False; timeout={joy_timeout_s}s"
        )

    def _on_msg(self, msg: WirelessController) -> None:
        self._last_msg_time_ns = self.get_clock().now().nanoseconds
        self._button_held = bool(int(msg.keys) & self._button_mask)

    def _tick(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        estopped = deadman_should_estop(
            last_input_ns=self._last_msg_time_ns,
            button_held=self._button_held,
            now_ns=now_ns,
            timeout_s=self._joy_timeout_s,
        )
        msg = Bool()
        msg.data = bool(estopped)
        self._pub.publish(msg)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--button-mask",
        type=lambda s: int(s, 0),
        default=0x02,
        help="WirelessController.keys bitmask of the deadman button (default 0x02 = L1)",
    )
    p.add_argument(
        "--joy-timeout-s",
        type=float,
        default=0.5,
        help="seconds without /wirelesscontroller before forcing estop True (default 0.5)",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="publish rate on /phoenix/estop (default 10 Hz)",
    )
    p.add_argument(
        "--estop-topic",
        default="/phoenix/estop",
        help="std_msgs/Bool topic to publish (default /phoenix/estop)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    rclpy.init()
    node = WirelessEstopNode(
        button_mask=args.button_mask,
        joy_timeout_s=args.joy_timeout_s,
        publish_rate_hz=args.rate,
        estop_topic=args.estop_topic,
    )
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
