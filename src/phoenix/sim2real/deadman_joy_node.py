"""Joystick-driven deadman for /phoenix/estop.

Phoenix's policy node treats ``/phoenix/estop`` as an external estop: a
publisher on that topic MUST be live before the policy will command
anything, and ``msg.data == True`` latches an abort. This node is the
joystick side of that: hold the deadman button to keep estop ``False``;
release it (or stop the gamepad) and estop goes ``True``.

Why this and not ``scripts/estop_publisher.sh``:

* ``estop_publisher.sh`` just heartbeats ``False`` at 10 Hz regardless of
  operator attention. It satisfies the "publisher exists" check but is
  not an actual deadman — if the human operator walks away, the robot
  keeps running.
* This node ties estop to a physical input. If the gamepad disconnects
  or the button is released, estop flips to ``True`` within one
  publish period.

Default mapping matches the Logitech F710 / common gamepads:
* Button index 4  = LB / L1 (deadman — must hold)

If your controller indexes LB differently, override with ``--button``. The
Unitree wireless remote surfaced via ``/wirelesscontroller_unprocessed`` is
a different topic and is NOT a deadman-suitable input by default.

Usage::

    python3 -m phoenix.sim2real.deadman_joy_node            # default L1
    python3 -m phoenix.sim2real.deadman_joy_node --button 5 # R1 instead
"""

from __future__ import annotations

import argparse
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

from phoenix.sim2real.safety import deadman_should_estop


class DeadmanNode(Node):
    def __init__(
        self,
        button_index: int,
        joy_timeout_s: float,
        publish_rate_hz: float,
        estop_topic: str,
    ) -> None:
        super().__init__("phoenix_deadman")
        self._button_index = button_index
        self._joy_timeout_s = joy_timeout_s

        self._last_joy_time_ns: int | None = None
        self._button_held: bool = False

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._sub = self.create_subscription(Joy, "/joy", self._on_joy, qos)
        self._pub = self.create_publisher(Bool, estop_topic, qos)
        self._timer = self.create_timer(1.0 / publish_rate_hz, self._tick)

        self.get_logger().info(
            f"deadman up: hold button {button_index} to keep {estop_topic}=False; "
            f"joy-timeout={joy_timeout_s}s"
        )

    def _on_joy(self, msg: Joy) -> None:
        self._last_joy_time_ns = self.get_clock().now().nanoseconds
        if self._button_index < len(msg.buttons):
            self._button_held = bool(msg.buttons[self._button_index])
        else:
            # Controller has fewer buttons than expected — treat as released.
            self._button_held = False

    def _tick(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        estopped = deadman_should_estop(
            last_input_ns=self._last_joy_time_ns,
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
        "--button",
        type=int,
        default=4,
        help="joy buttons[] index of the deadman (default 4 = LB/L1)",
    )
    p.add_argument(
        "--joy-timeout-s",
        type=float,
        default=0.5,
        help="seconds without any /joy message before forcing estop True (default 0.5)",
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
    node = DeadmanNode(
        button_index=args.button,
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
