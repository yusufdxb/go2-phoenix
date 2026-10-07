#!/usr/bin/env python3
"""Passive STOP-path recorder for the FREE_FLOOR drill and walk. Subscribes only; publishes nothing.

Why. The bridge log shows the moment the bridge READS the release (the first non-policy tick) and
the moment it publishes a zero-motion hold. It cannot show how long the release took to get there:
the GO2 remote publishes only while a button is pressed, so the deadman node
(wireless_estop_node, --joy-timeout-s 0.5, 10 Hz) turns silence into estop=True only after the
timeout, and the policy keeps commanding until then. This recorder stamps both topics with
``time.monotonic_ns()``, the same clock as the bridge's ``t_mono_ns`` (one host, CLOCK_MONOTONIC),
so the analyzer can measure last-held-message -> zero command end to end.

One JSON object per line: {"t_mono_ns": int, "topic": "wireless"|"estop", "value": bool}
(``value`` is the L1 bit for wireless, the estop value for estop). Resolution is the remote's
50 Hz message period (about 20 ms). The recorder cannot see the human's finger or the radio hop
between the controller and the robot: the last message it receives is the earliest it can know.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--button-mask", type=lambda s: int(s, 0), default=0x02, help="deadman bit (L1)"
    )
    ap.add_argument("--estop-topic", default="/phoenix/estop")
    args = ap.parse_args(argv)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Bool
    from unitree_go.msg import WirelessController

    qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50
    )
    rclpy.init()
    node = Node("phoenix_free_floor_remote_trace")
    out = args.out.open("w", buffering=1)  # line-buffered: survives a kill

    def on_remote(msg: WirelessController) -> None:
        held = bool(int(msg.keys) & args.button_mask)
        out.write(
            json.dumps({"t_mono_ns": time.monotonic_ns(), "topic": "wireless", "value": held})
            + "\n"
        )

    def on_estop(msg: Bool) -> None:
        out.write(
            json.dumps(
                {"t_mono_ns": time.monotonic_ns(), "topic": "estop", "value": bool(msg.data)}
            )
            + "\n"
        )

    node.create_subscription(WirelessController, "/wirelesscontroller", on_remote, qos)
    node.create_subscription(Bool, args.estop_topic, on_estop, qos)
    out.write(
        json.dumps({"t_mono_ns": time.monotonic_ns(), "topic": "start", "value": None}) + "\n"
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        out.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
