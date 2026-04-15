"""Standalone /lowcmd_dry (or /lowcmd) inspector.

Subscribes to /lowcmd_dry and /lowstate for N seconds, prints simple stats:

* Cmd publish rate (Hz)
* Distinct kp values seen — tells you whether the bridge stayed in
  hold-pose (kp=hold_kp, default 20) or transitioned into active mode
  (kp=cfg.kp, default 25). If you only see hold_kp, the bridge isn't
  receiving the policy commands (likely a QoS mismatch).
* mean / max |cmd.q − state.q| across the run — close to zero means
  the bridge is in hold mode; close to the 0.175 rad/step clip means
  the policy is driving the bridge into active control with saturated
  per-step deltas (i.e. policy output is large vs current pose).

Usage::

    python3 scripts/lowcmd_inspect.py 6      # 6 seconds, defaults to /lowcmd_dry
    python3 scripts/lowcmd_inspect.py 6 /lowcmd
"""

from __future__ import annotations

import statistics
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from unitree_go.msg import LowCmd, LowState


class Inspect(Node):
    def __init__(self, cmd_topic: str) -> None:
        super().__init__("lowcmd_inspect")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.cmd_count = 0
        self.kp_seen: set[float] = set()
        self.last_q: list[float] | None = None
        self.last_state_q: list[float] | None = None
        self.diffs: list[float] = []
        self.create_subscription(LowCmd, cmd_topic, self._on_cmd, qos)
        self.create_subscription(LowState, "/lowstate", self._on_state, qos)

    def _on_cmd(self, msg: LowCmd) -> None:
        self.cmd_count += 1
        self.kp_seen.add(round(float(msg.motor_cmd[0].kp), 2))
        q = [float(msg.motor_cmd[i].q) for i in range(12)]
        self.last_q = q
        if self.last_state_q is not None:
            self.diffs.append(max(abs(a - b) for a, b in zip(q, self.last_state_q)))

    def _on_state(self, msg: LowState) -> None:
        self.last_state_q = [float(msg.motor_state[i].q) for i in range(12)]


def main() -> int:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    cmd_topic = sys.argv[2] if len(sys.argv) > 2 else "/lowcmd_dry"

    rclpy.init()
    node = Inspect(cmd_topic)
    t0 = time.time()
    while time.time() - t0 < duration:
        rclpy.spin_once(node, timeout_sec=0.05)

    elapsed = time.time() - t0
    print(f"topic={cmd_topic}  elapsed={elapsed:.2f}s")
    print(f"cmd_count={node.cmd_count} -> {node.cmd_count / max(elapsed, 1e-6):.1f} Hz")
    print(f"kp values seen: {sorted(node.kp_seen)}")
    if node.last_q is not None:
        print(f"last cmd q: {[round(x, 3) for x in node.last_q]}")
    if node.last_state_q is not None:
        print(f"last state q: {[round(x, 3) for x in node.last_state_q]}")
    if node.diffs:
        print(f"max |cmd.q - state.q|: {max(node.diffs):.4f} rad")
        print(f"mean |cmd.q - state.q|: {statistics.mean(node.diffs):.4f} rad")
        print("   [~0 → hold mode; close to 0.175 → active mode hitting clip]")

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
