"""GO2 remote gestures for single-operator hardware gate synchronization.

This module automates only synchronization. Start confirms that physical setup
is ready, A records the operator's explicit safe-result judgement, B records an
explicit NO-GO, and releasing L1 is always HALT or the request to DAMP. No robot
condition is inferred from button timing, and a missing answer never passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

DEADMAN_MASK = 0x02
START_MASK = 0x04
A_MASK = 0x100
B_MASK = 0x200
MODES = ("arm", "judgement", "release")


@dataclass
class RemoteGesture:
    """Pure rising-edge recognizer for one remote interaction."""

    mode: str
    neutral_seen: bool = False

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown remote gesture mode {self.mode!r}")

    def observe(self, keys: int) -> str | None:
        keys = int(keys)
        deadman = bool(keys & DEADMAN_MASK)
        if self.mode == "release":
            return "released" if not deadman else None
        if self.mode == "judgement" and not deadman:
            return "halt"
        if not deadman:
            self.neutral_seen = False
            return None

        confirm_mask = START_MASK if self.mode == "arm" else A_MASK | B_MASK
        pressed = keys & confirm_mask
        if pressed == 0:
            self.neutral_seen = True
            return None
        if not self.neutral_seen:
            return None
        if self.mode == "arm":
            return "armed" if pressed == START_MASK else None
        if pressed == A_MASK:
            return "yes"
        if pressed == B_MASK:
            return "no"
        return "ambiguous"


def _instruction(mode: str) -> str:
    if mode == "arm":
        return "Hold L1, then press Start once to confirm the physical setup is ready."
    if mode == "judgement":
        return (
            "Keep holding L1. Press A for YES, safe result observed. Press B for NO-GO. "
            "Release L1 to HALT and record NO-GO."
        )
    return "Release L1 now. The bridge will DAMP and end the attempt."


def wait_for_remote(mode: str, timeout_s: float) -> dict:
    """Wait for a fresh remote gesture. ROS imports stay out of CI import paths."""

    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from unitree_go.msg import WirelessController

    rclpy.init()
    node = Node("phoenix_operator_remote")
    qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )
    gesture = RemoteGesture(mode)
    result: str | None = None
    messages = 0

    def on_remote(msg) -> None:
        nonlocal messages, result
        messages += 1
        if result is None:
            result = gesture.observe(int(msg.keys))

    node.create_subscription(WirelessController, "/wirelesscontroller", on_remote, qos)
    started = time.monotonic()
    deadline = started + timeout_s
    print(f"\a[remote] {_instruction(mode)}", flush=True)
    try:
        while result is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        decided = time.monotonic()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return {
        "mode": mode,
        "result": result or "no_answer",
        "messages": messages,
        "elapsed_s": decided - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record = wait_for_remote(args.mode, args.timeout_s)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"[remote] {args.mode}: {record['result']} -> {args.out}")
    return 2 if record["result"] in ("no_answer", "ambiguous") else 0


if __name__ == "__main__":
    sys.exit(main())
