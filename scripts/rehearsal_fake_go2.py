"""LOCALHOST-ONLY rehearsal fakes: a synthetic /lowstate and a GO2 remote.

Exists so the staged hardware gates (scripts/harness_preflight.sh), both bridges,
the policy node, the probes and the evaluators can be exercised end to end on a
workstation with no robot. Everything it produces is recorded as rehearsal
evidence (PHOENIX_REHEARSAL=1), which the ledger never counts.

It must never run next to a GO2, so it refuses to start unless
ROS_LOCALHOST_ONLY=1 and PHOENIX_REHEARSAL=1, and after a discovery wait it
refuses to publish if any other node already publishes /lowstate or
/wirelesscontroller.

Publishes:
* ``/lowstate`` (unitree_go/LowState) at 500 Hz: joints at a fixed pose
  (``--pose folded`` is Unitree's own folded pose, ``stand`` the training pose),
  zero velocity, a level IMU quaternion. The joints do not respond to /lowcmd.
* ``/wirelesscontroller`` (unitree_go/WirelessController) at 50 Hz while the
  deadman (L1, keys 0x02) is held, and NOTHING while released, mirroring the real
  remote, which publishes only while a button is pressed. ``--deadman`` is a
  schedule such as ``hold`` or ``hold:3.5,release:3,hold``; ``--deadman-file``
  instead reads ``hold`` or ``release`` from a file every tick, so a rehearsal
  driver can follow the deadman prompts of ``scripts/harness_preflight.sh C``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phoenix.sim2real.go2_model import (  # noqa: E402
    TRAINING_DEFAULT_JOINT_POS,
    UNITREE_EXAMPLE_FOLDED_POSE,
    UNITREE_MOTOR_ORDER,
)


def parse_schedule(text: str) -> list[tuple[bool, float]]:
    """``hold:3.5,release:3,hold`` -> [(True, 3.5), (False, 3.0), (True, inf)]."""
    out: list[tuple[bool, float]] = []
    for part in text.split(","):
        name, _, seconds = part.partition(":")
        if name not in ("hold", "release"):
            raise ValueError(f"bad deadman schedule entry {part!r}")
        out.append((name == "hold", float(seconds) if seconds else float("inf")))
    return out


def held_at(schedule: list[tuple[bool, float]], elapsed_s: float) -> bool:
    t = 0.0
    for held, duration in schedule:
        if elapsed_s < t + duration:
            return held
        t += duration
    return schedule[-1][0]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pose", choices=("folded", "stand"), default="folded")
    p.add_argument(
        "--deadman", default="hold", help="schedule, e.g. hold or hold:3.5,release:3,hold"
    )
    p.add_argument("--deadman-file", type=Path, default=None, help="file holding hold or release")
    p.add_argument("--no-lowstate", action="store_true")
    args = p.parse_args(argv)

    if os.environ.get("ROS_LOCALHOST_ONLY") != "1" or os.environ.get("PHOENIX_REHEARSAL") != "1":
        print(
            "REFUSING: rehearsal fakes need ROS_LOCALHOST_ONLY=1 and PHOENIX_REHEARSAL=1",
            file=sys.stderr,
        )
        return 2
    schedule = parse_schedule(args.deadman)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from unitree_go.msg import LowState, WirelessController

    if args.pose == "folded":
        q = list(UNITREE_EXAMPLE_FOLDED_POSE)
    else:
        q = [TRAINING_DEFAULT_JOINT_POS[n] for n in UNITREE_MOTOR_ORDER]

    rclpy.init()
    node = Node("phoenix_rehearsal_fake_go2")
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    for topic in ("/lowstate", "/wirelesscontroller"):
        others = [i.node_name for i in node.get_publishers_info_by_topic(topic)]
        if others:
            print(
                f"REFUSING: {topic} already has publishers {others}; this is not an empty rehearsal graph",
                file=sys.stderr,
            )
            node.destroy_node()
            rclpy.shutdown()
            return 3

    qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10
    )
    lowstate_pub = None if args.no_lowstate else node.create_publisher(LowState, "/lowstate", qos)
    remote_pub = node.create_publisher(WirelessController, "/wirelesscontroller", qos)
    started = time.monotonic()

    def publish_lowstate() -> None:
        msg = LowState()
        for i in range(12):
            msg.motor_state[i].q = float(q[i])
            msg.motor_state[i].dq = 0.0
        msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]  # Unitree order w, x, y, z
        lowstate_pub.publish(msg)

    def deadman_held() -> bool:
        if args.deadman_file is not None:
            try:
                return args.deadman_file.read_text().strip() == "hold"
            except OSError:
                return False
        return held_at(schedule, time.monotonic() - started)

    def publish_remote() -> None:
        if deadman_held():
            msg = WirelessController()
            msg.keys = 0x02
            remote_pub.publish(msg)

    if lowstate_pub is not None:
        node.create_timer(1.0 / 500.0, publish_lowstate)
    node.create_timer(1.0 / 50.0, publish_remote)
    print(f"[fake_go2] localhost rehearsal: pose={args.pose} deadman={args.deadman}", flush=True)
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
