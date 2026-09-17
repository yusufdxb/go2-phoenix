"""Motors-off ROS 2 probes for the hardware preflight.

Two subcommands, each writing one JSON file that
:mod:`phoenix.sim2real.preflight_eval` turns into a verdict:

``rates``    subscribe BEST_EFFORT to the deploy topics for ``--duration`` seconds
             and record, per topic: message count, rate, maximum inter-message gap
             (receive time, monotonic clock), the publishing node names, and the
             content statistics the freshness stage needs (LowState joint range and
             non-finite count, IMU attitude extremes, joint-state names).
``deadman``  Records every ``/phoenix/estop`` message and advances only after it
             observes sustained HOLD, RELEASE and HOLD AGAIN states. The
             operator never has to touch the terminal.

Neither subcommand publishes anything. Rates are measured from receive times on
this host, never from message stamps: the robot's and the payload's clocks are
both known to be wrong (``docs/go2_field_notes.md`` section 6).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

COMMAND_TOPIC = "/joint_group_position_controller/command"

#: topic -> (python module, message class)
TOPIC_TYPES: dict[str, tuple[str, str]] = {
    "/lowstate": ("unitree_go.msg", "LowState"),
    "/joint_states": ("sensor_msgs.msg", "JointState"),
    "/imu/data": ("sensor_msgs.msg", "Imu"),
    COMMAND_TOPIC: ("std_msgs.msg", "Float64MultiArray"),
    "/lowcmd_dry": ("unitree_go.msg", "LowCmd"),
    "/lowcmd": ("unitree_go.msg", "LowCmd"),
    "/phoenix/estop": ("std_msgs.msg", "Bool"),
}


def timing_stats(times_s: list[float]) -> dict[str, Any]:
    """Count, rate and max gap from monotonic receive times. Pure."""
    n = len(times_s)
    if n < 2:
        return {"count": n, "rate_hz": None, "max_gap_s": None}
    t = np.asarray(times_s, dtype=np.float64)
    span = float(t[-1] - t[0])
    return {
        "count": n,
        "rate_hz": (n - 1) / span if span > 0 else None,
        "max_gap_s": float(np.max(np.diff(t))),
    }


def roll_pitch_from_quat_xyzw(x: float, y: float, z: float, w: float) -> tuple[float, float]:
    roll = float(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    pitch = float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    return roll, pitch


def _extract_lowstate(extra: dict[str, Any], msg: Any) -> None:
    q = np.asarray([float(msg.motor_state[i].q) for i in range(12)])
    dq = np.asarray([float(msg.motor_state[i].dq) for i in range(12)])
    extra.setdefault("non_finite", 0)
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq))):
        extra["non_finite"] += 1
        return
    extra["q_min"] = [float(v) for v in np.minimum(extra.get("q_min", q), q)]
    extra["q_max"] = [float(v) for v in np.maximum(extra.get("q_max", q), q)]


def _extract_joint_states(extra: dict[str, Any], msg: Any) -> None:
    extra["names"] = sorted(set(extra.get("names", [])) | {str(n) for n in msg.name})


def _extract_imu(extra: dict[str, Any], msg: Any) -> None:
    o = msg.orientation
    g = msg.angular_velocity
    values = np.asarray([o.x, o.y, o.z, o.w, g.x, g.y, g.z], dtype=np.float64)
    extra.setdefault("non_finite", 0)
    if not np.all(np.isfinite(values)):
        extra["non_finite"] += 1
        return
    roll, pitch = roll_pitch_from_quat_xyzw(o.x, o.y, o.z, o.w)
    extra["max_abs_roll_rad"] = max(extra.get("max_abs_roll_rad", 0.0), abs(roll))
    extra["max_abs_pitch_rad"] = max(extra.get("max_abs_pitch_rad", 0.0), abs(pitch))


def _extract_command(extra: dict[str, Any], msg: Any) -> None:
    dims = getattr(msg.layout, "dim", None) or []
    extra["labels"] = sorted(set(extra.get("labels", [])) | {dims[0].label if dims else ""})
    extra["lengths"] = sorted(set(extra.get("lengths", [])) | {len(msg.data)})


def _extract_estop(extra: dict[str, Any], msg: Any) -> None:
    extra["values_seen"] = sorted(set(extra.get("values_seen", [])) | {bool(msg.data)})


EXTRACTORS: dict[str, Callable[[dict[str, Any], Any], None]] = {
    "/lowstate": _extract_lowstate,
    "/joint_states": _extract_joint_states,
    "/imu/data": _extract_imu,
    COMMAND_TOPIC: _extract_command,
    "/phoenix/estop": _extract_estop,
}


class TopicRecorder:
    """Per-topic receive times plus the content statistics each stage needs. Pure."""

    def __init__(self, topic: str, clock: Callable[[], float] = time.monotonic) -> None:
        self.topic = topic
        self.times: list[float] = []
        self.extra: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self._clock = clock
        self._lock = threading.Lock()
        self._record_events = topic == "/phoenix/estop"

    def on_message(self, msg: Any) -> None:
        now_s = self._clock()
        with self._lock:
            self.times.append(now_s)
            extract = EXTRACTORS.get(self.topic)
            if extract is not None:
                extract(self.extra, msg)
            if self._record_events:
                self.events.append({"t": now_s, "value": bool(msg.data)})

    def report(self) -> dict[str, Any]:
        with self._lock:
            out = timing_stats(list(self.times))
            out.update(self.extra)
        return out

    def event_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.events)


class ObservedBooleanPhase:
    """Recognize one sustained boolean state from timestamped observations."""

    def __init__(self, *, desired: bool, duration_s: float) -> None:
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        self.desired = bool(desired)
        self.duration_s = float(duration_s)
        self.started_s: float | None = None

    def observe(self, timestamp_s: float, value: bool) -> tuple[float, float] | None:
        if bool(value) != self.desired:
            self.started_s = None
            return None
        if self.started_s is None:
            self.started_s = float(timestamp_s)
        if float(timestamp_s) - self.started_s >= self.duration_s:
            return self.started_s, float(timestamp_s)
        return None


class _Probe:
    """rclpy wiring shared by both subcommands. Imports ROS only when constructed."""

    def __init__(self, topics: list[str]) -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        self.rclpy = rclpy
        rclpy.init()
        self.node = Node("phoenix_hw_probe")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50
        )
        self.recorders: dict[str, TopicRecorder] = {}
        for topic in topics:
            module, cls = TOPIC_TYPES[topic]
            rec = TopicRecorder(topic)
            self.recorders[topic] = rec
            msg_type = getattr(importlib.import_module(module), cls)
            self.node.create_subscription(msg_type, topic, rec.on_message, qos)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def publishers(self, topic: str) -> list[str]:
        return sorted(i.node_name for i in self.node.get_publishers_info_by_topic(topic))

    def close(self) -> None:
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


def cmd_rates(args: argparse.Namespace) -> int:
    probe = _Probe(list(args.topics))
    try:
        started = time.monotonic()
        while time.monotonic() - started < args.duration:
            probe.executor.spin_once(timeout_sec=0.05)
        report: dict[str, Any] = {"duration_s": args.duration, "topics": {}}
        for topic, rec in probe.recorders.items():
            entry = rec.report()
            entry["publishers"] = probe.publishers(topic)
            report["topics"][topic] = entry
    finally:
        probe.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    for topic, entry in report["topics"].items():
        print(
            f"[probe] {topic:45s} n={entry['count']:6d} rate={entry['rate_hz']} "
            f"max_gap={entry['max_gap_s']} publishers={entry['publishers']}"
        )
    return 0


def cmd_deadman(args: argparse.Namespace) -> int:
    probe = _Probe([args.estop_topic])
    recorder = probe.recorders[args.estop_topic]
    polls: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    stop = threading.Event()

    def spin() -> None:
        last_poll = 0.0
        while not stop.is_set():
            probe.executor.spin_once(timeout_sec=0.02)
            now = time.monotonic()
            if now - last_poll >= 0.5:
                polls.append({"t": now, "publishers": probe.publishers(args.estop_topic)})
                last_poll = now

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    requested = (
        ("hold", False, "HOLD the deadman and keep holding it"),
        ("release", True, "RELEASE the deadman completely"),
        ("rehold", False, "HOLD the deadman again and keep holding it"),
    )
    failure: str | None = None
    try:
        event_index = 0
        for name, desired, instruction in requested:
            print(
                f"\a\n[deadman] {instruction}. The probe advances after observing "
                f"{args.phase_s:g} continuous seconds."
            )
            tracker = ObservedBooleanPhase(desired=desired, duration_s=args.phase_s)
            deadline = time.monotonic() + args.transition_timeout_s
            completed: tuple[float, float] | None = None
            while time.monotonic() < deadline and completed is None:
                events = recorder.event_snapshot()
                for event in events[event_index:]:
                    completed = tracker.observe(float(event["t"]), bool(event["value"]))
                    if completed is not None:
                        break
                event_index = len(events)
                if completed is None:
                    time.sleep(0.02)
            if completed is None:
                failure = f"timed out waiting for observed {name} state"
                phases.append({"name": name, "status": "timeout"})
                break
            start, end = completed
            phases.append({"name": name, "t_start": start, "t_end": end, "status": "observed"})
            print(f"[deadman] observed '{name}' continuously for {end - start:.2f} s")
    finally:
        stop.set()
        thread.join(timeout=2.0)
        messages = list(recorder.events)
        probe.close()
    trace = {
        "estop_topic": args.estop_topic,
        "estop_timeout_s": args.estop_timeout_s,
        "phase_s": args.phase_s,
        "messages": messages,
        "publisher_polls": polls,
        "phases": phases,
        "failure": failure,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, indent=2) + "\n")
    print(f"[deadman] {len(messages)} messages, {len(polls)} publisher polls -> {out}")
    return 1 if failure else 0


def active_units(candidates: Sequence[str]) -> list[str]:
    """Which of ``candidates`` systemd reports as active. Pure-ish, no ROS.

    Uses ``systemctl is-active``, which exits non-zero for an inactive or
    unknown unit, so an absent unit is simply not active. A systemctl that
    cannot be run at all raises: reporting "nothing competing" because the
    check could not run would turn a broken probe into a green gate.
    """

    import shutil
    import subprocess

    if shutil.which("systemctl") is None:
        raise RuntimeError(
            "systemctl not found, so competing services cannot be checked. "
            "Run this probe on the payload."
        )
    found = []
    for unit in candidates:
        result = subprocess.run(
            ["systemctl", "is-active", unit],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.strip() == "active":
            found.append(unit)
    return found


def cmd_contention(args: argparse.Namespace) -> int:  # pragma: no cover - needs ROS 2
    """Record who else is sharing the robot right now."""
    import rclpy
    from rclpy.node import Node

    from .preflight_eval import COMPETING_SERVICES

    units = active_units(COMPETING_SERVICES)

    rclpy.init()
    node = Node("phoenix_contention_probe")
    try:
        # Give discovery a moment; an empty graph read as "nobody else is here"
        # is the same false green this probe exists to prevent.
        end = time.monotonic() + float(args.discovery_s)
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
        names = sorted(f"{ns.rstrip('/')}/{n}" for n, ns in node.get_node_names_and_namespaces())
        topics = len(node.get_topic_names_and_types())
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    probe = {
        "active_units": units,
        "ros_nodes": names,
        "topic_count": topics,
        "discovery_s": float(args.discovery_s),
    }
    if args.lowstate_rate_hz is not None:
        probe["lowstate_rate_hz"] = float(args.lowstate_rate_hz)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(probe, indent=2) + "\n")
    print(f"[contention] units={units} nodes={len(names)} topics={topics} -> {out}")
    return 1 if units else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    rates = sub.add_parser("rates", help="topic rates, gaps, publishers and content statistics")
    rates.add_argument("--duration", type=float, default=10.0)
    rates.add_argument("--topics", nargs="+", default=list(TOPIC_TYPES), choices=list(TOPIC_TYPES))
    rates.add_argument("--out", required=True)
    rates.set_defaults(func=cmd_rates)
    dead = sub.add_parser("deadman", help="interactive physical deadman trace")
    dead.add_argument("--estop-topic", default="/phoenix/estop")
    dead.add_argument("--estop-timeout-s", type=float, required=True)
    dead.add_argument("--phase-s", type=float, default=3.0, help="recording time per prompt")
    dead.add_argument(
        "--transition-timeout-s",
        type=float,
        default=60.0,
        help="fail after this long without each requested physical transition",
    )
    dead.add_argument("--out", required=True)
    dead.set_defaults(func=cmd_deadman)
    cont = sub.add_parser(
        "contention", help="record competing services and nodes sharing the robot"
    )
    cont.add_argument("--discovery-s", type=float, default=5.0)
    cont.add_argument(
        "--lowstate-rate-hz",
        type=float,
        default=None,
        help="measured /lowstate rate, from a prior 'rates' probe, to gate starvation",
    )
    cont.add_argument("--out", required=True)
    cont.set_defaults(func=cmd_contention)
    return p


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs ROS 2
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
