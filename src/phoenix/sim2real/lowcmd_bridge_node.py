"""Phoenix -> Unitree LowCmd bridge: the final actuator safety boundary.

A thin ROS 2 shell. Every decision about what the motors are told is made by
:class:`phoenix.sim2real.actuator_gate.ActuatorGate`, a pure state machine covered
by ``tests/test_actuator_gate.py``; read that module's docstring for the modes,
the joint limits, LowState freshness, the estop and deadman rules, and what
changed relative to the previous bridge. This file only:

* feeds ``/lowstate``, ``/phoenix/estop``, the ``/phoenix/estop`` publisher set and
  the policy command (wire v2, :mod:`phoenix.sim2real.command_wire`) into the gate,
  stamped with ``time.monotonic_ns()``, never the ROS wall clock;
* publishes the gate's decision as ``unitree_go/msg/LowCmd`` with the firmware CRC;
* writes one telemetry line per tick (:mod:`phoenix.sim2real.bridge_telemetry`).

Safety posture:

* **Dry-run by default.** Without ``--live`` it publishes ``/lowcmd_dry`` only.
* **Live refuses to start** unless it has a deploy config that passes the deploy
  contract, a lock file whose hashes match every artifact that config reaches, a
  telemetry path, and ``--expect-sha`` matching the running code's commit.
* **Live requires a real deadman.** Exactly one ``/phoenix/estop`` publisher, and
  it must be ``wireless_estop_node`` or ``deadman_joy_node``.
* **Shutdown damps.** On Ctrl-C the gate latches damping and the bridge sends
  ``kp=0`` for one watchdog period before exiting, so the last command the motors
  received is damping rather than a stiff hold of a pose nobody is measuring.

Usage (the preflight and the run card give the exact, complete commands)::

    python3 -m phoenix.sim2real.lowcmd_bridge_node --config <deploy.yaml> \\
        --lock <lock.yaml> --telemetry <run_dir>/bridge.jsonl          # dry
    python3 -m phoenix.sim2real.lowcmd_bridge_node --live --config ... --lock ... \\
        --telemetry ... --expect-sha <commit> --stage E                  # live
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64MultiArray
from unitree_go.msg import LowCmd, LowState

from phoenix.sim2real.actuator_gate import REAL_DEADMAN_NODE_NAMES, ActuatorGate, GateParams
from phoenix.sim2real.bridge_telemetry import (
    HARDWARE_SLEW_METRIC,
    HARDWARE_SLEW_METRIC_DEFINITION,
    TelemetryWriter,
    utc_now_iso,
)
from phoenix.sim2real.deploy_contract import load_lock, validate_deploy_contract, verify_lock
from phoenix.sim2real.go2_model import (
    JOINT_LIMITS_PROVENANCE,
    JOINT_POSITION_LIMITS_RAD,
    LIMIT_ABORT_BAND_RAD,
    POLICY_JOINT_ORDER,
    UNITREE_MOTOR_ORDER,
)
from phoenix.sim2real.motor_crc import PHOENIX_FOR_MOTOR, build_raw_from_motor_values, compute_crc
from phoenix.sim2real.provenance import identity_problems, resolve_code_identity
from phoenix.sim2real.safety import MAX_DELTA_PER_STEP_RAD

# MAX_DELTA_PER_STEP_RAD is re-exported from phoenix.sim2real.safety so
# the policy node and the bridge cannot drift on the slew-rate cap.

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class BridgeConfig:
    rate_hz: float
    watchdog_s: float
    kp: float
    kd: float
    hold_kp: float
    hold_kd: float
    live: bool
    dry_topic: str
    live_topic: str
    cmd_topic: str
    lowstate_topic: str
    estop_topic: str
    #: /lowcmd PUBLISH rate. Unitree's own examples publish LowCmd at 500 Hz,
    #: holding the latest target between policy updates; the policy itself
    #: stays at the deploy config's rate_hz (typically 50 Hz). Decoupled from
    #: ``rate_hz`` (which remains a display/manifest value describing the
    #: policy's own cadence) so this bridge can hold the last-known-good
    #: target 10x more often than the policy refreshes it. ActuatorGate.tick
    #: is a pure function of (now_ns, last received command/lowstate), so
    #: calling it faster than the policy publishes is exactly "hold the
    #: latest target": nothing about its staleness/watchdog logic is rate-
    #: dependent on this timer.
    publish_rate_hz: float = 500.0
    # If no /phoenix/estop message has been received within this window we
    # treat the publisher as dead and force hold-pose. Default matches the
    # 0.5s window the wireless/joystick adapters use.
    estop_timeout_s: float = 0.5
    # LowState older than this revokes policy authority (deploy
    # safety.sensor_timeout_s, the same freshness authority the policy node uses).
    lowstate_timeout_s: float = 0.2
    # How long a stale LowState may still be held before damping.
    stale_hold_s: float = 0.2
    # Startup window for the first estop message / deadman discovery
    # (deploy safety.first_message_timeout_s).
    first_message_timeout_s: float = 15.0
    joint_order: tuple[str, ...] = POLICY_JOINT_ORDER
    config_path: Path | None = None
    lock_path: Path | None = None
    telemetry_path: Path | None = None
    expect_sha: str | None = None
    stage: str = "unlabelled"
    standup_s: float = 0.0
    standup_kp: float = 60.0
    standup_kd: float = 5.0
    extra: dict[str, Any] = field(default_factory=dict)

    def gate_params(self) -> GateParams:
        return GateParams(
            live=self.live,
            kp=self.kp,
            kd=self.kd,
            hold_kp=self.hold_kp,
            hold_kd=self.hold_kd,
            watchdog_s=self.watchdog_s,
            estop_timeout_s=self.estop_timeout_s,
            lowstate_timeout_s=self.lowstate_timeout_s,
            stale_hold_s=self.stale_hold_s,
            first_message_timeout_s=self.first_message_timeout_s,
            joint_order=tuple(self.joint_order),
            standup_s=self.standup_s,
            standup_kp=self.standup_kp,
            standup_kd=self.standup_kd,
        )


def _load_deploy_config(path: Path) -> dict[str, Any]:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def lowcmd_fields(target_unitree, kp: float, kd: float) -> tuple[list[float], int]:
    """The 12 motor targets and the firmware CRC for a LowCmd. Pure; tested."""
    q = [float(v) for v in target_unitree]
    raw = build_raw_from_motor_values(q, [float(kp)] * 12, [float(kd)] * 12)
    return q, compute_crc(raw)


class LowCmdBridge(Node):
    """See module docstring for safety posture."""

    def __init__(self, cfg: BridgeConfig, telemetry: TelemetryWriter | None) -> None:
        super().__init__("phoenix_lowcmd_bridge")
        self._cfg = cfg
        self._telemetry = telemetry
        self._gate = ActuatorGate(cfg.gate_params(), time.monotonic_ns())
        self._last_mode: str | None = None
        self._last_fault: str | None = None

        # BEST_EFFORT everywhere: the policy node publishes BEST_EFFORT (a
        # RELIABLE subscriber would receive nothing), LowState comes from the
        # firmware as sensor data, and /lowcmd is a 50 Hz control stream.
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._sub_cmd = self.create_subscription(
            Float64MultiArray, cfg.cmd_topic, self._on_cmd, qos_be
        )
        self._sub_state = self.create_subscription(
            LowState, cfg.lowstate_topic, self._on_lowstate, qos_be
        )
        self._sub_estop = self.create_subscription(Bool, cfg.estop_topic, self._on_estop, qos_be)
        self._pub = self.create_publisher(
            LowCmd, cfg.live_topic if cfg.live else cfg.dry_topic, qos_be
        )
        self._timer = self.create_timer(1.0 / cfg.publish_rate_hz, self._tick)
        self._graph_timer = self.create_timer(1.0, self._poll_estop_publishers)

        mode_label = "LIVE (/lowcmd)" if cfg.live else "DRY (/lowcmd_dry)"
        self.get_logger().info(
            f"lowcmd bridge up in {mode_label} mode; stage={cfg.stage} policy_rate={cfg.rate_hz} Hz, "
            f"publish_rate={cfg.publish_rate_hz} Hz, "
            f"kp={cfg.kp}, kd={cfg.kd}, hold_kp={cfg.hold_kp}, hold_kd={cfg.hold_kd}, "
            f"watchdog={cfg.watchdog_s}s, lowstate_timeout={cfg.lowstate_timeout_s}s, "
            f"stale_hold={cfg.stale_hold_s}s, clip={MAX_DELTA_PER_STEP_RAD} rad/step, "
            f"hard joint limits from {JOINT_LIMITS_PROVENANCE['repository']}"
        )

    # --- inputs -------------------------------------------------------------
    def _on_cmd(self, msg: Float64MultiArray) -> None:
        dims = getattr(msg.layout, "dim", None) or []
        label = dims[0].label if dims else ""
        self._gate.on_command(time.monotonic_ns(), label, list(msg.data))

    def _on_lowstate(self, msg: LowState) -> None:
        q = [float(msg.motor_state[i].q) for i in range(12)]
        dq = [float(msg.motor_state[i].dq) for i in range(12)]
        self._gate.on_lowstate(time.monotonic_ns(), q, dq)

    def _on_estop(self, msg: Bool) -> None:
        self._gate.on_estop(time.monotonic_ns(), bool(msg.data))

    def _poll_estop_publishers(self) -> None:
        infos = self.get_publishers_info_by_topic(self._cfg.estop_topic)
        self._gate.on_estop_publishers([info.node_name for info in infos])

    # --- tick ---------------------------------------------------------------
    def _tick(self) -> None:
        rec = self._gate.tick(time.monotonic_ns())
        if rec["publish"]:
            self._publish(rec["final_target_unitree"], rec["kp"], rec["kd"])
        self._report(rec)

    def _report(self, rec: dict[str, Any]) -> None:
        if rec["mode"] != self._last_mode:
            self.get_logger().warn(
                f"mode {self._last_mode} -> {rec['mode']} (cause={rec['hold_cause']})"
            )
            self._last_mode = rec["mode"]
        if rec["fault"] != self._last_fault:
            self.get_logger().error(f"LATCHED FAULT: {rec['fault']} (all: {rec['faults']})")
            self._last_fault = rec["fault"]
        if self._telemetry is not None:
            self._telemetry.write_tick(rec)

    def shutdown_damp(self) -> None:
        """Latch damping and send it for one watchdog period. Best effort."""
        self._gate.request_shutdown()
        ticks = max(1, int(round(self._cfg.watchdog_s * self._cfg.publish_rate_hz)))
        for _ in range(ticks):
            rec = self._gate.tick(time.monotonic_ns())
            if rec["publish"]:
                # Telemetry must record what went out, not what the gate asked for.
                if not rclpy.ok():
                    rec = {**rec, "publish": False, "publish_skipped": "ros_context_invalid"}
                else:
                    try:
                        self._publish(rec["final_target_unitree"], rec["kp"], rec["kd"])
                    except Exception as exc:  # keep trying the remaining damp ticks
                        rec = {**rec, "publish": False, "publish_skipped": repr(exc)}
            self._report(rec)
            time.sleep(1.0 / self._cfg.publish_rate_hz)

    # --- publish ------------------------------------------------------------
    def _publish(self, target_unitree, kp: float, kd: float) -> None:
        q, crc = lowcmd_fields(target_unitree, kp, kd)
        msg = LowCmd()
        msg.head[0] = 0xFE
        msg.head[1] = 0xEF
        msg.level_flag = 0xFF
        for i in range(12):
            msg.motor_cmd[i].mode = 0x01
            msg.motor_cmd[i].q = q[i]
            msg.motor_cmd[i].dq = 0.0
            msg.motor_cmd[i].tau = 0.0
            msg.motor_cmd[i].kp = float(kp)
            msg.motor_cmd[i].kd = float(kd)
        msg.crc = crc
        self._pub.publish(msg)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sim2real/deploy.yaml"),
        help="deploy config: rate, topics, joint order, timeouts (required to exist when --live)",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="publish on /lowcmd (default: /lowcmd_dry). Required for motor motion.",
    )
    p.add_argument(
        "--publish-rate-hz",
        type=float,
        default=500.0,
        help="/lowcmd publish rate: hold the latest target between policy updates, "
        "matching Unitree's own 500 Hz examples (default 500). Independent of the "
        "policy's own control.rate_hz in --config (typically 50 Hz).",
    )
    p.add_argument("--kp", type=float, default=25.0, help="active-control kp (default 25)")
    p.add_argument("--kd", type=float, default=0.5, help="active-control kd (default 0.5)")
    p.add_argument("--hold-kp", type=float, default=20.0, help="hold kp (default 20)")
    p.add_argument("--hold-kd", type=float, default=1.0, help="hold and damping kd (default 1.0)")
    p.add_argument(
        "--watchdog-s",
        type=float,
        default=0.2,
        help="seconds without a policy command before falling back to hold (default 0.2)",
    )
    p.add_argument(
        "--estop-timeout-s",
        type=float,
        default=None,
        help="estop heartbeat timeout; default safety.estop_timeout_s from --config, else 0.5",
    )
    p.add_argument(
        "--stale-hold-s",
        type=float,
        default=None,
        help="how long a stale LowState may be held before damping (default: --watchdog-s)",
    )
    p.add_argument(
        "--lock", type=Path, default=None, help="deploy lock file (required when --live)"
    )
    p.add_argument(
        "--telemetry", type=Path, default=None, help="JSONL telemetry path, never overwritten"
    )
    p.add_argument("--expect-sha", default=None, help="commit the running code must be (live)")
    p.add_argument("--stage", default="unlabelled", help="gate stage label for the record")
    p.add_argument(
        "--standup-s",
        type=float,
        default=0.0,
        help="ramp from the measured posture to the training stance over this long before "
        "following the policy (0 = off, the default)",
    )
    p.add_argument("--standup-kp", type=float, default=60.0, help="standup kp (default 60)")
    p.add_argument("--standup-kd", type=float, default=5.0, help="standup kd (default 5)")
    return p.parse_args(argv)


def _build_config(args: argparse.Namespace) -> BridgeConfig:
    rate_hz = 50.0
    cmd_topic = "/joint_group_position_controller/command"
    lowstate_topic = "/lowstate"
    estop_topic = "/phoenix/estop"
    yaml_estop_timeout: float | None = None
    lowstate_timeout_s = 0.2
    first_message_timeout_s = 15.0
    joint_order: tuple[str, ...] = POLICY_JOINT_ORDER
    if args.config.exists():
        cfg = _load_deploy_config(args.config)
        rate_hz = float(cfg.get("control", {}).get("rate_hz", rate_hz))
        t = cfg.get("topics", {})
        cmd_topic = t.get("joint_command", cmd_topic)
        s = cfg.get("safety", {})
        estop_topic = s.get("emergency_stop_topic", estop_topic)
        if "estop_timeout_s" in s:
            yaml_estop_timeout = float(s["estop_timeout_s"])
        lowstate_timeout_s = float(s.get("sensor_timeout_s", lowstate_timeout_s))
        first_message_timeout_s = float(s.get("first_message_timeout_s", first_message_timeout_s))
        if cfg.get("joint_order"):
            joint_order = tuple(cfg["joint_order"])

    # Resolution order for the estop heartbeat timeout, strict-to-loose:
    #   1. CLI flag --estop-timeout-s if explicitly passed (not None).
    #   2. safety.estop_timeout_s in deploy.yaml.
    #   3. Hard-coded 0.5 s last-resort default (matches the BridgeConfig
    #      default and the wireless/joystick adapter defaults).
    if args.estop_timeout_s is not None:
        estop_timeout_s = float(args.estop_timeout_s)
    elif yaml_estop_timeout is not None:
        estop_timeout_s = yaml_estop_timeout
    else:
        estop_timeout_s = 0.5

    stale_hold_s = getattr(args, "stale_hold_s", None)
    return BridgeConfig(
        rate_hz=rate_hz,
        publish_rate_hz=float(getattr(args, "publish_rate_hz", 500.0)),
        watchdog_s=args.watchdog_s,
        kp=args.kp,
        kd=args.kd,
        hold_kp=args.hold_kp,
        hold_kd=args.hold_kd,
        live=args.live,
        dry_topic="/lowcmd_dry",
        live_topic="/lowcmd",
        cmd_topic=cmd_topic,
        lowstate_topic=lowstate_topic,
        estop_topic=estop_topic,
        estop_timeout_s=estop_timeout_s,
        lowstate_timeout_s=lowstate_timeout_s,
        stale_hold_s=float(args.watchdog_s if stale_hold_s is None else stale_hold_s),
        first_message_timeout_s=first_message_timeout_s,
        joint_order=joint_order,
        config_path=args.config,
        lock_path=getattr(args, "lock", None),
        telemetry_path=getattr(args, "telemetry", None),
        expect_sha=getattr(args, "expect_sha", None),
        stage=getattr(args, "stage", "unlabelled"),
        standup_s=float(getattr(args, "standup_s", 0.0)),
        standup_kp=float(getattr(args, "standup_kp", 60.0)),
        standup_kd=float(getattr(args, "standup_kd", 5.0)),
    )


def startup_problems(cfg: BridgeConfig) -> tuple[list[str], dict[str, Any]]:
    """Everything that must refuse startup, plus the manifest to record. Pure-ish (reads files)."""
    problems: list[str] = []
    deploy_cfg: dict[str, Any] | None = None
    if cfg.config_path is not None and cfg.config_path.is_file():
        deploy_cfg = _load_deploy_config(cfg.config_path)
        problems.extend(f"deploy contract: {p}" for p in validate_deploy_contract(deploy_cfg))
    elif cfg.live:
        problems.append(f"--live requires an existing --config (got {cfg.config_path})")

    observed: dict[str, Any] = {}
    lock_record: dict[str, Any] | None = None
    if cfg.lock_path is not None:
        try:
            lock = load_lock(cfg.lock_path)
        except (OSError, ValueError) as exc:
            problems.append(f"lock unreadable: {exc}")
        else:
            if deploy_cfg is None:
                problems.append("a lock was given but the deploy config could not be read")
            else:
                lock_problems, observed = verify_lock(lock, deploy_cfg, cfg.config_path)
                problems.extend(f"lock: {p}" for p in lock_problems)
            lock_record = {"path": str(cfg.lock_path), "name": lock.get("name")}
    elif cfg.live:
        problems.append("--live requires --lock")

    identity = resolve_code_identity(REPO_ROOT)
    id_problems = identity_problems(identity, expected_sha=cfg.expect_sha)
    if cfg.live:
        if not cfg.expect_sha:
            problems.append("--live requires --expect-sha")
        problems.extend(f"code identity: {p}" for p in id_problems)
        if cfg.telemetry_path is None:
            problems.append("--live requires --telemetry")

    params = cfg.gate_params()
    manifest = {
        "node": "phoenix_lowcmd_bridge",
        "stage": cfg.stage,
        "live": cfg.live,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "start_utc": utc_now_iso(),
        "start_mono_ns": time.monotonic_ns(),
        "code_identity": identity.to_dict(),
        "code_identity_problems": id_problems,
        "deploy_config": observed.get("deploy_config"),
        "artifacts": observed.get("artifacts"),
        "lock": lock_record,
        "startup_problems": list(problems),
        "gate_params": {**asdict(params), "deadman_required": params.deadman_required},
        "topics": {
            "command": cfg.cmd_topic,
            "lowstate": cfg.lowstate_topic,
            "estop": cfg.estop_topic,
            "output": cfg.live_topic if cfg.live else cfg.dry_topic,
        },
        "joint_order_policy": list(cfg.joint_order),
        "motor_order_unitree": list(UNITREE_MOTOR_ORDER),
        "phoenix_for_motor": list(PHOENIX_FOR_MOTOR),
        "joint_limits_rad": {k: list(v) for k, v in JOINT_POSITION_LIMITS_RAD.items()},
        "joint_limits_provenance": JOINT_LIMITS_PROVENANCE,
        "limit_abort_band_rad": LIMIT_ABORT_BAND_RAD,
        "real_deadman_node_names": sorted(REAL_DEADMAN_NODE_NAMES),
        "hardware_slew_metric": HARDWARE_SLEW_METRIC,
        "hardware_slew_metric_definition": HARDWARE_SLEW_METRIC_DEFINITION,
        "array_orders": {
            "q_unitree, dq_unitree, *_unitree, slew_*, limit_*": "motor_order_unitree",
            "policy.*": "joint_order_policy",
        },
    }
    return problems, manifest


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    cfg = _build_config(args)
    problems, manifest = startup_problems(cfg)
    if problems:
        print("REFUSING TO START lowcmd bridge:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    telemetry = TelemetryWriter(cfg.telemetry_path, manifest) if cfg.telemetry_path else None
    # rclpy's own SIGINT handler shuts the context down before ``finally`` runs, which made
    # shutdown_damp skip every damp publish. Keep the context alive; SIGINT/SIGTERM only set
    # a flag (raising inside rclpy's C calls surfaces as a RuntimeError), and the loop below
    # exits cleanly so the damp window is actually sent.
    from rclpy.signals import SignalHandlerOptions

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    stop: list[str] = []

    def _request_stop(signum, _frame) -> None:
        stop.append(signal.Signals(signum).name.lower())

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    node = LowCmdBridge(cfg, telemetry)
    end_reason = "spin_returned"
    try:
        while not stop and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
        if stop:
            end_reason = stop[0]
    except KeyboardInterrupt:
        end_reason = "sigint"
    finally:
        try:
            node.shutdown_damp()
        finally:
            if telemetry is not None:
                telemetry.close({"reason": end_reason, "faults": list(node._gate.faults)})
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
