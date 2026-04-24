"""Phoenix → Unitree LowCmd bridge.

Converts the Phoenix policy's ``/joint_group_position_controller/command``
(std_msgs/Float64MultiArray of 12 joint targets in deploy.yaml:joint_order)
into ``unitree_go/msg/LowCmd`` at 50 Hz, with the CRC the GO2 firmware
requires. This is the single missing link between the policy and the robot.

Safety posture (this is a hardware-actuation node — read carefully):

* **Dry-run by default.** Without ``--live``, publishes to ``/lowcmd_dry``
  only. No motors move. ``--live`` must be passed explicitly per run; there
  is no config toggle and no env var.
* **Requires /lowstate before publishing.** If the node has never seen a
  valid LowState, it stays silent.
* **Per-step delta clip.** Each motor's target is clipped to ``±0.175 rad``
  from the last measured joint position. Matches the ``_clip_to_limits``
  behaviour already in ros2_policy_node.
* **Stale-command watchdog.** If no new policy command has arrived in
  ``watchdog_s`` seconds (default 0.2 s = 2×5 control periods at 50 Hz),
  the bridge holds the last measured joint positions with softer gains
  (``hold_kp``, ``hold_kd``) so the robot doesn't collapse when the policy
  stops or is estopped.
* **Respects /phoenix/estop.** If the estop publisher goes True, the bridge
  immediately switches to hold-current regardless of policy output. The
  policy node itself also enforces this, but the bridge double-checks
  because the policy may have already crashed.
* **Conservative default gains.** ``kp=25``, ``kd=0.5``. The Unitree stand
  example uses kp=60/kd=5 which is fine for stand but too stiff for a
  not-yet-tuned policy. Override via CLI if you know better for a given
  run.

Usage on the Jetson::

    # Terminal A: GO2 driver (provides /lowstate, consumes /lowcmd)
    # Terminal B: estop publisher (deadman)
    # Terminal C: this bridge — DRY RUN (default)
    python3 -m phoenix.sim2real.lowcmd_bridge_node
    # Verify /lowcmd_dry is publishing with sane values, joints permuted
    # correctly vs a known Phoenix command, and CRC is non-zero.
    #
    # Once that passes, go live:
    python3 -m phoenix.sim2real.lowcmd_bridge_node --live
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64MultiArray
from unitree_go.msg import LowCmd, LowState

from phoenix.sim2real.motor_crc import (
    PHOENIX_FOR_MOTOR,
    LowCmdRaw,
    compute_crc,
)
from phoenix.sim2real.safety import (
    MAX_DELTA_PER_STEP_RAD,
    estop_is_active,
    per_step_clip_array,
)

# MAX_DELTA_PER_STEP_RAD is re-exported from phoenix.sim2real.safety so
# the policy node and the bridge cannot drift on the slew-rate cap.


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
    # If no /phoenix/estop message has been received within this window we
    # treat the publisher as dead and force hold-pose. Default matches the
    # 0.5s window the wireless/joystick adapters use.
    estop_timeout_s: float = 0.5


def _load_deploy_config(path: Path) -> dict[str, Any]:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


class LowCmdBridge(Node):
    """See module docstring for safety posture."""

    def __init__(self, cfg: BridgeConfig) -> None:
        super().__init__("phoenix_lowcmd_bridge")
        self._cfg = cfg

        self._last_cmd_phoenix: np.ndarray | None = None
        self._last_cmd_time_ns: int | None = None
        self._last_measured_unitree: np.ndarray | None = None
        self._estop_latched: bool = False
        # Freshness tracking for the estop topic. The bridge is the
        # actuation gate, so it cannot trust a stale "estop is False"
        # value left over from a publisher that has since died.
        self._last_estop_value: bool | None = None
        self._last_estop_ns: int | None = None

        # The Phoenix policy node uses a single QoSProfile(depth=1, BEST_EFFORT)
        # for every pub/sub it owns (see ros2_policy_node.py). To match its
        # publishers as a subscriber, we have to also be BEST_EFFORT — a
        # RELIABLE subscriber against a BEST_EFFORT publisher is treated as
        # incompatible by DDS and receives nothing. Sensor-style topics
        # (LowState from the firmware) and our own /lowcmd output stay
        # BEST_EFFORT for the same reason and for low-latency control.
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
        self._timer = self.create_timer(1.0 / cfg.rate_hz, self._tick)

        mode_label = "LIVE (/lowcmd)" if cfg.live else "DRY (/lowcmd_dry)"
        self.get_logger().info(
            f"lowcmd bridge up in {mode_label} mode; rate={cfg.rate_hz} Hz, "
            f"kp={cfg.kp}, kd={cfg.kd}, hold_kp={cfg.hold_kp}, "
            f"hold_kd={cfg.hold_kd}, watchdog={cfg.watchdog_s}s, "
            f"clip={MAX_DELTA_PER_STEP_RAD} rad/step"
        )

    # --- subscriptions ------------------------------------------------------

    def _on_cmd(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != 12:
            self.get_logger().warn(f"ignoring command with len={len(msg.data)} (expected 12)")
            return
        arr = np.asarray(msg.data, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            self.get_logger().warn("ignoring command containing NaN/Inf")
            return
        self._last_cmd_phoenix = arr
        self._last_cmd_time_ns = self.get_clock().now().nanoseconds

    def _on_lowstate(self, msg: LowState) -> None:
        q = np.zeros(12, dtype=np.float32)
        for i in range(12):
            q[i] = float(msg.motor_state[i].q)
        if not np.all(np.isfinite(q)):
            self.get_logger().warn("LowState has NaN in motor_state.q; discarding")
            return
        self._last_measured_unitree = q

    def _on_estop(self, msg: Bool) -> None:
        self._last_estop_value = bool(msg.data)
        self._last_estop_ns = self.get_clock().now().nanoseconds
        if msg.data and not self._estop_latched:
            self.get_logger().warn("/phoenix/estop True; bridge latching to hold")
        if msg.data:
            self._estop_latched = True

    # --- tick ---------------------------------------------------------------

    def _tick(self) -> None:
        if self._last_measured_unitree is None:
            # Haven't observed any LowState yet — stay silent.
            return

        now_ns = self.get_clock().now().nanoseconds

        # Fail-closed estop: stale heartbeat ≡ estop True. The latch is
        # sticky once flipped so a recovering publisher can't lift it.
        estop_signal = estop_is_active(
            last_msg_received_ns=self._last_estop_ns,
            latest_value=self._last_estop_value,
            now_ns=now_ns,
            timeout_s=self._cfg.estop_timeout_s,
        )
        if estop_signal and not self._estop_latched:
            reason = (
                "no /phoenix/estop publisher"
                if self._last_estop_ns is None
                else "stale estop heartbeat"
            )
            self.get_logger().warn(f"bridge latching to hold: {reason}")
            self._estop_latched = True

        use_hold = self._estop_latched or self._is_command_stale(now_ns)
        if use_hold or self._last_cmd_phoenix is None:
            target_unitree = self._last_measured_unitree.copy()
            kp, kd = self._cfg.hold_kp, self._cfg.hold_kd
        else:
            phoenix_vec = self._last_cmd_phoenix
            # Reorder to Unitree motor layout.
            target_unitree = np.array(
                [phoenix_vec[PHOENIX_FOR_MOTOR[k]] for k in range(12)],
                dtype=np.float32,
            )
            # Clip per-step delta vs measured. Shared helper with the
            # policy node — see phoenix.sim2real.safety.
            target_unitree = per_step_clip_array(
                target_unitree, self._last_measured_unitree, MAX_DELTA_PER_STEP_RAD
            ).astype(np.float32, copy=False)
            kp, kd = self._cfg.kp, self._cfg.kd

        self._publish(target_unitree, kp, kd)

    def _is_command_stale(self, now_ns: int) -> bool:
        if self._last_cmd_time_ns is None:
            return True
        age_s = (now_ns - self._last_cmd_time_ns) / 1e9
        return age_s > self._cfg.watchdog_s

    # --- publish ------------------------------------------------------------

    def _publish(self, target_unitree: np.ndarray, kp: float, kd: float) -> None:
        # Build the raw (packed-C) view, compute CRC, then copy into a ROS msg.
        raw = LowCmdRaw()
        raw.head[0] = 0xFE
        raw.head[1] = 0xEF
        raw.levelFlag = 0xFF
        for i in range(12):
            raw.motorCmd[i].mode = 0x01
            raw.motorCmd[i].q = float(target_unitree[i])
            raw.motorCmd[i].dq = 0.0
            raw.motorCmd[i].tau = 0.0
            raw.motorCmd[i].Kp = kp
            raw.motorCmd[i].Kd = kd
        crc = compute_crc(raw)

        msg = LowCmd()
        msg.head[0] = 0xFE
        msg.head[1] = 0xEF
        msg.level_flag = 0xFF
        for i in range(12):
            msg.motor_cmd[i].mode = 0x01
            msg.motor_cmd[i].q = float(target_unitree[i])
            msg.motor_cmd[i].dq = 0.0
            msg.motor_cmd[i].tau = 0.0
            msg.motor_cmd[i].kp = kp
            msg.motor_cmd[i].kd = kd
        msg.crc = crc

        self._pub.publish(msg)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sim2real/deploy.yaml"),
        help="deploy.yaml, read for rate and topic names (default: configs/sim2real/deploy.yaml)",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="publish on /lowcmd (default: /lowcmd_dry). Required for motor motion.",
    )
    p.add_argument("--kp", type=float, default=25.0, help="active-control kp (default 25)")
    p.add_argument("--kd", type=float, default=0.5, help="active-control kd (default 0.5)")
    p.add_argument(
        "--hold-kp",
        type=float,
        default=20.0,
        help="hold-pose kp when stale/estopped (default 20)",
    )
    p.add_argument(
        "--hold-kd",
        type=float,
        default=1.0,
        help="hold-pose kd when stale/estopped (default 1.0)",
    )
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
        help=(
            "seconds without a /phoenix/estop heartbeat before forcing hold. "
            "Default: read safety.estop_timeout_s from --config (deploy.yaml), "
            "falling back to 0.5 s if absent."
        ),
    )
    return p.parse_args(argv)


def _build_config(args: argparse.Namespace) -> BridgeConfig:
    rate_hz = 50.0
    cmd_topic = "/joint_group_position_controller/command"
    lowstate_topic = "/lowstate"
    estop_topic = "/phoenix/estop"
    yaml_estop_timeout: float | None = None
    if args.config.exists():
        cfg = _load_deploy_config(args.config)
        rate_hz = float(cfg.get("control", {}).get("rate_hz", rate_hz))
        t = cfg.get("topics", {})
        cmd_topic = t.get("joint_command", cmd_topic)
        s = cfg.get("safety", {})
        estop_topic = s.get("emergency_stop_topic", estop_topic)
        if "estop_timeout_s" in s:
            yaml_estop_timeout = float(s["estop_timeout_s"])

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

    return BridgeConfig(
        rate_hz=rate_hz,
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
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    cfg = _build_config(args)

    rclpy.init()
    node = LowCmdBridge(cfg)
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
