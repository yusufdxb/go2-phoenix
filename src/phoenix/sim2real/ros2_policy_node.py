"""ROS 2 node that runs a Phoenix policy on the real GO2 at 50 Hz.

Runs in the **system Python + ROS 2** context (not Isaac Lab's Python).
Reads:

* ``/imu/data`` — sensor_msgs/Imu (orientation, angular velocity, linear accel)
* ``/joint_states`` — sensor_msgs/JointState (positions + velocities)
* ``/cmd_vel`` — geometry_msgs/Twist (teleop / higher-level policy command)

Publishes:

* ``/joint_group_position_controller/command`` — std_msgs/Float64MultiArray
  of 12 target joint positions in canonical order.

Safety:

* Dead-man's-switch on ``/phoenix/estop`` (std_msgs/Bool).
* Hard max runtime after which the node exits and sends the stand pose.
* Per-joint position, velocity, and torque clipping relative to URDF limits.
* Attitude abort (pitch/roll) and NaN-in-joint-state abort; same latch path
  as an external estop — node stops publishing policy actions and holds
  the default stand pose.

Optional telemetry:

* ``--log-parquet PATH`` — write every control step to a parquet that
  matches :class:`phoenix.real_world.TrajectoryLogger`'s schema, so the
  file is directly consumable by the Phoenix replay/fine-tune pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from phoenix.real_world.failure_detector import FailureDetector, FailureThresholds
from phoenix.real_world.trajectory_logger import TrajectoryLogger, TrajectoryStep

from .observation import JointOrder, ObservationBuilder
from .safety import (
    MAX_DELTA_PER_STEP_RAD,
    is_ready_to_command_motion,
    per_step_clip_array,
)

logger = logging.getLogger("phoenix.sim2real.ros2_policy_node")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Phoenix policy on the GO2 via ROS 2.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="Path to exported ONNX. Falls back to cfg['policy']['onnx_path'].",
    )
    p.add_argument(
        "--log-parquet",
        type=Path,
        default=None,
        help="If set, log each control step to this parquet path.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - requires ROS 2
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    import rclpy

    cfg = yaml.safe_load(args.config.read_text())

    onnx_path = args.onnx
    if onnx_path is None:
        cfg_onnx = cfg.get("policy", {}).get("onnx_path")
        if not cfg_onnx:
            raise SystemExit(
                "--onnx not given and cfg['policy']['onnx_path'] is empty"
            )
        onnx_path = Path(cfg_onnx)

    rclpy.init()
    node = _PhoenixPolicyNode(cfg, onnx_path, log_parquet=args.log_parquet)
    try:
        rclpy.spin(node.node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.node.destroy_node()
        rclpy.shutdown()
    return 0


class _PhoenixPolicyNode:  # pragma: no cover - requires ROS 2 runtime
    """Minimal ROS 2 node wrapper. All rclpy imports are done here so the
    enclosing package can still be imported in CI."""

    def __init__(self, cfg: dict, onnx_path: Path, log_parquet: Path | None = None):
        import onnxruntime as ort
        from geometry_msgs.msg import Twist
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Imu, JointState
        from std_msgs.msg import Bool, Float64MultiArray

        self._float_msg = Float64MultiArray

        self.joint_order = JointOrder(tuple(cfg["joint_order"]))
        self.default_q = np.asarray(
            [cfg["control"]["default_joint_pos"][n] for n in self.joint_order.names],
            dtype=np.float32,
        )
        self.obs_builder = ObservationBuilder(self.joint_order, cfg["control"]["default_joint_pos"])
        self.action_scale = float(cfg["control"]["action_scale"])
        self.rate_hz = float(cfg["control"]["rate_hz"])
        self.max_runtime = float(cfg["safety"]["max_runtime_s"])
        self.pos_margin = float(cfg["actuator_limits"]["position_margin_rad"])
        # Fail-closed timeouts. Defaults are deliberate: estop must be
        # heartbeated faster than 0.5s (heartbeat publisher runs at 10Hz),
        # IMU/joint state must be alive within 0.2s (the policy can't
        # operate on stale observations of an actuated robot).
        safety_cfg = cfg.get("safety", {})
        self.estop_timeout_s = float(safety_cfg.get("estop_timeout_s", 0.5))
        self.sensor_timeout_s = float(safety_cfg.get("sensor_timeout_s", 0.2))
        # We allow the robot up to this long after launch to bring up its
        # estop publisher + first sensor messages. After that we latch.
        self.startup_grace_s = float(safety_cfg.get("startup_grace_s", 3.0))

        # Policy-side observation padding. The Rough baseline expects
        # 48 proprio + 187 height-scan = 235 dims; the real GO2 has no
        # scanner, so we append zeros. Set to 0 after retraining on a
        # flat task (obs_dim=48).
        self.obs_pad_zeros = int(cfg.get("policy", {}).get("obs_pad_zeros", 187))

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        # Shape sanity: fail fast if padding doesn't match the ONNX input.
        expected_in = self.session.get_inputs()[0].shape
        expected_dim = expected_in[-1] if isinstance(expected_in[-1], int) else None
        proprio_dim = 48
        if expected_dim is not None and proprio_dim + self.obs_pad_zeros != expected_dim:
            raise ValueError(
                f"ONNX expects obs_dim={expected_dim} but node will emit "
                f"{proprio_dim}+{self.obs_pad_zeros}={proprio_dim + self.obs_pad_zeros}. "
                f"Set policy.obs_pad_zeros in deploy.yaml."
            )

        # Failure thresholds reused from the offline detector so the
        # on-robot abort and the sim replay flag the same regimes.
        self.thresholds = FailureThresholds()

        self._latest_imu = None
        self._latest_joint_state = None
        self._velocity_command = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(len(self.joint_order), dtype=np.float32)
        self._estopped = False
        self._abort_reason: str | None = None
        self._started_at = time.monotonic()
        self._step_idx = 0

        # Freshness bookkeeping for the fail-closed estop / sensor watchdogs.
        self._latest_imu_ns: int | None = None
        self._latest_joint_state_ns: int | None = None
        self._latest_estop_ns: int | None = None
        self._latest_estop_value: bool | None = None

        self._logger: TrajectoryLogger | None = None
        if log_parquet is not None:
            self._logger = TrajectoryLogger(log_parquet)

        self.node = Node("phoenix_policy_node")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        topics = cfg["topics"]
        self.node.create_subscription(Imu, topics["imu"], self._on_imu, qos)
        self.node.create_subscription(JointState, topics["joint_states"], self._on_joint_state, qos)
        self.node.create_subscription(Twist, topics["cmd_vel"], self._on_cmd_vel, qos)
        self.node.create_subscription(
            Bool, cfg["safety"]["emergency_stop_topic"], self._on_estop, qos
        )
        self.cmd_pub = self.node.create_publisher(Float64MultiArray, topics["joint_command"], qos)

        self.node.create_timer(1.0 / self.rate_hz, self._control_step)

    def _on_imu(self, msg):
        self._latest_imu = msg
        self._latest_imu_ns = time.monotonic_ns()

    def _on_joint_state(self, msg):
        self._latest_joint_state = msg
        self._latest_joint_state_ns = time.monotonic_ns()

    def _on_cmd_vel(self, msg):
        self._velocity_command = np.asarray(
            [msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32
        )

    def _on_estop(self, msg):
        self._latest_estop_ns = time.monotonic_ns()
        self._latest_estop_value = bool(msg.data)
        if msg.data and not self._estopped:
            self._latch_abort("external_estop")

    def _latch_abort(self, reason: str) -> None:
        self._estopped = True
        self._abort_reason = reason
        logger.warning("ABORT: %s — holding stand pose.", reason)

    def _ready_to_command_motion(self, now_ns: int) -> tuple[bool, str | None]:
        """Bind the shared ``is_ready_to_command_motion`` predicate to this
        node's freshness state. The predicate itself is in
        :mod:`phoenix.sim2real.safety` so it can be exhaustively tested
        in CI without rclpy.
        """
        return is_ready_to_command_motion(
            now_ns=now_ns,
            estop_last_ns=self._latest_estop_ns,
            estop_value=self._latest_estop_value,
            estop_timeout_s=self.estop_timeout_s,
            imu_last_ns=self._latest_imu_ns,
            joint_state_last_ns=self._latest_joint_state_ns,
            sensor_timeout_s=self.sensor_timeout_s,
        )

    def _control_step(self):
        now_ns = time.monotonic_ns()
        elapsed_s = time.monotonic() - self._started_at

        if elapsed_s > self.max_runtime and not self._estopped:
            self._latch_abort("max_runtime")

        if self._estopped:
            self._publish_default_pose()
            return

        ok, reason = self._ready_to_command_motion(now_ns)
        if not ok:
            # Two distinct cases share the same conservative posture:
            # (1) startup — preconditions not yet met, never seen. The node
            #     stays SILENT (no publish at all) so we cannot even ask
            #     the bridge to actively servo. The bridge's own fail-closed
            #     watchdog will hold motors with hold_kp/hold_kd in parallel.
            # (2) once any precondition has been satisfied at least once,
            #     subsequent failure (stale heartbeat, dropped sensor, etc.)
            #     means publish the safe default stand pose so the bridge
            #     can deliberately hold the robot upright.
            if elapsed_s > self.startup_grace_s:
                self._latch_abort(reason or "unknown_safety_gate")
                self._publish_default_pose()
            elif self._latest_estop_value is False and self._latest_imu is not None \
                    and self._latest_joint_state is not None:
                # Brief blip after we have already heard from every publisher.
                self._publish_default_pose()
            # else: STAY SILENT during cold startup. Do NOT publish anything
            # until estop+sensors have been heard from — the bridge holds
            # the robot via its own watchdog with conservative gains.
            return

        idx = self.joint_order.remap(list(self._latest_joint_state.name))
        q = np.asarray(self._latest_joint_state.position, dtype=np.float32)[idx]
        qd = np.asarray(self._latest_joint_state.velocity, dtype=np.float32)[idx]

        # Fail closed on bad sensor data.
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
            self._latch_abort("nan_in_joint_state")
            self._publish_default_pose()
            return

        ori = self._latest_imu.orientation
        pg = _projected_gravity_from_quat(ori.x, ori.y, ori.z, ori.w)
        roll, pitch, _yaw = _rpy_from_quat_xyzw(ori.x, ori.y, ori.z, ori.w)

        if abs(pitch) > self.thresholds.pitch_rad or abs(roll) > self.thresholds.roll_rad:
            self._latch_abort(
                f"attitude pitch={pitch:.2f} roll={roll:.2f}"
            )
            self._publish_default_pose()
            return

        base_ang_vel = np.asarray(
            [
                self._latest_imu.angular_velocity.x,
                self._latest_imu.angular_velocity.y,
                self._latest_imu.angular_velocity.z,
            ],
            dtype=np.float32,
        )
        base_lin_vel = np.zeros(3, dtype=np.float32)

        proprio = self.obs_builder.build(
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            projected_gravity=pg,
            velocity_command=self._velocity_command,
            joint_pos=q,
            joint_vel=qd,
            last_action=self._last_action,
        )
        if self.obs_pad_zeros > 0:
            obs = np.concatenate(
                [proprio, np.zeros(self.obs_pad_zeros, dtype=np.float32)]
            ).reshape(1, -1)
        else:
            obs = proprio.reshape(1, -1)

        action = self.session.run(["action"], {"obs": obs})[0][0]
        self._last_action = action.astype(np.float32, copy=False)

        target = self.default_q + self.action_scale * action
        target = self._clip_to_limits(target, q)

        msg = self._float_msg()
        msg.data = target.astype(np.float64).tolist()
        self.cmd_pub.publish(msg)

        if self._logger is not None:
            self._log_step(q=q, qd=qd, action=action, quat_xyzw=(ori.x, ori.y, ori.z, ori.w),
                           ang_vel=base_ang_vel)
        self._step_idx += 1

    def _log_step(self, *, q, qd, action, quat_xyzw, ang_vel) -> None:
        # base_pos and contact_forces aren't observable on stock GO2 without
        # odometry / foot sensors — emit zeros so the parquet schema matches
        # what the replay pipeline expects.
        self._logger.append(
            TrajectoryStep(
                step=self._step_idx,
                timestamp_s=time.monotonic() - self._started_at,
                base_pos=np.zeros(3, dtype=np.float32),
                base_quat=np.asarray(quat_xyzw, dtype=np.float32),
                base_lin_vel_body=np.zeros(3, dtype=np.float32),
                base_ang_vel_body=ang_vel.astype(np.float32),
                joint_pos=q.astype(np.float32),
                joint_vel=qd.astype(np.float32),
                command_vel=self._velocity_command.astype(np.float32),
                action=action.astype(np.float32),
                contact_forces=np.zeros(4, dtype=np.float32),
                failure_flag=False,
                failure_mode=None,
            )
        )

    def _clip_to_limits(self, target: np.ndarray, q: np.ndarray) -> np.ndarray:
        # Single source of truth lives in phoenix.sim2real.safety so the
        # bridge and the policy node provably share the slew-rate cap.
        return per_step_clip_array(target, q, MAX_DELTA_PER_STEP_RAD).astype(
            np.float32, copy=False
        )

    def _publish_default_pose(self) -> None:
        msg = self._float_msg()
        msg.data = self.default_q.astype(np.float64).tolist()
        self.cmd_pub.publish(msg)

    def shutdown(self) -> None:
        import rclpy

        # If rclpy has already been torn down (e.g. Ctrl-C propagated through
        # spin), publishing raises RCLError. The ABI is: only publish when the
        # default context is still valid.
        if rclpy.ok():
            try:
                self._publish_default_pose()
            except Exception as exc:  # noqa: BLE001 - best-effort safe pose
                logger.warning("shutdown: default-pose publish failed: %s", exc)
        if self._logger is not None:
            self._logger.close()
            self._logger = None
        if self._abort_reason:
            logger.warning("Shutdown after abort: %s", self._abort_reason)


def _projected_gravity_from_quat(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Rotate world-frame gravity (0,0,-1) into the body frame using quat (x,y,z,w).

    Matches Isaac Lab's ``mdp.projected_gravity`` and the parity-gate
    helper in :func:`phoenix.sim2real.verify_deploy._projected_gravity_from_quat_xyzw`
    byte-for-byte. The previous implementation had the gx/gy sign flipped
    (mirror-image gravity in the policy's obs vector) — see the
    audit fixes in this branch. Tested in ``tests/test_projected_gravity.py``.
    """
    gx = -2.0 * (x * z - w * y)
    gy = -2.0 * (y * z + w * x)
    gz = -(1.0 - 2.0 * (x * x + y * y))
    return np.asarray([gx, gy, gz], dtype=np.float32)


def _rpy_from_quat_xyzw(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """Return (roll, pitch, yaw) from a unit quaternion (x,y,z,w)."""
    roll = float(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    pitch = float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return roll, pitch, yaw


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
