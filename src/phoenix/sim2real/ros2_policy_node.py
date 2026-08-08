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
* Per-step joint slew-rate clip, shared with the bridge via
  ``per_step_clip_array`` in ``phoenix.sim2real.safety``. NOTE: per-joint
  position / velocity / torque limit clipping is not implemented here.
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

from phoenix.real_world.failure_detector import FailureThresholds
from phoenix.real_world.trajectory_logger import TrajectoryLogger, TrajectoryStep

from .gate import GateConfig, Outcome, SensorSnapshot, evaluate_gates
from .mode_switch import ModeSwitchCfg, State, initial_state
from .mode_switch import step as mode_step
from .observation import JointOrder, ObservationBuilder
from .safety import MAX_DELTA_PER_STEP_RAD, per_step_clip_array

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
            raise SystemExit("--onnx not given and cfg['policy']['onnx_path'] is empty")
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
        # Fail-closed timeouts. estop must be heartbeated well inside
        # estop_timeout_s (code default 0.5s; deploy.yaml sets 0.8s; the
        # heartbeat publisher runs at 10Hz). IMU/joint state must be alive
        # within sensor_timeout_s (default 0.2s); the policy can't operate
        # on stale observations of an actuated robot.
        safety_cfg = cfg.get("safety", {})
        self.estop_timeout_s = float(safety_cfg.get("estop_timeout_s", 0.5))
        self.sensor_timeout_s = float(safety_cfg.get("sensor_timeout_s", 0.2))
        # New contract: gate startup on per-topic first-message receipt
        # rather than wallclock grace. DDS discovery regularly takes >3s
        # on the Jetson; the old wallclock gate (startup_grace_s) raced
        # discovery and aborted healthy bringups.
        self.first_message_timeout_s = float(
            safety_cfg.get(
                "first_message_timeout_s",
                # Back-compat: if only the old key is present, honor it and
                # emit a deprecation warning so deploy.yaml migrations land.
                safety_cfg.get("startup_grace_s", 15.0),
            )
        )
        if "startup_grace_s" in safety_cfg and "first_message_timeout_s" not in safety_cfg:
            import warnings

            warnings.warn(
                "safety.startup_grace_s is deprecated; use safety.first_message_timeout_s",
                DeprecationWarning,
                stacklevel=2,
            )
        self._seen_estop = False
        self._seen_imu = False
        self._seen_joint_state = False

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

        # Two-policy mode switch (stand-v2 + v3b). Opt-in via deploy.yaml.
        # When disabled, ``self.session`` is the only policy and the
        # behaviour below is a straight passthrough — backwards-compatible
        # with all pre-mode-switch bringups.
        ms_cfg = cfg.get("policy", {}).get("mode_switch", {}) or {}
        self.mode_switch_enabled = bool(ms_cfg.get("enabled", False))
        self.stand_session = None
        self.walk_session = None
        self.mode_cfg: ModeSwitchCfg | None = None
        self.mode_state: State | None = None
        self.mode_ticks: int = 0
        self._last_action_stand = np.zeros(len(self.joint_order), dtype=np.float32)
        self._last_action_walk = np.zeros(len(self.joint_order), dtype=np.float32)
        if self.mode_switch_enabled:
            stand_path = Path(ms_cfg["stand_onnx_path"])
            walk_path = Path(ms_cfg["walk_onnx_path"])
            if not stand_path.is_file():
                raise FileNotFoundError(f"mode_switch.stand_onnx_path missing: {stand_path}")
            if not walk_path.is_file():
                raise FileNotFoundError(f"mode_switch.walk_onnx_path missing: {walk_path}")
            self.stand_session = ort.InferenceSession(
                str(stand_path), providers=["CPUExecutionProvider"]
            )
            self.walk_session = ort.InferenceSession(
                str(walk_path), providers=["CPUExecutionProvider"]
            )
            self.mode_cfg = ModeSwitchCfg(
                enter_walk_thresh=float(ms_cfg.get("enter_walk_thresh", 0.15)),
                enter_stand_thresh=float(ms_cfg.get("enter_stand_thresh", 0.05)),
                yaw_scale=float(ms_cfg.get("yaw_scale", 0.3)),
                transition_ticks=int(ms_cfg.get("transition_ticks", 25)),
            )
            self.mode_state, self.mode_ticks = initial_state(self.mode_cfg)
            logger.info(
                "mode_switch enabled: stand=%s walk=%s thresh=[%.2f,%.2f] trans=%d ticks",
                stand_path.name,
                walk_path.name,
                self.mode_cfg.enter_stand_thresh,
                self.mode_cfg.enter_walk_thresh,
                self.mode_cfg.transition_ticks,
            )

        # Reliability shield (Phase 4). Opt-in via deploy.yaml. When enabled,
        # the policy ONNX must emit a `latent` output (export with
        # ``--emit-latent``); the shield scores it every tick and returns a
        # blend weight toward the verified fallback. Disabled by default, so
        # every pre-shield bringup behaves exactly as before.
        rel_cfg = cfg.get("reliability", {}) or {}
        self.shield_enabled = bool(rel_cfg.get("enabled", False))
        self.shield = None
        self.shield_op = None
        self._shield_outputs = ["action"]
        self._shield_blend = 0.0
        self._shield_engagements = 0
        if self.shield_enabled:
            from phoenix.reliability.deploy import build_shield

            out_names = [o.name for o in self.session.get_outputs()]
            if "latent" not in out_names:
                # Fail closed at construction rather than at 50 Hz: a policy
                # with no latent cannot be monitored, and silently running it
                # unshielded is the failure mode this whole layer exists to
                # prevent.
                raise ValueError(
                    f"reliability.enabled but {onnx_path} emits {out_names} with no 'latent'. "
                    "Re-export with `python -m phoenix.sim2real.export --emit-latent`."
                )
            latent_dim = self.session.get_outputs()[out_names.index("latent")].shape[-1]
            # Only the artifact path is configurable. Every parameter that
            # changes the shield's behaviour — threshold, K, arming window,
            # ramps, release policy — is frozen inside the artifact, because
            # they were all calibrated together. Refit to change them.
            # Whitelist, not a denylist: a denylist silently accepts whatever
            # key someone invents next (it already missed `clear_persistence`),
            # and every one of these keys changes the calibrated behaviour.
            allowed = {"enabled", "artifact"}
            extra = set(rel_cfg) - allowed
            if extra:
                raise ValueError(
                    f"reliability config sets {sorted(extra)}; only {sorted(allowed)} are "
                    "settable here. Everything else is frozen in the artifact — change it by "
                    "refitting with scripts/reliability_fit_deploy.py."
                )
            self.shield, self.shield_op, shield_meta = build_shield(
                Path(rel_cfg["artifact"]),
                expected_dim=latent_dim if isinstance(latent_dim, int) else None,
            )
            self._shield_outputs = ["action", "latent"]
            logger.info(
                "reliability shield ENABLED: artifact=%s dim=%d trip=%.1f K=%d arming=%d ticks "
                "(calibrated: nominal episode FPR %.3f, %.0f%% of falls warned, "
                "%.2fs to decision / %.2fs to full fallback)",
                rel_cfg["artifact"],
                self.shield.dim,
                self.shield_op.trip_threshold,
                self.shield_op.trip_persistence,
                self.shield_op.arming_ticks,
                self.shield_op.nominal_episode_fpr,
                100.0 * self.shield_op.falls_warned,
                self.shield_op.median_lead_s,
                self.shield_op.median_full_fallback_lead_s,
            )
            ckpt = (shield_meta.get("provenance") or {}).get("checkpoint_sha256")
            logger.info("shield fit on checkpoint sha256=%s", ckpt)
            if self.mode_switch_enabled:
                # The shield scores self.session's latent, but mode-switch mode
                # commands from stand_session / walk_session instead. Refuse
                # rather than monitor a policy that is not driving the robot.
                raise ValueError(
                    "reliability.enabled is not supported together with policy.mode_switch; "
                    "the shield would score a policy that is not in control."
                )

        # Failure thresholds reused from the offline detector so the
        # on-robot abort and the sim replay flag the same regimes.
        self.thresholds = FailureThresholds()

        # Every threshold the gate ladder consults, bound once. Collecting
        # them here is what lets the ladder be a pure function; note that
        # pitch and roll are deliberately asymmetric (0.8 vs 0.6 rad).
        self._gate_config = GateConfig(
            max_runtime_s=self.max_runtime,
            estop_timeout_s=self.estop_timeout_s,
            sensor_timeout_s=self.sensor_timeout_s,
            first_message_timeout_s=self.first_message_timeout_s,
            pitch_rad=self.thresholds.pitch_rad,
            roll_rad=self.thresholds.roll_rad,
        )

        self._latest_imu = None
        self._latest_joint_state = None
        self._velocity_command = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(len(self.joint_order), dtype=np.float32)
        self._estopped = False
        self._abort_reason: str | None = None
        self._started_at = time.monotonic()
        self._started_ns = time.monotonic_ns()
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
        self.shield_pub = None
        if self.shield_enabled:
            self.shield_pub = self.node.create_publisher(
                Float64MultiArray, topics.get("shield_status", "/phoenix/shield"), qos
            )

        self.node.create_timer(1.0 / self.rate_hz, self._control_step)

    def _on_imu(self, msg):
        self._latest_imu = msg
        self._latest_imu_ns = time.monotonic_ns()
        self._seen_imu = True

    def _on_joint_state(self, msg):
        self._latest_joint_state = msg
        self._latest_joint_state_ns = time.monotonic_ns()
        self._seen_joint_state = True

    def _on_cmd_vel(self, msg):
        self._velocity_command = np.asarray(
            [msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32
        )

    def _on_estop(self, msg):
        self._latest_estop_ns = time.monotonic_ns()
        self._latest_estop_value = bool(msg.data)
        self._seen_estop = True
        if msg.data and not self._estopped:
            self._latch_abort("external_estop")

    def _latch_abort(self, reason: str) -> None:
        self._estopped = True
        self._abort_reason = reason
        logger.warning("ABORT: %s — holding stand pose.", reason)
        # Flush the parquet footer at abort time. A hard kill during the
        # post-abort default-pose hold (e.g. operator SIGKILL after
        # max_runtime latched) otherwise skips shutdown() and leaves a
        # footerless parquet.
        if self._logger is not None:
            try:
                self._logger.close()
            finally:
                self._logger = None

    def _build_snapshot(self, now_ns: int, elapsed_s: float) -> SensorSnapshot:
        """Sample every input the gate ladder is allowed to look at.

        The joint remap happens here because it is resolved by joint *name*;
        a positional copy would be a silent leg swap.

        During startup a sensor may not have arrived at all. Those payload
        fields are filled with neutral placeholders, which is safe because the
        startup gate (rank 3) fires strictly before the joint-state and IMU
        validity gates (rank 4) that would inspect them.
        """
        if self._latest_joint_state is not None:
            idx = self.joint_order.remap(list(self._latest_joint_state.name))
            joint_pos = np.asarray(self._latest_joint_state.position, dtype=np.float32)[idx]
            joint_vel = np.asarray(self._latest_joint_state.velocity, dtype=np.float32)[idx]
        else:
            joint_pos = np.zeros(len(self.joint_order.names), dtype=np.float32)
            joint_vel = np.zeros(len(self.joint_order.names), dtype=np.float32)

        if self._latest_imu is not None:
            ori = self._latest_imu.orientation
            ang = self._latest_imu.angular_velocity
            quat_xyzw = (ori.x, ori.y, ori.z, ori.w)
            ang_vel = (ang.x, ang.y, ang.z)
            roll, pitch, _yaw = _rpy_from_quat_xyzw(*quat_xyzw)
        else:
            quat_xyzw = (0.0, 0.0, 0.0, 1.0)
            ang_vel = (0.0, 0.0, 0.0)
            roll = pitch = 0.0

        return SensorSnapshot(
            now_ns=now_ns,
            elapsed_s=elapsed_s,
            node_started_ns=self._started_ns,
            seen_estop=self._seen_estop,
            seen_imu=self._seen_imu,
            seen_joint_state=self._seen_joint_state,
            estop_last_ns=self._latest_estop_ns,
            estop_value=self._latest_estop_value,
            imu_last_ns=self._latest_imu_ns,
            joint_state_last_ns=self._latest_joint_state_ns,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            quat_xyzw=quat_xyzw,
            ang_vel=ang_vel,
            roll=roll,
            pitch=pitch,
        )

    def _control_step(self):
        now_ns = time.monotonic_ns()
        elapsed_s = time.monotonic() - self._started_at

        # The ladder is a pure function so its composition — which gate wins,
        # and what each one publishes — is exhaustively testable without
        # rclpy. See tests/test_gate_ladder.py and docs/NATIVE_RUNTIME_PLAN.md.
        # It is also the parity oracle for the native C++ runtime.
        snap = self._build_snapshot(now_ns, elapsed_s)
        decision = evaluate_gates(
            snap, self._gate_config, already_latched=self._estopped
        )

        if decision.latches:
            self._latch_abort(decision.reason or "unknown_safety_gate")
        if decision.publishes_default:
            self._publish_default_pose()
        if decision.outcome is not Outcome.RUN_POLICY:
            # Post-abort silence is load-bearing: per-tick rebroadcast of
            # default_q walked the commanded pose against real posture →
            # motor fight → Jetson brownout. The bridge's 0.2 s command-stale
            # watchdog holds at last-measured q instead.
            return

        q = snap.joint_pos
        qd = snap.joint_vel
        pg = _projected_gravity_from_quat(*snap.quat_xyzw)
        base_ang_vel = np.asarray(snap.ang_vel, dtype=np.float32)
        base_lin_vel = np.zeros(3, dtype=np.float32)

        if self.mode_switch_enabled:
            target, action = self._compute_mode_switch_target(
                q=q,
                qd=qd,
                pg=pg,
                base_ang_vel=base_ang_vel,
                base_lin_vel=base_lin_vel,
            )
        else:
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

            outputs = self.session.run(self._shield_outputs, {"obs": obs})
            action = outputs[0][0]
            self._last_action = action.astype(np.float32, copy=False)

            target = self.default_q + self.action_scale * action

            if self.shield is not None:
                target = self._apply_shield(outputs[1][0], target)

        target = self._clip_to_limits(target, q)

        msg = self._float_msg()
        msg.data = target.astype(np.float64).tolist()
        self.cmd_pub.publish(msg)

        if self._logger is not None:
            self._log_step(
                q=q,
                qd=qd,
                action=action,
                quat_xyzw=snap.quat_xyzw,
                ang_vel=base_ang_vel,
            )
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

    def _compute_mode_switch_target(
        self,
        *,
        q: np.ndarray,
        qd: np.ndarray,
        pg: np.ndarray,
        base_ang_vel: np.ndarray,
        base_lin_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run both policies, advance the state machine, blend joint targets.

        The ``self._last_action`` field is kept in sync
        with whichever policy is currently active so external consumers
        (logger, tests) see a coherent last_action.
        """
        assert self.mode_cfg is not None
        assert self.stand_session is not None
        assert self.walk_session is not None

        # Each policy receives its own last_action so observations stay
        # on-distribution relative to training. The inactive policy's
        # last_action is held at zero (set on state entry; see below).
        proprio_stand = self.obs_builder.build(
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            projected_gravity=pg,
            velocity_command=self._velocity_command,
            joint_pos=q,
            joint_vel=qd,
            last_action=self._last_action_stand,
        )
        proprio_walk = self.obs_builder.build(
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            projected_gravity=pg,
            velocity_command=self._velocity_command,
            joint_pos=q,
            joint_vel=qd,
            last_action=self._last_action_walk,
        )

        def _prepare(obs: np.ndarray) -> np.ndarray:
            if self.obs_pad_zeros > 0:
                return np.concatenate(
                    [obs, np.zeros(self.obs_pad_zeros, dtype=np.float32)]
                ).reshape(1, -1)
            return obs.reshape(1, -1)

        stand_action = self.stand_session.run(["action"], {"obs": _prepare(proprio_stand)})[0][0]
        walk_action = self.walk_session.run(["action"], {"obs": _prepare(proprio_walk)})[0][0]
        stand_target = self.default_q + self.action_scale * stand_action
        walk_target = self.default_q + self.action_scale * walk_action

        prev_state = self.mode_state
        new_state, new_ticks, alpha = mode_step(
            prev_state, self.mode_ticks, self._velocity_command, self.mode_cfg
        )

        if prev_state is State.STAND:
            target = stand_target
            active = "stand"
        elif prev_state is State.WALK:
            target = walk_target
            active = "walk"
        elif prev_state is State.TRANS_TO_WALK:
            target = (1.0 - alpha) * stand_target + alpha * walk_target
            active = "walk"  # walk is accumulating last_action during the blend
        elif prev_state is State.TRANS_TO_STAND:
            target = (1.0 - alpha) * walk_target + alpha * stand_target
            active = "stand"
        else:  # pragma: no cover - defensive; prev_state comes from mode_step
            raise RuntimeError(f"unexpected mode state: {prev_state!r}")

        # Update per-policy last_action per the spec's contract. The policy
        # that is (or is becoming) active owns a fresh last_action; the
        # other is held at zero so its next entry starts on-distribution.
        if active == "stand":
            self._last_action_stand = stand_action.astype(np.float32, copy=False)
            self._last_action_walk = np.zeros_like(self._last_action_walk)
        else:
            self._last_action_walk = walk_action.astype(np.float32, copy=False)
            self._last_action_stand = np.zeros_like(self._last_action_stand)

        # Legacy ``self._last_action`` kept coherent with the active policy
        # so parquet logging stays on-distribution.
        self._last_action = self._last_action_stand if active == "stand" else self._last_action_walk

        self.mode_state = new_state
        self.mode_ticks = new_ticks

        # Return both the blended target and the active policy's action;
        # the action is what the caller logs to parquet.
        active_action = stand_action if active == "stand" else walk_action
        return target.astype(np.float32, copy=False), active_action.astype(np.float32, copy=False)

    def _apply_shield(self, latent: np.ndarray, learned_target: np.ndarray) -> np.ndarray:
        """Score the policy latent and blend toward the verified fallback.

        The fallback is the default stand pose — the same posture every abort
        path in this node commands, and the only controller here that is
        verified in the Simplex sense. ``blend`` ramps rather than snaps
        (see :mod:`phoenix.reliability.arbiter`) so the handoff cannot yank a
        moving robot into a stand in one tick.

        Note on ``last_action``: the policy keeps consuming *its own* action as
        ``last_action`` even while the fallback is partly in control, matching
        how the Isaac study scored the shield. Once the blend is non-zero the
        policy's observations are off-distribution regardless — which is the
        point, and why the arbiter's release path requires a sustained return
        to nominal scores rather than a single quiet tick.
        """
        decision = self.shield.step(latent)
        self._shield_blend = decision.blend
        if decision.blend <= 0.0:
            self._publish_shield_telemetry(decision)
            return learned_target

        if decision.blend >= 1.0 and self._shield_engagements == 0:
            self._shield_engagements += 1
            logger.warning(
                "SHIELD ENGAGED (score=%.1f > trip=%.1f): fallback in control.",
                decision.raw_score,
                self.shield_op.trip_threshold,
            )
        blended = (1.0 - decision.blend) * learned_target + decision.blend * self.default_q
        self._publish_shield_telemetry(decision)
        return blended.astype(np.float32, copy=False)

    def _publish_shield_telemetry(self, decision) -> None:
        """Publish ``[blend, raw_score, trip_threshold, state_code]`` for the lab log."""
        if self.shield_pub is None:
            return
        states = ("nominal", "handoff", "fallback", "recovering")
        msg = self._float_msg()
        # A non-finite score is a real signal (bad latent), so clamp only for
        # transport — the arbiter already saw the true value.
        score = decision.raw_score if np.isfinite(decision.raw_score) else -1.0
        msg.data = [
            float(decision.blend),
            float(score),
            float(self.shield_op.trip_threshold),
            float(states.index(decision.state.value)),
        ]
        self.shield_pub.publish(msg)

    def _clip_to_limits(self, target: np.ndarray, q: np.ndarray) -> np.ndarray:
        # Single source of truth lives in phoenix.sim2real.safety so the
        # bridge and the policy node provably share the slew-rate cap.
        return per_step_clip_array(target, q, MAX_DELTA_PER_STEP_RAD).astype(np.float32, copy=False)

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
