"""Log real-robot rollouts to Apache Parquet for later sim replay.

Each row captures one control step (50 Hz). The schema is intentionally
stable: the :mod:`phoenix.replay` module reads it back into Isaac Sim.

.. code-block:: text

    step (int64)
    timestamp_s (float64)
    base_pos (list<float32>[3])
    base_quat (list<float32>[4])           # (x,y,z,w)
    base_lin_vel_body (list<float32>[3])
    base_ang_vel_body (list<float32>[3])
    joint_pos (list<float32>[12])
    joint_vel (list<float32>[12])
    command_vel (list<float32>[3])
    action (list<float32>[12])             # raw policy output, unscaled
    contact_forces (list<float32>[4])      # per-foot normal force
    failure_flag (bool)
    failure_mode (str, nullable)
    odom_valid (bool)                      # see note below
    capture_source (str)                   # "sim" | "hardware" | "unknown"
    base_lin_vel_source (str)              # provenance of the LOGGED base_lin_vel_body
    obs_base_lin_vel_source (str)          # provenance of what the POLICY was fed
    contact_forces_units (str)             # "newtons" | "raw_counts_uncalibrated"
    sim_termination_terms (str, nullable)  # simulator GROUND TRUTH, null on hardware
    sim_termination_index (int64, nullable)
    sim_termination_time_s (float64, nullable)
    failure_onset_source (str, nullable)   # what failure_flag keys off

Provenance is in the schema on purpose. A consumer must never have to infer
where a row came from by looking at the file name:

``capture_source`` is ``"sim"`` for simulator captures, ``"hardware"`` for
anything written off the real robot, and ``"unknown"`` for a writer that has
not been taught the difference. ``"unknown"`` is the dataclass default, so an
un-updated call site is visibly unlabelled rather than quietly mislabelled.

``contact_forces_units``: sim captures (Isaac Lab contact sensor) are true
Newtons, ``"newtons"``. Real-robot captures are
``unitree_go/msg/LowState.foot_force`` passed through by
:func:`phoenix.sim2real.telemetry.foot_force_to_array`, raw int16 sensor
counts with no documented calibration and no documented per-index leg
ordering, ``"raw_counts_uncalibrated"``. Do not mix the two, and do not call
the hardware numbers Newtons.

``base_lin_vel_source`` records how the LOGGED ``base_lin_vel_body`` column
was obtained: ``sim_ground_truth``, ``odom:body_passthrough``,
``odom:rotated_from_world``, ``absent``, ``stale``,
``unrecognized_child_frame:<id>``, ``unmeasured``, or ``unknown``.

``obs_base_lin_vel_source`` records, separately, what the POLICY consumed in
observation dims 0..2 on that step: ``sim_ground_truth``,
``odom:<frame resolution>``, or ``zeros:operator_selected``. The two differ
on purpose. A hardware run configured with
``observation.base_lin_vel_source: zeros`` logs the real odometry velocity in
``base_lin_vel_body`` while feeding the policy zeros, and these two columns
are what makes that visible instead of leaving the capture looking as though
the policy saw the logged value.

``odom_valid`` note: real-robot captures set this True only when
``/utlidar/robot_odom`` had published a fresh message this step; when False,
``base_pos`` and ``base_lin_vel_body`` are the zero fallback rather than a
real zero reading (that topic is optional/LiDAR-stack-dependent and may be
absent). It covers the POSE only. A fresh message whose twist could not be
resolved into the body frame still has a real ``base_pos`` and therefore
``odom_valid=True``, while ``base_lin_vel_body`` is the zero fallback and
``base_lin_vel_source`` names the reason. Check that column before trusting
the velocity. Sim captures leave ``odom_valid`` at the dataclass default
``True`` because their base_pos/vel are ground truth. Use ``capture_source``,
not ``odom_valid``, to tell sim from hardware.

Simulator ground truth versus detector measurement. ``failure_flag`` and
``failure_mode`` are the :class:`phoenix.real_world.FailureDetector`'s
OPINION. In simulation the episode's termination is GROUND TRUTH, and the two
do not agree: the detector misses a substantial share of genuine falls
(nothing in its ontology corresponds to Isaac's ``base_contact``, see the
FailureDetector module docstring). A capture that carried only the detector's
columns left a consumer unable to tell "the detector said not this mode" from
"the detector said nothing at all", so a mode-subset filter silently dropped
real failures into the wrong bucket.

The simulator columns fix that, and they are null on hardware because no such
ground truth exists there:

* ``sim_termination_terms``: the Isaac termination terms that fired, joined
  with ``"|"`` (for example ``"base_contact"`` or ``"base_contact|time_out"``).
  Null for a hardware capture, and null for a sim writer that has not been
  taught to record it, which is visibly different from ``""`` (terminated
  with no term).
* ``sim_termination_index``: row index of the simulator's terminal step.
* ``sim_termination_time_s``: that row's ``timestamp_s``.
* ``failure_onset_source``: ``"detector"`` when ``failure_flag`` is the
  detector's verdict, ``"simulator_termination"`` when it keys off the
  simulator's terminal step instead. Null when the writer has not declared it.

``failure_mode`` stays strictly the detector's label and stays null when the
detector did not fire, even on a row the simulator considers terminal. A mode
filter must therefore never treat a null ``failure_mode`` as "not that mode"
on a trajectory whose ``sim_termination_terms`` is non-null; that row is an
unlabelled real failure, not a nominal one.

``base_pos`` note, real-robot captures: ``/utlidar/robot_odom`` has its
origin at the boot pose (docs/go2_field_notes.md section 3), so ``base_pos``
is displacement from wherever the robot booted. ``base_pos[2]`` is NOT height
above the floor and must not be used as one; the logger stores the raw odom
value plus ``odom_valid`` and derives nothing from it.

Writer uses row-group buffering to keep memory bounded on long rollouts.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("phoenix.real_world.trajectory_logger")

#: ``capture_source`` values. See the module docstring.
CAPTURE_SOURCE_SIM = "sim"
CAPTURE_SOURCE_HARDWARE = "hardware"
CAPTURE_SOURCE_UNKNOWN = "unknown"


@dataclass
class TrajectoryStep:
    step: int
    timestamp_s: float
    base_pos: np.ndarray  # (3,)
    base_quat: np.ndarray  # (4,) xyzw
    base_lin_vel_body: np.ndarray  # (3,)
    base_ang_vel_body: np.ndarray  # (3,)
    joint_pos: np.ndarray  # (12,)
    joint_vel: np.ndarray  # (12,)
    command_vel: np.ndarray  # (3,)
    action: np.ndarray  # (12,)
    contact_forces: np.ndarray  # (4,)
    failure_flag: bool = False
    failure_mode: str | None = None
    # True unless a capture path explicitly marks base_pos/base_lin_vel_body
    # as an unmeasured fallback (real-robot odometry absent/stale). Defaults
    # True because every pre-existing call site (sim captures) already has
    # ground-truth position data, only ros2_policy_node's real-robot path
    # sets this False.
    odom_valid: bool = True
    # Provenance. Every one defaults to "unknown" rather than to a plausible
    # value: an unlabelled row is a bug someone can see, a wrongly-labelled
    # one is not.
    capture_source: str = CAPTURE_SOURCE_UNKNOWN
    base_lin_vel_source: str = "unknown"
    obs_base_lin_vel_source: str = "unknown"
    contact_forces_units: str = "unknown"
    # Simulator ground truth, per-row constant for the whole trajectory. All
    # four default to None: a hardware capture has no simulator termination to
    # report and must not fabricate one, and a sim writer that has not been
    # taught to fill them says "unknown" by being null rather than by
    # inventing a value. See the module docstring.
    sim_termination_terms: str | None = None
    sim_termination_index: int | None = None
    sim_termination_time_s: float | None = None
    failure_onset_source: str | None = None


_SCHEMA = pa.schema(
    [
        ("step", pa.int64()),
        ("timestamp_s", pa.float64()),
        ("base_pos", pa.list_(pa.float32(), 3)),
        ("base_quat", pa.list_(pa.float32(), 4)),
        ("base_lin_vel_body", pa.list_(pa.float32(), 3)),
        ("base_ang_vel_body", pa.list_(pa.float32(), 3)),
        ("joint_pos", pa.list_(pa.float32(), 12)),
        ("joint_vel", pa.list_(pa.float32(), 12)),
        ("command_vel", pa.list_(pa.float32(), 3)),
        ("action", pa.list_(pa.float32(), 12)),
        ("contact_forces", pa.list_(pa.float32(), 4)),
        ("failure_flag", pa.bool_()),
        ("failure_mode", pa.string()),
        ("odom_valid", pa.bool_()),
        ("capture_source", pa.string()),
        ("base_lin_vel_source", pa.string()),
        ("obs_base_lin_vel_source", pa.string()),
        ("contact_forces_units", pa.string()),
        ("sim_termination_terms", pa.string()),
        ("sim_termination_index", pa.int64()),
        ("sim_termination_time_s", pa.float64()),
        ("failure_onset_source", pa.string()),
    ]
)


class TrajectoryLogger:
    """Buffered Parquet writer. Call :meth:`append` then :meth:`close`.

    Use as a context manager to guarantee flush on exceptions::

        with TrajectoryLogger("rollout.parquet") as log:
            for step in rollout():
                log.append(step)
    """

    def __init__(self, path: str | Path, row_group_size: int = 512) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.row_group_size = row_group_size
        self._writer: pq.ParquetWriter | None = None
        self._buffer: list[dict] = []
        self._rows_written = 0

    def __enter__(self) -> TrajectoryLogger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def rows_written(self) -> int:
        return self._rows_written + len(self._buffer)

    def append(self, step: TrajectoryStep) -> None:
        row = asdict(step)
        # dataclasses.asdict doesn't descend into numpy arrays cleanly, so we
        # convert array fields by name.
        for k in (
            "base_pos",
            "base_quat",
            "base_lin_vel_body",
            "base_ang_vel_body",
            "joint_pos",
            "joint_vel",
            "command_vel",
            "action",
            "contact_forces",
        ):
            row[k] = np.asarray(row[k], dtype=np.float32).tolist()
        self._buffer.append(row)
        if len(self._buffer) >= self.row_group_size:
            self._flush()

    def close(self) -> None:
        if self._buffer:
            self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=_SCHEMA)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, _SCHEMA, compression="zstd")
        self._writer.write_table(table)
        self._rows_written += len(self._buffer)
        self._buffer.clear()


def _parse_standalone_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Subscribe to GO2 ROS 2 topics and log to parquet.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rate-hz", type=float, default=50.0)
    p.add_argument("--flush-on-estop", action="store_true")
    p.add_argument("--estop-topic", type=str, default="/phoenix/estop")
    return p.parse_args(argv)


def _standalone_main(argv: list[str] | None = None) -> int:  # pragma: no cover - requires ROS 2
    """Standalone subscriber: log one row per control tick from live topics.

    Complements the in-node logger in ``sim2real.ros2_policy_node``. Useful
    when capturing teleop or no-policy rollouts, or when you want a
    separate parquet unlinked from the policy process lifetime.
    """
    args = _parse_standalone_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Imu, JointState
    from std_msgs.msg import Bool

    rclpy.init()
    node = Node("phoenix_trajectory_logger")
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

    state = {"imu": None, "joint": None, "cmd": np.zeros(3, dtype=np.float32), "estop": False}

    def _on_imu(msg):
        state["imu"] = msg

    def _on_joint(msg):
        state["joint"] = msg

    def _on_cmd(msg):
        state["cmd"] = np.asarray([msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32)

    def _on_estop(msg):
        if msg.data:
            state["estop"] = True

    node.create_subscription(Imu, "/imu/data", _on_imu, qos)
    node.create_subscription(JointState, "/joint_states", _on_joint, qos)
    node.create_subscription(Twist, "/cmd_vel", _on_cmd, qos)
    node.create_subscription(Bool, args.estop_topic, _on_estop, qos)

    started = time.monotonic()
    step_idx = 0

    with TrajectoryLogger(args.output) as log:

        def _tick():
            nonlocal step_idx
            if state["imu"] is None or state["joint"] is None:
                return
            if args.flush_on_estop and state["estop"]:
                logger.info("Estop received, finalizing parquet.")
                rclpy.shutdown()
                return
            js = state["joint"]
            imu = state["imu"]
            q = np.asarray(js.position, dtype=np.float32)[:12]
            qd = np.asarray(js.velocity, dtype=np.float32)[:12]
            quat = np.asarray(
                [imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w],
                dtype=np.float32,
            )
            ang = np.asarray(
                [imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z],
                dtype=np.float32,
            )
            log.append(
                TrajectoryStep(
                    step=step_idx,
                    timestamp_s=time.monotonic() - started,
                    base_pos=np.zeros(3, dtype=np.float32),
                    base_quat=quat,
                    base_lin_vel_body=np.zeros(3, dtype=np.float32),
                    base_ang_vel_body=ang,
                    joint_pos=q,
                    joint_vel=qd,
                    command_vel=state["cmd"],
                    action=np.zeros(12, dtype=np.float32),
                    contact_forces=np.zeros(4, dtype=np.float32),
                    failure_flag=False,
                    failure_mode=None,
                    # This standalone subscriber reads only /imu/data,
                    # /joint_states and /cmd_vel. base_pos, base_lin_vel_body
                    # and contact_forces above are structural zeros, not
                    # measurements, and the row says so.
                    odom_valid=False,
                    capture_source=CAPTURE_SOURCE_HARDWARE,
                    base_lin_vel_source="unmeasured",
                    obs_base_lin_vel_source="unmeasured",
                    contact_forces_units="unmeasured",
                    # No policy and no detector run here, so failure_flag is a
                    # structural False rather than any verdict.
                    failure_onset_source=None,
                )
            )
            step_idx += 1

        node.create_timer(1.0 / args.rate_hz, _tick)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    logger.info("Wrote %d rows to %s", step_idx, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_standalone_main())
