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
    p = argparse.ArgumentParser(
        description="Subscribe to GO2 ROS 2 topics and log to parquet."
    )
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
        state["cmd"] = np.asarray(
            [msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32
        )

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
                logger.info("Estop received — finalizing parquet.")
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
