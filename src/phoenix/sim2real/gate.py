"""The deploy-path safety gate ladder, as a pure function.

This module exists because the ladder it encodes previously lived inline in
``ros2_policy_node._control_step`` and therefore had **zero** test coverage:
every individual predicate in :mod:`phoenix.sim2real.safety` was well tested,
but the *composition* (which gate wins, what gets published when it fires,
and whether the abort latches) was not exercised by anything.

``evaluate_gates`` is a pure function of ``(snapshot, config, already_latched)``.
It touches no ROS, no ONNX, no clock and no ``self``, so it can be exhaustively
tested in CI, and it serves as the parity oracle for the native C++ port.

Ordering here is a literal transcription of the ladder as it ran in
``_control_step``. The one deliberate addition is the ``nan_in_imu`` gate: the
attitude gate below it compares ``abs(pitch) > threshold``, and ``abs(NaN)`` is
never greater than anything, so a non-finite quaternion previously slipped past
it and went on to poison the observation with NaN projected gravity.

The output contract is deliberately explicit. Two abort causes publish nothing
at all (``max_runtime`` and a previously-latched abort), and after any latch the
node must go silent permanently: re-broadcasting ``default_q`` every tick walked
the commanded pose against the robot's real posture, which caused motor fight
and a Jetson brownout. That silence is load-bearing behaviour, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .safety import is_ready_to_command_motion, startup_state


class Outcome(Enum):
    """What the caller must do with the tick.

    The enum encodes the full output contract: whether to publish, what to
    publish, and whether to latch. It is not merely an error code.
    """

    #: Abort already latched on a previous tick. Publish nothing, ever again.
    SILENT = "silent"
    #: Latch now, publish nothing. Used only by ``max_runtime``, matching the
    #: pre-existing behaviour in which the latch fell through to the
    #: already-latched early return without emitting a pose.
    LATCH_SILENT = "latch_silent"
    #: Publish the default stand pose, do not latch. The only repeatedly
    #: publishing state; used while waiting for every topic to be seen once.
    PUBLISH_DEFAULT = "publish_default"
    #: Latch now and publish the default stand pose exactly once.
    LATCH_AND_PUBLISH_DEFAULT = "latch_and_publish_default"
    #: All gates passed. Run inference, apply the shield, clip and publish.
    RUN_POLICY = "run_policy"


@dataclass(frozen=True)
class GateConfig:
    """Thresholds governing the ladder. Every one has exactly one owner."""

    max_runtime_s: float
    estop_timeout_s: float
    sensor_timeout_s: float
    first_message_timeout_s: float
    pitch_rad: float
    roll_rad: float


@dataclass(frozen=True)
class SensorSnapshot:
    """Everything the ladder is allowed to look at, sampled once per tick.

    ``joint_pos`` / ``joint_vel`` are already remapped into policy joint order
    by the caller, because the remap is resolved by joint *name* and a
    positional copy would be a silent leg swap.
    """

    now_ns: int
    elapsed_s: float
    node_started_ns: int

    seen_estop: bool
    seen_imu: bool
    seen_joint_state: bool

    estop_last_ns: int | None
    estop_value: bool | None
    imu_last_ns: int | None
    joint_state_last_ns: int | None

    joint_pos: np.ndarray
    joint_vel: np.ndarray

    quat_xyzw: tuple[float, float, float, float]
    ang_vel: tuple[float, float, float]

    roll: float
    pitch: float


@dataclass(frozen=True)
class GateDecision:
    """The ladder's verdict for one tick."""

    outcome: Outcome
    #: Populated only for the latching outcomes. ``None`` otherwise.
    reason: str | None = None

    @property
    def latches(self) -> bool:
        return self.outcome in (
            Outcome.LATCH_SILENT,
            Outcome.LATCH_AND_PUBLISH_DEFAULT,
        )

    @property
    def publishes_default(self) -> bool:
        return self.outcome in (
            Outcome.PUBLISH_DEFAULT,
            Outcome.LATCH_AND_PUBLISH_DEFAULT,
        )


def evaluate_gates(
    snapshot: SensorSnapshot,
    config: GateConfig,
    *,
    already_latched: bool,
) -> GateDecision:
    """Evaluate the ladder. First gate to fire wins and short-circuits.

    Preserves the original ordering exactly. The ``already_latched`` flag is
    passed in rather than read from node state so that the function stays pure.
    """
    # Rank 1: runtime watchdog. Latches but emits nothing, because the
    # original code latched here and then hit the already-latched early
    # return on the very same tick.
    if snapshot.elapsed_s > config.max_runtime_s and not already_latched:
        return GateDecision(Outcome.LATCH_SILENT, "max_runtime")

    # Rank 0/1 continued: an abort latched on any previous tick, including
    # the out-of-band external estop, which latches between ticks.
    if already_latched:
        return GateDecision(Outcome.SILENT)

    # Rank 3: startup gate.
    startup, startup_reason = startup_state(
        seen_estop=snapshot.seen_estop,
        seen_imu=snapshot.seen_imu,
        seen_joint_state=snapshot.seen_joint_state,
        node_started_ns=snapshot.node_started_ns,
        now_ns=snapshot.now_ns,
        first_message_timeout_s=config.first_message_timeout_s,
    )
    if startup == "waiting":
        return GateDecision(Outcome.PUBLISH_DEFAULT)
    if startup == "abort":
        return GateDecision(
            Outcome.LATCH_AND_PUBLISH_DEFAULT,
            startup_reason or "first_message_timeout_unknown",
        )

    # Rank 2/4: estop-chain integrity and sensor freshness.
    ok, reason = is_ready_to_command_motion(
        now_ns=snapshot.now_ns,
        estop_last_ns=snapshot.estop_last_ns,
        estop_value=snapshot.estop_value,
        estop_timeout_s=config.estop_timeout_s,
        imu_last_ns=snapshot.imu_last_ns,
        joint_state_last_ns=snapshot.joint_state_last_ns,
        sensor_timeout_s=config.sensor_timeout_s,
    )
    if not ok:
        return GateDecision(Outcome.LATCH_AND_PUBLISH_DEFAULT, reason or "unknown_safety_gate")

    # Rank 4: joint-state validity.
    if not np.all(np.isfinite(snapshot.joint_pos)) or not np.all(np.isfinite(snapshot.joint_vel)):
        return GateDecision(Outcome.LATCH_AND_PUBLISH_DEFAULT, "nan_in_joint_state")

    # Rank 4: IMU validity. Must precede the attitude gate, which cannot
    # fire on NaN because abs(NaN) > threshold is False.
    if not np.all(np.isfinite(snapshot.quat_xyzw)) or not np.all(np.isfinite(snapshot.ang_vel)):
        return GateDecision(Outcome.LATCH_AND_PUBLISH_DEFAULT, "nan_in_imu")

    # Rank 6: attitude abort.
    if abs(snapshot.pitch) > config.pitch_rad or abs(snapshot.roll) > config.roll_rad:
        return GateDecision(
            Outcome.LATCH_AND_PUBLISH_DEFAULT,
            f"attitude pitch={snapshot.pitch:.2f} roll={snapshot.roll:.2f}",
        )

    # Rank 8: nothing seized the output; the policy may command motion.
    return GateDecision(Outcome.RUN_POLICY)
