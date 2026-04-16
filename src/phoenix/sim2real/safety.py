"""Pure-python safety predicates shared by the policy node + bridges.

The ROS nodes are heavy and can't be exercised in CI. The decision
logic ("should we treat estop as latched?", "is the joint state stale?",
"did the deadman release?") is all pull-out-able to plain functions
that take an injected clock — those live here so the unit tests in
``tests/test_safety.py`` cover the fail-closed semantics directly.
"""

from __future__ import annotations


def estop_is_active(
    *,
    last_msg_received_ns: int | None,
    latest_value: bool | None,
    now_ns: int,
    timeout_s: float,
) -> bool:
    """Return True if the estop should be treated as ASSERTED right now.

    Fail-closed semantics:

    * No message ever seen → True (publisher hasn't started).
    * Last message older than ``timeout_s`` → True (publisher died).
    * Last message says ``True`` → True.
    * Last message says ``False`` AND fresh → False.

    The bridge / policy node is expected to call this every control tick
    using its own monotonic clock.
    """
    if last_msg_received_ns is None or latest_value is None:
        return True
    age_s = (now_ns - last_msg_received_ns) / 1e9
    if age_s > timeout_s:
        return True
    return bool(latest_value)


def sensor_is_stale(
    *,
    last_msg_received_ns: int | None,
    now_ns: int,
    timeout_s: float,
) -> bool:
    """Return True if a sensor topic has gone silent past its timeout.

    Used by the policy node to abort if /imu/data or /joint_states stops
    publishing. Treats "never seen" as stale (fail-closed) but the policy
    node startup loop is allowed to no-op while waiting for the first
    message; callers should distinguish startup from steady state.
    """
    if last_msg_received_ns is None:
        return True
    return (now_ns - last_msg_received_ns) / 1e9 > timeout_s


def deadman_should_estop(
    *,
    last_input_ns: int | None,
    button_held: bool,
    now_ns: int,
    timeout_s: float,
) -> bool:
    """Translate a deadman button state + input freshness into an estop value.

    True means: publish ``estop=True``. Either the gamepad/wireless
    controller stopped sending (release of attention) or the operator
    released the deadman button.
    """
    if last_input_ns is None:
        return True
    if (now_ns - last_input_ns) / 1e9 > timeout_s:
        return True
    return not button_held


def per_step_clip(
    target: float, current: float, max_delta: float
) -> float:
    """Clip ``target`` to ``current ± max_delta``. Scalar form used by tests
    and small call sites; the policy node + bridge use
    :func:`per_step_clip_array` for the vectorized version (single source
    of truth for the slew-rate cap)."""
    if target > current + max_delta:
        return current + max_delta
    if target < current - max_delta:
        return current - max_delta
    return target


# Slew-rate limit applied to every joint command on every control tick.
# The policy is trained with action_scale=0.25 at 50 Hz; 0.175 rad/step is
# generous enough to not clip normal gait commands and tight enough to
# prevent a misbehaving policy from snapping a joint. Both the policy
# node and the lowcmd bridge import this constant so they cannot drift.
MAX_DELTA_PER_STEP_RAD: float = 0.175


def per_step_clip_array(target, current, max_delta: float = MAX_DELTA_PER_STEP_RAD):
    """Clip ``target`` to ``current ± max_delta`` element-wise.

    Pure numpy. Used by both ``ros2_policy_node._clip_to_limits`` and
    ``lowcmd_bridge_node._tick`` so the slew-rate cap is provably the
    same on both sides of the deploy stack.
    """
    import numpy as np

    target_arr = np.asarray(target)
    current_arr = np.asarray(current)
    return np.clip(target_arr, current_arr - max_delta, current_arr + max_delta)


def is_ready_to_command_motion(
    *,
    now_ns: int,
    estop_last_ns: int | None,
    estop_value: bool | None,
    estop_timeout_s: float,
    imu_last_ns: int | None,
    joint_state_last_ns: int | None,
    sensor_timeout_s: float,
) -> tuple[bool, str | None]:
    """Decide whether the policy node may run inference + publish a motion command.

    Returns ``(True, None)`` only when EVERY precondition holds:

    * an estop heartbeat has been received at least once,
    * that heartbeat is fresh (within ``estop_timeout_s``),
    * the heartbeat value is ``False`` (publisher asserts safe-to-run),
    * IMU and joint_state have each been received at least once,
    * both sensors are fresh (within ``sensor_timeout_s``).

    Returns ``(False, "<reason>")`` otherwise. The caller — typically
    ``ros2_policy_node._control_step`` — interprets the failure: during
    the startup grace window it stays silent (or publishes the safe
    default stand pose if it has heard from every publisher at least
    once); after the grace window it latches the abort.

    The point of this predicate being a free function is that it can be
    exhaustively tested in CI without rclpy or onnxruntime in scope.
    """
    if estop_last_ns is None or estop_value is None:
        return False, "estop_publisher_missing"
    if estop_is_active(
        last_msg_received_ns=estop_last_ns,
        latest_value=estop_value,
        now_ns=now_ns,
        timeout_s=estop_timeout_s,
    ):
        if estop_value:
            return False, "external_estop"
        return False, "estop_heartbeat_stale"
    if imu_last_ns is None or joint_state_last_ns is None:
        return False, "sensor_missing"
    if sensor_is_stale(
        last_msg_received_ns=imu_last_ns,
        now_ns=now_ns,
        timeout_s=sensor_timeout_s,
    ) or sensor_is_stale(
        last_msg_received_ns=joint_state_last_ns,
        now_ns=now_ns,
        timeout_s=sensor_timeout_s,
    ):
        return False, "sensor_stale"
    return True, None


__all__ = [
    "estop_is_active",
    "sensor_is_stale",
    "deadman_should_estop",
    "is_ready_to_command_motion",
    "per_step_clip",
    "per_step_clip_array",
    "MAX_DELTA_PER_STEP_RAD",
]
