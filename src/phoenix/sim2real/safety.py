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


def per_step_clip(target: float, current: float, max_delta: float) -> float:
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

    Pure numpy. Retained for the callers that still want a plain position
    slew cap (e.g. FIX_STAND's own per-tick step bound). The final actuator
    safety boundary (``ActuatorGate`` POLICY mode, ``SafetyFilter``) uses
    :func:`torque_limited_target_array` instead -- see that function's
    docstring for why the position-only clip was replaced there.
    """
    import numpy as np

    target_arr = np.asarray(target)
    current_arr = np.asarray(current)
    return np.clip(target_arr, current_arr - max_delta, current_arr + max_delta)


#: Actions leaving the policy network are clamped to this range before being
#: scaled and added to the default pose, matching training's
#: ``clip_actions=1.0`` (evaluate.py:223, fine_tune.py:265, reconstruct.py:224,
#: ppo_runner.py:124). The 2026-09-21 stage F1 failure traced to a raw action
#: of -7.2 on RR_thigh reaching the bridge with no clamp at all; clamp here,
#: at the earliest point after inference, not only downstream in the actuator
#: gate, so an unclamped action never enters the target-scaling arithmetic.
ACTION_CLIP: float = 1.0


def clip_actions(action, limit: float = ACTION_CLIP):
    """Clamp a raw policy action to ``[-limit, limit]`` element-wise.

    Returns ``(clipped, saturated)`` where ``saturated`` is a bool array,
    True for every element that hit either bound, so the caller can log a
    per-tick and rolling saturation rate (a saturating policy heading toward
    the clamp on every tick is a signal worth surfacing even though the
    clamp itself keeps the robot safe).
    """
    import numpy as np

    arr = np.asarray(action, dtype=np.float64)
    clipped = np.clip(arr, -limit, limit)
    saturated = np.abs(arr) > limit
    return clipped, saturated


def torque_limited_target_array(
    target,
    q_measured,
    dq_measured,
    kp,
    kd,
    torque_limit,
):
    """Clip ``target`` so the PD torque it implies cannot exceed ``torque_limit``.

    Replaces the measured-q +-0.175 rad/step slew clip as the deploy-time
    safety net. That clip was a POSITION bound with no relationship to what
    the motor could actually do: at kp=25 it acted as an approximately 4.4 N m
    torque cap on every joint, well under the real actuator limits (23.5 N m
    hip/thigh, 45.43 N m calf -- see ``go2_model.JOINT_TORQUE_LIMITS_NM``),
    and it clipped identically whether the joint was loaded or swinging free.
    2026-09-21 stage F1: the loaded diagonal's calves pinned at the clip and
    never rose while the unloaded diagonal flailed, because the safety net
    could not tell "under load, needs more torque" from "runaway target".

    The PD law the firmware executes is ``tau = Kp*(target - q) - Kd*dq``.
    Solving for the target range that keeps ``|tau| <= torque_limit`` at the
    MEASURED ``q``/``dq``::

        target_lo = q + (-torque_limit + Kd*dq) / Kp
        target_hi = q + ( torque_limit + Kd*dq) / Kp

    ``target`` is clipped into ``[target_lo, target_hi]``. Where ``Kp`` is
    zero (damping: no position term, so no position clip is meaningful here)
    the element passes through unclipped by this function; callers in a
    damping mode do not reach this path in practice because damping holds
    the measured position directly.

    This is a per-step bound, same as the clip it replaces, computed fresh
    every tick from the current measured state -- it is NOT a substitute for
    the hard position-limit / illegal-target abort band, which stays in
    place and fires only on a target that could not come from a legal
    measured state (see ``go2_model.LIMIT_ABORT_BAND_RAD``). A policy action
    clamped to [-1,1] and scaled by 0.25 can deviate at most 0.25 rad from
    the default pose, well inside the abort band in nominal operation, so
    this torque clip -- not the abort band -- is what shapes a nominal
    policy's authority near a limit.
    """
    import numpy as np

    target_arr = np.asarray(target, dtype=np.float64)
    q = np.asarray(q_measured, dtype=np.float64)
    dq = np.asarray(dq_measured, dtype=np.float64)
    kp_arr = np.broadcast_to(np.asarray(kp, dtype=np.float64), target_arr.shape).copy()
    kd_arr = np.broadcast_to(np.asarray(kd, dtype=np.float64), target_arr.shape)
    limit_arr = np.broadcast_to(np.asarray(torque_limit, dtype=np.float64), target_arr.shape)

    has_kp = kp_arr > 0.0
    safe_kp = np.where(has_kp, kp_arr, 1.0)
    lo = q + (-limit_arr + kd_arr * dq) / safe_kp
    hi = q + (limit_arr + kd_arr * dq) / safe_kp
    clipped = np.clip(target_arr, lo, hi)
    return np.where(has_kp, clipped, target_arr)


def expected_torque(target, q_measured, dq_measured, kp, kd):
    """``tau = Kp*(target - q) - Kd*dq`` element-wise. Pure numpy; for telemetry."""
    import numpy as np

    target_arr = np.asarray(target, dtype=np.float64)
    q = np.asarray(q_measured, dtype=np.float64)
    dq = np.asarray(dq_measured, dtype=np.float64)
    return np.asarray(kp, dtype=np.float64) * (target_arr - q) - np.asarray(
        kd, dtype=np.float64
    ) * dq


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


def startup_state(
    *,
    seen_estop: bool,
    seen_imu: bool,
    seen_joint_state: bool,
    node_started_ns: int,
    now_ns: int,
    first_message_timeout_s: float,
) -> tuple[str, str | None]:
    """Classify the node's startup state based on per-topic first-message seen flags.

    Returns one of:
      ("waiting", None) — at least one required first message still pending,
                          within the configured timeout. The node should hold
                          the default pose and refuse policy inference.
      ("ready",   None) — all three required first messages have arrived at
                          least once. Caller transitions to normal
                          freshness-based gating via is_ready_to_command_motion.
      ("abort",   reason) — the timeout expired with one or more topics still
                            missing. ``reason`` is ``first_message_timeout_<csv>``
                            where the CSV lists missing topics in the stable
                            order (estop, imu, joint_state).

    Fail-closed semantics:
      * All three seen → "ready" regardless of elapsed time.
      * Before timeout + any missing → "waiting".
      * After timeout + any missing → "abort".
      * After timeout + all seen → "ready" (would already have returned
        "ready" in the first clause; kept explicit for clarity).
    """
    if seen_estop and seen_imu and seen_joint_state:
        return ("ready", None)

    elapsed_s = (now_ns - node_started_ns) / 1e9
    if elapsed_s <= first_message_timeout_s:
        return ("waiting", None)

    missing: list[str] = []
    if not seen_estop:
        missing.append("estop")
    if not seen_imu:
        missing.append("imu")
    if not seen_joint_state:
        missing.append("joint_state")
    return ("abort", "first_message_timeout_" + ",".join(missing))


__all__ = [
    "estop_is_active",
    "sensor_is_stale",
    "deadman_should_estop",
    "is_ready_to_command_motion",
    "per_step_clip",
    "per_step_clip_array",
    "clip_actions",
    "expected_torque",
    "torque_limited_target_array",
    "ACTION_CLIP",
    "MAX_DELTA_PER_STEP_RAD",
    "startup_state",
]
