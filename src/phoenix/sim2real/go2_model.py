"""GO2 joint model constants the deploy path is allowed to rely on, with provenance.

Every number in this module is transcribed from a named source, and each source
is recorded next to the value it supplied. Nothing here is tuned. If one of these
tables disagrees with the robot, fix the source reference and the table together;
never edit a value to make a gate pass.

Three facts live here because the final actuator gate
(:mod:`phoenix.sim2real.actuator_gate`) and the preflight both need them without
importing ROS:

* the joint ORDERS: the policy's training order and the Unitree motor order, plus
  a check that ``PHOENIX_FOR_MOTOR`` really maps one onto the other by name;
* the training nominal pose (``default_joint_pos``);
* the hard joint-position limits of the GO2 leg joints.

Joint-position limits, and why these and not the simulator's soft limits
------------------------------------------------------------------------
The values are the ``<limit lower= upper=>`` attributes of Unitree's own GO2
description, ``unitreerobotics/unitree_ros`` at
``robots/go2_description/urdf/go2_description.urdf``. The file was fetched on
2026-09-12; the latest upstream commit touching that path was
``a3b70cae6fd4a82c0e1ece633d5c6f97e88c9d76`` (2025-06-19) and the fetched file
hashed to ``sha256 7d19fe48e2e689ee1a032ab99f2a4a8b671d87e73de48d3e65811682a5b48b9e``.
Two independent downstream copies of the GO2 URDF (different byte content,
different exporters' formatting) parse to exactly the same twelve limit pairs.

Isaac Lab trains with ``soft_joint_pos_limit_factor=0.9`` on top of the asset's
limits. Those soft limits are NOT used as the actuator envelope, deliberately:
Unitree's own low-level stand example folds the calf to ``-2.65`` rad
(``unitree_ros2/example/src/src/go2/go2_stand_example.cpp``), which is inside the
hard calf range ``[-2.7227, -0.83776]`` but outside the 0.9 soft range
``[-2.628, -0.932]``. A soft-limit envelope would therefore abort on a robot
lying in Unitree's own folded pose, before any policy ran. The hard range is the
mechanical authority; the soft range is a training reward shaping choice.

The abort band
--------------
:data:`LIMIT_ABORT_BAND_RAD` is not a robot measurement and does not pretend to
be. It is the per-step slew cap :data:`phoenix.sim2real.safety.MAX_DELTA_PER_STEP_RAD`,
reused, for one reason that follows from the order of operations in the final
gate: a target is slew-clipped to within one cap of the MEASURED joint position
before its absolute limit is checked. So if the measured position is inside the
hard range, a slew-clipped target can overshoot a limit by at most one cap, which
is the small, legitimate case that gets clipped to the limit. Anything further out
cannot come from a legal measured state, so it is treated as a command or a state
that must abort, not as something to clamp quietly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .safety import MAX_DELTA_PER_STEP_RAD

#: The policy's joint order: Isaac Lab's GO2 articulation order, grouped by joint
#: type. Identical to ``joint_order`` in every ``configs/sim2real/deploy*.yaml``
#: and to ``kPolicyJointNames`` in ``runtime/phoenix_core/src/joint_map.cpp``.
POLICY_JOINT_ORDER: tuple[str, ...] = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)

#: Unitree ``motor_cmd[k]`` / ``motor_state[k]`` order, grouped by LEG with FR
#: first. Source: ``unitree_ros2/example/src/include/common/motor_crc.h``,
#: ``FR_0 = 0``, ``FL_0 = 3``, ``RR_0 = 6``, ``RL_0 = 9`` with hip/thigh/calf as
#: ``_0/_1/_2``.
UNITREE_MOTOR_ORDER: tuple[str, ...] = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)

#: Training nominal pose. Source: Isaac Lab ``UNITREE_GO2_CFG.init_state.joint_pos``
#: in ``source/isaaclab_assets/isaaclab_assets/robots/unitree.py``, written there
#: as regex patterns and expanded here. It is the action offset AND the joint
#: observation reference at deploy (``use_default_offset=True``).
TRAINING_DEFAULT_JOINT_POS: dict[str, float] = {
    "FL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RL_hip_joint": 0.1,
    "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RL_calf_joint": -1.5,
    "RR_calf_joint": -1.5,
}

#: Where :data:`JOINT_POSITION_LIMITS_RAD` came from. Recorded into every
#: hardware telemetry manifest so a run names the envelope it was held to.
JOINT_LIMITS_PROVENANCE: dict[str, str] = {
    "repository": "https://github.com/unitreerobotics/unitree_ros",
    "path": "robots/go2_description/urdf/go2_description.urdf",
    "latest_commit_touching_path": "a3b70cae6fd4a82c0e1ece633d5c6f97e88c9d76",
    "commit_date": "2025-06-19",
    "fetched": "2026-09-12",
    "fetched_file_sha256": "7d19fe48e2e689ee1a032ab99f2a4a8b671d87e73de48d3e65811682a5b48b9e",
    "field": "joint/limit@lower, joint/limit@upper (radians)",
    "kind": "hard URDF limits, not Isaac Lab soft limits",
}

#: Hard joint-position limits in radians, ``(lower, upper)``, transcribed verbatim
#: from the URDF named in :data:`JOINT_LIMITS_PROVENANCE`. Front and rear thighs
#: differ; that asymmetry is in the source, not a typo.
JOINT_POSITION_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "FL_hip_joint": (-1.0472, 1.0472),
    "FR_hip_joint": (-1.0472, 1.0472),
    "RL_hip_joint": (-1.0472, 1.0472),
    "RR_hip_joint": (-1.0472, 1.0472),
    "FL_thigh_joint": (-1.5708, 3.4907),
    "FR_thigh_joint": (-1.5708, 3.4907),
    "RL_thigh_joint": (-0.5236, 4.5379),
    "RR_thigh_joint": (-0.5236, 4.5379),
    "FL_calf_joint": (-2.7227, -0.83776),
    "FR_calf_joint": (-2.7227, -0.83776),
    "RL_calf_joint": (-2.7227, -0.83776),
    "RR_calf_joint": (-2.7227, -0.83776),
}

#: See the module docstring: a target beyond a hard limit by more than one slew
#: cap cannot come from a legal measured state, so it aborts instead of clipping.
LIMIT_ABORT_BAND_RAD: float = MAX_DELTA_PER_STEP_RAD

#: Per-joint continuous torque limit, in newton-meters, used to bound the
#: commanded POSITION target so the resulting PD torque
#: ``tau = Kp*(target-q) - Kd*dq`` cannot exceed the motor's rating.
#:
#: Hip and thigh joints share the GO2's standard actuator: 23.5 N m continuous,
#: the same number Isaac Lab used for every joint before the calf fix below.
#: The calf joint uses a stronger actuator: 45.43 N m / 15.7 rad/s. Source:
#: Isaac Lab PR #7479 (merged 2026-09-04, closed the 2026-03-16 UNITREE_GO2_CFG
#: bug that gave every joint including the calf the weaker 23.5 N m / 30 rad/s
#: DCMotor group -- see docs/hardware/ and
#: Projects/go2-phoenix/ANALYSIS_2026-09-24_why-phoenix-failed-vs-proven-stacks.md
#: row "Calf actuator"). This replaces the pre-2026-09-24 deploy-time safety net
#: (a measured-q +-0.175 rad/step slew clip, an approximately 4.4 N m cap at
#: kp=25 that starved loaded legs on 2026-09-21 stage F1) with the actual motor
#: envelope.
JOINT_TORQUE_LIMITS_NM: dict[str, float] = {
    "FL_hip_joint": 23.5,
    "FR_hip_joint": 23.5,
    "RL_hip_joint": 23.5,
    "RR_hip_joint": 23.5,
    "FL_thigh_joint": 23.5,
    "FR_thigh_joint": 23.5,
    "RL_thigh_joint": 23.5,
    "RR_thigh_joint": 23.5,
    "FL_calf_joint": 45.43,
    "FR_calf_joint": 45.43,
    "RL_calf_joint": 45.43,
    "RR_calf_joint": 45.43,
}


def torque_limits_in_order(order: Sequence[str]) -> np.ndarray:
    """Return the per-joint torque limit (N m) as a float64 array in ``order``.

    Raises ``KeyError`` for a name that is not a GO2 leg joint, matching
    :func:`limits_in_order`.
    """
    return np.asarray([JOINT_TORQUE_LIMITS_NM[n] for n in order], dtype=np.float64)


#: Unitree's low-level stand example poses, in Unitree motor order
#: (``go2_stand_example.cpp`` ``target_pos_1_`` folded and ``target_pos_2_``
#: standing). Used only as sanity fixtures: a limit table that rejected Unitree's
#: own postures would be wrong.
UNITREE_EXAMPLE_FOLDED_POSE: tuple[float, ...] = (
    0.0, 1.36, -2.65, 0.0, 1.36, -2.65, -0.2, 1.36, -2.65, 0.2, 1.36, -2.65,
)  # fmt: skip
UNITREE_EXAMPLE_STAND_POSE: tuple[float, ...] = (
    0.0, 0.67, -1.3, 0.0, 0.67, -1.3, 0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
)  # fmt: skip


def limits_in_order(order: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lower, upper)`` float64 arrays in the given joint-name order.

    Raises ``KeyError`` for a name that is not a GO2 leg joint, so a typo in a
    deploy config cannot silently produce an unlimited joint.
    """
    lower = np.asarray([JOINT_POSITION_LIMITS_RAD[n][0] for n in order], dtype=np.float64)
    upper = np.asarray([JOINT_POSITION_LIMITS_RAD[n][1] for n in order], dtype=np.float64)
    return lower, upper


def verify_joint_model(
    joint_order: Sequence[str],
    phoenix_for_motor: Sequence[int],
    motor_names: Sequence[str] = UNITREE_MOTOR_ORDER,
) -> list[str]:
    """Return every reason the command permutation is NOT a correct name mapping.

    Empty means correct. The check is by NAME: ``motor_cmd[k]`` must receive the
    policy output for the joint whose name is ``motor_names[k]``. A positional
    table that happens to be a valid permutation but swaps two legs passes every
    shape check and fails here.
    """
    problems: list[str] = []
    order = tuple(joint_order)
    if order != POLICY_JOINT_ORDER:
        problems.append(
            f"joint_order {list(order)} is not the training order {list(POLICY_JOINT_ORDER)}"
        )
    perm = tuple(int(i) for i in phoenix_for_motor)
    if len(perm) != len(motor_names) or sorted(perm) != list(range(len(motor_names))):
        problems.append(
            f"PHOENIX_FOR_MOTOR {perm} is not a permutation of 0..{len(motor_names) - 1}"
        )
        return problems
    if tuple(motor_names) != UNITREE_MOTOR_ORDER:
        problems.append(f"motor names {list(motor_names)} are not the Unitree motor order")
    for k, src in enumerate(perm):
        if src >= len(order):
            problems.append(f"motor {k} reads policy index {src}, beyond {len(order)} joints")
            continue
        if order[src] != motor_names[k]:
            problems.append(
                f"motor_cmd[{k}] ({motor_names[k]}) would receive policy joint "
                f"{src} ({order[src]})"
            )
    return problems


def verify_default_pose(default_joint_pos: Mapping[str, float]) -> list[str]:
    """Return every joint whose deploy nominal pose differs from training. Exact match."""
    problems: list[str] = []
    if set(default_joint_pos) != set(TRAINING_DEFAULT_JOINT_POS):
        problems.append(
            f"default_joint_pos joints {sorted(default_joint_pos)} != training joints "
            f"{sorted(TRAINING_DEFAULT_JOINT_POS)}"
        )
    for name, expected in TRAINING_DEFAULT_JOINT_POS.items():
        if name in default_joint_pos and float(default_joint_pos[name]) != expected:
            problems.append(
                f"default_joint_pos[{name}]={default_joint_pos[name]} but training uses {expected}"
            )
    return problems


__all__ = [
    "JOINT_LIMITS_PROVENANCE",
    "JOINT_POSITION_LIMITS_RAD",
    "JOINT_TORQUE_LIMITS_NM",
    "LIMIT_ABORT_BAND_RAD",
    "POLICY_JOINT_ORDER",
    "TRAINING_DEFAULT_JOINT_POS",
    "UNITREE_EXAMPLE_FOLDED_POSE",
    "UNITREE_EXAMPLE_STAND_POSE",
    "UNITREE_MOTOR_ORDER",
    "limits_in_order",
    "torque_limits_in_order",
    "verify_default_pose",
    "verify_joint_model",
]
