"""Build the 45-D PhoenixVelocity actor observation from raw robot state.

This is the ONLY function that assembles the actor observation outside Isaac Lab.
The hardware policy node and the MuJoCo sim-to-sim runner both call
:func:`build_actor_observation`, so a mismatch between them is impossible by
construction; parity against Isaac Lab's own terms is tested separately.

Pure numpy. Fails closed: wrong shape, non-finite value or a quaternion that is
not close to unit norm raises :class:`ObservationError` instead of returning a
plausible vector.

Quaternion convention: **(w, x, y, z)**, body-to-world rotation, which is what
the GO2 ``LowState.imu_state.quaternion`` and MuJoCo ``qpos[3:7]`` both carry.
Callers holding (x, y, z, w) data must convert explicitly with
:func:`quat_xyzw_to_wxyz`; nothing here guesses.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .contract import ACTOR_OBS_DIM, DEFAULT_JOINT_POS, JOINT_ORDER, NUM_JOINTS, obs_slices

#: A measured quaternion whose norm is further than this from 1 is treated as a
#: corrupt sample, not renormalized away.
QUAT_NORM_TOLERANCE = 0.05

_DEFAULT_Q = np.asarray([DEFAULT_JOINT_POS[j] for j in JOINT_ORDER], dtype=np.float64)
_GRAVITY_WORLD = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)


class ObservationError(ValueError):
    """The inputs cannot produce a trustworthy observation."""


def _vec(name: str, value: Sequence[float] | np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (n,):
        raise ObservationError(f"{name} must have shape ({n},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ObservationError(f"{name} contains NaN/Inf: {arr.tolist()}")
    return arr


def quat_xyzw_to_wxyz(q_xyzw: Sequence[float] | np.ndarray) -> np.ndarray:
    q = _vec("quat_xyzw", q_xyzw, 4)
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64)


def normalized_quat_wxyz(quat_wxyz: Sequence[float] | np.ndarray) -> np.ndarray:
    q = _vec("quat_wxyz", quat_wxyz, 4)
    norm = float(np.linalg.norm(q))
    if abs(norm - 1.0) > QUAT_NORM_TOLERANCE:
        raise ObservationError(f"quaternion norm {norm:.4f} is not ~1 (corrupt sample?)")
    return q / norm


def rotation_matrix_wxyz(quat_wxyz: Sequence[float] | np.ndarray) -> np.ndarray:
    """Body-to-world rotation matrix of a (w, x, y, z) quaternion."""
    w, x, y, z = normalized_quat_wxyz(quat_wxyz)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def projected_gravity_wxyz(quat_wxyz: Sequence[float] | np.ndarray) -> np.ndarray:
    """World gravity direction (0, 0, -1) expressed in the body frame.

    Equals Isaac Lab ``mdp.projected_gravity`` (``quat_apply_inverse(root_quat,
    GRAVITY_VEC_W)``). Upright gives (0, 0, -1). Invariant under q -> -q.
    """
    return rotation_matrix_wxyz(quat_wxyz).T @ _GRAVITY_WORLD


def tilt_from_projected_gravity(g_body: Sequence[float] | np.ndarray) -> float:
    """Angle in rad between the body z axis and world up. 0 upright, pi upside down."""
    g = _vec("projected_gravity", g_body, 3)
    norm = float(np.linalg.norm(g))
    if norm < 1e-6:
        raise ObservationError("projected gravity has zero norm")
    return float(np.arccos(np.clip(-g[2] / norm, -1.0, 1.0)))


def build_actor_observation(
    *,
    gyro_body: Sequence[float] | np.ndarray,
    quat_wxyz: Sequence[float] | np.ndarray,
    command: Sequence[float] | np.ndarray,
    joint_pos: Sequence[float] | np.ndarray,
    joint_vel: Sequence[float] | np.ndarray,
    last_action: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Assemble the 45-D actor observation. All joint arrays are in :data:`JOINT_ORDER`.

    Args:
        gyro_body: body-frame angular velocity, rad/s.
        quat_wxyz: base orientation, (w, x, y, z), body-to-world.
        command: (vx m/s, vy m/s, wz rad/s).
        joint_pos: ABSOLUTE joint positions, rad (the default pose is subtracted here).
        joint_vel: joint velocities, rad/s.
        last_action: previous raw policy action (dimensionless, pre-scale).
    """
    parts = {
        "base_ang_vel": _vec("gyro_body", gyro_body, 3),
        "projected_gravity": projected_gravity_wxyz(quat_wxyz),
        "velocity_command": _vec("command", command, 3),
        "joint_pos_rel": _vec("joint_pos", joint_pos, NUM_JOINTS) - _DEFAULT_Q,
        "joint_vel": _vec("joint_vel", joint_vel, NUM_JOINTS),
        "last_action": _vec("last_action", last_action, NUM_JOINTS),
    }
    obs = np.empty(ACTOR_OBS_DIM, dtype=np.float64)
    for name, sl in obs_slices().items():
        obs[sl] = parts[name]
    return obs.astype(np.float32)


def actions_to_joint_targets(
    action: Sequence[float] | np.ndarray, action_scale: float
) -> np.ndarray:
    """Raw policy action -> absolute joint-position targets (rad), JOINT_ORDER.

    Mirrors Isaac Lab ``JointPositionAction`` with ``use_default_offset=True``:
    ``target = default + scale * action``. No clipping here: clipping is a safety
    decision and must be logged by the safety layer, not hidden in the mapping.
    """
    a = _vec("action", action, NUM_JOINTS)
    return _DEFAULT_Q + float(action_scale) * a


__all__ = [
    "ObservationError",
    "QUAT_NORM_TOLERANCE",
    "actions_to_joint_targets",
    "build_actor_observation",
    "normalized_quat_wxyz",
    "projected_gravity_wxyz",
    "quat_xyzw_to_wxyz",
    "rotation_matrix_wxyz",
    "tilt_from_projected_gravity",
]
