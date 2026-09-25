"""Base orientation for evaluation: one explicit quaternion convention per source.

Why this module exists
----------------------
Isaac Lab 3.0 changed ``root_quat_w`` (and every quaternion in its math utils)
from (w, x, y, z) to (x, y, z, w). The installed Isaac Lab is the editable
checkout at ``~/Sim/IsaacLab`` (``v3.0.0-beta``, pip name ``isaaclab 4.5.22``):

* ``source/isaaclab/docs/CHANGELOG.rst`` 1.0.0 (2026-01-30): "Changed the
  quaternion ordering to match warp, PhysX, and Newton native XYZW quaternion
  ordering."
* ``docs/source/migration/migrating_to_isaaclab_3-0.rst``: "The quaternion
  format changed from WXYZ to XYZW."
* ``isaaclab/utils/math.py`` ``matrix_from_quat``: "quaternions: The quaternion
  orientation in (x, y, z, w)".
* ``isaaclab_physx/.../articulation_data.py`` ``root_link_quat_w`` returns a
  ``wp.quatf`` (warp native xyzw); ``write_root_pose_to_sim_index`` documents
  "quaternion orientation in (x, y, z, w)".

Phoenix's evaluator kept reading ``root_quat_w`` as wxyz. Reading an xyzw
quaternion as wxyz is an exact, deterministic remap (derived in
``docs/forensics/evaluation_repair.md`` and locked by
``tests/test_evaluation_orientation.py``)::

    reported_roll  = true_yaw
    reported_pitch = -true_pitch
    reported_yaw   = pi - true_roll

so every sim "attitude" event was a heading test and true roll was never
checked. Nothing here guesses a convention: every entry point takes one, and
:func:`check_against_projected_gravity` cross-checks the result against the
simulator's own ``projected_gravity_b`` so a wrong convention is caught at run
time instead of producing plausible numbers.

Pure numpy, CI-safe. Quaternion math is delegated to
:mod:`phoenix.velocity.observation` (the one (w, x, y, z) implementation).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from phoenix.velocity.observation import (
    normalized_quat_wxyz,
    projected_gravity_wxyz,
    tilt_from_projected_gravity,
)

CONVENTION_WXYZ = "wxyz"
CONVENTION_XYZW = "xyzw"
CONVENTIONS = (CONVENTION_WXYZ, CONVENTION_XYZW)

#: Convention of ``articulation.data.root_quat_w`` in the Isaac Lab this repo
#: runs (3.0.0-beta checkout, see the module docstring for the sources).
ISAACLAB_ROOT_QUAT_CONVENTION = CONVENTION_XYZW
#: Unitree ``LowState.imu_state.quaternion`` is (w, x, y, z)
#: (``phoenix.sim2real.lowstate_bridge_node`` maps it field by field).
UNITREE_IMU_QUAT_CONVENTION = CONVENTION_WXYZ
#: ``phoenix.real_world.trajectory_logger`` ``base_quat`` column.
TRAJECTORY_LOGGER_QUAT_CONVENTION = CONVENTION_XYZW

#: Max disagreement (unit-vector components) between Phoenix's projected gravity
#: and the simulator's own before the evaluator declares itself inconsistent.
#: float32 round-off is ~1e-6; a convention error produces O(1) differences.
PROJECTED_GRAVITY_TOLERANCE = 1e-3


class OrientationError(ValueError):
    """The orientation cannot be computed trustworthily."""


@dataclass(frozen=True)
class Attitude:
    """Roll / pitch / yaw (intrinsic Z-Y-X, radians) and total tilt from vertical."""

    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    #: Angle between body z and world up, [0, pi]. Independent of yaw and of
    #: Euler-angle singularities; 0 upright, pi/2 on its side, pi upside down.
    tilt_rad: float


def to_wxyz(quat: Sequence[float] | np.ndarray, convention: str) -> np.ndarray:
    """Reorder one quaternion or a batch (..., 4) to (w, x, y, z). No guessing."""
    q = np.asarray(quat, dtype=np.float64)
    if q.shape[-1] != 4:
        raise OrientationError(f"quaternion last dim must be 4, got shape {q.shape}")
    if convention == CONVENTION_WXYZ:
        return q.copy()
    if convention == CONVENTION_XYZW:
        return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)
    raise OrientationError(f"unknown quaternion convention {convention!r}; use {CONVENTIONS}")


def euler_zyx_wxyz(quat_wxyz: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """(roll, pitch, yaw) of a (w, x, y, z) body-to-world quaternion, radians.

    Intrinsic Z-Y-X (yaw, then pitch, then roll), the aerospace convention and
    the one ``phoenix.sim2real.hw_probe`` reports on hardware. Pitch is clamped
    at the gimbal-lock pole instead of raising. Invariant under q -> -q.
    """
    w, x, y, z = normalized_quat_wxyz(quat_wxyz)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def attitude_from_quat(quat: Sequence[float] | np.ndarray, convention: str) -> Attitude:
    """Attitude of one quaternion given in an explicitly named ``convention``."""
    q = to_wxyz(quat, convention)
    if q.shape != (4,):
        raise OrientationError(f"attitude_from_quat takes one quaternion, got shape {q.shape}")
    roll, pitch, yaw = euler_zyx_wxyz(q)
    tilt = tilt_from_projected_gravity(projected_gravity_wxyz(q))
    return Attitude(roll, pitch, yaw, tilt)


def attitude_batch(quats: np.ndarray, convention: str) -> dict[str, np.ndarray]:
    """Vectorised :func:`attitude_from_quat` over (N, 4). Returns arrays keyed like Attitude."""
    q = np.atleast_2d(np.asarray(quats, dtype=np.float64))
    out = {k: np.empty(len(q)) for k in ("roll_rad", "pitch_rad", "yaw_rad", "tilt_rad")}
    for i, row in enumerate(q):
        a = attitude_from_quat(row, convention)
        out["roll_rad"][i] = a.roll_rad
        out["pitch_rad"][i] = a.pitch_rad
        out["yaw_rad"][i] = a.yaw_rad
        out["tilt_rad"][i] = a.tilt_rad
    return out


def projected_gravity_batch(quats: np.ndarray, convention: str) -> np.ndarray:
    """(N, 3) world gravity direction in the body frame for (N, 4) quaternions."""
    q = to_wxyz(np.atleast_2d(np.asarray(quats, dtype=np.float64)), convention)
    return np.stack([projected_gravity_wxyz(row) for row in q])


def check_against_projected_gravity(
    quats: np.ndarray,
    convention: str,
    simulator_projected_gravity: np.ndarray,
    *,
    tol: float = PROJECTED_GRAVITY_TOLERANCE,
) -> float:
    """Max abs difference between our projected gravity and the simulator's.

    Raises :class:`OrientationError` when it exceeds ``tol``: the evaluator is
    reading orientation in a different convention or frame than the simulator
    that produced it, and every attitude number it reports would be wrong. This
    is the run-time proof of the convention; the constants above are only the
    documented expectation.
    """
    ours = projected_gravity_batch(quats, convention)
    sim = np.atleast_2d(np.asarray(simulator_projected_gravity, dtype=np.float64))
    if ours.shape != sim.shape:
        raise OrientationError(f"shape mismatch: ours {ours.shape} vs simulator {sim.shape}")
    diff = float(np.max(np.abs(ours - sim))) if ours.size else 0.0
    if not math.isfinite(diff) or diff > tol:
        raise OrientationError(
            f"projected gravity from root_quat_w read as {convention} disagrees with the "
            f"simulator's projected_gravity_b by {diff:.4g} (> {tol}); the quaternion "
            "convention or frame is wrong, refusing to report attitude"
        )
    return diff


def legacy_misread_euler(quat_xyzw: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """What the pre-2026-09-22 evaluator reported for an Isaac Lab 3.0 quaternion.

    It passed the xyzw quaternion straight to a wxyz Euler routine. Kept only to
    reproduce and test the defect; never use it to score anything.
    """
    q = np.asarray(quat_xyzw, dtype=np.float64)
    return euler_zyx_wxyz(q)  # the bug: xyzw components consumed as (w, x, y, z)


__all__ = [
    "CONVENTIONS",
    "CONVENTION_WXYZ",
    "CONVENTION_XYZW",
    "ISAACLAB_ROOT_QUAT_CONVENTION",
    "PROJECTED_GRAVITY_TOLERANCE",
    "TRAJECTORY_LOGGER_QUAT_CONVENTION",
    "UNITREE_IMU_QUAT_CONVENTION",
    "Attitude",
    "OrientationError",
    "attitude_batch",
    "attitude_from_quat",
    "check_against_projected_gravity",
    "euler_zyx_wxyz",
    "legacy_misread_euler",
    "projected_gravity_batch",
    "to_wxyz",
]
