"""Translate :class:`VariationSample` instances into per-env perturbations.

The replay path needs to apply each sampled variation to a distinct sim
env. Isaac Lab's per-env physics knobs (mass, joint damping, root
velocity) are exposed as tensors of shape ``(num_envs, ...)``. Building
those tensors is pure numpy — keep it here so we can unit-test it
without Isaac Lab.

The Isaac Lab caller in :mod:`phoenix.replay.reconstruct` consumes the
output dict and routes each field to the right ``robot.write_*`` API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np

from .trajectory_reader import InitialState
from .variations import VariationSample


class PerEnvInitialConditions(TypedDict):
    """Per-env tensors ready for Isaac Lab ``write_*_to_sim`` calls.

    All arrays have leading dimension ``num_envs == len(variations)``.
    """

    base_pos: np.ndarray  # (N, 3)   world-frame, env_origin must be added by caller
    base_quat_wxyz: np.ndarray  # (N, 4)   wxyz for Isaac Lab — converted from xyzw
    base_lin_vel: np.ndarray  # (N, 3)   logged + push_velocity_delta along x
    base_ang_vel: np.ndarray  # (N, 3)   logged + push_yaw_delta on z
    joint_pos: np.ndarray  # (N, 12)
    joint_vel: np.ndarray  # (N, 12)
    base_mass_delta_kg: np.ndarray  # (N,)  added to body[0] mass via root_physx_view
    friction_scale: np.ndarray  # (N,)  multiplicative factor on the scene friction range


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Roll an (..., 4) array's last axis from (x,y,z,w) to (w,x,y,z)."""
    if quat_xyzw.shape[-1] != 4:
        raise ValueError(f"quat last dim must be 4 (xyzw), got {quat_xyzw.shape}")
    return np.roll(quat_xyzw, shift=1, axis=-1)


def build_per_env_initial_conditions(
    initial: InitialState,
    variations: Sequence[VariationSample],
) -> PerEnvInitialConditions:
    """Build per-env tensors from a logged initial state + N variation samples.

    The returned dict's leading dim is ``len(variations)`` everywhere.

    * ``push_velocity_delta`` is added to the body-frame x-component of
      ``base_lin_vel`` so each env starts with a different forward push.
    * ``push_yaw_delta`` is added to the body-frame z-component of
      ``base_ang_vel`` so each env starts with a different yaw spin.
    * ``mass_delta_kg`` flows through unchanged for the caller to apply
      to body[0] (the trunk) via ``root_physx_view.set_masses``.
    * ``friction_delta`` is converted to a multiplicative scale around 1.0
      so the caller can apply it to the scene friction range. Callers that
      cannot apply per-env friction should reduce this to a single scalar
      (e.g. ``mean``) and document the loss of fidelity.
    """
    n = len(variations)
    if n == 0:
        raise ValueError("variations must be non-empty")

    base_pos = np.broadcast_to(initial.base_pos.reshape(1, 3), (n, 3)).astype(np.float32, copy=True)
    base_quat_wxyz = np.broadcast_to(
        _quat_xyzw_to_wxyz(initial.base_quat).reshape(1, 4), (n, 4)
    ).astype(np.float32, copy=True)
    joint_pos = np.broadcast_to(
        initial.joint_pos.reshape(1, -1), (n, initial.joint_pos.shape[0])
    ).astype(np.float32, copy=True)
    joint_vel = np.broadcast_to(
        initial.joint_vel.reshape(1, -1), (n, initial.joint_vel.shape[0])
    ).astype(np.float32, copy=True)

    base_lin_vel = np.broadcast_to(initial.base_lin_vel_body.reshape(1, 3), (n, 3)).astype(
        np.float32, copy=True
    )
    base_ang_vel = np.broadcast_to(initial.base_ang_vel_body.reshape(1, 3), (n, 3)).astype(
        np.float32, copy=True
    )

    push_lin = np.asarray([v.push_velocity_delta for v in variations], dtype=np.float32)
    push_yaw = np.asarray([v.push_yaw_delta for v in variations], dtype=np.float32)
    base_lin_vel[:, 0] += push_lin
    base_ang_vel[:, 2] += push_yaw

    mass_delta = np.asarray([v.mass_delta_kg for v in variations], dtype=np.float32)
    # friction_delta is interpreted as Δμ around the scene's nominal μ; we
    # express it as a multiplicative factor so callers that can only set a
    # single per-scene friction range still get a meaningful sweep.
    friction_scale = (
        1.0 + np.asarray([v.friction_delta for v in variations], dtype=np.float32)
    ).clip(min=0.05)

    return PerEnvInitialConditions(
        base_pos=base_pos,
        base_quat_wxyz=base_quat_wxyz,
        base_lin_vel=base_lin_vel,
        base_ang_vel=base_ang_vel,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        base_mass_delta_kg=mass_delta,
        friction_scale=friction_scale,
    )


__all__ = [
    "PerEnvInitialConditions",
    "build_per_env_initial_conditions",
]
