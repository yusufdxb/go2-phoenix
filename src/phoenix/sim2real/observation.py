"""Build the policy observation vector from ROS 2 messages.

Layout must match what the policy was trained with. On flat terrain the
observation is 48-dim proprioception:

.. code-block:: text

    [ base_lin_vel (3) ,
      base_ang_vel (3) ,
      projected_gravity (3) ,
      velocity_command (3) ,
      joint_pos_rel_default (12) ,
      joint_vel (12) ,
      last_action (12) ]  -> 48 dims

On rough terrain Isaac Lab's task also appends a 187-dim height scanner
reading (total 235 dims); a deployed rough-terrain policy therefore
needs an equivalent scanner feed on the real robot. The builder below
returns only the proprioceptive 48-dim prefix — deploy scripts that
need the full 235-dim vector are responsible for concatenating the
height scan (either from a real sensor or a stubbed zero vector).

This module is pure numpy (no rclpy), so it can be unit-tested in CI.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointOrder:
    """The canonical joint order used at training time.

    ROS 2 ``/joint_states`` messages may emit joints in a different
    order — :meth:`remap` builds an index vector that reorders a
    joint-state array into the policy's canonical layout.
    """

    names: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.names)

    def remap(self, ros_joint_names: list[str]) -> np.ndarray:
        """Return indices into ``ros_joint_names`` such that
        ``arr[indices]`` yields values in canonical order."""
        lookup = {name: i for i, name in enumerate(ros_joint_names)}
        missing = [n for n in self.names if n not in lookup]
        if missing:
            raise KeyError(f"ROS /joint_states is missing joints: {missing}")
        return np.asarray([lookup[n] for n in self.names], dtype=np.int64)


class ObservationBuilder:
    """Stateless builder for the policy observation vector."""

    def __init__(self, joint_order: JointOrder, default_joint_pos: Mapping[str, float]) -> None:
        self.joint_order = joint_order
        self.default_q = np.asarray(
            [default_joint_pos[n] for n in joint_order.names], dtype=np.float32
        )
        self._zero_action = np.zeros(len(joint_order), dtype=np.float32)

    @property
    def dim(self) -> int:
        return 3 + 3 + 3 + 3 + 3 * len(self.joint_order)

    def build(
        self,
        *,
        base_lin_vel: np.ndarray,  # (3,) m/s in body frame
        base_ang_vel: np.ndarray,  # (3,) rad/s in body frame
        projected_gravity: np.ndarray,  # (3,) body-frame gravity unit vector
        velocity_command: np.ndarray,  # (3,) [vx, vy, wz]
        joint_pos: np.ndarray,  # (N,) canonical order, absolute rad
        joint_vel: np.ndarray,  # (N,) canonical order, rad/s
        last_action: np.ndarray | None = None,  # (N,) policy output from prev step
    ) -> np.ndarray:
        if last_action is None:
            last_action = self._zero_action
        parts = [
            base_lin_vel.astype(np.float32, copy=False),
            base_ang_vel.astype(np.float32, copy=False),
            projected_gravity.astype(np.float32, copy=False),
            velocity_command.astype(np.float32, copy=False),
            (joint_pos - self.default_q).astype(np.float32, copy=False),
            joint_vel.astype(np.float32, copy=False),
            last_action.astype(np.float32, copy=False),
        ]
        obs = np.concatenate(parts, axis=-1)
        if obs.shape[-1] != self.dim:
            raise ValueError(f"Observation dim mismatch: got {obs.shape[-1]}, expected {self.dim}")
        return obs
