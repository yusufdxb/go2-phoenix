"""Sim-to-real bridge for the trained Phoenix policy.

* :mod:`phoenix.sim2real.export`, export a rsl_rl checkpoint to ONNX, with a
  numerical-parity verification step against the original torch policy.
* :mod:`phoenix.sim2real.observation`, build the observation vector from
  ROS 2 messages in exactly the order the policy was trained with, and
  resolve the one term (``base_lin_vel``) the robot cannot measure.
* :mod:`phoenix.sim2real.obs_parity`, gate the assembled observation against
  the training term contract. Complements ``verify_deploy``, which compares
  Torch to ONNX on an already-assembled vector and therefore cannot see a
  term that was never filled in.
* :mod:`phoenix.sim2real.ros2_policy_node`, ROS 2 node running the policy
  on the real GO2 at 50 Hz.
"""

from .observation import (
    BASE_LIN_VEL_SOURCES,
    OBS_TERM_ORDER,
    BaseLinVelSample,
    BaseLinVelUnavailableError,
    JointOrder,
    ObservationBuilder,
    assemble_policy_observation,
    resolve_base_lin_vel,
)

__all__ = [
    "BASE_LIN_VEL_SOURCES",
    "OBS_TERM_ORDER",
    "BaseLinVelSample",
    "BaseLinVelUnavailableError",
    "JointOrder",
    "ObservationBuilder",
    "assemble_policy_observation",
    "resolve_base_lin_vel",
]
