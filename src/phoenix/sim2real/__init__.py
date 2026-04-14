"""Sim-to-real bridge for the trained Phoenix policy.

* :mod:`phoenix.sim2real.export` — export a rsl_rl checkpoint to ONNX, with a
  numerical-parity verification step against the original torch policy.
* :mod:`phoenix.sim2real.observation` — build the observation vector from
  ROS 2 messages in exactly the order the policy was trained with.
* :mod:`phoenix.sim2real.ros2_policy_node` — ROS 2 node running the policy
  on the real GO2 at 50 Hz.
"""

from .observation import JointOrder, ObservationBuilder

__all__ = ["JointOrder", "ObservationBuilder"]
