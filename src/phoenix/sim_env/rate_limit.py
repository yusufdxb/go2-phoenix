"""Per-step joint-target rate limiter, shared between sim training and deploy.

The deploy stack (``sim2real.lowcmd_bridge_node`` and ``ros2_policy_node``)
hard-clips every joint position target to ``measured_q ± MAX_DELTA_PER_STEP_RAD``
on every 50 Hz control tick (see :func:`phoenix.sim2real.safety.per_step_clip_array`).

Until now that limiter existed ONLY at deploy: the training env applied the raw
policy target with no rate cap and only *penalised* approaching it via the
``slew_sat_hinge_l2`` reward. A policy trained without the limiter meets it for
the first time on hardware, which is the closed-loop mismatch behind the
0.33% (sim) -> 33% (hardware) slew-saturation blowup observed on 2026-04-21.

This module makes the limiter part of the MDP. :func:`rate_limit_targets`
is the single pure clamp; :mod:`phoenix.sim_env.rate_limited_action` wraps it
in an Isaac Lab ``JointPositionAction`` so the policy is trained against the
exact constraint it will face on the GO2. The clamp is intentionally identical
in form to ``per_step_clip_array`` so sim and deploy provably agree.
"""

from __future__ import annotations

from typing import Any

# Single source of truth for the slew-rate cap. Imported from the deploy side
# so sim training and the Jetson bridge cannot drift on the value.
from phoenix.sim2real.safety import MAX_DELTA_PER_STEP_RAD

__all__ = ["rate_limit_targets", "MAX_DELTA_PER_STEP_RAD"]


def rate_limit_targets(processed: Any, joint_pos: Any, max_delta: float) -> Any:
    """Clip ``processed`` joint targets to ``joint_pos ± max_delta`` element-wise.

    Backend-agnostic: accepts torch tensors (training, via ``.clamp``) or numpy
    arrays (tests). Semantics match :func:`phoenix.sim2real.safety.per_step_clip_array`
    so the in-sim limiter and the on-robot limiter are the same operation.

    Args:
        processed: Desired joint position targets (default_offset + scale*action).
        joint_pos: Current measured joint positions (deploy clips against the
            robot's measured ``q``; sim clips against ``asset.data.joint_pos``).
        max_delta: Maximum allowed per-step change, in radians.

    Returns:
        Targets clipped to ``[joint_pos - max_delta, joint_pos + max_delta]``.
    """
    lower = joint_pos - max_delta
    upper = joint_pos + max_delta
    clamp = getattr(processed, "clamp", None)
    if clamp is not None:  # torch.Tensor, supports tensor min/max bounds
        return clamp(lower, upper)
    import numpy as np

    return np.clip(processed, lower, upper)
