"""Phoenix-owned reward functions not provided by upstream Isaac Lab.

Added 2026-04-19 (Phase 2b retrain) as a template for custom reward
terms. Each function here follows
the upstream Isaac Lab signature `func(env, **params) -> Tensor[E]`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import torch

# 85% of MAX_DELTA_PER_STEP_RAD=0.175, in JOINT-TARGET RADIANS.  Isaac Lab's
# action-manager tensors are dimensionless policy actions, so the reward must
# apply the same 0.25 action scale used by the action term before comparing
# against this threshold.
_DEFAULT_HINGE_THRESHOLD = 0.15
_DEFAULT_ACTION_SCALE = 0.25


def slew_sat_hinge_l2(
    env,
    threshold: float = _DEFAULT_HINGE_THRESHOLD,
    action_scale: float = _DEFAULT_ACTION_SCALE,
) -> torch.Tensor:
    """Per-motor squared-hinge penalty on action deltas approaching
    the hardware slew clip.

    For each env at each control step, compute
    ``action_scale * |a_t^i - a_{t-1}^i|`` per motor ``i`` to convert the
    dimensionless policy-action delta to a joint-target delta in radians,
    apply a hinge at ``threshold``, square, and sum across motors. Returns a
    positive-magnitude tensor; caller applies a negative weight via
    ``RewTerm``.

    Targets the same failure mode that ``slew_saturation_pct`` in
    ``phoenix.training.evaluate`` measures: any single motor hitting
    the clip is sufficient to activate the penalty, unlike
    ``action_rate_l2`` which is an L2 norm across all motors and can
    stay small while individual motors saturate.

    Args:
        env: IsaacLab ``ManagerBasedRLEnv`` (duck-typed; tests use a
            stand-in with ``action_manager.action`` /
            ``action_manager.prev_action``).
        threshold: Hinge threshold in joint-target radians. Motors with a
            scaled target delta ``<= threshold`` contribute 0.
        action_scale: Radians per unit policy action. Must match the action
            term and deploy config (0.25 for the GO2 policies).
    """
    import torch

    action = env.action_manager.action  # [E, num_actions]
    prev = env.action_manager.prev_action  # [E, num_actions]
    delta = torch.abs(action - prev) * action_scale
    excess = torch.clamp(delta - threshold, min=0.0)
    return (excess**2).sum(dim=-1)  # [E]
