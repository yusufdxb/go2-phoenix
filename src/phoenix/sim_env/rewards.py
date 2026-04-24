"""Phoenix-owned reward functions not provided by upstream Isaac Lab.

Added 2026-04-19 (Phase 2b retrain, spec
`docs/superpowers/specs/2026-04-19-phoenix-gate8-slewhinge-design.md`)
as a template for custom reward terms. Each function here follows
the upstream Isaac Lab signature `func(env, **params) -> Tensor[E]`.
"""

from __future__ import annotations

import torch

# 85% of MAX_DELTA_PER_STEP_RAD=0.175, the per-step slew clip enforced
# in phoenix.sim2real.safety and on the GO2 hardware deploy path.
_DEFAULT_HINGE_THRESHOLD = 0.15


def slew_sat_hinge_l2(
    env,
    threshold: float = _DEFAULT_HINGE_THRESHOLD,
) -> torch.Tensor:
    """Per-motor squared-hinge penalty on action deltas approaching
    the hardware slew clip.

    For each env at each control step, compute
    ``|a_t^i - a_{t-1}^i|`` per motor ``i``, apply a hinge at
    ``threshold``, square, and sum across motors. Returns a
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
        threshold: Hinge threshold in radians. Motors with
            ``|delta| <= threshold`` contribute 0.
    """
    action = env.action_manager.action  # [E, num_actions]
    prev = env.action_manager.prev_action  # [E, num_actions]
    delta = torch.abs(action - prev)
    excess = torch.clamp(delta - threshold, min=0.0)
    return (excess**2).sum(dim=-1)  # [E]
