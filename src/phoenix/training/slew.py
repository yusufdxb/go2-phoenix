"""Deploy-equivalent slew-clip metric, shared by the sim rollout evaluator and
Jetson dryrun analysis.

What deployment actually does, once per 50 Hz control tick, in
``phoenix.sim2real.ros2_policy_node._tick``::

    target = default_q + action_scale * action
    target = per_step_clip_array(target, measured_q, MAX_DELTA_PER_STEP_RAD)

The cap therefore acts on JOINT-POSITION TARGETS measured against the
MEASURED joint position. It does not act on raw policy-action deltas.

:func:`slew_clip_activation_rate` reproduces that computation exactly, calling
the shared deploy helper :func:`phoenix.sim2real.safety.per_step_clip_array`
so the sim-side metric and the deploy-side limiter cannot drift apart.

:func:`legacy_raw_action_delta_saturation_rate` is the ORIGINAL, INCORRECT
definition (``|action[t] - action[t-1]| >= 0.175``). It compares a different
quantity in different units against the deploy cap, so its percentages are not
deploy-equivalent. It is retained only to reproduce numbers already published
in ``reliability_eval/``, ``logs/``, ``checkpoints/`` and ``data/``, which were
all produced by that definition. Do not use it for new results.

Keeping this helper pure-python (no torch, no warp) lets it live in the
CI-gated test suite alongside ``tests/test_safety.py``.
"""

from __future__ import annotations

import numpy as np

from phoenix.sim2real.safety import MAX_DELTA_PER_STEP_RAD, per_step_clip_array


def slew_clip_activation_rate(
    *,
    actions: np.ndarray,
    measured_q: np.ndarray,
    default_q: np.ndarray,
    action_scale: float | np.ndarray,
    max_delta: float = MAX_DELTA_PER_STEP_RAD,
) -> float:
    """Return the fraction of (env, motor) samples the deploy clip would alter.

    Args:
        actions: raw policy output, shape ``[num_envs, num_motors]``. The
            deploy node feeds the ONNX output straight into the affine map,
            so no action-space clipping is applied here either.
        measured_q: joint position the clip is referenced against, same shape.
            This must be the position read BEFORE the step the actions are
            applied on, which is the value the deploy node and the Isaac Lab
            ``RateLimitedJointPositionAction`` both clip against.
        default_q: joint-position offset, shape ``[num_motors]`` or
            ``[num_envs, num_motors]``.
        action_scale: policy-output scale, scalar or broadcastable array.
        max_delta: per-step cap in rad, defaults to the canonical deploy value.

    Returns:
        Fraction in ``[0, 1]`` of samples where
        ``per_step_clip_array(target, measured_q, max_delta) != target``.
        A sample sitting exactly ``max_delta`` away is NOT counted: the clip
        returns it unchanged, so the limiter did not actually act.
    """
    if max_delta <= 0.0:
        raise ValueError(f"max_delta must be positive, got {max_delta}")
    actions = np.asarray(actions)
    measured_q = np.asarray(measured_q)
    default_q = np.asarray(default_q)
    if actions.shape != measured_q.shape:
        raise ValueError(f"shape mismatch: actions={actions.shape} measured_q={measured_q.shape}")
    if default_q.shape not in (actions.shape, actions.shape[-1:]):
        raise ValueError(
            f"default_q shape {default_q.shape} must be {actions.shape} or {actions.shape[-1:]}"
        )
    target = default_q + np.asarray(action_scale) * actions
    if target.shape != actions.shape:
        raise ValueError(
            f"action_scale broadcast produced shape {target.shape}, not {actions.shape}"
        )
    if not np.isfinite(target).all() or not np.isfinite(measured_q).all():
        raise ValueError("non-finite joint target or measured joint position")
    clipped = per_step_clip_array(target, measured_q, max_delta)
    return float(np.mean(clipped != target))


def legacy_raw_action_delta_saturation_rate(
    prev_actions: np.ndarray,
    current_actions: np.ndarray,
    threshold: float,
) -> float:
    """LEGACY, NOT DEPLOY-EQUIVALENT. Fraction of action deltas >= threshold.

    This compares consecutive RAW POLICY ACTIONS against the deploy cap
    ``MAX_DELTA_PER_STEP_RAD``, which is a joint-position bound. The two are
    different quantities (the action is scaled by ``action_scale`` and offset
    by ``default_q`` before the cap applies) and the clip is referenced
    against measured ``q``, not against the previous action. Every
    ``slew_saturation_pct`` number recorded before 2026-09-11 came from this
    function; keep it only to reproduce those, and use
    :func:`slew_clip_activation_rate` for anything new.

    Both arrays have shape ``[num_envs, num_motors]`` for a single step.
    """
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if prev_actions.shape != current_actions.shape:
        raise ValueError(f"shape mismatch: prev={prev_actions.shape} curr={current_actions.shape}")
    deltas = np.abs(current_actions - prev_actions)
    return float(np.mean(deltas >= threshold))
