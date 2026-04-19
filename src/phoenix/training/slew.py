"""Slew-saturation metric shared between the sim rollout evaluator and
Jetson dryrun analysis.

The metric measures the fraction of (env, timestep, motor) action-delta
samples whose absolute value is at or above a saturation threshold. The
Jetson ``lowcmd_bridge`` hard-clips per-step joint deltas to
``MAX_DELTA_PER_STEP_RAD`` (0.175 rad); any time the clip activates the
policy is trying to ask for a larger step than the bridge will allow.

Keeping this helper pure-python (no torch, no warp) lets it live in the
CI-gated test suite alongside ``tests/test_safety.py``.
"""
from __future__ import annotations

import numpy as np


def slew_saturation_rate(
    prev_actions: np.ndarray,
    current_actions: np.ndarray,
    threshold: float,
) -> float:
    """Return the fraction of action-delta samples >= ``threshold``.

    Both arrays have shape ``[num_envs, num_motors]`` for a single step.
    For multi-step rollouts, call once per step and average the return
    values weighted by sample count (or equivalently: stack arrays and
    call once with shape ``[steps * envs, motors]``).
    """
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if prev_actions.shape != current_actions.shape:
        raise ValueError(
            f"shape mismatch: prev={prev_actions.shape} curr={current_actions.shape}"
        )
    deltas = np.abs(current_actions - prev_actions)
    return float(np.mean(deltas >= threshold))
