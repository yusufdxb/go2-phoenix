"""The policy node's action map, as pure functions shared with tests and replays.

Layer 1 is the raw ONNX output. Layer 2, what the trained plant would execute, is
``default_q + action_scale * clip(raw, -action_clip, action_clip)``: Isaac Lab's
``RslRlVecEnvWrapper(clip_actions=1.0)`` applies that clamp before the action term in
PPO training and evaluation, and the ``last_action`` observation term reads the
CLAMPED action (``ActionManager.action``). The exported ONNX has no clamp. Before
Phoenix v2 the deploy node skipped it on both paths: on the 2026-09-22 F1 run it fed
raw actions up to -9.7 back into the observation and requested joint targets the
trained plant could never produce. ``action_clip=None`` reproduces that legacy map for
old configs and replays; every v2 deploy config must set 1.0 (deploy contract).
"""

from __future__ import annotations

import numpy as np

from .safety import LIMITER_MODES, per_step_clip_array


def policy_action_map(
    raw_action: np.ndarray,
    default_q: np.ndarray,
    action_scale: float,
    action_clip: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(fed_back_action, requested_target)``.

    ``fed_back_action`` is what the next observation's ``last_action`` must hold;
    ``requested_target`` is layer 2 in radians, policy joint order.
    """
    raw = np.asarray(raw_action, dtype=np.float32)
    act = raw if action_clip is None else np.clip(raw, -action_clip, action_clip)
    requested = np.asarray(default_q, dtype=np.float32) + np.float32(action_scale) * act
    return act.astype(np.float32, copy=False), requested.astype(np.float32, copy=False)


def node_soft_limit(
    requested: np.ndarray, q: np.ndarray, limiter_mode: str, max_delta: float
) -> np.ndarray:
    """The policy node's own soft limiter.

    ``measured_q`` (incumbent): clip to the node's measured ``q +/- max_delta``.
    ``prev_command``: none here. The bridge is the single command-rate limiter, keyed
    on what it actually sent; a second copy in the node would hold its own state and
    could disagree after a dropped message, and the node's output would stop being the
    policy's request.
    """
    if limiter_mode not in LIMITER_MODES:
        raise ValueError(f"unknown limiter mode {limiter_mode!r}")
    if limiter_mode == "prev_command":
        return np.asarray(requested, dtype=np.float32).copy()
    return np.asarray(per_step_clip_array(requested, q, max_delta), dtype=np.float32)


__all__ = ["node_soft_limit", "policy_action_map"]
