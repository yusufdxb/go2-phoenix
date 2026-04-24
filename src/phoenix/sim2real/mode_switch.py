"""Deploy-layer mode switch for the two-policy runtime.

Stand-v2 handles ``cmd_vel ≈ 0`` (Gate 7 policy, slew 0.00254 in sim).
v3b handles walking (tracking lin_err 0.091 / ang_err 0.087).

The ROS 2 policy node loads both ONNX sessions and picks one target per
tick based on this state machine. Transitions between modes use a
linear blend of joint *targets* over ``transition_ticks`` ticks; the
per-step slew clip (``MAX_DELTA_PER_STEP_RAD``) runs afterwards so the
hardware slew constraint is always respected.

This module has no rclpy / onnxruntime dependency on purpose: every
branch is covered in ``tests/test_mode_switch.py`` without Isaac Lab
or the robot.

See docs/superpowers/specs/2026-04-19-phoenix-gate8-mode-switch-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class State(str, Enum):
    STAND = "stand"
    TRANS_TO_WALK = "trans_to_walk"
    WALK = "walk"
    TRANS_TO_STAND = "trans_to_stand"


@dataclass(frozen=True)
class ModeSwitchCfg:
    enter_walk_thresh: float = 0.15
    enter_stand_thresh: float = 0.05
    yaw_scale: float = 0.3
    transition_ticks: int = 25

    def __post_init__(self) -> None:
        if not self.enter_stand_thresh < self.enter_walk_thresh:
            raise ValueError("enter_stand_thresh must be < enter_walk_thresh (hysteresis)")
        if self.transition_ticks < 1:
            raise ValueError("transition_ticks must be >= 1")
        if self.yaw_scale < 0:
            raise ValueError("yaw_scale must be non-negative")


def cmd_magnitude(cmd_vel: np.ndarray, yaw_scale: float) -> float:
    """Scalar magnitude used by the state machine.

    Takes the larger of the planar linear-velocity norm and the
    yaw-scaled absolute yaw rate. Puts both signals on the same
    m/s-equivalent axis without forcing the caller to pick between
    them.
    """
    vx, vy, vyaw = float(cmd_vel[0]), float(cmd_vel[1]), float(cmd_vel[2])
    lin = float(np.hypot(vx, vy))
    yaw = abs(vyaw) * float(yaw_scale)
    return max(lin, yaw)


def initial_state(cfg: ModeSwitchCfg) -> tuple[State, int]:
    """Always boot into STAND regardless of initial cmd.

    The first safe tick of the node publishes the stand target; the
    state machine re-evaluates from there. This matches the spec's
    ``initial_state: stand`` contract and avoids a spurious
    TRANS_TO_WALK kick on startup if cmd_vel happens to be non-zero
    at boot.
    """
    del cfg
    return State.STAND, 0


def step(
    state: State,
    ticks_in_state: int,
    cmd_vel: np.ndarray,
    cfg: ModeSwitchCfg,
) -> tuple[State, int, float]:
    """Advance the state machine by one control tick.

    Returns ``(next_state, next_ticks_in_state, blend_alpha)`` where
    ``blend_alpha`` is:

    * ``0.0`` when the next state is a pure state (STAND / WALK), or
    * ``ticks_in_state / transition_ticks`` for TRANS_TO_WALK and
      TRANS_TO_STAND — representing how far through the blend we are
      at the *start* of this tick.

    Transitions run to completion. A reverse command received during
    a transition is ignored until the destination pure state has been
    entered for at least one tick (keeps the blend math unambiguous).
    """
    magnitude = cmd_magnitude(cmd_vel, cfg.yaw_scale)

    if state is State.STAND:
        if magnitude > cfg.enter_walk_thresh:
            return State.TRANS_TO_WALK, 0, 0.0
        return State.STAND, ticks_in_state + 1, 0.0

    if state is State.WALK:
        if magnitude < cfg.enter_stand_thresh:
            return State.TRANS_TO_STAND, 0, 0.0
        return State.WALK, ticks_in_state + 1, 0.0

    # For both TRANS states: alpha describes the blend for THIS tick
    # (``target = (1-alpha)*src + alpha*dst``). It walks from 0/N on
    # tick 0 up to (N-1)/N on tick N-1; on tick N-1 we also flip to
    # the destination pure state so the NEXT tick publishes the
    # destination target directly (no alpha=1 step).
    if state is State.TRANS_TO_WALK:
        alpha = ticks_in_state / cfg.transition_ticks
        next_ticks = ticks_in_state + 1
        if next_ticks >= cfg.transition_ticks:
            return State.WALK, 0, alpha
        return State.TRANS_TO_WALK, next_ticks, alpha

    if state is State.TRANS_TO_STAND:
        alpha = ticks_in_state / cfg.transition_ticks
        next_ticks = ticks_in_state + 1
        if next_ticks >= cfg.transition_ticks:
            return State.STAND, 0, alpha
        return State.TRANS_TO_STAND, next_ticks, alpha

    raise ValueError(f"unknown state: {state!r}")
