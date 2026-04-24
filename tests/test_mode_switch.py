"""Tests for the two-policy (stand-v2 / v3b) deploy-layer mode switch.

All tests are pure-numpy. The state machine in ``phoenix.sim2real.mode_switch``
has no rclpy / onnxruntime dependency so it can be exhaustively covered in
CI without the robot or Isaac Lab.

See docs/superpowers/specs/2026-04-19-phoenix-gate8-mode-switch-design.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real.mode_switch import (
    ModeSwitchCfg,
    State,
    cmd_magnitude,
    initial_state,
    step,
)


@pytest.fixture
def cfg() -> ModeSwitchCfg:
    return ModeSwitchCfg(
        enter_walk_thresh=0.15,
        enter_stand_thresh=0.05,
        yaw_scale=0.3,
        transition_ticks=25,
    )


# --- cmd_magnitude ----------------------------------------------------------


def test_cmd_magnitude_uses_lin_vel_norm(cfg: ModeSwitchCfg) -> None:
    # Pure linear command: magnitude = ||(vx, vy)||_2
    assert cmd_magnitude(np.array([0.3, 0.4, 0.0]), cfg.yaw_scale) == pytest.approx(0.5)


def test_cmd_magnitude_scales_yaw(cfg: ModeSwitchCfg) -> None:
    # Pure yaw command: magnitude = |vyaw| * yaw_scale
    # 1.0 rad/s * 0.3 = 0.3 m/s-equivalent
    assert cmd_magnitude(np.array([0.0, 0.0, 1.0]), cfg.yaw_scale) == pytest.approx(0.3)


def test_cmd_magnitude_max_of_lin_and_scaled_yaw(cfg: ModeSwitchCfg) -> None:
    # Combined: max of lin norm and scaled |yaw|.
    # lin = 0.2, yaw-scaled = 0.3*0.3 = 0.09 → 0.2 wins.
    assert cmd_magnitude(np.array([0.2, 0.0, 0.3]), cfg.yaw_scale) == pytest.approx(0.2)


# --- initial_state ---------------------------------------------------------


def test_boots_in_stand(cfg: ModeSwitchCfg) -> None:
    s, ticks = initial_state(cfg)
    assert s is State.STAND
    assert ticks == 0


# --- step transitions ------------------------------------------------------


def test_stand_stays_below_enter_walk_threshold(cfg: ModeSwitchCfg) -> None:
    cmd = np.array([0.10, 0.0, 0.0])  # magnitude 0.10, between 0.05 and 0.15
    s, ticks, alpha = step(State.STAND, ticks_in_state=5, cmd_vel=cmd, cfg=cfg)
    assert s is State.STAND
    assert ticks == 6
    assert alpha == 0.0


def test_stand_enters_walk_on_threshold_cross(cfg: ModeSwitchCfg) -> None:
    cmd = np.array([0.20, 0.0, 0.0])  # above enter_walk_thresh
    s, ticks, alpha = step(State.STAND, ticks_in_state=10, cmd_vel=cmd, cfg=cfg)
    assert s is State.TRANS_TO_WALK
    assert ticks == 0
    assert alpha == 0.0  # tick 0 of blend → still publishing pure STAND target


def test_trans_to_walk_progresses_blend(cfg: ModeSwitchCfg) -> None:
    cmd = np.array([0.20, 0.0, 0.0])
    # After 10 ticks in the transition, alpha = 10/25
    s, ticks, alpha = step(State.TRANS_TO_WALK, ticks_in_state=10, cmd_vel=cmd, cfg=cfg)
    assert s is State.TRANS_TO_WALK
    assert ticks == 11
    assert alpha == pytest.approx(10 / 25)


def test_trans_to_walk_completes(cfg: ModeSwitchCfg) -> None:
    cmd = np.array([0.20, 0.0, 0.0])
    # At transition_ticks-1 → transition completes; next state is WALK.
    # Alpha on the exit tick is (transition_ticks-1)/transition_ticks — the
    # final blend step, published by the caller this tick. The NEXT tick
    # the caller sees state=WALK and publishes walk_target directly.
    s, ticks, alpha = step(
        State.TRANS_TO_WALK, ticks_in_state=cfg.transition_ticks - 1, cmd_vel=cmd, cfg=cfg
    )
    assert s is State.WALK
    assert ticks == 0
    assert alpha == pytest.approx((cfg.transition_ticks - 1) / cfg.transition_ticks)


def test_walk_returns_to_trans_to_stand_below_threshold(cfg: ModeSwitchCfg) -> None:
    cmd = np.array([0.03, 0.0, 0.0])  # below enter_stand_thresh
    s, ticks, alpha = step(State.WALK, ticks_in_state=100, cmd_vel=cmd, cfg=cfg)
    assert s is State.TRANS_TO_STAND
    assert ticks == 0
    assert alpha == 0.0


def test_walk_stays_above_enter_stand_threshold(cfg: ModeSwitchCfg) -> None:
    cmd = np.array([0.10, 0.0, 0.0])  # in hysteresis band, > enter_stand
    s, ticks, alpha = step(State.WALK, ticks_in_state=100, cmd_vel=cmd, cfg=cfg)
    assert s is State.WALK
    assert ticks == 101
    assert alpha == 0.0


def test_trans_to_stand_completes(cfg: ModeSwitchCfg) -> None:
    cmd = np.array([0.0, 0.0, 0.0])
    s, ticks, alpha = step(
        State.TRANS_TO_STAND, ticks_in_state=cfg.transition_ticks - 1, cmd_vel=cmd, cfg=cfg
    )
    assert s is State.STAND
    assert ticks == 0
    assert alpha == pytest.approx((cfg.transition_ticks - 1) / cfg.transition_ticks)


# --- hysteresis --------------------------------------------------------------


def test_hysteresis_no_flutter_in_band(cfg: ModeSwitchCfg) -> None:
    """Commands oscillating inside [enter_stand, enter_walk] must not flip state."""
    # Start in STAND, oscillate in-band for many ticks — stays STAND.
    s, ticks = State.STAND, 0
    for k in range(100):
        cmd = np.array([0.07 if k % 2 == 0 else 0.12, 0.0, 0.0])
        s, ticks, _ = step(s, ticks, cmd, cfg)
    assert s is State.STAND

    # Same experiment starting in WALK — stays WALK.
    s, ticks = State.WALK, 0
    for k in range(100):
        cmd = np.array([0.07 if k % 2 == 0 else 0.12, 0.0, 0.0])
        s, ticks, _ = step(s, ticks, cmd, cfg)
    assert s is State.WALK


# --- blend_alpha invariants --------------------------------------------------


def test_blend_alpha_monotonic_increasing_during_trans(cfg: ModeSwitchCfg) -> None:
    """alpha must be non-decreasing while still in TRANS_TO_WALK."""
    cmd = np.array([0.20, 0.0, 0.0])
    alphas = []
    s, ticks = State.TRANS_TO_WALK, 0
    while s is State.TRANS_TO_WALK:
        s_next, ticks_next, alpha = step(s, ticks, cmd, cfg)
        alphas.append(alpha)
        s, ticks = s_next, ticks_next
    # Should see N alphas: 0/N .. (N-1)/N. The (N-1)/N tick also transitions out to WALK.
    assert len(alphas) == cfg.transition_ticks
    assert all(alphas[i] < alphas[i + 1] for i in range(len(alphas) - 1))
    assert alphas[0] == 0.0
    assert alphas[-1] == pytest.approx((cfg.transition_ticks - 1) / cfg.transition_ticks)
    assert max(alphas) < 1.0  # alpha never hits 1.0; destination pure-state handles that


def test_blend_alpha_zero_in_pure_states(cfg: ModeSwitchCfg) -> None:
    cmd_walk = np.array([0.20, 0.0, 0.0])
    cmd_stand = np.array([0.0, 0.0, 0.0])
    _, _, alpha_s = step(State.STAND, 0, cmd_stand, cfg)
    _, _, alpha_w = step(State.WALK, 0, cmd_walk, cfg)
    assert alpha_s == 0.0
    assert alpha_w == 0.0


# --- interrupted transitions ------------------------------------------------
# Contract (documented in spec §"State machine"): transitions RUN TO COMPLETION.
# A reverse command received mid-transition does not flip direction. The
# policy completes the current blend, enters the destination pure state for
# at least one tick, then re-evaluates and potentially enters the reverse
# transition. This keeps blend math simple and avoids alpha-direction ambiguity.


def test_trans_to_walk_completes_even_if_cmd_drops_mid_blend(cfg: ModeSwitchCfg) -> None:
    cmd_walking = np.array([0.20, 0.0, 0.0])
    cmd_stopping = np.array([0.0, 0.0, 0.0])
    s, ticks = State.STAND, 0
    # Enter the transition.
    s, ticks, _ = step(s, ticks, cmd_walking, cfg)
    assert s is State.TRANS_TO_WALK
    # Mid-transition, cmd drops to zero.
    for _ in range(5):
        s, ticks, _ = step(s, ticks, cmd_stopping, cfg)
    # Still completing the transition, not reversed.
    assert s is State.TRANS_TO_WALK
    # Finish the transition.
    for _ in range(cfg.transition_ticks):
        s, ticks, _ = step(s, ticks, cmd_stopping, cfg)
    # Now in WALK briefly, then TRANS_TO_STAND on the re-evaluation tick.
    assert s in (State.WALK, State.TRANS_TO_STAND)


def test_yaw_only_cmd_enters_walk_via_yaw_scale(cfg: ModeSwitchCfg) -> None:
    # |vyaw| = 0.6 rad/s → scaled magnitude = 0.6*0.3 = 0.18 > 0.15
    cmd = np.array([0.0, 0.0, 0.6])
    s, ticks, _ = step(State.STAND, 0, cmd, cfg)
    assert s is State.TRANS_TO_WALK


# --- rollout integration ---------------------------------------------------
# Exercises the full state machine against a cmd sequence that matches the
# CaresLab hardware gate 8a plan: 100 ticks of cmd=0, 100 ticks of cmd=0.3,
# 100 ticks of cmd=0. Expect exactly one STAND→WALK and one WALK→STAND
# transition, each 25 ticks.


def test_rollout_cmd_step_0_03_0(cfg: ModeSwitchCfg) -> None:
    cmds = (
        [np.array([0.0, 0.0, 0.0])] * 100
        + [np.array([0.3, 0.0, 0.0])] * 100
        + [np.array([0.0, 0.0, 0.0])] * 100
    )

    s, ticks = initial_state(cfg)
    state_trace: list[State] = []
    alpha_trace: list[float] = []
    for cmd in cmds:
        # Record the state *before* advancing (this is what the caller would
        # have used to compute this tick's published target).
        state_trace.append(s)
        s, ticks, alpha = step(s, ticks, cmd, cfg)
        alpha_trace.append(alpha)

    # Partition trace into runs of consecutive identical states.
    from itertools import groupby

    runs = [(k, sum(1 for _ in g)) for k, g in groupby(state_trace)]

    # Expect STAND → TRANS_TO_WALK → WALK → TRANS_TO_STAND → STAND.
    run_states = [r[0] for r in runs]
    assert run_states == [
        State.STAND,
        State.TRANS_TO_WALK,
        State.WALK,
        State.TRANS_TO_STAND,
        State.STAND,
    ]

    # Transitions must each span exactly transition_ticks.
    run_lengths = {s: n for s, n in runs if s in (State.TRANS_TO_WALK, State.TRANS_TO_STAND)}
    # Note: because of the "exit tick also publishes the last blend step"
    # semantics, TRANS_TO_WALK shows up for transition_ticks ticks, then
    # state flips to WALK on the following tick.
    assert run_lengths[State.TRANS_TO_WALK] == cfg.transition_ticks
    assert run_lengths[State.TRANS_TO_STAND] == cfg.transition_ticks

    # Every TRANS_TO_WALK alpha is inside [0, 1).
    trans_walk_alphas = [
        a for s_, a in zip(state_trace, alpha_trace, strict=True) if s_ is State.TRANS_TO_WALK
    ]
    assert len(trans_walk_alphas) == cfg.transition_ticks
    assert 0.0 <= min(trans_walk_alphas) < 1.0
    assert 0.0 <= max(trans_walk_alphas) < 1.0


def test_rollout_target_blend_is_slew_bounded_with_clip(cfg: ModeSwitchCfg) -> None:
    """Blended targets with per-step clip must respect the ±0.175 rad/step slew bound.

    Stand target and walk target differ by 0.5 rad on one joint (a step
    much larger than 0.175 rad). Across the 25-tick blend the unclipped
    per-step delta on that joint is 0.5/25 = 0.02 rad — well under the
    clip, so the clip shouldn't bite. But we run the test through the
    clip anyway to document the contract.
    """
    from phoenix.sim2real.safety import MAX_DELTA_PER_STEP_RAD, per_step_clip_array

    stand_target = np.zeros(12, dtype=np.float32)
    walk_target = np.zeros(12, dtype=np.float32)
    walk_target[0] = 0.5  # differ by 0.5 rad on joint 0

    # Simulate: q starts equal to stand_target, we hold cmd=walking for
    # enough ticks to drive one full transition and beyond.
    cmd = np.array([0.2, 0.0, 0.0])
    s, ticks = initial_state(cfg)
    q = stand_target.copy()
    deltas: list[float] = []
    for _ in range(50):
        # Step state machine based on THIS tick's cmd.
        prev_state = s
        s_next, ticks_next, alpha = step(s, ticks, cmd, cfg)
        if prev_state is State.STAND:
            target = stand_target
        elif prev_state is State.WALK:
            target = walk_target
        elif prev_state is State.TRANS_TO_WALK:
            target = (1.0 - alpha) * stand_target + alpha * walk_target
        else:
            target = (1.0 - alpha) * walk_target + alpha * stand_target
        clipped = per_step_clip_array(target, q, MAX_DELTA_PER_STEP_RAD)
        deltas.append(float(np.max(np.abs(clipped - q))))
        q = clipped
        s, ticks = s_next, ticks_next

    assert max(deltas) <= MAX_DELTA_PER_STEP_RAD + 1e-6
    # q should have converged to walk_target by the end.
    np.testing.assert_allclose(q, walk_target, atol=1e-6)
