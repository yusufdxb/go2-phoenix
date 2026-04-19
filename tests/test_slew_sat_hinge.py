"""Tests for phoenix.sim_env.rewards.slew_sat_hinge_l2.

Pure-torch tests; no Isaac Lab dependency. Uses a trivial stand-in
`env` object whose only interface is `env.action_manager.action` and
`env.action_manager.prev_action` — matches how upstream Isaac Lab
action-rate rewards access actions (see
IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:action_rate_l2).
"""

from __future__ import annotations

import pytest
import torch

from phoenix.sim_env.rewards import slew_sat_hinge_l2


class _FakeActionManager:
    def __init__(self, action: torch.Tensor, prev: torch.Tensor):
        self.action = action
        self.prev_action = prev


class _FakeEnv:
    def __init__(self, action: torch.Tensor, prev: torch.Tensor):
        self.action_manager = _FakeActionManager(action, prev)


def test_zero_below_threshold() -> None:
    # All deltas = 0.1 (< 0.15 threshold)
    prev   = torch.zeros(2, 12)
    action = torch.full((2, 12), 0.1)
    env = _FakeEnv(action, prev)
    r = slew_sat_hinge_l2(env)
    assert torch.allclose(r, torch.zeros(2))


def test_squared_above_threshold() -> None:
    # Env 0: one motor at 0.175 (at clip), rest zero
    prev = torch.zeros(1, 12)
    action = torch.zeros(1, 12)
    action[0, 3] = 0.175
    env = _FakeEnv(action, prev)
    r = slew_sat_hinge_l2(env)
    # excess = 0.025, squared = 6.25e-4
    assert torch.allclose(r, torch.tensor([0.025 ** 2]), atol=1e-8)


def test_sums_across_motors() -> None:
    # 3 motors at 0.175 -> 3 * (0.025)^2
    prev = torch.zeros(1, 12)
    action = torch.zeros(1, 12)
    action[0, 0] = 0.175
    action[0, 5] = 0.175
    action[0, 11] = 0.175
    env = _FakeEnv(action, prev)
    r = slew_sat_hinge_l2(env)
    assert torch.allclose(r, torch.tensor([3 * 0.025 ** 2]), atol=1e-8)


def test_per_env_independent() -> None:
    # Env 0: all quiet. Env 1: one motor at clip.
    prev = torch.zeros(2, 12)
    action = torch.zeros(2, 12)
    action[1, 0] = 0.175
    env = _FakeEnv(action, prev)
    r = slew_sat_hinge_l2(env)
    assert r.shape == (2,)
    assert torch.allclose(r[0], torch.tensor(0.0))
    assert torch.allclose(r[1], torch.tensor(0.025 ** 2), atol=1e-8)


def test_threshold_parameter() -> None:
    # Motor at 0.175. With threshold=0.175 (strict >), penalty must be 0.
    prev = torch.zeros(1, 12)
    action = torch.zeros(1, 12)
    action[0, 0] = 0.175
    env = _FakeEnv(action, prev)
    r = slew_sat_hinge_l2(env, threshold=0.175)
    assert torch.allclose(r, torch.zeros(1))


def test_default_hinge_threshold_tracks_hardware_slew_clip() -> None:
    """The default hinge threshold is designed to sit 0.025 rad below
    the hardware slew clip MAX_DELTA_PER_STEP_RAD. If either constant
    changes without the other, the safety margin drifts silently; this
    test catches that."""
    from phoenix.sim2real.safety import MAX_DELTA_PER_STEP_RAD
    from phoenix.sim_env.rewards import _DEFAULT_HINGE_THRESHOLD

    margin = MAX_DELTA_PER_STEP_RAD - _DEFAULT_HINGE_THRESHOLD
    assert margin == pytest.approx(0.025, abs=1e-9), (
        f"Expected 0.025 rad margin between _DEFAULT_HINGE_THRESHOLD "
        f"({_DEFAULT_HINGE_THRESHOLD}) and MAX_DELTA_PER_STEP_RAD "
        f"({MAX_DELTA_PER_STEP_RAD}); got {margin}. If hardware clip "
        f"changed, update the hinge threshold to match."
    )
