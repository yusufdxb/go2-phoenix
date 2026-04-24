"""Unit tests for the post-export canonical-stand bench."""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real.bench_export import (
    DEFAULT_MAX_ABS_ACTION,
    build_canonical_stand_obs,
    check_canonical_action,
)
from phoenix.sim2real.observation import JointOrder, ObservationBuilder


@pytest.fixture
def builder() -> ObservationBuilder:
    names = [
        "FL_hip_joint",
        "FR_hip_joint",
        "RL_hip_joint",
        "RR_hip_joint",
        "FL_thigh_joint",
        "FR_thigh_joint",
        "RL_thigh_joint",
        "RR_thigh_joint",
        "FL_calf_joint",
        "FR_calf_joint",
        "RL_calf_joint",
        "RR_calf_joint",
    ]
    defaults = {n: 0.0 for n in names}
    for n in ("FL_thigh_joint", "FR_thigh_joint"):
        defaults[n] = 0.8
    for n in ("RL_thigh_joint", "RR_thigh_joint"):
        defaults[n] = 1.0
    for n in ("FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"):
        defaults[n] = -1.5
    return ObservationBuilder(JointOrder(tuple(names)), defaults)


def test_canonical_stand_obs_is_mostly_zero_for_flat_policy(builder):
    obs = build_canonical_stand_obs(builder, pad_zeros=0)
    # 48-dim proprio: 12 zero components, gravity at (0,0,-1), rest zero.
    assert obs.shape == (48,)
    # gravity slot is at [6:9]
    np.testing.assert_allclose(obs[6:9], np.asarray([0.0, 0.0, -1.0]))
    mask = np.ones_like(obs, dtype=bool)
    mask[6:9] = False
    assert np.all(obs[mask] == 0.0)


def test_canonical_stand_obs_padding_for_rough_policy(builder):
    obs = build_canonical_stand_obs(builder, pad_zeros=187)
    assert obs.shape == (48 + 187,)
    assert np.all(obs[48:] == 0.0)


def test_check_canonical_action_pass(builder):
    obs = build_canonical_stand_obs(builder)

    def infer(_: np.ndarray) -> np.ndarray:
        # Small action magnitude — a converged policy output.
        return np.full(12, 0.1, dtype=np.float32)

    report = check_canonical_action(infer, obs)
    assert report.passed
    assert report.max_abs_action == pytest.approx(0.1)
    assert report.threshold == DEFAULT_MAX_ABS_ACTION
    assert report.obs_dim == 48
    assert report.action_dim == 12


def test_check_canonical_action_fail(builder):
    obs = build_canonical_stand_obs(builder)

    def infer(_: np.ndarray) -> np.ndarray:
        # Large action magnitude — under-trained policy output.
        a = np.zeros(12, dtype=np.float32)
        a[3] = 0.9
        return a

    report = check_canonical_action(infer, obs)
    assert not report.passed
    assert report.max_abs_action == pytest.approx(0.9)


def test_check_canonical_action_custom_threshold(builder):
    obs = build_canonical_stand_obs(builder)

    def infer(_: np.ndarray) -> np.ndarray:
        return np.full(12, 0.35, dtype=np.float32)

    assert not check_canonical_action(infer, obs, threshold=0.3).passed
    assert check_canonical_action(infer, obs, threshold=0.5).passed


def test_check_canonical_action_rejects_empty(builder):
    obs = build_canonical_stand_obs(builder)

    def infer(_: np.ndarray) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)

    with pytest.raises(ValueError, match="empty action"):
        check_canonical_action(infer, obs)
