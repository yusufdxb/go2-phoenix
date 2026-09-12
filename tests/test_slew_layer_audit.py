"""clip_layer_audit: what a second, lagged slew clip does to an already clipped command."""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.training.slew import clip_layer_audit


def _audit(actions, q):
    return clip_layer_audit(
        actions=np.asarray(actions, dtype=np.float64),
        measured_q=np.asarray(q, dtype=np.float64),
        default_q=np.zeros(np.asarray(actions).shape[1]),
        action_scale=1.0,
    )


def test_static_joints_second_clip_never_binds() -> None:
    actions = [[0.5, 0.0], [0.5, 0.0], [0.5, 0.0]]
    q = [[0.0, 0.0]] * 3
    out = _audit(actions, q)
    assert out["policy_layer_pct"] == pytest.approx(50.0)  # joint 0 clipped, joint 1 not
    assert out["second_layer_pct"] == 0.0
    assert out["end_to_end_pct"] == pytest.approx(50.0)
    assert out["mean_second_adjust_rad"] == 0.0


def test_joint_moving_away_rebinds_an_already_clipped_target() -> None:
    actions = [[0.5], [0.5]]
    q = [[0.0], [-0.01]]
    out = _audit(actions, q)
    assert out["policy_layer_pct"] == pytest.approx(100.0)
    assert out["second_layer_pct"] == pytest.approx(100.0)
    assert out["second_only_pct"] == 0.0
    assert out["mean_second_adjust_rad"] == pytest.approx(0.01)


def test_second_clip_can_bind_where_the_first_did_not() -> None:
    actions = [[0.17], [0.17]]
    q = [[0.0], [-0.01]]
    out = _audit(actions, q)
    assert out["policy_layer_pct"] == 0.0
    assert out["second_only_pct"] == pytest.approx(100.0)
    assert out["end_to_end_pct"] == pytest.approx(100.0)


def test_shape_and_lag_validation() -> None:
    with pytest.raises(ValueError):
        clip_layer_audit(
            actions=np.zeros((3, 2)),
            measured_q=np.zeros((3, 3)),
            default_q=np.zeros(2),
            action_scale=1.0,
        )
    with pytest.raises(ValueError):
        clip_layer_audit(
            actions=np.zeros((2, 2)),
            measured_q=np.zeros((2, 2)),
            default_q=np.zeros(2),
            action_scale=1.0,
            lag_steps=2,
        )
