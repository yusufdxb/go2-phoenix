"""Tests for the deployment-mismatch ablation transformations.

Pure numpy, no simulator. These pin the arithmetic that lets an existing
checkpoint be evaluated under a historical deploy bug without retraining, so a
silent error here would invalidate the whole ablation.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.reliability.deploy_ablation import (
    FLAT_OBS_DIM,
    HISTORICAL_HIP_POSE,
    N_JOINTS,
    TRAINING_HIP_POSE,
    AblationSpec,
    apply_action_ablation,
    apply_observation_ablation,
    default_grid,
    deploy_slew_saturation,
    hip_offset_action_shift,
    hip_offset_observation_shift,
    zero_base_lin_vel,
)


def obs(n: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, FLAT_OBS_DIM))


def test_zero_base_lin_vel_zeroes_only_the_first_three_dims() -> None:
    src = obs()
    out = zero_base_lin_vel(src)
    assert np.all(out[:, 0:3] == 0.0)
    assert np.allclose(out[:, 3:], src[:, 3:])


def test_transformations_do_not_mutate_their_input() -> None:
    src = obs()
    before = src.copy()
    zero_base_lin_vel(src)
    hip_offset_observation_shift(src)
    assert np.array_equal(src, before)


def test_rough_terrain_observation_is_refused_rather_than_mis_sliced() -> None:
    with pytest.raises(ValueError, match="235|flat observation"):
        zero_base_lin_vel(np.zeros((2, 235)))


def test_hip_observation_shift_moves_only_the_four_hip_entries() -> None:
    src = obs()
    out = hip_offset_observation_shift(src)
    delta = out - src
    # joint_pos_rel occupies 12:24; hips are its first four entries.
    assert np.allclose(delta[:, 12:16], TRAINING_HIP_POSE - HISTORICAL_HIP_POSE)
    assert np.allclose(delta[:, 0:12], 0.0)
    assert np.allclose(delta[:, 16:], 0.0)


def test_hip_action_shift_reproduces_the_deployed_target_exactly() -> None:
    """The whole point: sim must emit the target a buggy deploy config would.

    Deploy computes target = deployed_default + scale * action.
    Sim computes  target = training_default + scale * action_sim.
    The shift must make those two equal.
    """
    scale = 0.25
    rng = np.random.default_rng(1)
    action = rng.normal(size=(5, N_JOINTS))

    training_default = np.zeros(N_JOINTS)
    training_default[0:4] = TRAINING_HIP_POSE
    deployed_default = np.zeros(N_JOINTS)
    deployed_default[0:4] = HISTORICAL_HIP_POSE

    deployed_target = deployed_default + scale * action
    action_sim = hip_offset_action_shift(action, scale)
    sim_target = training_default + scale * action_sim

    assert np.allclose(sim_target, deployed_target)


def test_hip_action_shift_is_identity_when_the_poses_agree() -> None:
    action = np.ones((3, N_JOINTS))
    out = hip_offset_action_shift(
        action, 0.25, training_hips=TRAINING_HIP_POSE, deployed_hips=TRAINING_HIP_POSE
    )
    assert np.allclose(out, action)


def test_hip_action_shift_rejects_a_zero_scale() -> None:
    with pytest.raises(ValueError):
        hip_offset_action_shift(np.zeros((2, N_JOINTS)), 0.0)


# --------------------------------------------------------------------------
# The deploy-equivalent slew metric
# --------------------------------------------------------------------------


def test_deploy_slew_counts_target_versus_measured_not_action_delta() -> None:
    """A large action delta that lands close to measured q must NOT saturate."""
    default_q = np.zeros(N_JOINTS)
    scale = 1.0
    # Successive actions differ hugely, but each target sits exactly on the
    # measured joint position, so deploy would clip nothing.
    actions = np.array([[0.0] * N_JOINTS, [5.0] * N_JOINTS])
    measured = np.array([[0.0] * N_JOINTS, [5.0] * N_JOINTS])
    assert deploy_slew_saturation(actions, measured, default_q, scale, 0.175) == 0.0


def test_deploy_slew_flags_targets_beyond_the_cap() -> None:
    default_q = np.zeros(N_JOINTS)
    actions = np.array([[1.0] * N_JOINTS])
    measured = np.array([[0.0] * N_JOINTS])
    assert deploy_slew_saturation(actions, measured, default_q, 1.0, 0.175) == 1.0


def test_deploy_slew_honours_the_default_pose_offset() -> None:
    """default_q is part of the target, so it must shift saturation."""
    actions = np.zeros((1, N_JOINTS))
    measured = np.zeros((1, N_JOINTS))
    no_offset = deploy_slew_saturation(actions, measured, np.zeros(N_JOINTS), 1.0, 0.175)
    with_offset = deploy_slew_saturation(actions, measured, np.full(N_JOINTS, 0.5), 1.0, 0.175)
    assert no_offset == 0.0
    assert with_offset == 1.0


def test_deploy_slew_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        deploy_slew_saturation(
            np.zeros((2, N_JOINTS)), np.zeros((3, N_JOINTS)), np.zeros(N_JOINTS), 1.0, 0.175
        )


def test_deploy_slew_rejects_a_nonpositive_cap() -> None:
    with pytest.raises(ValueError):
        deploy_slew_saturation(
            np.zeros((1, N_JOINTS)), np.zeros((1, N_JOINTS)), np.zeros(N_JOINTS), 1.0, 0.0
        )


# --------------------------------------------------------------------------
# Grid composition
# --------------------------------------------------------------------------


def test_default_grid_covers_control_singles_pairs_and_all_on() -> None:
    names = [s.name for s in default_grid()]
    assert "correct" in names
    assert len(names) == len(set(names))
    control = next(s for s in default_grid() if s.name == "correct")
    assert not any(
        [control.zero_base_lin_vel, control.historical_hip_offset, control.enforce_deploy_limiter]
    )
    all_on = next(s for s in default_grid() if s.name == "all_three")
    assert all(
        [all_on.zero_base_lin_vel, all_on.historical_hip_offset, all_on.enforce_deploy_limiter]
    )


def test_control_cell_is_a_no_op_on_both_paths() -> None:
    spec = AblationSpec("correct")
    src = obs()
    act = np.ones((4, N_JOINTS))
    assert np.allclose(apply_observation_ablation(src, spec), src)
    assert np.allclose(apply_action_ablation(act, spec, 0.25), act)


def test_combined_cell_applies_both_observation_transforms() -> None:
    spec = AblationSpec("both", zero_base_lin_vel=True, historical_hip_offset=True)
    src = obs()
    out = apply_observation_ablation(src, spec)
    assert np.all(out[:, 0:3] == 0.0)
    assert np.allclose(out[:, 12:16] - src[:, 12:16], TRAINING_HIP_POSE - HISTORICAL_HIP_POSE)


def test_spec_serializes_for_machine_readable_output() -> None:
    d = AblationSpec("x", zero_base_lin_vel=True).as_dict()
    assert d == {
        "name": "x",
        "zero_base_lin_vel": True,
        "historical_hip_offset": False,
        "enforce_deploy_limiter": False,
    }
