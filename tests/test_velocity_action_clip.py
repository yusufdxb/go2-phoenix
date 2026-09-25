"""ActionSpec.clip_actions: the 2026-09-25 fix for the clip=1.0 curriculum-stall bug."""

from __future__ import annotations

import dataclasses

from phoenix.velocity import spec as s


def test_default_clip_actions_is_wide_not_unit() -> None:
    # legged_gym convention (rl_sar / robot_lab / himloco use +-100): the rsl_rl
    # wrapper clip is a dimensionless sanity clip, not a radians limit. 1.0 with
    # action_scale=0.25 caps every joint offset at 0.25 rad and stalled training
    # (seed 42, iteration 2965: action_abs_mean 0.9985, curriculum stuck level 0).
    default = s.default_spec()
    assert default.action.clip_actions == 100.0
    assert default.action.scale == 0.25


def test_clip_actions_round_trips_through_to_dict_and_back() -> None:
    spec = s.default_spec().replace(
        action=dataclasses.replace(s.default_spec().action, clip_actions=1.0)
    )
    restored = s.spec_from_dict(spec.to_dict())
    assert restored.action.clip_actions == 1.0


def test_validate_rejects_non_positive_clip_actions() -> None:
    spec = s.default_spec().replace(
        action=dataclasses.replace(s.default_spec().action, clip_actions=0.0)
    )
    problems = spec.validate()
    assert any("clip_actions" in p for p in problems)

    spec_neg = s.default_spec().replace(
        action=dataclasses.replace(s.default_spec().action, clip_actions=-5.0)
    )
    assert any("clip_actions" in p for p in spec_neg.validate())


def test_default_spec_still_validates_clean() -> None:
    assert s.default_spec().validate() == []


def test_smoke_spec_still_validates_clean() -> None:
    assert s.smoke_spec().validate() == []


def test_curriculum_disabled_starts_at_final_ranges() -> None:
    # 2026-09-25: seed 42 (curriculum on) stalled at level 0 (yaw_score 0.31 <
    # 0.5 threshold at iteration ~1972) despite low termination rate and decent
    # absolute tracking error -- the normalized-against-standstill score gate
    # is too strict on the narrow level-0 yaw range. The --curriculum off arm
    # must train on the FINAL ranges from iteration 0, like stock Isaac Lab Go2
    # (no curriculum at all). CommandCurriculum.from_spec sets level=num_levels
    # when spec.enabled is False, so .ranges is exactly final_ranges().
    from phoenix.velocity.curriculum import CommandCurriculum

    spec = s.default_spec().replace(
        curriculum=dataclasses.replace(s.default_spec().curriculum, enabled=False)
    )
    cur = CommandCurriculum.from_spec(
        spec.commands, spec.curriculum, tracking_std=0.5, num_envs=8192
    )
    assert cur.level == cur.max_level
    assert cur.ranges.lin_vel_x == spec.commands.final_lin_vel_x
    assert cur.ranges.lin_vel_y == spec.commands.final_lin_vel_y
    assert cur.ranges.ang_vel_z == spec.commands.final_ang_vel_z


def test_curriculum_enabled_starts_at_initial_ranges() -> None:
    from phoenix.velocity.curriculum import CommandCurriculum

    spec = s.default_spec()
    assert spec.curriculum.enabled is True
    cur = CommandCurriculum.from_spec(
        spec.commands, spec.curriculum, tracking_std=0.5, num_envs=8192
    )
    assert cur.level == 0
    assert cur.ranges.lin_vel_x == spec.commands.initial_lin_vel_x


def test_action_l2_reward_term_present_and_valid() -> None:
    # 2026-09-25: seed 43/44/42-curriculum episode length collapsed over the
    # second half of a 9000-iteration run (e.g. seed 43: 946@2000 -> 60@8500)
    # while action_abs_p99 grew smoothly (4.09 -> 15.49) with action std
    # roughly flat (0.19 -> 0.25). rsl_rl's GaussianDistribution has no output
    # bound on the mean (mean = mlp_output, no tanh/clip), and
    # action_rate_l2/slew_sat_hinge_l2 only penalize CHANGE between steps, so
    # nothing directly bounded the mean's magnitude. action_l2 closes that gap.
    default = s.default_spec()
    assert default.validate() == []
    names = [t.name for t in default.rewards]
    assert "action_l2" in names
    term = default.reward("action_l2")
    assert term.weight < 0.0
    assert term.func == "isaac:action_l2"


def test_entropy_coef_matches_stock_go2_ppo_cfg() -> None:
    # Stock UnitreeGo2{Rough,Flat}PPORunnerCfg uses entropy_coef=0.01; ours was
    # 0.005 with no recorded reason. Secondary alignment while diagnosing the
    # 2026-09-25 collapse, not shown alone to be the cause.
    assert s.default_spec().ppo.entropy_coef == 0.01


def test_with_reward_weight_isolates_one_term() -> None:
    # 2026-09-25: sim2sim found seed42@3000 drives FR/RL calves to their hard
    # limit during CCW yaw. joint_pos_limits weight raised toward the
    # legged_gym/unitree_rl_gym convention (-10.0) for an isolated arm (seed
    # 46) via this method, WITHOUT changing default_spec() (a resumed run,
    # e.g. seed 45, must keep its own original -1.0).
    default = s.default_spec()
    assert default.reward("joint_pos_limits").weight == -1.0

    bumped = default.with_reward_weight("joint_pos_limits", -10.0)
    assert bumped.reward("joint_pos_limits").weight == -10.0
    assert bumped.validate() == []

    # Original untouched (immutability), and only the named term changed.
    assert default.reward("joint_pos_limits").weight == -1.0
    for a, b in zip(default.rewards, bumped.rewards, strict=True):
        if a.name != "joint_pos_limits":
            assert a == b


def test_with_reward_weight_rejects_unknown_name() -> None:
    import pytest

    with pytest.raises(KeyError):
        s.default_spec().with_reward_weight("not_a_real_term", -1.0)


def test_trunk_bodies_is_base_only_not_head() -> None:
    # 2026-09-25 bug (fixed): TRUNK_BODIES used to include "Head_.*". Head_upper/
    # Head_lower are near-massless (0.001 kg) bodies rigidly fixed to the base;
    # PhysX's internal constraint reaction force for keeping them attached during
    # any joint acceleration spikes through the net-contact-force channel (measured
    # up to 533 N with only action_std=0.5 noise and zero real ground contact),
    # which produced a near-100% false-positive trunk_contact termination on any
    # nonzero action (mean episode length ~7 steps vs stock Go2's ~973-993 with
    # the same reward/PPO setup). See scripts/diag_zero_action_rollout.py.
    assert s.TRUNK_BODIES == ("base",)
    assert not any("Head" in b for b in s.TRUNK_BODIES)
