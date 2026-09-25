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
    spec = s.default_spec().replace(action=dataclasses.replace(s.default_spec().action, clip_actions=1.0))
    restored = s.spec_from_dict(spec.to_dict())
    assert restored.action.clip_actions == 1.0


def test_validate_rejects_non_positive_clip_actions() -> None:
    spec = s.default_spec().replace(action=dataclasses.replace(s.default_spec().action, clip_actions=0.0))
    problems = spec.validate()
    assert any("clip_actions" in p for p in problems)

    spec_neg = s.default_spec().replace(action=dataclasses.replace(s.default_spec().action, clip_actions=-5.0))
    assert any("clip_actions" in p for p in spec_neg.validate())


def test_default_spec_still_validates_clean() -> None:
    assert s.default_spec().validate() == []


def test_smoke_spec_still_validates_clean() -> None:
    assert s.smoke_spec().validate() == []


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
