"""The PhoenixVelocity 45-D contract and the fail-closed checkpoint manifest."""

from __future__ import annotations

import copy

import pytest

from phoenix.velocity import contract as c


def _manifest(**over) -> dict:
    cmds = over.pop(
        "commands",
        c.CommandRanges((-1.0, 1.0), (-0.5, 0.5), (-1.0, 1.0), 0.1),
    )
    m = c.build_manifest(
        checkpoint_sha256="a" * 64,
        commands=cmds,
        git_sha="b" * 40,
        git_dirty=False,
        seed=1,
        task="Phoenix-Velocity-Flat-Go2-v0",
        simulator="isaaclab",
        reward_scales={"track_lin_vel_xy": 1.0},
        domain_randomization={},
        curriculum={},
    )
    m.update(over)
    return m


ENVELOPE = {"lin_vel_x": 0.3, "lin_vel_y": 0.0, "ang_vel_z": 0.3}


def test_actor_observation_is_45_dims_in_order_without_base_lin_vel() -> None:
    assert c.ACTOR_OBS_DIM == 45
    names = [t[0] for t in c.ACTOR_OBS_TERMS]
    assert names == [
        "base_ang_vel",
        "projected_gravity",
        "velocity_command",
        "joint_pos_rel",
        "joint_vel",
        "last_action",
    ]
    assert "base_lin_vel" not in names
    assert not c.FORBIDDEN_ACTOR_TERMS.intersection(names)
    sl = c.obs_slices()
    assert sl["base_ang_vel"] == slice(0, 3)
    assert sl["projected_gravity"] == slice(3, 6)
    assert sl["velocity_command"] == slice(6, 9)
    assert sl["joint_pos_rel"] == slice(9, 21)
    assert sl["joint_vel"] == slice(21, 33)
    assert sl["last_action"] == slice(33, 45)


def test_control_constants() -> None:
    assert c.CONTROL_HZ == 50 and c.PHYSICS_HZ == 200 and c.DECIMATION == 4
    assert c.ACTION_SCALE == 0.25 and c.ACTION_DIM == 12


def test_schema_hash_is_stable_and_sensitive() -> None:
    h = c.obs_schema_sha256()
    assert h == c.obs_schema_sha256() and len(h) == 64
    d = c.obs_schema_dict()
    d["joint_order"] = list(reversed(d["joint_order"]))
    assert c._canonical_sha256(d) != h


def test_zero_command_training_is_not_locomotion_capable() -> None:
    h25 = c.CommandRanges((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 1.0)
    capable, reasons = c.derive_locomotion_capable(h25)
    assert capable is False and len(reasons) == 3
    m = _manifest(commands=h25)
    assert m["locomotion_capable"] is False and m["classification"] == "stand_only"
    probs = c.validate_manifest_for_mode(m, c.MODE_VELOCITY, max_deploy_command=ENVELOPE)
    assert any("not locomotion-capable" in p for p in probs)
    # A stand-only checkpoint may still be used as a stand controller.
    assert c.validate_manifest_for_mode(m, c.MODE_STAND) == []


def test_good_velocity_manifest_passes() -> None:
    m = _manifest()
    assert m["locomotion_capable"] is True
    assert (
        c.validate_manifest_for_mode(
            m,
            c.MODE_VELOCITY,
            deploy_action_scale=0.25,
            deploy_control_hz=50,
            deploy_joint_order=c.JOINT_ORDER,
            max_deploy_command=ENVELOPE,
            checkpoint_sha256="a" * 64,
        )
        == []
    )


def test_forged_locomotion_flag_is_refused() -> None:
    h25 = c.CommandRanges((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 1.0)
    m = _manifest(commands=h25)
    m["locomotion_capable"] = True
    m["classification"] = "velocity_candidate"
    probs = c.validate_manifest_for_mode(m, c.MODE_VELOCITY, max_deploy_command=ENVELOPE)
    assert any("disagrees" in p for p in probs)
    assert any("not locomotion-capable" in p for p in probs)


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda m: m.__setitem__("schema", "legacy"), "manifest schema"),
        (lambda m: m["observation"].__setitem__("actor_dim", 48), "actor observation dim"),
        (
            lambda m: m["observation"].__setitem__(
                "actor_terms", ["base_lin_vel"] + m["observation"]["actor_terms"]
            ),
            "unavailable on hardware",
        ),
        (lambda m: m["observation"].__setitem__("schema_sha256", "0" * 64), "hash"),
        (lambda m: m["action"].__setitem__("scale", 0.5), "action scale"),
        (lambda m: m.__setitem__("control_hz", 100), "control_hz"),
        (lambda m: m.__setitem__("joint_order", list(reversed(m["joint_order"]))), "joint_order"),
        (lambda m: m["default_joint_pos"].__setitem__("FL_calf_joint", -1.4), "default_joint_pos"),
        (lambda m: m["provenance"].__setitem__("git_sha", "unknown"), "git_sha"),
        (lambda m: m["provenance"].__setitem__("git_dirty", True), "dirty"),
        (lambda m: m.pop("commands"), "commands record"),
    ],
)
def test_every_mismatch_fails_closed(mutate, needle) -> None:
    m = copy.deepcopy(_manifest())
    mutate(m)
    probs = c.validate_manifest_for_mode(m, c.MODE_VELOCITY, max_deploy_command=ENVELOPE)
    assert any(needle in p for p in probs), probs


def test_missing_manifest_and_unknown_mode_refuse() -> None:
    assert c.validate_manifest_for_mode(None, c.MODE_STAND)
    assert c.validate_manifest_for_mode(_manifest(), "walk_fast")


def test_deploy_side_mismatches_refuse() -> None:
    m = _manifest()
    probs = c.validate_manifest_for_mode(
        m,
        c.MODE_VELOCITY,
        deploy_action_scale=0.3,
        deploy_control_hz=100,
        deploy_joint_order=list(reversed(c.JOINT_ORDER)),
        max_deploy_command=ENVELOPE,
        checkpoint_sha256="c" * 64,
    )
    joined = " | ".join(probs)
    for needle in ("deploy action_scale", "deploy control rate", "deploy joint_order", "hash"):
        assert needle in joined


def test_deploy_envelope_must_fit_trained_ranges() -> None:
    m = _manifest()
    probs = c.validate_manifest_for_mode(
        m, c.MODE_VELOCITY, max_deploy_command={"lin_vel_x": 1.5, "lin_vel_y": 0, "ang_vel_z": 0}
    )
    assert any("exceeds the trained envelope" in p for p in probs)
    assert any(
        "needs the deploy command envelope" in p
        for p in c.validate_manifest_for_mode(m, c.MODE_VELOCITY)
    )


def test_command_ranges_reject_garbage() -> None:
    with pytest.raises(ValueError):
        c.CommandRanges.from_mapping(
            {"lin_vel_x": [1, -1], "lin_vel_y": [0, 0], "ang_vel_z": [0, 0], "rel_standing_envs": 0}
        )
    with pytest.raises(ValueError):
        c.CommandRanges.from_mapping(
            {"lin_vel_x": [0, 1], "lin_vel_y": [0, 0], "ang_vel_z": [0, 0], "rel_standing_envs": 2}
        )
