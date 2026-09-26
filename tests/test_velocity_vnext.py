"""PhoenixVelocity vNext: observation schema v2, lin-only curriculum gate, health guard."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from phoenix.velocity import contract as c
from phoenix.velocity.curriculum import CommandCurriculum, standstill_tracking_baseline
from phoenix.velocity.health import ABORT, OK, WARN, HealthSample, TrainingHealthGuard
from phoenix.velocity.observation import ObservationError, build_actor_observation
from phoenix.velocity.spec import (
    LAB_CARD_ENVELOPE,
    TRUNK_BODIES,
    default_spec,
    smoke_spec,
    spec_from_dict,
    vnext_spec,
)

#: obs_schema_sha256(v1) as committed before versioning (ad81c70..21540b1) and as
#: recorded in walk_v2_seed46 model_1500's manifest. Must never change.
V1_SHA = "80f50c6d862931a94bf7298d95d7f44e834cf3d51f8111e9fb26719b36897540"

DEFAULT_Q = np.asarray([c.DEFAULT_JOINT_POS[j] for j in c.JOINT_ORDER])


def _manifest(version: str) -> dict:
    return c.build_manifest(
        checkpoint_sha256="a" * 64,
        commands=c.CommandRanges((-0.3, 0.6), (-0.3, 0.3), (-1.0, 1.0), 0.1),
        git_sha="b" * 40,
        git_dirty=False,
        seed=1,
        task="t",
        simulator="isaaclab",
        reward_scales={},
        domain_randomization={},
        curriculum={},
        obs_schema_version=version,
    )


def _obs_inputs(version: str) -> dict:
    return dict(
        gyro_body=[1.0, -2.0, 3.0],
        quat_wxyz=[1, 0, 0, 0],
        command=[0.5, -0.3, 0.8],
        joint_pos=DEFAULT_Q + 0.01 * np.arange(12),
        joint_vel=np.arange(12) * -2.0,
        last_action=np.linspace(-1, 1, 12),
        obs_schema_version=version,
    )


# ------------------------------------------------------------------ contract / obs


def test_v1_schema_hash_is_unchanged_by_versioning() -> None:
    assert c.obs_schema_sha256(c.OBS_SCHEMA_V1) == V1_SHA
    assert c.obs_schema_sha256() == V1_SHA  # default stays v1 for resumed runs
    assert c.obs_schema_sha256(c.OBS_SCHEMA_V2) != V1_SHA


def test_v2_scales_are_unitree_go2() -> None:
    s = c.actor_obs_scales(c.OBS_SCHEMA_V2)
    assert s == {
        "base_ang_vel": 0.2,
        "projected_gravity": 1.0,
        "velocity_command": 1.0,
        "joint_pos_rel": 1.0,
        "joint_vel": 0.05,
        "last_action": 1.0,
    }
    assert set(c.actor_obs_scales(c.OBS_SCHEMA_V1).values()) == {1.0}
    with pytest.raises(c.UnknownObsSchemaError):
        c.actor_obs_scales("phoenix-velocity-obs/v9")


def test_v2_observation_scales_exactly_two_terms() -> None:
    raw = build_actor_observation(**_obs_inputs(c.OBS_SCHEMA_V1))
    scaled = build_actor_observation(**_obs_inputs(c.OBS_SCHEMA_V2))
    sl = c.obs_slices()
    assert np.allclose(scaled[sl["base_ang_vel"]], 0.2 * raw[sl["base_ang_vel"]])
    assert np.allclose(scaled[sl["joint_vel"]], 0.05 * raw[sl["joint_vel"]])
    for name in ("projected_gravity", "velocity_command", "joint_pos_rel", "last_action"):
        assert np.array_equal(scaled[sl[name]], raw[sl[name]])


def test_observation_schema_is_required_and_checked() -> None:
    kw = _obs_inputs(c.OBS_SCHEMA_V2)
    kw.pop("obs_schema_version")
    with pytest.raises(TypeError):
        build_actor_observation(**kw)
    with pytest.raises(ObservationError):
        build_actor_observation(**_obs_inputs("phoenix-velocity-obs/v9"))


def test_manifest_records_and_validates_both_schemas() -> None:
    env = {"lin_vel_x": 0.5, "lin_vel_y": 0.3, "ang_vel_z": 0.8}
    for v in (c.OBS_SCHEMA_V1, c.OBS_SCHEMA_V2):
        m = _manifest(v)
        assert m["observation"]["actor_scales"] == c.ACTOR_OBS_SCALES[v]
        assert c.manifest_obs_schema(m) == v
        assert c.validate_manifest_for_mode(m, c.MODE_VELOCITY, max_deploy_command=env) == []


def test_manifest_schema_tampering_fails_closed() -> None:
    m = _manifest(c.OBS_SCHEMA_V2)
    m["observation"]["schema_version"] = c.OBS_SCHEMA_V1  # v2 policy labelled raw
    assert c.validate_manifest_for_mode(m, c.MODE_STAND)
    with pytest.raises(c.UnknownObsSchemaError):
        c.manifest_obs_schema(m)
    m = _manifest(c.OBS_SCHEMA_V2)
    m["observation"]["actor_scales"] = dict(m["observation"]["actor_scales"], joint_vel=1.0)
    assert any("actor_scales" in p for p in c.validate_manifest_for_mode(m, c.MODE_STAND))
    m = _manifest(c.OBS_SCHEMA_V2)
    m["observation"]["schema_version"] = "phoenix-velocity-obs/v9"
    assert c.validate_manifest_for_mode(m, c.MODE_STAND)


def test_build_manifest_requires_the_schema() -> None:
    with pytest.raises(TypeError):
        c.build_manifest(  # type: ignore[call-arg]
            checkpoint_sha256=None,
            commands=c.CommandRanges((0, 1), (0, 0), (-1, 1), 0.1),
            git_sha="b" * 40,
            git_dirty=False,
            seed=1,
            task="t",
            simulator="s",
            reward_scales={},
            domain_randomization={},
            curriculum={},
        )


def test_mujoco_runner_refuses_without_schema() -> None:
    from phoenix.sim2sim.mujoco_runner import RunConfig, ZeroActionPolicy, run_scenario

    with pytest.raises(ValueError, match="obs_schema_version"):
        run_scenario(ZeroActionPolicy(), object(), RunConfig())  # type: ignore[arg-type]


def test_controller_fsm_default_builder_needs_schema() -> None:
    # controller_fsm imports phoenix.sim2real.handoff, which exists on the deploy
    # branch but not on this one: skip here, run there.
    fsm_mod = pytest.importorskip("phoenix.sim2real.controller_fsm")

    policy = lambda obs: np.zeros(12)  # noqa: E731
    with pytest.raises(ValueError, match="obs_schema_version"):
        fsm_mod.ControllerFSM(c.JOINT_ORDER, policy=policy)
    fsm = fsm_mod.ControllerFSM(c.JOINT_ORDER, policy=policy, obs_schema_version=c.OBS_SCHEMA_V2)
    kw = _obs_inputs(c.OBS_SCHEMA_V2)
    kw.pop("obs_schema_version")
    assert np.array_equal(
        fsm._obs_builder(**kw), build_actor_observation(**_obs_inputs(c.OBS_SCHEMA_V2))
    )


# --------------------------------------------------------------------------- spec


def test_vnext_spec_is_default_plus_only_the_listed_changes() -> None:
    d, v = default_spec(), vnext_spec()
    assert v.validate() == []
    assert v.obs_schema == c.OBS_SCHEMA_V2 and d.obs_schema == c.OBS_SCHEMA_V1
    assert v.ppo == d.ppo  # entropy 0.01, adaptive, desired_kl, init std: untouched
    assert v.ppo.actor_obs_normalization is False
    assert v.ppo.entropy_coef == 0.01 and v.ppo.schedule == "adaptive"
    assert v.action == d.action and v.action.clip_actions == 100.0
    assert TRUNK_BODIES == ("base",)
    assert v.terminations == d.terminations
    assert v.domain_randomization == d.domain_randomization
    assert v.obs_noise == d.obs_noise
    changed = {t.name for t, u in zip(v.rewards, d.rewards, strict=True) if t != u}
    assert changed == {"joint_pos_limits"}
    assert v.reward("joint_pos_limits").weight == -10.0
    assert v.reward("action_l2").weight == -0.01
    assert v.curriculum.gate == "unitree_lin_kernel" and d.curriculum.gate == "normalized_lin_yaw"
    assert v.health.enabled and not d.health.enabled


def test_vnext_guarantees_the_lab_card_envelope_at_level_zero() -> None:
    v = vnext_spec()
    assert LAB_CARD_ENVELOPE == (0.5, 0.3, 0.8)
    cur = CommandCurriculum.from_spec(v.commands, v.curriculum, 0.5, 4096)
    assert cur.level == 0
    r = cur.ranges
    assert r.lin_vel_x[1] >= 0.5
    assert r.lin_vel_y[0] <= -0.3 and r.lin_vel_y[1] >= 0.3
    assert r.ang_vel_z == (-1.0, 1.0)
    capable, _ = c.derive_locomotion_capable(r)
    assert capable
    # A spec that breaks the guarantee is rejected.
    bad = v.replace(commands=dataclasses.replace(v.commands, initial_lin_vel_y=(-0.2, 0.2)))
    assert any("lin_vel_y" in p for p in bad.validate())
    bad = v.replace(commands=dataclasses.replace(v.commands, initial_ang_vel_z=(-0.5, 0.5)))
    probs = bad.validate()
    assert any("ang_vel_z" in p for p in probs) and any("yaw must be" in p for p in probs)


def test_vnext_round_trips_and_old_spec_json_loads_as_v1() -> None:
    v = vnext_spec()
    assert spec_from_dict(json.loads(v.to_json())) == v
    old = json.loads(default_spec().to_json())
    for k in ("obs_schema", "health"):
        old.pop(k)
    old["curriculum"].pop("gate")
    old["commands"].pop("required_envelope")
    legacy = spec_from_dict(old)
    assert legacy.obs_schema == c.OBS_SCHEMA_V1
    assert legacy.curriculum.gate == "normalized_lin_yaw"
    assert legacy.health.enabled is False
    assert smoke_spec(v).obs_schema == c.OBS_SCHEMA_V2


def test_actor_normalization_is_refused() -> None:
    v = vnext_spec()
    bad = v.replace(ppo=dataclasses.replace(v.ppo, actor_obs_normalization=True))
    assert any("actor_obs_normalization" in p for p in bad.validate())


# --------------------------------------------------------------------- curriculum


def _curriculum(spec):
    return CommandCurriculum.from_spec(spec.commands, spec.curriculum, 0.5, num_envs=64)


def test_unitree_gate_ignores_yaw_and_expands_only_linear() -> None:
    v = vnext_spec()
    cur = _curriculum(v)
    n = cur.window_episodes
    # Yaw kernel 0 (yaw score very negative), lin kernel 0.85, no falls: expand.
    d = cur.observe(n, 0, 0.9 * n, 0.0, 0.85 * n)
    assert d.evaluated and d.expanded and cur.level == 1
    assert d.yaw_score < 0  # telemetry only
    assert cur.ranges.ang_vel_z == (-1.0, 1.0)
    assert cur.ranges.lin_vel_y[1] > 0.3 and cur.ranges.lin_vel_x[1] > 0.6
    # Below the 0.8 survival-weighted kernel: hold.
    d = cur.observe(n, 0, 0.99 * n, 0.99 * n, 0.79 * n)
    assert d.evaluated and not d.expanded and "lin kernel" in d.reason
    # Termination gate still applies.
    d = cur.observe(n, n // 2, 0.9 * n, 0.9 * n, 0.9 * n)
    assert not d.expanded and "termination" in d.reason


def test_unitree_gate_reaches_the_final_envelope() -> None:
    v = vnext_spec()
    cur = _curriculum(v)
    n = cur.window_episodes
    for _ in range(v.curriculum.num_levels + 3):
        cur.observe(n, 0, 0.9 * n, 0.1 * n, 0.9 * n)
    assert cur.level == cur.max_level
    assert cur.ranges.to_dict() == v.commands.final_ranges().to_dict()
    assert cur.manifest_dict()["stepping_rule"]["rule"].startswith("phoenix-velocity-curriculum/v2")


def test_seed46_counterfactual() -> None:
    """Seed 46 at iteration 750: lin_score 0.572 (normalized), yaw_score 0.406,
    termination 0.029, Episode_Reward/track_lin_vel_xy_exp 1.339 (weight 1.5 ->
    survival-weighted kernel 0.893). The v1 gate held level 0 (both lin and yaw
    short); the vNext gate expands on the same numbers."""
    d = default_spec()
    base_lin, base_yaw = standstill_tracking_baseline(d.commands.initial_ranges(), 0.5)
    lin_k = base_lin + 0.572 * (1 - base_lin)
    yaw_k = base_yaw + 0.406 * (1 - base_yaw)
    n = 1000
    legacy = _curriculum(d)
    legacy.window_episodes = n
    dec = legacy.observe(n, 29, lin_k * n, yaw_k * n, (1.339 / 1.5) * n)
    assert dec.evaluated and not dec.expanded
    assert "lin score" in dec.reason and "yaw score" in dec.reason
    # Removing ONLY the yaw blocker from the v1 gate would still have held (0.572 < 0.6).
    no_yaw = default_spec().curriculum
    assert 0.572 < no_yaw.lin_score_threshold
    vn = _curriculum(vnext_spec())
    vn.window_episodes = n
    dec = vn.observe(n, 29, lin_k * n, yaw_k * n, (1.339 / 1.5) * n)
    assert dec.expanded


# -------------------------------------------------------------------- health guard


def _guard(**over):
    spec = dataclasses.replace(vnext_spec().health, **over)
    return TrainingHealthGuard(spec, desired_kl=0.01)


def _healthy(it: int, **over) -> HealthSample:
    base = dict(
        iteration=it,
        std_max=0.26,
        std_mean=0.23,
        action_p99=2.4,
        episode_length=985.0,
        kl=0.01,
        learning_rate=4.4e-4,
        curriculum_level=1.0,
    )
    base.update(over)
    return HealthSample(**base)


def test_guard_healthy_seed46_region_stays_ok() -> None:
    g = _guard()
    for it in range(0, 1600):
        v = g.update(_healthy(it))
        assert v.status == OK, v


def test_guard_ignores_warmup_and_needs_a_sustained_breach() -> None:
    g = _guard()
    # std 0.99 at init (warmup) never trips.
    for it in range(0, 250):
        assert g.update(_healthy(it, std_max=0.99)).status != ABORT
    # 49 breaching iterations, one healthy, 49 more: no abort (streak resets).
    for it in range(250, 299):
        assert g.update(_healthy(it, std_max=0.31)).status == WARN
    assert g.update(_healthy(299)).status == OK
    for it in range(300, 349):
        assert g.update(_healthy(it, std_max=0.31)).status == WARN
    v = g.update(_healthy(349, std_max=0.31))
    assert v.status == ABORT and "std" in v.abort_reasons[0]


@pytest.mark.parametrize(
    ("field", "value", "n"),
    [
        ("action_p99", 3.6, 50),
        ("episode_length", 700.0, 50),
        ("learning_rate", 1e-5, 10),
        ("kl", 0.06, 10),
    ],
)
def test_guard_aborts_on_each_signal_when_hard(field: str, value: float, n: int) -> None:
    g = _guard(action_p99_hard=True, lr_hard=True, kl_hard=True)
    for i in range(n - 1):
        assert g.update(_healthy(1000 + i, **{field: value})).status != ABORT
    v = g.update(_healthy(1000 + n - 1, **{field: value}))
    assert v.status == ABORT, v
    rep = g.report(v)
    assert rep["abort_reasons"] and rep["thresholds"]["std_abort"] == 0.30


def test_vnext_hard_and_warn_only_signals() -> None:
    h = vnext_spec().health
    assert (h.action_p99_hard, h.lr_hard, h.kl_hard) == (False, False, False)
    assert (h.std_abort, h.action_p99_abort, h.episode_length_abort, h.lr_abort) == (
        0.30,
        3.5,
        800.0,
        1.0e-4,
    )
    assert h.sustain_iterations == 50
    assert default_spec().health.action_p99_hard  # dataclass default stays hard


@pytest.mark.parametrize(
    ("field", "value", "n", "hard"),
    [
        ("action_p99", 3.6, 50, False),
        ("learning_rate", 1e-5, 10, False),
        ("kl", 0.06, 10, False),
        ("episode_length", 700.0, 50, True),
        ("std_max", 0.31, 50, True),
    ],
)
def test_vnext_guard_policy(field: str, value: float, n: int, hard: bool) -> None:
    g = _guard()  # vnext_spec().health
    v = None
    for i in range(n + 20):
        v = g.update(_healthy(1000 + i, **{field: value}))
        if v.status == ABORT:
            break
    assert v is not None
    if hard:
        assert v.status == ABORT and v.iteration == 1000 + n - 1
    else:
        assert v.status == WARN
        assert any("consecutive iterations" in w for w in v.warnings)


def test_guard_nonfinite_aborts_immediately_even_in_warmup() -> None:
    g = _guard()
    v = g.update(_healthy(3, std_max=math.nan))
    assert v.status == ABORT and "non-finite" in v.abort_reasons[0]


def test_guard_curriculum_stall_only_warns() -> None:
    g = _guard()
    for it in range(0, 1500):
        g.update(_healthy(it, curriculum_level=0.0))
    v = g.update(_healthy(1500, curriculum_level=0.0))
    assert v.status == WARN and "level 0" in v.warnings[0]
    g2 = _guard()
    g2.update(_healthy(10, curriculum_level=2.0))
    assert g2.update(_healthy(1600, curriculum_level=2.0)).status == OK


def test_guard_missing_signals_are_not_failures() -> None:
    g = _guard()
    v = g.update(HealthSample(iteration=500))
    assert v.status == OK


# ---------------------------------------------------------------- cold start (vnext2)


def test_vnext2_is_vnext_plus_cold_start_only() -> None:
    from phoenix.velocity.spec import vnext2_spec

    v, v2 = vnext_spec(), vnext2_spec()
    assert v2.validate() == []
    assert not v.cold_start.enabled and v2.cold_start.enabled
    assert v2.replace(cold_start=v.cold_start) == v
    cs = v2.cold_start
    assert 0.10 <= cs.fraction <= 0.20
    assert cs.settle_s == (0.5, 1.5)


def test_cold_start_families_cover_the_requested_transitions() -> None:
    from phoenix.velocity.spec import vnext2_spec

    fams = vnext2_spec().cold_start.families
    covers = lambda axis, x: any(f[0] == axis and f[1] <= x <= f[2] for f in fams)  # noqa: E731
    assert covers(0, 0.3) and covers(0, -0.3)  # vx +-0.3
    assert covers(1, 0.3) and covers(1, -0.3)  # vy +-0.3
    assert covers(2, 0.8) and covers(2, -0.8) and covers(2, 0.5)  # low/moderate yaw
    back = sum(f[3] for f in fams if f[0] == 0 and f[2] < 0)
    assert back / sum(f[3] for f in fams) == pytest.approx(2 / 7)


def test_cold_start_validation_rejects_bad_families() -> None:
    from phoenix.velocity.spec import vnext2_spec

    v2 = vnext2_spec()
    cs = v2.cold_start
    bad = v2.replace(cold_start=dataclasses.replace(cs, families=((0, -0.5, -0.2, 1.0),)))
    assert any("outside level-0" in p for p in bad.validate())  # backward level 0 is -0.3
    bad = v2.replace(cold_start=dataclasses.replace(cs, families=((0, 0.0, 0.3, 1.0),)))
    assert any("stand-command band" in p for p in bad.validate())
    bad = v2.replace(cold_start=dataclasses.replace(cs, fraction=0.9))
    assert any("fraction" in p for p in bad.validate())
    bad = v2.replace(cold_start=dataclasses.replace(cs, settle_s=(0.5, 6.0)))
    assert any("settle_s" in p for p in bad.validate())


def test_cold_start_round_trips_and_old_specs_load_without_it() -> None:
    from phoenix.velocity.spec import vnext2_spec

    v2 = vnext2_spec()
    assert spec_from_dict(json.loads(v2.to_json())) == v2
    old = json.loads(vnext_spec().to_json())
    old.pop("cold_start")
    assert spec_from_dict(old) == vnext_spec()


# ------------------------------------------------ vnext3: linear tracking std 0.35 only


def test_vnext3_changes_only_linear_tracking_std() -> None:
    from phoenix.velocity.spec import vnext2_spec, vnext3_spec

    v2, v3 = vnext2_spec(), vnext3_spec()
    assert v3.validate() == []
    assert v3.reward("track_lin_vel_xy_exp").params["std"] == 0.35
    assert v3.reward("track_ang_vel_z_exp").params["std"] == 0.5
    assert v3.reward("action_l2").weight == -0.01
    assert v3.reward("feet_slide") == v2.reward("feet_slide")
    changed = [a.name for a, b in zip(v3.rewards, v2.rewards, strict=True) if a != b]
    assert changed == ["track_lin_vel_xy_exp"]
    # the curriculum gate keeps measuring the linear kernel at vnext2's 0.5
    assert v3.curriculum.gate_kernel_std == 0.5 and v2.curriculum.gate_kernel_std is None
    assert v3.replace(rewards=v2.rewards, curriculum=v2.curriculum) == v2
    assert v3.cold_start == v2.cold_start and v3.health == v2.health
    assert spec_from_dict(json.loads(v3.to_json())) == v3


def test_gate_kernel_std_requires_cold_start() -> None:
    from phoenix.velocity.spec import vnext3_spec

    v3 = vnext3_spec()
    bad = v3.replace(cold_start=dataclasses.replace(v3.cold_start, enabled=False))
    assert any("gate_kernel_std needs cold_start" in p for p in bad.validate())
