"""Build the PhoenixVelocity Isaac Lab env cfg from :mod:`phoenix.velocity.spec`.

Isaac-only module (imports torch / isaaclab at module load), so it is only ever
imported through the gym entry points registered by :func:`register` or from a
training / eval script that has already accepted running on the GPU workstation.
Never import this from CI-reachable code (see ``feedback_phoenix_lazy_torch``).

Design: start from the upstream ``UnitreeGo2FlatEnvCfg`` (proven scene, terrain,
sensors, sim settings) and override every MDP piece from the ONE spec object
(:func:`phoenix.velocity.spec.default_spec`), so nothing here restates a number
that already lives in ``spec.py``. Overridden, not layered YAML: this repo's
documented "unwired YAML key" history (``reference_phoenix_unwired_config_blocks``)
is exactly the failure mode a spec-driven configclass avoids.
"""

from __future__ import annotations

from typing import Any

from .contract import ACTOR_OBS_DIM, JOINT_ORDER
from .spec import (
    CRITIC_OBS_DIM,
    FOOT_BODIES,
    TRUNK_BODIES,
    UNDESIRED_CONTACT_BODIES,
    VelocityTaskSpec,
    default_spec,
)

# ---------------------------------------------------------------- actuators
#: Real GO2 EDU actuator envelope, split by joint group. The stock Isaac Lab
#: ``UNITREE_GO2_CFG`` (source at ~/Sim/IsaacLab/source/isaaclab_assets/isaaclab_assets
#: /robots/unitree.py, verified 2026-09-24) puts ALL joints in ONE DCMotorCfg at
#: effort_limit=23.5 N m / velocity_limit=30.0 rad/s. That is correct for hip and
#: thigh but wrong for calf: the real GO2 calf motor is 45.43 N m / 15.7 rad/s
#: (IsaacLab issue #7479, fixed upstream 2026-09-04 in a version newer than the
#: ~/Sim/IsaacLab checkout this repo shares with other projects). Overriding here,
#: in Phoenix's own env cfg, matches the fix without touching the shared IsaacLab
#: checkout other projects depend on.
HIP_THIGH_EFFORT_LIMIT_NM = 23.5
HIP_THIGH_VELOCITY_LIMIT_RAD_S = 30.0
CALF_EFFORT_LIMIT_NM = 45.43
CALF_VELOCITY_LIMIT_RAD_S = 15.7
ACTUATOR_STIFFNESS = 25.0
ACTUATOR_DAMPING = 0.5


def _build_actuators() -> dict[str, Any]:
    from isaaclab.actuators import DCMotorCfg

    return {
        "hip_thigh": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            effort_limit=HIP_THIGH_EFFORT_LIMIT_NM,
            saturation_effort=HIP_THIGH_EFFORT_LIMIT_NM,
            velocity_limit=HIP_THIGH_VELOCITY_LIMIT_RAD_S,
            stiffness=ACTUATOR_STIFFNESS,
            damping=ACTUATOR_DAMPING,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=CALF_EFFORT_LIMIT_NM,
            saturation_effort=CALF_EFFORT_LIMIT_NM,
            velocity_limit=CALF_VELOCITY_LIMIT_RAD_S,
            stiffness=ACTUATOR_STIFFNESS,
            damping=ACTUATOR_DAMPING,
            friction=0.0,
        ),
    }


def _apply_actuator_latency(robot_cfg: Any, lo: int, hi: int) -> None:
    """Swap both DCMotor actuator groups for DelayedDCMotor (see go2_env_cfg.py)."""
    from dataclasses import fields as dc_fields

    from isaaclab.actuators import DCMotorCfg

    from phoenix.sim_env.delayed_dc_motor import DelayedDCMotorCfg

    for name, act in list(robot_cfg.actuators.items()):
        if not isinstance(act, DCMotorCfg) or isinstance(act, DelayedDCMotorCfg):
            continue
        kwargs = {f.name: getattr(act, f.name) for f in dc_fields(act) if f.name != "class_type"}
        kwargs["min_delay"] = lo
        kwargs["max_delay"] = hi
        robot_cfg.actuators[name] = DelayedDCMotorCfg(**kwargs)


# ------------------------------------------------------------ reward mapping


def _resolve_reward_func(func_name: str):
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as itmdp

    from . import mdp as pmdp

    prefix, _, name = func_name.partition(":")
    if prefix == "isaac":
        return getattr(itmdp, name)
    if prefix == "phoenix":
        return getattr(pmdp, name)
    raise ValueError(f"unknown reward func namespace in {func_name!r}")


def _reward_params(spec: VelocityTaskSpec, term) -> dict[str, Any]:
    from isaaclab.managers import SceneEntityCfg

    p = dict(term.params)
    name = term.name
    if name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"):
        p["command_name"] = "base_velocity"
    elif name == "feet_slide":
        p["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=[FOOT_BODIES])
        p["asset_cfg"] = SceneEntityCfg("robot", body_names=[FOOT_BODIES])
    elif name == "undesired_contacts":
        p["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=list(UNDESIRED_CONTACT_BODIES))
        p["threshold"] = term.params.get("threshold", 1.0)
    elif name == "joint_pos_limits":
        p["asset_cfg"] = SceneEntityCfg("robot")
    elif name == "slew_sat_hinge_l2":
        p["action_name"] = "joint_pos"
    elif name == "feet_air_time":
        p["command_name"] = "base_velocity"
        p["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=[FOOT_BODIES])
        p["lin_threshold"] = spec.commands.small_lin_cmd
        p["yaw_threshold"] = spec.commands.small_yaw_cmd
    elif name == "stand_still":
        p["command_name"] = "base_velocity"
        p["lin_threshold"] = spec.commands.small_lin_cmd
        p["yaw_threshold"] = spec.commands.small_yaw_cmd
        p["asset_cfg"] = SceneEntityCfg("robot")
    return p


def _build_rewards_cfg(spec: VelocityTaskSpec):
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.utils import configclass

    attrs: dict[str, Any] = {}
    for term in spec.rewards:
        func = _resolve_reward_func(term.func)
        attrs[term.name] = RewTerm(func=func, weight=term.weight, params=_reward_params(spec, term))
    cls = configclass(type("RewardsCfg", (), attrs))
    return cls()


# ------------------------------------------------------------ observations


def _obs_noise_term(name: str, spec: VelocityTaskSpec):
    from isaaclab.utils.noise import UniformNoiseCfg as Unoise

    half = getattr(spec.obs_noise, name)
    return None if half == 0.0 else Unoise(n_min=-half, n_max=half)


def _build_observations_cfg(spec: VelocityTaskSpec):
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as itmdp
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils import configclass

    from . import mdp as pmdp

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=itmdp.base_ang_vel, noise=_obs_noise_term("base_ang_vel", spec))
        projected_gravity = ObsTerm(
            func=itmdp.projected_gravity, noise=_obs_noise_term("projected_gravity", spec)
        )
        velocity_command = ObsTerm(
            func=itmdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=itmdp.joint_pos_rel, noise=_obs_noise_term("joint_pos_rel", spec))
        joint_vel = ObsTerm(func=itmdp.joint_vel_rel, noise=_obs_noise_term("joint_vel", spec))
        last_action = ObsTerm(func=itmdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=itmdp.base_ang_vel)
        projected_gravity = ObsTerm(func=itmdp.projected_gravity)
        velocity_command = ObsTerm(func=itmdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=itmdp.joint_pos_rel)
        joint_vel = ObsTerm(func=itmdp.joint_vel_rel)
        last_action = ObsTerm(func=itmdp.last_action)
        base_lin_vel = ObsTerm(func=itmdp.base_lin_vel)
        base_height = ObsTerm(func=pmdp.base_height, params={"asset_cfg": SceneEntityCfg("robot")})
        feet_contact = ObsTerm(
            func=pmdp.feet_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[FOOT_BODIES]),
                "threshold": 1.0,
            },
        )
        feet_contact_force = ObsTerm(
            func=pmdp.feet_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[FOOT_BODIES])},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ObservationsCfg:
        policy: PolicyCfg = PolicyCfg()
        critic: CriticCfg = CriticCfg()

    assert len(spec.obs_noise.as_dict()) == 6  # documents the 6 actor terms this mirrors
    return ObservationsCfg()


# ---------------------------------------------------------------- commands


def _build_commands_cfg(spec: VelocityTaskSpec):
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as itmdp
    from isaaclab.utils import configclass

    c = spec.commands

    @configclass
    class CommandsCfg:
        base_velocity = itmdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=c.resampling_time_range_s,
            rel_standing_envs=c.rel_standing_envs,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            ranges=itmdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=c.initial_lin_vel_x,
                lin_vel_y=c.initial_lin_vel_y,
                ang_vel_z=c.initial_ang_vel_z,
            ),
        )

    return CommandsCfg()


# ----------------------------------------------------------------- events


def _build_events_cfg(spec: VelocityTaskSpec):
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as itmdp
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils import configclass

    from . import isaac_terms as pterms

    dr = spec.domain_randomization

    @configclass
    class EventsCfg:
        assert_contract = EventTerm(
            func=pterms.assert_joint_contract,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "action_name": "joint_pos",
                "critic_group": "critic",
            },
        )
        physics_material = EventTerm(
            func=itmdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": dr.static_friction,
                "dynamic_friction_range": dr.dynamic_friction,
                "restitution_range": dr.restitution,
                "num_buckets": dr.material_buckets,
            },
        )
        add_base_mass = EventTerm(
            func=itmdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": dr.base_mass_add_kg,
                "operation": "add",
            },
        )
        base_com = EventTerm(
            func=itmdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {"x": dr.base_com_x, "y": dr.base_com_y, "z": dr.base_com_z},
            },
        )
        joint_armature = EventTerm(
            func=itmdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "armature_distribution_params": dr.joint_armature_add,
                "operation": "add",
            },
        )
        motor_strength = EventTerm(
            func=None,  # bound below to phoenix.sim_env's explicit-actuator-safe scaler
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "stiffness_distribution_params": dr.motor_stiffness_scale,
                "damping_distribution_params": dr.motor_damping_scale,
            },
        )
        reset_base = EventTerm(
            func=itmdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": dr.reset_xy_m, "y": dr.reset_xy_m, "yaw": dr.reset_yaw_rad},
                "velocity_range": {
                    "x": dr.reset_lin_vel,
                    "y": dr.reset_lin_vel,
                    "z": dr.reset_lin_vel,
                    "roll": dr.reset_ang_vel,
                    "pitch": dr.reset_ang_vel,
                    "yaw": dr.reset_ang_vel,
                },
            },
        )
        reset_robot_joints = EventTerm(
            func=itmdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "position_range": dr.reset_joint_pos_offset,
                "velocity_range": (0.0, 0.0),
            },
        )
        push_robot = (
            EventTerm(
                func=itmdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=dr.push_interval_s,
                params={"velocity_range": {"x": dr.push_velocity_xy, "y": dr.push_velocity_xy}},
            )
            if dr.push_enabled
            else None
        )

    from phoenix.sim_env.go2_env_cfg import scale_explicit_actuator_gains

    cfg = EventsCfg()
    cfg.motor_strength.func = scale_explicit_actuator_gains
    return cfg


# ------------------------------------------------------------ terminations


def _build_terminations_cfg(spec: VelocityTaskSpec):
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as itmdp
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.utils import configclass

    from . import mdp as pmdp

    t = spec.terminations

    @configclass
    class TerminationsCfg:
        time_out = DoneTerm(func=itmdp.time_out, time_out=True)
        trunk_contact = DoneTerm(
            func=pmdp.trunk_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(TRUNK_BODIES)),
                "threshold": t.trunk_contact_threshold_n,
            },
        )
        bad_orientation = DoneTerm(
            func=pmdp.bad_tilt,
            params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": t.tilt_limit_rad},
        )
        numerical_failure = DoneTerm(
            func=pmdp.numerical_failure,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_joint_vel": t.max_joint_vel,
                "max_root_lin_vel": t.max_root_lin_vel,
            },
        )

    return TerminationsCfg()


# ------------------------------------------------------------- curriculum


def _build_curriculum_cfg(spec: VelocityTaskSpec):
    from isaaclab.managers import CurriculumTermCfg as CurrTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils import configclass

    from . import isaac_terms as pterms

    @configclass
    class CurriculumCfg:
        command_curriculum = CurrTerm(
            func=pterms.VelocityCommandCurriculum,
            params={"command_name": "base_velocity", "spec": spec.to_dict()},
        )
        telemetry = CurrTerm(
            func=pterms.VelocityTelemetry,
            params={
                "action_name": "joint_pos",
                "asset_cfg": SceneEntityCfg("robot"),
                "slew_clip_rad": 0.175,
            },
        )

    return CurriculumCfg()


# --------------------------------------------------------------- env cfg


def build_velocity_env_cfg(spec: VelocityTaskSpec | None = None):
    """Build a ``ManagerBasedRLEnvCfg`` for the PhoenixVelocity walking task."""
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
        UnitreeGo2FlatEnvCfg,
    )

    spec = spec or default_spec()
    problems = spec.validate()
    if problems:
        raise ValueError("PhoenixVelocity spec failed validation:\n  " + "\n  ".join(problems))

    cfg = UnitreeGo2FlatEnvCfg()

    # scene / timing
    cfg.scene.num_envs = spec.sim.num_envs
    cfg.scene.env_spacing = spec.sim.env_spacing
    cfg.episode_length_s = spec.sim.episode_length_s
    cfg.decimation = spec.sim.decimation
    cfg.sim.dt = spec.sim.physics_dt
    cfg.sim.render_interval = spec.sim.decimation
    cfg.seed = spec.seed

    # actuators: hip/thigh vs calf split at real GO2 limits
    cfg.scene.robot.actuators = _build_actuators()
    lat_lo, lat_hi = spec.domain_randomization.actuator_latency_physics_steps
    if lat_hi > 0:
        _apply_actuator_latency(cfg.scene.robot, lat_lo, lat_hi)

    # actions
    cfg.actions.joint_pos.asset_name = "robot"
    cfg.actions.joint_pos.joint_names = list(JOINT_ORDER)
    cfg.actions.joint_pos.scale = spec.action.scale
    cfg.actions.joint_pos.use_default_offset = spec.action.use_default_offset

    cfg.observations = _build_observations_cfg(spec)
    cfg.commands = _build_commands_cfg(spec)
    cfg.events = _build_events_cfg(spec)
    cfg.rewards = _build_rewards_cfg(spec)
    cfg.terminations = _build_terminations_cfg(spec)
    cfg.curriculum = _build_curriculum_cfg(spec)

    return cfg


class PhoenixVelocityFlatEnvCfg:  # populated by build_velocity_env_cfg() at import time via __new__
    """Gym entry point target. ``gym.make`` calls this like a class; delegate to the builder."""

    def __new__(cls, *args: Any, **kwargs: Any):  # noqa: D102
        return build_velocity_env_cfg()


class PhoenixVelocityFlatEnvCfgPlay:
    """Small-scene, DR-off variant for interactive play / sim2sim checks."""

    def __new__(cls, *args: Any, **kwargs: Any):  # noqa: D102
        import dataclasses

        spec = default_spec().replace(
            sim=dataclasses.replace(default_spec().sim, num_envs=50),
        )
        cfg = build_velocity_env_cfg(spec)
        cfg.observations.policy.enable_corruption = False
        cfg.events.push_robot = None
        return cfg


def register() -> None:
    """Register the PhoenixVelocity gym tasks. Idempotent."""
    import gymnasium as gym

    from .spec import (
        AGENT_CFG_ENTRY,
        ENV_CFG_ENTRY,
        PLAY_ENV_CFG_ENTRY,
        PLAY_TASK_ID,
        TASK_ID,
    )

    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": ENV_CFG_ENTRY,
                "rsl_rl_cfg_entry_point": AGENT_CFG_ENTRY,
            },
        )
    if PLAY_TASK_ID not in gym.registry:
        gym.register(
            id=PLAY_TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": PLAY_ENV_CFG_ENTRY,
                "rsl_rl_cfg_entry_point": AGENT_CFG_ENTRY,
            },
        )


def build_ppo_runner_cfg(spec: VelocityTaskSpec | None = None):
    """Build the rsl_rl ``RslRlOnPolicyRunnerCfg`` from :class:`.spec.PPOSpec`."""
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
    )

    spec = spec or default_spec()
    p = spec.ppo
    return RslRlOnPolicyRunnerCfg(
        seed=spec.seed,
        num_steps_per_env=p.num_steps_per_env,
        max_iterations=p.max_iterations,
        save_interval=p.save_interval,
        experiment_name=p.experiment_name,
        obs_groups={"actor": ["policy"], "critic": ["critic"]},
        # deprecated runner-level flag; explicit per-network flags below are what
        # actually apply (see phoenix.training.agent_cfg for why this must not be MISSING).
        empirical_normalization=False,
        # rsl_rl clip_actions: a wide dimensionless sanity clip (legged_gym
        # convention: rl_sar/robot_lab/himloco use +-100), not a radians limit.
        # Safety against illegal joint targets comes from actuator torque/velocity
        # limits and the deploy slew clip, not this value. See ActionSpec.clip_actions
        # for the 2026-09-25 bug this replaces (1.0 here with scale=0.25 pinned every
        # joint at 0.25 rad and stalled the curriculum). The action stored for the
        # "last_action" observation term is the CLIPPED action (the wrapper clips
        # before env.step), so train and deploy must clip identically at this value.
        clip_actions=spec.action.clip_actions,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=p.init_noise_std,
            actor_obs_normalization=p.actor_obs_normalization,
            critic_obs_normalization=p.critic_obs_normalization,
            actor_hidden_dims=list(p.actor_hidden_dims),
            critic_hidden_dims=list(p.critic_hidden_dims),
            activation=p.activation,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=p.value_loss_coef,
            use_clipped_value_loss=p.use_clipped_value_loss,
            clip_param=p.clip_param,
            entropy_coef=p.entropy_coef,
            num_learning_epochs=p.num_learning_epochs,
            num_mini_batches=p.num_mini_batches,
            learning_rate=p.learning_rate,
            schedule=p.schedule,
            gamma=p.gamma,
            lam=p.lam,
            desired_kl=p.desired_kl,
            max_grad_norm=p.max_grad_norm,
        ),
    )


PhoenixVelocityPPORunnerCfg = build_ppo_runner_cfg


__all__ = [
    "ACTOR_OBS_DIM",
    "CALF_EFFORT_LIMIT_NM",
    "CALF_VELOCITY_LIMIT_RAD_S",
    "CRITIC_OBS_DIM",
    "HIP_THIGH_EFFORT_LIMIT_NM",
    "HIP_THIGH_VELOCITY_LIMIT_RAD_S",
    "PhoenixVelocityFlatEnvCfg",
    "PhoenixVelocityFlatEnvCfgPlay",
    "PhoenixVelocityPPORunnerCfg",
    "build_env_cfg_realized_dump",
    "build_ppo_runner_cfg",
    "build_velocity_env_cfg",
    "register",
]


def build_env_cfg_realized_dump(spec: VelocityTaskSpec | None = None) -> dict[str, Any]:
    """Static (pre-construction) dump of what the cfg WILL apply. See scripts/dump_realized_dr.py
    for the post-construction dump (reads back the actually-instantiated env's live event terms,
    which is the one that matters for "is this DR term actually live").
    """
    spec = spec or default_spec()
    cfg = build_velocity_env_cfg(spec)
    events = {}
    for name in dir(cfg.events):
        if name.startswith("_"):
            continue
        term = getattr(cfg.events, name)
        if term is None:
            continue
        events[name] = {
            "mode": getattr(term, "mode", None),
            "func": getattr(getattr(term, "func", None), "__name__", str(getattr(term, "func", None))),
            "params": {k: (v if not hasattr(v, "__dict__") else str(v)) for k, v in getattr(term, "params", {}).items()},
        }
    actuators = {
        name: {
            "joint_names_expr": act.joint_names_expr,
            "effort_limit": act.effort_limit,
            "velocity_limit": act.velocity_limit,
            "stiffness": act.stiffness,
            "damping": act.damping,
            "class": type(act).__name__,
            "min_delay": getattr(act, "min_delay", None),
            "max_delay": getattr(act, "max_delay", None),
        }
        for name, act in cfg.scene.robot.actuators.items()
    }
    return {"events": events, "actuators": actuators, "spec": spec.to_dict()}
