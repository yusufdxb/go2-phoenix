"""Every number of the PhoenixVelocity walking task, in one pure-python spec.

The Isaac Lab configclass builder (:mod:`phoenix.velocity.env_cfg`), the training
entry point (``scripts/train_velocity.py``), the checkpoint manifest and the CI
tests all read THIS module. Nothing here imports torch or Isaac Lab, so the spec
is unit-testable in CI and is recorded verbatim (``to_dict``) into every run.

Why a spec instead of the YAML factory
--------------------------------------
``phoenix.sim_env.go2_env_cfg`` layers YAML on top of an upstream task and has a
documented history of keys that were declared but never applied ("unwired key"
traps). Here there is no layering: a value either lives in a dataclass below and
is consumed by name in ``env_cfg.py``, or it does not exist. ``validate`` checks
the invariants that tie the spec to the 45-D contract.

Units are SI throughout: m, s, rad, N, kg. Command and push ranges are
``(lo, hi)`` tuples. Reward weights follow Isaac Lab's convention: the manager
multiplies ``term(env) * weight * step_dt`` every control step.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .contract import (
    ACTION_SCALE,
    ACTOR_OBS_DIM,
    ACTOR_OBS_TERMS,
    CONTROL_HZ,
    DECIMATION,
    DEFAULT_JOINT_POS,
    JOINT_ORDER,
    PHYSICS_HZ,
    CommandRanges,
    derive_locomotion_capable,
)

Range = tuple[float, float]

SPEC_SCHEMA = "phoenix-velocity-spec/v1"

#: Gym ids registered by :func:`phoenix.velocity.tasks.register`.
TASK_ID = "Phoenix-Velocity-Flat-Go2-v0"
PLAY_TASK_ID = "Phoenix-Velocity-Flat-Go2-Play-v0"
ENV_CFG_ENTRY = "phoenix.velocity.env_cfg:PhoenixVelocityFlatEnvCfg"
PLAY_ENV_CFG_ENTRY = "phoenix.velocity.env_cfg:PhoenixVelocityFlatEnvCfgPlay"
AGENT_CFG_ENTRY = "phoenix.velocity.env_cfg:PhoenixVelocityPPORunnerCfg"

#: Name of the velocity command term (Isaac ``CommandsCfg`` attribute).
COMMAND_NAME = "base_velocity"
#: Name of the joint-position action term (Isaac ``ActionsCfg`` attribute).
ACTION_NAME = "joint_pos"


# --------------------------------------------------------------------------- sim


@dataclass(frozen=True)
class SimSpec:
    """Timing and scene size. Control rate is DERIVED, never stated twice."""

    physics_dt: float = 1.0 / PHYSICS_HZ  # s (200 Hz)
    decimation: int = DECIMATION  # physics steps per policy step (4 -> 50 Hz)
    episode_length_s: float = 20.0  # s; 1000 policy steps
    num_envs: int = 4096  # fits a 12 GB GPU for a flat GO2 task
    env_spacing: float = 2.5  # m

    @property
    def step_dt(self) -> float:
        return self.physics_dt * self.decimation

    @property
    def control_hz(self) -> float:
        return 1.0 / self.step_dt

    @property
    def max_episode_steps(self) -> int:
        return int(math.ceil(self.episode_length_s / self.step_dt))


@dataclass(frozen=True)
class ActionSpec:
    """Joint-position action: ``target = default_q + scale * a`` (raw ``a`` unclipped).

    No action clip: the contract's ``last_action`` is the RAW policy output, so a
    training-side clip would silently change what the deploy side must feed back.
    Large actions are discouraged by rewards instead (action_rate, torques,
    joint_pos_limits, slew hinge) and measured by telemetry.
    """

    scale: float = ACTION_SCALE
    use_default_offset: bool = True
    clip: float | None = None
    #: Train against the deploy per-step slew clip (current_q +/- 0.175 rad) as part of
    #: the MDP. OFF by default: clipping the target to measured q +/- 0.175 rad caps
    #: the PD torque at 25 * 0.175 = 4.4 N m, below what trotting needs. The deploy
    #: side must relax that clip for velocity mode or this must be turned on (see
    #: docs/velocity/mdp.md, "Deploy slew clip").
    rate_limit_enabled: bool = False


# ------------------------------------------------------------------ observations


@dataclass(frozen=True)
class ObsNoiseSpec:
    """Uniform noise half-widths on the ACTOR terms (critic sees clean values).

    Audit (units match the contract terms):
      base_ang_vel      0.2 rad/s  gyro white noise + vibration + bias drift on a
                                   trotting trunk; conservative vs datasheet noise.
      projected_gravity 0.05       ~2.9 deg attitude error of the IMU fusion.
      joint_pos_rel     0.01 rad   encoder quantization plus gearbox backlash.
      joint_vel         1.5 rad/s  GO2 dq is a finite difference and is jittery;
                                   the policy must not rely on it being clean.
      velocity_command  0          commanded, exact on both sides.
      last_action       0          the policy's own output, exact on both sides.
    These are the upstream Isaac Lab values; they are kept because each is at or
    above the hardware error it stands for and training with them is known to work.
    """

    base_ang_vel: float = 0.2
    projected_gravity: float = 0.05
    velocity_command: float = 0.0
    joint_pos_rel: float = 0.01
    joint_vel: float = 1.5
    last_action: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name, *_ in ACTOR_OBS_TERMS}


#: Privileged critic group: clean actor terms + these. ``(name, dim, unit)``.
CRITIC_EXTRA_TERMS: tuple[tuple[str, int, str], ...] = (
    ("base_lin_vel", 3, "m/s body"),
    ("base_height", 1, "m world z (flat plane)"),
    ("feet_contact", 4, "0/1 per foot"),
    ("feet_contact_force", 4, "N per foot (norm)"),
)
CRITIC_OBS_DIM = ACTOR_OBS_DIM + sum(t[1] for t in CRITIC_EXTRA_TERMS)

#: GO2 body names (read from go2.usd): base, Head_upper, Head_lower, {FL,FR,RL,RR}_{hip,thigh,calf,foot}.
FOOT_BODIES = ".*_foot"
TRUNK_BODIES: tuple[str, ...] = ("base", "Head_.*")
UNDESIRED_CONTACT_BODIES: tuple[str, ...] = (".*_hip", ".*_thigh")
#: Contact is "on" above this force (N). Same threshold everywhere for consistency.
CONTACT_FORCE_THRESHOLD_N = 1.0


# ---------------------------------------------------------------------- commands


def _pair(r: Sequence[float]) -> Range:
    return float(r[0]), float(r[1])


@dataclass(frozen=True)
class CommandSpec:
    """Direct yaw-rate velocity commands (deploy sends yaw RATE, not heading)."""

    initial_lin_vel_x: Range = (-0.3, 0.6)
    initial_lin_vel_y: Range = (-0.2, 0.2)
    initial_ang_vel_z: Range = (-0.5, 0.5)
    final_lin_vel_x: Range = (-0.8, 1.0)
    final_lin_vel_y: Range = (-0.5, 0.5)
    final_ang_vel_z: Range = (-1.0, 1.0)
    #: Fraction of (re)samples forced to a zero command. Resampled with the command,
    #: so an env walking at 0.8 m/s can be told to stop mid-episode.
    rel_standing_envs: float = 0.10
    #: Seconds between resamples. With 20 s episodes this gives 2 to 4 command
    #: segments per episode: start, transition and stop-from-walking are all trained.
    resampling_time_range_s: Range = (5.0, 10.0)
    heading_command: bool = False
    #: Below both of these a command counts as "stand" for command-gated rewards.
    small_lin_cmd: float = 0.1  # m/s, norm of (vx, vy)
    small_yaw_cmd: float = 0.1  # rad/s, |wz|

    def initial_ranges(self) -> CommandRanges:
        return CommandRanges(
            _pair(self.initial_lin_vel_x),
            _pair(self.initial_lin_vel_y),
            _pair(self.initial_ang_vel_z),
            float(self.rel_standing_envs),
        )

    def final_ranges(self) -> CommandRanges:
        return CommandRanges(
            _pair(self.final_lin_vel_x),
            _pair(self.final_lin_vel_y),
            _pair(self.final_ang_vel_z),
            float(self.rel_standing_envs),
        )


@dataclass(frozen=True)
class CurriculumSpec:
    """Performance-gated command-range curriculum (rule in :mod:`.curriculum`).

    Ranges move from ``CommandSpec.initial_*`` to ``final_*`` in ``num_levels``
    equal linear steps. After every window of finished episodes the level goes up
    by one iff ALL of:
      * lin tracking score  >= ``lin_score_threshold``
      * yaw tracking score  >= ``yaw_score_threshold``
      * termination rate    <= ``max_termination_rate`` (non-timeout ends / all ends)
    A score is the measured mean tracking kernel normalized against a policy that
    never moves: ``(k - k_still) / (1 - k_still)``, so 0 = no better than standing
    still and 1 = perfect tracking. Normalizing matters: with std 0.5 a robot that
    stands still already earns ~0.71 of the lin kernel on the initial ranges.
    The level never decreases and never exceeds ``num_levels``.
    """

    enabled: bool = True
    num_levels: int = 8
    #: Window length, in finished episodes, as a multiple of num_envs (>= min).
    window_episodes_per_env: float = 1.0
    min_window_episodes: int = 64
    lin_score_threshold: float = 0.6
    yaw_score_threshold: float = 0.5
    max_termination_rate: float = 0.10

    def window_episodes(self, num_envs: int) -> int:
        return max(int(self.min_window_episodes), int(round(self.window_episodes_per_env * num_envs)))


# ----------------------------------------------------------------------- rewards


@dataclass(frozen=True)
class RewardTermSpec:
    """One reward term. ``func`` is resolved by name in ``env_cfg.py``."""

    name: str
    func: str
    weight: float
    unit: str
    prevents: str
    params: Mapping[str, Any] = field(default_factory=dict)


#: Isaac/Phoenix tracking-kernel std, m/s and rad/s. exp(-err^2 / std^2).
TRACKING_STD = 0.5


def default_reward_terms() -> tuple[RewardTermSpec, ...]:
    return (
        RewardTermSpec(
            "track_lin_vel_xy_exp",
            "isaac:track_lin_vel_xy_exp",
            1.5,
            "exp kernel in [0,1] of body-frame (vx,vy) error",
            "the task itself: not following the planar velocity command",
            {"std": TRACKING_STD},
        ),
        RewardTermSpec(
            "track_ang_vel_z_exp",
            "isaac:track_ang_vel_z_exp",
            0.75,
            "exp kernel in [0,1] of body-frame wz error",
            "the task itself: not following the yaw-rate command",
            {"std": TRACKING_STD},
        ),
        RewardTermSpec(
            "lin_vel_z_l2",
            "isaac:lin_vel_z_l2",
            -2.0,
            "(m/s)^2 body vz",
            "bouncing / hopping gaits that pump the trunk vertically",
        ),
        RewardTermSpec(
            "ang_vel_xy_l2",
            "isaac:ang_vel_xy_l2",
            -0.05,
            "(rad/s)^2 body roll+pitch rate",
            "trunk wobble; the IMU and any payload see it",
        ),
        RewardTermSpec(
            "flat_orientation_l2",
            "isaac:flat_orientation_l2",
            -2.5,
            "projected gravity xy squared (~tilt^2)",
            "walking with a steady lean (static tilt is not caught by ang_vel_xy)",
        ),
        RewardTermSpec(
            "joint_torques_l2",
            "isaac:joint_torques_l2",
            -2.0e-4,
            "(N m)^2 summed over 12 joints",
            "high-torque gaits that heat motors and saturate the DC-motor curve",
        ),
        RewardTermSpec(
            "joint_acc_l2",
            "isaac:joint_acc_l2",
            -2.5e-7,
            "(rad/s^2)^2 summed",
            "jerky, vibrating joint motion that the real gearbox will not follow",
        ),
        RewardTermSpec(
            "action_rate_l2",
            "isaac:action_rate_l2",
            -0.01,
            "(dimensionless action delta)^2 summed",
            "high-frequency action chatter across ALL motors (L2 smoothness)",
        ),
        RewardTermSpec(
            "slew_sat_hinge_l2",
            "phoenix:slew_sat_hinge_l2",
            -10.0,
            "rad^2: sum over motors of max(0, scale*|a_t - a_{t-1}| - 0.15)^2",
            "any SINGLE motor approaching the 0.175 rad/step deploy slew clip "
            "(action_rate_l2 can stay small while one motor saturates)",
            {"threshold": 0.15},
        ),
        RewardTermSpec(
            "feet_air_time",
            "phoenix:feet_air_time_gated",
            0.25,
            "s: sum over touchdowns of (air_time - 0.5 s), only when commanded to move",
            "shuffling / foot dragging; zero when the command is a stand command",
            {"threshold": 0.5},
        ),
        RewardTermSpec(
            "feet_slide",
            "isaac:feet_slide",
            -0.1,
            "m/s: sum of planar foot speed while that foot is in contact",
            "stance feet skating on the ground (slip that the real floor punishes)",
        ),
        RewardTermSpec(
            "undesired_contacts",
            "isaac:undesired_contacts",
            -1.0,
            "count of hip/thigh bodies with contact force > 1 N",
            "knee / thigh / hip touching the ground (kneeling, crawling gaits)",
        ),
        RewardTermSpec(
            "joint_pos_limits",
            "isaac:joint_pos_limits",
            -1.0,
            "rad beyond the soft joint limits (0.9 x hard range), summed",
            "gaits that ride the mechanical joint stops",
        ),
        RewardTermSpec(
            "stand_still",
            "phoenix:stand_still_joint_deviation_l1",
            -0.5,
            "rad: sum |q - q_default| when |v_cmd| < 0.1 m/s and |wz_cmd| < 0.1 rad/s",
            "fidgeting / marching in place or a crouched pose when told to stand",
        ),
    )


# ------------------------------------------------------------------ terminations


@dataclass(frozen=True)
class TerminationSpec:
    """Terminations. See docs/velocity/mdp.md for the justification of each value.

    * time_out: 20 s episodes (truncation, bootstrapped by PPO).
    * trunk_contact: base or head touches the ground with > 1 N: a fall.
    * bad_orientation: tilt (angle between body z and world up, from projected
      gravity) > 1.0 rad (57 deg). Aggressive turns and 0.5 m/s pushes stay well
      under ~0.6 rad; past ~1 rad a 0.3 m tall quadruped does not come back
      without a trunk contact, so waiting for that contact only adds
      uninformative flailing steps.
    * numerical_failure: any non-finite root/joint state, or |qd| > 100 rad/s, or
      |root v| > 50 m/s. Those are > 3x anything the GO2 can physically do
      (motor velocity limit 30 rad/s), so they only fire on a PhysX blow-up,
      before the NaN reaches the reward (rsl_rl raises on NaN rewards).
    Thigh/hip contacts are a PENALTY, not a termination: brushing a knee while
    recovering from a stumble is survivable and the policy should learn from the
    recovery rather than have the episode cut.
    """

    tilt_limit_rad: float = 1.0
    trunk_contact_threshold_n: float = CONTACT_FORCE_THRESHOLD_N
    max_joint_vel: float = 100.0
    max_root_lin_vel: float = 50.0


# ----------------------------------------------------------- domain randomization


@dataclass(frozen=True)
class DomainRandSpec:
    """Domain randomization. Mode (startup / reset / interval) noted per field."""

    # startup: per-shape materials, 64 buckets; ground is mu=1.0 with "multiply"
    # combine, so these ARE the effective foot/ground friction.
    static_friction: Range = (0.4, 1.2)
    dynamic_friction: Range = (0.4, 1.0)
    restitution: Range = (0.0, 0.1)
    material_buckets: int = 64
    # startup: added to the trunk ("base") mass, kg (GO2 trunk ~6.9 kg).
    base_mass_add_kg: Range = (-1.0, 3.0)
    # startup: trunk CoM shift, m.
    base_com_enabled: bool = True
    base_com_x: Range = (-0.03, 0.03)
    base_com_y: Range = (-0.03, 0.03)
    base_com_z: Range = (-0.02, 0.02)
    # startup: per-env, per-joint multiplier on the DC-motor PD gains (motor strength).
    motor_stiffness_scale: Range = (0.85, 1.15)
    motor_damping_scale: Range = (0.85, 1.15)
    # startup: joint armature added, kg m^2. ASSUMPTION: GO2 reflected rotor inertia
    # is O(0.005-0.01) kg m^2; the USD models none, so the range brackets [0, that].
    joint_armature_add: Range = (0.0, 0.01)
    # every reset: per-env actuator command delay in PHYSICS steps (5 ms each).
    actuator_latency_physics_steps: tuple[int, int] = (1, 4)
    # reset: root pose / velocity perturbation.
    reset_xy_m: Range = (-0.5, 0.5)
    reset_yaw_rad: Range = (-math.pi, math.pi)
    reset_lin_vel: Range = (-0.2, 0.2)
    reset_ang_vel: Range = (-0.2, 0.2)
    # reset: joint offsets from the default pose, rad (velocity zero).
    reset_joint_pos_offset: Range = (-0.1, 0.1)
    # interval: mid-episode shove by setting root planar velocity, m/s.
    push_enabled: bool = True
    push_interval_s: Range = (6.0, 12.0)
    push_velocity_xy: Range = (-0.5, 0.5)


# --------------------------------------------------------------------------- PPO


@dataclass(frozen=True)
class PPOSpec:
    num_steps_per_env: int = 24
    max_iterations: int = 2000
    save_interval: int = 100
    experiment_name: str = "phoenix_velocity_flat"
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    init_noise_std: float = 1.0
    #: Actor normalization OFF: the deployed network is then exactly the MLP on raw
    #: SI observations, with no normalizer stats to lose (rsl_rl 5.0.1 does save and
    #: export them, but OFF removes that failure class entirely). Critic ON: its
    #: privileged terms (contact forces in N) span 3 orders of magnitude and it is
    #: never deployed.
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = True
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    entropy_coef: float = 0.005
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    max_grad_norm: float = 1.0


# ---------------------------------------------------------------------- the spec


@dataclass(frozen=True)
class VelocityTaskSpec:
    sim: SimSpec = field(default_factory=SimSpec)
    action: ActionSpec = field(default_factory=ActionSpec)
    obs_noise: ObsNoiseSpec = field(default_factory=ObsNoiseSpec)
    commands: CommandSpec = field(default_factory=CommandSpec)
    curriculum: CurriculumSpec = field(default_factory=CurriculumSpec)
    rewards: tuple[RewardTermSpec, ...] = field(default_factory=default_reward_terms)
    terminations: TerminationSpec = field(default_factory=TerminationSpec)
    domain_randomization: DomainRandSpec = field(default_factory=DomainRandSpec)
    ppo: PPOSpec = field(default_factory=PPOSpec)
    seed: int = 42
    task_id: str = TASK_ID

    def reward_scales(self) -> dict[str, float]:
        return {t.name: float(t.weight) for t in self.rewards}

    def reward(self, name: str) -> RewardTermSpec:
        for t in self.rewards:
            if t.name == name:
                return t
        raise KeyError(name)

    def replace(self, **changes: Any) -> VelocityTaskSpec:
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        out = _jsonable(dataclasses.asdict(self))
        out["schema"] = SPEC_SCHEMA
        out["derived"] = {
            "control_hz": self.sim.control_hz,
            "step_dt": self.sim.step_dt,
            "max_episode_steps": self.sim.max_episode_steps,
            "actor_obs_dim": ACTOR_OBS_DIM,
            "critic_obs_dim": CRITIC_OBS_DIM,
            "critic_extra_terms": [list(t) for t in CRITIC_EXTRA_TERMS],
            "joint_order": list(JOINT_ORDER),
            "default_joint_pos": {j: DEFAULT_JOINT_POS[j] for j in JOINT_ORDER},
            "initial_command_ranges": self.commands.initial_ranges().to_dict(),
            "final_command_ranges": self.commands.final_ranges().to_dict(),
            "curriculum_window_episodes": self.curriculum.window_episodes(self.sim.num_envs),
        }
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def validate(self) -> list[str]:
        """Every reason this spec violates the 45-D contract or its own invariants."""
        p: list[str] = []
        if abs(self.action.scale - ACTION_SCALE) > 1e-12:
            p.append(f"action scale {self.action.scale} != contract {ACTION_SCALE}")
        if abs(self.sim.control_hz - CONTROL_HZ) > 1e-9:
            p.append(f"control rate {self.sim.control_hz} Hz != contract {CONTROL_HZ}")
        if abs(1.0 / self.sim.physics_dt - PHYSICS_HZ) > 1e-6:
            p.append(f"physics rate {1.0 / self.sim.physics_dt} Hz != contract {PHYSICS_HZ}")
        if self.action.clip is not None:
            p.append("action clip set: contract last_action is the RAW policy action")
        if set(self.obs_noise.as_dict()) != {t[0] for t in ACTOR_OBS_TERMS}:
            p.append("obs noise terms do not match the actor terms")
        if self.commands.heading_command:
            p.append("heading_command=True: deploy commands yaw RATE, train it directly")
        c = self.commands
        for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            lo0, hi0 = getattr(c, f"initial_{axis}")
            lo1, hi1 = getattr(c, f"final_{axis}")
            if not (lo0 <= hi0 and lo1 <= hi1):
                p.append(f"{axis}: range with lo > hi")
            if not (lo1 <= lo0 and hi0 <= hi1):
                p.append(f"{axis}: initial range {lo0, hi0} not inside final {lo1, hi1}")
        if not 0.0 <= c.rel_standing_envs <= 1.0:
            p.append("rel_standing_envs outside [0, 1]")
        lo, hi = c.resampling_time_range_s
        if not (0 < lo <= hi < self.sim.episode_length_s):
            p.append("resampling time must be positive and shorter than the episode")
        capable, reasons = derive_locomotion_capable(c.final_ranges())
        if not capable:
            p.append("final command ranges are not locomotion-capable: " + "; ".join(reasons))
        cur = self.curriculum
        if cur.num_levels < 1:
            p.append("curriculum needs >= 1 level")
        if not (0.0 < cur.lin_score_threshold <= 1.0 and 0.0 < cur.yaw_score_threshold <= 1.0):
            p.append("curriculum score thresholds must be in (0, 1]")
        if not 0.0 <= cur.max_termination_rate < 1.0:
            p.append("curriculum max_termination_rate must be in [0, 1)")
        names = [t.name for t in self.rewards]
        if len(names) != len(set(names)):
            p.append(f"duplicate reward names: {names}")
        funcs = [(t.func, tuple(sorted(t.params.items()))) for t in self.rewards]
        if len(funcs) != len(set(funcs)):
            p.append("two reward terms compute the same function with the same params")
        for t in self.rewards:
            positive = t.name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp", "feet_air_time")
            if positive and t.weight <= 0:
                p.append(f"{t.name} must have a positive weight")
            if not positive and t.weight >= 0:
                p.append(f"penalty {t.name} must have a negative weight")
        dr = self.domain_randomization
        lat_lo, lat_hi = dr.actuator_latency_physics_steps
        if not (0 <= lat_lo <= lat_hi):
            p.append("actuator latency range invalid")
        if lat_hi * self.sim.physics_dt > self.sim.step_dt + 1e-12:
            p.append("actuator latency longer than one control step")
        for name in ("static_friction", "dynamic_friction", "motor_stiffness_scale"):
            lo_, hi_ = getattr(dr, name)
            if not (0.0 < lo_ <= hi_):
                p.append(f"{name} must be a positive range")
        t = self.terminations
        if not (0.0 < t.tilt_limit_rad < math.pi):
            p.append("tilt limit must be in (0, pi): pi is unreachable (tautological)")
        return p


def default_spec() -> VelocityTaskSpec:
    return VelocityTaskSpec()


def smoke_spec() -> VelocityTaskSpec:
    """Tiny, fast variant for pipeline smoke tests (same MDP, fewer envs/iters)."""
    base = default_spec()
    return base.replace(
        sim=dataclasses.replace(base.sim, num_envs=64),
        ppo=dataclasses.replace(base.ppo, max_iterations=6, save_interval=2),
        curriculum=dataclasses.replace(base.curriculum, min_window_episodes=16),
    )


def spec_from_dict(data: Mapping[str, Any]) -> VelocityTaskSpec:
    """Inverse of :meth:`VelocityTaskSpec.to_dict` (ignores ``schema``/``derived``)."""

    def build(cls: type, blob: Mapping[str, Any]) -> Any:
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name not in blob:
                continue
            v = blob[f.name]
            kwargs[f.name] = tuple(v) if isinstance(v, list) else v
        return cls(**kwargs)

    return VelocityTaskSpec(
        sim=build(SimSpec, data["sim"]),
        action=build(ActionSpec, data["action"]),
        obs_noise=build(ObsNoiseSpec, data["obs_noise"]),
        commands=build(CommandSpec, data["commands"]),
        curriculum=build(CurriculumSpec, data["curriculum"]),
        rewards=tuple(RewardTermSpec(**r) for r in data["rewards"]),
        terminations=build(TerminationSpec, data["terminations"]),
        domain_randomization=build(DomainRandSpec, data["domain_randomization"]),
        ppo=build(PPOSpec, data["ppo"]),
        seed=int(data["seed"]),
        task_id=str(data["task_id"]),
    )


def joint_order_problems(
    joint_names: Sequence[str], default_joint_pos: Sequence[float] | None = None
) -> list[str]:
    """Why an articulation's joint order / default pose disagrees with the contract.

    Called at env build (``env_cfg.assert_joint_contract``) with the Isaac
    articulation's ``joint_names`` and ``default_joint_pos[0]``; the joint_pos_rel,
    joint_vel_rel and action terms all index joints in that articulation order.
    """
    problems: list[str] = []
    if list(joint_names) != list(JOINT_ORDER):
        problems.append(
            f"articulation joint order {list(joint_names)} != contract {list(JOINT_ORDER)}"
        )
        return problems
    if default_joint_pos is not None:
        if len(default_joint_pos) != len(JOINT_ORDER):
            problems.append(f"default_joint_pos has {len(default_joint_pos)} entries")
        else:
            for name, value in zip(JOINT_ORDER, default_joint_pos, strict=True):
                if abs(float(value) - DEFAULT_JOINT_POS[name]) > 1e-5:
                    problems.append(
                        f"default pose {name}={float(value):.5f} != contract {DEFAULT_JOINT_POS[name]}"
                    )
    return problems


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    return obj


__all__ = [
    "ACTION_NAME",
    "AGENT_CFG_ENTRY",
    "COMMAND_NAME",
    "CONTACT_FORCE_THRESHOLD_N",
    "CRITIC_EXTRA_TERMS",
    "CRITIC_OBS_DIM",
    "ENV_CFG_ENTRY",
    "FOOT_BODIES",
    "PLAY_ENV_CFG_ENTRY",
    "PLAY_TASK_ID",
    "SPEC_SCHEMA",
    "TASK_ID",
    "TRACKING_STD",
    "TRUNK_BODIES",
    "UNDESIRED_CONTACT_BODIES",
    "ActionSpec",
    "CommandSpec",
    "CurriculumSpec",
    "DomainRandSpec",
    "ObsNoiseSpec",
    "PPOSpec",
    "RewardTermSpec",
    "SimSpec",
    "TerminationSpec",
    "VelocityTaskSpec",
    "default_reward_terms",
    "default_spec",
    "joint_order_problems",
    "smoke_spec",
    "spec_from_dict",
]
