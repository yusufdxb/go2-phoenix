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
    OBS_SCHEMA_V1,
    OBS_SCHEMA_V2,
    OBS_SCHEMA_VERSIONS,
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
    #: rsl_rl ``RslRlOnPolicyRunnerCfg.clip_actions``: the raw policy output is
    #: clamped to [-clip_actions, clip_actions] BEFORE ``scale`` is applied and
    #: before it is stored as the "last_action" observation. This is a dimensionless
    #: sanity clip on the network output, not a radians limit; safety against illegal
    #: joint targets comes from the actuator torque/velocity limits and the deploy
    #: slew clip, not from this value (legged_gym convention: rl_sar/robot_lab/
    #: himloco configs use +-100, effectively unclipped for any action magnitude a
    #: converged policy would output).
    #:
    #: 2026-09-25 bug (fixed here): this was hardcoded to 1.0 in env_cfg.py. With
    #: scale=0.25 that caps every joint offset at 0.25 rad, too small for a trot.
    #: Seed 42 of the walk_v1 run pinned at the clip almost every step
    #: (Curriculum/telemetry/action_abs_mean 0.9985, p99 1.0 at iteration 2965),
    #: the command curriculum never left level 0, and lin_score went negative
    #: (worse than standing still). Kept running as the clip=1.0 control arm;
    #: 100.0 is the new default for every arm launched after this fix.
    clip_actions: float = 100.0


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
#: "base" only, matching the upstream Go2 termination body. 2026-09-25 bug (fixed
#: here): TRUNK_BODIES used to also include "Head_.*". Head_upper/Head_lower are
#: near-massless (0.001 kg, isaac_reference.py) bodies rigidly fixed to the base;
#: under any joint acceleration the internal constraint reaction force needed to
#: keep a near-massless body attached to an accelerating trunk spikes through
#: PhysX's net-contact-force channel (measured: mean 3.7 N, MAX 235 N with only
#: action_std=0.5 noise and zero real ground contact), far above the 1 N
#: threshold. This produced a near-100% false-positive "trunk_contact"
#: termination on any nonzero action (mean episode length ~7 steps at 4096 envs,
#: vs stock Go2's ~973-993 steps with the same reward/PPO setup, "base" only).
#: scripts/diag_zero_action_rollout.py reproduces this in isolation.
TRUNK_BODIES: tuple[str, ...] = ("base",)
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
    #: ``(vx_forward, |vy|, |wz|)`` the trained envelope MUST cover whatever the
    #: curriculum does, or None. Checked against the INITIAL ranges by
    #: ``validate``: curriculum levels are monotone, so what level 0 covers is
    #: covered by every checkpoint of the run. A performance-gated curriculum
    #: cannot guarantee reaching a level; construction can.
    required_envelope: tuple[float, float, float] | None = None
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

    Two gates (``gate``):

    ``normalized_lin_yaw`` (v1, walk_v1/v2 runs; default so they resume unchanged)
        described below. 2026-09-25: it never left level 0 on seeds 42-46; the
        yaw score blocked and, on seed 46, the lin score also peaked at 0.572
        (iteration 750) below its 0.6 threshold.
    ``unitree_lin_kernel`` (vNext)
        Unitree's GO2 rule (unitree_rl_lab ``mdp.lin_vel_cmd_levels``): expand
        when ``mean(episode_sum_track_lin) / max_episode_length_s`` exceeds
        ``lin_kernel_threshold * weight``, i.e. the tracking kernel averaged over
        the FULL episode duration (an early fall counts as zero tracking for the
        rest of it) is > 0.8. Only lin x/y move; yaw must be at its final range
        from level 0 (``validate`` enforces initial == final ang_vel_z). The yaw
        score is still computed and logged, but never blocks. Phoenix keeps its
        termination-rate gate on top (Unitree has none; it never bound on seed
        46, whose rate was 0.02-0.04 over iterations 250-2500). Seed 46 logged
        ``Episode_Reward/track_lin_vel_xy_exp`` (the same statistic times the
        weight 1.5) at 1.30-1.34 over iterations 500-2000, above the 1.2 bar.

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
    gate: str = "normalized_lin_yaw"
    lin_kernel_threshold: float = 0.8
    #: None: the gate's lin kernel comes from the track_lin_vel_xy_exp reward's
    #: episode sums (so it moves with that reward's std). A value: the kernel is
    #: accumulated separately from the velocity error at THIS std (command-term
    #: accumulator, needs cold_start enabled), so a reward std change does not
    #: also change how the curriculum expands. vnext3 pins 0.5 = vnext2's gate.
    gate_kernel_std: float | None = None
    num_levels: int = 8
    #: Window length, in finished episodes, as a multiple of num_envs (>= min).
    window_episodes_per_env: float = 1.0
    min_window_episodes: int = 64
    lin_score_threshold: float = 0.6
    yaw_score_threshold: float = 0.5
    max_termination_rate: float = 0.10

    def window_episodes(self, num_envs: int) -> int:
        return max(
            int(self.min_window_episodes), int(round(self.window_episodes_per_env * num_envs))
        )


CURRICULUM_GATES = ("normalized_lin_yaw", "unitree_lin_kernel")


# ------------------------------------------------------------------ health guard


@dataclass(frozen=True)
class HealthGuardSpec:
    """Fail-fast training-health guard (:mod:`phoenix.velocity.health`).

    Every iteration logs Health/* scalars; a signal past its ABORT threshold for
    ``sustain_iterations`` consecutive iterations (after ``warmup_iterations``)
    stops the run with a checkpoint and ``health_abort.json``. Non-finite values
    abort at once. WARN thresholds only log.

    Thresholds come from walk_v2 seed 46 (8192 envs, 9000 iterations,
    2026-09-25; phoenix-seed46-autopsy/training_all_iterations.csv and
    checkpoint_std_by_joint.csv). Healthy = iterations 250-1500 (model_1000 and
    model_1500 passed Isaac eval and every MuJoCo gate_v3 physical check);
    degradation onset ~2000 (calf upper-limit hugging 0.157 of envs at +0.6
    yaw, 0.574 at 2500); collapse 6000+ (episode length 843 -> 7).

    * std: per-joint policy std max. Pre-degradation max 0.2923 (model_250;
      0.2750 over 500-1750). Abort > 0.30, "slightly above", which is also the
      upper bound rsl_rl ``std_range`` would get; rsl_rl-lib 5.0.1 (installed)
      has no ``std_range``, so this guard is the only std bound. Seed 46 would
      trip it ~3750 (checkpoint max 0.309). Std is a LATE signal there: it was
      still 0.27 when calf hugging began.
    * action |a| p99: healthy max 2.94, pre-degradation max 3.19; 3.5 sustained
      50 iterations first held at 2166 on seed 46 (the degradation onset).
      Seed-46-derived under the v1 observation; vNext changes the observation
      scaling and the command ranges, so a false abort here is possible and is
      the first thing to check if it trips early.
    * episode length (steps, rsl_rl lenbuffer mean): healthy min 904; abort < 800
      (seed 46: 6234). Backstop only.
    * learning rate: healthy min 2.96e-4 (adaptive, desired_kl 0.01); abort when
      <= 1e-4 for 10 iterations (seed 46: 3190). Repeated KL > 2 x desired is
      what drives it there.
    * KL (mean over the update's minibatches): not logged on seed 46, so no
      calibration. ASSUMPTION: abort at > 5 x desired_kl for 10 iterations is a
      blow-up, not ordinary adaptive-LR hunting (which targets 0.5-2 x).
    * hard vs warn-only: ``*_hard = False`` turns that signal's sustained
      breach into a WARNING (logged, run continues). Episode length, std and
      non-finite values are always hard. vNext (first run, seed 47,
      2026-09-25, Yusuf's call) runs action p99, learning rate and KL
      warn-only: they are the seed-46/v1-obs-derived or uncalibrated signals,
      so a first vNext run must not be killed by them; their thresholds are
      unchanged.
    * curriculum: warn only, if the level is still 0 at
      ``curriculum_stall_warn_iteration``; ``required_envelope`` already holds at
      level 0, so a stall costs range, not usability.
    """

    enabled: bool = False
    warmup_iterations: int = 250
    sustain_iterations: int = 50
    std_warn: float = 0.28
    std_abort: float = 0.30
    action_p99_warn: float = 3.2
    action_p99_abort: float = 3.5
    episode_length_warn: float = 900.0
    episode_length_abort: float = 800.0
    lr_abort: float = 1.0e-4
    lr_sustain_iterations: int = 10
    kl_abort_factor: float = 5.0
    kl_sustain_iterations: int = 10
    action_p99_hard: bool = True
    lr_hard: bool = True
    kl_hard: bool = True
    curriculum_stall_warn_iteration: int = 1500


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
            "action_l2",
            "isaac:action_l2",
            -0.01,
            "(dimensionless RAW action, post rsl_rl clip)^2 summed",
            "unbounded drift of the actor's mean output. 2026-09-25 bug: the "
            "Gaussian actor has no output activation on its mean (rsl_rl "
            "GaussianDistribution.update: mean = mlp_output, no tanh/clip) and "
            "action_rate_l2/slew_sat_hinge_l2 only penalize CHANGE between "
            "steps, not magnitude, so a policy can drift its mean action to "
            "arbitrarily large values as long as it does so slowly (small "
            "per-step delta). Measured: seed 43's action_abs_p99 grew smoothly "
            "4.09 (iter 2000) -> 9.70 (iter 5000) -> 15.49 (iter 8500) while "
            "action std stayed roughly flat (0.19 -> 0.20 -> 0.25), and episode "
            "length collapsed from 946 to 60 over the same span. Same weight "
            "as action_rate_l2 (-0.01): both are action-processing penalties "
            "at the same order of magnitude, not tuned against each other.",
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
        # 2026-09-25: default kept at -1.0 here (do not change default_spec(),
        # a resumed run must keep its own original weight). The sim2sim agent
        # found seed42@3000 drives FR/RL calves to their upper hard limit
        # during counter-clockwise yaw (+0.6, worse at +0.8), targets 0.25-0.31
        # rad past the stop, 20-30 steps past Isaac's own soft limit. The soft
        # limit factor is already correct (UNITREE_GO2_CFG,
        # ~/Sim/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/
        # unitree.py:168, soft_joint_pos_limit_factor=0.9, inherited since
        # env_cfg.py only overrides scene.robot.actuators, not the whole robot
        # asset), matching the legged_gym / unitree_rl_gym convention; the gap
        # is the reward WEIGHT, -1.0 here versus that convention's -10.0. Used
        # via VelocityTaskSpec.with_reward_weight for the isolated seed 46 arm
        # (clip100-curriculum-limitpen); see train_velocity.py
        # --joint-pos-limits-weight.
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
    # startup: per-env, per-joint Coulomb joint friction, N m (PhysX >= 5 joint
    # friction: static = dynamic = the sample, viscous 0). Off by default so every
    # earlier spec.json rebuilds unchanged. Measured on the hung GO2 2026-10-07
    # (docs/hardware/2026-10-07_friction_probe): hips 0.24-0.35, thighs 0.17-0.34,
    # calves 0.42-0.96 N m; the MuJoCo replay matches the robot at 1.25-1.5x.
    joint_friction_enabled: bool = False
    joint_friction_hip_thigh_nm: Range = (0.0, 0.0)
    joint_friction_calf_nm: Range = (0.0, 0.0)
    # startup + every control step: joint friction MODEL (phoenix.physics.joint_friction
    # FrictionDistribution): Coulomb (static = dynamic) + viscous + load-dependent Coulomb
    # k_load * |applied torque|, refreshed every control step. "off" keeps the legacy
    # joint_friction_* fields above in charge; "hardware" = measured per joint x a shared
    # scale with per-joint jitter (identity kept); "broad" = every joint U(broad), no identity.
    joint_friction_model: str = "off"
    joint_friction_scale: Range = (1.0, 1.0)
    joint_friction_jitter: float = 0.0
    joint_friction_broad_nm: Range = (0.0, 0.0)
    joint_viscous_hip: Range = (0.0, 0.0)
    joint_viscous_thigh: Range = (0.0, 0.0)
    joint_viscous_calf: Range = (0.0, 0.0)
    joint_friction_load: Range = (0.0, 0.0)
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


# --------------------------------------------------------------------- cold start

#: ``(axis, lo, hi, weight)``: axis 0 = vx m/s, 1 = vy m/s, 2 = wz rad/s; the
#: command is ``lo..hi`` uniform on that axis and zero on the others; family drawn
#: with probability proportional to ``weight``.
ColdStartFamily = tuple[int, float, float, float]


@dataclass(frozen=True)
class ColdStartSpec:
    """Cold-start reset mixture: the deployment start state, trained explicitly.

    2026-09-25 finding (walk_vnext_seed47): every checkpoint 750-1500 stood
    perfectly still in MuJoCo gate v3 at forward 0.3 from the settled default
    stance (vx exactly 0.0), while walking at 0.5 and still tracking 0.3 when it
    came DOWN from 0.5 (vx 0.38). Isaac never showed it because every Isaac reset
    perturbs joint positions (+-0.1 rad) and root velocity (+-0.2), so training
    never started from a dead-still stance, which is exactly how the robot is
    handed to the policy (FixStand, then the policy). The same low-command
    deadband shows in Isaac for backward: model_1000 from reset, vx -0.2 -> 0.000,
    -0.3 -> -0.153, -0.5 -> -0.412, -0.8 -> -0.674; entering -0.3 from -0.6
    -> -0.213.

    On a fraction of resets (event ``cold_start`` after the normal reset terms):
    joint position = default pose exactly, joint velocity 0, root linear and
    angular velocity 0 (no joint perturbation). That episode then holds a zero
    command for ``settle_s`` and switches to ONE low command drawn from
    ``families`` (stand -> low-command transition), held for a normal resampling
    interval, after which ordinary sampling resumes. Every family must lie inside
    the level-0 command ranges, so it is valid at every curriculum level.
    Backward low speed gets double weight: it is the weakest direction.
    """

    enabled: bool = False
    fraction: float = 0.15
    settle_s: Range = (0.5, 1.5)
    families: tuple[ColdStartFamily, ...] = (
        (0, 0.15, 0.40, 1.0),  # forward low
        (0, -0.30, -0.15, 2.0),  # backward low (weakest direction, x2)
        (1, 0.15, 0.30, 1.0),  # left
        (1, -0.30, -0.15, 1.0),  # right
        (2, 0.30, 0.80, 1.0),  # yaw CCW low/moderate
        (2, -0.80, -0.30, 1.0),  # yaw CW low/moderate
    )


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
    #: Matches stock UnitreeGo2{Rough,Flat}PPORunnerCfg (0.01). Was 0.005 (half
    #: stock's value) with no recorded justification; aligned 2026-09-25 while
    #: diagnosing the 3-seed overnight collapse (see the action_l2 reward term
    #: for the primary fix; this is a secondary alignment, not shown alone to
    #: be the collapse's cause).
    entropy_coef: float = 0.01
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
    health: HealthGuardSpec = field(default_factory=HealthGuardSpec)
    cold_start: ColdStartSpec = field(default_factory=ColdStartSpec)
    #: Actor observation schema (contract.ACTOR_OBS_SCALES). v1 = raw SI.
    obs_schema: str = OBS_SCHEMA_V1
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

    def with_reward_params(self, name: str, **params: Any) -> VelocityTaskSpec:
        """A copy with ONE reward term's params updated (e.g. std), others untouched."""
        if name not in {t.name for t in self.rewards}:
            raise KeyError(name)
        new_rewards = tuple(
            dataclasses.replace(t, params={**dict(t.params), **params}) if t.name == name else t
            for t in self.rewards
        )
        return self.replace(rewards=new_rewards)

    def with_reward_weight(self, name: str, weight: float) -> VelocityTaskSpec:
        """A copy with ONE reward term's weight changed, others untouched.

        Used to isolate a single per-arm change (e.g. joint_pos_limits weight
        for the 2026-09-25 calf-limit finding) without touching
        default_spec(), so an unrelated run resumed later keeps its own
        original weight instead of silently picking up a new global default.
        """
        if name not in {t.name for t in self.rewards}:
            raise KeyError(name)
        new_rewards = tuple(
            dataclasses.replace(t, weight=float(weight)) if t.name == name else t
            for t in self.rewards
        )
        return self.replace(rewards=new_rewards)

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
        if self.action.clip_actions <= 0:
            p.append("action.clip_actions must be positive")
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
        if c.required_envelope is not None:
            fwd, lat, yaw = (float(v) for v in c.required_envelope)
            lo_x, hi_x = c.initial_lin_vel_x
            lo_y, hi_y = c.initial_lin_vel_y
            lo_z, hi_z = c.initial_ang_vel_z
            if hi_x < fwd:
                p.append(f"initial lin_vel_x {c.initial_lin_vel_x} does not reach required {fwd}")
            if lo_y > -lat or hi_y < lat:
                p.append(f"initial lin_vel_y {c.initial_lin_vel_y} does not cover +-{lat}")
            if lo_z > -yaw or hi_z < yaw:
                p.append(f"initial ang_vel_z {c.initial_ang_vel_z} does not cover +-{yaw}")
        if self.obs_schema not in OBS_SCHEMA_VERSIONS:
            p.append(f"obs_schema {self.obs_schema!r} not in {list(OBS_SCHEMA_VERSIONS)}")
        if self.ppo.actor_obs_normalization:
            p.append("actor_obs_normalization must stay OFF (deploy has no normalizer stats)")
        cur = self.curriculum
        if cur.gate not in CURRICULUM_GATES:
            p.append(f"curriculum gate {cur.gate!r} not in {CURRICULUM_GATES}")
        if cur.gate_kernel_std is not None:
            if not cur.gate_kernel_std > 0:
                p.append("curriculum.gate_kernel_std must be positive")
            if not self.cold_start.enabled:
                p.append(
                    "curriculum.gate_kernel_std needs cold_start (its accumulator lives there)"
                )
        if cur.gate == "unitree_lin_kernel":
            if tuple(c.initial_ang_vel_z) != tuple(c.final_ang_vel_z):
                p.append("unitree_lin_kernel gate: yaw must be at its final range from level 0")
            if not 0.0 < cur.lin_kernel_threshold <= 1.0:
                p.append("curriculum lin_kernel_threshold must be in (0, 1]")
        cs = self.cold_start
        if cs.enabled:
            if not 0.0 < cs.fraction <= 0.5:
                p.append("cold_start.fraction must be in (0, 0.5]")
            lo_s, hi_s = cs.settle_s
            if not 0.0 <= lo_s <= hi_s < c.resampling_time_range_s[0]:
                p.append("cold_start.settle_s must be within [0, resampling min)")
            if not cs.families:
                p.append("cold_start needs at least one command family")
            init = (c.initial_lin_vel_x, c.initial_lin_vel_y, c.initial_ang_vel_z)
            for fam in cs.families:
                axis, lo_f, hi_f, weight = fam
                if axis not in (0, 1, 2) or not lo_f <= hi_f or weight <= 0:
                    p.append(f"cold_start family {fam} malformed")
                    continue
                if lo_f < init[axis][0] or hi_f > init[axis][1]:
                    p.append(f"cold_start family {fam} outside level-0 range {init[axis]}")
                small = (c.small_lin_cmd, c.small_lin_cmd, c.small_yaw_cmd)[axis]
                if min(abs(lo_f), abs(hi_f)) <= small and lo_f * hi_f >= 0:
                    p.append(f"cold_start family {fam} reaches the stand-command band")
        h = self.health
        if h.std_warn > h.std_abort or h.action_p99_warn > h.action_p99_abort:
            p.append("health guard warn threshold above its abort threshold")
        if h.episode_length_warn < h.episode_length_abort:
            p.append("health guard episode-length warn below its abort threshold")
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
        if dr.joint_friction_enabled:
            for name in ("joint_friction_hip_thigh_nm", "joint_friction_calf_nm"):
                lo_, hi_ = getattr(dr, name)
                if not (0.0 <= lo_ <= hi_ <= 5.0):
                    p.append(f"{name} must satisfy 0 <= lo <= hi <= 5 N m")
        if dr.joint_friction_model != "off":
            if dr.joint_friction_enabled:
                p.append("joint_friction_model and the legacy joint_friction_enabled are exclusive")
            fd = friction_distribution(dr)
            p.extend(fd.validate() if fd is not None else [])
        t = self.terminations
        if not (0.0 < t.tilt_limit_rad < math.pi):
            p.append("tilt limit must be in (0, pi): pi is unreachable (tautological)")
        return p


def default_spec() -> VelocityTaskSpec:
    return VelocityTaskSpec()


def smoke_spec(base: VelocityTaskSpec | None = None) -> VelocityTaskSpec:
    """Tiny, fast variant for pipeline smoke tests (same MDP, fewer envs/iters)."""
    base = base or default_spec()
    return base.replace(
        sim=dataclasses.replace(base.sim, num_envs=64),
        ppo=dataclasses.replace(base.ppo, max_iterations=6, save_interval=2),
        curriculum=dataclasses.replace(base.curriculum, min_window_episodes=16),
    )


#: Lab-card command caps (forward m/s, |lateral| m/s, |yaw| rad/s) that every
#: vNext checkpoint must have been trained on.
LAB_CARD_ENVELOPE: tuple[float, float, float] = (0.5, 0.3, 0.8)


def vnext_spec() -> VelocityTaskSpec:
    """PhoenixVelocity vNext: default_spec() plus exactly these changes.

    1. obs_schema v2: base_ang_vel x0.2, joint_vel x0.05 (Unitree GO2 scales);
       actor normalization stays OFF.
    2. Commands: yaw at its full (-1, 1) from iteration 0; initial lin_vel_y
       widened (-0.2, 0.2) -> (-0.3, 0.3); initial lin_vel_x unchanged
       (-0.3, 0.6); finals unchanged (vx -0.8..1.0, vy +-0.5). So level 0 already
       covers the lab-card envelope (0.5 / 0.3 / 0.8), by construction.
    3. Curriculum gate ``unitree_lin_kernel``: only lin x/y expand; yaw score is
       telemetry, never a blocker.
    4. joint_pos_limits -1.0 -> -10.0 (the seed 46 arm's validated value).
    5. Health guard enabled. Hard abort: non-finite, episode length < 800,
       per-joint std > 0.30 (both 50 iterations). Warn-only: action p99 > 3.5,
       learning rate <= 1e-4, KL > 0.05, curriculum stuck at level 0.
    Unchanged on purpose: clip_actions 100, action_l2 -0.01, base-only trunk
    termination, entropy 0.01, adaptive schedule, desired_kl 0.01, init std 1.0,
    every other reward weight.
    """
    base = default_spec().with_reward_weight("joint_pos_limits", -10.0)
    return base.replace(
        obs_schema=OBS_SCHEMA_V2,
        commands=dataclasses.replace(
            base.commands,
            initial_lin_vel_y=(-0.3, 0.3),
            initial_ang_vel_z=base.commands.final_ang_vel_z,
            required_envelope=LAB_CARD_ENVELOPE,
        ),
        curriculum=dataclasses.replace(base.curriculum, gate="unitree_lin_kernel"),
        health=dataclasses.replace(
            base.health, enabled=True, action_p99_hard=False, lr_hard=False, kl_hard=False
        ),
    )


def vnext2_spec() -> VelocityTaskSpec:
    """vnext_spec() plus the cold-start reset mixture (:class:`ColdStartSpec`), nothing else."""
    base = vnext_spec()
    return base.replace(cold_start=dataclasses.replace(base.cold_start, enabled=True))


def vnext3_spec() -> VelocityTaskSpec:
    """vnext2_spec() with ONE reward change: linear tracking std 0.5 -> 0.35.

    Yaw tracking stays at 0.5. The curriculum gate keeps measuring the linear
    kernel at 0.5 (gate_kernel_std) so the curriculum expands on the same
    statistic as vnext2. Evidence: scripts/analyze_startup_reward.py
    (2026-09-26), on fixed trajectories of vnext/vnext2 seed-47 checkpoints,
    tracking std 0.35 raised the 2 s advantage of starting over standing by
    +0.32 (+0.3), +0.25 (-0.3), +0.09 (+0.2); action_l2 = 0 by <= +0.02.
    """
    base = vnext2_spec().with_reward_params("track_lin_vel_xy_exp", std=0.35)
    return base.replace(curriculum=dataclasses.replace(base.curriculum, gate_kernel_std=0.5))


def vnext3f_spec() -> VelocityTaskSpec:
    """vnext3_spec() with ONE change: joint Coulomb friction DR (N m, per env and joint).

    Ranges preregistered in docs/preregistration/2026-10-07-joint-friction-dr.md:
    hips/thighs U(0.1, 0.6), calves U(0.3, 1.5), which bracket the measured GO2
    values (x0.5 to x1.5+ of the hung-robot probe).
    """
    base = vnext3_spec()
    return base.replace(
        domain_randomization=dataclasses.replace(
            base.domain_randomization,
            joint_friction_enabled=True,
            joint_friction_hip_thigh_nm=(0.1, 0.6),
            joint_friction_calf_nm=(0.3, 1.5),
        )
    )


def friction_distribution(dr: DomainRandSpec):
    """The joint-friction FrictionDistribution a DR spec asks for, or None when off."""
    if dr.joint_friction_model == "off":
        return None
    from phoenix.physics.joint_friction import FrictionDistribution

    return FrictionDistribution(
        kind=dr.joint_friction_model,
        scale=tuple(dr.joint_friction_scale),
        jitter=float(dr.joint_friction_jitter),
        broad_coulomb=tuple(dr.joint_friction_broad_nm),
        viscous_hip=tuple(dr.joint_viscous_hip),
        viscous_thigh=tuple(dr.joint_viscous_thigh),
        viscous_calf=tuple(dr.joint_viscous_calf),
        load=tuple(dr.joint_friction_load),
    )


#: Identified on the incumbent (results/friction_physics/ident_I, fit = yaw05_3s free-floor
#: hardware runs, held-out = vx03_4s tape distance): the 6 hypotheses that reproduce the
#: robot's yaw within its run-to-run spread span measured Coulomb x0.75..1.5 with k_load
#: 0..0.1 and viscous 0..0.05; they predict the held-out forward distance 0.62-0.68 m (robot
#: 0.71 m, frictionless 1.06 m). The two training arms share every other setting and budget.
HW_FRICTION = {
    "joint_friction_model": "hardware",
    "joint_friction_scale": (0.75, 1.6),
    "joint_friction_jitter": 0.15,
    "joint_viscous_hip": (0.0, 0.05),
    "joint_viscous_thigh": (0.0, 0.05),
    "joint_viscous_calf": (0.0, 0.05),
    "joint_friction_load": (0.0, 0.1),
}
BROAD_FRICTION = {
    "joint_friction_model": "broad",
    "joint_friction_broad_nm": (0.0, 1.6),
    "joint_viscous_hip": (0.0, 0.1),
    "joint_viscous_thigh": (0.0, 0.1),
    "joint_viscous_calf": (0.0, 0.1),
    "joint_friction_load": (0.0, 0.1),
}


def vnext3hw_spec() -> VelocityTaskSpec:
    """Arm B: vnext3_spec() + the hardware-identified joint friction model (HW_FRICTION)."""
    base = vnext3_spec()
    return base.replace(domain_randomization=dataclasses.replace(base.domain_randomization, **HW_FRICTION))


def vnext3broad_spec() -> VelocityTaskSpec:
    """Arm C: vnext3_spec() + conventional broad friction DR (BROAD_FRICTION): every joint
    U(0, 1.6) N m with no joint identity (covers B's support: max measured 0.96 x 1.6),
    wider viscous, same load range."""
    base = vnext3_spec()
    return base.replace(domain_randomization=dataclasses.replace(base.domain_randomization, **BROAD_FRICTION))


def _cold_start_from_dict(blob: Mapping[str, Any] | None) -> ColdStartSpec:
    if not blob:  # pre-cold-start spec.json: that run had none
        return ColdStartSpec()
    return ColdStartSpec(
        enabled=bool(blob["enabled"]),
        fraction=float(blob["fraction"]),
        settle_s=(float(blob["settle_s"][0]), float(blob["settle_s"][1])),
        families=tuple(
            (int(f[0]), float(f[1]), float(f[2]), float(f[3])) for f in blob["families"]
        ),
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
        # Absent in pre-vNext spec.json files: those runs had no guard and raw obs.
        health=build(HealthGuardSpec, data.get("health", {})),
        cold_start=_cold_start_from_dict(data.get("cold_start")),
        obs_schema=str(data.get("obs_schema", OBS_SCHEMA_V1)),
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
    "CURRICULUM_GATES",
    "ColdStartSpec",
    "CommandSpec",
    "CurriculumSpec",
    "HealthGuardSpec",
    "LAB_CARD_ENVELOPE",
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
    "vnext2_spec",
    "vnext3_spec",
    "vnext3f_spec",
    "vnext3hw_spec",
    "vnext3broad_spec",
    "friction_distribution",
    "vnext_spec",
]
