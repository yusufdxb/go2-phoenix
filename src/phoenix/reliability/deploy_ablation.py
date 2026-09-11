"""Deployment-mismatch ablations: can a known deploy bug explain a hardware failure?

The 2026-04-21 Gate 7 hardware run saturated the per-step slew clip on 33% of
motor-steps against a sim figure of 0.33%. That was attributed to one cause, the
per-step rate limiter existing only at deploy and never in training. A later
audit found a SECOND independent defect live in the same run: the deploy config
set all four hip joints to 0.0 while training used the IsaacLab nominal pose of
+0.1 left and -0.1 right. A third was found after that: the policy is fed a
zeroed ``base_lin_vel`` even though it is a trained observation term.

Three candidate causes, one historical number, and no way to tell which mattered
by argument. This module makes each one a switch that can be applied in
simulation to an EXISTING checkpoint, so the question becomes a measurement and
needs no retraining.

Every ablation is a pure transformation of the observation or action path, which
is what makes this possible. Nothing here modifies a policy or an environment's
physics.

Observation layout, from ``phoenix.sim2real.observation`` and matching the
trained term order in IsaacLab's velocity task:

    [ base_lin_vel (3), base_ang_vel (3), projected_gravity (3),
      velocity_command (3), joint_pos_rel (12), joint_vel (12),
      last_action (12) ] = 48

Only the flat proprioceptive 48-dim vector is handled. A rough-terrain policy
appends a 187-dim height scan and its slices differ; the builder raises rather
than silently mis-slicing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Slice of the flat observation vector holding body linear velocity.
BASE_LIN_VEL_SLICE = slice(0, 3)
#: Slice holding joint position relative to the training default pose.
JOINT_POS_REL_SLICE = slice(12, 24)
#: Width of the flat proprioceptive observation.
FLAT_OBS_DIM = 48
#: Number of actuated joints.
N_JOINTS = 12

#: Training nominal hip angles in Phoenix joint order (FL, FR, RL, RR), read
#: from IsaacLab's UNITREE_GO2_CFG.init_state: left +0.1, right -0.1.
TRAINING_HIP_POSE = np.array([0.1, -0.1, 0.1, -0.1], dtype=np.float64)
#: The value five deploy configs carried historically.
HISTORICAL_HIP_POSE = np.zeros(4, dtype=np.float64)


def _check_obs(obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != FLAT_OBS_DIM:
        raise ValueError(
            f"expected a (n_envs, {FLAT_OBS_DIM}) flat observation, got {arr.shape}. "
            "A rough-terrain policy's 235-dim vector has different slices and is "
            "not supported; mis-slicing it would silently corrupt the ablation."
        )
    return arr


def zero_base_lin_vel(obs: np.ndarray) -> np.ndarray:
    """Force the body-linear-velocity observation to zero.

    Reproduces ``ros2_policy_node`` feeding ``np.zeros(3)`` for a term the
    policy was trained on. Returns a copy; the input is not mutated.
    """
    out = _check_obs(obs).copy()
    out[:, BASE_LIN_VEL_SLICE] = 0.0
    return out


def hip_offset_observation_shift(
    obs: np.ndarray,
    *,
    training_hips: np.ndarray = TRAINING_HIP_POSE,
    deployed_hips: np.ndarray = HISTORICAL_HIP_POSE,
) -> np.ndarray:
    """Apply the observation half of a wrong ``default_joint_pos``.

    The joint-position observation is ``q - default_q``. A deploy config whose
    default hips are ``deployed_hips`` instead of ``training_hips`` therefore
    reports ``q - deployed`` where the policy expects ``q - training``, an
    offset of ``training - deployed`` on the four hip entries.
    """
    out = _check_obs(obs).copy()
    delta = np.asarray(training_hips, dtype=np.float64) - np.asarray(
        deployed_hips, dtype=np.float64
    )
    if delta.shape != (4,):
        raise ValueError(f"hip pose must have 4 entries, got {delta.shape}")
    joint_block = out[:, JOINT_POS_REL_SLICE]
    joint_block[:, 0:4] += delta
    out[:, JOINT_POS_REL_SLICE] = joint_block
    return out


def hip_offset_action_shift(
    action: np.ndarray,
    action_scale: float,
    *,
    training_hips: np.ndarray = TRAINING_HIP_POSE,
    deployed_hips: np.ndarray = HISTORICAL_HIP_POSE,
) -> np.ndarray:
    """Apply the action half of a wrong ``default_joint_pos``.

    Deploy computes ``target = deployed_default + scale * action`` while the
    simulator applies ``target = training_default + scale * action_sim``. To make
    the simulator produce the deployed target from the same policy output, shift
    the action by ``(deployed - training) / scale`` on the hip entries.

    This is what lets the ablation run without touching the environment's action
    term or retraining anything.
    """
    if not np.isfinite(action_scale) or action_scale == 0.0:
        raise ValueError("action_scale must be finite and nonzero")
    arr = np.asarray(action, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != N_JOINTS:
        raise ValueError(f"expected (n_envs, {N_JOINTS}) actions, got {arr.shape}")
    delta = np.asarray(deployed_hips, dtype=np.float64) - np.asarray(
        training_hips, dtype=np.float64
    )
    out = arr.copy()
    out[:, 0:4] += delta / action_scale
    return out


def deploy_slew_saturation(
    actions: np.ndarray,
    measured_q: np.ndarray,
    default_q: np.ndarray,
    action_scale: float,
    max_delta: float,
) -> float:
    """Fraction of motor-steps the DEPLOY limiter would actually clip.

    This is the deploy-equivalent definition, not the raw action delta. Deploy
    builds ``target = default_q + action_scale * action`` and then clips that
    target against MEASURED joint position, so saturation is
    ``|target - measured_q| > max_delta``. Comparing successive raw policy
    outputs to the same threshold measures a different quantity entirely and is
    not comparable to a hardware slew number.

    Shapes are ``(n_steps, n_envs, 12)`` or ``(n, 12)``.
    """
    if max_delta <= 0:
        raise ValueError("max_delta must be positive")
    a = np.asarray(actions, dtype=np.float64)
    q = np.asarray(measured_q, dtype=np.float64)
    if a.shape != q.shape:
        raise ValueError(f"shape mismatch: actions={a.shape} measured_q={q.shape}")
    d = np.asarray(default_q, dtype=np.float64)
    if d.shape != (N_JOINTS,):
        raise ValueError(f"default_q must have {N_JOINTS} entries, got {d.shape}")
    target = d + action_scale * a
    return float(np.mean(np.abs(target - q) > max_delta))


@dataclass(frozen=True)
class AblationSpec:
    """One cell of the ablation grid."""

    name: str
    zero_base_lin_vel: bool = False
    historical_hip_offset: bool = False
    #: The April checkpoints were trained BEFORE the in-training rate limiter
    #: existed. Enforcing the limiter at evaluation reproduces the historical
    #: deploy mismatch on an unmodified checkpoint; disabling it is the
    #: as-trained baseline.
    enforce_deploy_limiter: bool = False

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "zero_base_lin_vel": self.zero_base_lin_vel,
            "historical_hip_offset": self.historical_hip_offset,
            "enforce_deploy_limiter": self.enforce_deploy_limiter,
        }


def default_grid() -> list[AblationSpec]:
    """Every single cause, the pairs, and the all-on cell, plus a clean control."""
    return [
        AblationSpec("correct"),
        AblationSpec("zero_base_lin_vel", zero_base_lin_vel=True),
        AblationSpec("historical_hip_offset", historical_hip_offset=True),
        AblationSpec("deploy_limiter_only", enforce_deploy_limiter=True),
        AblationSpec("hip_plus_limiter", historical_hip_offset=True, enforce_deploy_limiter=True),
        AblationSpec("velocity_plus_limiter", zero_base_lin_vel=True, enforce_deploy_limiter=True),
        AblationSpec("velocity_plus_hip", zero_base_lin_vel=True, historical_hip_offset=True),
        AblationSpec(
            "all_three",
            zero_base_lin_vel=True,
            historical_hip_offset=True,
            enforce_deploy_limiter=True,
        ),
    ]


def apply_observation_ablation(obs: np.ndarray, spec: AblationSpec) -> np.ndarray:
    out = np.asarray(obs, dtype=np.float64)
    if spec.zero_base_lin_vel:
        out = zero_base_lin_vel(out)
    if spec.historical_hip_offset:
        out = hip_offset_observation_shift(out)
    return out


def apply_action_ablation(
    action: np.ndarray, spec: AblationSpec, action_scale: float
) -> np.ndarray:
    if not spec.historical_hip_offset:
        return np.asarray(action, dtype=np.float64)
    return hip_offset_action_shift(action, action_scale)


__all__ = [
    "BASE_LIN_VEL_SLICE",
    "FLAT_OBS_DIM",
    "HISTORICAL_HIP_POSE",
    "JOINT_POS_REL_SLICE",
    "N_JOINTS",
    "TRAINING_HIP_POSE",
    "AblationSpec",
    "apply_action_ablation",
    "apply_observation_ablation",
    "default_grid",
    "deploy_slew_saturation",
    "hip_offset_action_shift",
    "hip_offset_observation_shift",
    "zero_base_lin_vel",
]
