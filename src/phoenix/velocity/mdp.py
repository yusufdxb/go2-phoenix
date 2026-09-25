"""PhoenixVelocity MDP terms: rewards, terminations, critic observations.

Every function takes ``env`` first (Isaac Lab manager signature) and imports
torch lazily, so this module is importable in CI. Each Isaac-facing function is a
thin adapter over a pure-tensor helper (``*_math``) that the unit tests call
directly with plain torch tensors.

Isaac Lab 3.0 stores asset and sensor data as warp arrays; :func:`as_torch`
converts them (``wp.to_torch``) and passes torch tensors through unchanged, which
is what lets the tests use duck-typed stand-ins holding torch tensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .spec import ACTION_NAME

if TYPE_CHECKING:  # pragma: no cover
    import torch


def as_torch(x: Any) -> torch.Tensor:
    """Torch view of an Isaac Lab 3.0 data field (warp array) or a torch tensor."""
    import torch

    if isinstance(x, torch.Tensor):
        return x
    import warp as wp  # Isaac-only dependency; never reached with torch inputs

    return wp.to_torch(x)


def _ids(ids: Any) -> Any:
    """SceneEntityCfg ids -> an indexer (``slice(None)`` or a list)."""
    return slice(None) if ids is None else ids


# ------------------------------------------------------------------ pure math


def moving_command_mask_math(
    command: torch.Tensor, lin_threshold: float, yaw_threshold: float
) -> torch.Tensor:
    """True where the command asks the robot to move: ||(vx,vy)|| > lin OR |wz| > yaw."""
    import torch

    lin = torch.linalg.norm(command[:, :2], dim=1)
    return (lin > lin_threshold) | (torch.abs(command[:, 2]) > yaw_threshold)


def slew_hinge_math(
    action: torch.Tensor, prev_action: torch.Tensor, action_scale: float, threshold: float
) -> torch.Tensor:
    """Sum over motors of max(0, scale*|a_t - a_{t-1}| - threshold)^2, in rad^2."""
    import torch

    delta = torch.abs(action - prev_action) * float(action_scale)
    return torch.clamp(delta - float(threshold), min=0.0).square().sum(dim=-1)


def feet_air_time_math(
    last_air_time: torch.Tensor, first_contact: torch.Tensor, threshold: float, moving: torch.Tensor
) -> torch.Tensor:
    """Sum over feet touching down this step of (air_time - threshold); 0 when not moving."""
    import torch

    reward = torch.sum((last_air_time - threshold) * first_contact.float(), dim=1)
    return reward * moving.float()


def stand_still_math(
    joint_pos: torch.Tensor, default_joint_pos: torch.Tensor, moving: torch.Tensor
) -> torch.Tensor:
    """L1 joint deviation from the default pose, only where the command is a stand."""
    import torch

    dev = torch.sum(torch.abs(joint_pos - default_joint_pos), dim=1)
    return dev * (~moving).float()


def tilt_angle_math(projected_gravity: torch.Tensor) -> torch.Tensor:
    """Angle (rad) between body z and world up from the body-frame gravity direction.

    Upright g_b = (0, 0, -1) gives 0; upside down gives pi. Normalized and clamped,
    so a slightly non-unit vector cannot produce NaN (upstream ``acos(-g_z)`` can).
    """
    import torch

    g = projected_gravity / torch.linalg.norm(projected_gravity, dim=1, keepdim=True).clamp(
        min=1e-9
    )
    return torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0))


def contact_exceeds_math(
    net_forces_w_history: torch.Tensor, body_ids: Any, threshold: float
) -> torch.Tensor:
    """True per env if any selected body's force norm exceeded ``threshold`` in the history.

    ``net_forces_w_history`` is (envs, history, bodies, 3), as in Isaac Lab.
    """
    import torch

    forces = net_forces_w_history[:, :, _ids(body_ids)]
    peak = torch.max(torch.linalg.norm(forces, dim=-1), dim=1)[0]  # (envs, bodies)
    return torch.any(peak > threshold, dim=1)


def numerical_failure_math(
    tensors: list[torch.Tensor],
    joint_vel: torch.Tensor,
    root_lin_vel: torch.Tensor,
    max_joint_vel: float,
    max_root_lin_vel: float,
) -> torch.Tensor:
    """True per env on any non-finite state value or a physically impossible speed."""
    import torch

    bad = torch.zeros(joint_vel.shape[0], dtype=torch.bool, device=joint_vel.device)
    for t in tensors:
        bad |= ~torch.isfinite(t.reshape(t.shape[0], -1)).all(dim=1)
    # nan_to_num so a NaN does not silently compare False below (already caught above).
    jv = torch.nan_to_num(joint_vel, nan=0.0, posinf=float("inf"), neginf=float("inf"))
    bad |= torch.abs(jv).amax(dim=1) > max_joint_vel
    v = torch.nan_to_num(root_lin_vel, nan=0.0)
    bad |= torch.linalg.norm(v, dim=1) > max_root_lin_vel
    return bad


def episode_mean_kernel_math(
    episode_sum: torch.Tensor, weight: float, step_dt: float, episode_steps: torch.Tensor
) -> torch.Tensor:
    """Time-averaged tracking kernel of an episode from Isaac's weighted episode sum.

    Isaac's RewardManager accumulates ``kernel * weight * step_dt`` per step, so the
    mean kernel is ``sum / (weight * step_dt * steps)``. Clamped to [0, 1].
    """
    import torch

    denom = float(weight) * float(step_dt) * episode_steps.float().clamp(min=1.0)
    return torch.clamp(episode_sum / denom, 0.0, 1.0)


def slew_clip_fraction_math(
    action: torch.Tensor, prev_action: torch.Tensor, action_scale: float, clip_rad: float
) -> torch.Tensor:
    """Fraction of motor-steps whose joint-target step exceeds the deploy slew clip."""
    delta = (action - prev_action).abs() * float(action_scale)
    return (delta > clip_rad).float().mean()


# ------------------------------------------------------------------- rewards


def slew_sat_hinge_l2(env: Any, threshold: float, action_name: str = ACTION_NAME) -> torch.Tensor:
    """Per-motor squared hinge on joint-target slew (rad), scale read from the action term.

    The action scale comes from the LIVE action term cfg, so it cannot drift from
    the action pipeline. Delta is in joint-target radians, as in the corrected
    ``phoenix.sim_env.rewards.slew_sat_hinge_l2``.
    """
    scale = env.action_manager.get_term(action_name).cfg.scale
    if not isinstance(scale, (int, float)):
        raise TypeError(f"slew hinge needs a scalar action scale, got {scale!r}")
    am = env.action_manager
    return slew_hinge_math(am.action, am.prev_action, float(scale), threshold)


def feet_air_time_gated(
    env: Any,
    command_name: str,
    sensor_cfg: Any,
    threshold: float,
    lin_threshold: float,
    yaw_threshold: float,
) -> torch.Tensor:
    """Upstream ``feet_air_time`` but gated on ANY motion command (incl. turn in place).

    Upstream gates on ||(vx, vy)|| > 0.1 only, so turning on the spot got no
    stepping reward. Here yaw commands count as motion too.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    ids = _ids(sensor_cfg.body_ids)
    first_contact = as_torch(sensor.compute_first_contact(env.step_dt))[:, ids]
    last_air_time = as_torch(sensor.data.last_air_time)[:, ids]
    command = env.command_manager.get_command(command_name)
    moving = moving_command_mask_math(command, lin_threshold, yaw_threshold)
    return feet_air_time_math(last_air_time, first_contact, threshold, moving)


def stand_still_joint_deviation_l1(
    env: Any, command_name: str, lin_threshold: float, yaw_threshold: float, asset_cfg: Any
) -> torch.Tensor:
    """L1 deviation from the default pose while commanded to stand (lin AND yaw small)."""
    asset = env.scene[asset_cfg.name]
    ids = _ids(asset_cfg.joint_ids)
    q = as_torch(asset.data.joint_pos)[:, ids]
    q0 = as_torch(asset.data.default_joint_pos)[:, ids]
    moving = moving_command_mask_math(
        env.command_manager.get_command(command_name), lin_threshold, yaw_threshold
    )
    return stand_still_math(q, q0, moving)


# --------------------------------------------------------------- terminations


def bad_tilt(env: Any, limit_angle: float, asset_cfg: Any) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return tilt_angle_math(as_torch(asset.data.projected_gravity_b)) > limit_angle


def trunk_contact(env: Any, threshold: float, sensor_cfg: Any) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    return contact_exceeds_math(
        as_torch(sensor.data.net_forces_w_history), sensor_cfg.body_ids, threshold
    )


def numerical_failure(
    env: Any, max_joint_vel: float, max_root_lin_vel: float, asset_cfg: Any
) -> torch.Tensor:
    d = env.scene[asset_cfg.name].data
    jv = as_torch(d.joint_vel)
    lin = as_torch(d.root_lin_vel_w)
    tensors = [as_torch(d.root_pos_w), as_torch(d.root_quat_w), lin, as_torch(d.root_ang_vel_w)]
    tensors += [as_torch(d.joint_pos), jv]
    return numerical_failure_math(tensors, jv, lin, max_joint_vel, max_root_lin_vel)


# ------------------------------------------------------- critic observations


def feet_contact(env: Any, sensor_cfg: Any, threshold: float) -> torch.Tensor:
    """(envs, feet) 0/1 contact state: peak force norm over the sensor history > threshold."""
    import torch

    forces = as_torch(env.scene.sensors[sensor_cfg.name].data.net_forces_w_history)
    forces = forces[:, :, _ids(sensor_cfg.body_ids)]
    peak = torch.max(torch.linalg.norm(forces, dim=-1), dim=1)[0]
    return (peak > threshold).float()


def feet_contact_force(env: Any, sensor_cfg: Any) -> torch.Tensor:
    """(envs, feet) current net contact force norm, N."""
    import torch

    forces = as_torch(env.scene.sensors[sensor_cfg.name].data.net_forces_w)
    return torch.linalg.norm(forces[:, _ids(sensor_cfg.body_ids)], dim=-1)


def base_height(env: Any, asset_cfg: Any) -> torch.Tensor:
    """(envs, 1) root z in the world frame, m (flat plane at z=0)."""
    return as_torch(env.scene[asset_cfg.name].data.root_pos_w)[:, 2:3]


__all__ = [
    "as_torch",
    "base_height",
    "bad_tilt",
    "contact_exceeds_math",
    "episode_mean_kernel_math",
    "feet_air_time_gated",
    "feet_air_time_math",
    "feet_contact",
    "feet_contact_force",
    "moving_command_mask_math",
    "numerical_failure",
    "numerical_failure_math",
    "slew_clip_fraction_math",
    "slew_hinge_math",
    "slew_sat_hinge_l2",
    "stand_still_joint_deviation_l1",
    "stand_still_math",
    "tilt_angle_math",
    "trunk_contact",
]
