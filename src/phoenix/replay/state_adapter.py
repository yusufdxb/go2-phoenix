"""Explicit Isaac Lab state/command adapters, importable without the simulator."""

from __future__ import annotations

import numpy as np


def body_to_world(vector, quat_xyzw):
    """Rotate vectors using active body-to-world ROS xyzw orientation."""
    v = np.asarray(vector, dtype=float)
    q = np.asarray(quat_xyzw, dtype=float)
    if v.shape[-1] != 3 or q.shape[-1] != 4:
        raise ValueError("Expected (...,3) vectors and (...,4) xyzw quaternions")
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(q)) or np.any(norms < 1e-10):
        raise ValueError("State must be finite and quaternion must be nonzero")
    q = q / norms
    xyz, w = q[..., :3], q[..., 3:]
    return v + 2 * (w * np.cross(xyz, v) + np.cross(xyz, np.cross(xyz, v)))


class VelocityCommandAdapter:
    """Restore UniformVelocityCommand buffers and disable heading/stand overrides.

    Targets CommandManager.get_term and UniformVelocityCommand.vel_command_b.
    Unsupported managers fail explicitly. Commands are held for hold_seconds;
    infinity fixes the logged seed command for the whole reconstructed episode.
    Nominal resets invoke the normal command reset and restore its distribution.
    """

    def __init__(self, env, name="base_velocity"):
        try:
            self.term = env.command_manager.get_term(name)
        except (AttributeError, KeyError, ValueError) as exc:
            raise RuntimeError(f"Cannot restore command term {name!r}") from exc
        required = ("vel_command_b", "time_left", "is_heading_env", "is_standing_env")
        missing = [field for field in required if not hasattr(self.term, field)]
        if missing:
            raise RuntimeError(f"Unsupported velocity command adapter: missing {missing}")
        # NormalVelocityCommand has additional zero-velocity overrides. Reject
        # until these semantics have a separately verified adapter.
        if any(
            hasattr(self.term, name)
            for name in ("is_zero_vel_x_env", "is_zero_vel_y_env", "is_zero_vel_yaw_env")
        ):
            raise RuntimeError("NormalVelocityCommand requires a dedicated adapter")

    def restore(self, env_ids, command, *, hold_seconds=float("inf")):
        import torch

        if hold_seconds <= 0 or np.isnan(hold_seconds):
            raise ValueError("hold_seconds must be positive")
        buffer = self.term.vel_command_b
        value = torch.as_tensor(command, dtype=buffer.dtype, device=buffer.device)
        if value.shape == (3,):
            value = value.expand(len(env_ids), 3)
        if value.shape != (len(env_ids), 3) or not torch.isfinite(value).all():
            raise ValueError("command must have three finite components per environment")
        self.term.is_heading_env[env_ids] = False
        self.term.is_standing_env[env_ids] = False
        self.term.vel_command_b[env_ids] = value
        self.term.time_left[env_ids] = hold_seconds
        if not torch.equal(self.term.command[env_ids], value):
            raise RuntimeError("Command restoration readback failed")
        return value.detach().cpu().tolist()


def restore_state(env, state, env_id: int, *, command_adapter=None, hold_seconds=float("inf")):
    """Write a validated seed pose, velocities, joints, and command; return telemetry."""
    import torch

    robot = env.scene["robot"]
    ids = torch.as_tensor([env_id], device=env.device, dtype=torch.long)

    def tensor(value):
        return torch.as_tensor(value, device=env.device, dtype=torch.float32)

    position = tensor(state.base_pos)
    if hasattr(env.scene, "env_origins"):
        position = position + env.scene.env_origins[env_id]
    quat = np.asarray(state.base_quat, dtype=float)
    quat = quat / np.linalg.norm(quat)
    linear = body_to_world(state.base_lin_vel_body, quat)
    angular = body_to_world(state.base_ang_vel_body, quat)
    adapter = command_adapter or VelocityCommandAdapter(env)
    robot.write_root_pose_to_sim(torch.cat((position, tensor(np.roll(quat, 1))))[None], env_ids=ids)
    robot.write_root_velocity_to_sim(tensor(np.concatenate((linear, angular)))[None], env_ids=ids)
    robot.write_joint_state_to_sim(
        tensor(state.joint_pos)[None], tensor(state.joint_vel)[None], env_ids=ids
    )
    command = adapter.restore(ids, state.command_vel, hold_seconds=hold_seconds)
    return {
        "env_id": env_id,
        "restored_command": command[0],
        "restored_linear_velocity_world": linear.tolist(),
        "restored_angular_velocity_world": angular.tolist(),
        "command_hold_seconds": None if np.isinf(hold_seconds) else hold_seconds,
        "command_hold": "episode" if np.isinf(hold_seconds) else "seconds",
        "restoration_evidence": "simulator_write_calls_and_command_buffer_readback",
    }
