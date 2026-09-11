"""Explicit Isaac Lab state/command adapters, importable without the simulator.

Coordinate frames are declared, never inferred. A stored base position is
either ``env_local`` (measured from that environment's origin, the Phoenix
capture convention written by
:func:`phoenix.training.episode_outcomes.snapshot_manager_state`) or ``world``
(absolute simulator coordinates). :func:`to_world_position` and
:func:`to_stored_position` are exact inverses for both, so a capture/restore
round trip is the identity for any environment origin, including the nonzero
X and Y offsets a multi-environment scene layout always has.

Command restoration is a declared policy, not an implicit forever-hold. See
:func:`resolve_command_hold`.
"""

from __future__ import annotations

import numpy as np

#: Frames a stored ``base_pos`` may be expressed in.
POSITION_FRAMES = ("env_local", "world")

#: The frame Phoenix captures write. Consumers must still record which frame
#: they resolved and where that declaration came from.
DEFAULT_POSITION_FRAME = "env_local"

#: Declared command-restoration policies (:func:`resolve_command_hold`).
COMMAND_POLICIES = (
    "source_hold_to_onset",
    "fixed_hold",
    "match_control_process",
    "hold_to_episode_end_legacy",
)


def as_numpy(value):
    """Convert a torch tensor, warp array, or sequence to a plain numpy array."""
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        return value.numpy()
    return np.asarray(value)


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


def _checked_position(position, env_origin):
    p = np.asarray(position, dtype=float)
    o = np.asarray(env_origin, dtype=float)
    if p.shape[-1] != 3 or o.shape[-1] != 3:
        raise ValueError("Positions and environment origins must have three components")
    if not np.isfinite(p).all() or not np.isfinite(o).all():
        raise ValueError("Positions and environment origins must be finite")
    return p, o


def to_world_position(position, env_origin, frame=DEFAULT_POSITION_FRAME):
    """Map a stored base position into world coordinates.

    ``env_local`` adds the FULL environment origin (x, y and z). Adding only
    part of it, or subtracting only part of it on capture, displaces a
    restored robot by the rest of the origin, which for the usual grid layout
    is metres of x/y error and lands the robot on another environment's tile.
    """
    if frame not in POSITION_FRAMES:
        raise ValueError(f"Unknown position_frame={frame!r}; expected {POSITION_FRAMES}")
    p, o = _checked_position(position, env_origin)
    return p + o if frame == "env_local" else p


def to_stored_position(position_world, env_origin, frame=DEFAULT_POSITION_FRAME):
    """Exact inverse of :func:`to_world_position`."""
    if frame not in POSITION_FRAMES:
        raise ValueError(f"Unknown position_frame={frame!r}; expected {POSITION_FRAMES}")
    p, o = _checked_position(position_world, env_origin)
    return p - o if frame == "env_local" else p


def resolve_command_hold(policy, *, time_before_onset_seconds=None, hold_seconds=None):
    """Resolve a declared command policy into a hold and its telemetry.

    Returns ``(hold, telemetry)`` where ``hold`` is the number of seconds the
    restored command is pinned before the environment's own velocity-command
    process resumes, ``None`` meaning "do not touch the reset's resample clock
    at all", and ``inf`` meaning "hold for the whole episode".

    Policies:

    ``source_hold_to_onset``
        Replay the source command for exactly the interval that was replayed
        (seed row to failure onset), then return to the normal command
        process. This is the default because it reproduces the source command
        over the window under study and leaves treatment and control envs on
        the same command process afterwards.
    ``fixed_hold``
        Replay the source command for an explicit number of seconds, then
        return to the normal command process.
    ``match_control_process``
        Write the source command value but leave the reset's own resample
        clock untouched, so the seeded env and the control envs share one
        command process from the first step.
    ``hold_to_episode_end_legacy``
        The historical behaviour: pin the source command for the entire
        episode. This makes the seeded env differ from control envs in BOTH
        state and command process, so it is a confound; it is kept only to
        reproduce runs recorded before this was fixed and must be requested
        by name.
    """
    if policy not in COMMAND_POLICIES:
        raise ValueError(f"Unknown command_policy={policy!r}; expected {COMMAND_POLICIES}")
    telemetry = {"command_policy": policy, "command_hold_source": policy}
    if policy == "match_control_process":
        hold = None
    elif policy == "hold_to_episode_end_legacy":
        hold = float("inf")
    elif policy == "fixed_hold":
        if hold_seconds is None or not np.isfinite(hold_seconds) or hold_seconds <= 0:
            raise ValueError("fixed_hold needs a finite positive command_hold_seconds")
        hold = float(hold_seconds)
    else:
        if time_before_onset_seconds is None:
            raise ValueError(
                "source_hold_to_onset needs the source's time to onset; the trajectory has no "
                "timestamps or control_dt, so pick fixed_hold or match_control_process explicitly"
            )
        if not np.isfinite(time_before_onset_seconds) or time_before_onset_seconds < 0:
            raise ValueError("time_before_onset_seconds must be finite and nonnegative")
        hold = float(time_before_onset_seconds)
        if hold <= 0:
            # Seeding at onset itself replays no command interval, so there is
            # nothing to hold; fall through to the normal process rather than
            # inventing a duration.
            hold = None
            telemetry["command_hold_source"] = "source_hold_to_onset_zero_interval"
    telemetry["command_hold_seconds"] = (
        None if hold is None or np.isinf(hold) else float(hold)
    )
    telemetry["command_hold"] = (
        "reset_process" if hold is None else ("episode" if np.isinf(hold) else "seconds")
    )
    return hold, telemetry


class VelocityCommandAdapter:
    """Restore UniformVelocityCommand buffers and disable heading/stand overrides.

    Targets CommandManager.get_term and UniformVelocityCommand.vel_command_b.
    Unsupported managers fail explicitly. ``hold_seconds=None`` leaves the
    reset's own resample clock alone so the seeded env stays on the control
    envs' command process; a finite hold pins the command for that long and
    the term then resamples normally; infinity pins the logged seed command
    for the whole reconstructed episode and is a command-process confound.
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

    def restore(self, env_ids, command, *, hold_seconds=None):
        import torch

        if hold_seconds is not None and (hold_seconds <= 0 or np.isnan(hold_seconds)):
            raise ValueError("hold_seconds must be positive, or None to keep the reset clock")
        buffer = self.term.vel_command_b
        value = torch.as_tensor(command, dtype=buffer.dtype, device=buffer.device)
        if value.shape == (3,):
            value = value.expand(len(env_ids), 3)
        if value.shape != (len(env_ids), 3) or not torch.isfinite(value).all():
            raise ValueError("command must have three finite components per environment")
        self.term.is_heading_env[env_ids] = False
        self.term.is_standing_env[env_ids] = False
        self.term.vel_command_b[env_ids] = value
        if hold_seconds is not None:
            self.term.time_left[env_ids] = hold_seconds
        if not torch.equal(self.term.command[env_ids], value):
            raise RuntimeError("Command restoration readback failed")
        return value.detach().cpu().tolist()


def restore_state(
    env,
    state,
    env_id: int,
    *,
    command_adapter=None,
    command_hold_seconds=None,
    command_telemetry=None,
    position_frame=None,
    controller_history=None,
    require_exact_replay=False,
):
    """Write a validated seed pose, velocities, joints, and command; return telemetry.

    ``position_frame`` overrides the frame declared by the state itself; the
    resolved frame is inverted EXACTLY (full-XYZ origin add for ``env_local``,
    no origin add for ``world``) and reported in the telemetry so a consumer
    cannot guess wrong.

    ``controller_history`` is a
    :class:`phoenix.replay.controller_history.ControllerHistory`; when given,
    the controller state a single state row cannot carry is restored too and
    the returned telemetry reports the achieved replay fidelity. When it is
    ``None`` the telemetry says so explicitly: this is a state-only seed, not
    an exact replay.
    """
    import torch

    from .controller_history import REPLAY_STATE_ONLY, restore_controller_history

    robot = env.scene["robot"]
    ids = torch.as_tensor([env_id], device=env.device, dtype=torch.long)

    def tensor(value):
        return torch.as_tensor(value, device=env.device, dtype=torch.float32)

    frame = position_frame or getattr(state, "position_frame", DEFAULT_POSITION_FRAME)
    if hasattr(env.scene, "env_origins"):
        env_origin = as_numpy(env.scene.env_origins)[env_id]
    else:
        env_origin = np.zeros(3)
    position = to_world_position(state.base_pos, env_origin, frame)
    quat = np.asarray(state.base_quat, dtype=float)
    quat = quat / np.linalg.norm(quat)
    linear = body_to_world(state.base_lin_vel_body, quat)
    angular = body_to_world(state.base_ang_vel_body, quat)
    adapter = command_adapter or VelocityCommandAdapter(env)
    robot.write_root_pose_to_sim(
        torch.cat((tensor(position), tensor(np.roll(quat, 1))))[None], env_ids=ids
    )
    robot.write_root_velocity_to_sim(tensor(np.concatenate((linear, angular)))[None], env_ids=ids)
    robot.write_joint_state_to_sim(
        tensor(state.joint_pos)[None], tensor(state.joint_vel)[None], env_ids=ids
    )
    command = adapter.restore(ids, state.command_vel, hold_seconds=command_hold_seconds)
    record = {
        "env_id": env_id,
        "restored_command": command[0],
        "restored_linear_velocity_world": linear.tolist(),
        "restored_angular_velocity_world": angular.tolist(),
        "position_frame": frame,
        "env_origin": [float(v) for v in env_origin],
        "restored_position_world": [float(v) for v in position],
        "command_hold_seconds": (
            None
            if command_hold_seconds is None or np.isinf(command_hold_seconds)
            else float(command_hold_seconds)
        ),
        "command_hold": (
            "reset_process"
            if command_hold_seconds is None
            else ("episode" if np.isinf(command_hold_seconds) else "seconds")
        ),
        "restoration_evidence": "simulator_write_calls_and_command_buffer_readback",
    }
    if command_telemetry:
        record.update(command_telemetry)
    if controller_history is None:
        if require_exact_replay:
            raise RuntimeError(
                "Exact replay requested but no controller history was supplied; a single state "
                "row cannot restore last_action, the rate limiter, or the actuator delay buffer"
            )
        record.update(
            {
                "replay_fidelity": REPLAY_STATE_ONLY,
                "controller_history_restored": [],
                "controller_state_not_reconstructed": ["controller_history_not_supplied"],
            }
        )
    else:
        record.update(
            restore_controller_history(
                env, env_id, controller_history, require_exact=require_exact_replay
            )
        )
    return record
