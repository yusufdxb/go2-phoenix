"""Unloaded-feet (stand-fixture) scenario — rigidly pin the trunk in the air.

Reproduces the CaresLab stand fixture in sim: the trunk is clamped and the feet
hang unloaded. That off-distribution contact state (training has feet loaded
on the ground; the rig has them dangling) drove the 100% RL/RR-thigh slew
saturation on hardware 2026-04-21. This module lets us MEASURE whether a policy
saturates on the fixture in sim, instead of discovering it on the robot.

Mechanism: monkeypatch ``scene.write_data_to_sim`` (called once per physics
substep inside the decimation loop) to re-write the trunk root pose and zero its
velocity for the fixture envs every substep. The trunk becomes effectively
kinematic — a rigid clamp — while the legs actuate in free space, exactly like a
stand. The rig's measured ~0.32 rad L/R asymmetry is modelled as a fixed trunk
roll (``roll_rad``); the held height lifts the feet clear of the ground.

Sim-gated: imports torch and touches the live Isaac Lab env, so it is
lazy-imported from the eval/training entry points only, never from CI.
The pure quaternion helper lives in :mod:`phoenix.sim_env.fixture_math`.
"""

from __future__ import annotations

from typing import Any

from phoenix.sim_env.fixture_math import roll_quat_wxyz

__all__ = ["install_fixture_hold"]


def install_fixture_hold(env: Any, fixture: dict[str, Any]) -> int:
    """Pin the trunk of the fixture envs in the air on every physics substep.

    Args:
        env: The constructed (possibly wrapped) Isaac Lab env.
        fixture: The ``fixture:`` config block — ``enabled``, ``rel_fixture_envs``
            (fraction of envs on the fixture, 1.0 for eval), ``hold_height_m``
            (trunk world height so feet clear the ground), ``roll_rad`` (fixed
            trunk roll modelling the rig's L/R asymmetry).

    Returns:
        Number of envs placed in fixture mode (0 if disabled).
    """
    if not fixture or not fixture.get("enabled", False):
        return 0

    import math

    import torch

    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    robot = unwrapped.scene["robot"]
    device = unwrapped.device
    num_envs = unwrapped.num_envs

    rel = float(fixture.get("rel_fixture_envs", 1.0))
    hold_h = float(fixture.get("hold_height_m", 0.55))
    roll = float(fixture.get("roll_rad", 0.0))

    # Deterministic env selection (first n) so eval rollouts are reproducible.
    n_fix = num_envs if rel >= 1.0 else max(1, int(math.ceil(rel * num_envs)))
    ids = torch.arange(n_fix, device=device, dtype=torch.long)

    # World-frame held pose: env origin + (0,0,hold_h), trunk rolled by roll_rad.
    held_pos = unwrapped.scene.env_origins[ids].clone()
    held_pos[:, 2] = hold_h
    w, x, y, z = roll_quat_wxyz(roll)
    quat = torch.tensor([w, x, y, z], device=device, dtype=held_pos.dtype).repeat(n_fix, 1)
    held_pose = torch.cat([held_pos, quat], dim=1)  # (n_fix, 7)
    zero_vel = torch.zeros((n_fix, 6), device=device, dtype=held_pos.dtype)

    scene = unwrapped.scene
    original_write_data_to_sim = scene.write_data_to_sim

    def _patched_write_data_to_sim(*args, **kwargs):
        original_write_data_to_sim(*args, **kwargs)
        # Re-clamp the trunk after joint targets are written, before sim.step.
        robot.write_root_pose_to_sim(held_pose, env_ids=ids)
        robot.write_root_velocity_to_sim(zero_vel, env_ids=ids)

    scene.write_data_to_sim = _patched_write_data_to_sim
    return n_fix
