"""Synthesize a failure trajectory parquet without the real GO2.

Runs a short rollout of a *deliberately under-trained* policy in a
slippery / perturbed env, logs every step via
:class:`phoenix.real_world.TrajectoryLogger`, and flags the steps that
the :class:`phoenix.real_world.FailureDetector` flags. The output is a
drop-in substitute for a parquet captured on the real robot — same
schema, same reader, usable by ``replay`` and the failure curriculum.

Use sparingly: a synthesized failure is still a *sim* failure. Real
hardware parquets remain the ground truth, but this script lets the
full Phoenix loop smoke-test end-to-end before the robot is available.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("phoenix.real_world.synthesize_failure")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a synthetic failure parquet for testing.")
    p.add_argument("--env-config", type=Path, default=Path("configs/env/slippery.yaml"))
    p.add_argument(
        "--checkpoint", type=Path, default=None, help="Optional checkpoint; else random actions"
    )
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument(
        "--steps", type=int, default=400, help="Control steps to log (50 Hz -> 8s default)"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--headless", action="store_true", default=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=args.headless).app
    try:
        return _run(args, app)
    finally:
        app.close()


def _run(args: argparse.Namespace, simulation_app) -> int:  # noqa: ANN001
    import gymnasium as gym
    import numpy as np
    import torch

    from phoenix.real_world.failure_detector import FailureDetector
    from phoenix.real_world.trajectory_logger import TrajectoryLogger, TrajectoryStep
    from phoenix.sim_env import build_env_cfg, load_layered_config

    env_cfg_loaded = load_layered_config(args.env_config)
    env_cfg = build_env_cfg(env_cfg_loaded)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed

    task = env_cfg_loaded.to_container()["env"]["task_name"]
    env = gym.make(task, cfg=env_cfg, render_mode=None)
    obs, _ = env.reset(seed=args.seed)

    policy = _load_policy(args.checkpoint, args.device) if args.checkpoint else None

    detector = FailureDetector()
    dt_ctrl = env_cfg.decimation * env_cfg.sim.dt
    action_dim = int(env.action_space.shape[-1])

    rng = np.random.default_rng(args.seed)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TrajectoryLogger(args.output) as log:
        for step in range(args.steps):
            if policy is not None:
                obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
                with torch.inference_mode():
                    action = policy(obs_tensor)
            else:
                action = torch.as_tensor(
                    rng.standard_normal((args.num_envs, action_dim)) * 0.6,
                    device=args.device,
                    dtype=torch.float32,
                )

            obs, _reward, terminated, truncated, _info = env.step(action)
            done = terminated | truncated

            base_pos = _to_numpy(robot.data.root_pos_w[0])
            base_quat = _to_numpy(robot.data.root_quat_w[0])  # wxyz in IsaacLab
            # convert wxyz→xyzw to match ROS / parquet schema
            base_quat = np.roll(base_quat, -1)
            base_lin_vel_b = _to_numpy(robot.data.root_lin_vel_b[0])
            base_ang_vel_b = _to_numpy(robot.data.root_ang_vel_b[0])
            joint_pos = _to_numpy(robot.data.joint_pos[0])
            joint_vel = _to_numpy(robot.data.joint_vel[0])
            cmd = _to_numpy(unwrapped.command_manager.get_command("base_velocity")[0])
            act = _to_numpy(action[0])

            roll, pitch, _yaw = _rpy_from_quat_xyzw(base_quat)
            height = float(base_pos[2])

            event = detector.step(
                timestamp_s=step * dt_ctrl,
                pitch_rad=float(pitch),
                roll_rad=float(roll),
                base_height_m=height,
                cmd_lin_vel=cmd[:2],
                actual_lin_vel=base_lin_vel_b[:2],
            )

            log.append(
                TrajectoryStep(
                    step=step,
                    timestamp_s=step * dt_ctrl,
                    base_pos=base_pos.astype(np.float32),
                    base_quat=base_quat.astype(np.float32),
                    base_lin_vel_body=base_lin_vel_b.astype(np.float32),
                    base_ang_vel_body=base_ang_vel_b.astype(np.float32),
                    joint_pos=joint_pos[:12].astype(np.float32),
                    joint_vel=joint_vel[:12].astype(np.float32),
                    command_vel=cmd[:3].astype(np.float32),
                    action=act[:12].astype(np.float32),
                    contact_forces=np.zeros(4, dtype=np.float32),
                    failure_flag=event is not None,
                    failure_mode=event.mode.value if event else None,
                )
            )

            if bool(done[0].item()):
                obs, _ = env.reset()

    logger.info("Wrote %d steps to %s", args.steps, args.output)
    env.close()
    return 0


def _load_policy(ckpt_path: Path, device: str):
    import torch
    from rsl_rl.modules import ActorCritic

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]

    def dims(prefix: str) -> list[int]:
        out: list[int] = []
        i = 0
        while f"{prefix}{i * 2}.weight" in state:
            out.append(state[f"{prefix}{i * 2}.weight"].shape[0])
            i += 1
        return out[:-1] if len(out) >= 2 else [512, 256, 128]

    obs_dim = state["actor.0.weight"].shape[1]
    act_dim = 12
    actor_critic = ActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=act_dim,
        actor_hidden_dims=dims("actor."),
        critic_hidden_dims=dims("critic."),
        activation="elu",
    ).to(device)
    actor_critic.load_state_dict(state, strict=False)
    actor_critic.eval()
    return lambda obs: actor_critic.actor(obs)


def _to_numpy(x):
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return x.numpy() if hasattr(x, "numpy") else x


def _rpy_from_quat_xyzw(q):
    """Return (roll, pitch, yaw) from a unit quaternion (x,y,z,w)."""
    import numpy as np

    x, y, z, w = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    sys.exit(main())
