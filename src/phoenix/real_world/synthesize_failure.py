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
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    args = parse_args(argv)
    print("[synth] args:", args, flush=True)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=args.headless).app
    print("[synth] app launched", flush=True)
    try:
        return _run(args, app)
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
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
    print("[synth] env made", flush=True)
    obs, _ = env.reset(seed=args.seed)
    print("[synth] env reset, action_space=", env.action_space.shape, flush=True)

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

            # Some fields are warp arrays in Isaac Lab v3 — convert the whole
            # tensor first, then index, to avoid "Item indexing is not supported
            # on wp.array objects".
            # base_pos is stored *relative to the env origin* so replay can
            # add back whatever origin the replay env uses (broadcast across
            # many envs, each with a different origin).
            env_origin = _to_numpy(unwrapped.scene.env_origins)[0]
            base_pos = _to_numpy(robot.data.root_pos_w)[0] - env_origin
            base_quat = _to_numpy(robot.data.root_quat_w)[0]  # wxyz in IsaacLab
            base_quat = np.roll(base_quat, -1)  # -> xyzw
            base_lin_vel_b = _to_numpy(robot.data.root_lin_vel_b)[0]
            base_ang_vel_b = _to_numpy(robot.data.root_ang_vel_b)[0]
            joint_pos = _to_numpy(robot.data.joint_pos)[0]
            joint_vel = _to_numpy(robot.data.joint_vel)[0]
            cmd = _to_numpy(unwrapped.command_manager.get_command("base_velocity"))[0]
            act = _to_numpy(action)[0]

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
    """Load the actor MLP from a rsl_rl 3.0 checkpoint as a plain torch Sequential."""
    import torch
    import torch.nn as nn

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_sd = ckpt.get("actor_state_dict") or {
        k[len("actor.") :]: v
        for k, v in ckpt.get("model_state_dict", {}).items()
        if k.startswith("actor.")
    }
    if not actor_sd:
        raise RuntimeError(f"Checkpoint {ckpt_path} has no actor weights (keys={list(ckpt)})")

    def layer_idx(k: str) -> int:
        for p in k.split("."):
            if p.isdigit():
                return int(p)
        return -1

    layer_keys = sorted(
        (k for k in actor_sd if k.endswith(".weight") and ("mlp" in k or "actor" in k)),
        key=layer_idx,
    )
    obs_dim = int(actor_sd[layer_keys[0]].shape[1])
    act_dim = int(actor_sd[layer_keys[-1]].shape[0])
    hidden = [int(actor_sd[k].shape[0]) for k in layer_keys[:-1]]

    layers: list[nn.Module] = []
    prev = obs_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ELU())
        prev = h
    layers.append(nn.Linear(prev, act_dim))
    actor = nn.Sequential(*layers).to(device).eval()

    for k in layer_keys:
        idx = layer_idx(k)
        with torch.no_grad():
            actor[idx].weight.copy_(actor_sd[k])
            actor[idx].bias.copy_(actor_sd[k.replace(".weight", ".bias")])

    return lambda obs: actor(obs)


def _to_numpy(x):
    """Convert a torch tensor / warp array / ndarray to a plain numpy array."""
    import numpy as np

    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        # warp arrays need .numpy() but no cpu(); numpy arrays already have .numpy as attribute
        return x.numpy()
    return np.asarray(x)


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
