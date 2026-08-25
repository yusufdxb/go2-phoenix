"""Headless rollout + video capture for a trained checkpoint.

Used by:
* ``scripts/demo.sh`` (via :mod:`phoenix.demo.benchmark`) to record
  ``sim_baseline.mp4`` / ``sim_adapted.mp4``.
* Unit smoke tests to sanity-check checkpoint loading.

Produces a ``metrics.json`` alongside the video with success rate,
average tracking error, and failure count under the Phoenix
failure-detection rules.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("phoenix.training.evaluate")


@dataclass
class RolloutMetrics:
    num_episodes: int
    mean_episode_return: float
    mean_episode_length_s: float
    success_rate: float
    failure_rate: float
    mean_lin_vel_error: float
    mean_ang_vel_error: float
    # Per-reward-term mean step contribution (reward/step averaged over envs).
    # Keyed by Isaac Lab reward_manager term name (e.g. "track_lin_vel_xy_exp").
    # Empty dict if reward_manager is unavailable on the env (older Isaac Lab).
    per_term_rewards: dict[str, float]
    # Fraction of per-(env, step, motor) action-delta samples whose absolute
    # value >= MAX_DELTA_PER_STEP_RAD (0.175 rad). Matches the Jetson
    # dryrun definition so sim and hardware can be compared 1:1.
    slew_saturation_pct: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a Phoenix checkpoint and record video.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-config", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--num-episodes", type=int, default=64)
    p.add_argument("--video-out", type=Path, default=None)
    p.add_argument(
        "--video-length", type=int, default=500, help="Video length in env steps (50 Hz)"
    )
    p.add_argument("--metrics-out", type=Path, default=None)
    p.add_argument(
        "--slew-saturation-max",
        type=float,
        default=None,
        help="If set, exit non-zero when slew_saturation_pct exceeds this value",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--gui",
        action="store_true",
        help="Run non-headless (visible Isaac Sim window) for screen-recorded demos",
    )
    p.add_argument(
        "--telemetry-out",
        type=Path,
        default=None,
        help="If set, write a per-step CSV of commanded vs actual base velocity",
    )
    p.add_argument(
        "--cam-dist",
        type=float,
        default=2.6,
        help="Follow-cam per-axis offset: smaller frames the robot closer",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    args = parse_args(argv)
    print("[eval] args:", args, flush=True)

    from isaaclab.app import AppLauncher

    # This Isaac Lab version decoupled `headless` from window creation:
    # headless=False alone still runs windowless. A Kit GUI window requires
    # the `visualizer="kit"` argument. --gui uses it; default stays headless.
    launcher_kwargs: dict = {"enable_cameras": args.video_out is not None}
    if args.gui:
        launcher_kwargs["visualizer"] = "kit"
    else:
        launcher_kwargs["headless"] = True
    app_launcher = AppLauncher(**launcher_kwargs)
    simulation_app = app_launcher.app
    print("[eval] app launched", flush=True)
    try:
        return _run(args, simulation_app)
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


def _run(args: argparse.Namespace, simulation_app) -> int:  # noqa: ANN001
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from phoenix.sim_env import build_env_cfg, load_layered_config

    env_cfg_loaded = load_layered_config(args.env_config)
    env_cfg = build_env_cfg(env_cfg_loaded)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed

    # Track the robot in env 0 so the captured video shows the GO2 walking
    # rather than the world origin (which is empty terrain). The default
    # ViewerCfg points /OmniverseKit_Persp at world (0,0,0), for rough
    # terrain that's empty space and produces black frames.
    if args.video_out is not None:
        # Note: env_cfg.viewer.* settings are intentionally NOT set here.
        # In headless+rendering mode ManagerBasedEnv skips constructing the
        # ViewportCameraController, so any env_cfg.viewer.eye/lookat/origin
        # settings would be silently ignored. We instead call
        # sim.set_camera_view() directly below, after env construction.
        env_cfg.viewer.resolution = (1920, 1080)

    task_name = env_cfg_loaded.to_container()["env"]["task_name"]
    render_mode = "rgb_array" if args.video_out else None
    env = gym.make(task_name, cfg=env_cfg, render_mode=render_mode)

    if args.video_out:
        args.video_out.parent.mkdir(parents=True, exist_ok=True)
        # gym >=0.29 RecordVideo emits "<name_prefix>-episode-<N>.mp4". To get
        # the exact path the caller asked for, we record into a temp folder
        # under name_prefix, then rename to the requested video_out path
        # after env.close(). step_trigger (not episode_trigger) so recording
        # starts at step 0, episode_trigger fires on reset and a 16-env
        # vec env's "episode 0" boundary is fuzzy.
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(args.video_out.parent),
            name_prefix=args.video_out.stem,
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    # Unloaded-feet stand-fixture scenario (Phase A): pin the trunk in the air
    # so the rollout measures slew on the OOD fixture contact state. No-op
    # unless the env config carries a `fixture:` block.
    _fixture = getattr(env_cfg, "phoenix_fixture", None)
    if _fixture:
        from phoenix.sim_env.fixture_hold import install_fixture_hold

        n_fix = install_fixture_hold(env, _fixture)
        logger.warning(
            "fixture-hold ACTIVE: %d env(s) trunk-pinned (hold_height=%.2fm, roll=%.2frad)",
            n_fix,
            float(_fixture.get("hold_height_m", 0.55)),
            float(_fixture.get("roll_rad", 0.0)),
        )

    # Use the upstream rsl_rl cfg to stay compatible with whatever version
    # of rsl_rl is installed (actor/critic cfg shape changed in 3.0).
    import importlib.metadata as metadata

    import yaml  # noqa: E402
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    from phoenix.training.agent_cfg import build_runner_cfg

    eval_yaml = {
        "run": {
            "name": "eval",
            "output_dir": "/tmp",
            "log_interval": 1,
            "save_interval": 1,
            "max_iterations": 1,
            "seed": args.seed,
            "device": args.device,
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "runner": {"num_steps_per_env": 24, "empirical_normalization": True},
    }
    _ = yaml  # quiet unused-import warning (yaml reserved for future config loading)
    runner_cfg = build_runner_cfg(eval_yaml, task_name)
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=args.device)
    from phoenix.training.checkpoint import load_runner_checkpoint

    ckpt_info = load_runner_checkpoint(
        runner,
        args.checkpoint,
        load_actor=True,
        load_critic=True,
        load_optimizer=False,
        load_iteration=False,
    )
    if not ckpt_info.get("actor_match", False):
        raise RuntimeError(f"Actor weights did not round-trip from {args.checkpoint}: {ckpt_info}")
    policy = runner.get_inference_policy(device=args.device)

    # ---- Renderer warmup + camera framing --------------------------------
    # Two diagnosed issues on RTX 5070 + Isaac Sim 5.0 headless+rendering
    # (verified 2026-04-30 against Phoenix and IsaacLab test_record_video.py):
    #
    #   1. The /OmniverseKit_Persp annotator returns all-zero (black) frames
    #      for the first several render calls. Empirically ~5+
    #      simulation_app.update() ticks are required AFTER the annotator
    #      is lazily created (by the first env.render() call) before it
    #      starts returning non-empty pixels.
    #
    #   2. ManagerBasedEnv only constructs a ViewportCameraController when
    #      sim.has_gui or a visualizer is active (manager_based_env.py:171).
    #      In fully-headless mode neither holds, so env_cfg.viewer.eye/lookat
    #      are silently ignored and the camera sits at kit-default world
    #      pose, framing the scene from far away. Workaround: call
    #      sim.set_camera_view(eye, target) directly to position
    #      /OmniverseKit_Persp at a useful pose for env 0.
    if args.video_out is not None:
        unwrapped_env = env.unwrapped

        # Manually frame env 0. Use env 0's origin offset so this also works
        # for multi-env scenes where the cloner spreads envs across the world.
        try:
            import torch as _torch  # local import, torch is already loaded

            env_origin = unwrapped_env.scene.env_origins[0]
            if hasattr(env_origin, "cpu"):
                ox, oy, oz = (float(v) for v in env_origin.cpu().tolist())
            else:
                ox, oy, oz = float(env_origin[0]), float(env_origin[1]), float(env_origin[2])
            cam_eye = (ox + 1.8, oy + 1.8, oz + 1.0)
            cam_target = (ox + 0.0, oy + 0.0, oz + 0.3)
            unwrapped_env.sim.set_camera_view(eye=cam_eye, target=cam_target)
            logger.info(
                "framed camera: env0 origin=(%.2f, %.2f, %.2f) eye=(%.2f, %.2f, %.2f)",
                ox,
                oy,
                oz,
                *cam_eye,
            )
            del _torch
        except Exception as cam_err:  # noqa: BLE001
            logger.warning("set_camera_view failed: %r, using default pose", cam_err)

        # Annotator warmup, first env.render() lazily creates it, then we
        # need several Kit ticks for the renderer to populate its buffer.
        try:
            _ = unwrapped_env.render()
        except Exception as warmup_err:  # noqa: BLE001
            logger.warning("warmup render() raised: %r, continuing", warmup_err)
        for _ in range(12):
            simulation_app.update()
        logger.info("renderer warmup: 12 simulation_app.update() ticks complete")

    # ---- Rollout -----------------------------------------------------------
    # rsl_rl 3.0's RslRlVecEnvWrapper.get_observations returns a dict
    # {group_name: tensor}; flatten to the policy obs here.
    obs_raw = env.get_observations()
    if isinstance(obs_raw, tuple):
        obs = obs_raw[0]
    else:
        obs = obs_raw
    if isinstance(obs, dict):
        obs = obs.get("policy", next(iter(obs.values())))
    episode_return = torch.zeros(args.num_envs, device=args.device)
    episode_length = torch.zeros(args.num_envs, device=args.device)

    returns: list[float] = []
    lengths: list[float] = []
    successes = 0
    failures = 0
    lin_err_acc = 0.0
    ang_err_acc = 0.0
    n_steps = 0
    tracking_steps = 0

    # Per-reward-term accumulator. Keyed by term name, stores sum of
    # (mean across envs) per-step reward contributions. Post-rollout we divide
    # by n_steps to recover the mean per-step contribution of each term.
    term_acc: dict[str, float] = {}
    reward_mgr = getattr(env.unwrapped, "reward_manager", None)
    if reward_mgr is not None:
        for term_name in reward_mgr.active_terms:
            term_acc[term_name] = 0.0

    dt_ctrl = env_cfg.decimation * env_cfg.sim.dt  # seconds per env step

    from phoenix.sim2real.safety import MAX_DELTA_PER_STEP_RAD
    from phoenix.training.slew import slew_saturation_rate

    prev_actions_np: np.ndarray | None = None
    slew_sat_acc = 0.0
    slew_sat_steps = 0
    telemetry_rows: list | None = [] if args.telemetry_out else None

    print("[eval] rollout started", flush=True)
    with torch.inference_mode():
        while len(returns) < args.num_episodes:
            actions = policy(obs)
            actions_np = _to_numpy(actions)
            if prev_actions_np is not None and actions_np.shape == prev_actions_np.shape:
                slew_sat_acc += slew_saturation_rate(
                    prev_actions_np, actions_np, threshold=MAX_DELTA_PER_STEP_RAD
                )
                slew_sat_steps += 1
            # First step after an env reset will contribute a small delta because
            # the policy sees the default stand obs again, acceptable noise at
            # thousands of steps; no per-env reset bookkeeping needed.
            prev_actions_np = actions_np
            obs, reward, dones, extras = env.step(actions)
            episode_return += reward
            episode_length += 1
            n_steps += 1

            # Follow-cam: keep the GO2 framed as the velocity-tracking policy
            # walks it away from spawn. Active for GUI (screen-recorded demos)
            # and for video_out captures; metric-only runs are unaffected.
            if args.gui or args.video_out is not None:
                try:
                    rpos = _to_numpy(env.unwrapped.scene["robot"].data.root_pos_w)[0]
                    rx, ry, rz = float(rpos[0]), float(rpos[1]), float(rpos[2])
                    env.unwrapped.sim.set_camera_view(
                        eye=(
                            rx - args.cam_dist,
                            ry - args.cam_dist,
                            rz + args.cam_dist * 0.4,
                        ),
                        target=(rx, ry, rz + 0.30),
                    )
                except Exception:  # noqa: BLE001, S110
                    pass

            # tracking error, done on numpy to avoid torch/warp interop pitfalls.
            # Root velocities are Warp arrays in Isaac Lab 3.x; the matching
            # conversion pattern in synthesize_failure.py proves the .numpy()
            # route works. Keep this block loud: if it skips, we want to know.
            unwrapped = env.unwrapped
            if hasattr(unwrapped, "command_manager"):
                try:
                    cmd_np = _to_numpy(unwrapped.command_manager.get_command("base_velocity"))
                    root = unwrapped.scene["robot"].data
                    lin_b_np = _to_numpy(root.root_lin_vel_b)
                    ang_b_np = _to_numpy(root.root_ang_vel_b)
                    if cmd_np.ndim >= 2 and lin_b_np.ndim >= 2:
                        lin_err_acc += float(
                            np.mean(np.linalg.norm(cmd_np[:, :2] - lin_b_np[:, :2], axis=-1))
                        )
                        ang_err_acc += float(np.mean(np.abs(cmd_np[:, 2] - ang_b_np[:, 2])))
                        tracking_steps += 1
                        if telemetry_rows is not None:
                            telemetry_rows.append(
                                (
                                    n_steps,
                                    time.time(),
                                    float(cmd_np[0, 0]),
                                    float(cmd_np[0, 1]),
                                    float(cmd_np[0, 2]),
                                    float(lin_b_np[0, 0]),
                                    float(lin_b_np[0, 1]),
                                    float(ang_b_np[0, 2]),
                                )
                            )
                    else:
                        if n_steps <= 1:
                            logger.warning(
                                "tracking-error skip: cmd ndim=%d lin ndim=%d",
                                cmd_np.ndim,
                                lin_b_np.ndim,
                            )
                except Exception as err:  # noqa: BLE001
                    if n_steps <= 1:
                        logger.warning("tracking-error exception step=%d: %r", n_steps, err)

            # Per-reward-term accumulation. reward_manager._step_reward is
            # [num_envs, num_terms] with each column being that term's
            # contribution (already weighted, scaled by 1/dt). We average
            # across envs at this step and add to the term's accumulator.
            if reward_mgr is not None and hasattr(reward_mgr, "_step_reward"):
                try:
                    step_rew = reward_mgr._step_reward  # [num_envs, num_terms]
                    term_means = step_rew.mean(dim=0).detach().cpu().numpy()
                    for idx, name in enumerate(reward_mgr.active_terms):
                        term_acc[name] += float(term_means[idx])
                except Exception as err:  # noqa: BLE001
                    if n_steps <= 1:
                        logger.warning(
                            "per-term reward capture failed step=%d: %r",
                            n_steps,
                            err,
                        )

            done_idx = dones.nonzero(as_tuple=False).flatten()
            if len(done_idx) > 0:
                for i in done_idx.tolist():
                    returns.append(float(episode_return[i].item()))
                    ep_len = float(episode_length[i].item())
                    lengths.append(ep_len * dt_ctrl)
                    # termination reason via the time_out buffer
                    time_out = bool(extras.get("time_outs", torch.zeros_like(dones))[i].item())
                    if time_out:
                        successes += 1
                    else:
                        failures += 1
                    episode_return[i] = 0.0
                    episode_length[i] = 0.0

    n_eps = max(len(returns), 1)
    slew_pct = slew_sat_acc / max(slew_sat_steps, 1)
    metrics = RolloutMetrics(
        num_episodes=n_eps,
        mean_episode_return=float(np.mean(returns)),
        mean_episode_length_s=float(np.mean(lengths)),
        success_rate=successes / n_eps,
        failure_rate=failures / n_eps,
        mean_lin_vel_error=lin_err_acc / max(tracking_steps, 1),
        mean_ang_vel_error=ang_err_acc / max(tracking_steps, 1),
        per_term_rewards={k: v / max(n_steps, 1) for k, v in term_acc.items()},
        slew_saturation_pct=slew_pct,
    )
    logger.info("Metrics: %s", metrics)

    if args.slew_saturation_max is not None and slew_pct > args.slew_saturation_max:
        logger.error(
            "slew_saturation_pct=%.4f exceeds --slew-saturation-max=%.4f",
            slew_pct,
            args.slew_saturation_max,
        )
        env.close()
        return 1

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(asdict(metrics), indent=2))

    if args.telemetry_out and telemetry_rows:
        args.telemetry_out.parent.mkdir(parents=True, exist_ok=True)
        with args.telemetry_out.open("w") as tf:
            tf.write("step,wall_time,vx_cmd,vy_cmd,vyaw_cmd,vx_act,vy_act,vyaw_act\n")
            for row in telemetry_rows:
                tf.write(",".join(str(x) for x in row) + "\n")
        print(
            f"[eval] telemetry: {len(telemetry_rows)} rows -> {args.telemetry_out}",
            flush=True,
        )

    env.close()

    # gym >=0.29 RecordVideo writes "<stem>-step-<N>.mp4" with step_trigger
    # (or "<stem>-episode-<N>.mp4" with episode_trigger). Rename to the exact
    # path the caller requested so downstream tools (video_compose) find it.
    if args.video_out is not None:
        target = args.video_out
        # Prefer the canonical step-0 name; fall back to episode-0 then any
        # matching pattern. Skip the target itself in the glob so we don't
        # pick a stale file with the requested name.
        candidates: list[Path] = []
        for name in (f"{target.stem}-step-0.mp4", f"{target.stem}-episode-0.mp4"):
            p = target.parent / name
            if p.exists() and p != target:
                candidates.append(p)
        if not candidates:
            for p in sorted(target.parent.glob(f"{target.stem}-*.mp4")):
                if p != target and p.exists():
                    candidates.append(p)
        if candidates:
            src = candidates[0]
            if target.exists():
                target.unlink()
            src.rename(target)
            logger.info("Renamed %s -> %s", src.name, target.name)
        else:
            logger.warning("No recorded video found at %s", target)
    return 0


def _to_numpy(x):
    """Convert a torch tensor / warp array / ndarray to a plain numpy array.

    Mirrors the helper in phoenix.real_world.synthesize_failure so the
    tracking-error block handles every Isaac Lab 3.x buffer kind without
    silently returning zeros.
    """
    import numpy as np

    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        return x.numpy()
    return np.asarray(x)


def _as_torch(x):
    """Coerce a torch tensor / warp array / ndarray into a contiguous torch.Tensor."""
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        return torch.as_tensor(x.numpy())
    return torch.as_tensor(x)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    sys.exit(main())
