#!/usr/bin/env python3
"""Phoenix v2 phases B/C/K (stand): physically scored Isaac Lab rollouts of one checkpoint.

One run = one env config x one checkpoint x ``--num-envs`` independent 20 s episodes
(one episode per env; data after an env's first termination is discarded). Every
policy step logs the four command layers the deploy stack distinguishes:

1. ``raw``        the actor output (unclamped network mean),
2. ``requested``  what the TRAINED plant would execute: ``default + scale * clip(raw, -1, 1)``
                  (``RslRlVecEnvWrapper(clip_actions=1.0)`` clamps before the action term),
3. ``sent``       the joint target the action term actually applied after the soft
                  limiter (``processed_actions``),
4. ``q``          the measured joint position (and computed / applied torque).

Attitude comes from ``projected_gravity_b``, which has no quaternion-order ambiguity;
the run also checks which quaternion order of ``root_quat_w`` reproduces it and records
the answer (audit H6: the legacy evaluator assumed wxyz).

Physical standing success (preregistered in docs/research/EXPERIMENT.md, amendment 1):
no trunk contact, |roll| and |pitch| <= 0.40 rad for the whole episode, no request
beyond the deploy abort band, and the per-episode execution-fidelity gate
(altered <= 5 % overall and on every joint, RMS <= 0.01 rad). Timeout is not success.

Usage::

    source scripts/_activate.sh
    PYTHONPATH=src python scripts/phoenix_v2_sim_stand.py --checkpoint <model.pt> \
        --env-config configs/env/phoenix_v2/limiter_b1_measured_q.yaml \
        --num-envs 128 --seed 1001 --out results/phoenix_v2/sim_limiter/b1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path


def alive_to_valid(alive):
    """Steps that belong to each env's first episode (the terminating step included)."""
    import numpy as np

    T = alive.shape[0]
    first_end = np.where((~alive).any(axis=0), np.argmax(~alive, axis=0), T)
    return np.arange(T)[:, None] < first_end[None, :]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-config", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--seed", type=int, default=1001)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--save-steps", action="store_true", help="save per-step arrays (npz)")
    p.add_argument(
        "--bridge-records-envs",
        type=int,
        default=0,
        help="write the first N envs as bridge-format tick records (monitor input)",
    )
    p.add_argument(
        "--degrade",
        default=None,
        help="JOINT:scale override written into provenance only (the env config applies it)",
    )
    return p.parse_args(argv)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    args = parse_args(argv)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    from phoenix.sim_app_exit import run_isaac_main

    return run_isaac_main(lambda: _run(args), app, label="phoenix_v2_sim_stand")


def _quat_gravity(q, order):
    """Projected gravity implied by a unit quaternion (body<-world), numpy (N,4)."""
    import numpy as np

    if order == "wxyz":
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    else:
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # R (body->world) third row gives world z in body coords: g_b = R^T [0,0,-1]
    gx = -(2 * (x * z - w * y))
    gy = -(2 * (y * z + w * x))
    gz = -(1 - 2 * (x * x + y * y))
    return np.stack([gx, gy, gz], axis=1)


def _run(args) -> int:
    import gymnasium as gym
    import importlib.metadata as metadata

    import numpy as np
    import torch
    import warp as wp
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    from phoenix.monitor.stand_metrics import score_stand_rollout
    from phoenix.sim2real.export import checkpoint_has_obs_normalizer
    from phoenix.sim2real.go2_model import LIMIT_ABORT_BAND_RAD, POLICY_JOINT_ORDER, limits_in_order
    from phoenix.sim_env import build_env_cfg, load_layered_config
    from phoenix.training.agent_cfg import build_runner_cfg
    from phoenix.training.checkpoint import load_runner_checkpoint

    def t2n(x):
        if isinstance(x, wp.array):
            x = wp.to_torch(x)
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    t0 = time.time()
    loaded = load_layered_config(args.env_config)
    container = loaded.to_container()
    env_cfg = build_env_cfg(loaded)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    task = container["env"]["task_name"]
    env = RslRlVecEnvWrapper(gym.make(task, cfg=env_cfg), clip_actions=1.0)
    u = env.unwrapped
    robot = u.scene["robot"]
    term = u.action_manager.get_term("joint_pos")
    joint_names = [robot.joint_names[i] for i in (term._joint_ids if not isinstance(term._joint_ids, slice) else range(12))]
    if tuple(joint_names) != tuple(POLICY_JOINT_ORDER):
        raise RuntimeError(f"sim joint order {joint_names} != POLICY_JOINT_ORDER")
    lo, hi = limits_in_order(POLICY_JOINT_ORDER)
    default = t2n(term._offset)
    if default.ndim == 2:
        if not np.allclose(default, default[:1]):
            raise RuntimeError("per-env default offsets differ; layer 2 would be ambiguous")
        default = default[0]
    scale = float(term._scale) if isinstance(term._scale, (int, float)) else t2n(term._scale)
    limiter = {
        k: getattr(term.cfg, k, None) for k in ("max_delta_per_step", "clip_mode", "clip_ref_noise")
    }
    limiter["class"] = type(term).__name__

    use_norm = checkpoint_has_obs_normalizer(args.checkpoint)
    runner_yaml = {
        "run": {"name": "eval", "output_dir": "/tmp", "log_interval": 1, "save_interval": 1,
                "max_iterations": 1, "seed": args.seed, "device": args.device},
        "algorithm": {"class_name": "PPO", "value_loss_coef": 1.0, "use_clipped_value_loss": True,
                      "clip_param": 0.2, "entropy_coef": 0.005, "num_learning_epochs": 5,
                      "num_mini_batches": 4, "learning_rate": 1.0e-3, "schedule": "adaptive",
                      "gamma": 0.99, "lam": 0.95, "desired_kl": 0.01, "max_grad_norm": 1.0},
        "policy": {"class_name": "ActorCritic", "init_noise_std": 1.0,
                   "actor_hidden_dims": [512, 256, 128], "critic_hidden_dims": [512, 256, 128],
                   "activation": "elu"},
        "runner": {"num_steps_per_env": 24, "empirical_normalization": use_norm},
    }
    rcfg = handle_deprecated_rsl_rl_cfg(build_runner_cfg(runner_yaml, task), metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, rcfg.to_dict(), log_dir=None, device=args.device)
    info = load_runner_checkpoint(runner, args.checkpoint, load_actor=True, load_critic=True,
                                  load_optimizer=False, load_iteration=False)
    if not info.get("actor_match", False):
        raise RuntimeError(f"actor did not round-trip: {info}")
    policy = runner.get_inference_policy(device=args.device)

    dt = env_cfg.decimation * env_cfg.sim.dt
    T = int(round(env_cfg.episode_length_s / dt))
    N = args.num_envs
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    # rsl_rl 3.x policies take the whole observation TensorDict.

    shape = (T, N, 12)
    raw = np.zeros(shape, np.float32)
    req = np.zeros(shape, np.float32)
    sent = np.zeros(shape, np.float32)
    q0 = np.zeros(shape, np.float32)
    q1 = np.zeros(shape, np.float32)
    tau_c = np.zeros(shape, np.float32)
    tau_a = np.zeros(shape, np.float32)
    kp_eff = np.zeros((N, 12), np.float32)
    grav = np.zeros((T, N, 3), np.float32)
    height = np.zeros((T, N), np.float32)
    linv = np.zeros((T, N, 3), np.float32)
    angv = np.zeros((T, N, 3), np.float32)
    cmd = np.zeros((T, N, 3), np.float32)
    alive = np.ones((T, N), bool)
    contact_term = np.zeros(N, bool)
    ended = np.zeros(N, bool)
    quat_err = {"wxyz": [], "xyzw": []}

    term_names = list(u.termination_manager.active_terms)
    with torch.inference_mode():
        for k in range(T):
            a = policy(obs)
            raw[k] = t2n(a)
            req[k] = default + scale * np.clip(raw[k], -1.0, 1.0)
            q0[k] = t2n(robot.data.joint_pos)
            cmd[k] = t2n(u.command_manager.get_command("base_velocity"))
            obs, _, dones, _ = env.step(a)
            if isinstance(obs, tuple):
                obs = obs[0]
            sent[k] = t2n(term.processed_actions)
            q1[k] = t2n(robot.data.joint_pos)
            tau_c[k] = t2n(robot.data.computed_torque)
            tau_a[k] = t2n(robot.data.applied_torque)
            g = t2n(robot.data.projected_gravity_b)
            grav[k] = g
            quat = t2n(robot.data.root_quat_w)
            if k % 50 == 0:
                for order in quat_err:
                    quat_err[order].append(float(np.abs(_quat_gravity(quat, order) - g).max()))
            height[k] = t2n(robot.data.root_pos_w)[:, 2] - t2n(u.scene.env_origins)[:, 2]
            linv[k] = t2n(robot.data.root_lin_vel_b)
            angv[k] = t2n(robot.data.root_ang_vel_b)
            d = t2n(dones).astype(bool)
            alive[k] = ~ended
            if d.any():
                for name in term_names:
                    if name == "time_out":
                        continue
                    hit = t2n(u.termination_manager.get_term(name)).astype(bool)
                    contact_term |= hit & d & ~ended
                ended |= d
    # Explicit (DC-motor) actuators hold the PD gains; the PhysX joint drive is zero.
    for act in robot.actuators.values():
        ids = act.joint_indices
        ids = list(range(12)) if isinstance(ids, slice) else [int(i) for i in t2n(ids)]
        kp_eff[:, ids] = t2n(act.stiffness)
    env.close()

    metrics, episodes = score_stand_rollout(
        raw=raw, req=req, sent=sent, q0=q0, q1=q1, tau_c=tau_c, tau_a=tau_a, grav=grav,
        height=height, linv=linv, angv=angv, cmd=cmd, alive=alive, contact_term=contact_term,
        ended=ended, default=default, lo=lo, hi=hi, dt=dt, abort_band=LIMIT_ABORT_BAND_RAD,
        joint_names=list(POLICY_JOINT_ORDER),
    )
    valid = alive_to_valid(alive)
    summary = {
        "schema": "phoenix-v2-sim-stand/v1",
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha(args.checkpoint),
        "env_config": str(args.env_config),
        "env_config_sha256": _sha(args.env_config),
        "env_resolved": container,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()),
        "seed": args.seed,
        "num_envs": N,
        "steps_per_episode": T,
        "dt_s": dt,
        "limiter": limiter,
        "action_scale": scale,
        "quaternion_order_check": {o: max(v) for o, v in quat_err.items()},
        **metrics,
        "wall_s": time.time() - t0,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    with (args.out / "episodes.jsonl").open("w") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep) + "\n")
    if args.save_steps:
        np.savez_compressed(args.out / "steps.npz", raw=raw, req=req, sent=sent, q0=q0, q1=q1,
                            tau_c=tau_c, tau_a=tau_a, grav=grav, valid=valid, cmd=cmd, linv=linv,
                            angv=angv, height=height)
    if args.bridge_records_envs:
        _write_bridge_records(args, raw, req, sent, q0, tau_a, valid, dt, default, scale,
                              t2n_kp=kp_eff)
    print(json.dumps({k: summary[k] for k in (
        "label", "success_rate", "survival_rate", "mean_primary_score", "fidelity_pass_rate",
        "altered_fraction", "rms_modification_rad", "distortion_D", "raw_out_of_range_fraction",
        "effort_saturation_fraction", "attitude_violation_episode_rate", "max_abs_roll_rad",
        "max_abs_pitch_rad", "quaternion_order_check", "wall_s")}, indent=1), flush=True)
    return 0


def _write_bridge_records(args, raw, req, sent, q0, tau_a, valid, dt, default, scale, t2n_kp):
    """Per-env bridge-format tick records for the actuator monitor (Unitree motor order)."""
    import numpy as np

    from phoenix.sim2real.motor_crc import PHOENIX_FOR_MOTOR

    perm = np.asarray(PHOENIX_FOR_MOTOR)
    outdir = args.out / "bridge_records"
    outdir.mkdir(exist_ok=True)
    T = raw.shape[0]
    for e in range(min(args.bridge_records_envs, raw.shape[1])):
        with (outdir / f"env{e:03d}.jsonl").open("w") as fh:
            for k in range(T):
                if not valid[k, e]:
                    break
                rec = {
                    "record": "tick",
                    "t_mono_ns": int(round(k * dt * 1e9)),
                    "tick": k,
                    "mode": "policy",
                    "publish": True,
                    "cmd_is_new": True,
                    "q_unitree": [float(v) for v in q0[k, e][perm]],
                    "dq_unitree": [0.0] * 12,
                    "tau_est_unitree": [float(v) for v in tau_a[k, e][perm]],
                    "final_target_unitree": [float(v) for v in sent[k, e][perm]],
                    "requested_target_unitree": [float(v) for v in sent[k, e][perm]],
                    "kp_unitree": [float(v) for v in t2n_kp[e][perm]],
                    "kd_unitree": [0.5] * 12,
                    "slew_clip": [False] * 12,
                    "limit_clip": [False] * 12,
                    "policy": {
                        "raw_action": [float(v) for v in raw[k, e]],
                        "requested_target": [float(v) for v in req[k, e]],
                        "target": [float(v) for v in sent[k, e]],
                    },
                }
                fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    sys.exit(main())
