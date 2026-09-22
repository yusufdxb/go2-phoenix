#!/usr/bin/env python3
"""Stage W sign / asymmetry audit (diagnostic): is the forward/backward asymmetry a bug?

Runs against the LIVE training env (``walk_w1.yaml``, the W1/W2 plant) and writes
``<out>/audit.json``. Four parts, each a direct measurement, not a config reading:

A. command sampler: the task's own ``_resample_command`` / ``_update_command`` called
   repeatedly on every env; sign counts, quantiles, tail symmetry, standing fraction.
B. reward: Isaac Lab's ``track_lin_vel_xy_exp`` / ``track_ang_vel_z_exp`` evaluated on
   the six matched sign cases through a stand-in env holding warp arrays.
C. frame: which body axis points at the head (foot link positions in the body frame),
   and the sign of ``root_lin_vel_b`` and of the tracking reward after writing a known
   world velocity along the robot's heading, both directions, four yaw angles.
D. reset / default pose: initial pitch, height, joint pose and the COM position relative
   to the feet's support polygon, after reset and after 1 s of zero action.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--env-config", type=Path, default=Path("configs/env/phoenix_v2/walk_w1.yaml"))
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--resamples", type=int, default=100)
    p.add_argument("--seed", type=int, default=5901)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    from phoenix.sim_app_exit import run_isaac_main

    return run_isaac_main(lambda: _run(args), app, label="phoenix_v2_asymmetry_audit")


def _run(args) -> int:  # noqa: PLR0915
    from types import SimpleNamespace

    import gymnasium as gym
    import numpy as np
    import torch
    import warp as wp
    from isaaclab.envs import mdp

    from phoenix.monitor.stand_metrics import attitude_from_gravity
    from phoenix.sim_env import build_env_cfg, load_layered_config

    def t2n(x):
        if isinstance(x, wp.array):
            x = wp.to_torch(x)
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    loaded = load_layered_config(args.env_config)
    env_cfg = build_env_cfg(loaded)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    env = gym.make(loaded.to_container()["env"]["task_name"], cfg=env_cfg)
    env.reset(seed=args.seed)
    u = env.unwrapped
    robot = u.scene["robot"]
    term = u.command_manager.get_term("base_velocity")
    rew_cfg = u.reward_manager.get_term_cfg("track_lin_vel_xy_exp")
    std_lin = float(rew_cfg.params["std"])
    std_yaw = float(u.reward_manager.get_term_cfg("track_ang_vel_z_exp").params["std"])
    out: dict = {"env_config": str(args.env_config), "seed": args.seed}

    # ---- A. command sampler -------------------------------------------------------
    ids = torch.arange(args.num_envs, device=args.device)
    vx, vy, wz, standing = [], [], [], []
    for _ in range(args.resamples):
        term._resample_command(ids)
        term._update_command()
        c = t2n(term.vel_command_b)
        vx.append(c[:, 0])
        vy.append(c[:, 1])
        wz.append(c[:, 2])
        standing.append(t2n(term.is_standing_env))
    vx, vy, wz, standing = map(np.concatenate, (vx, vy, wz, standing))
    moving = ~standing.astype(bool)
    tails = {}
    for a in (0.1, 0.25, 0.4, 0.6, 0.8, 0.95):
        p_pos, p_neg = float((vx > a).mean()), float((vx < -a).mean())
        se = math.sqrt((p_pos + p_neg) / vx.size)  # SE of the difference under symmetry
        tails[f"{a}"] = {"P(vx>a)": p_pos, "P(vx<-a)": p_neg, "z": (p_pos - p_neg) / se if se else 0.0}
    q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    out["A_command_sampler"] = {
        "n_samples": int(vx.size),
        "ranges_cfg": {k: list(getattr(term.cfg.ranges, k)) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")},
        "heading_command": bool(term.cfg.heading_command),
        "rel_standing_envs_cfg": float(term.cfg.rel_standing_envs),
        "standing_fraction": float(standing.mean()),
        "vx_positive": int((vx > 0).sum()),
        "vx_negative": int((vx < 0).sum()),
        "vx_zero": int((vx == 0).sum()),
        "vx_mean_moving": float(vx[moving].mean()),
        "vx_median_moving": float(np.median(vx[moving])),
        "vx_quantiles": dict(zip(map(str, q), np.quantile(vx, q).tolist(), strict=True)),
        "abs_vx_quantiles_positive": np.quantile(vx[vx > 0], [0.25, 0.5, 0.75]).tolist(),
        "abs_vx_quantiles_negative": np.quantile(-vx[vx < 0], [0.25, 0.5, 0.75]).tolist(),
        "vy_mean": float(vy[moving].mean()),
        "wz_mean": float(wz[moving].mean()),
        "tail_symmetry": tails,
        "curriculum_terms": list(getattr(u.curriculum_manager, "active_terms", [])),
    }

    # ---- B. reward symmetry (the real reward functions) ----------------------------
    cases = [(0.75, 0.75), (0.75, 0.0), (0.75, -0.75), (-0.75, -0.75), (-0.75, 0.0), (-0.75, 0.75)]
    cmd_t = torch.zeros(len(cases), 3, device=args.device)
    vel_t = torch.zeros(len(cases), 3, device=args.device)
    for i, (c, v) in enumerate(cases):
        cmd_t[i, 0], vel_t[i, 0] = c, v
        cmd_t[i, 2], vel_t[i, 2] = c, v  # same values on the yaw channel
    fake_asset = SimpleNamespace(data=SimpleNamespace(
        root_lin_vel_b=wp.from_torch(vel_t.contiguous(), dtype=wp.vec3f),
        root_ang_vel_b=wp.from_torch(vel_t.contiguous(), dtype=wp.vec3f),
    ))
    fake_env = SimpleNamespace(
        scene={"robot": fake_asset},
        command_manager=SimpleNamespace(get_command=lambda _name: cmd_t),
    )
    r_lin = t2n(mdp.track_lin_vel_xy_exp(fake_env, std=std_lin, command_name="base_velocity"))
    r_yaw = t2n(mdp.track_ang_vel_z_exp(fake_env, std=std_yaw, command_name="base_velocity"))
    out["B_reward_symmetry"] = {
        "std_lin": std_lin,
        "std_yaw": std_yaw,
        "cases": [{"cmd": c, "measured": v, "track_lin_vel_xy_exp": float(r_lin[i]),
                   "track_ang_vel_z_exp": float(r_yaw[i])} for i, (c, v) in enumerate(cases)],
        "max_abs_mirror_diff_lin": float(max(abs(r_lin[i] - r_lin[i + 3]) for i in range(3))),
        "max_abs_mirror_diff_yaw": float(max(abs(r_yaw[i] - r_yaw[i + 3]) for i in range(3))),
    }

    # ---- C. frame / forward axis --------------------------------------------------
    env.reset(seed=args.seed)
    quat = torch.as_tensor(t2n(robot.data.root_quat_w), device=args.device)  # xyzw (verified in Phase B)
    root = t2n(robot.data.root_pos_w)
    names = list(robot.body_names)
    link = t2n(robot.data.body_link_pos_w)

    def to_body(p_w, e):
        x, y, z, w = t2n(quat)[e]
        rot = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        return rot.T @ (p_w - root[e])

    feet = {n: to_body(link[0, names.index(n)], 0).tolist() for n in names if n.endswith("_foot")}
    front_x = np.mean([feet[n][0] for n in feet if n.startswith("F")])
    rear_x = np.mean([feet[n][0] for n in feet if n.startswith("R")])
    heading = t2n(robot.data.heading_w)
    probes = []
    for sgn in (+1.0, -1.0):
        vel = torch.zeros(args.num_envs, 6, device=args.device)
        vel[:, 0] = sgn * 0.5 * torch.cos(torch.as_tensor(heading, device=args.device))
        vel[:, 1] = sgn * 0.5 * torch.sin(torch.as_tensor(heading, device=args.device))
        robot.write_root_link_velocity_to_sim_index(root_velocity=vel)
        u.scene.write_data_to_sim()
        u.sim.step(render=False)
        u.scene.update(dt=u.physics_dt)
        vb = t2n(robot.data.root_lin_vel_b)
        term.vel_command_b[:, :] = 0.0
        term.vel_command_b[:, 0] = sgn * 0.5
        rew = t2n(mdp.track_lin_vel_xy_exp(u, std=std_lin, command_name="base_velocity"))
        term.vel_command_b[:, 0] = -sgn * 0.5
        rew_wrong = t2n(mdp.track_lin_vel_xy_exp(u, std=std_lin, command_name="base_velocity"))
        probes.append({
            "world_velocity_along_heading": sgn * 0.5,
            "root_lin_vel_b_x_mean": float(vb[:, 0].mean()),
            "root_lin_vel_b_x_min_max": [float(vb[:, 0].min()), float(vb[:, 0].max())],
            "root_lin_vel_b_y_abs_max": float(np.abs(vb[:, 1]).max()),
            "reward_matching_command_mean": float(rew.mean()),
            "reward_opposite_command_mean": float(rew_wrong.mean()),
        })
        env.reset(seed=args.seed)
    out["C_frame"] = {
        "feet_in_body_frame_env0": feet,
        "front_feet_mean_x": float(front_x),
        "rear_feet_mean_x": float(rear_x),
        "body_plus_x_points_to_head": bool(front_x > rear_x),
        "heading_spread_rad": [float(heading.min()), float(heading.max())],
        "velocity_probes": probes,
    }

    # ---- D. reset and default pose -------------------------------------------------
    env.reset(seed=args.seed)
    g0 = t2n(robot.data.projected_gravity_b)
    roll0, pitch0, _ = attitude_from_gravity(g0)
    h0 = t2n(robot.data.root_pos_w)[:, 2]
    q0 = t2n(robot.data.joint_pos)
    v0 = t2n(robot.data.root_lin_vel_b)
    zero = torch.zeros(args.num_envs, 12, device=args.device)
    for _ in range(50):  # 1 s of zero action = default-pose targets
        env.step(zero)
    g1 = t2n(robot.data.projected_gravity_b)
    roll1, pitch1, _ = attitude_from_gravity(g1)
    link = t2n(robot.data.body_link_pos_w)
    com = t2n(robot.data.body_com_pos_w)
    mass = robot.data.body_mass
    mass = t2n(mass() if callable(mass) else mass)
    root = t2n(robot.data.root_pos_w)
    quat = torch.as_tensor(t2n(robot.data.root_quat_w), device=args.device)
    foot_idx = [names.index(n) for n in names if n.endswith("_foot")]
    offs = []
    for e in range(min(args.num_envs, 256)):
        com_w = (com[e] * mass[e][:, None]).sum(0) / mass[e].sum()
        cb = to_body(com_w, e)
        fb = np.array([to_body(link[e, i], e) for i in foot_idx])
        offs.append((cb[0] - fb[:, 0].mean(), cb[1] - fb[:, 1].mean()))
    offs = np.array(offs)
    out["D_reset_default_pose"] = {
        "at_reset": {
            "pitch_mean_rad": float(pitch0.mean()), "roll_mean_rad": float(roll0.mean()),
            "height_mean_m": float(h0.mean()), "lin_vel_b_abs_max": float(np.abs(v0).max()),
            "joint_pos_env0": q0[0].tolist(), "joint_names": list(robot.joint_names),
        },
        "after_1s_zero_action": {
            "pitch_mean_rad": float(pitch1.mean()), "pitch_std_rad": float(pitch1.std()),
            "roll_mean_rad": float(roll1.mean()),
            "com_minus_support_centroid_x_m": float(offs[:, 0].mean()),
            "com_minus_support_centroid_x_std_m": float(offs[:, 0].std()),
            "com_minus_support_centroid_y_m": float(offs[:, 1].mean()),
            "sign_convention": "pitch > 0 = nose down (projected gravity x > 0); x > 0 = towards head",
        },
        "command_after_reset": "CommandManager.reset resamples each reset env's command at once "
                               "(no stand period); resampling_time_range fixes the next change",
    }
    env.close()
    out["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("C_frame",)} | {
        "C_frame_summary": {k: out["C_frame"][k] for k in ("front_feet_mean_x", "rear_feet_mean_x",
                                                           "body_plus_x_points_to_head", "velocity_probes")}},
        indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
