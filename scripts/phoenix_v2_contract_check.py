#!/usr/bin/env python3
"""Deploy contract v3 against the LIVE Isaac Lab training plant (Phase 1).

Feeds scripted action sequences, including pathological values, through the same
``RslRlVecEnvWrapper(clip_actions=1.0)`` the trainer uses, and after every step compares
the env's own tensors with the deploy-side pure functions:

* ``ActionManager.action`` (what ``last_action`` and ``action_rate_l2`` read) against
  ``policy_action_map(...)[0]``;
* the action term's applied target against ``policy_action_map(...)[1]``;
* the full 48-dim policy observation against the deploy ``ObservationBuilder`` fed the
  simulator's state (observation noise off for this check only);
* the env's joint order against ``POLICY_JOINT_ORDER``.

Usage::

    source scripts/_activate.sh
    PYTHONPATH=src python scripts/phoenix_v2_contract_check.py \
        --env-config configs/env/phoenix_v2/walk_w1.yaml --out results/phoenix_v2/contract/w1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PATHOLOGICAL = [-10.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 10.0]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--env-config", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    from phoenix.sim_app_exit import run_isaac_main

    return run_isaac_main(lambda: _run(args), app, label="phoenix_v2_contract_check")


def _run(args) -> int:
    import gymnasium as gym
    import numpy as np
    import torch
    import warp as wp
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from phoenix.sim2real.action_map import policy_action_map
    from phoenix.sim2real.go2_model import POLICY_JOINT_ORDER, TRAINING_DEFAULT_JOINT_POS
    from phoenix.sim2real.observation import JointOrder, ObservationBuilder
    from phoenix.sim2real.safety import TRAINED_ACTION_CLIP
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
    env_cfg.observations.policy.enable_corruption = False  # exact comparison only
    task = loaded.to_container()["env"]["task_name"]
    env = RslRlVecEnvWrapper(gym.make(task, cfg=env_cfg), clip_actions=1.0)
    u = env.unwrapped
    robot = u.scene["robot"]
    term = u.action_manager.get_term("joint_pos")
    ids = term._joint_ids
    ids = list(range(12)) if isinstance(ids, slice) else list(ids)
    names = tuple(robot.joint_names[i] for i in ids)
    scale = float(term._scale) if isinstance(term._scale, (int, float)) else None
    default_env = t2n(term._offset)[0]
    default_dep = np.asarray([TRAINING_DEFAULT_JOINT_POS[n] for n in POLICY_JOINT_ORDER], np.float32)
    builder = ObservationBuilder(JointOrder(POLICY_JOINT_ORDER), TRAINING_DEFAULT_JOINT_POS)
    limiter_class = type(term).__name__

    rng = np.random.default_rng(args.seed)
    seq = []
    for v in PATHOLOGICAL:  # each value held on every joint, then mixed rows
        seq += [np.full(12, v, np.float32)] * 3
    while len(seq) < args.steps:
        seq.append(rng.choice(PATHOLOGICAL, 12).astype(np.float32))
    seq = seq[: args.steps]

    worst = {"action_manager": 0.0, "target": 0.0, "obs": 0.0, "obs_last_action": 0.0}
    per_term = {}
    obs = env.get_observations()
    obs_t = obs["policy"] if hasattr(obs, "keys") else obs
    last_fed = np.zeros((args.num_envs, 12), np.float32)
    n_reset = 0
    with torch.inference_mode():
        for row in seq:
            raw = np.tile(row, (args.num_envs, 1))
            # Deploy side, BEFORE the env acts: build the obs from the sim state.
            dep_obs = np.stack([
                builder.build(
                    base_lin_vel=t2n(robot.data.root_lin_vel_b)[e],
                    base_ang_vel=t2n(robot.data.root_ang_vel_b)[e],
                    projected_gravity=t2n(robot.data.projected_gravity_b)[e],
                    velocity_command=t2n(u.command_manager.get_command("base_velocity"))[e],
                    joint_pos=t2n(robot.data.joint_pos)[e][ids],
                    joint_vel=t2n(robot.data.joint_vel)[e][ids],
                    last_action=last_fed[e],
                )
                for e in range(args.num_envs)
            ])
            env_obs = t2n(obs_t)
            d = np.abs(env_obs - dep_obs)
            worst["obs"] = max(worst["obs"], float(d.max()))
            worst["obs_last_action"] = max(worst["obs_last_action"], float(d[:, 36:48].max()))
            for name, sl in builder.term_slices().items():
                per_term[name] = max(per_term.get(name, 0.0), float(d[:, sl].max()))
            fed, target = policy_action_map(raw, default_dep, scale, TRAINED_ACTION_CLIP)
            obs, _, dones, _ = env.step(torch.as_tensor(raw, device=args.device))
            obs_t = obs["policy"] if hasattr(obs, "keys") else obs
            worst["action_manager"] = max(
                worst["action_manager"], float(np.abs(t2n(u.action_manager.action) - fed).max())
            )
            worst["target"] = max(worst["target"], float(np.abs(t2n(term.processed_actions) - target).max()))
            dn = t2n(dones).astype(bool)
            last_fed = np.where(dn[:, None], 0.0, fed).astype(np.float32)  # reset zeroes _action
            n_reset += int(dn.sum())
    env.close()

    tol = {"action_manager": 0.0, "target": 1e-6, "obs": 1e-5, "obs_last_action": 0.0}
    passed = {k: worst[k] <= tol[k] for k in tol}
    passed["joint_order"] = names == tuple(POLICY_JOINT_ORDER)
    passed["default_pose"] = bool(np.allclose(default_env, default_dep, atol=1e-6))
    passed["action_scale"] = scale == 0.25
    result = {
        "schema": "phoenix-v2-contract-check/v1",
        "env_config": str(args.env_config),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()),
        "action_term_class": limiter_class,
        "action_scale": scale,
        "joint_names_env": list(names),
        "steps": len(seq),
        "num_envs": args.num_envs,
        "episode_resets_during_check": n_reset,
        "pathological_values": PATHOLOGICAL,
        "max_abs_diff": worst,
        "max_abs_diff_per_obs_term": per_term,
        "tolerance": tol,
        "passed": passed,
        "all_passed": all(passed.values()),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "contract_check.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)
    return 0 if result["all_passed"] else 3


if __name__ == "__main__":
    sys.exit(main())
