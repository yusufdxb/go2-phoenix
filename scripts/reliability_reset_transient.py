"""Measure how long the OOD score stays elevated after a hard reset.

The closed-loop study forces a synchronised env reset at the start of every
block; the deployed robot likewise starts from a single standing pose. The
Phase 4 monitor was calibrated on FREE-RUNNING rollouts whose resets happened
asynchronously mid-stream, so its 15-tick arming window reflects that
distribution, not a hard reset. This script boots the healthy policy, forces a
reset, and records the deploy-monitor score per tick so the arming window can be
set to actually cover the hard-reset transient.
"""

from __future__ import annotations

import argparse
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="deploy/shield_stand_v3.npz")
    ap.add_argument("--checkpoint", default="checkpoints/phoenix-stand-v3-h25-final/latest.pt")
    ap.add_argument("--env-config", default="configs/env/stand_v3_h25.yaml")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--ticks", type=int, default=100)
    ap.add_argument("--resets", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        import importlib.metadata as md

        import gymnasium as gym
        import torch
        import torch.nn as nn
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from rsl_rl.runners import OnPolicyRunner

        from phoenix.reliability.deploy import load_artifact
        from phoenix.sim_env import build_env_cfg, load_layered_config
        from phoenix.training.agent_cfg import build_runner_cfg
        from phoenix.training.checkpoint import load_runner_checkpoint

        monitor, op, meta = load_artifact(args.artifact)
        cfg = load_layered_config(args.env_config)
        env_cfg = build_env_cfg(cfg)
        env_cfg.scene.num_envs = args.envs
        task = cfg.to_container()["env"]["task_name"]
        env = RslRlVecEnvWrapper(gym.make(task, cfg=env_cfg, render_mode=None), clip_actions=1.0)

        eval_yaml = {
            "run": {"name": "rt", "output_dir": "/tmp", "log_interval": 1, "save_interval": 1,
                    "max_iterations": 1, "seed": 0, "device": args.device},
            "algorithm": {"class_name": "PPO", "value_loss_coef": 1.0, "use_clipped_value_loss": True,
                          "clip_param": 0.2, "entropy_coef": 0.005, "num_learning_epochs": 5,
                          "num_mini_batches": 4, "learning_rate": 1e-3, "schedule": "adaptive",
                          "gamma": 0.99, "lam": 0.95, "desired_kl": 0.01, "max_grad_norm": 1.0},
            "policy": {"class_name": "ActorCritic", "init_noise_std": 1.0,
                       "actor_hidden_dims": [512, 256, 128], "critic_hidden_dims": [512, 256, 128],
                       "activation": "elu"},
            "runner": {"num_steps_per_env": 24, "empirical_normalization": True},
        }
        runner_cfg = handle_deprecated_rsl_rl_cfg(build_runner_cfg(eval_yaml, task), md.version("rsl-rl-lib"))
        runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=args.device)
        load_runner_checkpoint(runner, args.checkpoint, load_actor=True, load_critic=False,
                               load_optimizer=False, load_iteration=False)
        policy = runner.get_inference_policy(device=args.device)
        actor = runner.alg.actor
        linears = [m for m in actor.modules() if isinstance(m, nn.Linear)]
        captured: dict[int, torch.Tensor] = {}
        for i, m in enumerate(linears):
            m.register_forward_pre_hook(lambda _m, inp, i=i: captured.__setitem__(i, inp[0].clone()))
        hidden = list(range(1, len(linears)))
        tap = sorted({hidden[(len(hidden) - 1) // 2], hidden[-1]})

        def _pol(o):
            o = o[0] if isinstance(o, tuple) else o
            try:
                return o["policy"]
            except (KeyError, TypeError, IndexError):
                return o

        trip = op.trip_threshold
        per_reset = []
        with torch.inference_mode():
            for _ in range(args.resets):
                env.reset()
                obs = env.get_observations()
                obs = obs[0] if isinstance(obs, tuple) else obs
                frac_over = []
                for _t in range(args.ticks):
                    captured.clear()
                    actions = policy(obs)
                    latent = torch.cat([captured[i] for i in tap], dim=1).detach().cpu().numpy()
                    scores = monitor.score(latent)
                    frac_over.append(float(np.mean(scores > trip)))
                    obs2, _r, _d, _e = env.step(actions)
                    obs = obs2[0] if isinstance(obs2, tuple) else obs2
                per_reset.append(frac_over)

        frac = np.array(per_reset).mean(axis=0)  # (ticks,)
        # First tick after which the over-threshold fraction stays below 5%.
        settle = args.ticks
        for t in range(args.ticks):
            if np.all(frac[t:] < 0.05):
                settle = t
                break
        print(f"[rt] trip={trip:.0f} arming(now)={op.arming_ticks}")
        print("[rt] fraction of envs over trip, per tick after hard reset:")
        for t in range(0, min(args.ticks, 60), 3):
            bar = "#" * int(frac[t] * 40)
            print(f"  t={t:>3} {frac[t]:.3f} {bar}")
        print(f"[rt] score settles below 5% over-rate at tick {settle} "
              f"(current arming {op.arming_ticks} is {'ENOUGH' if settle <= op.arming_ticks else 'TOO SHORT'})")
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
