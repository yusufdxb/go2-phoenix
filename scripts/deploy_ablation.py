"""Run the deployment-mismatch ablation grid in simulation. Machine-readable output.

Question this answers: can the 2026-04-21 Gate 7 hardware result (33% of
motor-steps hitting the per-step slew clip, against 0.33% in sim) be reproduced
from KNOWN deployment mismatches, without retraining anything?

Three mismatches are known to have been live in that run or in the deploy path:

1. ``base_lin_vel`` fed to the policy as zeros, although it is a trained
   observation term (IsaacLab velocity_env_cfg.py:135).
2. ``default_joint_pos`` hips set to 0.0 in five deploy configs while training
   used +0.1 left and -0.1 right.
3. The per-step rate limiter existing only at deploy. The April checkpoints
   predate the in-training limiter, so enforcing it at evaluation reproduces
   that mismatch on an unmodified checkpoint.

Each is a pure transformation of the observation or action path, so all eight
cells of the grid run against the SAME checkpoint. The decision layer and the
arithmetic live in ``phoenix.reliability.deploy_ablation`` and are unit-tested
without a simulator; this script is only the driver.

The reported slew figure is the DEPLOY-EQUIVALENT definition: build
``target = default_q + action_scale * action`` and ask whether the deploy
limiter would clip it against MEASURED joint position. It is NOT the legacy
raw-action-delta figure, and the two are not comparable. Historical numbers in
this repo used the legacy definition.

Usage:
    python scripts/deploy_ablation.py \
        --checkpoint checkpoints/phoenix-flat-v4/latest.pt \
        --env-config configs/env/flat_v4.yaml \
        --num-envs 64 --steps 600 --out reliability_eval/deploy_ablation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

logger = logging.getLogger("phoenix.deploy_ablation")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-config", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "reliability_eval/deploy_ablation")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    args = parse_args(argv)
    print(f"[ablation] args: {args}", flush=True)

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    print("[ablation] app launched", flush=True)
    try:
        return _run(args)
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


def _build(args, enforce_limiter: bool):
    """Build one env. The limiter is a training-time MDP term, so it needs a rebuild."""
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from omegaconf import OmegaConf

    from phoenix.sim_env import build_env_cfg, load_layered_config

    loaded = load_layered_config(args.env_config)
    # The in-training limiter defaults to ENABLED. Disabling it reproduces the
    # pre-limiter training condition the April checkpoints were produced under,
    # so enforcing it on one of those checkpoints reproduces the historical
    # deploy mismatch without retraining. Mutate the loaded OmegaConf tree in
    # place: build_env_cfg wants a PhoenixConfig (or a path) and calls
    # .to_container() itself, so a bare dict is not a valid substitute.
    OmegaConf.update(loaded.cfg, "action.rate_limit.enabled", enforce_limiter, force_add=True)
    container = loaded.to_container()
    env_cfg = build_env_cfg(loaded)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    env = gym.make(container["env"]["task_name"], cfg=env_cfg, render_mode=None)
    return RslRlVecEnvWrapper(env, clip_actions=1.0), container["env"]["task_name"]


def _run(args) -> int:  # noqa: ANN001
    from collections.abc import Mapping
    from importlib import metadata

    import numpy as np
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from phoenix.real_world.synthesize_failure import _to_numpy as to_numpy
    from phoenix.reliability.deploy_ablation import (
        apply_action_ablation,
        apply_observation_ablation,
        default_grid,
        deploy_slew_saturation,
    )
    from phoenix.sim2real.export import checkpoint_has_obs_normalizer
    from phoenix.sim2real.safety import MAX_DELTA_PER_STEP_RAD
    from phoenix.training.agent_cfg import build_runner_cfg
    from phoenix.training.checkpoint import load_runner_checkpoint

    def policy_obs(raw):
        """Unwrap to the policy observation tensor, iteratively.

        rsl_rl and the Isaac wrapper return a tensor, an (obs, extras) tuple,
        or a group mapping depending on version, and the mapping can nest.

        The mapping is a tensordict.TensorDict, which is a Mapping but NOT a
        dict subclass, so an isinstance(raw, dict) test skips it entirely. It
        then reaches _to_numpy, which sees .cpu(), calls .numpy(), and gets back
        a plain dict of arrays. Test against Mapping, not dict.
        """
        for _ in range(6):
            if isinstance(raw, tuple):
                raw = raw[0]
                continue
            if isinstance(raw, Mapping):
                keys = list(raw.keys())
                if not keys:
                    raise RuntimeError("empty observation mapping")
                raw = raw["policy"] if "policy" in keys else raw[keys[0]]
                continue
            break
        if isinstance(raw, tuple | Mapping):
            raise RuntimeError(f"could not resolve a policy observation, got {type(raw)}")
        return raw

    def as_float_array(value, what):
        """Unwrap and convert in one place, failing loudly on an unknown shape.

        Doing the unwrap and the conversion separately let a container slip
        through the unwrap and only fail later inside the arithmetic, where the
        message named .astype rather than the real problem.
        """
        resolved = policy_obs(value)
        arr = to_numpy(resolved)
        if not isinstance(arr, np.ndarray) or arr.dtype == object:
            raise RuntimeError(
                f"{what} did not resolve to a numeric array: "
                f"input {type(value)}, after unwrap {type(resolved)}, "
                f"after convert {type(arr)}"
            )
        return arr.astype(np.float64)

    use_norm = checkpoint_has_obs_normalizer(args.checkpoint)
    print(f"[ablation] empirical_normalization from checkpoint: {use_norm}", flush=True)

    results = []
    grid = default_grid()
    # Group by limiter setting so the env is rebuilt twice, not eight times.
    for enforce in sorted({s.enforce_deploy_limiter for s in grid}):
        env, task_name = _build(args, enforce)
        runner_yaml = {
            "run": {
                "name": "ablation",
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
            "runner": {"num_steps_per_env": 24, "empirical_normalization": use_norm},
        }
        cfg = build_runner_cfg(runner_yaml, task_name)
        try:
            from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

            cfg = handle_deprecated_rsl_rl_cfg(cfg, metadata.version("rsl-rl-lib"))
        except ImportError:
            pass
        runner = OnPolicyRunner(env, cfg.to_dict(), log_dir=None, device=args.device)
        info = load_runner_checkpoint(
            runner, args.checkpoint, load_actor=True, load_critic=True,
            load_optimizer=False, load_iteration=False,
        )
        if not info.get("actor_match", False):
            raise RuntimeError(f"actor weights did not round-trip: {info}")
        policy = runner.get_inference_policy(device=args.device)

        robot = env.unwrapped.scene["robot"]
        default_q = to_numpy(robot.data.default_joint_pos)[0].astype(np.float64)
        action_scale = float(
            getattr(env.unwrapped.action_manager.get_term("joint_pos").cfg, "scale", 0.25)
        )

        for spec in [s for s in grid if s.enforce_deploy_limiter == enforce]:
            obs = policy_obs(env.get_observations())
            falls = 0
            episodes = 0
            sat_num = 0.0
            sat_den = 0
            with torch.inference_mode():
                for _ in range(args.steps):
                    obs_np = as_float_array(obs, "observation")
                    ablated = apply_observation_ablation(obs_np, spec)
                    action = policy(torch.as_tensor(ablated, dtype=torch.float32,
                                                    device=args.device))
                    action_np = as_float_array(action, "action")
                    applied = apply_action_ablation(action_np, spec, action_scale)

                    measured_q = to_numpy(robot.data.joint_pos).astype(np.float64)
                    sat_num += deploy_slew_saturation(
                        applied, measured_q, default_q, action_scale, MAX_DELTA_PER_STEP_RAD
                    ) * measured_q.shape[0]
                    sat_den += measured_q.shape[0]

                    # env.step's arity varies across rsl_rl versions, so the
                    # observation goes through the same normalizer used at reset
                    # rather than being unpacked positionally.
                    stepped = env.step(
                        torch.as_tensor(applied, dtype=torch.float32, device=args.device)
                    )
                    obs = policy_obs(stepped)
                    dones = stepped[2] if isinstance(stepped, tuple) and len(stepped) > 2 else None
                    if dones is not None:
                        d = to_numpy(dones).astype(bool)
                        episodes += int(d.sum())
                        term = to_numpy(env.unwrapped.termination_manager.terminated).astype(bool)
                        falls += int((d & term).sum())

            cell = spec.as_dict()
            cell.update(
                {
                    "deploy_slew_saturation": sat_num / sat_den if sat_den else None,
                    "terminations": falls,
                    "episodes_ended": episodes,
                    "fall_rate_per_episode": (falls / episodes) if episodes else None,
                    "steps": args.steps,
                    "num_envs": args.num_envs,
                    "metric_definition": "deploy_equivalent_target_vs_measured_q",
                }
            )
            results.append(cell)
            print(
                f"[ablation] {spec.name:<24} slew={cell['deploy_slew_saturation']} "
                f"falls={falls}/{episodes}",
                flush=True,
            )
        env.close()

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": args.checkpoint.name,
        "env_config": args.env_config.name,
        "seed": args.seed,
        "max_delta_per_step_rad": MAX_DELTA_PER_STEP_RAD,
        "metric_definition": (
            "deploy-equivalent: target = default_q + action_scale * action, clipped "
            "against measured q. NOT the legacy raw-action-delta definition; the two "
            "are not comparable."
        ),
        "cells": results,
    }
    (args.out / "deploy_ablation.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[ablation] wrote {args.out / 'deploy_ablation.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
