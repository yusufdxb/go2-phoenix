"""Record policy-internal rollouts for the reliability-shield Phase 3 study.

Boots Isaac Sim once for ONE condition (nominal or a pinned held-out shift),
runs the trained policy, and logs per control step, for every parallel env:

  obs, policy latent activations (mid + penultimate ELU outputs), action,
  critic value, and the independent failure-oracle signals (base pitch/roll,
  base height, commanded and actual planar velocity).

Output is a single ``.npz`` per condition plus a JSON sidecar carrying the
checkpoint SHA-256, the resolved DR override, and library versions, so the
offline evaluator can fit/calibrate on nominal only and score the shifts
without ever touching Isaac. One condition per process keeps DR changes
robust (each condition gets a clean env build).

Run inside the Isaac Lab venv:
  ~/Sim/isaac-sim-venv/bin/python scripts/reliability_rollout.py \
      --checkpoint checkpoints/phoenix-flat/policy.pt \
      --env-config configs/env/flat.yaml \
      --condition nominal --out reliability_eval/raw/nominal_seed0.npz \
      --num-envs 64 --max-steps 400 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Record reliability-shield rollouts.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-config", type=Path, required=True)
    p.add_argument("--condition", type=str, required=True, help="condition label")
    p.add_argument("--dr-override", type=str, default="", help="JSON dict of DR overrides")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args(argv)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    args = parse_args(argv)
    print(f"[roll] condition={args.condition} dr={args.dr_override!r}", flush=True)

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    print("[roll] app launched", flush=True)
    try:
        return _run(args, app)
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


def _run(args, app) -> int:  # noqa: ANN001
    import importlib.metadata as md

    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from omegaconf import OmegaConf
    from rsl_rl.runners import OnPolicyRunner

    from phoenix.sim2real.export import checkpoint_has_obs_normalizer
    from phoenix.sim_env import build_env_cfg, load_layered_config
    from phoenix.training.agent_cfg import build_runner_cfg
    from phoenix.training.checkpoint import load_runner_checkpoint

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Config + DR override ----------------------------------------------
    cfg = load_layered_config(args.env_config)
    dr_override = json.loads(args.dr_override) if args.dr_override else {}
    for key, val in dr_override.items():
        OmegaConf.update(cfg.cfg, f"domain_randomization.{key}", val, force_add=True)
    env_cfg = build_env_cfg(cfg)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    task_name = cfg.to_container()["env"]["task_name"]
    print(f"[roll] task={task_name} num_envs={args.num_envs}", flush=True)

    env = gym.make(task_name, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)  # evaluate.py clips actions to [-1,1]

    # Whether to normalize observations is a PROPERTY OF THE CHECKPOINT, never a
    # constant. rsl_rl only serializes ``obs_normalizer.*`` buffers when the run
    # was trained with empirical normalization; if we ask for normalization on a
    # checkpoint that has none, rsl_rl builds a fresh EmpiricalNormalization whose
    # buffers are still mean=0/std=1, and its forward is
    # ``(x - 0) / (1 + eps)`` with eps=1e-2 -- a silent 1% shrink of every
    # observation. That is invisible in behaviour but shifts every recorded latent
    # by ~1% relative, which is exactly the gap that made deploy/flat_v4_latent.onnx
    # fail the deploy parity gate (worst latent diff 6.44 on a latent of scale 673)
    # while stand-v3, whose checkpoint DOES carry the buffers, passed.
    normalize_obs = checkpoint_has_obs_normalizer(args.checkpoint)
    print(f"[roll] checkpoint obs normalization: {normalize_obs}", flush=True)

    # ---- Runner + checkpoint (mirrors training/evaluate.py exactly) --------
    # Use the proven rsl_rl inference path so behavior matches evaluate.py;
    # we only ADD activation hooks on the actor. Reconstructing the MLP by
    # hand diverged from this path (the policy fell), so we do not do that.
    eval_yaml = {
        "run": {
            "name": "reliability_roll",
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
        "runner": {"num_steps_per_env": 24, "empirical_normalization": normalize_obs},
    }
    runner_cfg = build_runner_cfg(eval_yaml, task_name)
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, md.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=args.device)
    info = load_runner_checkpoint(
        runner, args.checkpoint, load_actor=True, load_critic=True, load_optimizer=False, load_iteration=False
    )
    if not info.get("actor_match", False):
        raise RuntimeError(f"actor weights did not round-trip: {info}")
    policy = runner.get_inference_policy(device=args.device)
    actor = runner.alg.actor
    critic = getattr(runner.alg, "critic", None)
    print(f"[roll] runner loaded; actor_match={info.get('actor_match')}", flush=True)

    # ---- Hook the actor's Linear layers; capture their INPUTS --------------
    # The input to hidden Linear k is exactly the post-activation output of
    # layer k-1, so the inputs to Linears 1.. are the hidden activations. This
    # is robust even when the activation module is shared/reused across layers
    # (hooking the activation module itself then fires once and overwrites).
    import torch.nn as nn

    linear_modules = [m for m in actor.modules() if isinstance(m, nn.Linear)]
    if len(linear_modules) < 2:
        raise RuntimeError(f"expected >=2 Linear layers in actor, found {len(linear_modules)}")
    captured: dict[int, torch.Tensor] = {}
    handles = []
    _no_hooks = bool(os.environ.get("ROLL_NO_HOOKS"))
    if not _no_hooks:
        for i, m in enumerate(linear_modules):
            handles.append(
                m.register_forward_pre_hook(lambda _mod, inp, i=i: captured.__setitem__(i, inp[0].clone()))
            )
    hidden_lin = list(range(1, len(linear_modules)))  # Linears whose input is a hidden activation
    n_hidden = len(hidden_lin)
    tap_idx = sorted({hidden_lin[(n_hidden - 1) // 2], hidden_lin[-1]})  # mid + penultimate
    print(f"[roll] {len(linear_modules)} Linear layers; tapping inputs of {tap_idx}", flush=True)

    def _np(t):
        if t is None:
            return None
        if hasattr(t, "detach"):  # torch tensor
            return t.detach().cpu().numpy()
        if hasattr(t, "numpy"):  # warp array
            return t.numpy()
        return np.asarray(t)

    def _roll_pitch(quat_wxyz):
        """Roll (x) and pitch (y) in radians from a batch of wxyz quaternions."""
        q = np.asarray(quat_wxyz, dtype=np.float64)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        return roll, pitch

    # ---- Rollout -----------------------------------------------------------
    # The policy/critic are rsl_rl MLPModels whose get_latent() indexes the
    # observation by group name (obs["policy"]) and normalizes internally, so
    # we must pass the raw TensorDict straight through (NOT a flattened tensor).
    def _unwrap(o):
        return o[0] if isinstance(o, tuple) else o

    def _policy_group(o):
        """The 'policy' observation tensor from a TensorDict / dict / tensor."""
        o = _unwrap(o)
        try:
            return o["policy"]
        except (KeyError, TypeError, IndexError):
            return o

    # Mirror evaluate.py: do NOT force a synchronized reset (that drops all
    # robots at once and the mediocre policy handles it worse than the warmed
    # post-construction state). Only reset if obs come back empty.
    obs = _unwrap(env.get_observations())
    if _policy_group(obs).shape[-1] == 0:
        print("[roll] empty obs; issuing one reset", flush=True)
        env.reset()
        obs = _unwrap(env.get_observations())
    pol = _policy_group(obs)
    print(f"[roll] policy-group obs shape={tuple(pol.shape)}", flush=True)
    if pol.shape[-1] == 0:
        raise RuntimeError("policy obs has 0 features")

    log = {
        k: []
        for k in ("obs", "latent", "action", "value", "grav", "base_h", "cmd_vxy", "act_vxy", "done", "time_out")
    }
    robot = env.unwrapped.scene["robot"]

    print("[roll] rollout started", flush=True)
    with torch.inference_mode():
        for step in range(args.max_steps):
            captured.clear()
            actions = policy(obs)
            latent = (
                torch.cat([captured[i] for i in tap_idx], dim=1)
                if captured
                else torch.zeros((actions.shape[0], 1), device=actions.device)
            )

            value = None
            if critic is not None and not os.environ.get("ROLL_NO_VALUE"):
                try:
                    value = critic(obs)  # MLPModel normalizes + selects groups internally
                except Exception:  # noqa: BLE001
                    value = None

            d = robot.data
            obs_pol = _np(_policy_group(obs))  # (N, 48)
            grav = _np(getattr(d, "projected_gravity_b", None))
            if grav is None:  # fallback: projected gravity lives at obs dims 6:9
                grav = obs_pol[:, 6:9]
            pos = _np(d.root_pos_w)  # (N, 3)
            lin_b = _np(d.root_lin_vel_b)  # (N, 3) body frame
            cmd = _np(env.unwrapped.command_manager.get_command("base_velocity"))  # (N, 3)

            obs2, _reward, dones, extras = env.step(actions)
            time_out = extras.get("time_outs") if isinstance(extras, dict) else None

            val_np = _np(value)
            if val_np is None:
                val_col = np.full(args.num_envs, np.nan)
            else:
                val_col = val_np[:, 0] if val_np.ndim == 2 else val_np.reshape(-1)
            to_np = _np(time_out)

            log["obs"].append(obs_pol)
            log["latent"].append(_np(latent))
            log["action"].append(_np(actions))
            log["value"].append(val_col)
            log["grav"].append(grav)
            log["base_h"].append(pos[:, 2])
            log["cmd_vxy"].append(cmd[:, :2])
            log["act_vxy"].append(lin_b[:, :2])
            log["done"].append(_np(dones).astype(bool).reshape(-1))
            log["time_out"].append(
                to_np.astype(bool).reshape(-1) if to_np is not None else np.zeros(args.num_envs, bool)
            )
            obs = _unwrap(obs2)

            if step % 100 == 0:
                print(f"[roll] step {step}/{args.max_steps}", flush=True)

    for h in handles:
        h.remove()

    arrays = {k: np.asarray(v) for k, v in log.items()}  # (T, N, ...)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    meta = {
        "condition": args.condition,
        "dr_override": dr_override,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "env_config": str(args.env_config),
        "task_name": task_name,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "tap_indices": tap_idx,
        "n_elu_layers": n_hidden,
        "latent_dim": int(arrays["latent"].shape[-1]),
        "obs_dim": int(arrays["obs"].shape[-1]),
        # Provenance for the deploy parity gate: a recording made under a
        # different normalization decision than the export is not comparable.
        "empirical_normalization": bool(normalize_obs),
        "versions": {
            "isaaclab": md.version("isaaclab") if _has(md, "isaaclab") else "?",
            "rsl_rl_lib": md.version("rsl-rl-lib"),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[roll] wrote {args.out} shape latent={arrays['latent'].shape} + meta", flush=True)
    return 0


def _has(md, name) -> bool:
    try:
        md.version(name)
        return True
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    sys.exit(main())
