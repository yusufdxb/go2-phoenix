"""Paired closed-loop intervention study: does the shield PREVENT falls?

Every result so far is about *warning*. The shield's warnings were scored
against what the unshielded policy did, and the fallback was never actually
engaged. This harness runs the experiment that tests the causal claim.

One process per arm. Within a process, each pre-registered scenario block is
replayed as its own episode: seed the RNG from the block, reset, run to the
horizon, and apply the block's motor-degradation disturbance at its registered onset tick
(never before ``MIN_ONSET_TICK``, so the disturbance always lands after the
shield has armed and the robot has stabilised).

Arms
----
``unshielded``  the learned policy alone, with the shield running as a passive
                monitor so its would-be decisions are still logged.
``shielded``    the frozen bundle: per-environment ``DeployShield`` blending the
                learned action toward the fallback.
``sham``        the same fallback, engaged with the shield's own realised
                switching frequency and timing but on a schedule permuted across
                blocks, so the switch time carries no information about the
                episode it lands in. This is the arm that separates "the
                monitor's timing matters" from "standing still sometimes helps".

The blend acts on the policy action rather than on joint targets, which is the
same operation the ROS node performs: on-robot,
``target = default_q + action_scale * action``, and the fallback target is
exactly ``default_q``, so blending targets toward ``default_q`` is identical to
blending the action toward zero.

Usage::

    # 1. freeze the bundle + protocol (once, before any arm runs)
    python scripts/reliability_closed_loop.py --freeze --out-dir reliability_eval/closed_loop
    # 2. run the arms
    python scripts/reliability_closed_loop.py --arm unshielded --out-dir ...
    python scripts/reliability_closed_loop.py --arm shielded   --out-dir ...
    python scripts/reliability_closed_loop.py --arm sham       --out-dir ...   # needs shielded first
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["unshielded", "shielded", "sham", "oracle"], default=None)
    p.add_argument(
        "--oracle-delay-ticks",
        type=int,
        default=0,
        help="oracle arm only: switch this many ticks AFTER the true disturbance onset "
        "(0 = a perfect detector). Nominal blocks never switch (a perfect detector has no "
        "false positive). This arm is a NON-confirmatory diagnostic: it upper-bounds what the "
        "static fallback can do with ideal timing, decoupled from all calibration error.",
    )
    p.add_argument("--freeze", action="store_true", help="Write the bundle manifest + protocol, then exit")
    p.add_argument("--out-dir", default="reliability_eval/closed_loop")
    p.add_argument("--artifact", default="deploy/shield_stand_v3.npz")
    p.add_argument("--onnx", default="deploy/stand_v3_latent.onnx")
    p.add_argument("--checkpoint", default="checkpoints/phoenix-stand-v3-h25-final/latest.pt")
    p.add_argument("--env-config", default="configs/env/stand_v3_h25.yaml")
    p.add_argument("--envs", type=int, default=16, help="Environments per block")
    p.add_argument("--n-disturbed", type=int, default=32)
    p.add_argument("--n-nominal", type=int, default=16)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pilot", type=int, default=0, help="Run only the first N blocks (pilot; excluded from results)")
    return p.parse_args()


# --- controller snapshot -----------------------------------------------------


def controller_snapshot(env_config: str, artifact_meta: dict) -> dict:
    """The resolved controller values that determine closed-loop behaviour."""
    from phoenix.sim2real.safety import MAX_DELTA_PER_STEP_RAD
    from phoenix.sim_env import load_layered_config

    cfg = load_layered_config(env_config).to_container()
    control = cfg.get("control", {}) or {}
    return {
        "control_dt_s": float((artifact_meta.get("provenance") or {}).get("control_dt_s", 0.02)),
        "action_scale": float(control.get("action_scale", 0.25)),
        "max_delta_per_step_rad": float(MAX_DELTA_PER_STEP_RAD),
        "default_joint_pos": control.get("default_joint_pos", cfg.get("default_joint_pos", {})),
        "joint_order": cfg.get("joint_order", []),
    }


def do_freeze(args) -> int:
    from phoenix.reliability.bundle import build_manifest
    from phoenix.reliability.deploy import load_artifact
    from phoenix.reliability.study import generate_blocks, write_protocol

    _, op, meta = load_artifact(args.artifact)
    controller = controller_snapshot(args.env_config, meta)
    manifest = build_manifest(
        files={
            "shield_artifact": args.artifact,
            "policy_onnx": args.onnx,
            "policy_checkpoint": args.checkpoint,
        },
        controller=controller,
        versions=(meta.get("provenance") or {}).get("versions", {}),
        note="paired closed-loop intervention study",
    )
    out = Path(args.out_dir)
    manifest.write(out / "bundle.json")

    blocks = generate_blocks(
        n_disturbed=args.n_disturbed, n_nominal=args.n_nominal, horizon_ticks=args.horizon
    )
    params = {
        "envs_per_block": args.envs,
        "horizon_ticks": args.horizon,
        "artifact_trip": op.trip_threshold,
        "artifact_K": op.trip_persistence,
        "artifact_arming_ticks": op.arming_ticks,
        "primary_estimand": "paired block-level fall-rate difference, unshielded minus shielded",
        "secondary_estimand": "paired block-level fall-rate difference, sham minus shielded",
        "analysis_unit": "scenario block",
    }
    digest = write_protocol(out / "protocol.json", blocks, bundle_id=manifest.bundle_id, params=params)
    print(f"[freeze] bundle_id={manifest.bundle_id} dirty={manifest.code_dirty}")
    print(f"[freeze] protocol_hash={digest[:16]} blocks={len(blocks)}")
    print(f"[freeze] wrote {out}/bundle.json and {out}/protocol.json")
    if manifest.code_dirty:
        print("[freeze] WARNING: working tree is dirty; commit before the definitive run.")
    return 0


# --- the run -----------------------------------------------------------------


def main() -> int:
    args = parse_args()
    if args.freeze:
        return do_freeze(args)
    if args.arm is None:
        raise SystemExit("give --arm or --freeze")

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    print("[cl] app launched", flush=True)
    try:
        return run_arm(args)
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


def run_arm(args) -> int:  # noqa: C901 - one long, linear experimental loop
    import importlib.metadata as md

    import gymnasium as gym
    import torch
    import torch.nn as nn
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    from phoenix.reliability.bundle import BundleManifest, verify_bundle
    from phoenix.reliability.deploy import load_artifact
    from phoenix.reliability.study import VectorShield, read_protocol, sham_schedule
    from phoenix.sim_env import build_env_cfg, load_layered_config
    from phoenix.training.agent_cfg import build_runner_cfg
    from phoenix.training.checkpoint import load_runner_checkpoint

    out_dir = Path(args.out_dir)
    blocks, protocol = read_protocol(out_dir / "protocol.json")
    manifest = BundleManifest.read(out_dir / "bundle.json")
    _, op, meta = load_artifact(args.artifact)

    # Gate 1: the controller must be the one the protocol was frozen against.
    controller = controller_snapshot(args.env_config, meta)
    verify_bundle(
        manifest,
        files={
            "shield_artifact": args.artifact,
            "policy_onnx": args.onnx,
            "policy_checkpoint": args.checkpoint,
        },
        controller=controller,
        artifact_control_dt_s=controller["control_dt_s"],
    )
    if manifest.bundle_id != protocol["bundle_id"]:
        raise SystemExit(
            f"FAIL CLOSED: bundle {manifest.bundle_id} != the one the protocol was frozen "
            f"against ({protocol['bundle_id']})"
        )
    print(f"[cl] bundle verified: {manifest.bundle_id}", flush=True)

    if args.pilot:
        blocks = blocks[: args.pilot]
        print(f"[cl] PILOT: {len(blocks)} blocks — results are NOT part of the study", flush=True)

    # Gate 2: the sham arm needs the shielded arm's realised switching behaviour.
    sham_sched = None
    if args.arm == "sham":
        shielded_path = out_dir / "arm_shielded.npz"
        if not shielded_path.is_file():
            raise SystemExit("FAIL CLOSED: run the shielded arm before the sham arm")
        prev = np.load(shielded_path)
        realised = {
            int(b): [None if t < 0 else int(t) for t in prev["switch_tick"][i]]
            for i, b in enumerate(prev["block_id"])
        }
        sham_sched = sham_schedule(realised, seed=int(protocol["params"].get("envs_per_block", 16)))
        print(f"[cl] sham schedule permuted from {len(realised)} shielded blocks", flush=True)

    # ---- env + policy (mirrors reliability_rollout.py exactly) --------------
    cfg = load_layered_config(args.env_config)
    env_cfg = build_env_cfg(cfg)
    env_cfg.scene.num_envs = args.envs
    task_name = cfg.to_container()["env"]["task_name"]
    env = gym.make(task_name, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    eval_yaml = {
        "run": {
            "name": "closed_loop", "output_dir": "/tmp", "log_interval": 1, "save_interval": 1,
            "max_iterations": 1, "seed": 0, "device": args.device,
        },
        "algorithm": {
            "class_name": "PPO", "value_loss_coef": 1.0, "use_clipped_value_loss": True,
            "clip_param": 0.2, "entropy_coef": 0.005, "num_learning_epochs": 5,
            "num_mini_batches": 4, "learning_rate": 1.0e-3, "schedule": "adaptive",
            "gamma": 0.99, "lam": 0.95, "desired_kl": 0.01, "max_grad_norm": 1.0,
        },
        "policy": {
            "class_name": "ActorCritic", "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128], "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "runner": {"num_steps_per_env": 24, "empirical_normalization": True},
    }
    runner_cfg = handle_deprecated_rsl_rl_cfg(build_runner_cfg(eval_yaml, task_name), md.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=args.device)
    info = load_runner_checkpoint(
        runner, Path(args.checkpoint), load_actor=True, load_critic=True,
        load_optimizer=False, load_iteration=False,
    )
    if not info.get("actor_match", False):
        raise RuntimeError(f"actor weights did not round-trip: {info}")
    policy = runner.get_inference_policy(device=args.device)
    actor = runner.alg.actor

    linear_modules = [m for m in actor.modules() if isinstance(m, nn.Linear)]
    captured: dict[int, torch.Tensor] = {}
    for i, m in enumerate(linear_modules):
        m.register_forward_pre_hook(lambda _mod, inp, i=i: captured.__setitem__(i, inp[0].clone()))
    hidden_lin = list(range(1, len(linear_modules)))
    n_hidden = len(hidden_lin)
    tap_idx = sorted({hidden_lin[(n_hidden - 1) // 2], hidden_lin[-1]})

    def _unwrap(o):
        return o[0] if isinstance(o, tuple) else o

    def _policy_group(o):
        o = _unwrap(o)
        try:
            return o["policy"]
        except (KeyError, TypeError, IndexError):
            return o

    robot = env.unwrapped.scene["robot"]

    def _tensor(x):
        """Actuator/robot data is a torch tensor for some fields, a warp array for others."""
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x.numpy(), device=env.unwrapped.device)

    # Baseline actuator gains, captured once. The `scale_motor_strength` term is
    # a STARTUP event, so these are constant across resets and can be restored
    # exactly at the start of every block.
    base_gains = {
        name: (_tensor(a.stiffness).clone(), _tensor(a.damping).clone())
        for name, a in robot.actuators.items()
    }

    def set_motor_scale(scale: float) -> bool:
        """Scale actuator stiffness and damping mid-episode.

        Writing the actuator gains directly is the only disturbance that can
        actually be retargeted mid-episode: Isaac Lab's physics-material event
        term caches a fixed bucket pool at startup, so re-invoking it resamples
        from the ORIGINAL friction range and silently ignores a new one. Motor
        strength is also the stronger held-out shift, having never been
        randomised during training.
        """
        try:
            for name, actuator in robot.actuators.items():
                stiffness, damping = base_gains[name]
                actuator.stiffness = stiffness * scale
                actuator.damping = damping * scale
            return True
        except Exception as exc:  # noqa: BLE001 - reported, never silently ignored
            print(f"[cl] MOTOR INJECTION FAILED: {type(exc).__name__}: {exc}", flush=True)
            return False

    shield = VectorShield(args.artifact, args.envs)
    handoff = int(meta["timings"]["handoff_ticks"])

    # ---- per-block loop -----------------------------------------------------
    rows = {k: [] for k in ("block_id", "fell", "switch_tick", "engaged", "fall_tick")}
    with torch.inference_mode():
        for bi, block in enumerate(blocks):
            torch.manual_seed(block.seed)
            np.random.seed(block.seed)
            set_motor_scale(1.0)  # restore healthy gains before every block
            env.reset()
            shield.reset()
            obs = _unwrap(env.get_observations())

            fell = np.zeros(args.envs, bool)
            fall_tick = np.full(args.envs, -1, np.int32)
            switch_tick = np.full(args.envs, -1, np.int32)
            engaged = np.zeros(args.envs, bool)
            sham_switch = (
                np.array([-1 if t is None else t for t in sham_sched[block.block_id]], np.int32)
                if sham_sched is not None
                else None
            )
            injected = False

            for tick in range(block.horizon_ticks):
                if block.disturbed and not injected and tick == block.onset_tick:
                    injected = set_motor_scale(block.motor_scale)

                captured.clear()
                actions = policy(obs)
                latent = torch.cat([captured[i] for i in tap_idx], dim=1)
                blend_np, _score, _armed = shield.step(latent.detach().cpu().numpy())

                if args.arm == "unshielded":
                    applied_blend = np.zeros(args.envs)  # passive monitor only
                elif args.arm == "shielded":
                    applied_blend = blend_np
                elif args.arm == "oracle":
                    # Perfect detector: on disturbed blocks, ramp in exactly at the
                    # true onset (+ optional delay) and hold; nominal blocks never
                    # switch. Upper-bounds the static fallback's control authority.
                    if block.disturbed:
                        since = tick - (block.onset_tick + args.oracle_delay_ticks)
                        b = min(1.0, since / handoff) if since >= 0 else 0.0
                    else:
                        b = 0.0
                    applied_blend = np.full(args.envs, b)
                else:  # sham: ramp on the permuted schedule, same handoff length
                    since = tick - sham_switch
                    applied_blend = np.where(
                        (sham_switch >= 0) & (since >= 0), np.minimum(1.0, since / handoff), 0.0
                    )

                newly = (applied_blend > 0) & (switch_tick < 0)
                switch_tick[newly] = tick
                engaged |= applied_blend > 0

                # Blending the action toward zero == blending the joint target
                # toward the default stand pose, which is what the ROS node does.
                blended = actions * torch.as_tensor(
                    1.0 - applied_blend, device=actions.device, dtype=actions.dtype
                ).unsqueeze(1)

                obs2, _r, dones, extras = env.step(blended)
                time_out = extras.get("time_outs") if isinstance(extras, dict) else None
                d = np.asarray(dones.detach().cpu()).astype(bool).reshape(-1)
                t_out = (
                    np.asarray(time_out.detach().cpu()).astype(bool).reshape(-1)
                    if time_out is not None
                    else np.zeros(args.envs, bool)
                )
                new_falls = d & (~t_out) & (~fell)
                fall_tick[new_falls] = tick
                fell |= new_falls
                obs = _unwrap(obs2)

            rows["block_id"].append(block.block_id)
            rows["fell"].append(fell.copy())
            rows["switch_tick"].append(switch_tick.copy())
            rows["engaged"].append(engaged.copy())
            rows["fall_tick"].append(fall_tick.copy())
            print(
                f"[cl] block {bi + 1}/{len(blocks)} id={block.block_id} "
                f"{'disturbed' if block.disturbed else 'nominal  '} "
                f"fell={fell.sum()}/{args.envs} engaged={engaged.sum()}/{args.envs}",
                flush=True,
            )

    suffix = "_pilot" if args.pilot else ""
    if args.arm == "oracle" and args.oracle_delay_ticks:
        suffix += f"_d{args.oracle_delay_ticks}"
    out = out_dir / f"arm_{args.arm}{suffix}.npz"
    np.savez(
        out,
        block_id=np.asarray(rows["block_id"]),
        fell=np.asarray(rows["fell"]),
        switch_tick=np.asarray(rows["switch_tick"]),
        engaged=np.asarray(rows["engaged"]),
        fall_tick=np.asarray(rows["fall_tick"]),
    )
    (out_dir / f"arm_{args.arm}{suffix}.meta.json").write_text(
        json.dumps(
            {
                "arm": args.arm,
                "bundle_id": manifest.bundle_id,
                "protocol_hash": protocol.get("protocol_hash"),
                "blocks": len(blocks),
                "envs_per_block": args.envs,
                "pilot": bool(args.pilot),
                "diagnostic": args.arm == "oracle",
                "oracle_delay_ticks": args.oracle_delay_ticks if args.arm == "oracle" else None,
            },
            indent=2,
        )
    )
    print(f"[cl] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
