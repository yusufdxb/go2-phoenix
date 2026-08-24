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
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_SOURCE_PATHS = (
    "scripts/reliability_closed_loop.py",
    "scripts/reliability_oracle_screen.py",
    "scripts/reliability_replication.py",
    "src/phoenix/reliability/bundle.py",
    "src/phoenix/reliability/deploy.py",
    "src/phoenix/reliability/oracle_screen.py",
    "src/phoenix/reliability/replication.py",
    "src/phoenix/reliability/study.py",
    "src/phoenix/sim_env/config_loader.py",
    "src/phoenix/sim_env/go2_env_cfg.py",
    "src/phoenix/training/checkpoint.py",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_snapshot() -> dict:
    files = {}
    for relative in EXPERIMENT_SOURCE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"experimental source file is missing: {path}")
        files[relative] = file_sha256(path)
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"sha256": digest, "files": files}


def resolved_env_config_hash(path: str | Path) -> str:
    from phoenix.reliability.bundle import value_sha256
    from phoenix.sim_env import load_layered_config

    return value_sha256(load_layered_config(path).to_container())


def _arm_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    suffix = "_preflight" if args.preflight_subset else ("_pilot" if args.pilot else "")
    if args.arm == "oracle" and args.oracle_delay_ticks:
        suffix += f"_d{args.oracle_delay_ticks}"
    out_dir = Path(args.out_dir)
    return (
        out_dir / f"arm_{args.arm}{suffix}.npz",
        out_dir / f"arm_{args.arm}{suffix}.meta.json",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--arm",
        choices=["unshielded", "shielded", "sham", "sham_stratified", "oracle"],
        default=None,
    )
    p.add_argument(
        "--oracle-delay-ticks",
        type=int,
        default=0,
        help="oracle arm only: switch this many ticks AFTER the true disturbance onset "
        "(0 = a perfect detector). Nominal blocks never switch (a perfect detector has no "
        "false positive). This is a privileged onset-timing diagnostic, not an "
        "optimal-timing upper bound.",
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
    p.add_argument(
        "--disturbance",
        choices=["motor", "obs", "command"],
        default="motor",
        help="motor = actuator degradation (static fallback cannot survive it); "
        "obs = perceptual corruption of the policy's observations; "
        "command = an out-of-distribution velocity command (policy trained at cmd=0 "
        "tries to move and may fall, fallback ignores it and stands -- the canonical "
        "regime a stand-fallback Simplex shield exists for).",
    )
    p.add_argument("--obs-noise-lo", type=float, default=1.0)
    p.add_argument("--obs-noise-hi", type=float, default=3.0)
    p.add_argument("--command-speed-lo", type=float, default=0.6)
    p.add_argument("--command-speed-hi", type=float, default=1.4)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--protocol-seed", type=int, default=None)
    p.add_argument("--process-seed", type=int, default=None)
    p.add_argument("--replicate-id", default=None)
    p.add_argument("--cell-id", default=None)
    p.add_argument("--policy-name", choices=["stand", "walk"], default=None)
    p.add_argument("--motor-scale-lo", type=float, default=0.30)
    p.add_argument("--motor-scale-hi", type=float, default=0.55)
    p.add_argument(
        "--reset-settle-ticks",
        type=int,
        default=0,
        help="Step the environment this many ticks with zero actions after reset before "
        "capturing initial_obs. Guards against stale root-state buffers leaking the previous "
        "block terminal state across arms.",
    )
    p.add_argument("--onset-lo", type=int, default=100)
    p.add_argument("--onset-hi", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pilot", type=int, default=0, help="Run only the first N blocks (pilot; excluded from results)")
    p.add_argument(
        "--preflight-subset",
        action="store_true",
        help="Run two disturbed and two nominal blocks as a contract-only preflight",
    )
    return p.parse_args()


# --- controller snapshot -----------------------------------------------------


def bundle_files(args: argparse.Namespace) -> dict[str, str]:
    files = {
        "shield_artifact": args.artifact,
        "policy_onnx": args.onnx,
        "policy_checkpoint": args.checkpoint,
        "env_config": args.env_config,
    }
    external_onnx_data = Path(f"{args.onnx}.data")
    if external_onnx_data.is_file():
        files["policy_onnx_data"] = str(external_onnx_data)
    return files


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
    from phoenix.reliability.study import REPLICATION_ARMS, generate_blocks, write_protocol

    required = {
        "--protocol-seed": args.protocol_seed,
        "--process-seed": args.process_seed,
        "--replicate-id": args.replicate_id,
        "--cell-id": args.cell_id,
        "--policy-name": args.policy_name,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"FAIL CLOSED: freeze requires {', '.join(missing)}")
    if args.pilot or args.preflight_subset:
        raise SystemExit("FAIL CLOSED: do not combine --freeze with a pilot option")
    if args.horizon != 500:
        raise SystemExit("FAIL CLOSED: replication horizon must be exactly 500 ticks")
    if args.envs != 16 or args.n_disturbed != 32 or args.n_nominal != 16:
        raise SystemExit(
            "FAIL CLOSED: replication requires 16 envs, 32 disturbed blocks, "
            "and 16 nominal blocks"
        )

    _, op, meta = load_artifact(args.artifact)
    controller = controller_snapshot(args.env_config, meta)
    manifest = build_manifest(
        files=bundle_files(args),
        controller=controller,
        versions=(meta.get("provenance") or {}).get("versions", {}),
        note="paired closed-loop intervention study",
    )
    out = Path(args.out_dir)
    for path in (out / "bundle.json", out / "protocol.json"):
        if path.exists():
            raise SystemExit(f"FAIL CLOSED: refusing to overwrite frozen artifact {path}")
    manifest.write(out / "bundle.json")

    blocks = generate_blocks(
        n_disturbed=args.n_disturbed,
        n_nominal=args.n_nominal,
        disturbance=args.disturbance,
        motor_scale_range=(args.motor_scale_lo, args.motor_scale_hi),
        obs_noise_range=(args.obs_noise_lo, args.obs_noise_hi),
        command_speed_range=(args.command_speed_lo, args.command_speed_hi),
        onset_range=(args.onset_lo, args.onset_hi),
        horizon_ticks=args.horizon,
        seed=args.protocol_seed,
    )
    snapshot = source_snapshot()
    params = {
        "study_id": "phoenix_causal_viability_replication_v1",
        "replicate_id": args.replicate_id,
        "cell_id": args.cell_id,
        "policy_name": args.policy_name,
        "protocol_seed": args.protocol_seed,
        "process_seed": args.process_seed,
        "envs_per_block": args.envs,
        "n_disturbed": args.n_disturbed,
        "n_nominal": args.n_nominal,
        "horizon_ticks": args.horizon,
        "disturbance_kind": args.disturbance,
        "motor_scale_range": [args.motor_scale_lo, args.motor_scale_hi],
        "obs_noise_range": [args.obs_noise_lo, args.obs_noise_hi],
        "command_speed_range": [args.command_speed_lo, args.command_speed_hi],
        "onset_range": [args.onset_lo, args.onset_hi],
        "artifact_trip": op.trip_threshold,
        "artifact_K": op.trip_persistence,
        "artifact_arming_ticks": op.arming_ticks,
        "oracle_handoff_ticks": int(meta["timings"]["handoff_ticks"]),
        "reset_settle_ticks": int(args.reset_settle_ticks),
        "primary_estimand": (
            "paired block-level post-onset fall-rate difference among jointly "
            "onset-eligible environment pairs, unshielded minus oracle"
        ),
        "pre_onset_negative_control": (
            "paired block-level pre-onset fall-rate difference, unshielded minus oracle"
        ),
        "eligibility_rule": (
            "exclude an environment pair from the post-onset estimand if either arm "
            "falls before the registered onset"
        ),
        "analysis_unit": "scenario block",
        "source_snapshot_sha256": snapshot["sha256"],
        "source_files": snapshot["files"],
        "resolved_env_config_sha256": resolved_env_config_hash(args.env_config),
    }
    digest = write_protocol(
        out / "protocol.json",
        blocks,
        bundle_id=manifest.bundle_id,
        params=params,
        arms=REPLICATION_ARMS,
    )
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
    if args.process_seed is None:
        raise SystemExit("FAIL CLOSED: arm runs require --process-seed")
    out_path, meta_path = _arm_output_paths(args)
    for path in (out_path, meta_path):
        if path.exists():
            raise SystemExit(f"FAIL CLOSED: refusing to overwrite arm artifact {path}")

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

    from phoenix.reliability.bundle import BundleManifest, git_state, value_sha256, verify_bundle
    from phoenix.reliability.deploy import load_artifact
    from phoenix.reliability.study import VectorShield, read_protocol, sham_schedule
    from phoenix.sim2real.export import checkpoint_has_obs_normalizer
    from phoenix.sim_env import build_env_cfg, load_layered_config
    from phoenix.training.agent_cfg import build_runner_cfg
    from phoenix.training.checkpoint import load_runner_checkpoint

    out_dir = Path(args.out_dir)
    blocks, protocol = read_protocol(out_dir / "protocol.json")
    manifest = BundleManifest.read(out_dir / "bundle.json")
    _, op, meta = load_artifact(args.artifact)
    disturbance_kind = protocol["params"].get("disturbance_kind", "motor")
    print(f"[cl] disturbance = {disturbance_kind}", flush=True)

    params = protocol["params"]
    expected_cli = {
        "--protocol-seed": (args.protocol_seed, params.get("protocol_seed")),
        "--process-seed": (args.process_seed, params.get("process_seed")),
        "--replicate-id": (args.replicate_id, params.get("replicate_id")),
        "--cell-id": (args.cell_id, params.get("cell_id")),
        "--policy-name": (args.policy_name, params.get("policy_name")),
    }
    mismatched = [
        f"{flag}={actual!r}, protocol={expected!r}"
        for flag, (actual, expected) in expected_cli.items()
        if actual != expected
    ]
    if mismatched:
        raise SystemExit("FAIL CLOSED: CLI/protocol mismatch: " + "; ".join(mismatched))
    if args.arm not in protocol.get("arms", []):
        raise SystemExit(f"FAIL CLOSED: arm {args.arm!r} was not frozen in the protocol")
    if args.disturbance != disturbance_kind:
        raise SystemExit(
            f"FAIL CLOSED: --disturbance {args.disturbance!r} != protocol {disturbance_kind!r}"
        )
    if args.envs != int(params["envs_per_block"]):
        raise SystemExit("FAIL CLOSED: --envs does not match the frozen protocol")
    if args.horizon != int(params["horizon_ticks"]):
        raise SystemExit("FAIL CLOSED: --horizon does not match the frozen protocol")
    if any(block.horizon_ticks != args.horizon for block in blocks):
        raise SystemExit("FAIL CLOSED: one or more block horizons do not match the protocol")

    snapshot = source_snapshot()
    if snapshot["sha256"] != params.get("source_snapshot_sha256"):
        raise SystemExit(
            "FAIL CLOSED: experimental source snapshot changed after protocol freeze"
        )
    env_contract_hash = resolved_env_config_hash(args.env_config)
    if env_contract_hash != params.get("resolved_env_config_sha256"):
        raise SystemExit("FAIL CLOSED: resolved environment config changed after freeze")
    current_commit, current_dirty = git_state(REPO_ROOT)
    if current_commit != manifest.code_commit:
        raise SystemExit(
            f"FAIL CLOSED: current commit {current_commit} != bundle commit {manifest.code_commit}"
        )

    # Gate 1: the controller must be the one the protocol was frozen against.
    controller = controller_snapshot(args.env_config, meta)
    verify_bundle(
        manifest,
        files=bundle_files(args),
        controller=controller,
        artifact_control_dt_s=controller["control_dt_s"],
    )
    if manifest.bundle_id != protocol["bundle_id"]:
        raise SystemExit(
            f"FAIL CLOSED: bundle {manifest.bundle_id} != the one the protocol was frozen "
            f"against ({protocol['bundle_id']})"
        )
    print(f"[cl] bundle verified: {manifest.bundle_id}", flush=True)

    if args.pilot and args.preflight_subset:
        raise SystemExit("FAIL CLOSED: choose --pilot or --preflight-subset, not both")
    if args.preflight_subset:
        # A-B-N-N-A: the repeated first scenario after intervening blocks is a
        # direct reset/carryover check and is never accepted as study evidence.
        blocks = blocks[:2] + blocks[-2:] + blocks[:1]
        print(
            f"[cl] PREFLIGHT: {len(blocks)} blocks, results are NOT part of the study",
            flush=True,
        )
    elif args.pilot:
        blocks = blocks[: args.pilot]
        print(f"[cl] PILOT: {len(blocks)} blocks — results are NOT part of the study", flush=True)

    # Gate 2: the sham arm needs the shielded arm's realised switching behaviour.
    sham_sched = None
    if args.arm in ("sham", "sham_stratified"):
        shielded_path = out_dir / "arm_shielded.npz"
        if not shielded_path.is_file():
            raise SystemExit("FAIL CLOSED: run the shielded arm before the sham arm")
        prev = np.load(shielded_path)
        realised = {
            int(b): [None if t < 0 else int(t) for t in prev["switch_tick"][i]]
            for i, b in enumerate(prev["block_id"])
        }
        strata = (
            {block.block_id: block.disturbed for block in blocks}
            if args.arm == "sham_stratified"
            else None
        )
        sham_sched = sham_schedule(
            realised,
            seed=int(protocol["params"].get("envs_per_block", 16)),
            strata=strata,
        )
        qualifier = " within disturbance strata" if strata is not None else ""
        print(
            f"[cl] sham schedule permuted{qualifier} from {len(realised)} shielded blocks",
            flush=True,
        )

    # ---- env + policy (mirrors reliability_rollout.py exactly) --------------
    cfg = load_layered_config(args.env_config)
    env_cfg = build_env_cfg(cfg)
    env_cfg.scene.num_envs = args.envs
    env_cfg.seed = int(args.process_seed)
    task_name = cfg.to_container()["env"]["task_name"]
    torch.manual_seed(int(args.process_seed))
    np.random.seed(int(args.process_seed))
    env = gym.make(task_name, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    # Same rule as scripts/reliability_rollout.py: whether observations are
    # normalized is a property of the CHECKPOINT, never a constant. Asking for
    # normalization on a checkpoint that carries no ``obs_normalizer`` buffers
    # makes rsl_rl build an untrained EmpiricalNormalization whose forward is
    # still ``(x - 0) / (1 + 1e-2)`` -- a silent 1% shrink of every observation.
    # The shield artifact is fit on latents recorded through the rollout script,
    # so if the two harnesses answer this question differently the closed-loop
    # study scores its latents against a monitor fit in a different space.
    normalize_obs = checkpoint_has_obs_normalizer(Path(args.checkpoint))
    print(f"[cl] checkpoint obs normalization: {normalize_obs}", flush=True)

    eval_yaml = {
        "run": {
            "name": "closed_loop", "output_dir": "/tmp", "log_interval": 1, "save_interval": 1,
            "max_iterations": 1, "seed": int(args.process_seed), "device": args.device,
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
        "runner": {"num_steps_per_env": 24, "empirical_normalization": normalize_obs},
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

    fallback_contract = {
        "implementation": "policy action multiplied by one minus applied blend",
        "joint_names": list(robot.joint_names),
        "default_joint_pos": _tensor(robot.data.default_joint_pos[0])
        .detach()
        .cpu()
        .tolist(),
        "handoff_ticks": int(meta["timings"]["handoff_ticks"]),
        "action_scale": float(controller["action_scale"]),
        "control_dt_s": float(controller["control_dt_s"]),
    }
    fallback_contract_hash = value_sha256(fallback_contract)

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
                if not torch.allclose(
                    _tensor(actuator.stiffness),
                    stiffness * scale,
                    rtol=0.0,
                    atol=1e-7,
                ) or not torch.allclose(
                    _tensor(actuator.damping),
                    damping * scale,
                    rtol=0.0,
                    atol=1e-7,
                ):
                    raise RuntimeError(f"actuator gain readback mismatch for {name}")
            return True
        except Exception as exc:  # noqa: BLE001 - reported, never silently ignored
            print(f"[cl] MOTOR INJECTION FAILED: {type(exc).__name__}: {exc}", flush=True)
            return False

    def corrupt_obs(obs, std: float, generator):
        """Additive Gaussian corruption of the policy's observation input.

        Applied identically in every arm (it is the disturbance, not the
        treatment). Only the policy's *input* is corrupted; the environment's
        physical state is untouched, so the static fallback stays a safe
        attractor. A per-block generator makes the noise sequence byte-identical
        across arms regardless of how their trajectories diverge after switching.

        ``obs`` is the container the policy expects (a ``TensorDict`` keyed by
        observation group, or a bare tensor). The container type must be
        preserved -- the model indexes ``obs["policy"]`` -- so only the policy
        group's tensor is replaced.
        """
        if isinstance(obs, torch.Tensor):
            noise = torch.randn(obs.shape, generator=generator, device=obs.device, dtype=obs.dtype)
            return obs + noise * std
        pol = obs["policy"]
        noise = torch.randn(pol.shape, generator=generator, device=pol.device, dtype=pol.dtype)
        try:
            new = obs.clone()
        except AttributeError:
            new = dict(obs)
        new["policy"] = pol + noise * std
        return new

    def set_command(speed: float) -> bool:
        """Write a forward velocity command into the env's command buffer.

        The policy was trained only at zero command, so a nonzero command is out
        of distribution. Re-written every disturbed tick because the command
        manager resamples (to zero, for this stand env) on its own schedule.
        """
        try:
            cmd = env.unwrapped.command_manager.get_command("base_velocity")
            cmd[:] = 0.0
            cmd[:, 0] = speed
            return True
        except Exception as exc:  # noqa: BLE001 - reported, never silently ignored
            print(f"[cl] COMMAND INJECTION FAILED: {type(exc).__name__}: {exc}", flush=True)
            return False

    shield = VectorShield(args.artifact, args.envs)
    handoff = int(meta["timings"]["handoff_ticks"])

    # ---- per-block loop -----------------------------------------------------
    run_started_unix_s = time.time()
    process_uuid = str(uuid.uuid4())
    rows = {
        key: []
        for key in (
            "block_id",
            "fell",
            "pre_onset_fall",
            "post_onset_fall",
            "switch_tick",
            "full_fallback_tick",
            "engaged",
            "fall_tick",
            "return_until_first_fall",
            "active_ticks",
            "blend_sum",
            "reset_count",
            "fault_injected",
            "reset_state",
            "initial_obs",
            "onset_obs",
        )
    }
    with torch.inference_mode():
        for bi, block in enumerate(blocks):
            torch.manual_seed(block.seed)
            np.random.seed(block.seed)
            corrupt_gen = torch.Generator(device=env.unwrapped.device).manual_seed(
                int(block.seed) ^ 0x0B5EED
            )
            if not set_motor_scale(1.0):
                raise RuntimeError("FAIL CLOSED: could not restore baseline motor gains")
            env.seed(int(block.seed))
            reset_obs, _reset_extras = env.reset()
            shield.reset()
            reset_state = np.concatenate(
                [
                    _tensor(robot.data.root_state_w).detach().cpu().numpy(),
                    _tensor(robot.data.joint_pos).detach().cpu().numpy(),
                    _tensor(robot.data.joint_vel).detach().cpu().numpy(),
                ],
                axis=1,
            )
            obs = _unwrap(reset_obs)
            zero_actions = torch.zeros(
                (args.envs, env.num_actions),
                device=env.unwrapped.device,
                dtype=torch.float32,
            )
            for _ in range(int(params["reset_settle_ticks"])):
                settle_obs, _settle_reward, settle_dones, _settle_extras = env.step(
                    zero_actions
                )
                if bool(settle_dones.any()):
                    raise RuntimeError("FAIL CLOSED: environment terminated during reset settling")
                obs = _unwrap(settle_obs)
            initial_obs = _policy_group(obs).detach().cpu().numpy().copy()
            onset_obs = np.full_like(initial_obs, np.nan)

            fell = np.zeros(args.envs, bool)
            pre_onset_fall = np.zeros(args.envs, bool)
            post_onset_fall = np.zeros(args.envs, bool)
            fall_tick = np.full(args.envs, -1, np.int32)
            switch_tick = np.full(args.envs, -1, np.int32)
            full_fallback_tick = np.full(args.envs, -1, np.int32)
            engaged = np.zeros(args.envs, bool)
            return_until_first_fall = np.zeros(args.envs, np.float64)
            active_ticks = np.zeros(args.envs, np.int32)
            blend_sum = np.zeros(args.envs, np.float64)
            reset_count = np.zeros(args.envs, np.int32)
            sham_switch = (
                np.array([-1 if t is None else t for t in sham_sched[block.block_id]], np.int32)
                if sham_sched is not None
                else None
            )
            injected = False

            for tick in range(block.horizon_ticks):
                # Apply the block's disturbance to the policy's input / actuators.
                pol_in = obs
                if tick == block.onset_tick:
                    onset_obs = _policy_group(obs).detach().cpu().numpy().copy()
                if disturbance_kind == "motor":
                    if block.disturbed and not injected and tick == block.onset_tick:
                        injected = set_motor_scale(block.motor_scale)
                        if not injected:
                            raise RuntimeError(
                                f"FAIL CLOSED: motor fault injection failed for block {block.block_id}"
                            )
                elif disturbance_kind == "obs":
                    if block.disturbed and tick >= block.onset_tick:
                        injected = True
                        # Perceptual OOD: corrupt the policy input from onset onward.
                        pol_in = corrupt_obs(obs, block.obs_noise, corrupt_gen)
                elif disturbance_kind == "command":
                    # OOD command: inject from onset onward (persistent). The
                    # command reaches the policy via the next step's observation.
                    if block.disturbed and tick >= block.onset_tick:
                        injected = set_command(block.command_speed)
                        if not injected:
                            raise RuntimeError(
                                f"FAIL CLOSED: command fault injection failed for block {block.block_id}"
                            )

                captured.clear()
                actions = policy(pol_in)
                latent = torch.cat([captured[i] for i in tap_idx], dim=1)
                blend_np, _score, _armed = shield.step(latent.detach().cpu().numpy())
                alive = ~fell

                if args.arm == "unshielded":
                    applied_blend = np.zeros(args.envs)  # passive monitor only
                elif args.arm == "shielded":
                    applied_blend = blend_np
                elif args.arm == "oracle":
                    # Perfect-onset schedule: on disturbed blocks, the first
                    # positive ramp command is applied at the registered onset
                    # (+ optional diagnostic delay). Nominal blocks never switch.
                    if block.disturbed:
                        since = tick - (block.onset_tick + args.oracle_delay_ticks)
                        b = min(1.0, (since + 1) / handoff) if since >= 0 else 0.0
                    else:
                        b = 0.0
                    applied_blend = np.full(args.envs, b)
                else:  # sham: ramp on the permuted schedule, same handoff length
                    since = tick - sham_switch
                    applied_blend = np.where(
                        (sham_switch >= 0) & (since >= 0), np.minimum(1.0, since / handoff), 0.0
                    )

                # A terminal environment is no longer part of this registered
                # trial. Isaac may auto-reset it internally, but replacement
                # episodes receive no treatment and contribute no outcome,
                # return, or dose.
                applied_blend = np.where(alive, applied_blend, 0.0)
                newly = (applied_blend > 0) & (switch_tick < 0)
                switch_tick[newly] = tick
                newly_full = (applied_blend >= 1.0) & (full_fallback_tick < 0)
                full_fallback_tick[newly_full] = tick
                engaged |= applied_blend > 0
                blend_sum += applied_blend * alive
                active_ticks += alive.astype(np.int32)

                # Blending the action toward zero == blending the joint target
                # toward the default stand pose, which is what the ROS node does.
                blended = actions * torch.as_tensor(
                    1.0 - applied_blend, device=actions.device, dtype=actions.dtype
                ).unsqueeze(1)
                blended[torch.as_tensor(~alive, device=actions.device)] = 0.0

                obs2, reward, dones, extras = env.step(blended)
                time_out = extras.get("time_outs") if isinstance(extras, dict) else None
                d = np.asarray(dones.detach().cpu()).astype(bool).reshape(-1)
                reward_np = np.asarray(reward.detach().cpu(), dtype=np.float64).reshape(-1)
                return_until_first_fall += reward_np * alive
                t_out = (
                    np.asarray(time_out.detach().cpu()).astype(bool).reshape(-1)
                    if time_out is not None
                    else np.zeros(args.envs, bool)
                )
                new_falls = d & (~t_out) & alive
                fall_tick[new_falls] = tick
                pre_onset_fall |= new_falls & (tick < block.onset_tick)
                post_onset_fall |= new_falls & (tick >= block.onset_tick)
                fell |= new_falls
                reset_count += d.astype(np.int32)
                if d.any():
                    shield.reset(np.flatnonzero(d))
                obs = _unwrap(obs2)

            if block.disturbed and not injected:
                raise RuntimeError(
                    f"FAIL CLOSED: block {block.block_id} never received its registered fault"
                )
            rows["block_id"].append(block.block_id)
            rows["fell"].append(fell.copy())
            rows["pre_onset_fall"].append(pre_onset_fall.copy())
            rows["post_onset_fall"].append(post_onset_fall.copy())
            rows["switch_tick"].append(switch_tick.copy())
            rows["full_fallback_tick"].append(full_fallback_tick.copy())
            rows["engaged"].append(engaged.copy())
            rows["fall_tick"].append(fall_tick.copy())
            rows["return_until_first_fall"].append(return_until_first_fall.copy())
            rows["active_ticks"].append(active_ticks.copy())
            rows["blend_sum"].append(blend_sum.copy())
            rows["reset_count"].append(reset_count.copy())
            rows["fault_injected"].append(bool(injected) if block.disturbed else True)
            rows["reset_state"].append(reset_state)
            rows["initial_obs"].append(initial_obs)
            rows["onset_obs"].append(onset_obs)
            print(
                f"[cl] block {bi + 1}/{len(blocks)} id={block.block_id} "
                f"{'disturbed' if block.disturbed else 'nominal  '} "
                f"fell={fell.sum()}/{args.envs} pre={pre_onset_fall.sum()} "
                f"post={post_onset_fall.sum()} engaged={engaged.sum()}/{args.envs}",
                flush=True,
            )

    out, meta_out = _arm_output_paths(args)
    np.savez(
        out,
        **{key: np.asarray(value) for key, value in rows.items()},
        task_complete=~np.asarray(rows["fell"]),
    )
    trajectory_digest = hashlib.sha256()
    trajectory_digest.update(np.asarray(rows["initial_obs"]).tobytes())
    trajectory_digest.update(np.asarray(rows["onset_obs"]).tobytes())
    trajectory_digest.update(np.asarray(rows["fall_tick"]).tobytes())
    runtime_versions = {
        package: md.version(distribution)
        for package, distribution in {
            "isaaclab": "isaaclab",
            "numpy": "numpy",
            "rsl_rl_lib": "rsl-rl-lib",
            "torch": "torch",
        }.items()
    }
    run_finished_unix_s = time.time()
    meta_out.write_text(
        json.dumps(
            {
                "arm": args.arm,
                "study_id": params["study_id"],
                "cell_id": params["cell_id"],
                "replicate_id": params["replicate_id"],
                "policy_name": params["policy_name"],
                "bundle_id": manifest.bundle_id,
                "protocol_hash": protocol.get("protocol_hash"),
                "protocol_seed": params["protocol_seed"],
                "process_seed": params["process_seed"],
                "process_uuid": process_uuid,
                "process_pid": os.getpid(),
                "blocks": len(blocks),
                "envs_per_block": args.envs,
                "empirical_normalization": bool(normalize_obs),
                "pilot": bool(args.pilot or args.preflight_subset),
                "preflight_subset": bool(args.preflight_subset),
                "diagnostic": args.arm == "oracle",
                "oracle_delay_ticks": args.oracle_delay_ticks if args.arm == "oracle" else None,
                "code_commit": current_commit,
                "code_dirty": current_dirty,
                "source_snapshot_sha256": snapshot["sha256"],
                "source_files": snapshot["files"],
                "resolved_env_config_sha256": env_contract_hash,
                "runtime_versions": runtime_versions,
                "bundle_file_hashes": manifest.files,
                "fallback_contract": fallback_contract,
                "fallback_contract_sha256": fallback_contract_hash,
                "fault_configuration": {
                    key: params[key]
                    for key in (
                        "disturbance_kind",
                        "motor_scale_range",
                        "obs_noise_range",
                        "onset_range",
                        "horizon_ticks",
                    )
                },
                "trajectory_sha256": trajectory_digest.hexdigest(),
                "raw_output_sha256": file_sha256(out),
                "run_started_unix_s": run_started_unix_s,
                "run_finished_unix_s": run_finished_unix_s,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"[cl] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
