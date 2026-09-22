#!/usr/bin/env python3
"""Phase D: the exact deployment path, closed around Isaac Lab physics.

Everything between the sensors and the motor command is the deploy code, not the
training action term:

    sim state -> ObservationBuilder / assemble_policy_observation (deploy obs path,
    base_lin_vel from the deploy config's source) -> ONNX Runtime on the EXPORTED
    policy.onnx -> policy_action_map (trained clamp, fed-back last_action) ->
    node_soft_limit -> command_wire.encode -> ActuatorGate.on_command/tick (the bridge's
    state machine, limiter and hard envelope from the deploy config) -> the gate's
    final target AND per-motor kp/kd written into the simulated DC-motor actuators.

The gate's tick records are written through the bridge's own ``TelemetryWriter``
(schema v2), one file per simulated robot, and scored by the same
``phoenix.monitor.fidelity`` gate a hardware run is scored by. The simulated env has
its own soft limiter DISABLED; the gate is the only limiter.

What is NOT exercised: ROS transport and timing, the policy node's own sensor
watchdogs and attitude/runtime aborts (they run in rclpy), the Jetson's CPU timing,
and any real-hardware effect. A controlled degradation (``--degrade RR_thigh:0.6``)
is applied by the gate exactly as on the robot (``DegradationSpec``: kp/kd scaled on
one motor in policy mode, ramped over 2 s); ``--allow-degradation`` is required.

Usage::

    source scripts/_activate.sh
    PYTHONPATH=src python scripts/phoenix_v2_sim2sim_deploy.py \
        --deploy-config configs/sim2real/deploy_stand_h25_v2.yaml \
        --env-config configs/env/phoenix_v2/sim2sim_nominal.yaml \
        --num-envs 32 --seed 2001 --out results/phoenix_v2/sim2sim/nominal_v2
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--deploy-config", type=Path, required=True)
    p.add_argument("--env-config", type=Path, required=True)
    p.add_argument("--onnx", type=Path, default=None, help="default: policy.onnx_path in the deploy config")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--seed", type=int, default=2001)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--duration-s", type=float, default=20.0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--degrade", default=None, help="JOINT:scale, e.g. RR_thigh:0.6 (gate-applied)")
    p.add_argument("--allow-degradation", action="store_true")
    p.add_argument("--kp", type=float, default=25.0)
    p.add_argument("--kd", type=float, default=0.5)
    p.add_argument("--save-steps", action="store_true")
    p.add_argument(
        "--factorial",
        default=None,
        choices=["clamp_measured_q", "noclamp_prev_command", "clamp_prev_command", "noclamp_measured_q"],
        help="RESEARCH ONLY: override the action clamp and limiter of the deploy config to "
        "isolate each change (recorded in the manifest; never a deploy configuration)",
    )
    p.add_argument(
        "--walk",
        action="store_true",
        help="SIMULATION ONLY: follow the env's velocity commands, feed true body velocity "
        "as an idealised odometry source, and lift the gate's walking block in THIS process "
        "(recorded in the manifest). The deploy contract still refuses walking configs.",
    )
    p.add_argument("--policy-onnx-override", type=Path, default=None)
    p.add_argument("--walk-thresholds", type=Path, default=None)
    return p.parse_args(argv)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    args = parse_args(argv)
    if args.degrade and not args.allow_degradation:
        print("--degrade needs --allow-degradation", file=sys.stderr)
        return 2
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    from phoenix.sim_app_exit import run_isaac_main

    return run_isaac_main(lambda: _run(args), app, label="phoenix_v2_sim2sim_deploy")


def _run(args) -> int:
    import gymnasium as gym
    import numpy as np
    import onnxruntime as ort
    import torch
    import warp as wp
    import yaml

    from phoenix.monitor.fidelity import fidelity_report
    from phoenix.monitor.layers import read_bridge_telemetry
    from phoenix.monitor.stand_metrics import score_stand_rollout
    from phoenix.sim2real.action_map import node_soft_limit, policy_action_map
    from phoenix.sim2real.actuator_gate import (
        ActuatorGate,
        GateParams,
        limiter_params_from_config,
    )
    from phoenix.sim2real.bridge_telemetry import TelemetryWriter
    from phoenix.sim2real.command_wire import KIND_POLICY, encode
    from phoenix.sim2real.degradation import parse_spec
    from phoenix.sim2real.deploy_contract import semantic_config_sha256, validate_deploy_contract
    from phoenix.sim2real.go2_model import (
        LIMIT_ABORT_BAND_RAD,
        POLICY_JOINT_ORDER,
        UNITREE_MOTOR_ORDER,
        limits_in_order,
    )
    from phoenix.sim2real.motor_crc import PHOENIX_FOR_MOTOR
    from phoenix.sim2real.observation import (
        JointOrder,
        ObservationBuilder,
        assemble_policy_observation,
    )
    from phoenix.sim_env import build_env_cfg, load_layered_config

    def t2n(x):
        if isinstance(x, wp.array):
            x = wp.to_torch(x)
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    t0 = time.time()
    if args.walk:
        import phoenix.sim2real.actuator_gate as _gate_mod

        _gate_mod.WALKING_ENABLED = True  # this process only; see --walk
    dcfg = yaml.safe_load(args.deploy_config.read_text())
    problems = validate_deploy_contract(dcfg)
    walk_block_lifted = []
    if args.walk:
        # SIMULATION ONLY: the walking refusal is the one problem this process may lift
        # (hardware prerequisites, deploy_contract.WALKING_PREREQUISITES); recorded below.
        walk_block_lifted = [p for p in problems if p.startswith("walking deploy configs are blocked")]
        problems = [p for p in problems if p not in walk_block_lifted]
    if problems:
        raise SystemExit(f"deploy contract refuses {args.deploy_config}: {problems}")
    order = tuple(dcfg["joint_order"])
    if order != tuple(POLICY_JOINT_ORDER):
        raise SystemExit("deploy joint order differs from the policy order")
    control = dcfg["control"]
    scale = float(control["action_scale"])
    clip = control.get("action_clip")
    clip = None if clip is None else float(clip)
    default_p = np.asarray([control["default_joint_pos"][n] for n in order], dtype=np.float32)
    limiter = limiter_params_from_config(dcfg)
    if args.factorial:
        clamp_part, lim_part = args.factorial.split("_", 1)
        clip = 1.0 if clamp_part == "clamp" else None
        if lim_part == "prev_command":
            limiter = limiter_params_from_config(
                yaml.safe_load(Path("configs/sim2real/deploy_stand_h25_v2.yaml").read_text())
            )
        else:
            limiter = limiter_params_from_config({})
    lin_src = "odom" if args.walk else dcfg["observation"]["base_lin_vel_source"]
    onnx_path = args.policy_onnx_override or args.onnx or Path(dcfg["policy"]["onnx_path"])
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    builder = ObservationBuilder(JointOrder(order), control["default_joint_pos"])
    perm = np.asarray(PHOENIX_FOR_MOTOR)  # motor k <- policy perm[k]
    inv = np.argsort(perm)  # policy j <- motor inv[j]
    degradation = parse_spec(args.degrade) if args.degrade else None

    loaded = load_layered_config(args.env_config)
    container = loaded.to_container()
    if (container.get("action", {}).get("rate_limit", {}) or {}).get("enabled", True):
        raise SystemExit("the env's own soft limiter must be disabled: the gate is the limiter")
    env_cfg = build_env_cfg(loaded)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    # A session is one uninterrupted episode: never let the env's own time-out cut a
    # longer run short (it resets the robot and ends that session's telemetry).
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), args.duration_s + 1.0)
    env = gym.make(container["env"]["task_name"], cfg=env_cfg)
    env.reset(seed=args.seed)
    u = env.unwrapped
    robot = u.scene["robot"]
    term = u.action_manager.get_term("joint_pos")
    sim_default = t2n(term._offset)[0]
    if not np.allclose(sim_default, default_p, atol=1e-6):
        raise SystemExit(f"sim default pose {sim_default} != deploy default {default_p}")
    acts = list(robot.actuators.values())
    if len(acts) != 1 or list(acts[0].joint_names) != list(order):
        raise SystemExit("expected one actuator group in policy order")
    act = acts[0]
    kp_base = t2n(act.stiffness).copy()
    kd_base = t2n(act.damping).copy()
    kp_fac = kp_base / args.kp  # DR factors, if the env randomised gains
    kd_fac = kd_base / args.kd
    lo_p, hi_p = limits_in_order(order)

    N = args.num_envs
    dt = env_cfg.decimation * env_cfg.sim.dt
    T = int(round(args.duration_s / dt))
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    params = GateParams(
        live=False,
        kp=args.kp,
        kd=args.kd,
        hold_kp=20.0,
        hold_kd=1.0,
        watchdog_s=0.2,
        estop_timeout_s=float(dcfg["safety"]["estop_timeout_s"]),
        lowstate_timeout_s=float(dcfg["safety"]["sensor_timeout_s"]),
        stale_hold_s=0.2,
        first_message_timeout_s=float(dcfg["safety"]["first_message_timeout_s"]),
        joint_order=order,
        degradation=degradation,
        **limiter,
    )
    manifest_common = {
        "node": "phoenix_sim2sim_deploy",
        "stage": "SIM2SIM",
        "live": False,
        "deploy_config": {
            "path": str(args.deploy_config),
            "sha256": _sha(args.deploy_config),
            "semantic_sha256": semantic_config_sha256(dcfg),
        },
        "artifacts": {"policy.onnx": {"path": str(onnx_path), "sha256": _sha(Path(onnx_path))}},
        "gate_params": {k: v for k, v in params.__dict__.items() if k != "degradation"},
        "controlled_degradation": None if degradation is None else degradation.to_dict(),
        "env_config": str(args.env_config),
        "seed": args.seed,
        "factorial_override": args.factorial,
        "walk_sim_override": bool(args.walk),
        "walk_contract_refusal_lifted_in_sim": walk_block_lifted,
        "effective_action_clip": clip,
        "effective_limiter": limiter,
    }
    tel_dir = out / "bridge"
    tel_dir.mkdir()
    writers = [
        TelemetryWriter(tel_dir / f"robot{e:03d}.jsonl", {**manifest_common, "robot": e})
        for e in range(N)
    ]
    ns = lambda k: int(round((k + 1) * dt * 1e9))  # noqa: E731
    gates = [ActuatorGate(params, 0) for _ in range(N)]
    last_action = np.zeros((N, 12), np.float32)

    shape = (T, N, 12)
    arrs = {
        k: np.zeros(shape, np.float32)
        for k in ("raw", "req", "sent", "q0", "q1", "qd1", "tau_c", "tau_a", "kp_sent")
    }
    grav = np.zeros((T, N, 3), np.float32)
    height = np.zeros((T, N), np.float32)
    linv = np.zeros((T, N, 3), np.float32)
    angv = np.zeros((T, N, 3), np.float32)
    cmd = np.zeros((T, N, 3), np.float32)
    alive = np.ones((T, N), bool)
    hold = np.zeros((T, N), bool)
    contact_term = np.zeros(N, bool)
    ended = np.zeros(N, bool)
    names = list(u.termination_manager.active_terms)
    kp_t = torch.as_tensor(kp_base, device=args.device)
    kd_t = torch.as_tensor(kd_base, device=args.device)

    with torch.inference_mode():
        for k in range(T):
            now = ns(k)
            q = t2n(robot.data.joint_pos)
            dq = t2n(robot.data.joint_vel)
            tau = t2n(robot.data.applied_torque)
            quat = t2n(robot.data.root_quat_w)  # xyzw in this Isaac Lab (checked in Phase B)
            angb = t2n(robot.data.root_ang_vel_b)
            linb = t2n(robot.data.root_lin_vel_b)
            lin_fed = linb if lin_src == "odom" else np.zeros_like(linb)
            vcmd = (
                t2n(u.command_manager.get_command("base_velocity")).astype(np.float32)
                if args.walk
                else np.zeros((N, 3), np.float32)
            )
            obs = np.stack(
                [
                    assemble_policy_observation(
                        builder,
                        base_lin_vel=lin_fed[e],
                        quat_xyzw=tuple(float(v) for v in quat[e]),
                        base_ang_vel=angb[e],
                        velocity_command=vcmd[e],
                        joint_pos=q[e],
                        joint_vel=dq[e],
                        last_action=last_action[e],
                    )
                    for e in range(N)
                ]
            ).astype(np.float32)
            raw = sess.run(None, {in_name: obs})[0]
            final_p = np.zeros((N, 12), np.float32)
            kp_p = np.zeros((N, 12), np.float32)
            kd_p = np.zeros((N, 12), np.float32)
            for e in range(N):
                fed, req = policy_action_map(raw[e], default_p, scale, clip)
                last_action[e] = fed
                node_t = node_soft_limit(req, q[e], limiter["limiter_mode"], limiter["max_delta"])
                label, data = encode(
                    order,
                    seq=k + 1,
                    kind=KIND_POLICY,
                    target=node_t,
                    requested_target=req,
                    raw_action=raw[e],
                    q_policy=q[e],
                    base_lin_vel_fed=lin_fed[e],
                    cmd_vel_received=vcmd[e],
                    velocity_command_fed=vcmd[e],
                    obs_source_code=0.0 if lin_src == "zeros" else 1.0,
                    stand_only=0.0 if args.walk else 1.0,
                )
                g = gates[e]
                g.on_lowstate(now, q[e][perm], dq[e][perm], tau[e][perm])
                g.on_estop(now, False)
                g.on_command(now, label, data)
                rec = g.tick(now)
                if not ended[e]:
                    writers[e].write_tick(rec)
                final_p[e] = np.asarray(rec["final_target_unitree"])[inv]
                kp_p[e] = np.asarray(rec["kp_unitree"])[inv]
                kd_p[e] = np.asarray(rec["kd_unitree"])[inv]
                hold[k, e] = rec["mode"] != "policy"
                arrs["raw"][k, e] = raw[e]
                arrs["req"][k, e] = req
            arrs["sent"][k] = final_p
            arrs["kp_sent"][k] = kp_p
            arrs["q0"][k] = q
            cmd[k] = vcmd
            # Gains exactly as the gate sent them, times the env's DR factor (1 when off).
            kp_t[:] = torch.as_tensor(kp_p * kp_fac, device=args.device)
            kd_t[:] = torch.as_tensor(kd_p * kd_fac, device=args.device)
            act.stiffness[:] = kp_t
            act.damping[:] = kd_t
            action = torch.as_tensor((final_p - default_p) / scale, device=args.device)
            _, _, terminated, truncated, _ = env.step(action)
            d = (t2n(terminated) | t2n(truncated)).astype(bool)
            arrs["q1"][k] = t2n(robot.data.joint_pos)
            arrs["qd1"][k] = t2n(robot.data.joint_vel)
            arrs["tau_c"][k] = t2n(robot.data.computed_torque)
            arrs["tau_a"][k] = t2n(robot.data.applied_torque)
            grav[k] = t2n(robot.data.projected_gravity_b)
            height[k] = t2n(robot.data.root_pos_w)[:, 2] - t2n(u.scene.env_origins)[:, 2]
            linv[k] = t2n(robot.data.root_lin_vel_b)
            angv[k] = t2n(robot.data.root_ang_vel_b)
            alive[k] = ~ended
            if d.any():
                for name in names:
                    if name == "time_out":
                        continue
                    hit = t2n(u.termination_manager.get_term(name)).astype(bool)
                    contact_term |= hit & d & ~ended
                for e in np.flatnonzero(d & ~ended):
                    writers[e].close({"reason": "episode_end"})
                ended |= d
    for e in range(N):
        writers[e].close({"reason": "duration_complete"})
    env.close()

    metrics, episodes = score_stand_rollout(
        raw=arrs["raw"], req=arrs["req"], sent=arrs["sent"], q0=arrs["q0"], q1=arrs["q1"],
        tau_c=arrs["tau_c"], tau_a=arrs["tau_a"], grav=grav, height=height, linv=linv,
        angv=angv, cmd=cmd, alive=alive, contact_term=contact_term, ended=ended,
        default=default_p, lo=lo_p, hi=hi_p, dt=dt, abort_band=LIMIT_ABORT_BAND_RAD,
        joint_names=list(order), safety_hold=hold,
    )
    if args.walk:
        from phoenix.monitor.stand_metrics import score_walk_episodes

        T_, N_ = alive.shape
        first_end = np.where((~alive).any(axis=0), np.argmax(~alive, axis=0), T_)
        valid = np.arange(T_)[:, None] < first_end[None, :]
        metrics.update(score_walk_episodes(episodes, linv=linv, angv=angv, cmd=cmd, valid=valid, dt=dt))
        from phoenix.monitor.stand_metrics import score_walk_v2

        wth = json.loads(args.walk_thresholds.read_text()) if args.walk_thresholds else None
        metrics.update(score_walk_v2(episodes, linv=linv, angv=angv, cmd=cmd, valid=valid,
                                     height=height, qd=arrs["qd1"], req=arrs["req"],
                                     tau_c=arrs["tau_c"], tau_a=arrs["tau_a"], dt=dt,
                                     thresholds=wth))
    # The hardware fidelity gate, on the gate's own telemetry, per simulated robot.
    fid = []
    for e in range(N):
        rep = fidelity_report(read_bridge_telemetry(tel_dir / f"robot{e:03d}.jsonl"))
        fid.append({k: rep.get(k) for k in ("verdict", "reasons", "altered_fraction",
                                            "rms_distortion_rad", "authority_s")})
    faults = sorted({f for g in gates for f in g.faults})
    summary = {
        "schema": "phoenix-v2-sim2sim-deploy/v1",
        "label": args.label,
        **manifest_common,
        "env_resolved": container,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()),
        "num_robots": N,
        "steps": T,
        "dt_s": dt,
        **metrics,
        "hardware_fidelity_gate": {
            "pass_rate": float(np.mean([f["verdict"] == "PASS" for f in fid])),
            "per_robot": fid,
        },
        "gate_faults": faults,
        "wall_s": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    with (out / "episodes.jsonl").open("w") as fh:
        for ep, f in zip(episodes, fid):
            fh.write(json.dumps({**ep, "hardware_fidelity": f["verdict"]}) + "\n")
    if args.save_steps:
        np.savez_compressed(out / "steps.npz", grav=grav, alive=alive, hold=hold, **arrs)
    print(json.dumps({k: summary[k] for k in (
        "label", "success_rate", "survival_rate", "mean_primary_score", "fidelity_pass_rate",
        "altered_fraction", "rms_modification_rad", "raw_out_of_range_fraction",
        "attitude_violation_episode_rate", "safety_hold_episode_rate", "mean_base_height_m",
        "gate_faults", "wall_s")} | {k: v for k, v in summary.items() if k.startswith("walk")} | {"hw_gate_pass_rate": summary["hardware_fidelity_gate"]["pass_rate"]},
        indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
