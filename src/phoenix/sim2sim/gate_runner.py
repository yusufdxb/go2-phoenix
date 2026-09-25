"""Run a deploy spec + policy through the sim2sim gate scenarios in MuJoCo.

Loop per control step (``spec.control_hz``, 50 Hz for every policy here):

1. observation from the MuJoCo state via ``spec.build_obs`` (for Phoenix
   manifests: :func:`phoenix.velocity.observation.build_actor_observation`, the
   hardware node's function);
2. policy -> raw action; count joints with ``|raw| > clip`` (pre-clip
   saturation), then ``spec.postprocess``: clamp, scale, add default pose (for
   Phoenix manifests: :func:`phoenix.velocity.observation.actions_to_joint_targets`);
   ``last_action`` is the clamped action, as Isaac's action manager and rl_sar
   both store it;
3. ``physics_hz / control_hz`` physics steps, each computing
   ``tau = kp (q_des - q) - kd qd`` from the CURRENT joint state and clipping it
   with the real GO2 DCMotor envelope per joint group (hip/thigh 23.5 Nm,
   30.1 rad/s; calf 45.43 Nm, 15.70 rad/s), optionally through a target delay.

MuJoCo, onnxruntime and imageio are imported lazily. The PD, clip and metric
math is numpy and shared with the tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phoenix.sim2real.go2_model import limits_in_order
from phoenix.velocity.contract import JOINT_ORDER

from .dc_motor import DCMotorParams, TargetDelay, clip_dc_motor_effort
from .deploy_spec import JOINT_GROUPS, DeploySpec, ObsHistory, joint_group
from .gate import (
    REPORT_SCHEMA,
    GateConfig,
    GateScenario,
    envelope_checks,
    evaluate_scenario,
    gate_verdict,
    performance_table,
    roll_pitch_from_quat_wxyz,
    saturation_mask,
    tag_spec_checks,
    tracking_rmse,
)
from .model import load_go2_model

#: MuJoCo joint limits are soft constraints; a joint pressed against its stop
#: penetrates by a few mrad. Only excursions beyond this count as violations.
JOINT_LIMIT_TOL_RAD = 0.005
_HARD_LO, _HARD_HI = limits_in_order(JOINT_ORDER)
_SIM_GROUPS = np.asarray([joint_group(n) for n in JOINT_ORDER])


def real_motor_params(cfg: GateConfig) -> DCMotorParams:
    """Per-joint (JOINT_ORDER) DCMotor envelope from the gate config. Gains unused here."""
    eff = np.asarray([cfg.actuator[joint_group(n)]["effort_limit"] for n in JOINT_ORDER], dtype=np.float64)
    vel = np.asarray([cfg.actuator[joint_group(n)]["velocity_limit"] for n in JOINT_ORDER], dtype=np.float64)
    return DCMotorParams(stiffness=0.0, damping=0.0, effort_limit=eff, saturation_effort=eff,  # type: ignore[arg-type]
                         velocity_limit=vel)  # type: ignore[arg-type]


def pd_torque(q_des: np.ndarray, q: np.ndarray, qd: np.ndarray, kp: np.ndarray, kd: np.ndarray,
              motor: DCMotorParams) -> tuple[np.ndarray, np.ndarray]:
    """``(computed, applied)``: position PD with zero velocity target, then the DCMotor clip."""
    computed = kp * (q_des - q) - kd * qd
    return computed, clip_dc_motor_effort(computed, qd, motor)


def spec_permutation(spec: DeploySpec) -> np.ndarray:
    """``perm[i]`` = index in JOINT_ORDER (the sim's array order) of the spec's joint ``i``."""
    return np.asarray([JOINT_ORDER.index(n) for n in spec.joint_order], dtype=np.int64)


# --------------------------------------------------------------------- policies


class OnnxPolicy:
    def __init__(self, path: str | Path, obs_dim: int) -> None:
        import onnxruntime as ort

        self.path = Path(path)
        self.name = f"onnx:{self.path.name}"
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
        inputs = self._sess.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"{self.path}: expected one input, found {len(inputs)}")
        shape = inputs[0].shape
        if len(shape) != 2 or (isinstance(shape[1], int) and shape[1] != obs_dim):
            raise ValueError(f"{self.path}: input shape {shape} does not match spec obs_dim {obs_dim}")
        self._in = inputs[0].name
        outs = [o.name for o in self._sess.get_outputs()]
        self._out = "action" if "action" in outs else ("actions" if "actions" in outs else outs[0])
        self.obs_dim = obs_dim

    def reset(self) -> None:
        return None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(1, self.obs_dim)
        (y,) = self._sess.run([self._out], {self._in: x})
        return np.asarray(y, dtype=np.float64).reshape(-1)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------- sim


class GateSim:
    def __init__(self, cfg: GateConfig, scenario: GateScenario, *, scene_xml: str | None = None) -> None:
        import mujoco

        self._mj = mujoco
        phys = cfg.physics
        self.model, self.idx, self.model_info = load_go2_model(
            phys["profile"],
            scene_xml=scene_xml,
            foot_friction=scenario.foot_friction if scenario.foot_friction is not None
            else float(phys["nominal_foot_friction"]),
            timestep=1.0 / float(phys["physics_hz"]),
            payload_kg=scenario.payload_kg,
        )
        self.data = mujoco.MjData(self.model)

    def reset_to_pose(self, q_sim: np.ndarray, clearance: float) -> float:
        """Place the robot at joint pose ``q_sim`` (JOINT_ORDER), lowest foot ``clearance`` above floor."""
        mj, m, d = self._mj, self.model, self.data
        mj.mj_resetData(m, d)
        d.qpos[0:3] = (0.0, 0.0, 1.0)
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        d.qpos[self.idx.qpos_adr] = q_sim
        mj.mj_forward(m, d)
        low = min(float(d.geom_xpos[g][2] - m.geom_size[g][0]) for g in self.idx.foot_geom_ids)
        h = 1.0 - low + clearance
        d.qpos[2] = h
        d.qvel[:] = 0.0
        mj.mj_forward(m, d)
        return h

    def q(self) -> np.ndarray:
        return self.data.qpos[self.idx.qpos_adr].copy()

    def qd(self) -> np.ndarray:
        return self.data.qvel[self.idx.dof_adr].copy()

    def quat(self) -> np.ndarray:
        return self.data.qpos[3:7].copy()

    def gyro_body(self) -> np.ndarray:
        # MuJoCo free-joint angular velocity qvel[3:6] is in the body frame.
        return self.data.qvel[3:6].copy()

    def lin_vel_body(self) -> np.ndarray:
        w, x, y, z = self.quat() / np.linalg.norm(self.quat())
        r = np.asarray([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        return r.T @ self.data.qvel[0:3]

    def step(self, tau_sim: np.ndarray) -> None:
        self.data.ctrl[self.idx.actuator_id] = tau_sim
        self._mj.mj_step(self.model, self.data)


class _Video:
    def __init__(self, sim: GateSim, cfg: Mapping[str, Any]) -> None:
        import mujoco

        self._mj = mujoco
        self.renderer = mujoco.Renderer(sim.model, int(cfg["height"]), int(cfg["width"]))
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = sim.idx.base_body_id
        self.cam.distance = 1.6
        self.cam.elevation = -15.0
        self.cam.azimuth = 120.0
        self.frames: list[np.ndarray] = []
        self.fps = int(cfg["fps"])

    def capture(self, data: Any) -> None:
        self.renderer.update_scene(data, camera=self.cam)
        self.frames.append(self.renderer.render().copy())

    def write(self, path: Path) -> None:
        import imageio.v2 as imageio

        imageio.mimwrite(str(path), self.frames, fps=self.fps, codec="libx264", quality=6,
                         macro_block_size=8)

    def close(self) -> None:
        self.renderer.close()


# --------------------------------------------------------------------- scenario


@dataclass
class RunOptions:
    latency_ms: float
    video_dir: Path | None = None
    scene_xml: str | None = None
    #: Debug: if a list, one dict per control step is appended (t, cmd, q and
    #: targets in JOINT_ORDER, raw action in spec order, base height, gyro z).
    trace: list | None = None


def run_gate_scenario(
    spec: DeploySpec,
    policy: Callable[[np.ndarray], np.ndarray],
    scenario: GateScenario,
    cfg: GateConfig,
    opts: RunOptions,
) -> dict[str, Any]:
    phys = cfg.physics
    physics_hz = int(phys["physics_hz"])
    if physics_hz % spec.control_hz:
        raise ValueError(f"physics_hz {physics_hz} is not a multiple of control_hz {spec.control_hz}")
    decim = physics_hz // spec.control_hz
    control_dt = 1.0 / spec.control_hz
    sim = GateSim(cfg, scenario, scene_xml=opts.scene_xml)
    motor = real_motor_params(cfg)
    perm = spec_permutation(spec)
    kp_sim = np.empty(12)
    kd_sim = np.empty(12)
    kp_sim[perm] = spec.kp
    kd_sim[perm] = spec.kd
    default_sim = np.empty(12)
    default_sim[perm] = spec.default_joint_pos
    init_h = sim.reset_to_pose(default_sim, float(phys["init_foot_clearance_m"]))
    lag = int(round(float(opts.latency_ms) * 1e-3 * physics_hz))
    delay = TargetDelay(lag)
    if hasattr(policy, "reset"):
        policy.reset()  # type: ignore[attr-defined]
    history = (
        ObsHistory(spec.history_length, spec.obs_dim // spec.history_length)
        if spec.history_length > 1 else None
    )

    video = None
    video_error = None
    if opts.video_dir is not None:
        try:
            video = _Video(sim, cfg.raw["video"])
        except Exception as exc:  # noqa: BLE001  (headless GL failures vary by driver)
            video_error = f"{type(exc).__name__}: {exc}"
    frame_every = max(1, int(round(spec.control_hz / int(cfg.raw["video"]["fps"])))) if video else 0

    n_ctrl = int(round(scenario.duration_s * spec.control_hz))
    last_action = np.zeros(12)
    fall = cfg.fall
    groups = _SIM_GROUPS
    g_mask = {g: groups == g for g in JOINT_GROUPS}
    near_margin = float(cfg.thresholds["all"]["near_limit_margin_rad"])

    times, cmds, vxy, wz, heights = [], [], [], [], []
    sat_pre = 0
    sat_undefined = False
    raw_gt1 = 0
    raw_max = 0.0
    n_act = 0
    tq_sat = {g: 0 for g in JOINT_GROUPS}
    tq_peak_applied = {g: 0.0 for g in JOINT_GROUPS}
    tq_peak_demand = {g: 0.0 for g in JOINT_GROUPS}
    vel_over = {g: 0 for g in JOINT_GROUPS}
    phys_samples = 0
    near = {g: 0 for g in JOINT_GROUPS}
    min_margin = {g: math.inf for g in JOINT_GROUPS}
    violation_steps = 0
    fell = False
    fall_reason = None
    t_fall = None
    nonfinite = False
    max_rp = 0.0
    start_xy = sim.data.qpos[0:2].copy()
    limit_lo, limit_hi = _HARD_LO, _HARD_HI
    vel_lim = np.asarray(motor.velocity_limit)

    k_done = 0
    for k in range(n_ctrl):
        t = k * control_dt
        cmd = scenario.command_at(t)
        q_sim, qd_sim = sim.q(), sim.qd()
        obs = spec.build_obs(
            gyro_body=sim.gyro_body(), quat_wxyz=sim.quat(), command=cmd,
            joint_pos=q_sim[perm], joint_vel=qd_sim[perm], last_action=last_action,
        )
        if history is not None:
            obs = history.push(obs)
        raw =np.asarray(policy(obs), dtype=np.float64).reshape(-1)
        if raw.shape != (12,) or not np.all(np.isfinite(raw)):
            nonfinite = True
            fell, fall_reason, t_fall = True, "nonfinite_action", t
            break
        action, targets, _sat = spec.postprocess(raw)
        sat = saturation_mask(raw, spec.action_clip, cfg)
        if sat is None:
            sat_undefined = True
        else:
            sat_pre += int(sat.sum())
        raw_gt1 += int(np.sum(np.abs(raw) > 1.0))
        raw_max = max(raw_max, float(np.max(np.abs(raw))))
        n_act += 12
        last_action = action
        target_sim = np.empty(12)
        target_sim[perm] = targets

        for _ in range(decim):
            q, qd = sim.q(), sim.qd()
            computed, applied = pd_torque(delay(target_sim), q, qd, kp_sim, kd_sim, motor)
            clipped = np.abs(computed - applied) > 1e-9
            over = np.abs(qd) > vel_lim
            for g, m in g_mask.items():
                tq_sat[g] += int(clipped[m].sum())
                tq_peak_applied[g] = max(tq_peak_applied[g], float(np.max(np.abs(applied[m]))))
                tq_peak_demand[g] = max(tq_peak_demand[g], float(np.max(np.abs(computed[m]))))
                vel_over[g] += int(over[m].sum())
            phys_samples += 1
            sim.step(applied)

        k_done = k + 1
        t_end = (k + 1) * control_dt
        q = sim.q()
        margin = np.minimum(q - limit_lo, limit_hi - q)
        if float(np.min(margin)) < -JOINT_LIMIT_TOL_RAD:
            violation_steps += 1
        for g, m in g_mask.items():
            mg = float(np.min(margin[m]))
            min_margin[g] = min(min_margin[g], mg)
            if mg < near_margin:
                near[g] += 1
        roll, pitch = roll_pitch_from_quat_wxyz(sim.quat())
        max_rp = max(max_rp, abs(roll), abs(pitch))
        h = float(sim.data.qpos[2])
        heights.append(h)
        v = sim.lin_vel_body()
        times.append(t_end)
        cmds.append(cmd)
        vxy.append(v[:2])
        wz.append(float(sim.gyro_body()[2]))
        if opts.trace is not None:
            opts.trace.append({"t": t_end, "cmd": list(cmd), "q": q.tolist(), "target": target_sim.tolist(),
                               "raw": raw.tolist(), "base_height": h, "wz": wz[-1]})
        if video is not None and k % frame_every == 0:
            video.capture(sim.data)
        reason = None
        if h < float(fall["base_height_min_m"]):
            reason = "base_height"
        elif max(abs(roll), abs(pitch)) > float(fall["max_abs_roll_pitch_rad"]):
            reason = "roll_pitch"
        if reason:
            fell, fall_reason, t_fall = True, reason, t_end
            break

    trk = tracking_rmse(
        np.asarray(times), np.asarray(cmds, dtype=np.float64).reshape(-1, 3),
        np.asarray(vxy).reshape(-1, 2), np.asarray(wz), scenario.change_times(),
        control_hz=spec.control_hz,
        smoothing_window_s=float(cfg.tracking["smoothing_window_s"]),
        exclude_after_change_s=float(cfg.tracking["exclude_after_command_change_s"]),
    )
    n_group_phys = {g: phys_samples * int(g_mask[g].sum()) for g in JOINT_GROUPS}
    eff = {g: float(cfg.actuator[g]["effort_limit"]) for g in JOINT_GROUPS}
    metrics = {
        "fell": fell,
        "fall_reason": fall_reason,
        "time_to_fall_s": t_fall,
        "nonfinite_action": nonfinite,
        "simulated_s": round(k_done * control_dt, 6),
        "init_base_height_m": init_h,
        "lin_vel_rmse_mps": trk["lin_vel_rmse_mps"],
        "yaw_rate_rmse_radps": trk["yaw_rate_rmse_radps"],
        "tracking_samples": trk["n_samples"],
        "tracked_mean_vx": trk["mean_vx"],
        "tracked_mean_vy": trk["mean_vy"],
        "tracked_mean_wz": trk["mean_wz"],
        "mean_base_height_m": float(np.mean(heights)) if heights else None,
        "min_base_height_m": float(np.min(heights)) if heights else None,
        "planar_displacement_m": float(np.linalg.norm(sim.data.qpos[0:2] - start_xy)),
        "max_abs_roll_pitch_rad": max_rp,
        # None = no deploy clip under a gate whose saturation.missing_clip is "fail".
        "pre_clip_saturation_rate": (sat_pre / n_act if n_act and not sat_undefined else None),
        "pre_clip_saturation_definition": (
            f"|raw| {'>=' if cfg.saturation_ge else '>'} deploy clip {spec.action_clip}"),
        "abs_raw_action_gt1_rate": raw_gt1 / n_act if n_act else None,
        "max_abs_raw_action": raw_max,
        "torque_saturation_fraction": {
            g: (tq_sat[g] / n_group_phys[g] if n_group_phys[g] else None) for g in JOINT_GROUPS},
        "peak_applied_torque_over_limit": {g: tq_peak_applied[g] / eff[g] for g in JOINT_GROUPS},
        "peak_demanded_torque_over_limit": {g: tq_peak_demand[g] / eff[g] for g in JOINT_GROUPS},
        "joint_vel_over_limit_fraction": {
            g: (vel_over[g] / n_group_phys[g] if n_group_phys[g] else None) for g in JOINT_GROUPS},
        "hard_limit_violation_steps": violation_steps,
        "min_limit_margin_rad": {g: (None if math.isinf(min_margin[g]) else min_margin[g]) for g in JOINT_GROUPS},
        "near_limit_fraction": {g: (near[g] / k_done if k_done else None) for g in JOINT_GROUPS},
    }
    out: dict[str, Any] = {"scenario": scenario.to_dict(), "metrics": metrics,
                           "latency_physics_steps": lag, "model": sim.model_info}
    if video is not None:
        path = Path(opts.video_dir) / f"{scenario.name}.mp4"  # type: ignore[arg-type]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            video.write(path)
            out["video"] = str(path)
        except Exception as exc:  # noqa: BLE001
            out["video"] = None
            out["video_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            video.close()
    elif opts.video_dir is not None:
        out["video"] = None
        out["video_error"] = video_error
    return out


def _git_sha_of(path: str) -> str | None:
    try:
        r = subprocess.run(["git", "log", "-1", "--format=%H", "--", path], capture_output=True, text=True,
                           cwd=Path(path).resolve().parent, check=False, timeout=10)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def run_gate(
    spec: DeploySpec,
    policy: Callable[[np.ndarray], np.ndarray],
    cfg: GateConfig,
    *,
    policy_info: Mapping[str, Any],
    latency_ms: float | None = None,
    video_dir: str | Path | None = None,
    scenario_names: list[str] | None = None,
    diagnostic_reasons: list[str] | None = None,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    lat = float(cfg.physics["latency_ms"]) if latency_ms is None else float(latency_ms)
    diag = list(diagnostic_reasons or [])
    if abs(lat - float(cfg.physics["latency_ms"])) > 1e-12:
        diag.append(f"latency_ms {lat} != gate {cfg.physics['latency_ms']}")
    scenarios = list(cfg.scenarios) if not scenario_names else [cfg.scenario(n) for n in scenario_names]
    if scenario_names and set(scenario_names) != {s.name for s in cfg.scenarios}:
        diag.append("scenario subset")
    spec_checks = [{"check": "manifest_valid", "value": list(spec.manifest_problems), "op": "==",
                    "threshold": [], "pass": not spec.manifest_problems}]
    spec_checks += envelope_checks(spec, cfg.required_envelope)
    from .joint_audit import sign_audit

    audit = sign_audit(spec, physics_hz=int(cfg.physics["physics_hz"]))
    bad = [j["joint"] for j in audit["joints"] if not j["pass"]]
    spec_checks.append({"check": "joint_sign_audit", "value": bad, "op": "==", "threshold": [],
                        "pass": audit["pass"]})
    tag_spec_checks(spec_checks, cfg)
    cfg_sha = hashlib.sha256(Path(cfg.path).read_bytes()).hexdigest()
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "gate": {"config": cfg.path, "config_sha256": cfg_sha, "config_commit": _git_sha_of(cfg.path),
                 "name": cfg.raw.get("name"), "version": cfg.version},
        "policy": dict(policy_info),
        "spec": spec.summary(),
        "run": {"latency_ms": lat, "physics_hz": int(cfg.physics["physics_hz"]),
                "profile": cfg.physics["profile"], "actuator": cfg.actuator,
                "mujoco_gl": os.environ.get("MUJOCO_GL")},
        "spec_checks": spec_checks,
        "joint_audit": audit,
        "scenarios": {},
    }
    opts = RunOptions(latency_ms=lat, video_dir=None if video_dir is None else Path(video_dir))
    for s in scenarios:
        r = run_gate_scenario(spec, policy, s, cfg, opts)
        r["checks"] = evaluate_scenario(r["metrics"], s, cfg)
        # Scenario pass = its BLOCKING checks pass; performance is reported separately.
        r["pass"] = all(c["pass"] for c in r["checks"] if c["tier"] != "performance")
        r["performance_pass"] = all(c["pass"] for c in r["checks"] if c["tier"] == "performance")
        report["scenarios"][s.name] = r
        if progress:
            progress(s.name, r)
    v = gate_verdict(report)
    report["would_be_verdict"] = v["verdict"]
    report["failures"] = v["failures"]
    report["performance_failures"] = v["performance_failures"]
    if cfg.version >= 2:
        report["tiers"] = {
            "safety": {"blocking": True, "verdict": v["verdict"], "failures": v["failures"]},
            "performance": {"blocking": False, **performance_table(report, cfg)},
        }
    report["diagnostic_reasons"] = diag
    report["verdict"] = "DIAGNOSTIC" if diag else v["verdict"]
    return report


def write_gate_report(report: Mapping[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=float) + "\n")
    return p


__all__ = [
    "GateSim",
    "JOINT_LIMIT_TOL_RAD",
    "OnnxPolicy",
    "RunOptions",
    "pd_torque",
    "real_motor_params",
    "run_gate",
    "run_gate_scenario",
    "sha256_file",
    "spec_permutation",
    "write_gate_report",
]
