"""Headless MuJoCo runner for PhoenixVelocity policies (sim-to-sim validation).

Loop, matching Isaac Lab ``ManagerBasedRLEnv.step`` with decimation 4:

1. observation from the current state via
   :func:`phoenix.velocity.observation.build_actor_observation` (the function the
   hardware node uses); gyro = MuJoCo free-joint ``qvel[3:6]``, which is the
   angular velocity in the BODY frame (verified numerically in
   ``tests/test_sim2sim_gyro_frame.py``), quaternion = ``qpos[3:7]`` (w, x, y, z);
2. policy -> raw action; clipped to ``[-action_clip, action_clip]`` as the
   training wrapper does (``RslRlVecEnvWrapper(env, clip_actions=1.0)`` in
   ``phoenix.training.ppo_runner``); ``last_action`` is that clipped action,
   which is what Isaac's action manager stores;
3. targets = :func:`phoenix.velocity.observation.actions_to_joint_targets`;
4. four physics steps of 0.005 s, each computing the Isaac Lab ``DCMotor`` torque
   (:mod:`phoenix.sim2sim.dc_motor`) from the CURRENT joint state, optionally
   through a physics-step target delay, written to MuJoCo ``motor`` actuators.

MuJoCo and the ONNX / torch runtimes are imported lazily.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from phoenix.sim2real.go2_model import limits_in_order
from phoenix.velocity.contract import (
    ACTION_DIM,
    ACTION_SCALE,
    ACTOR_OBS_DIM,
    CONTROL_HZ,
    DECIMATION,
    DEFAULT_JOINT_POS,
    JOINT_ORDER,
    MANIFEST_NAME,
    MODE_VELOCITY,
    validate_manifest_for_mode,
)
from phoenix.velocity.observation import (
    actions_to_joint_targets,
    build_actor_observation,
    projected_gravity_wxyz,
    rotation_matrix_wxyz,
    tilt_from_projected_gravity,
)

from .dc_motor import GO2_DC_MOTOR, DCMotorParams, TargetDelay, dc_motor_torque
from .isaac_reference import ISAAC_JOINT_MAX_VEL_RAD_S
from .model import PROFILE_ISAAC, load_go2_model
from .scenarios import (
    COMMAND_KEYS,
    FALL_BASE_HEIGHT_M,
    FALL_TILT_RAD,
    INIT_BASE_HEIGHT_M,
    SETTLE_S,
    Scenario,
    empty_report,
    envelope_problems,
)

CONTROL_DT = 1.0 / CONTROL_HZ
#: A joint is counted as beyond its hard limit only past this tolerance: MuJoCo
#: joint limits are soft constraints and allow sub-millimetre-scale penetration.
JOINT_LIMIT_TOL_RAD = 0.005
_DEFAULT_Q = np.asarray([DEFAULT_JOINT_POS[j] for j in JOINT_ORDER], dtype=np.float64)
_HARD_LO, _HARD_HI = limits_in_order(JOINT_ORDER)
_ISAAC_VEL_CLAMP = np.asarray([ISAAC_JOINT_MAX_VEL_RAD_S[j] for j in JOINT_ORDER])


class PolicyRefusedError(RuntimeError):
    """The checkpoint may not run in velocity mode and no diagnostic override was given."""


# --------------------------------------------------------------------------- policies


class Policy(Protocol):
    def reset(self) -> None: ...

    def __call__(self, obs: np.ndarray) -> np.ndarray: ...


class CallablePolicy:
    """Wrap ``fn(obs45) -> action12`` (tests, custom controllers)."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], name: str = "callable") -> None:
        self._fn = fn
        self.name = name

    def reset(self) -> None:
        return None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self._fn(obs), dtype=np.float64).reshape(-1)


class ZeroActionPolicy(CallablePolicy):
    """Action 0 forever: the DC-motor PD holds the default pose. Plumbing / physics check."""

    def __init__(self) -> None:
        super().__init__(lambda _obs: np.zeros(ACTION_DIM), name="builtin:zero")


class RandomBoundedPolicy:
    """Low-pass filtered uniform noise in ``[-amplitude, amplitude]``, seeded. Plumbing only."""

    def __init__(self, amplitude: float = 0.3, seed: int = 0, alpha: float = 0.2) -> None:
        self.amplitude = float(amplitude)
        self.seed = int(seed)
        self.alpha = float(alpha)
        self.name = f"builtin:random(amp={self.amplitude},seed={self.seed},alpha={self.alpha})"
        self.reset()

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._a = np.zeros(ACTION_DIM)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        noise = self._rng.uniform(-self.amplitude, self.amplitude, size=ACTION_DIM)
        self._a = (1.0 - self.alpha) * self._a + self.alpha * noise
        return np.clip(self._a, -self.amplitude, self.amplitude)


class OnnxPolicy:
    """ONNX actor as exported by ``phoenix.sim2real.export`` (input ``obs``, output ``action``)."""

    def __init__(self, path: str | Path) -> None:
        import onnxruntime as ort

        self.path = Path(path)
        self.name = f"onnx:{self.path.name}"
        self._sess = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        inputs = self._sess.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"{self.path}: expected one input, found {len(inputs)}")
        self._in = inputs[0].name
        shape = inputs[0].shape
        if len(shape) != 2 or (isinstance(shape[1], int) and shape[1] != ACTOR_OBS_DIM):
            raise ValueError(f"{self.path}: input shape {shape} is not [batch, {ACTOR_OBS_DIM}]")
        outs = [o.name for o in self._sess.get_outputs()]
        self._out = "action" if "action" in outs else outs[0]

    def reset(self) -> None:
        return None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(1, ACTOR_OBS_DIM)
        (y,) = self._sess.run([self._out], {self._in: x})
        return np.asarray(y, dtype=np.float64).reshape(-1)


class TorchScriptPolicy:
    """TorchScript actor (the ``.pt`` fallback ``phoenix.sim2real.export`` writes)."""

    def __init__(self, path: str | Path) -> None:
        import torch

        self.path = Path(path)
        self.name = f"torchscript:{self.path.name}"
        try:
            self._mod = torch.jit.load(str(self.path), map_location="cpu").eval()
        except RuntimeError as exc:
            raise ValueError(
                f"{self.path} is not TorchScript. A raw rsl_rl training checkpoint must be "
                "exported first: python -m phoenix.sim2real.export --checkpoint <model.pt> "
                "--output <policy.onnx> --verify (this also normalizes observations the way "
                "training did)."
            ) from exc
        self._torch = torch

    def reset(self) -> None:
        return None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            x = self._torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1))
            y = self._mod(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
        return y.detach().cpu().numpy().astype(np.float64).reshape(-1)


def load_policy(path: str | Path) -> Policy:
    p = Path(path)
    if p.suffix == ".onnx":
        return OnnxPolicy(p)
    if p.suffix in (".pt", ".jit", ".ts"):
        return TorchScriptPolicy(p)
    raise ValueError(f"unsupported policy file {p} (expected .onnx or TorchScript .pt)")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- manifest gate


def find_manifest(policy_path: str | Path) -> Path | None:
    p = Path(policy_path).resolve()
    for d in (p.parent, p.parent.parent):
        cand = d / MANIFEST_NAME
        if cand.is_file():
            return cand
    return None


@dataclass
class GateResult:
    diagnostic: bool
    problems: list[str]
    manifest_path: str | None
    trained_commands: dict[str, Any] | None
    skipped: dict[str, list[str]] = field(default_factory=dict)
    runnable: list[Scenario] = field(default_factory=list)


def gate_checkpoint(
    manifest: Mapping[str, Any] | None,
    scenarios: Sequence[Scenario],
    *,
    allow_unvalidated_for_diagnosis: bool,
    checkpoint_sha256: str | None = None,
    manifest_path: str | None = None,
) -> GateResult:
    """Apply the contract manifest gate to a checkpoint for these scenarios.

    Scenarios whose commands leave the trained envelope are skipped (with the
    reason). The rest are checked with :func:`validate_manifest_for_mode` in
    velocity mode against their largest command. Any problem refuses the run
    unless ``allow_unvalidated_for_diagnosis``, in which case the result is
    DIAGNOSTIC.
    """
    trained = None
    if isinstance(manifest, Mapping) and isinstance(manifest.get("commands"), Mapping):
        trained = dict(manifest["commands"])
    skipped: dict[str, list[str]] = {}
    runnable: list[Scenario] = []
    for s in scenarios:
        try:
            probs = envelope_problems(s, trained)
        except (KeyError, TypeError, ValueError) as exc:
            probs = [f"trained command ranges unusable: {exc!r}"]
        if probs:
            skipped[s.name] = probs
        else:
            runnable.append(s)
    envelope = {k: 0.0 for k in COMMAND_KEYS}
    for s in runnable:
        for k, v in s.max_abs_command().items():
            envelope[k] = max(envelope[k], v)
    problems = validate_manifest_for_mode(
        manifest,
        MODE_VELOCITY,
        deploy_action_scale=ACTION_SCALE,
        deploy_control_hz=CONTROL_HZ,
        deploy_joint_order=JOINT_ORDER,
        max_deploy_command=envelope,
        checkpoint_sha256=checkpoint_sha256,
    )
    if problems and not allow_unvalidated_for_diagnosis:
        raise PolicyRefusedError(
            "checkpoint refused for velocity-mode sim2sim (pass "
            "--allow-unvalidated-for-diagnosis to run it as DIAGNOSTIC):\n  - "
            + "\n  - ".join(problems)
        )
    if problems:
        # Diagnostic runs still exercise every scenario, envelope or not.
        runnable = list(scenarios)
        skipped = {}
    return GateResult(
        diagnostic=bool(problems),
        problems=list(problems),
        manifest_path=manifest_path,
        trained_commands=trained,
        skipped=skipped,
        runnable=runnable,
    )


# --------------------------------------------------------------------------- simulator


def yaw_from_quat_wxyz(q: Sequence[float]) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class MujocoGo2:
    """One GO2 in MuJoCo with the Isaac Lab DCMotor actuator model in Python."""

    def __init__(
        self,
        *,
        profile: str = PROFILE_ISAAC,
        foot_friction: float | None = None,
        scene_xml: str | Path | None = None,
        motor: DCMotorParams = GO2_DC_MOTOR,
    ) -> None:
        import mujoco

        self._mj = mujoco
        self.model, self.idx, self.model_info = load_go2_model(
            profile, scene_xml=scene_xml, foot_friction=foot_friction
        )
        self.data = mujoco.MjData(self.model)
        self.motor = motor
        self.reset()

    # ---- state
    def reset(self, base_height: float = INIT_BASE_HEIGHT_M) -> None:
        mj = self._mj
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = (0.0, 0.0, base_height)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.idx.qpos_adr] = _DEFAULT_Q
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mj.mj_forward(self.model, self.data)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.idx.qpos_adr].copy()

    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self.idx.dof_adr].copy()

    def base_quat_wxyz(self) -> np.ndarray:
        return self.data.qpos[3:7].copy()

    def gyro_body(self) -> np.ndarray:
        """Body-frame angular velocity: MuJoCo free-joint ``qvel[3:6]`` is expressed locally."""
        return self.data.qvel[3:6].copy()

    def lin_vel_body(self) -> np.ndarray:
        """Base-origin linear velocity in the body frame (free-joint ``qvel[0:3]`` is world)."""
        r = rotation_matrix_wxyz(self.base_quat_wxyz())
        return r.T @ self.data.qvel[0:3]

    def base_height(self) -> float:
        return float(self.data.qpos[2])

    def tilt(self) -> float:
        return tilt_from_projected_gravity(projected_gravity_wxyz(self.base_quat_wxyz()))

    def trunk_contact(self) -> bool:
        trunk = set(self.idx.trunk_geom_ids)
        floor = self.idx.floor_geom_id
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 == floor and g2 in trunk) or (g2 == floor and g1 in trunk):
                return True
        return False

    def observation(self, command: Sequence[float], last_action: np.ndarray) -> np.ndarray:
        return build_actor_observation(
            gyro_body=self.gyro_body(),
            quat_wxyz=self.base_quat_wxyz(),
            command=command,
            joint_pos=self.joint_pos(),
            joint_vel=self.joint_vel(),
            last_action=last_action,
        )

    # ---- actuation
    def physics_step(self, q_des: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One 0.005 s step with DCMotor torques from the current state. Returns torques."""
        computed, applied = dc_motor_torque(q_des, self.joint_pos(), self.joint_vel(), self.motor)
        self.data.ctrl[self.idx.actuator_id] = applied
        self._mj.mj_step(self.model, self.data)
        return computed, applied

    def push(self, delta_vel_world: Sequence[float]) -> None:
        self.data.qvel[0:3] += np.asarray(delta_vel_world, dtype=np.float64)


# --------------------------------------------------------------------------- scenario run


@dataclass
class RunConfig:
    profile: str = PROFILE_ISAAC
    action_scale: float = ACTION_SCALE
    #: None disables clipping (mirrors a deploy path that does not clip).
    action_clip: float | None = 1.0
    scene_xml: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "action_scale": self.action_scale,
            "action_clip": self.action_clip,
            "physics_dt": 1.0 / (CONTROL_HZ * DECIMATION),
            "control_hz": CONTROL_HZ,
            "decimation": DECIMATION,
            "joint_limit_tol_rad": JOINT_LIMIT_TOL_RAD,
        }


def _rms(x: list[float]) -> float | None:
    return float(np.sqrt(np.mean(np.square(x)))) if x else None


def _mean(x: list[float]) -> float | None:
    return float(np.mean(x)) if x else None


def run_scenario(
    policy: Policy, scenario: Scenario, cfg: RunConfig | None = None
) -> dict[str, Any]:
    """Run one scenario from the nominal init state. Returns ``{"metrics": ..., ...}``."""
    cfg = cfg or RunConfig()
    sim = MujocoGo2(
        profile=cfg.profile, foot_friction=scenario.foot_friction, scene_xml=cfg.scene_xml
    )
    policy.reset()
    delay = TargetDelay(scenario.latency_physics_steps)
    n_ctrl = int(round(scenario.duration_s * CONTROL_HZ))
    last_action = np.zeros(ACTION_DIM)
    prev_action: np.ndarray | None = None
    pushed = False

    lin_err: list[float] = []
    yaw_err: list[float] = []
    tilts: list[float] = []
    heights: list[float] = []
    a_abs: list[float] = []
    a_max = 0.0
    rate_abs: list[float] = []
    rate_max = 0.0
    sat_count = 0
    torque_abs_sum = 0.0
    phys_samples = 0
    vel_clamp_exceed = 0
    limit_steps = 0
    max_excursion = 0.0
    cmd_fwd = ach_fwd = cmd_lat = ach_lat = cmd_yaw = ach_yaw = 0.0
    prev_yaw = yaw_from_quat_wxyz(sim.base_quat_wxyz())
    fell = False
    fall_reason: str | None = None
    time_to_fall: float | None = None
    nonfinite_action = False

    for k in range(n_ctrl):
        t = k * CONTROL_DT
        cmd = scenario.command_at(t)
        obs = sim.observation(cmd, last_action)
        raw = np.asarray(policy(obs), dtype=np.float64).reshape(-1)
        if raw.shape != (ACTION_DIM,):
            raise ValueError(f"policy returned shape {raw.shape}, expected ({ACTION_DIM},)")
        if not np.all(np.isfinite(raw)):
            nonfinite_action = True
            fell, fall_reason, time_to_fall = True, "nonfinite_action", t
            break
        action = raw if cfg.action_clip is None else np.clip(raw, -cfg.action_clip, cfg.action_clip)
        targets = actions_to_joint_targets(action, cfg.action_scale)
        a_abs.append(float(np.mean(np.abs(action))))
        a_max = max(a_max, float(np.max(np.abs(action))))
        if prev_action is not None:
            d = np.abs(action - prev_action)
            rate_abs.append(float(np.mean(d)))
            rate_max = max(rate_max, float(np.max(d)))
        prev_action = action
        last_action = action

        trunk_hit = False
        for _ in range(DECIMATION):
            if scenario.push is not None and not pushed and sim.time + 1e-9 >= scenario.push.t_s:
                sim.push(scenario.push.delta_vel_world)
                pushed = True
            computed, applied = sim.physics_step(delay(targets))
            sat_count += int(np.sum(np.abs(computed - applied) > 1e-9))
            torque_abs_sum += float(np.sum(np.abs(applied)))
            phys_samples += ACTION_DIM
            vel_clamp_exceed += int(np.sum(np.abs(sim.joint_vel()) > _ISAAC_VEL_CLAMP))
            trunk_hit = trunk_hit or sim.trunk_contact()

        t_end = (k + 1) * CONTROL_DT
        q = sim.joint_pos()
        exc = np.maximum(np.maximum(_HARD_LO - q, q - _HARD_HI), 0.0)
        max_excursion = max(max_excursion, float(np.max(exc)))
        if float(np.max(exc)) > JOINT_LIMIT_TOL_RAD:
            limit_steps += 1
        tilt = sim.tilt()
        h = sim.base_height()
        tilts.append(tilt)
        heights.append(h)
        yaw = yaw_from_quat_wxyz(sim.base_quat_wxyz())
        dyaw = math.atan2(math.sin(yaw - prev_yaw), math.cos(yaw - prev_yaw))
        prev_yaw = yaw
        if t >= SETTLE_S:
            v = sim.lin_vel_body()
            w = sim.gyro_body()
            lin_err.append(float(np.hypot(v[0] - cmd[0], v[1] - cmd[1])))
            yaw_err.append(abs(float(w[2]) - cmd[2]))
            cmd_fwd += cmd[0] * CONTROL_DT
            ach_fwd += float(v[0]) * CONTROL_DT
            cmd_lat += cmd[1] * CONTROL_DT
            ach_lat += float(v[1]) * CONTROL_DT
            cmd_yaw += cmd[2] * CONTROL_DT
            ach_yaw += dyaw

        reason = None
        if tilt > FALL_TILT_RAD:
            reason = "tilt"
        elif h < FALL_BASE_HEIGHT_M:
            reason = "base_height"
        elif trunk_hit:
            reason = "trunk_contact"
        if reason is not None:
            fell, fall_reason, time_to_fall = True, reason, t_end
            break

    metrics = {
        "fell": fell,
        "fall_reason": fall_reason,
        "time_to_fall_s": time_to_fall,
        "duration_s": round(sim.time, 6),
        "lin_vel_err_mean": _mean(lin_err),
        "lin_vel_err_rms": _rms(lin_err),
        "yaw_rate_err_mean": _mean(yaw_err),
        "yaw_rate_err_rms": _rms(yaw_err),
        "max_tilt_rad": max(tilts) if tilts else None,
        "mean_base_height_m": _mean(heights),
        "min_base_height_m": min(heights) if heights else None,
        "joint_limit_violation_steps": limit_steps,
        "max_joint_limit_excursion_rad": max_excursion,
        "torque_saturation_fraction": sat_count / phys_samples if phys_samples else None,
        "mean_abs_torque_nm": torque_abs_sum / phys_samples if phys_samples else None,
        "action_abs_mean": _mean(a_abs),
        "action_abs_max": a_max,
        "action_rate_abs_mean": _mean(rate_abs),
        "action_rate_abs_max": rate_max,
        "commanded_forward_m": cmd_fwd,
        "achieved_forward_m": ach_fwd,
        "commanded_lateral_m": cmd_lat,
        "achieved_lateral_m": ach_lat,
        "commanded_yaw_rad": cmd_yaw,
        "achieved_yaw_rad": ach_yaw,
    }
    return {
        "skipped": False,
        "scenario": scenario.to_dict(),
        "metrics": metrics,
        "mujoco_only": {
            "isaac_vel_clamp_exceed_fraction": (
                vel_clamp_exceed / phys_samples if phys_samples else None
            ),
            "nonfinite_action": nonfinite_action,
            "model": sim.model_info,
        },
    }


def run_suite(
    policy: Policy,
    scenarios: Iterable[Scenario],
    cfg: RunConfig | None = None,
    *,
    diagnostic: bool,
    policy_info: Mapping[str, Any] | None = None,
    skipped: Mapping[str, list[str]] | None = None,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    cfg = cfg or RunConfig()
    report = empty_report("mujoco", diagnostic=diagnostic)
    report["label"] = "DIAGNOSTIC" if diagnostic else "VALIDATED-CHECKPOINT"
    report["policy"] = dict(policy_info or {})
    report["config"].update(cfg.to_dict())
    for name, reasons in (skipped or {}).items():
        report["scenarios"][name] = {"skipped": True, "skip_reason": "; ".join(reasons)}
    for s in scenarios:
        result = run_scenario(policy, s, cfg)
        report["scenarios"][s.name] = result
        if progress is not None:
            progress(s.name, result)
    return report


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "YES" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def summary_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# MuJoCo sim2sim report ({report.get('label', '')})",
        "",
        f"- policy: `{report['policy'].get('name', '?')}`",
        f"- profile: `{report['config'].get('profile')}`, action clip "
        f"{report['config'].get('action_clip')}, scale {report['config'].get('action_scale')}",
    ]
    if report.get("diagnostic"):
        lines.append(
            "- **DIAGNOSTIC**: this policy did not pass the manifest gate; nothing here "
            "qualifies it for hardware."
        )
    lines += [
        "",
        "| scenario | fell | t_fall s | lin err rms | yaw err rms | max tilt | mean h | "
        "sat frac | fwd cmd/ach m | yaw cmd/ach rad |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in report["scenarios"].items():
        if r.get("skipped"):
            lines.append(f"| {name} | skipped: {r.get('skip_reason')} |||||||||")
            continue
        m = r["metrics"]
        lines.append(
            f"| {name} | {_fmt(m['fell'])} {m['fall_reason'] or ''} | {_fmt(m['time_to_fall_s'], 2)}"
            f" | {_fmt(m['lin_vel_err_rms'])} | {_fmt(m['yaw_rate_err_rms'])}"
            f" | {_fmt(m['max_tilt_rad'])} | {_fmt(m['mean_base_height_m'])}"
            f" | {_fmt(m['torque_saturation_fraction'])}"
            f" | {_fmt(m['commanded_forward_m'], 2)}/{_fmt(m['achieved_forward_m'], 2)}"
            f" | {_fmt(m['commanded_yaw_rad'], 2)}/{_fmt(m['achieved_yaw_rad'], 2)} |"
        )
    return "\n".join(lines) + "\n"


def write_report(report: Mapping[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "mujoco_report.json"
    mp = out / "mujoco_summary.md"
    jp.write_text(json.dumps(report, indent=2, sort_keys=False, default=float) + "\n")
    mp.write_text(summary_markdown(report))
    return jp, mp


__all__ = [
    "CallablePolicy",
    "GateResult",
    "JOINT_LIMIT_TOL_RAD",
    "MujocoGo2",
    "OnnxPolicy",
    "Policy",
    "PolicyRefusedError",
    "RandomBoundedPolicy",
    "RunConfig",
    "TorchScriptPolicy",
    "ZeroActionPolicy",
    "find_manifest",
    "gate_checkpoint",
    "load_policy",
    "run_scenario",
    "run_suite",
    "sha256_file",
    "summary_markdown",
    "write_report",
    "yaw_from_quat_wxyz",
]
