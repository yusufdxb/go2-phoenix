"""Behavioral locomotion metrics for one episode, raw policy and post-safety kept apart.

An :class:`EpisodeTrace` is the per-control-step record of one episode. It is
filled identically by the Isaac Lab evaluator and by the hardware forensics
script, so sim and robot are scored by the same code.

Two views are never collapsed:

* **raw policy**: what the network asked for (``raw_action``,
  ``requested_target = default + scale * raw_action``);
* **executed**: what reached the actuators after every safety layer
  (``executed_target``: the action clamp in sim, the policy-node and bridge
  clips on hardware).

Their difference is the modification ("intervention") rate. A policy whose
executed behavior looks fine only because a safety layer rewrote most of its
commands is reported as such, not as a good policy.

Pure numpy, CI-safe.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from phoenix.sim2real.safety import MAX_DELTA_PER_STEP_RAD

from .thresholds import ILLEGAL_TARGET_BAND_RAD

#: Targets that differ by less than this are "unmodified" (float32 round trip).
MODIFICATION_EPS_RAD = 1e-5
#: Training action clamp (RslRlVecEnvWrapper clip_actions=1.0).
TRAINING_ACTION_CLIP = 1.0


def _arr(value: Any, name: str, shape_tail: tuple[int, ...], n: int) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (n, *shape_tail):
        raise ValueError(f"{name} must have shape {(n, *shape_tail)}, got {a.shape}")
    return a


@dataclass
class EpisodeTrace:
    """Per-control-step signals of ONE episode. Arrays are (T, ...) with T >= 1.

    ``measured_q[t]`` is the joint position the step-``t`` command was computed
    against (read before the step). Optional signals are ``None`` when the source
    cannot observe them; the metric is then reported as ``None``, never as 0.
    """

    dt_s: float
    command: np.ndarray  # (T, 3) vx, vy (m/s), wz (rad/s), body frame
    lin_vel_b: np.ndarray | None  # (T, 2) achieved body planar velocity; None: unmeasured
    yaw_rate: np.ndarray | None  # (T,) achieved body yaw rate, rad/s
    tilt_rad: np.ndarray  # (T,) angle of body z from world up
    roll_rad: np.ndarray | None = None
    pitch_rad: np.ndarray | None = None
    raw_action: np.ndarray | None = None  # (T, J) dimensionless
    requested_target: np.ndarray | None = None  # (T, J) rad, policy request
    executed_target: np.ndarray | None = None  # (T, J) rad, post-safety
    measured_q: np.ndarray | None = None  # (T, J) rad
    joint_lower: np.ndarray | None = None  # (J,) hard limits, rad
    joint_upper: np.ndarray | None = None
    undesired_contact: np.ndarray | None = None  # (T,) bool
    planned_steps: int | None = None
    joint_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.dt_s) and self.dt_s > 0):
            raise ValueError("dt_s must be finite and positive")
        n = len(np.asarray(self.command))
        if n < 1:
            raise ValueError("an episode trace needs at least one step")
        self.command = _arr(self.command, "command", (3,), n)
        if self.lin_vel_b is not None:
            self.lin_vel_b = _arr(self.lin_vel_b, "lin_vel_b", (2,), n)
        if self.yaw_rate is not None:
            self.yaw_rate = _arr(self.yaw_rate, "yaw_rate", (), n)
        self.tilt_rad = _arr(self.tilt_rad, "tilt_rad", (), n)
        for name in ("roll_rad", "pitch_rad"):
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, _arr(v, name, (), n))
        j = None
        for name in ("raw_action", "requested_target", "executed_target", "measured_q"):
            v = getattr(self, name)
            if v is not None:
                v = np.asarray(v, dtype=np.float64)
                if v.ndim != 2 or v.shape[0] != n:
                    raise ValueError(f"{name} must be (T, J), got {v.shape}")
                if j is not None and v.shape[1] != j:
                    raise ValueError(f"{name} joint count {v.shape[1]} != {j}")
                j = v.shape[1]
                setattr(self, name, v)
        for name in ("joint_lower", "joint_upper"):
            v = getattr(self, name)
            if v is not None:
                v = np.asarray(v, dtype=np.float64)
                if j is not None and v.shape != (j,):
                    raise ValueError(f"{name} must be ({j},), got {v.shape}")
                setattr(self, name, v)
        if self.undesired_contact is not None:
            self.undesired_contact = np.asarray(self.undesired_contact, dtype=bool).reshape(n)

    @property
    def n_steps(self) -> int:
        return int(self.command.shape[0])


@dataclass
class RawPolicyMetrics:
    max_abs_action: float | None
    action_saturation_rate: float | None
    requested_beyond_hard_limit: int | None
    illegal_target_count: int | None
    illegal_targets: list[dict[str, Any]]
    deploy_slew_activation_rate: float | None


@dataclass
class ExecutedMetrics:
    modification_rate: float | None  # joint-ticks
    modified_tick_rate: float | None  # ticks with any joint modified
    modification_mean_abs_rad: float | None
    modification_rms_rad: float | None
    modification_max_abs_rad: float | None
    executed_beyond_hard_limit: int | None
    tracking_deviation_rms_rad: float | None  # executed[t] - measured_q[t+1]
    tracking_deviation_max_abs_rad: float | None
    measured_beyond_hard_limit: int | None


@dataclass
class EpisodeMetrics:
    n_steps: int
    duration_s: float
    planned_steps: int | None
    non_finite: bool
    non_finite_fields: list[str]
    lin_vel_xy_rmse_mps: float | None
    lin_vel_xy_max_err_mps: float | None
    yaw_rate_rmse_radps: float | None
    yaw_rate_max_err_radps: float | None
    commanded_distance_m: float
    achieved_distance_along_command_m: float | None
    distance_ratio: float | None
    commanded_yaw_rad: float
    achieved_yaw_rad: float | None
    yaw_ratio: float | None
    max_tilt_rad: float
    max_abs_roll_rad: float | None
    max_abs_pitch_rad: float | None
    undesired_contact_steps: int | None
    feet_slip: None  # not measurable from the available signals; see failure_detector docs
    raw: RawPolicyMetrics
    executed: ExecutedMetrics

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nonfinite_fields(trace: EpisodeTrace) -> list[str]:
    bad = []
    for name in (
        "command",
        "lin_vel_b",
        "yaw_rate",
        "tilt_rad",
        "roll_rad",
        "pitch_rad",
        "raw_action",
        "requested_target",
        "executed_target",
        "measured_q",
    ):
        v = getattr(trace, name)
        if v is not None and not np.all(np.isfinite(v)):
            bad.append(name)
    return bad


def _nan_safe_max(a: np.ndarray) -> float:
    a = np.abs(a[np.isfinite(a)])
    return float(a.max()) if a.size else float("nan")


def _illegal(
    targets: np.ndarray, lower: np.ndarray, upper: np.ndarray, names: tuple[str, ...], band: float
) -> tuple[int, int, list[dict[str, Any]]]:
    below = lower - targets
    above = targets - upper
    beyond = np.maximum(below, above)  # >0 means outside the hard range
    beyond_any = int(np.sum(beyond > 0))
    illegal_mask = beyond > band
    events = []
    for t, j in zip(*np.nonzero(illegal_mask), strict=True):
        events.append(
            {
                "step": int(t),
                "joint": names[j] if j < len(names) else int(j),
                "target_rad": float(targets[t, j]),
                "limit_rad": float(lower[j] if below[t, j] > 0 else upper[j]),
                "beyond_by_rad": float(beyond[t, j]),
            }
        )
    return beyond_any, int(illegal_mask.sum()), events


def compute_episode_metrics(trace: EpisodeTrace) -> EpisodeMetrics:
    """Every behavioral metric of one episode. Never raises on NaN: it reports it."""
    n = trace.n_steps
    dt = trace.dt_s
    bad = _nonfinite_fields(trace)

    cmd_speed = np.linalg.norm(trace.command[:, :2], axis=-1)
    safe_speed = np.where(cmd_speed > 1e-9, cmd_speed, 1.0)
    cmd_dir = np.where(cmd_speed[:, None] > 1e-9, trace.command[:, :2] / safe_speed[:, None], 0.0)
    commanded_distance = float(np.sum(cmd_speed) * dt)
    commanded_yaw = float(np.sum(trace.command[:, 2]) * dt)
    lin_rmse = lin_max = achieved_along = distance_ratio = None
    if trace.lin_vel_b is not None:
        lin_err = np.linalg.norm(trace.command[:, :2] - trace.lin_vel_b, axis=-1)
        lin_rmse = float(np.sqrt(np.mean(lin_err**2)))
        lin_max = _nan_safe_max(lin_err)
        achieved_along = float(np.sum(np.sum(trace.lin_vel_b * cmd_dir, axis=-1)) * dt)
        if abs(commanded_distance) > 1e-9:
            distance_ratio = achieved_along / commanded_distance
    yaw_rmse = yaw_max = achieved_yaw = yaw_ratio = None
    if trace.yaw_rate is not None:
        yaw_err = np.abs(trace.command[:, 2] - trace.yaw_rate)
        yaw_rmse = float(np.sqrt(np.mean(yaw_err**2)))
        yaw_max = _nan_safe_max(yaw_err)
        achieved_yaw = float(np.sum(trace.yaw_rate) * dt)
        if abs(commanded_yaw) > 1e-9:
            yaw_ratio = achieved_yaw / commanded_yaw

    names = trace.joint_names
    lower, upper = trace.joint_lower, trace.joint_upper
    have_limits = lower is not None and upper is not None

    # ---- raw policy --------------------------------------------------------
    raw = trace.raw_action
    req = trace.requested_target
    beyond_req = illegal_n = None
    illegal_events: list[dict[str, Any]] = []
    if req is not None and have_limits:
        beyond_req, illegal_n, illegal_events = _illegal(
            req, lower, upper, names, ILLEGAL_TARGET_BAND_RAD
        )
    slew_rate = None
    if req is not None and trace.measured_q is not None:
        slew_rate = float(np.mean(np.abs(req - trace.measured_q) > MAX_DELTA_PER_STEP_RAD))
    raw_metrics = RawPolicyMetrics(
        max_abs_action=_nan_safe_max(raw) if raw is not None else None,
        action_saturation_rate=(
            float(np.mean(np.abs(raw) > TRAINING_ACTION_CLIP)) if raw is not None else None
        ),
        requested_beyond_hard_limit=beyond_req,
        illegal_target_count=illegal_n,
        illegal_targets=illegal_events[:50],
        deploy_slew_activation_rate=slew_rate,
    )

    # ---- executed (post-safety) -------------------------------------------
    exe = trace.executed_target
    mod_rate = mod_tick = mod_mean = mod_rms = mod_max = None
    if exe is not None and req is not None:
        diff = exe - req
        modified = np.abs(diff) > MODIFICATION_EPS_RAD
        mod_rate = float(np.mean(modified))
        mod_tick = float(np.mean(np.any(modified, axis=1)))
        mod_mean = float(np.mean(np.abs(diff)))
        mod_rms = float(np.sqrt(np.mean(diff**2)))
        mod_max = _nan_safe_max(diff)
    exe_beyond = meas_beyond = None
    if have_limits and exe is not None:
        exe_beyond = int(np.sum((exe < lower - 1e-9) | (exe > upper + 1e-9)))
    if have_limits and trace.measured_q is not None:
        q = trace.measured_q
        meas_beyond = int(np.sum((q < lower - 1e-9) | (q > upper + 1e-9)))
    dev_rms = dev_max = None
    if exe is not None and trace.measured_q is not None and n >= 2:
        dev = exe[:-1] - trace.measured_q[1:]
        dev_rms = float(np.sqrt(np.mean(dev**2)))
        dev_max = _nan_safe_max(dev)
    executed = ExecutedMetrics(
        modification_rate=mod_rate,
        modified_tick_rate=mod_tick,
        modification_mean_abs_rad=mod_mean,
        modification_rms_rad=mod_rms,
        modification_max_abs_rad=mod_max,
        executed_beyond_hard_limit=exe_beyond,
        tracking_deviation_rms_rad=dev_rms,
        tracking_deviation_max_abs_rad=dev_max,
        measured_beyond_hard_limit=meas_beyond,
    )

    return EpisodeMetrics(
        n_steps=n,
        duration_s=n * dt,
        planned_steps=trace.planned_steps,
        non_finite=bool(bad),
        non_finite_fields=bad,
        lin_vel_xy_rmse_mps=lin_rmse,
        lin_vel_xy_max_err_mps=lin_max,
        yaw_rate_rmse_radps=yaw_rmse,
        yaw_rate_max_err_radps=yaw_max,
        commanded_distance_m=commanded_distance,
        achieved_distance_along_command_m=achieved_along,
        distance_ratio=distance_ratio,
        commanded_yaw_rad=commanded_yaw,
        achieved_yaw_rad=achieved_yaw,
        yaw_ratio=yaw_ratio,
        max_tilt_rad=_nan_safe_max(trace.tilt_rad),
        max_abs_roll_rad=_nan_safe_max(trace.roll_rad) if trace.roll_rad is not None else None,
        max_abs_pitch_rad=(
            _nan_safe_max(trace.pitch_rad) if trace.pitch_rad is not None else None
        ),
        undesired_contact_steps=(
            int(np.sum(trace.undesired_contact)) if trace.undesired_contact is not None else None
        ),
        feet_slip=None,
        raw=raw_metrics,
        executed=executed,
    )


__all__ = [
    "EpisodeMetrics",
    "EpisodeTrace",
    "ExecutedMetrics",
    "MODIFICATION_EPS_RAD",
    "RawPolicyMetrics",
    "TRAINING_ACTION_CLIP",
    "compute_episode_metrics",
]
