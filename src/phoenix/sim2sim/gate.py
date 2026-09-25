"""Sim2sim PASS/FAIL gate: config, scenarios, metric math and verdict. Pure numpy + yaml.

The thresholds live in ``configs/sim2sim/gate_v1.yaml``, committed before any
policy was run through the gate. This module only reads them; nothing here may
carry a threshold of its own.

MuJoCo lives in :mod:`phoenix.sim2sim.gate_runner`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deploy_spec import COMMAND_KEYS, JOINT_GROUPS, DeploySpec

GATE_SCHEMA = "phoenix-sim2sim-gate/v1"
GATE_SCHEMA_V2 = "phoenix-sim2sim-gate/v2"
GATE_SCHEMA_V3 = "phoenix-sim2sim-gate/v3"
GATE_SCHEMAS = {GATE_SCHEMA: 1, GATE_SCHEMA_V2: 2, GATE_SCHEMA_V3: 3}
REPORT_SCHEMA = "phoenix-sim2sim-gate-report/v2"
_REPO_ROOT = Path(__file__).resolve().parents[3]
GATE_CONFIGS = {
    "v1": _REPO_ROOT / "configs" / "sim2sim" / "gate_v1.yaml",
    "v2": _REPO_ROOT / "configs" / "sim2sim" / "gate_v2.yaml",
    "v3": _REPO_ROOT / "configs" / "sim2sim" / "gate_v3.yaml",
}
DEFAULT_GATE_CONFIG = GATE_CONFIGS["v3"]

#: Command component -> tracked-mean metric in the scenario metrics.
_RESPONSE_METRIC = {"lin_vel_x": ("tracked_mean_vx", 0), "lin_vel_y": ("tracked_mean_vy", 1),
                    "ang_vel_z": ("tracked_mean_wz", 2)}

TIERS = ("nominal", "stress")
#: Check tiers. v1: every check is "gate" (blocking). v2: "safety" blocks,
#: "performance" is reported only.
CHECK_TIER_GATE = "gate"
CHECK_TIER_SAFETY = "safety"
CHECK_TIER_PERFORMANCE = "performance"
_TRACKING_CHECKS = ("lin_vel_rmse_mps", "yaw_rate_rmse_radps")


class GateConfigError(ValueError):
    pass


@dataclass(frozen=True)
class GateScenario:
    name: str
    tier: str
    duration_s: float
    segments: tuple[tuple[float, tuple[float, float, float]], ...]
    foot_friction: float | None = None
    payload_kg: float = 0.0
    stand_checks: bool = False

    def command_at(self, t: float) -> tuple[float, float, float]:
        cmd = self.segments[0][1]
        for start, c in self.segments:
            if t + 1e-9 >= start:
                cmd = c
            else:
                break
        return cmd

    def change_times(self) -> list[float]:
        return [s for s, _ in self.segments]

    def max_abs_command(self) -> dict[str, float]:
        out = {k: 0.0 for k in COMMAND_KEYS}
        for _t, c in self.segments:
            for k, v in zip(COMMAND_KEYS, c, strict=True):
                out[k] = max(out[k], abs(float(v)))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "duration_s": self.duration_s,
            "segments": [[t, list(c)] for t, c in self.segments],
            "foot_friction": self.foot_friction,
            "payload_kg": self.payload_kg,
            "stand_checks": self.stand_checks,
        }


def _scenario(d: Mapping[str, Any]) -> GateScenario:
    segs = tuple((float(t), tuple(float(x) for x in c)) for t, c in d["segments"])
    if not segs or segs[0][0] != 0.0:
        raise GateConfigError(f"{d.get('name')}: first segment must start at 0")
    times = [s[0] for s in segs]
    if times != sorted(set(times)) or times[-1] >= float(d["duration_s"]):
        raise GateConfigError(f"{d.get('name')}: segment times must increase and end before duration")
    if any(len(c) != 3 for _t, c in segs):
        raise GateConfigError(f"{d.get('name')}: commands are (vx, vy, wz)")
    tier = d.get("tier", "nominal")
    if tier not in TIERS:
        raise GateConfigError(f"{d.get('name')}: unknown tier {tier!r}")
    return GateScenario(
        name=str(d["name"]),
        tier=tier,
        duration_s=float(d["duration_s"]),
        segments=segs,
        foot_friction=None if d.get("foot_friction") is None else float(d["foot_friction"]),
        payload_kg=float(d.get("payload_kg", 0.0)),
        stand_checks=bool(d.get("stand_checks", False)),
    )


@dataclass(frozen=True)
class GateConfig:
    raw: dict[str, Any]
    path: str
    scenarios: tuple[GateScenario, ...]

    @property
    def physics(self) -> dict[str, Any]:
        return self.raw["physics"]

    @property
    def actuator(self) -> dict[str, dict[str, float]]:
        return self.raw["actuator"]

    @property
    def fall(self) -> dict[str, float]:
        return self.raw["fall"]

    @property
    def tracking(self) -> dict[str, float]:
        return self.raw["tracking"]

    @property
    def version(self) -> int:
        return GATE_SCHEMAS[self.raw["schema"]]

    @property
    def thresholds(self) -> dict[str, dict[str, float]]:
        """Numeric thresholds in the v1 layout (all / stand / nominal / stress) for both versions."""
        if self.version == 1:
            return self.raw["thresholds"]
        s, p = self.raw["safety"], self.raw["performance"]
        return {"all": s["all"], "stand": s["stand"], "nominal": p["nominal"], "stress": p["stress"]}

    @property
    def saturation_ge(self) -> bool:
        """v2: saturated when |raw| >= clip. v1: when |raw| > clip."""
        return self.version >= 2 and self.raw["saturation"]["comparison"] == ">="

    @property
    def missing_clip_fails(self) -> bool:
        return self.version >= 2 and self.raw["saturation"].get("missing_clip") == "fail"

    def check_tier(self, check: str) -> str:
        if self.version == 1:
            return CHECK_TIER_GATE
        return CHECK_TIER_PERFORMANCE if check in _TRACKING_CHECKS else CHECK_TIER_SAFETY

    @property
    def required_envelope(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.raw["required_command_envelope"].items()}

    def scenario(self, name: str) -> GateScenario:
        for s in self.scenarios:
            if s.name == name:
                return s
        raise KeyError(f"unknown gate scenario {name!r}; known {[s.name for s in self.scenarios]}")


def load_gate_config(path: str | Path | None = None) -> GateConfig:
    """Load a gate file. ``path`` may also be ``"v1"`` / ``"v2"``. Default v2."""
    import yaml

    if isinstance(path, str) and path in GATE_CONFIGS:
        path = GATE_CONFIGS[path]
    p = Path(path or DEFAULT_GATE_CONFIG)
    raw = yaml.safe_load(p.read_text())
    if raw.get("schema") not in GATE_SCHEMAS:
        raise GateConfigError(f"{p}: schema {raw.get('schema')!r} not in {sorted(GATE_SCHEMAS)}")
    keys = ["physics", "actuator", "fall", "tracking", "required_command_envelope", "scenarios"]
    keys += ["thresholds"] if GATE_SCHEMAS[raw["schema"]] == 1 else ["safety", "performance", "saturation"]
    for key in keys:
        if key not in raw:
            raise GateConfigError(f"{p}: missing {key}")
    if set(raw["actuator"]) != set(JOINT_GROUPS):
        raise GateConfigError(f"{p}: actuator groups must be {JOINT_GROUPS}")
    scen = tuple(_scenario(s) for s in raw["scenarios"])
    if len({s.name for s in scen}) != len(scen):
        raise GateConfigError(f"{p}: duplicate scenario names")
    return GateConfig(raw=raw, path=str(p), scenarios=scen)


# ------------------------------------------------------------------ metric math


def roll_pitch_from_quat_wxyz(q: Sequence[float]) -> tuple[float, float]:
    """ZYX Euler roll and pitch (rad) of a body-to-world (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    s = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return roll, math.asin(s)


def trailing_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Mean over the trailing ``n`` samples (fewer at the start). Axis 0."""
    x = np.asarray(x, dtype=np.float64)
    if n <= 1:
        return x.copy()
    c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), x], axis=0), axis=0)
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - n, 0)
    cnt = (idx - lo).reshape((-1,) + (1,) * (x.ndim - 1))
    return (c[idx] - c[lo]) / cnt


def tracking_rmse(
    times: np.ndarray,
    commands: np.ndarray,
    lin_vel_xy: np.ndarray,
    yaw_rate: np.ndarray,
    change_times: Sequence[float],
    *,
    control_hz: float,
    smoothing_window_s: float,
    exclude_after_change_s: float,
) -> dict[str, Any]:
    """Velocity-tracking RMSE on smoothed body velocity, excluding command transients.

    ``commands`` is (N, 3) (vx, vy, wz); ``lin_vel_xy`` (N, 2) body frame;
    ``yaw_rate`` (N,). A sample is scored when at least ``exclude_after_change_s``
    has passed since the latest command change at or before it.
    """
    times = np.asarray(times, dtype=np.float64)
    n_win = max(1, int(round(smoothing_window_s * control_hz)))
    v = trailing_mean(np.asarray(lin_vel_xy), n_win)
    w = trailing_mean(np.asarray(yaw_rate), n_win)
    changes = np.asarray(sorted(change_times), dtype=np.float64)
    last = changes[np.searchsorted(changes, times + 1e-9, side="right") - 1]
    mask = (times - last) >= exclude_after_change_s - 1e-9
    if not np.any(mask):
        return {"n_samples": 0, "lin_vel_rmse_mps": None, "yaw_rate_rmse_radps": None,
                "mean_vx": None, "mean_vy": None, "mean_wz": None}
    cmd = np.asarray(commands, dtype=np.float64)[mask]
    ev = v[mask] - cmd[:, :2]
    ew = w[mask] - cmd[:, 2]
    return {
        "n_samples": int(mask.sum()),
        "lin_vel_rmse_mps": float(np.sqrt(np.mean(np.sum(ev * ev, axis=1)))),
        "yaw_rate_rmse_radps": float(np.sqrt(np.mean(ew * ew))),
        "mean_vx": float(np.mean(v[mask][:, 0])),
        "mean_vy": float(np.mean(v[mask][:, 1])),
        "mean_wz": float(np.mean(w[mask])),
    }


def saturation_mask(raw: np.ndarray, clip: float | None, cfg: GateConfig) -> np.ndarray | None:
    """Per-joint pre-clip saturation of one raw action, relative to the deploy clip.

    v2: ``|raw| >= clip``; a missing clip returns None (unbounded path, the check
    fails). v1: ``|raw| > clip``; a missing clip counts nothing (v1 behaviour kept).
    """
    a = np.abs(np.asarray(raw, dtype=np.float64))
    if clip is None:
        return None if cfg.missing_clip_fails else np.zeros(a.shape, dtype=bool)
    return a >= clip if cfg.saturation_ge else a > clip


# ------------------------------------------------------------------ verdict


def _check(name: str, value: Any, op: str, threshold: Any) -> dict[str, Any]:
    if value is None:
        ok = False
    elif op == "<=":
        ok = value <= threshold
    elif op == ">=":
        ok = value >= threshold
    elif op == "==":
        ok = value == threshold
    else:  # pragma: no cover
        raise ValueError(op)
    return {"check": name, "value": value, "op": op, "threshold": threshold, "pass": bool(ok)}


def envelope_checks(spec: DeploySpec, required: Mapping[str, float]) -> list[dict[str, Any]]:
    out = []
    for k in COMMAND_KEYS:
        need = float(required[k])
        if spec.trained_commands is None:
            out.append({"check": f"envelope.{k}", "value": None, "op": ">=", "threshold": need,
                        "pass": False, "note": "trained command envelope not declared"})
            continue
        lo, hi = spec.trained_commands[k]
        # Both signs of the required magnitude must be inside the trained range.
        covered = min(hi, -lo)
        out.append(_check(f"envelope.{k}", covered, ">=", need))
    return out


def evaluate_scenario(metrics: Mapping[str, Any], scenario: GateScenario, cfg: GateConfig) -> list[dict[str, Any]]:
    th_all = cfg.thresholds["all"]
    th_tier = cfg.thresholds[scenario.tier]
    checks = [
        _check("no_fall", bool(metrics["fell"]), "==", False),
        _check("finite_actions", bool(metrics["nonfinite_action"]), "==", False),
        _check("pre_clip_saturation_rate", metrics["pre_clip_saturation_rate"], "<=",
               th_all["max_pre_clip_saturation_rate"]),
        _check("hard_limit_violation_steps", metrics["hard_limit_violation_steps"], "<=",
               th_all["max_hard_limit_violation_steps"]),
    ]
    for g in JOINT_GROUPS:
        checks.append(_check(f"torque_saturation_fraction.{g}", metrics["torque_saturation_fraction"][g],
                             "<=", th_all["max_torque_saturation_fraction"]))
        checks.append(_check(f"near_limit_fraction.{g}", metrics["near_limit_fraction"][g], "<=",
                             th_all["max_near_limit_fraction"]))
    checks.append(_check("lin_vel_rmse_mps", metrics["lin_vel_rmse_mps"], "<=", th_tier["max_lin_vel_rmse_mps"]))
    checks.append(_check("yaw_rate_rmse_radps", metrics["yaw_rate_rmse_radps"], "<=",
                         th_tier["max_yaw_rate_rmse_radps"]))
    if scenario.stand_checks:
        checks.append(_check("mean_base_height_m", metrics["mean_base_height_m"], ">=",
                             cfg.thresholds["stand"]["min_mean_base_height_m"]))
    if cfg.version >= 3:
        resp = cfg.raw["safety"]["responsiveness"]
        component = resp["scenarios"].get(scenario.name)
        if component is not None:
            checks.append(responsiveness_check(metrics, scenario, component, float(resp["min_fraction"])))
    for c in checks:
        c["tier"] = cfg.check_tier(c["check"])
    return checks


def responsiveness_check(metrics: Mapping[str, Any], scenario: GateScenario, component: str,
                         min_fraction: float) -> dict[str, Any]:
    """Achieved mean (tracking window) along the command, as a fraction of |command|.

    value = sign(cmd) * mean / |cmd|; pass when >= ``min_fraction``. The command is
    ``component`` of the scenario's last segment. No tracking window (fall) fails.
    """
    key, idx = _RESPONSE_METRIC[component]
    cmd = float(scenario.segments[-1][1][idx])
    if cmd == 0.0:
        raise GateConfigError(f"responsiveness on {scenario.name}.{component}: last command is zero")
    mean = metrics.get(key)
    frac = None if mean is None else math.copysign(1.0, cmd) * float(mean) / abs(cmd)
    c = _check(f"responsiveness.{component}", frac, ">=", min_fraction)
    c["achieved_mean"] = mean
    c["command"] = cmd
    return c


def tag_spec_checks(checks: list[dict[str, Any]], cfg: GateConfig) -> list[dict[str, Any]]:
    """Spec-level checks are blocking in both versions (v2: SAFETY)."""
    tier = CHECK_TIER_GATE if cfg.version == 1 else CHECK_TIER_SAFETY
    for c in checks:
        c["tier"] = tier
    return checks


def _blocking(c: Mapping[str, Any]) -> bool:
    return c.get("tier", CHECK_TIER_GATE) != CHECK_TIER_PERFORMANCE


def gate_verdict(report: Mapping[str, Any]) -> dict[str, Any]:
    """Verdict from the BLOCKING checks only; performance failures are listed separately.

    v1: every check blocks. v2: SAFETY checks block, PERFORMANCE checks are reported.
    """
    failures: list[str] = []
    perf: list[str] = []
    for c in report["spec_checks"]:
        if not c["pass"]:
            (failures if _blocking(c) else perf).append(f"spec:{c['check']}")
    for name, r in report["scenarios"].items():
        for c in r["checks"]:
            if not c["pass"]:
                (failures if _blocking(c) else perf).append(f"{name}:{c['check']}")
    return {"verdict": "PASS" if not failures else "FAIL", "failures": failures,
            "performance_failures": perf}


def performance_table(report: Mapping[str, Any], cfg: GateConfig) -> dict[str, Any]:
    """Per-scenario tracking rows plus the named rows (v2 ``performance.named``)."""
    rows = []
    for name, r in report["scenarios"].items():
        for c in r["checks"]:
            if c["check"] in _TRACKING_CHECKS:
                rows.append({"scenario": name, "metric": c["check"], "value": c["value"],
                             "threshold": c["threshold"], "pass": c["pass"]})
    named = {}
    for label, ref in (cfg.raw.get("performance", {}).get("named") or {}).items():
        match = [row for row in rows if row["scenario"] == ref["scenario"] and row["metric"] == ref["metric"]]
        named[label] = match[0] if match else None
    return {"rows": rows, "named": named,
            "verdict": "PASS" if rows and all(row["pass"] for row in rows) else "FAIL"}


__all__ = [
    "CHECK_TIER_GATE",
    "CHECK_TIER_PERFORMANCE",
    "CHECK_TIER_SAFETY",
    "DEFAULT_GATE_CONFIG",
    "GATE_CONFIGS",
    "GATE_SCHEMA",
    "GATE_SCHEMA_V2",
    "GATE_SCHEMA_V3",
    "responsiveness_check",
    "performance_table",
    "saturation_mask",
    "tag_spec_checks",
    "GateConfig",
    "GateConfigError",
    "GateScenario",
    "REPORT_SCHEMA",
    "envelope_checks",
    "evaluate_scenario",
    "gate_verdict",
    "load_gate_config",
    "roll_pitch_from_quat_wxyz",
    "tracking_rmse",
    "trailing_mean",
]
