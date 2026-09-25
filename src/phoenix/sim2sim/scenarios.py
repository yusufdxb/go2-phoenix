"""Sim-to-sim scenario table and report schema, shared by MuJoCo and Isaac Lab.

Pure Python (stdlib only). The Isaac Lab evaluator imports this module to run
the SAME scenarios and emit a report with the SAME metric names, so
:mod:`phoenix.sim2sim.compare` can put the two side by side.

A scenario is a piecewise-constant velocity command schedule plus optional
perturbations (a base velocity push, an actuation latency, a foot friction
override). Times are seconds from reset. The robot starts from the Isaac Lab
nominal init state: base at (0, 0, :data:`INIT_BASE_HEIGHT_M`), identity
orientation, joints at the contract default pose, all velocities zero.

Tiers
-----
``critical``  in-distribution nominal driving that any hardware attempt will do
              first (stand, slow forward, stop, turn, command changes). A fall in
              either simulator is a hard FAIL.
``stress``    perturbed variants (push, latency, low friction, lateral). A fall
              is a WARN: it bounds the robustness margin but does not by itself
              prove the nominal behaviour is simulator-specific.

Report schema (one JSON object, both simulators)::

    {
      "schema": REPORT_SCHEMA,
      "simulator": "mujoco" | "isaaclab",
      "diagnostic": bool,          # True = policy did not pass the manifest gate
      "policy": {...},             # free-form provenance
      "config": {...},             # free-form run config
      "scenarios": {
         "<scenario name>": {"skipped": false, "metrics": {<METRIC_KEYS>}, ...}
         | {"skipped": true, "skip_reason": "..."}
      }
    }
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

REPORT_SCHEMA = "phoenix-sim2sim-report/v1"

#: Isaac Lab ``UNITREE_GO2_CFG.init_state.pos`` z (m).
INIT_BASE_HEIGHT_M = 0.4

#: Fall criteria, identical in both simulators.
FALL_TILT_RAD = 1.0
#: ``configs/env/base.yaml`` ``termination.base_height_min``.
FALL_BASE_HEIGHT_M = 0.15
#: Tracking metrics ignore the first ``SETTLE_S`` seconds (the drop from the
#: init height to stance).
SETTLE_S = 1.0

TIER_CRITICAL = "critical"
TIER_STRESS = "stress"

#: Command keys, in the (vx, vy, wz) order of the observation.
COMMAND_KEYS = ("lin_vel_x", "lin_vel_y", "ang_vel_z")


@dataclass(frozen=True)
class Push:
    """Add ``delta_vel_world`` (m/s, world frame) to the base linear velocity at ``t_s``.

    Mirrors Isaac Lab ``mdp.push_by_setting_velocity`` (it adds a sampled delta to
    the current root velocity in the world frame).
    """

    t_s: float
    delta_vel_world: tuple[float, float, float]


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    duration_s: float
    #: ``((t_start_s, (vx, vy, wz)), ...)``, sorted, first entry at t=0.
    segments: tuple[tuple[float, tuple[float, float, float]], ...]
    tier: str = TIER_CRITICAL
    push: Push | None = None
    #: Target delay in physics steps (Isaac ``DelayedDCMotor`` semantics). None = 0.
    latency_physics_steps: int = 0
    #: Foot/ground friction coefficient override. None = runner nominal.
    foot_friction: float | None = None

    def __post_init__(self) -> None:
        if not self.segments or self.segments[0][0] != 0.0:
            raise ValueError(f"{self.name}: first segment must start at t=0")
        times = [s[0] for s in self.segments]
        if times != sorted(times) or len(set(times)) != len(times):
            raise ValueError(f"{self.name}: segment start times must be strictly increasing")
        if times[-1] >= self.duration_s:
            raise ValueError(f"{self.name}: last segment starts after the scenario ends")
        if self.tier not in (TIER_CRITICAL, TIER_STRESS):
            raise ValueError(f"{self.name}: unknown tier {self.tier!r}")
        if self.latency_physics_steps < 0:
            raise ValueError(f"{self.name}: negative latency")
        # A scenario whose command exceeds the trained envelope is SKIPPED by the
        # runner (see envelope_problems), never silently run out of distribution.

    def command_at(self, t_s: float) -> tuple[float, float, float]:
        """The (vx, vy, wz) command active at time ``t_s``."""
        cmd = self.segments[0][1]
        for start, c in self.segments:
            if t_s + 1e-9 >= start:
                cmd = c
            else:
                break
        return cmd

    def max_abs_command(self) -> dict[str, float]:
        """Largest |command| per key over the schedule (for the envelope check)."""
        out = {k: 0.0 for k in COMMAND_KEYS}
        for _t, c in self.segments:
            for k, v in zip(COMMAND_KEYS, c, strict=True):
                out[k] = max(out[k], abs(float(v)))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "duration_s": self.duration_s,
            "segments": [[t, list(c)] for t, c in self.segments],
            "tier": self.tier,
            "push": (
                None
                if self.push is None
                else {"t_s": self.push.t_s, "delta_vel_world": list(self.push.delta_vel_world)}
            ),
            "latency_physics_steps": self.latency_physics_steps,
            "foot_friction": self.foot_friction,
        }


_ZERO = (0.0, 0.0, 0.0)
_FWD = (0.3, 0.0, 0.0)

SCENARIOS: tuple[Scenario, ...] = (
    Scenario("stand", "zero command for 10 s", 10.0, ((0.0, _ZERO),)),
    Scenario("slow_forward", "vx 0.3 m/s for 10 s", 10.0, ((0.0, _FWD),)),
    Scenario(
        "forward_then_stop",
        "vx 0.3 m/s for 5 s, then zero command for 5 s",
        10.0,
        ((0.0, _FWD), (5.0, _ZERO)),
    ),
    Scenario("yaw", "wz 0.5 rad/s for 10 s", 10.0, ((0.0, (0.0, 0.0, 0.5)),)),
    Scenario(
        "command_sequence",
        "stand, forward, forward+turn, turn the other way, stand",
        14.0,
        (
            (0.0, _ZERO),
            (2.0, _FWD),
            (5.0, (0.3, 0.0, 0.4)),
            (8.0, (0.0, 0.0, -0.4)),
            (11.0, _ZERO),
        ),
    ),
    Scenario(
        "lateral",
        "vy 0.2 m/s for 8 s (skipped unless trained with lateral commands)",
        8.0,
        ((0.0, (0.0, 0.2, 0.0)),),
        tier=TIER_STRESS,
    ),
    Scenario(
        "push_lateral",
        "vx 0.3 m/s, +0.5 m/s lateral base velocity push at t=4 s",
        10.0,
        ((0.0, _FWD),),
        tier=TIER_STRESS,
        push=Push(4.0, (0.0, 0.5, 0.0)),
    ),
    Scenario(
        "latency_25ms",
        "vx 0.3 m/s with a 5 physics-step (25 ms) target delay, top of the trained DR range",
        10.0,
        ((0.0, _FWD),),
        tier=TIER_STRESS,
        latency_physics_steps=5,
    ),
    Scenario(
        "low_friction",
        "vx 0.3 m/s with foot friction 0.4",
        10.0,
        ((0.0, _FWD),),
        tier=TIER_STRESS,
        foot_friction=0.4,
    ),
)

SCENARIO_BY_NAME: dict[str, Scenario] = {s.name: s for s in SCENARIOS}
assert len(SCENARIO_BY_NAME) == len(SCENARIOS), "duplicate scenario names"


def get_scenarios(names: Iterable[str] | None = None) -> list[Scenario]:
    if names is None:
        return list(SCENARIOS)
    out = []
    for n in names:
        if n not in SCENARIO_BY_NAME:
            raise KeyError(f"unknown scenario {n!r}; known: {sorted(SCENARIO_BY_NAME)}")
        out.append(SCENARIO_BY_NAME[n])
    return out


def envelope_problems(
    scenario: Scenario, trained_commands: Mapping[str, Sequence[float]] | None
) -> list[str]:
    """Why ``scenario`` would drive the policy outside its trained command envelope.

    ``trained_commands`` is the manifest ``commands`` block (``lin_vel_x`` etc. as
    ``[lo, hi]``). None (no manifest, diagnostic run) means nothing is checked.
    """
    if trained_commands is None:
        return []
    problems = []
    for _t, cmd in scenario.segments:
        for key, value in zip(COMMAND_KEYS, cmd, strict=True):
            lo, hi = (float(v) for v in trained_commands[key])
            if not (lo - 1e-9 <= float(value) <= hi + 1e-9):
                problems.append(f"{key}={value} outside trained range [{lo}, {hi}]")
    return sorted(set(problems))


#: Every metric a scenario result carries. Units in the name where not obvious.
METRIC_KEYS: tuple[str, ...] = (
    "fell",  # bool
    "fall_reason",  # str | None: "tilt", "base_height", "trunk_contact"
    "time_to_fall_s",  # float | None
    "duration_s",  # simulated seconds actually run
    "lin_vel_err_mean",  # m/s, |v_body_xy - cmd_xy|, after SETTLE_S
    "lin_vel_err_rms",  # m/s
    "yaw_rate_err_mean",  # rad/s, |w_body_z - cmd_wz|, after SETTLE_S
    "yaw_rate_err_rms",  # rad/s
    "max_tilt_rad",
    "mean_base_height_m",
    "min_base_height_m",
    "joint_limit_violation_steps",  # control steps with any joint beyond hard limits
    "max_joint_limit_excursion_rad",
    "torque_saturation_fraction",  # fraction of physics-step x joint samples clipped
    "mean_abs_torque_nm",
    "action_abs_mean",
    "action_abs_max",
    "action_rate_abs_mean",  # mean |a_t - a_{t-1}| per control step
    "action_rate_abs_max",
    "commanded_forward_m",  # integral of cmd vx over time, after SETTLE_S
    "achieved_forward_m",  # integral of body-frame vx
    "commanded_lateral_m",
    "achieved_lateral_m",
    "commanded_yaw_rad",  # integral of cmd wz
    "achieved_yaw_rad",  # unwrapped base yaw change over the same window
)


def empty_report(simulator: str, *, diagnostic: bool) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "simulator": simulator,
        "diagnostic": bool(diagnostic),
        "policy": {},
        "config": {
            "init_base_height_m": INIT_BASE_HEIGHT_M,
            "fall_tilt_rad": FALL_TILT_RAD,
            "fall_base_height_m": FALL_BASE_HEIGHT_M,
            "settle_s": SETTLE_S,
        },
        "scenarios": {},
    }


__all__ = [
    "COMMAND_KEYS",
    "FALL_BASE_HEIGHT_M",
    "FALL_TILT_RAD",
    "INIT_BASE_HEIGHT_M",
    "METRIC_KEYS",
    "Push",
    "REPORT_SCHEMA",
    "SCENARIOS",
    "SCENARIO_BY_NAME",
    "SETTLE_S",
    "Scenario",
    "TIER_CRITICAL",
    "TIER_STRESS",
    "empty_report",
    "envelope_problems",
    "get_scenarios",
]
