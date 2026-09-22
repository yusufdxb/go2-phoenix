"""From a health report to a TARGETED Isaac Lab actuator distribution.

This is the step that makes Phoenix Phoenix: the robot's own measurement decides
what the next training run randomises, and nothing else is widened.

Input: the monitor's per-joint health (:mod:`phoenix.monitor.health`), which
already refuses to localise a change seen on many joints at once.
Output: an env-config overlay the existing layered-YAML factory reads::

    defaults: [<parent env of the deployed policy>]
    domain_randomization:
      targeted_actuator:
        nominal_fraction: 0.5
        scale_damping: true
        joints:
          RR_thigh_joint: [0.52, 0.72]
    phoenix_condition: {provenance...}

Semantics (implemented in ``phoenix.sim_env.go2_env_cfg`` as a startup event):
a fixed ``1 - nominal_fraction`` share of the parallel envs multiplies the named
joint's actuator stiffness (and damping, when ``scale_damping``) by a factor drawn
uniformly from the range; the other envs are left as the parent recipe made them.
The factor is applied ON TOP of the parent's own motor-strength randomisation, so
the only difference between the parent recipe and the Phoenix recipe is the
targeted term. Hardware ``kp``/``kd`` 25 / 0.5 equal the sim ``DCMotor`` stiffness /
damping 25 / 0.5, so a factor here means the same thing as the controlled
degradation scale on the robot (:mod:`phoenix.sim2real.degradation`).

Range construction
------------------
``[max(S_MIN, s_lo - margin), min(1.0, s_hi + margin)]`` where ``s_lo``/``s_hi`` are
the 2.5 / 97.5 percentiles of the joint's window estimates over its persistence
span. A RANGE, not a point: the estimator is biased under saturation and under
policy compensation (see ``phoenix.monitor.residual``), and the margin absorbs
some of that. The upper end is capped at 1.0: Phoenix conditions on less authority
than nominal, never more.

``nominal_fraction`` 0.5 is the preregistered default. The one-joint model
(``phoenix.condition.toy_model``) shows why a mixture: training only on the
degraded range makes the policy worst at nominal, which the candidate gate would
then (correctly) reject.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from phoenix.monitor.health import JointHealth, JointState

SCHEMA = "phoenix-condition/v1"
S_MIN = 0.3  # never train on less than 30 % authority: below that it is a different robot
DEFAULT_MARGIN = 0.05
DEFAULT_NOMINAL_FRACTION = 0.5
MAX_TARGETED_JOINTS = 2


class ConditionError(ValueError):
    """The report does not justify a targeted distribution."""


@dataclass(frozen=True)
class TargetedActuatorSpec:
    joints: dict[str, tuple[float, float]]
    nominal_fraction: float = DEFAULT_NOMINAL_FRACTION
    scale_damping: bool = True

    def __post_init__(self) -> None:
        if not self.joints:
            raise ValueError("at least one joint is required")
        if len(self.joints) > MAX_TARGETED_JOINTS:
            raise ValueError(f"at most {MAX_TARGETED_JOINTS} joints may be targeted")
        for name, (lo, hi) in self.joints.items():
            if not (S_MIN <= lo <= hi <= 1.0):
                raise ValueError(f"{name}: range [{lo}, {hi}] outside [{S_MIN}, 1.0] or inverted")
        if not 0.0 <= self.nominal_fraction < 1.0:
            raise ValueError("nominal_fraction must be in [0, 1)")

    def to_yaml_block(self) -> dict[str, Any]:
        return {
            "nominal_fraction": float(self.nominal_fraction),
            "scale_damping": bool(self.scale_damping),
            "joints": {k: [float(lo), float(hi)] for k, (lo, hi) in self.joints.items()},
        }

    @classmethod
    def from_yaml_block(cls, block: Mapping[str, Any]) -> TargetedActuatorSpec:
        return cls(
            joints={k: (float(v[0]), float(v[1])) for k, v in block["joints"].items()},
            nominal_fraction=float(block.get("nominal_fraction", DEFAULT_NOMINAL_FRACTION)),
            scale_damping=bool(block.get("scale_damping", True)),
        )


@dataclass(frozen=True)
class ConditionResult:
    spec: TargetedActuatorSpec
    parent_env: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def overlay(self) -> dict[str, Any]:
        return {
            "defaults": [self.parent_env],
            "domain_randomization": {"targeted_actuator": self.spec.to_yaml_block()},
            "phoenix_condition": {"schema": SCHEMA, **self.provenance},
        }


def build_targeted_spec(
    report: Sequence[JointHealth],
    margin: float = DEFAULT_MARGIN,
    nominal_fraction: float = DEFAULT_NOMINAL_FRACTION,
) -> TargetedActuatorSpec:
    """Refuses unless the report has 1..MAX_TARGETED_JOINTS localised DEGRADED joints."""
    states = {h.state for h in report}
    if JointState.GLOBAL_SHIFT.value in states:
        raise ConditionError(
            "the change is global (many joints at once): not an actuator-localised change, "
            "no targeted distribution"
        )
    degraded = [h for h in report if h.state == JointState.DEGRADED.value]
    if not degraded:
        raise ConditionError("no joint is persistently DEGRADED")
    if len(degraded) > MAX_TARGETED_JOINTS:
        raise ConditionError(f"{len(degraded)} joints degraded, more than {MAX_TARGETED_JOINTS}")
    joints: dict[str, tuple[float, float]] = {}
    for h in degraded:
        if h.s_lo is None or h.s_hi is None:
            raise ConditionError(f"{h.joint}: no estimate interval")
        lo = float(np.clip(h.s_lo - margin, S_MIN, 1.0))
        hi = float(np.clip(h.s_hi + margin, S_MIN, 1.0))
        joints[h.joint] = (min(lo, hi), hi)
    return TargetedActuatorSpec(joints=joints, nominal_fraction=nominal_fraction)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def condition(
    report: Sequence[JointHealth],
    parent_env: str,
    telemetry_paths: Sequence[str | Path] = (),
    baseline: Mapping[str, Any] | None = None,
    margin: float = DEFAULT_MARGIN,
    nominal_fraction: float = DEFAULT_NOMINAL_FRACTION,
    extra: Mapping[str, Any] | None = None,
) -> ConditionResult:
    """Build the overlay plus a provenance record tying it to the exact evidence."""
    spec = build_targeted_spec(report, margin=margin, nominal_fraction=nominal_fraction)
    provenance: dict[str, Any] = {
        "health": [h.to_dict() for h in report],
        "margin": margin,
        "telemetry": [{"path": str(p), "sha256": sha256_file(p)} for p in telemetry_paths],
        "baseline": dict(baseline) if baseline is not None else None,
    }
    if extra:
        provenance.update(dict(extra))
    return ConditionResult(spec=spec, parent_env=parent_env, provenance=provenance)


def targeted_scale_factors(
    num_envs: int,
    joint_names: Sequence[str],
    spec: TargetedActuatorSpec,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-env, per-joint multiplicative factors ``(stiffness, damping)``, shape ``(N, J)``.

    Pure numpy so the exact sampling rule is unit-tested; the Isaac Lab startup
    event only multiplies the actuator tensors by these. Envs ``[0, n_nominal)``
    stay at 1.0; the rest draw the targeted joints uniformly from their ranges.
    A joint named in the spec but absent from ``joint_names`` is an error: a typo
    must not silently train the parent recipe.
    """
    names = list(joint_names)
    missing = [j for j in spec.joints if j not in names]
    if missing:
        raise KeyError(f"targeted joints not in the actuator: {missing}")
    stiff = np.ones((num_envs, len(names)))
    n_nom = int(round(spec.nominal_fraction * num_envs))
    for joint, (lo, hi) in spec.joints.items():
        j = names.index(joint)
        stiff[n_nom:, j] = rng.uniform(lo, hi, num_envs - n_nom)
    damp = stiff.copy() if spec.scale_damping else np.ones_like(stiff)
    return stiff, damp


__all__ = [
    "DEFAULT_MARGIN",
    "DEFAULT_NOMINAL_FRACTION",
    "MAX_TARGETED_JOINTS",
    "S_MIN",
    "SCHEMA",
    "ConditionError",
    "ConditionResult",
    "TargetedActuatorSpec",
    "build_targeted_spec",
    "condition",
    "sha256_file",
    "targeted_scale_factors",
]
