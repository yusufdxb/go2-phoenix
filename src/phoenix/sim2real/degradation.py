"""Controlled actuator degradation: a bounded, reversible software gain reduction on ONE joint.

This exists to validate Phoenix with a known ground truth, not to model any
particular motor fault. It is NOT motor damage and must never be reported as such.

What it does
------------
While the bridge is in POLICY mode, the ``kp`` and ``kd`` sent to one motor are
multiplied by ``kp_scale`` / ``kd_scale`` (both in ``[MIN_SCALE, 1.0]``). The GO2
motor driver then produces ``s * (kp (q_sent - q) - kd dq)`` on that joint instead
of the nominal torque: the joint delivers less authority for the same command.
In Isaac Lab the Go2 actuator is an explicit ``DCMotor`` running the same PD law
with ``stiffness`` 25 / ``damping`` 0.5, equal to the bridge's ``kp`` 25 / ``kd``
0.5, so the same scale applied to that joint's stiffness and damping in sim is the
same intervention. That identity is what makes the Phoenix experiment checkable:
the monitor's estimate and the conditioned sim range can both be compared with
the value that was actually applied.

Why this is safe to have in the final actuator gate
---------------------------------------------------
* It can only REDUCE gains (scale <= 1). It never raises a torque above what the
  nominal command would have produced, never moves a target, and never touches the
  joint-limit clip, the slew clip, the abort band or the LowState freshness rules.
* It applies in POLICY mode only. HOLD, DAMP and STANDUP keep their nominal gains,
  so an estop, a deadman release, a latched fault or a stale LowState behave exactly
  as without it.
* Bounded: ``MIN_SCALE`` = 0.5. One joint at a time.
* Off by default and impossible to enable by accident: it needs the CLI spec AND
  the environment variable :data:`ARM_ENV` set to :data:`ARM_VALUE` AND a stage
  label starting with :data:`STAGE_PREFIX`. Any one missing refuses startup.
* Recorded: the manifest carries the spec, and every tick record carries the
  per-motor scale that was actually applied (1.0 when not applied).

Reversal: restart the bridge without the flag. There is no persistent state.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

from .go2_model import UNITREE_MOTOR_ORDER

MIN_SCALE = 0.5
ARM_ENV = "PHOENIX_EXPERIMENT"
ARM_VALUE = "controlled_degradation"
STAGE_PREFIX = "X"


@dataclass(frozen=True)
class DegradationSpec:
    joint: str
    kp_scale: float
    kd_scale: float

    def __post_init__(self) -> None:
        if self.joint not in UNITREE_MOTOR_ORDER:
            raise ValueError(f"unknown joint {self.joint!r}; expected one of {UNITREE_MOTOR_ORDER}")
        for name in ("kp_scale", "kd_scale"):
            v = getattr(self, name)
            if not np.isfinite(v) or not (MIN_SCALE <= v <= 1.0):
                raise ValueError(
                    f"{name} must be in [{MIN_SCALE}, 1.0] (a reduction only), got {v}"
                )

    @property
    def motor_index(self) -> int:
        return UNITREE_MOTOR_ORDER.index(self.joint)

    def kp_scale_vector(self) -> np.ndarray:
        v = np.ones(len(UNITREE_MOTOR_ORDER))
        v[self.motor_index] = self.kp_scale
        return v

    def kd_scale_vector(self) -> np.ndarray:
        v = np.ones(len(UNITREE_MOTOR_ORDER))
        v[self.motor_index] = self.kd_scale
        return v

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_spec(text: str) -> DegradationSpec:
    """``JOINT:SCALE`` (same scale for kp and kd) or ``JOINT:KP_SCALE:KD_SCALE``.

    A bare leg-joint name such as ``RR_thigh`` is accepted for ``RR_thigh_joint``.
    """
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"expected JOINT:SCALE or JOINT:KP:KD, got {text!r}")
    joint = parts[0].strip()
    if not joint.endswith("_joint"):
        joint = joint + "_joint"
    kp = float(parts[1])
    kd = float(parts[2]) if len(parts) == 3 else kp
    return DegradationSpec(joint=joint, kp_scale=kp, kd_scale=kd)


def activation_problems(
    spec: DegradationSpec | None,
    stage: str,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Why this spec may not be armed. Empty when there is no spec or all locks agree."""
    if spec is None:
        return []
    env = os.environ if environ is None else environ
    problems: list[str] = []
    if env.get(ARM_ENV) != ARM_VALUE:
        problems.append(f"controlled degradation requires {ARM_ENV}={ARM_VALUE} in the environment")
    if not stage.startswith(STAGE_PREFIX):
        problems.append(
            f"controlled degradation is only allowed in an experiment stage "
            f"(label starting with {STAGE_PREFIX!r}); got stage {stage!r}"
        )
    return problems


__all__ = [
    "ARM_ENV",
    "ARM_VALUE",
    "MIN_SCALE",
    "STAGE_PREFIX",
    "DegradationSpec",
    "activation_problems",
    "parse_spec",
]
