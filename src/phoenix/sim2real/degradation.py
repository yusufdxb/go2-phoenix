"""Controlled actuator degradation: a bounded, reversible software gain reduction.

Applies to ONE joint, or to a named set of joints at the same scale (a leg, a pair
of legs, one motor class, or the whole robot). The multi-joint form was permitted by
the robot owner for the intervention-screening study and carries a HIGHER floor than
the single-joint form (:data:`MIN_SCALE_MULTI` vs :data:`MIN_SCALE`).

This exists to validate Phoenix with a known ground truth, not to model any
particular motor fault. It is NOT motor damage and must never be reported as such.

What it does
------------
While the bridge is in POLICY mode, the ``kp`` and ``kd`` sent to every affected
motor are multiplied by ``kp_scale`` / ``kd_scale``. The GO2 motor driver then
produces ``s * (kp (q_sent - q) - kd dq)`` on those joints instead of the nominal
torque: they deliver less authority for the same command.
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
* Bounded, and the bound tightens with reach: 0.5 for one joint
  (``MIN_SCALE``), 0.70 for two or more (``MIN_SCALE_MULTI``). Every affected joint
  takes the same scale. Ramped in over ``RAMP_S`` after policy authority begins.
* Watched: if ANY degraded joint stays pinned at the slew limit for
  ``SATURATION_LATCH_S`` the gate latches HOLD (``degradation_joint_saturated``).
  The watch is per joint, so widening the set cannot dilute it.
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

#: Floor for a spec that touches exactly ONE joint. Unchanged since the gate shipped.
MIN_SCALE = 0.5
#: Floor for a spec that touches TWO OR MORE joints, raised by the robot owner when
#: the multi-joint form was permitted (EXPERIMENT.md amendment 13). A single weak joint
#: is load-shared by the other eleven; a whole leg or the whole robot at the same scale
#: removes that margin, so the permitted envelope is narrower the more joints it covers.
MIN_SCALE_MULTI = 0.70
#: The scale is ramped in linearly over this long after policy authority begins, so
#: the hand-off from the stand-up gains (kp 60) or a fresh arm is not a gain step.
RAMP_S = 2.0
#: Latch HOLD when the degraded joint's target stays pinned at the slew limit for this
#: long: under the measured-q clip that joint's restoring torque is capped at
#: ``s * kp * 0.175`` N m, so a pinned joint is a leg that is sagging, and no other rule
#: in the gate fires until the joint crosses its hard limit.
SATURATION_LATCH_S = 0.5
#: "Pinned" means the policy asks the degraded joint for at least this much beyond its
#: measured position. The historical slew cap value, kept fixed when the soft limiter
#: changed to a command-rate limiter (Phoenix v2 amendment 1) so the latch is unchanged.
DEGRADATION_PIN_BAND_RAD = 0.175
ARM_ENV = "PHOENIX_EXPERIMENT"
ARM_VALUE = "controlled_degradation"
STAGE_PREFIX = "X"


@dataclass(frozen=True)
class DegradationSpec:
    """One scale, applied to one joint or to a named set of joints.

    ``joint`` is a single joint name (the historical form, floor :data:`MIN_SCALE`)
    or a tuple of two or more names (floor :data:`MIN_SCALE_MULTI`). Every affected
    joint gets the SAME ``kp_scale`` / ``kd_scale``: the intervention this study is
    allowed to apply is a uniform authority reduction over a joint set, not a
    per-joint profile. That keeps the manipulated variable one number, which is what
    the monitor has to estimate and the conditioner has to reproduce.
    """

    joint: str | tuple[str, ...]
    kp_scale: float
    kd_scale: float

    def __post_init__(self) -> None:
        if isinstance(self.joint, str):
            names: tuple[str, ...] = (self.joint,)
        else:
            names = tuple(self.joint)
            # A one-element set is the single-joint spec; normalise so that it
            # compares equal to, and is bounded exactly like, the historical form.
            object.__setattr__(self, "joint", names[0] if len(names) == 1 else names)
        if not names:
            raise ValueError("a degradation spec needs at least one joint")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate joints in {names}")
        for name in names:
            if name not in UNITREE_MOTOR_ORDER:
                raise ValueError(f"unknown joint {name!r}; expected one of {UNITREE_MOTOR_ORDER}")
        floor = MIN_SCALE if len(names) == 1 else MIN_SCALE_MULTI
        for attr in ("kp_scale", "kd_scale"):
            v = getattr(self, attr)
            if not np.isfinite(v) or not (floor <= v <= 1.0):
                raise ValueError(
                    f"{attr} must be in [{floor}, 1.0] (a reduction only) for a "
                    f"{len(names)}-joint spec, got {v}"
                )

    @property
    def joints(self) -> tuple[str, ...]:
        """Every affected joint, always a tuple."""
        return (self.joint,) if isinstance(self.joint, str) else self.joint

    @property
    def motor_index(self) -> int:
        """The affected motor, for a single-joint spec only."""
        if not isinstance(self.joint, str):
            raise ValueError(f"{len(self.joints)} joints affected; use motor_indices")
        return UNITREE_MOTOR_ORDER.index(self.joint)

    @property
    def motor_indices(self) -> tuple[int, ...]:
        return tuple(UNITREE_MOTOR_ORDER.index(j) for j in self.joints)

    @property
    def min_scale(self) -> float:
        """The floor this spec was bounded by."""
        return MIN_SCALE if len(self.joints) == 1 else MIN_SCALE_MULTI

    def _vector(self, scale: float) -> np.ndarray:
        v = np.ones(len(UNITREE_MOTOR_ORDER))
        v[list(self.motor_indices)] = scale
        return v

    def kp_scale_vector(self) -> np.ndarray:
        return self._vector(self.kp_scale)

    def kd_scale_vector(self) -> np.ndarray:
        return self._vector(self.kd_scale)

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["joints"] = list(self.joints)
        d["min_scale"] = self.min_scale
        return d


def _leg(prefix: str) -> tuple[str, ...]:
    return tuple(j for j in UNITREE_MOTOR_ORDER if j.startswith(prefix + "_"))


def _class(kind: str) -> tuple[str, ...]:
    return tuple(j for j in UNITREE_MOTOR_ORDER if j == f"{j[:2]}_{kind}_joint")


#: Named joint sets accepted by :func:`parse_spec`. Each one is a physically
#: meaningful group (a leg, a pair of legs, or one motor class across the robot),
#: so a spec cannot name an arbitrary combination hunted for effect.
JOINT_GROUPS: dict[str, tuple[str, ...]] = {
    "all": UNITREE_MOTOR_ORDER,
    **{f"leg_{p}": _leg(p) for p in ("FR", "FL", "RR", "RL")},
    "rear": _leg("RR") + _leg("RL"),
    "front": _leg("FR") + _leg("FL"),
    "diag_a": _leg("FR") + _leg("RL"),
    "diag_b": _leg("FL") + _leg("RR"),
    **{f"{k}s": _class(k) for k in ("hip", "thigh", "calf")},
}


def parse_spec(text: str) -> DegradationSpec:
    """``TARGET:SCALE`` (same scale for kp and kd) or ``TARGET:KP_SCALE:KD_SCALE``.

    ``TARGET`` is one of:

    * a joint name; a bare ``RR_thigh`` is accepted for ``RR_thigh_joint``
    * a named group from :data:`JOINT_GROUPS` (``all``, ``leg_RR``, ``rear``,
      ``thighs``, ...)
    * joint names joined by ``+``, e.g. ``RR_hip+RR_thigh``

    Any target covering two or more joints is bounded by :data:`MIN_SCALE_MULTI`.
    """
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"expected TARGET:SCALE or TARGET:KP:KD, got {text!r}")
    target = parts[0].strip()
    if target in JOINT_GROUPS:
        joints: str | tuple[str, ...] = JOINT_GROUPS[target]
    else:
        names = tuple(
            n if n.endswith("_joint") else n + "_joint"
            for n in (p.strip() for p in target.split("+"))
        )
        joints = names[0] if len(names) == 1 else names
    kp = float(parts[1])
    kd = float(parts[2]) if len(parts) == 3 else kp
    return DegradationSpec(joint=joints, kp_scale=kp, kd_scale=kd)


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
    "JOINT_GROUPS",
    "MIN_SCALE",
    "MIN_SCALE_MULTI",
    "RAMP_S",
    "SATURATION_LATCH_S",
    "STAGE_PREFIX",
    "DegradationSpec",
    "activation_problems",
    "parse_spec",
]
