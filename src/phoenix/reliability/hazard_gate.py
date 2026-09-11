"""H0 gate v2: delivery, precursor validity, and recoverability, measured separately.

Why this module replaces the v1 criterion
-----------------------------------------
The preregistered H0 asks whether curriculum-seeded states are "measurably
distinct from nominal reset states, and distinguishable in the direction of the
intended failure mode". The v1 probe (``scripts/h0_delivery_probe.py``) was
later re-anchored, during a robustness pass, onto each trajectory's OWN prior
history. That is statistically sounder as an estimator but it silently answers a
different question: whether the seeded state is ALREADY EXHIBITING the failure.

Those are not the same question, and the substitution made the gate actively
misleading. Measured on harvested rollout failures, a collapse case's z score
GREW from +2.55 to +3.95 as the seed row moved earlier, while its direction
inverted, because earlier in time the robot was genuinely standing higher.
Seeding earlier finds a healthier robot, which is the entire point of pre-onset
seeding, and the v1 criterion scored it as a failure. A curriculum that only
accepts states already exhibiting the failure can teach recovery and can never
teach avoidance.

This module does not modify the v1 probe or any recorded v1 evidence. It defines
a new, versioned gate that separates three questions the v1 criterion conflated.

The three criteria
------------------
**A. Treatment delivery.** Does the seeded state differ meaningfully from an
ORDINARY RESET? This is the preregistered question, and the reference is the
simulator's nominal reset distribution, not the trajectory's own history. Pure
geometry, no rollouts, computable without a simulator.

**B. Precursor validity.** Under the original or a matched failure context, does
the seeded state carry an ELEVATED probability of future failure relative to an
ordinary reset in that same context? A state that is merely unusual is not a
precursor. This requires rollouts and is the criterion v1 never tested at all.

**C. Recoverability.** Is the state still recoverable, or is it already doomed?
A state from which the policy fails with probability ~1 regardless of what it
does carries no learning signal: there is no counterfactual to learn. Seeding
those teaches nothing and inflates the failure rate of the treatment arm for
free.

B and C together define a RECOVERABLE HAZARD BAND: elevated failure probability,
but not certain. That band, not "does the state look like a failure", is what a
useful curriculum seed has to sit inside.

Statistics
----------
Failure counts are binomial. Point estimates alone are not decidable at the
sample sizes a rollout budget allows, so every rate carries a Wilson score
interval and every difference carries a Newcombe hybrid-score interval, which
behaves correctly near 0 and 1 where the normal approximation fails badly. Both
are standard and neither requires a bootstrap.

This module is pure numpy and has no simulator, torch or ROS dependency, so the
whole decision layer is testable in CI. The rollout driver that feeds it lives
in ``scripts/h0_gate_v2.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Gate protocol version. v1 is the superseded delivery probe. Any recorded
#: result must carry this so a reader can never mistake one gate for the other.
GATE_VERSION = "h0-gate-v2"

# Nominal reset distribution, read from the IsaacLab source that defines it, not
# from memory. UNITREE_GO2_CFG.init_state gives pos z and the joint pose; the
# velocity env's reset_base randomizes x, y and yaw ONLY (never roll, pitch or
# height) with root velocity uniform on +/-0.5, and reset_robot_joints uses
# velocity_range (0.0, 0.0), so nominal joint velocity is exactly zero.
NOMINAL_HEIGHT_M = 0.4
NOMINAL_TILT_DEG = 0.0
NOMINAL_JOINT_VEL_NORM = 0.0
NOMINAL_ROOT_VEL_RANGE = 0.5


@dataclass(frozen=True)
class HazardEstimate:
    """A binomial failure-rate estimate with a Wilson score interval."""

    n: int
    n_fail: int

    def __post_init__(self) -> None:
        if self.n < 0 or self.n_fail < 0:
            raise ValueError("counts must be nonnegative")
        if self.n_fail > self.n:
            raise ValueError(f"n_fail={self.n_fail} exceeds n={self.n}")

    @property
    def rate(self) -> float:
        return float(self.n_fail) / self.n if self.n else float("nan")

    def interval(self, z: float = 1.96) -> tuple[float, float]:
        return wilson_interval(self.n_fail, self.n, z)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because rollout failure rates sit
    near 0 or near 1 exactly where the normal interval misbehaves, producing
    bounds outside [0, 1] and coverage far from nominal at small n.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    if k < 0 or k > n:
        raise ValueError(f"k={k} out of range for n={n}")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def newcombe_difference(
    seeded: HazardEstimate, baseline: HazardEstimate, z: float = 1.96
) -> tuple[float, float, float]:
    """Return (difference, low, high) for seeded.rate - baseline.rate.

    Newcombe's hybrid-score method composes the two Wilson intervals. It keeps
    correct behaviour when either arm is at 0 or 1, which a pooled normal
    interval does not.
    """
    if seeded.n <= 0 or baseline.n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    l1, u1 = seeded.interval(z)
    l2, u2 = baseline.interval(z)
    diff = seeded.rate - baseline.rate
    low = diff - math.sqrt((seeded.rate - l1) ** 2 + (u2 - baseline.rate) ** 2)
    high = diff + math.sqrt((u1 - seeded.rate) ** 2 + (baseline.rate - l2) ** 2)
    return (diff, max(-1.0, low), min(1.0, high))


def tilt_deg_from_quat_xyzw(quat_xyzw) -> float:
    """Angle between the body +z axis and world +z. Invariant to yaw."""
    q = np.asarray(quat_xyzw, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion must be finite and nonzero")
    x, y, z, w = q / norm
    cos_tilt = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0))))


@dataclass(frozen=True)
class DeliveryVerdict:
    """Criterion A: does the seeded state differ from an ordinary reset?"""

    delivered: bool
    reasons: tuple[str, ...]
    height_delta_m: float
    tilt_deg: float
    joint_vel_norm: float
    root_vel_norm: float


def assess_delivery(
    *,
    base_height_m: float,
    base_quat_xyzw,
    joint_vel,
    base_lin_vel_body,
    base_ang_vel_body,
    height_tol_m: float = 0.02,
    tilt_tol_deg: float = 2.0,
    joint_vel_tol: float = 0.5,
) -> DeliveryVerdict:
    """Criterion A, against the NOMINAL RESET distribution.

    This is the preregistered comparison. A nominal reset has height exactly
    0.400 m, tilt exactly 0 deg and joint velocity exactly 0, because the reset
    events randomize x, y, yaw and root velocity but never height, roll, pitch
    or joint velocity. Any departure beyond tolerance is delivery.

    Root velocity is deliberately NOT a delivery signal on its own: nominal
    reset already randomizes it over +/-0.5 per axis, so a seeded root velocity
    inside that box is indistinguishable from an ordinary reset. It is reported
    for context only.
    """
    tilt = tilt_deg_from_quat_xyzw(base_quat_xyzw)
    jv = float(np.linalg.norm(np.asarray(joint_vel, dtype=float)))
    lin = np.asarray(base_lin_vel_body, dtype=float)
    ang = np.asarray(base_ang_vel_body, dtype=float)
    root_vel_norm = float(np.linalg.norm(np.concatenate([lin, ang])))
    height_delta = float(base_height_m) - NOMINAL_HEIGHT_M

    reasons: list[str] = []
    if abs(height_delta) > height_tol_m:
        reasons.append(f"height {base_height_m:.3f} m differs from nominal 0.400 m")
    if tilt - NOMINAL_TILT_DEG > tilt_tol_deg:
        reasons.append(f"tilt {tilt:.2f} deg exceeds nominal 0 deg")
    if jv - NOMINAL_JOINT_VEL_NORM > joint_vel_tol:
        reasons.append(f"joint velocity norm {jv:.2f} exceeds nominal 0")

    return DeliveryVerdict(
        delivered=bool(reasons),
        reasons=tuple(reasons),
        height_delta_m=height_delta,
        tilt_deg=tilt,
        joint_vel_norm=jv,
        root_vel_norm=root_vel_norm,
    )


#: Criterion B: the seeded state must raise failure probability by at least this
#: much over an ordinary reset in the SAME context, with the interval excluding
#: zero. A seed that does not raise hazard at all is not a precursor.
DEFAULT_MIN_ELEVATION = 0.10

#: Criterion C: above this failure probability the state is treated as already
#: doomed. There is no counterfactual left to learn from a state the policy
#: cannot escape, and seeding it inflates the treatment arm's failure rate for
#: free.
DEFAULT_MAX_HAZARD = 0.95


@dataclass(frozen=True)
class HazardVerdict:
    """Combined A + B + C verdict for one candidate seed state."""

    verdict: str
    gate_version: str = GATE_VERSION
    delivered: bool = False
    seeded_rate: float = float("nan")
    baseline_rate: float = float("nan")
    elevation: float = float("nan")
    elevation_ci: tuple[float, float] = (float("nan"), float("nan"))
    detail: tuple[str, ...] = field(default_factory=tuple)

    @property
    def usable_seed(self) -> bool:
        return self.verdict == "RECOVERABLE_HAZARD"


def classify_seed(
    *,
    delivery: DeliveryVerdict,
    seeded: HazardEstimate,
    baseline: HazardEstimate,
    min_elevation: float = DEFAULT_MIN_ELEVATION,
    max_hazard: float = DEFAULT_MAX_HAZARD,
    z: float = 1.96,
) -> HazardVerdict:
    """Apply A, B and C in order and return one verdict.

    Order matters and is fail-closed. A state that was never delivered cannot be
    a precursor no matter what its rollouts do, and a state that is already
    doomed is rejected even when its elevation is large, because a large
    elevation is exactly what being doomed produces.
    """
    diff, lo, hi = newcombe_difference(seeded, baseline, z)
    detail: list[str] = list(delivery.reasons)

    if not delivery.delivered:
        verdict = "NOT_DELIVERED"
        detail.append("state is indistinguishable from an ordinary reset")
    elif not np.isfinite(diff):
        verdict = "INSUFFICIENT_SAMPLES"
        detail.append("no rollouts in one or both arms")
    elif seeded.rate > max_hazard:
        verdict = "ALREADY_DOOMED"
        detail.append(
            f"failure probability {seeded.rate:.3f} exceeds {max_hazard:.2f}; "
            "no recoverable counterfactual remains"
        )
    elif lo <= 0.0 or diff < min_elevation:
        verdict = "NOT_ELEVATED"
        detail.append(
            f"elevation {diff:+.3f} [{lo:+.3f}, {hi:+.3f}] does not clear "
            f"{min_elevation:+.2f} with an interval excluding zero"
        )
    else:
        verdict = "RECOVERABLE_HAZARD"
        detail.append(
            f"elevation {diff:+.3f} [{lo:+.3f}, {hi:+.3f}], "
            f"failure probability {seeded.rate:.3f} still below {max_hazard:.2f}"
        )

    return HazardVerdict(
        verdict=verdict,
        delivered=delivery.delivered,
        seeded_rate=seeded.rate,
        baseline_rate=baseline.rate,
        elevation=diff,
        elevation_ci=(lo, hi),
        detail=tuple(detail),
    )


__all__ = [
    "GATE_VERSION",
    "DeliveryVerdict",
    "HazardEstimate",
    "HazardVerdict",
    "assess_delivery",
    "classify_seed",
    "newcombe_difference",
    "tilt_deg_from_quat_xyzw",
    "wilson_interval",
]
