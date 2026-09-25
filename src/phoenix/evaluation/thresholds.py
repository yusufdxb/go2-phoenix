"""PASS / WARN / FAIL thresholds for Phoenix evaluation, each with its justification.

Every number the verdict uses lives here, next to the reason it has that value.
Nothing else in :mod:`phoenix.evaluation` hard-codes a bar. A threshold without a
justification is rejected at import time (see :data:`THRESHOLDS` and
``tests/test_evaluation_thresholds.py``).

Hard FAIL rules (no WARN band, no tuning):

* any fall (terminated on base contact / bad orientation, or tilt past the sim
  fall bar) is FAIL;
* any NaN / Inf in a state or action is FAIL;
* any requested joint target beyond a hard URDF limit by more than the abort
  band (an "illegal target", what the LowCmd bridge aborts on) is FAIL;
* an evaluator inconsistency (quaternion cross-check fails, counters disagree)
  is FAIL, because every other number from that run is then untrustworthy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from phoenix.real_world.failure_detector import (
    DEFAULT_ATTITUDE_INTERVENTION_RAD,
    SIM_ANALYSIS_PITCH_RAD,
    SIM_ANALYSIS_ROLL_RAD,
)
from phoenix.sim2real.go2_model import LIMIT_ABORT_BAND_RAD
from phoenix.sim2real.safety import MAX_DELTA_PER_STEP_RAD

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
LEVELS = (PASS, WARN, FAIL)
_RANK = {PASS: 0, WARN: 1, FAIL: 2}


def worst(*levels: str) -> str:
    """The most severe of ``levels`` (PASS if none)."""
    out = PASS
    for level in levels:
        if _RANK[level] > _RANK[out]:
            out = level
    return out


@dataclass(frozen=True)
class Band:
    """``value <= warn`` PASS, ``warn < value <= fail`` WARN, ``value > fail`` FAIL."""

    name: str
    warn: float
    fail: float
    unit: str
    justification: str

    def __post_init__(self) -> None:
        if not (math.isfinite(self.warn) and math.isfinite(self.fail)) or self.warn > self.fail:
            raise ValueError(f"band {self.name}: need finite warn <= fail")
        if len(self.justification.strip()) < 40:
            raise ValueError(f"band {self.name}: a threshold needs a written justification")

    def level(self, value: float | None) -> str:
        """Level of ``value``. ``None`` (not measured) is PASS here; callers that
        REQUIRE a measurement must check for ``None`` themselves."""
        if value is None:
            return PASS
        if not math.isfinite(value):
            return FAIL
        if value > self.fail:
            return FAIL
        if value > self.warn:
            return WARN
        return PASS


# --- Attitude ---------------------------------------------------------------

#: Tilt (angle of body z from world up) beyond which a SIM episode counts as a
#: fall even when the simulator did not terminate it (Isaac Lab's GO2 task only
#: terminates on trunk contact). Equals the larger of the historical sim
#: analysis bars so no recorded "success" is rescored by a stricter number.
SIM_FALL_TILT_RAD = max(SIM_ANALYSIS_PITCH_RAD, SIM_ANALYSIS_ROLL_RAD)

ATTITUDE = Band(
    name="max_tilt_rad",
    warn=DEFAULT_ATTITUDE_INTERVENTION_RAD - 0.15,
    fail=DEFAULT_ATTITUDE_INTERVENTION_RAD,
    unit="rad",
    justification=(
        "FAIL at 0.40 rad: the hardware safety gate (failure_detector."
        "DEFAULT_ATTITUDE_INTERVENTION_RAD) takes authority away from the policy there, so a "
        "sim episode that exceeds it would have been an intervention on the robot. WARN from "
        "0.25 rad: within 0.15 rad (under 9 deg) of that intervention."
    ),
)

# --- Safety-layer modification (the 'intervention rate') ---------------------

MODIFICATION_RATE = Band(
    name="modification_rate",
    warn=0.01,
    fail=0.05,
    unit="fraction of joint-ticks",
    justification=(
        "Fraction of joint-ticks where the executed target differs from the policy's own "
        "request (slew clip, limit clip, action clamp). FAIL above 5%: the repo's own Gate 7 "
        "bar for per-step slew saturation (configs/env/stand_v3.yaml, '<5% bar'); above it the "
        "safety layer, not the policy, is shaping the motion. WARN above 1%."
    ),
)

ACTION_SATURATION = Band(
    name="action_saturation_rate",
    warn=0.01,
    fail=0.05,
    unit="fraction of action elements",
    justification=(
        "Fraction of raw policy action elements with |a| > 1.0, the RslRlVecEnvWrapper "
        "clip_actions=1.0 used in training (ppo_runner.py) and evaluation. Every such element "
        "asked for more than the training envelope; same 5% / 1% bars as modification."
    ),
)

DEPLOY_SLEW_ACTIVATION = Band(
    name="deploy_slew_activation_rate",
    warn=0.01,
    fail=0.05,
    unit="fraction of joint-steps",
    justification=(
        f"Fraction of joint-steps where target minus measured q exceeds the {MAX_DELTA_PER_STEP_RAD}"
        " rad/step deploy slew cap (phoenix.sim2real.safety.MAX_DELTA_PER_STEP_RAD), i.e. where "
        "the Jetson gate WOULD modify the command. Same Gate 7 5% bar."
    ),
)

# --- Command tracking --------------------------------------------------------

LIN_VEL_TRACKING = Band(
    name="lin_vel_xy_rmse_mps",
    warn=0.15,
    fail=0.30,
    unit="m/s",
    justification=(
        "Planar velocity RMSE. The Isaac Lab tracking kernel is exp(-e^2/0.25) (std 0.5 m/s); "
        "at 0.30 m/s it pays 0.70 of full reward and 0.30 m/s is 30% of the 1.0 m/s max "
        "command. PASS below 0.15 m/s, about 1.6x the 0.091 m/s the v3b walking policy "
        "achieved (sim2real/mode_switch.py docstring)."
    ),
)

YAW_RATE_TRACKING = Band(
    name="yaw_rate_rmse_radps",
    warn=0.20,
    fail=0.40,
    unit="rad/s",
    justification=(
        "Yaw-rate RMSE. Same exp(-e^2/0.25) kernel; 0.40 rad/s is 40% of the 1.0 rad/s max "
        "yaw command and pays 0.53 of reward. PASS below 0.20 rad/s, about 2.3x the 0.087 "
        "rad/s v3b reference."
    ),
)

#: Achieved / commanded distance (or yaw) is only scored when the commanded
#: amount is at least this large; below it the ratio is noise.
MIN_COMMANDED_DISTANCE_M = 0.5
MIN_COMMANDED_YAW_RAD = 0.5

PROGRESS_RATIO_ERROR = Band(
    name="progress_ratio_error",
    warn=0.2,
    fail=0.4,
    unit="|achieved/commanded - 1|",
    justification=(
        "Integrated progress along the command. FAIL when achieved distance or yaw is off by "
        "more than 40% of what was commanded (a policy that walks at 60% of the requested "
        "speed is not tracking); WARN beyond 20%. Only scored when the command integrates to "
        "at least 0.5 m or 0.5 rad."
    ),
)

# --- Actuation ---------------------------------------------------------------

TRACKING_DEVIATION = Band(
    name="executed_target_vs_measured_rms_rad",
    warn=MAX_DELTA_PER_STEP_RAD,
    fail=2 * MAX_DELTA_PER_STEP_RAD,
    unit="rad",
    justification=(
        "RMS of executed joint target minus the next measured joint position. Above one slew "
        "cap (0.175 rad) the next command is clipped against measured q, so the actuators are "
        "not following the policy; above two caps the joint is effectively not tracking."
    ),
)

#: A requested target further than this past a hard URDF limit is an illegal
#: target (the bridge's target_beyond_limit abort). go2_model.LIMIT_ABORT_BAND_RAD.
ILLEGAL_TARGET_BAND_RAD = LIMIT_ABORT_BAND_RAD

#: Required fraction of the planned duration. Anything shorter did not complete.
MIN_DURATION_FRACTION = 1.0

THRESHOLDS: tuple[Band, ...] = (
    ATTITUDE,
    MODIFICATION_RATE,
    ACTION_SATURATION,
    DEPLOY_SLEW_ACTIVATION,
    LIN_VEL_TRACKING,
    YAW_RATE_TRACKING,
    PROGRESS_RATIO_ERROR,
    TRACKING_DEVIATION,
)

HARD_FAIL_RULES: dict[str, str] = {
    "fall": (
        "terminated on base contact or bad orientation, or tilt above "
        f"{SIM_FALL_TILT_RAD} rad (historical sim bars pitch 0.8 / roll 0.6)"
    ),
    "numerical_failure": "any NaN or Inf in state, observation or action",
    "illegal_target": (
        f"any requested joint target beyond a hard URDF limit by more than {ILLEGAL_TARGET_BAND_RAD}"
        " rad (go2_model.LIMIT_ABORT_BAND_RAD, the bridge's abort band)"
    ),
    "evaluator_inconsistency": (
        "projected gravity recomputed from root_quat_w disagrees with the simulator's, or the "
        "evaluator's own counters disagree; nothing else from the run can be trusted"
    ),
    "incomplete": "episode ended before its planned duration for any reason",
}


def thresholds_table() -> list[dict[str, object]]:
    """Machine-readable copy of every threshold, for stamping into result files."""
    rows: list[dict[str, object]] = [
        {
            "name": b.name,
            "warn": b.warn,
            "fail": b.fail,
            "unit": b.unit,
            "justification": b.justification,
        }
        for b in THRESHOLDS
    ]
    rows.append({"name": "sim_fall_tilt_rad", "fail": SIM_FALL_TILT_RAD, "unit": "rad"})
    rows.append({"name": "illegal_target_band_rad", "fail": ILLEGAL_TARGET_BAND_RAD, "unit": "rad"})
    return rows


__all__ = [
    "ACTION_SATURATION",
    "ATTITUDE",
    "Band",
    "DEPLOY_SLEW_ACTIVATION",
    "FAIL",
    "HARD_FAIL_RULES",
    "ILLEGAL_TARGET_BAND_RAD",
    "LEVELS",
    "LIN_VEL_TRACKING",
    "MIN_COMMANDED_DISTANCE_M",
    "MIN_COMMANDED_YAW_RAD",
    "MIN_DURATION_FRACTION",
    "MODIFICATION_RATE",
    "PASS",
    "PROGRESS_RATIO_ERROR",
    "SIM_FALL_TILT_RAD",
    "THRESHOLDS",
    "TRACKING_DEVIATION",
    "WARN",
    "YAW_RATE_TRACKING",
    "thresholds_table",
    "worst",
]
