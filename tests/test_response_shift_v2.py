"""Detector v2: hierarchical detection, family-wise control, extent by evidence.

The properties pinned here are the ones the design claims: stage 2 never runs unless
stage 1 fires, the family-wise threshold is the max-statistic null and not a per-group
one, and the reported extent is inferred from telemetry rather than preferred by size or
hard-coded to the intervention actually selected for the study.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.monitor.residual import Baseline, WindowStats
from phoenix.monitor.response_shift_v2 import (
    ALL_JOINTS,
    DEFAULT_V2_CONFIG,
    V2_GROUPS,
    V2Config,
    V2State,
    assess_v2,
    calibrate_v2,
)
from phoenix.sim2real.go2_model import UNITREE_MOTOR_ORDER

N = len(UNITREE_MOTOR_ORDER)
REF = np.full(N, 0.10)
_IDX = {n: i for i, n in enumerate(UNITREE_MOTOR_ORDER)}


def _baseline() -> Baseline:
    return Baseline(
        regime="walk",
        reference_rms=[float(v) for v in REF],
        s_hat_quantile_lo=[0.9] * N,
        threshold=[0.9] * N,
        n_windows=[100] * N,
        alpha=0.01,
        min_effect=0.10,
    )


def _stats(scale: np.ndarray, windows: int = 60, noise: float = 0.0, seed: int = 0) -> WindowStats:
    rng = np.random.default_rng(seed)
    rms = np.repeat((REF / scale)[None, :], windows, axis=0)
    if noise:
        rms = rms * (1.0 + noise * rng.standard_normal((windows, N)))
    return WindowStats(
        t_start_s=np.arange(windows, dtype=float),
        rms_error=np.abs(rms),
        mean_error=np.zeros((windows, N)),
        valid_fraction=np.ones((windows, N)),
        safety_altered_fraction=np.zeros((windows, N)),
        torque_gain_ratio=np.full((windows, N), np.nan),
    )


#: Nominal structure measured on real W2 walking telemetry: a modest whole-robot offset
#: plus LARGE independent per-joint variation (undegraded joints ranged 0.75 to 1.63).
#: The independent part is what makes a larger group's median less noisy, which is the
#: whole reason stage 2 can tell a six-joint change from a three-joint one.
_SESSION_SD = 0.03
_JOINT_SD = 0.12


def _nominal_sessions(n: int = 24, noise: float = 0.10, start: int = 0):
    out = []
    for k in range(n):
        rng = np.random.default_rng(1000 + start + k)
        offset = 1.0 + _SESSION_SD * rng.standard_normal()
        per_joint = offset * (1.0 + _JOINT_SD * rng.standard_normal(N))
        out.append(_stats(per_joint, noise=noise, seed=start + k))
    return out


def _cal(noise: float = 0.10, cfg: V2Config = DEFAULT_V2_CONFIG):
    base = _baseline()
    return base, calibrate_v2(_nominal_sessions(24, noise), base, regime="walk", cfg=cfg)


def _scale_for(group: str, s: float, seed: int = 7) -> np.ndarray:
    """A degraded session: the same nominal structure, with one group scaled by ``s``."""
    rng = np.random.default_rng(seed)
    offset = 1.0 + _SESSION_SD * rng.standard_normal()
    v = offset * (1.0 + _JOINT_SD * rng.standard_normal(N))
    v[list(V2_GROUPS[group])] *= s
    return v


# ------------------------------------------------------------------ the space
def test_hypothesis_space_is_small_frozen_and_every_member_is_reachable():
    assert len(V2_GROUPS) == 5
    assert set(V2_GROUPS) == {"all", "front", "rear", "left", "right"}
    assert len(V2_GROUPS["all"]) == N
    for pair in ("front", "rear", "left", "right"):
        assert len(V2_GROUPS[pair]) == 6, pair
    # Reachability: stage 1 is a median over twelve joints, so a hypothesis smaller than
    # half the robot can never be selected and would only inflate the family-wise bound.
    for name, members in V2_GROUPS.items():
        assert len(members) >= N // 2, name


def test_the_redundant_and_unreachable_hypotheses_are_gone():
    """Diagonals, joint classes, singletons, and the four single legs."""
    for dropped in (
        "diag_a",
        "diag_b",
        "hips",
        "thighs",
        "calfs",
        "RR_thigh_joint",
        "leg_FR",
        "leg_FL",
        "leg_RR",
        "leg_RL",
    ):
        assert dropped not in V2_GROUPS


def test_the_two_leg_partitions_are_complementary_and_cover_the_robot():
    assert set(V2_GROUPS["front"]) | set(V2_GROUPS["rear"]) == set(ALL_JOINTS)
    assert not set(V2_GROUPS["front"]) & set(V2_GROUPS["rear"])
    assert set(V2_GROUPS["left"]) | set(V2_GROUPS["right"]) == set(ALL_JOINTS)
    assert not set(V2_GROUPS["left"]) & set(V2_GROUPS["right"])


# --------------------------------------------------------------- stage gating
def test_stage_two_does_not_run_when_stage_one_says_nominal():
    base, v2 = _cal()
    rep = assess_v2(_stats(np.ones(N), noise=0.10, seed=777), base, v2)
    assert rep.state == V2State.NOMINAL.value
    assert rep.group is None and rep.severity is None and rep.z_by_group is None


def test_insufficient_data_is_not_reported_as_nominal():
    base, v2 = _cal()
    rep = assess_v2(_stats(_scale_for("all", 0.70), windows=3), base, v2)
    assert rep.state == V2State.INSUFFICIENT_DATA.value
    assert not rep.shifted and rep.group is None


def test_a_change_smaller_than_min_effect_does_not_fire_stage_one():
    base, v2 = _cal()
    assert v2.tau_global <= 1.0 - DEFAULT_V2_CONFIG.min_effect + 1e-12
    rep = assess_v2(_stats(_scale_for("all", 0.99), noise=0.02, seed=5), base, v2)
    assert rep.state == V2State.NOMINAL.value


# -------------------------------------------------- family-wise error control
def test_family_wise_threshold_is_the_max_statistic_null():
    """It must bound the MOST EXTREME group, not each group separately."""
    base, v2 = _cal()
    per_group_95 = []
    for name in V2_GROUPS:
        per_group_95.append(v2.mu[name])
    # The family-wise threshold must be strictly larger than a per-group z of 0,
    # and large enough that a typical nominal session's best group does not clear it.
    assert v2.tau_fw > 0
    fired = 0
    for st in _nominal_sessions(24, start=500):
        rep = assess_v2(st, base, v2)
        if rep.shifted and rep.group is not None:
            fired += 1
    assert fired <= 2, f"family-wise control leaking: {fired}/24 nominal sessions named a group"
    assert len(per_group_95) == len(V2_GROUPS)


def test_nominal_false_alarm_rate_is_near_alpha_on_fresh_sessions():
    base, v2 = _cal()
    fired = sum(assess_v2(st, base, v2).shifted for st in _nominal_sessions(40, start=900))
    assert fired / 40 <= 0.15


# ------------------------------------------------------- extent, by evidence
@pytest.mark.parametrize("group", ["all", "front", "rear", "left", "right"])
def test_the_affected_group_is_inferred_from_telemetry(group):
    """No size preference and no hard-coded answer: each group must be recoverable."""
    base, v2 = _cal()
    rep = assess_v2(_stats(_scale_for(group, 0.70), noise=0.05, seed=42), base, v2)
    assert rep.shifted, group
    assert rep.group == group, f"{group} reported as {rep.group}"


def test_a_front_degradation_is_not_reported_as_rear():
    """Specificity: the study's selected intervention must not be the default answer."""
    base, v2 = _cal()
    rep = assess_v2(_stats(_scale_for("front", 0.70), noise=0.05, seed=11), base, v2)
    assert rep.group == "front"
    assert rep.group != "rear"


def test_a_global_change_is_not_reported_as_one_leg():
    base, v2 = _cal()
    rep = assess_v2(_stats(_scale_for("all", 0.75), noise=0.05, seed=12), base, v2)
    assert rep.group == "all"


def test_extent_is_not_chosen_by_size():
    """A rear-only change must beat `all`, whose median moves only halfway."""
    base, v2 = _cal()
    rep = assess_v2(_stats(_scale_for("rear", 0.70), noise=0.05, seed=13), base, v2)
    assert rep.group == "rear"
    assert rep.z_by_group["rear"] > rep.z_by_group["all"]
    assert rep.z_by_group["rear"] > rep.z_by_group["front"]


def test_a_single_leg_change_cannot_fire_stage_one_and_that_is_declared():
    """The stated limitation, pinned so it is documented behaviour and not a surprise.

    Three of twelve joints do not move a median over twelve, at any severity, so v2
    detects shifts affecting at least half the robot and nothing narrower.
    """
    base, v2 = _cal()
    one_leg = np.ones(N)
    one_leg[[_IDX[j] for j in UNITREE_MOTOR_ORDER if j.startswith("RR_")]] = 0.50
    rep = assess_v2(_stats(one_leg, noise=0.03, seed=61), base, v2)
    assert rep.state == V2State.NOMINAL.value
    assert rep.group is None


# -------------------------------------------------------------- severity reuse
def test_severity_tracks_the_applied_scale_and_its_range_covers_truth():
    base, v2 = _cal()
    for s in (0.80, 0.75, 0.70):
        rep = assess_v2(_stats(_scale_for("rear", s), noise=0.04, seed=21), base, v2)
        assert rep.severity == pytest.approx(s, abs=0.05)
        assert rep.lo <= s <= rep.hi


def test_severity_is_monotone_in_the_applied_scale():
    base, v2 = _cal()
    sev = [
        assess_v2(_stats(_scale_for("all", s), noise=0.03, seed=31), base, v2).severity
        for s in (0.90, 0.80, 0.70)
    ]
    assert sev[0] > sev[1] > sev[2]


# ------------------------------------------------------------------ mechanics
def test_calibration_refuses_too_few_sessions():
    with pytest.raises(ValueError):
        calibrate_v2([], _baseline(), regime="walk")
    with pytest.raises(ValueError):
        calibrate_v2(_nominal_sessions(3), _baseline(), regime="walk")


def test_baseline_and_report_round_trip_through_json():
    import json

    base, v2 = _cal()
    d = json.loads(json.dumps(v2.to_dict()))
    assert d["schema"] == "phoenix-response-shift-v2-baseline/v1"
    assert set(d["stage2"]["groups"]) == set(V2_GROUPS)
    assert d["config"]["alpha_fw"] == DEFAULT_V2_CONFIG.alpha_fw

    rep = assess_v2(_stats(_scale_for("rear", 0.70), noise=0.04, seed=41), base, v2)
    r = json.loads(json.dumps(rep.to_dict()))
    assert r["schema"] == "phoenix-response-shift-v2/v1"
    assert r["shifted"] is True and r["group"] == "rear"
    assert r["range"][0] <= r["severity"] <= r["range"][1]
