"""Group-level actuator-response shift (Phase H).

These tests pin the property the whole design rests on: because a group's statistic is a
MEDIAN over its members, a group flags only when a MAJORITY of its joints moved, so the
LARGEST flagged group is the true extent rather than the most extreme-looking subset.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.monitor.residual import Baseline, WindowStats
from phoenix.monitor.response_shift import (
    CANDIDATE_GROUPS,
    DEFAULT_SHIFT_CONFIG,
    ShiftConfig,
    ShiftState,
    assess_shift,
    calibrate_groups,
)
from phoenix.sim2real.go2_model import UNITREE_MOTOR_ORDER

N = len(UNITREE_MOTOR_ORDER)
REF = np.full(N, 0.10)  # nominal reference RMS, rad
IDX = {n: i for i, n in enumerate(UNITREE_MOTOR_ORDER)}


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


def _stats(scale: np.ndarray, windows: int = 30, noise: float = 0.0, seed: int = 0) -> WindowStats:
    """Windows whose RMS is ``REF / scale``, i.e. authority ratio ``s_hat = scale``."""
    rng = np.random.default_rng(seed)
    rms = REF[None, :] / scale[None, :]
    if noise:
        rms = rms * (1.0 + noise * rng.standard_normal((windows, N)))
    if rms.shape[0] == 1:
        rms = np.repeat(rms, windows, axis=0)
    return WindowStats(
        t_start_s=np.arange(windows, dtype=float),
        rms_error=np.abs(rms),
        mean_error=np.zeros((windows, N)),
        valid_fraction=np.ones((windows, N)),
        safety_altered_fraction=np.zeros((windows, N)),
        torque_gain_ratio=np.full((windows, N), np.nan),
    )


def _nominal_calibration(noise: float = 0.10, cfg: ShiftConfig = DEFAULT_SHIFT_CONFIG):
    base = _baseline()
    stats = [_stats(np.ones(N), noise=noise, seed=s) for s in range(8)]
    return base, calibrate_groups(stats, base, regime="walk", cfg=cfg)


def _scale_for(group: str, s: float) -> np.ndarray:
    v = np.ones(N)
    v[list(CANDIDATE_GROUPS[group])] = s
    return v


def test_nominal_session_is_not_flagged():
    base, gb = _nominal_calibration()
    rep = assess_shift(_stats(np.ones(N), noise=0.10, seed=99), base, gb)
    assert not rep.shifted
    assert rep.selected is None
    assert all(g.state != ShiftState.SHIFTED.value for g in rep.groups)


@pytest.mark.parametrize(("group", "s"), [("all", 0.75), ("rear", 0.75), ("leg_RR", 0.75)])
def test_the_largest_flagged_group_is_the_true_extent(group, s):
    """The property the selection rule depends on."""
    base, gb = _nominal_calibration()
    rep = assess_shift(_stats(_scale_for(group, s), noise=0.05, seed=7), base, gb)
    assert rep.shifted
    assert rep.selected is not None
    assert rep.selected.group == group


def test_a_global_change_does_not_read_as_one_bad_motor():
    """The v1 failure mode: a global shift must not be reported as a single joint."""
    base, gb = _nominal_calibration()
    rep = assess_shift(_stats(_scale_for("all", 0.75), noise=0.05, seed=3), base, gb)
    assert rep.selected.group == "all"
    assert len(rep.selected.joints) == N


def test_a_one_leg_change_does_not_spill_into_the_whole_robot():
    """Three of twelve joints cannot move the median of `all` or of `rear`."""
    base, gb = _nominal_calibration()
    rep = assess_shift(_stats(_scale_for("leg_RR", 0.70), noise=0.05, seed=4), base, gb)
    by = {g.group: g for g in rep.groups}
    assert by["leg_RR"].state == ShiftState.SHIFTED.value
    assert by["all"].state == ShiftState.NOMINAL.value
    assert by["rear"].state == ShiftState.NOMINAL.value
    assert rep.selected.group == "leg_RR"


def test_severity_and_range_track_the_applied_scale():
    base, gb = _nominal_calibration()
    for s in (0.90, 0.80, 0.70):
        rep = assess_shift(_stats(_scale_for("all", s), noise=0.03, seed=11), base, gb)
        assert rep.selected.shift == pytest.approx(s, abs=0.03)
        assert rep.selected.lo <= s <= rep.selected.hi


def test_severity_is_monotone_in_the_applied_scale():
    base, gb = _nominal_calibration()
    shifts = [
        assess_shift(_stats(_scale_for("all", s), noise=0.03, seed=5), base, gb).selected.shift
        for s in (0.90, 0.85, 0.75)
    ]
    assert shifts[0] > shifts[1] > shifts[2]


def test_a_change_smaller_than_min_effect_is_not_flagged():
    """The floor exists so the monitor cannot alarm on a change too small to adapt to."""
    base, gb = _nominal_calibration()
    rep = assess_shift(_stats(_scale_for("all", 0.97), noise=0.02, seed=12), base, gb)
    assert not rep.shifted


def test_an_incoherent_group_is_not_flagged_however_far_its_median_falls():
    """Exactly half of `rear` moving pulls its median down, but its members disagree."""
    base, gb = _nominal_calibration()
    rep = assess_shift(_stats(_scale_for("leg_RR", 0.70), noise=0.05, seed=4), base, gb)
    by = {g.group: g for g in rep.groups}
    assert by["rear"].member_spread > by["rear"].spread_bound
    assert by["rear"].coherent is False
    assert by["rear"].state == ShiftState.NOMINAL.value
    assert by["leg_RR"].coherent is True


def test_member_spread_is_small_for_a_uniform_reduction():
    """Phase H's consistency-across-joints term."""
    base, gb = _nominal_calibration()
    rep = assess_shift(_stats(_scale_for("all", 0.75), noise=0.02, seed=6), base, gb)
    assert rep.selected.member_spread < 0.15


def test_threshold_is_never_closer_to_one_than_min_effect():
    """A tight calibration set must not license flagging a change too small to matter."""
    base, gb = _nominal_calibration(noise=0.0)
    assert all(t <= 1.0 - DEFAULT_SHIFT_CONFIG.min_effect + 1e-12 for t in gb.threshold.values())


def test_too_few_usable_windows_is_insufficient_data_not_nominal():
    base, gb = _nominal_calibration()
    short = _stats(_scale_for("all", 0.75), windows=3, noise=0.0)
    rep = assess_shift(short, base, gb)
    assert all(g.state == ShiftState.INSUFFICIENT_DATA.value for g in rep.groups)
    assert not rep.shifted


def test_calibration_refuses_an_empty_session_set():
    with pytest.raises(ValueError):
        calibrate_groups([], _baseline(), regime="walk")


def test_baseline_round_trips_through_json():
    import json

    _base, gb = _nominal_calibration()
    d = json.loads(json.dumps(gb.to_dict()))
    assert d["schema"] == "phoenix-group-baseline/v1"
    assert d["regime"] == "walk"
    assert set(d["threshold"]) == set(CANDIDATE_GROUPS)
    assert d["config"]["alpha"] == DEFAULT_SHIFT_CONFIG.alpha


def test_report_round_trips_through_json():
    import json

    base, gb = _nominal_calibration()
    rep = assess_shift(_stats(_scale_for("all", 0.75), noise=0.03, seed=8), base, gb)
    d = json.loads(json.dumps(rep.to_dict()))
    assert d["schema"] == "phoenix-response-shift/v1"
    assert d["shifted"] is True
    assert d["selected"]["group"] == "all"
    assert len(d["per_joint_median"]) == N


def test_the_hypothesis_space_is_small_fixed_and_physical():
    """It must not be a search over every subset."""
    assert len(CANDIDATE_GROUPS) < 30
    assert "all" in CANDIDATE_GROUPS and len(CANDIDATE_GROUPS["all"]) == N
    for name, members in CANDIDATE_GROUPS.items():
        assert len(set(members)) == len(members), name
        assert all(0 <= i < N for i in members), name
