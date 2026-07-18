"""Unit tests for the Simplex safety arbiter (Phase 2)."""

from __future__ import annotations

import math

import pytest

from phoenix.reliability.arbiter import (
    ArbiterOutput,
    ShieldState,
    SimplexArbiter,
    SimplexArbiterCfg,
)


def _cfg(**kw) -> SimplexArbiterCfg:
    base = dict(
        trip_threshold=10.0,
        clear_threshold=3.0,
        trip_persistence=3,
        clear_persistence=5,
        handoff_ticks=4,
        recover_ticks=6,
        min_fallback_ticks=2,
    )
    base.update(kw)
    return SimplexArbiterCfg(**base)


def _run(arb: SimplexArbiter, score: float, n: int) -> ArbiterOutput:
    out = ArbiterOutput(arb.state, arb.blend)
    for _ in range(n):
        out = arb.update(score)
    return out


# --- config validation ------------------------------------------------------


def test_cfg_requires_hysteresis_ordering():
    with pytest.raises(ValueError):
        SimplexArbiterCfg(trip_threshold=5.0, clear_threshold=5.0)
    with pytest.raises(ValueError):
        SimplexArbiterCfg(trip_threshold=5.0, clear_threshold=6.0)


def test_cfg_rejects_nonpositive_counts():
    with pytest.raises(ValueError):
        _cfg(handoff_ticks=0)


# --- persistence / dwell ----------------------------------------------------


def test_single_spike_does_not_trip():
    arb = SimplexArbiter(_cfg(trip_persistence=3))
    arb.update(0.0)
    out = arb.update(99.0)  # one spike, persistence is 3
    assert out.state is ShieldState.NOMINAL
    out = arb.update(0.0)  # streak broken
    assert out.state is ShieldState.NOMINAL


def test_sustained_over_trip_engages_after_persistence():
    arb = SimplexArbiter(_cfg(trip_persistence=3))
    assert arb.update(20.0).state is ShieldState.NOMINAL
    assert arb.update(20.0).state is ShieldState.NOMINAL
    # third consecutive over-trip tick starts the handoff
    assert arb.update(20.0).state is ShieldState.HANDOFF


# --- bounded ramp -----------------------------------------------------------


def test_handoff_blend_ramps_monotonically_to_one():
    cfg = _cfg(trip_persistence=1, handoff_ticks=4)
    arb = SimplexArbiter(cfg)
    arb.update(20.0)  # trips immediately (persistence 1) -> HANDOFF
    blends = []
    for _ in range(4):
        out = arb.update(20.0)
        blends.append(out.blend)
    assert blends == sorted(blends)  # monotonic non-decreasing
    assert blends[-1] == pytest.approx(1.0)
    assert arb.state is ShieldState.FALLBACK


def test_no_instant_stand_first_handoff_tick_is_partial():
    cfg = _cfg(trip_persistence=1, handoff_ticks=5)
    arb = SimplexArbiter(cfg)
    arb.update(20.0)  # -> HANDOFF
    first = arb.update(20.0)
    assert 0.0 < first.blend < 1.0  # never a snap to full fallback


# --- fallback dwell + recovery ----------------------------------------------


def test_recovers_after_dwell_and_clear_persistence():
    cfg = _cfg(trip_persistence=1, handoff_ticks=2, min_fallback_ticks=2, clear_persistence=3, recover_ticks=3)
    arb = SimplexArbiter(cfg)
    _run(arb, 20.0, 1 + 2)  # trip + finish handoff -> FALLBACK
    assert arb.state is ShieldState.FALLBACK
    # low scores: must satisfy both dwell and clear-persistence before recovery
    _run(arb, 0.0, 3)
    assert arb.state in (ShieldState.RECOVERING, ShieldState.NOMINAL)
    out = _run(arb, 0.0, 5)
    assert out.state is ShieldState.NOMINAL
    assert out.blend == pytest.approx(0.0)


def test_latch_never_recovers():
    cfg = _cfg(trip_persistence=1, handoff_ticks=2, latch=True)
    arb = SimplexArbiter(cfg)
    _run(arb, 20.0, 3)
    assert arb.state is ShieldState.FALLBACK
    out = _run(arb, 0.0, 100)  # would clear if not latched
    assert out.state is ShieldState.FALLBACK
    assert out.blend == pytest.approx(1.0)


def test_hysteresis_midband_does_not_release():
    # score between clear (3) and trip (10): not below clear, so never releases.
    cfg = _cfg(trip_persistence=1, handoff_ticks=2, min_fallback_ticks=1, clear_persistence=2)
    arb = SimplexArbiter(cfg)
    _run(arb, 20.0, 3)
    assert arb.state is ShieldState.FALLBACK
    out = _run(arb, 5.0, 50)  # in the hysteresis band
    assert out.state is ShieldState.FALLBACK


# --- re-trip during recovery ------------------------------------------------


def test_retrip_during_recovery_snaps_back_to_fallback():
    cfg = _cfg(trip_persistence=1, handoff_ticks=2, min_fallback_ticks=1, clear_persistence=2, recover_ticks=10)
    arb = SimplexArbiter(cfg)
    _run(arb, 20.0, 3)
    _run(arb, 0.0, 3)  # enter RECOVERING
    assert arb.state is ShieldState.RECOVERING
    out = arb.update(20.0)  # trip again mid-recovery
    assert out.state is ShieldState.FALLBACK
    assert out.blend == pytest.approx(1.0)


# --- fail toward safe -------------------------------------------------------


def test_nonfinite_score_counts_as_trip():
    arb = SimplexArbiter(_cfg(trip_persistence=2))
    assert arb.update(math.inf).state is ShieldState.NOMINAL
    assert arb.update(math.nan).state is ShieldState.HANDOFF  # 2 consecutive -> engage


def test_nonfinite_score_never_clears():
    cfg = _cfg(trip_persistence=1, handoff_ticks=2, min_fallback_ticks=1, clear_persistence=1)
    arb = SimplexArbiter(cfg)
    _run(arb, 20.0, 3)
    assert arb.state is ShieldState.FALLBACK
    out = _run(arb, math.nan, 20)
    assert out.state is ShieldState.FALLBACK


# --- output helper ----------------------------------------------------------


def test_engaged_flag():
    arb = SimplexArbiter(_cfg(trip_persistence=1))
    assert not arb.update(0.0).engaged
    assert arb.update(20.0).engaged  # HANDOFF counts as engaged
