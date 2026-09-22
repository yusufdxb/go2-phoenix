"""Recipe W3 curriculum stage machine: both directions must qualify, no promotion on time."""

from __future__ import annotations

import pytest

from phoenix.sim_env.command_curriculum import CurriculumSpec, CurriculumState

SPEC = CurriculumSpec(window=100, min_segments_per_direction=50, check_every_steps=10,
                      min_stage_steps=20, max_stage_steps=100, consecutive_checks=2)


def _feed(st, fwd_ratio, bwd_ratio, n=60):
    v = st.factor * 0.8  # above min_cmd_frac * stage max
    for _ in range(n):
        st.record_segment(v, fwd_ratio * v, 50)
        st.record_segment(-v, bwd_ratio * -v, 50)


def test_backward_alone_never_advances_and_times_out():
    st = CurriculumState(SPEC)
    for step in range(0, 200, 10):
        _feed(st, fwd_ratio=0.05, bwd_ratio=1.0)
        assert st.maybe_advance(step) is False
    assert st.stage == 0 and st.timed_out


def test_both_directions_advance_after_consecutive_checks_and_min_duration():
    st = CurriculumState(SPEC)
    advanced = []
    for step in range(10, 400, 10):
        _feed(st, 0.9, 0.9)
        if st.maybe_advance(step):
            advanced.append((step, st.stage))
    assert [s for _, s in advanced] == [1, 2, 3]
    assert st.final and st.factor == 1.0
    assert all(b - a >= SPEC.min_stage_steps for (a, _), (b, _) in zip([(0, 0)] + advanced, advanced))


def test_small_commands_do_not_count_and_ratios_need_enough_segments():
    st = CurriculumState(SPEC)
    for _ in range(80):
        st.record_segment(0.05, 0.05, 50)  # below 0.5 * 0.2
    assert st.ratios() == (None, None)


def test_runaway_segment_is_clipped():
    st = CurriculumState(CurriculumSpec(window=10, min_segments_per_direction=1))
    st.record_segment(0.15, 3.0, 50)
    assert st.ratios()[0] == pytest.approx(2.0)


def test_spec_validates_factors():
    with pytest.raises(ValueError):
        CurriculumSpec(factors=(0.4, 0.2, 1.0))
