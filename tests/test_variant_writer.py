"""Per-variant trajectory writing: the properties loop closure depends on.

``scripts/loop_closure.sh`` treats "number of parquets the replay produced" as
a hard gate, and the curriculum then seeds training from those files. So the
things that must hold are not cosmetic: one file is exactly one episode, a file
that exists has rows in it, the frame is declared, and a replayed variant is
labelled as a simulator capture rather than inheriting the hardware defaults.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from phoenix.real_world.failure_detector import SIM_ANALYSIS_PITCH_RAD
from phoenix.real_world.trajectory_logger import (
    CAPTURE_SOURCE_SIM,
    POSITION_FRAME_ENV_LOCAL,
)
from phoenix.replay.trajectory_reader import TrajectoryReader
from phoenix.replay.variant_writer import VariantTrajectoryWriter

N = 3
DT = 0.02


def upright(n=N):
    q = np.zeros((n, 4), dtype=np.float32)
    q[:, 3] = 1.0  # xyzw identity
    return q


def tipped(pitch_rad, n=N):
    """xyzw quaternion for a pure pitch rotation."""
    q = np.zeros((n, 4), dtype=np.float32)
    q[:, 1] = np.sin(pitch_rad / 2.0)
    q[:, 3] = np.cos(pitch_rad / 2.0)
    return q


def step_kwargs(*, quat=None, n=N, cmd=None, actual=None):
    return {
        "base_pos": np.zeros((n, 3), dtype=np.float32),
        "base_quat_xyzw": upright(n) if quat is None else quat,
        "base_lin_vel_body": (np.zeros((n, 3), dtype=np.float32) if actual is None else actual),
        "base_ang_vel_body": np.zeros((n, 3), dtype=np.float32),
        "joint_pos": np.zeros((n, 12), dtype=np.float32),
        "joint_vel": np.zeros((n, 12), dtype=np.float32),
        "command_vel": np.zeros((n, 3), dtype=np.float32) if cmd is None else cmd,
        "action": np.zeros((n, 12), dtype=np.float32),
        "contact_forces": np.zeros((n, 4), dtype=np.float32),
    }


def write(tmp_path, steps, **init):
    w = VariantTrajectoryWriter(tmp_path, N, control_dt=DT, **init)
    for i in range(steps):
        w.append_step(i, **step_kwargs())
    return w


# ------------------------------------------------------------------ basics
def test_writes_one_parquet_per_env(tmp_path):
    results = write(tmp_path, 4).close()
    assert len(results) == N
    assert sorted(p.name for p in tmp_path.glob("*.parquet")) == [
        "variant_000.parquet",
        "variant_001.parquet",
        "variant_002.parquet",
    ]
    assert {r.rows for r in results} == {4}


def test_rejects_a_nonpositive_env_count(tmp_path):
    with pytest.raises(ValueError, match="num_envs must be positive"):
        VariantTrajectoryWriter(tmp_path, 0, control_dt=DT)


def test_rejects_a_nonpositive_control_dt(tmp_path):
    with pytest.raises(ValueError, match="control_dt must be finite and positive"):
        VariantTrajectoryWriter(tmp_path, N, control_dt=0.0)


def test_a_zero_row_run_writes_no_files(tmp_path):
    # An empty parquet would still be counted as a variant by loop_closure.sh
    # and would seed nothing, so it must not exist at all.
    results = VariantTrajectoryWriter(tmp_path, N, control_dt=DT).close()
    assert results == []
    assert list(tmp_path.glob("*.parquet")) == []


def test_close_is_idempotent(tmp_path):
    w = write(tmp_path, 2)
    assert w.close() == w.close()


def test_append_after_close_is_refused(tmp_path):
    w = write(tmp_path, 1)
    w.close()
    with pytest.raises(RuntimeError, match="closed VariantTrajectoryWriter"):
        w.append_step(1, **step_kwargs())


# ------------------------------------------------------- provenance / frame
def test_rows_are_labelled_as_simulator_captures(tmp_path):
    write(tmp_path, 3).close()
    reader = TrajectoryReader(tmp_path / "variant_000.parquet")
    assert set(reader.column("capture_source")) == {CAPTURE_SOURCE_SIM}


def test_the_position_frame_is_declared_not_inferred(tmp_path):
    write(tmp_path, 3).close()
    reader = TrajectoryReader(tmp_path / "variant_000.parquet")
    assert reader.declared_position_frame == POSITION_FRAME_ENV_LOCAL
    assert reader.resolve_position_frame() == (
        POSITION_FRAME_ENV_LOCAL,
        "declared_by_source",
    )


def test_timestamps_follow_the_control_period(tmp_path):
    write(tmp_path, 3).close()
    reader = TrajectoryReader(tmp_path / "variant_001.parquet")
    assert np.allclose(reader.column("timestamp_s"), [0.0, DT, 2 * DT])


# --------------------------------------------------------- input validation
def test_a_wrong_width_is_refused(tmp_path):
    w = VariantTrajectoryWriter(tmp_path, N, control_dt=DT)
    kwargs = step_kwargs()
    kwargs["joint_pos"] = np.zeros((N, 11), dtype=np.float32)
    with pytest.raises(ValueError, match=r"joint_pos must have shape \(3, 12\)"):
        w.append_step(0, **kwargs)
    w.close()


def test_a_non_finite_value_is_refused(tmp_path):
    w = VariantTrajectoryWriter(tmp_path, N, control_dt=DT)
    kwargs = step_kwargs()
    kwargs["base_pos"] = np.full((N, 3), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="base_pos contains non-finite"):
        w.append_step(0, **kwargs)
    w.close()


def test_a_degenerate_quaternion_is_refused(tmp_path):
    w = VariantTrajectoryWriter(tmp_path, N, control_dt=DT)
    with pytest.raises(ValueError, match="quaternion must be finite and nonzero"):
        w.append_step(0, **step_kwargs(quat=np.zeros((N, 4), dtype=np.float32)))
    w.close()


# ------------------------------------------------------- failure labelling
def test_an_attitude_failure_is_labelled(tmp_path):
    w = VariantTrajectoryWriter(tmp_path, N, control_dt=DT)
    w.append_step(0, **step_kwargs())
    w.append_step(1, **step_kwargs(quat=tipped(SIM_ANALYSIS_PITCH_RAD + 0.3)))
    results = w.close()
    assert all(r.failed for r in results)
    assert {r.failure_mode for r in results} == {"attitude"}
    assert {r.failure_step for r in results} == {1}


def test_a_tilt_below_the_sim_bar_is_not_a_failure(tmp_path):
    # 0.5 rad is past the 0.40 rad HARDWARE intervention and below the 0.8 rad
    # sim analysis bar. Scoring sim rollouts at the hardware threshold is the
    # coupling this writer must not reintroduce.
    w = VariantTrajectoryWriter(tmp_path, N, control_dt=DT)
    w.append_step(0, **step_kwargs(quat=tipped(0.5)))
    results = w.close()
    assert not any(r.failed for r in results)


def test_failure_rows_carry_the_detector_as_the_onset_source(tmp_path):
    w = VariantTrajectoryWriter(tmp_path, N, control_dt=DT)
    w.append_step(0, **step_kwargs())
    w.append_step(1, **step_kwargs(quat=tipped(1.2)))
    w.close()
    reader = TrajectoryReader(tmp_path / "variant_000.parquet")
    assert list(reader.column("failure_flag")) == [False, True]
    assert list(reader.column("failure_onset_source")) == [None, "detector"]


def test_collapse_is_unavailable_without_a_height_source(tmp_path):
    w = VariantTrajectoryWriter(tmp_path, N, control_dt=DT)
    for i in range(3):
        w.append_step(i, **step_kwargs())  # base_height defaults to None
    assert not any(r.failed for r in w.close())


def test_collapse_fires_when_a_height_source_is_supplied(tmp_path):
    w = VariantTrajectoryWriter(tmp_path, N, control_dt=DT)
    low = np.full((N, 1), 0.05, dtype=np.float32)
    w.append_step(0, base_height=low, **step_kwargs())
    results = w.close()
    assert {r.failure_mode for r in results} == {"collapse"}


# ----------------------------------------------- one file is one episode
def test_a_terminated_env_stops_recording(tmp_path):
    w = VariantTrajectoryWriter(tmp_path, N, control_dt=DT)
    w.append_step(0, **step_kwargs())
    w.append_step(1, terminated=[True, False, False], **step_kwargs())
    w.append_step(2, **step_kwargs())
    results = {r.env_index: r for r in w.close()}
    # env 0 recorded its terminal row and then stopped; the others kept going.
    assert results[0].rows == 2
    assert results[1].rows == 3
    assert results[0].terminated_step == 1
    assert results[1].terminated_step is None


def test_active_envs_tracks_terminations(tmp_path):
    w = VariantTrajectoryWriter(tmp_path, N, control_dt=DT)
    assert w.active_envs == N
    w.append_step(0, terminated=[True, True, False], **step_kwargs())
    assert w.active_envs == 1
    w.close()


def test_termination_without_a_detector_event_is_labelled_as_such(tmp_path):
    w = VariantTrajectoryWriter(tmp_path, N, control_dt=DT)
    w.append_step(0, terminated=[True, False, False], **step_kwargs())
    w.close()
    reader = TrajectoryReader(tmp_path / "variant_000.parquet")
    assert list(reader.column("failure_onset_source")) == ["simulator_termination"]
    assert list(reader.column("sim_termination_index")) == [0]


# ------------------------------------------------------------------- index
def test_index_reports_what_the_replay_produced(tmp_path):
    w = VariantTrajectoryWriter(tmp_path, N, control_dt=DT)
    w.append_step(0, **step_kwargs())
    w.append_step(1, **step_kwargs(quat=tipped(1.2)))
    out = w.write_index(tmp_path / "variants_index.json", extra={"policy": "model_799.pt"})
    payload = json.loads(out.read_text())
    assert payload["variants_written"] == N
    assert payload["variants_with_failure"] == N
    assert payload["policy"] == "model_799.pt"
    assert len(payload["variants"]) == N
    assert payload["control_dt"] == DT


def test_index_counts_only_files_that_exist(tmp_path):
    w = VariantTrajectoryWriter(tmp_path, N, control_dt=DT)
    out = w.write_index(tmp_path / "variants_index.json")
    payload = json.loads(out.read_text())
    assert payload["variants_written"] == 0
    assert payload["variants"] == []


def test_a_row_reports_the_mode_that_fired_on_that_step(tmp_path):
    """Regression: a later slip was labelled with the earlier attitude mode.

    The row's failure_mode used to carry the FIRST mode ever seen for that env,
    so once an attitude event had fired, every later failure row claimed
    "attitude" whatever actually fired. The schema contract in
    trajectory_logger's module docstring is that failure_mode is strictly the
    detector's label for THAT row, and a mode filter downstream trusts it.
    """
    w = VariantTrajectoryWriter(tmp_path, N, control_dt=DT)
    # Step 0: upright, nothing fires.
    w.append_step(0, **step_kwargs())
    # Step 1: tipped past the sim attitude bar -> attitude.
    w.append_step(1, **step_kwargs(quat=tipped(SIM_ANALYSIS_PITCH_RAD + 0.3)))
    # Now upright again, but commanded fast while measured speed stays ~0, held
    # long enough to trip the kinematic slip mode.
    cmd = np.tile(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), (N, 1))
    for i in range(2, 60):
        w.append_step(i, **step_kwargs(cmd=cmd))
    w.close()

    reader = TrajectoryReader(tmp_path / "variant_000.parquet")
    modes = list(reader.column("failure_mode"))
    flags = list(reader.column("failure_flag"))
    fired = [(i, m) for i, (f, m) in enumerate(zip(flags, modes, strict=True)) if f]

    assert fired, "no failure row was recorded at all"
    assert fired[0][1] == "attitude", fired
    later = [m for i, m in fired if i > 1]
    assert "slip" in later, f"expected a later slip row, got {fired}"
    # The bug: every later row said "attitude".
    assert "attitude" not in later, f"a later row inherited the first mode: {fired}"


def test_the_summary_still_reports_the_first_onset(tmp_path):
    """Per-row honesty must not cost the first-onset summary."""
    w = VariantTrajectoryWriter(tmp_path, N, control_dt=DT)
    w.append_step(0, **step_kwargs())
    w.append_step(1, **step_kwargs(quat=tipped(1.2)))
    cmd = np.tile(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), (N, 1))
    for i in range(2, 60):
        w.append_step(i, **step_kwargs(cmd=cmd))
    results = w.close()
    assert {r.failure_mode for r in results} == {"attitude"}
    assert {r.failure_step for r in results} == {1}


def test_close_is_safe_to_call_again_after_an_error(tmp_path):
    # close() marks itself closed before doing the work, so a second call used
    # to hit an unset _results attribute.
    w = VariantTrajectoryWriter(tmp_path, N, control_dt=DT)
    w.append_step(0, **step_kwargs())
    assert w.close() == w.close()


# ------------------------------------------------- curriculum seeding contract
def test_a_fast_failing_variant_seeds_from_its_replay_seed_row(tmp_path):
    """Regression for the 2026-09-21 loop-closure crash.

    A variant's row 0 is the replay seed, which reconstruct already resolved
    with the run's strategy (0.5 s before the SOURCE onset). A variant that
    falls within 0.5 s of that seed has no row 0.5 s before its OWN onset, so
    re-applying the strategy asked for row -1 and killed the fine-tune. The
    variant must seed from the row the replay seeded it from, not back off twice.
    """
    from phoenix.adaptation.reset_bridge import resolve_seed

    w = VariantTrajectoryWriter(tmp_path, N, control_dt=DT)
    for i in range(20):
        quat = tipped(SIM_ANALYSIS_PITCH_RAD + 0.3) if i >= 10 else None
        w.append_step(i, **step_kwargs(quat=quat))
    results = w.close()
    path = results[0].path
    onset = int(TrajectoryReader(path).failure_indices()[0])
    assert onset * DT < 0.5  # the pre-onset window the strategy wants is absent

    record = resolve_seed(path, "failure_onset_minus_seconds", 0, 0.5)
    assert record["resolved_row"] == 0
    assert record["requested_seed_row"] == 0
    assert record["failure_onset_row"] == onset
    assert record["seed_row_source"] == "replay_seed"
    assert record["time_before_onset_seconds"] == pytest.approx(onset * DT)
