"""A hardware capture must never be usable as a simulator seed.

``/utlidar/robot_odom`` has its origin at the BOOT pose, so ``base_pos[2]`` is
displacement from wherever the robot booted, not height above the floor. A
robot that boots standing and is captured while standing reads z ~ 0.

Before 2026-09-17 nothing declared that frame, so
:meth:`TrajectoryReader.resolve_position_frame` fell back to the SIMULATOR
convention ``env_local``, and ``reset_bridge`` wrote that z straight into
Isaac: the seeded trunk spawned in the floor and the resulting rollout looked
like a plausible failure. These tests pin the three layers that now stop it:

1. the writer DECLARES the frame in the Parquet footer,
2. the reader INFERS it from ``capture_source`` for captures written before
   the writer did (the parquets already on disk),
3. the state adapter REFUSES to map it into simulator coordinates at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phoenix.real_world.trajectory_logger import (
    CAPTURE_SOURCE_HARDWARE,
    CAPTURE_SOURCE_SIM,
    PARQUET_POSITION_FRAME_KEY,
    POSITION_FRAME_ENV_LOCAL,
    POSITION_FRAME_ODOM_BOOT_RELATIVE,
    TrajectoryLogger,
    TrajectoryStep,
)
from phoenix.replay import state_adapter
from phoenix.replay.trajectory_reader import (
    PARQUET_POSITION_FRAME_KEY as READER_KEY,
)
from phoenix.replay.trajectory_reader import (
    TrajectoryReader,
)


def step(i: int, capture_source: str) -> TrajectoryStep:
    return TrajectoryStep(
        step=i,
        timestamp_s=0.02 * i,
        # z ~ 0: exactly what a standing robot reports in boot-relative odom.
        base_pos=np.asarray([0.0, 0.0, 0.002], dtype=np.float32),
        base_quat=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        base_lin_vel_body=np.zeros(3, dtype=np.float32),
        base_ang_vel_body=np.zeros(3, dtype=np.float32),
        joint_pos=np.zeros(12, dtype=np.float32),
        joint_vel=np.zeros(12, dtype=np.float32),
        command_vel=np.zeros(3, dtype=np.float32),
        action=np.zeros(12, dtype=np.float32),
        contact_forces=np.zeros(4, dtype=np.float32),
        capture_source=capture_source,
    )


def write(path: Path, capture_source: str, *, position_frame: str | None = None) -> Path:
    with TrajectoryLogger(path, position_frame=position_frame) as log:
        for i in range(3):
            log.append(step(i, capture_source))
    return path


# ------------------------------------------------- the constants agree
def test_the_two_modules_use_the_same_metadata_key():
    # The writer must not import the replay stack (it runs on the payload), so
    # the key is duplicated. If these drift, every declaration goes unread.
    assert PARQUET_POSITION_FRAME_KEY == READER_KEY


def test_the_two_modules_use_the_same_frame_names():
    assert POSITION_FRAME_ODOM_BOOT_RELATIVE == state_adapter.POSITION_FRAME_ODOM_BOOT_RELATIVE
    assert POSITION_FRAME_ENV_LOCAL == state_adapter.DEFAULT_POSITION_FRAME


def test_the_boot_relative_frame_is_known_but_not_restorable():
    assert POSITION_FRAME_ODOM_BOOT_RELATIVE in state_adapter.POSITION_FRAMES
    assert POSITION_FRAME_ODOM_BOOT_RELATIVE not in state_adapter.RESTORABLE_POSITION_FRAMES


# --------------------------------------------------------- layer 1: writer
def test_the_logger_declares_the_frame_it_is_given(tmp_path):
    p = write(tmp_path / "hw.parquet", CAPTURE_SOURCE_HARDWARE,
              position_frame=POSITION_FRAME_ODOM_BOOT_RELATIVE)
    reader = TrajectoryReader(p)
    assert reader.declared_position_frame == POSITION_FRAME_ODOM_BOOT_RELATIVE
    assert reader.resolve_position_frame() == (
        POSITION_FRAME_ODOM_BOOT_RELATIVE,
        "declared_by_source",
    )


def test_an_undeclared_capture_writes_no_key(tmp_path):
    p = write(tmp_path / "plain.parquet", CAPTURE_SOURCE_SIM)
    assert TrajectoryReader(p).declared_position_frame is None


def test_the_logger_refuses_a_meaningless_frame(tmp_path):
    for bad in ("", "   "):
        with pytest.raises(ValueError, match="position_frame must be a non-empty string"):
            TrajectoryLogger(tmp_path / "x.parquet", position_frame=bad)


def test_an_unknown_declared_frame_is_refused_on_read(tmp_path):
    p = write(tmp_path / "weird.parquet", CAPTURE_SOURCE_SIM, position_frame="lidar_frame")
    with pytest.raises(ValueError, match="Unknown parquet position frame"):
        TrajectoryReader(p)


# -------------------------------------------------------- layer 2: reader
def test_an_undeclared_hardware_capture_is_inferred_not_defaulted(tmp_path):
    # This is the back-compat path: parquets captured on the robot BEFORE the
    # writer learned to declare its frame. They are already on disk.
    p = write(tmp_path / "legacy_hw.parquet", CAPTURE_SOURCE_HARDWARE)
    reader = TrajectoryReader(p)
    assert reader.declared_position_frame is None
    assert reader.resolve_position_frame() == (
        POSITION_FRAME_ODOM_BOOT_RELATIVE,
        "inferred_from_capture_source",
    )


def test_a_sim_capture_still_resolves_to_the_simulator_default(tmp_path):
    p = write(tmp_path / "sim.parquet", CAPTURE_SOURCE_SIM)
    assert TrajectoryReader(p).resolve_position_frame() == (
        POSITION_FRAME_ENV_LOCAL,
        "phoenix_capture_default",
    )


def test_a_caller_cannot_override_an_inferred_hardware_frame(tmp_path):
    # Asking for env_local on a hardware capture IS the bug this prevents.
    p = write(tmp_path / "legacy_hw.parquet", CAPTURE_SOURCE_HARDWARE)
    with pytest.raises(ValueError, match="A hardware capture is not a simulator seed"):
        TrajectoryReader(p).resolve_position_frame(POSITION_FRAME_ENV_LOCAL)


def test_requesting_the_frame_it_already_is_is_accepted(tmp_path):
    p = write(tmp_path / "legacy_hw.parquet", CAPTURE_SOURCE_HARDWARE)
    frame, source = TrajectoryReader(p).resolve_position_frame(
        POSITION_FRAME_ODOM_BOOT_RELATIVE
    )
    assert (frame, source) == (POSITION_FRAME_ODOM_BOOT_RELATIVE, "inferred_from_capture_source")


def test_a_declaration_still_outranks_the_inference(tmp_path):
    # A sim writer that labelled its rows "hardware" by mistake must not have
    # its explicit declaration silently overridden by the row inference.
    p = write(tmp_path / "declared.parquet", CAPTURE_SOURCE_HARDWARE,
              position_frame=POSITION_FRAME_ENV_LOCAL)
    assert TrajectoryReader(p).resolve_position_frame() == (
        POSITION_FRAME_ENV_LOCAL,
        "declared_by_source",
    )


def test_a_mixed_capture_does_not_infer(tmp_path, caplog):
    p = tmp_path / "mixed.parquet"
    with TrajectoryLogger(p) as log:
        log.append(step(0, CAPTURE_SOURCE_HARDWARE))
        log.append(step(1, CAPTURE_SOURCE_SIM))
    reader = TrajectoryReader(p)
    assert reader.inferred_position_frame is None
    assert reader.resolve_position_frame()[1] == "phoenix_capture_default"


# -------------------------------------------------- layer 3: state adapter
def test_to_world_position_refuses_the_boot_relative_frame():
    with pytest.raises(ValueError, match="cannot be mapped into simulator coordinates"):
        state_adapter.to_world_position(
            [0.0, 0.0, 0.002], [0.0, 0.0, 0.0], POSITION_FRAME_ODOM_BOOT_RELATIVE
        )


def test_to_stored_position_refuses_it_too():
    with pytest.raises(ValueError, match="cannot be mapped into simulator coordinates"):
        state_adapter.to_stored_position(
            [0.0, 0.0, 0.002], [0.0, 0.0, 0.0], POSITION_FRAME_ODOM_BOOT_RELATIVE
        )


def test_the_refusal_names_the_reason_not_just_the_frame():
    with pytest.raises(ValueError) as exc:
        state_adapter.to_world_position([0.0] * 3, [0.0] * 3, POSITION_FRAME_ODOM_BOOT_RELATIVE)
    message = str(exc.value)
    assert "boot-relative" in message
    assert "not height" in message


def test_an_unknown_frame_is_still_reported_as_unknown():
    with pytest.raises(ValueError, match="Unknown position_frame"):
        state_adapter.to_world_position([0.0] * 3, [0.0] * 3, "somewhere")


def test_the_restorable_frames_still_round_trip():
    origin = np.asarray([3.0, -2.0, 0.5])
    stored = np.asarray([0.1, 0.2, 0.35])
    for frame in state_adapter.RESTORABLE_POSITION_FRAMES:
        world = state_adapter.to_world_position(stored, origin, frame)
        back = state_adapter.to_stored_position(world, origin, frame)
        assert np.allclose(back, stored)


# ------------------------------------------------------------ end to end
def test_a_hardware_capture_cannot_be_restored_end_to_end(tmp_path):
    """The whole point: read a real capture, try to seed sim, get refused."""
    from phoenix.replay.trajectory_reader import load_initial_state

    p = write(tmp_path / "hw.parquet", CAPTURE_SOURCE_HARDWARE)
    initial = load_initial_state(p, row=0)
    assert initial.position_frame == POSITION_FRAME_ODOM_BOOT_RELATIVE
    # z ~ 0 is what the standing robot reported; had this resolved to
    # env_local it would have been written into Isaac as ground clearance.
    assert initial.base_pos[2] == pytest.approx(0.002, abs=1e-6)
    with pytest.raises(ValueError, match="cannot be mapped into simulator coordinates"):
        state_adapter.to_world_position(
            initial.base_pos, np.zeros(3), initial.position_frame
        )
