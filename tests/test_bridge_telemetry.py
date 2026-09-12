"""Bridge telemetry: crash-safe JSONL and an unambiguous hardware slew metric."""

from __future__ import annotations

import json

import numpy as np
import pytest

from phoenix.sim2real.actuator_gate import ActuatorGate, GateParams
from phoenix.sim2real.bridge_telemetry import (
    HARDWARE_SLEW_METRIC,
    TelemetryWriter,
    read_telemetry,
    summarize,
)
from phoenix.sim2real.command_wire import KIND_ABORT, KIND_POLICY, encode
from phoenix.sim2real.go2_model import POLICY_JOINT_ORDER, TRAINING_DEFAULT_JOINT_POS
from phoenix.sim2real.motor_crc import phoenix_to_unitree

ORDER = POLICY_JOINT_ORDER
DEFAULT_P = np.asarray([TRAINING_DEFAULT_JOINT_POS[n] for n in ORDER])
DEFAULT_U = np.asarray(phoenix_to_unitree(DEFAULT_P))
T0 = 1_000_000_000


def _gate() -> ActuatorGate:
    return ActuatorGate(
        GateParams(
            live=False,
            kp=25.0,
            kd=0.5,
            hold_kp=20.0,
            hold_kd=1.0,
            watchdog_s=0.2,
            estop_timeout_s=0.5,
            lowstate_timeout_s=0.2,
            stale_hold_s=0.2,
            first_message_timeout_s=15.0,
        ),
        T0,
    )


def _cmd(target, seq, requested=None, q_policy=None):
    return encode(
        ORDER,
        seq=seq,
        kind=KIND_POLICY,
        target=target,
        requested_target=target if requested is None else requested,
        q_policy=DEFAULT_P if q_policy is None else q_policy,
        obs_source_code=0.0,
        stand_only=1.0,
        velocity_command_fed=[0.0] * 3,
        cmd_vel_received=[0.0] * 3,
        roll_rad=0.01,
        pitch_rad=-0.02,
    )


def test_writer_refuses_to_overwrite(tmp_path) -> None:
    path = tmp_path / "t.jsonl"
    TelemetryWriter(path, {"live": False}).close()
    with pytest.raises(FileExistsError):
        TelemetryWriter(path, {"live": False})


def test_nan_is_written_as_null_and_file_reads_back(tmp_path) -> None:
    path = tmp_path / "t.jsonl"
    w = TelemetryWriter(path, {"live": False, "x": float("nan")})
    w.write_tick({"t_mono_ns": T0, "mode": "hold", "v": [1.0, float("inf")]})
    w.close({"reason": "test"})
    manifest, ticks, end = read_telemetry(path)
    assert manifest["x"] is None
    assert ticks[0]["v"] == [1.0, None]
    assert end["reason"] == "test"
    for line in path.read_text().splitlines():
        json.loads(line)  # strict JSON, no NaN tokens


def test_truncated_last_line_is_tolerated_and_counted(tmp_path) -> None:
    path = tmp_path / "t.jsonl"
    w = TelemetryWriter(path, {"live": False})
    w.write_tick({"t_mono_ns": T0, "mode": "hold"})
    w._fh.write('{"record": "tick", "t_mono_')  # simulate a kill mid-write
    w._fh.flush()
    manifest, ticks, end = read_telemetry(path)
    assert len(ticks) == 1 and end is None and manifest["_unparseable_lines"] == 1


def test_slew_metric_counts_each_command_once_and_matches_hand_count(tmp_path) -> None:
    gate = _gate()
    gate.on_lowstate(T0, DEFAULT_U, np.zeros(12))
    gate.on_estop(T0, False)
    rows = []
    # Command 1: two joints far beyond one slew step -> 2 of 12 clipped.
    big = DEFAULT_P.copy()
    big[0] += 0.5
    big[5] -= 0.5
    label, data = _cmd(big, seq=1)
    gate.on_command(T0 + 1, label, data)
    rows.append(gate.tick(T0 + 10_000_000))
    rows.append(gate.tick(T0 + 20_000_000))  # repeat tick, same command
    # Command 2: nothing clipped.
    label, data = _cmd(DEFAULT_P, seq=2)
    gate.on_command(T0 + 25_000_000, label, data)
    rows.append(gate.tick(T0 + 30_000_000))

    path = tmp_path / "t.jsonl"
    w = TelemetryWriter(path, {"live": False, "gate_params": {"max_delta": 0.175}})
    for r in rows:
        w.write_tick(r)
    w.close()
    manifest, ticks, _ = read_telemetry(path)
    s = summarize(manifest, ticks)
    assert s["metric"] == HARDWARE_SLEW_METRIC
    assert s["policy_ticks"] == 3 and s["policy_new_command_ticks"] == 2
    assert s["bridge_slew_clip_pct"] == pytest.approx(100.0 * 2 / 24)
    assert s["bridge_slew_clip_pct_all_policy_ticks"] == pytest.approx(100.0 * 4 / 36)
    assert s["bridge_slew_clip_pct_per_joint"]["FL_hip_joint"] == pytest.approx(50.0)
    assert s["end_to_end_clip_pct"] == pytest.approx(100.0 * 2 / 24)
    assert s["max_abs_pitch_rad"] == pytest.approx(0.02)


def test_policy_node_clip_is_recomputed_from_the_wire(tmp_path) -> None:
    gate = _gate()
    gate.on_lowstate(T0, DEFAULT_U, np.zeros(12))
    gate.on_estop(T0, False)
    requested = DEFAULT_P.copy()
    requested[3] += 0.3  # the policy node would have clipped this one
    clipped = DEFAULT_P.copy()
    clipped[3] += 0.175
    label, data = _cmd(clipped, seq=1, requested=requested)
    gate.on_command(T0 + 1, label, data)
    rec = gate.tick(T0 + 10_000_000)
    s = summarize({"gate_params": {"max_delta": 0.175}}, [rec])
    assert s["policy_node_slew_clip_pct"] == pytest.approx(100.0 / 12)
    assert s["bridge_slew_clip_pct"] == pytest.approx(0.0)
    # End-to-end sees the policy node's clip even though the bridge added none.
    assert s["end_to_end_clip_pct"] == pytest.approx(100.0 / 12)
    assert s["end_to_end_clip_pct_per_joint"]["RR_hip_joint"] == pytest.approx(100.0)


def test_summary_reports_faults_and_policy_abort_reasons() -> None:
    gate = _gate()
    gate.on_lowstate(T0, DEFAULT_U, np.zeros(12))
    gate.on_estop(T0, False)
    label, data = encode(
        ORDER, seq=9, kind=KIND_ABORT, target=DEFAULT_P, abort_reason="authority_window_complete"
    )
    gate.on_command(T0 + 1, label, data)
    rec = gate.tick(T0 + 10_000_000)
    s = summarize({}, [rec])
    assert s["first_fault"] == "policy_abort:authority_window_complete"
    assert s["policy_abort_reasons"] == ["authority_window_complete"]
    assert s["mode_counts"] == {"hold": 1}
