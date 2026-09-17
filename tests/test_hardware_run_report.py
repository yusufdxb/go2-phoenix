"""The post-run report, driven by fixture evidence in the real schema.

Fixtures are written through the actual :class:`TelemetryWriter`, so a schema
change breaks these tests rather than silently producing a report about fields
that no longer exist.

The behaviour that matters most here is the startup/settled split. A single
clip rate over a whole window described the GO2's FOLDED START POSE, not the
standing robot: folded calf readings sit below the URDF limit, so HOLD clips to
exactly the limit and the rate reads ~100% with the motors off. The report must
keep those windows disjoint and must never let startup contaminate settled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phoenix.sim2real.bridge_telemetry import TELEMETRY_SCHEMA, TelemetryWriter
from phoenix.sim2real.go2_model import UNITREE_MOTOR_ORDER
from phoenix.sim2real.hardware_run_report import (
    Unavailable,
    analyze_session,
    build_parser,
    main,
    render,
)
from phoenix.sim2real.motor_crc import PHOENIX_FOR_MOTOR

PERIOD_NS = 20_000_000  # 50 Hz
WATCHDOG_S = 0.1


def manifest(stage: str = "H", live: bool = True, commit: str = "deadbeef1234") -> dict:
    return {
        "node": "phoenix_lowcmd_bridge",
        "stage": stage,
        "live": live,
        "start_utc": "2026-09-18T03:00:00.000000Z",
        "code_identity": {"sha": commit, "branch": "feat/causal-viability-replication"},
        "deploy_config": {"path": "configs/sim2real/deploy_stand_h25.yaml"},
        "lock": {"name": "h25-stand"},
        "gate_params": {"watchdog_s": WATCHDOG_S, "max_delta": 0.175},
    }


def tick(
    i: int,
    *,
    mode: str = "policy",
    clipped: bool,
    cmd_is_new: bool = True,
    cmd_age_s: float = 0.01,
    estop_value: bool = False,
    deadman_ok: bool = True,
    fault: str | None = None,
    hold_cause: str | None = None,
    t_ns: int | None = None,
) -> dict:
    """One tick. ``clipped`` decides whether the final target differs from the request."""
    requested = [0.1 * (j + 1) for j in range(12)]  # POLICY order
    final = [requested[PHOENIX_FOR_MOTOR[j]] for j in range(12)]  # UNITREE order
    if clipped:
        final[0] += 0.05  # one joint did not get what the policy asked for
    return {
        "t_mono_ns": t_ns if t_ns is not None else i * PERIOD_NS,
        "tick": i,
        "live": True,
        "mode": mode,
        "publish": mode in ("policy", "hold"),
        "hold_cause": hold_cause,
        "fault": fault,
        "faults": [fault] if fault else [],
        "lowstate_age_s": 0.002,
        "lowstate_fresh": True,
        "q_unitree": [0.0] * 12,
        "dq_unitree": [0.0] * 12,
        "estop_value": estop_value,
        "estop_age_s": 0.01,
        "estop_state": "held" if not estop_value else "released",
        "deadman_source_ok": deadman_ok,
        "cmd_seq": i,
        "cmd_kind": 1,
        "cmd_is_new": cmd_is_new,
        "cmd_age_s": cmd_age_s,
        "final_target_unitree": final,
        "kp": 40.0,
        "kd": 1.0,
        "slew_clip": [clipped] + [False] * 11,
        "slew_margin": [0.1] * 12,
        "limit_clip": [False] * 12,
        "limit_margin": [0.5] * 12,
        "policy": {
            "kind": 1,
            "requested_target": requested,
            "roll_rad": 0.01,
            "pitch_rad": 0.02,
            "abort_reason": None,
        },
        "counters": {},
    }


def write_session(
    tmp_path: Path,
    ticks: list[dict],
    *,
    stage: str = "H",
    verdict: str = "GO",
    run_dir: str = "H1",
    with_stage_json: bool = True,
) -> Path:
    session = tmp_path / "session"
    session.mkdir(exist_ok=True)
    run = session / run_dir
    run.mkdir(parents=True, exist_ok=True)
    writer = TelemetryWriter(run / "bridge.jsonl", manifest(stage=stage))
    for t in ticks:
        writer.write_tick(t)
    writer.close()
    if with_stage_json:
        (session / f"stage_{stage}.json").write_text(
            json.dumps(
                {
                    "stage": stage,
                    "verdict": verdict,
                    "code_identity": {"sha": "deadbeef1234"},
                    "checks": [{"name": "a check", "ok": verdict == "GO"}],
                }
            )
        )
    return session


def clean_run(n_startup: int = 100, n_settled: int = 50) -> list[dict]:
    """Startup clips on every tick (folded), settled clips on none (standing)."""
    ticks = [tick(i, clipped=True) for i in range(n_startup)]
    ticks += [tick(n_startup + i, clipped=False) for i in range(n_settled)]
    return ticks


# ------------------------------------------------------------ the core split
def test_startup_does_not_contaminate_the_settled_clip_rate(tmp_path):
    session = write_session(tmp_path, clean_run())
    report = analyze_session(session)
    run = report["runs"][0]
    assert run["clip_startup"]["pct"] == pytest.approx(100.0 / 12, rel=1e-6)
    # The settled second contains only unclipped ticks. A single blended number
    # would have reported the folded start pose instead.
    assert run["clip_settled"]["pct"] == 0.0


def test_the_two_windows_are_disjoint_and_cover_the_policy_ticks(tmp_path):
    session = write_session(tmp_path, clean_run(100, 50))
    run = analyze_session(session)["runs"][0]
    windows = run["clip_window_ticks"]
    assert windows["startup"] + windows["settled"] == run["policy_ticks"]
    # 1.0 s of settle at 50 Hz is exactly the last 50 ticks. The tick sitting
    # on the boundary belongs to startup, or a folded-start sample leaks in.
    assert windows["settled"] == 50


def test_the_settle_window_is_configurable(tmp_path):
    session = write_session(tmp_path, clean_run(100, 50))
    wide = analyze_session(session, settle_window_s=3.0)["runs"][0]
    assert wide["clip_window_ticks"]["settled"] > 50
    # A window wide enough to swallow the folded start now shows contamination,
    # which is exactly why the default is short and the number is labelled.
    assert wide["clip_settled"]["pct"] > 0.0


def test_settled_clip_reports_its_sample_count(tmp_path):
    session = write_session(tmp_path, clean_run())
    run = analyze_session(session)["runs"][0]
    assert run["clip_settled"]["ticks"] > 0


# ------------------------------------------------------------------- rates
def test_rates_and_gaps_come_from_the_timestamps(tmp_path):
    session = write_session(tmp_path, clean_run())
    run = analyze_session(session)["runs"][0]
    assert run["bridge_rate_hz"] == pytest.approx(50.0, rel=1e-6)
    assert run["policy_rate_hz"] == pytest.approx(50.0, rel=1e-2)
    assert run["tick_gap"]["p50_s"] == pytest.approx(0.02, rel=1e-6)
    assert run["duration_s"] == pytest.approx(149 * 0.02, rel=1e-6)


def test_missed_ticks_are_counted_from_a_long_gap(tmp_path):
    ticks = [tick(i, clipped=False) for i in range(20)]
    # Skip three periods between tick 9 and 10.
    for i in range(10, 20):
        ticks[i]["t_mono_ns"] += 3 * PERIOD_NS
    session = write_session(tmp_path, ticks)
    run = analyze_session(session)["runs"][0]
    assert run["missed_ticks"]["count"] == 3
    assert run["missed_ticks"]["long_gaps"] == 1
    assert run["missed_ticks"]["observed_period_s"] == pytest.approx(0.02, rel=1e-6)


def test_a_clean_run_reports_no_missed_ticks(tmp_path):
    session = write_session(tmp_path, clean_run())
    assert analyze_session(session)["runs"][0]["missed_ticks"]["count"] == 0


# ------------------------------------------------------- command freshness
def test_stale_commands_are_counted_against_the_recorded_watchdog(tmp_path):
    ticks = [tick(i, clipped=False, cmd_age_s=0.01) for i in range(10)]
    for i in (4, 5, 6):
        ticks[i]["cmd_age_s"] = 0.5  # older than the 0.1 s watchdog
    session = write_session(tmp_path, ticks)
    run = analyze_session(session)["runs"][0]
    assert run["stale_commands"]["count"] == 3
    assert run["stale_commands"]["watchdog_s"] == WATCHDOG_S
    assert run["max_cmd_age_s"] == pytest.approx(0.5)


def test_stale_commands_are_unavailable_without_a_watchdog_in_the_manifest(tmp_path):
    session = tmp_path / "s"
    (session / "H1").mkdir(parents=True)
    m = manifest()
    m["gate_params"] = {}
    w = TelemetryWriter(session / "H1" / "bridge.jsonl", m)
    for t in clean_run(4, 2):
        w.write_tick(t)
    w.close()
    run = analyze_session(session)["runs"][0]
    assert isinstance(run["stale_commands"], Unavailable)
    assert "watchdog_s" in run["stale_commands"].reason


# ------------------------------------------------------------ interventions
def test_a_latched_fault_counts_as_one_trip_not_hundreds(tmp_path):
    ticks = [tick(i, clipped=False) for i in range(5)]
    ticks += [tick(5 + i, clipped=False, mode="hold", fault="estop_stale") for i in range(50)]
    session = write_session(tmp_path, ticks)
    run = analyze_session(session)["runs"][0]
    assert run["watchdog_trips"]["count"] == 1
    assert run["watchdog_trips"]["onsets"][0]["fault"] == "estop_stale"


def test_hold_causes_are_reported_as_intervention_reasons(tmp_path):
    ticks = [tick(i, clipped=False) for i in range(3)]
    ticks += [tick(3 + i, clipped=False, mode="hold", hold_cause="lowstate_stale") for i in range(3)]
    session = write_session(tmp_path, ticks)
    run = analyze_session(session)["runs"][0]
    assert run["intervention_reasons"]["hold_causes"]["lowstate_stale"] == 3


def test_deadman_transitions_are_counted(tmp_path):
    ticks = [tick(i, clipped=False) for i in range(5)]
    ticks += [tick(5 + i, clipped=False, estop_value=True, mode="hold") for i in range(5)]
    session = write_session(tmp_path, ticks)
    run = analyze_session(session)["runs"][0]
    assert run["deadman_transitions"]["count"] == 1


def test_release_to_safe_output_latency_is_measured(tmp_path):
    # Deadman goes unsafe while the policy still has authority, then the bridge
    # leaves policy mode two ticks later.
    ticks = [tick(i, clipped=False) for i in range(5)]
    ticks.append(tick(5, clipped=False, estop_value=True, mode="policy"))
    ticks.append(tick(6, clipped=False, estop_value=True, mode="policy"))
    ticks.append(tick(7, clipped=False, estop_value=True, mode="hold"))
    session = write_session(tmp_path, ticks)
    run = analyze_session(session)["runs"][0]
    rel = run["deadman_release_to_safe_output"]
    assert rel["seconds"] == pytest.approx(2 * PERIOD_NS / 1e9)
    assert rel["to_mode"] == "hold"


def test_release_latency_is_unavailable_when_it_never_happened(tmp_path):
    session = write_session(tmp_path, clean_run())
    rel = analyze_session(session)["runs"][0]["deadman_release_to_safe_output"]
    assert isinstance(rel, Unavailable)
    assert "deadman" in rel.reason


# ------------------------------------------------------- command path delta
def test_request_to_final_delta_uses_the_joint_order_permutation(tmp_path):
    session = write_session(tmp_path, clean_run(3, 1))
    run = analyze_session(session)["runs"][0]
    # Only the injected 0.05 offset should show; a joint-order mistake would
    # produce a much larger spurious delta.
    assert run["max_request_to_final_delta_rad"] == pytest.approx(0.05, abs=1e-9)
    assert len(UNITREE_MOTOR_ORDER) == 12


# -------------------------------------------------------- unavailable paths
def test_a_session_with_no_telemetry_is_insufficient_data(tmp_path):
    session = tmp_path / "empty"
    session.mkdir()
    report = analyze_session(session)
    assert report["runs"] == []
    assert "INSUFFICIENT DATA" in render(report)


def test_a_single_tick_cannot_support_rates(tmp_path):
    session = write_session(tmp_path, [tick(0, clipped=False)])
    run = analyze_session(session)["runs"][0]
    assert isinstance(run["duration_s"], Unavailable)
    assert isinstance(run["tick_gap"], Unavailable)
    assert "t_mono_ns" in run["duration_s"].reason


def test_unavailable_metrics_are_explained_in_the_rendered_report(tmp_path):
    session = write_session(tmp_path, [tick(0, clipped=False)])
    text = render(analyze_session(session))
    assert "UNAVAILABLE" in text
    assert "what would be needed" in text


def test_a_no_go_stage_makes_the_result_fail(tmp_path):
    session = write_session(tmp_path, clean_run(), verdict="NO-GO")
    assert "RESULT: FAIL" in render(analyze_session(session))


def test_a_go_stage_with_a_full_run_passes(tmp_path):
    session = write_session(tmp_path, clean_run())
    assert "RESULT: PASS" in render(analyze_session(session))


def test_a_run_without_any_stage_verdict_is_not_a_pass(tmp_path):
    session = write_session(tmp_path, clean_run(), with_stage_json=False)
    assert "RESULT: INSUFFICIENT DATA" in render(analyze_session(session))


def test_a_rehearsal_stage_is_marked_in_the_output(tmp_path):
    session = write_session(tmp_path, clean_run())
    data = json.loads((session / "stage_H.json").read_text())
    data["rehearsal"] = True
    (session / "stage_H.json").write_text(json.dumps(data))
    assert "REHEARSAL, never counts" in render(analyze_session(session))


# ------------------------------------------------------------------- CLI
def test_cli_writes_json_alongside_the_text(tmp_path, capsys):
    session = write_session(tmp_path, clean_run())
    out = tmp_path / "report.json"
    rc = main([str(session), "--json-out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["runs"][0]["clip_settled"]["pct"] == 0.0
    assert "PHOENIX HARDWARE RUN" in capsys.readouterr().out


def test_cli_json_encodes_unavailable_with_its_reason(tmp_path):
    session = write_session(tmp_path, [tick(0, clipped=False)])
    out = tmp_path / "report.json"
    main([str(session), "--json-out", str(out)])
    payload = json.loads(out.read_text())
    duration = payload["runs"][0]["duration_s"]
    assert duration["unavailable"] is True
    assert duration["reason"]


def test_cli_returns_nonzero_when_not_a_pass(tmp_path):
    session = write_session(tmp_path, clean_run(), verdict="NO-GO")
    assert main([str(session), "--json-out", str(tmp_path / "r.json")]) != 0


def test_parser_requires_a_session(tmp_path):
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_a_missing_session_directory_is_refused(tmp_path):
    with pytest.raises(NotADirectoryError):
        analyze_session(tmp_path / "nope")


def test_the_fixture_matches_the_shipped_telemetry_schema(tmp_path):
    session = write_session(tmp_path, clean_run(2, 1))
    first = (session / "H1" / "bridge.jsonl").read_text().splitlines()[0]
    assert json.loads(first)["schema"] == TELEMETRY_SCHEMA
