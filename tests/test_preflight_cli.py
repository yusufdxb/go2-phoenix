"""The preflight driver fails loud: evidence out of order, stale, or rehearsal never counts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phoenix.sim2real import preflight
from phoenix.sim2real import preflight_eval as pe
from phoenix.sim2real.activation import file_sha256
from phoenix.sim2real.provenance import EXPECTED_BRANCH, CodeIdentity

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "configs/sim2real/deploy_stand_h25.yaml"
LOCK = REPO_ROOT / "configs/sim2real/locks/deploy_stand_h25.lock.yaml"
SHA = "1234567" + "0" * 33


@pytest.fixture
def clean_identity(monkeypatch):
    ident = CodeIdentity(sha=SHA, source="git", branch=EXPECTED_BRANCH, dirty=False)
    monkeypatch.setattr(preflight, "resolve_code_identity", lambda root: ident)
    return ident


def _common(session: Path) -> list[str]:
    return ["--config", str(CONFIG), "--lock", str(LOCK), "--session", str(session)]


def _write_go(session: Path, stage: str, utc: str, **over) -> None:
    record = {
        "schema": pe.STAGE_SCHEMA,
        "stage": stage,
        "verdict": "GO",
        "utc": utc,
        "code_identity": {"sha": SHA},
        "lock_file_sha256": file_sha256(LOCK),
        "checks": [{"name": "x", "ok": True, "gating": True}],
    }
    record.update(over)
    (session / f"stage_{stage}.json").write_text(json.dumps(record))


def test_status_on_an_empty_session_is_not_ready(tmp_path, clean_identity, capsys) -> None:
    rc = preflight.main(["status", *_common(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "READY FOR STAND-ONLY LIVE GO2 TEST (stage F):  NO" in out
    assert "next permitted stage: A" in out


def test_status_ready_only_when_a_through_e_are_go(tmp_path, clean_identity, capsys) -> None:
    for i, stage in enumerate("ABCD"):
        _write_go(tmp_path, stage, f"2026-09-12T10:0{i}:00")
    assert preflight.main(["status", *_common(tmp_path)]) == 1
    assert "ready for live hold stage E:                   YES" in capsys.readouterr().out
    _write_go(tmp_path, "E", "2026-09-12T10:05:00")
    assert preflight.main(["status", *_common(tmp_path)]) == 0
    assert "READY FOR STAND-ONLY LIVE GO2 TEST (stage F):  YES" in capsys.readouterr().out


def test_rehearsal_evidence_is_never_ready(tmp_path, clean_identity, capsys) -> None:
    for i, stage in enumerate("ABCDE"):
        _write_go(tmp_path, stage, f"2026-09-12T10:0{i}:00", rehearsal=(stage == "E"))
    assert preflight.main(["status", *_common(tmp_path)]) == 1
    assert "REHEARSAL" in capsys.readouterr().out


def test_dirty_code_is_never_ready_even_with_evidence(tmp_path, monkeypatch, capsys) -> None:
    dirty = CodeIdentity(sha=SHA, source="git", branch=EXPECTED_BRANCH, dirty=True)
    monkeypatch.setattr(preflight, "resolve_code_identity", lambda root: dirty)
    for i, stage in enumerate("ABCDE"):
        _write_go(tmp_path, stage, f"2026-09-12T10:0{i}:00")
    assert preflight.main(["status", *_common(tmp_path)]) == 1
    assert "CODE IDENTITY PROBLEM" in capsys.readouterr().out


def test_require_refuses_a_stage_whose_predecessors_do_not_count(tmp_path, clean_identity) -> None:
    _write_go(tmp_path, "A", "2026-09-12T10:00:00")
    assert preflight.main(["require", "B", *_common(tmp_path)]) == 0
    assert preflight.main(["require", "E", *_common(tmp_path)]) == 1


def test_evaluate_out_of_order_records_a_no_go(tmp_path, clean_identity) -> None:
    trace = {
        "messages": [
            {"t": i * 0.1, "value": v}
            for i, v in enumerate([False] * 30 + [True] * 30 + [False] * 30)
        ],
        "phases": [
            {"name": "hold", "t_start": 0.0, "t_end": 2.9},
            {"name": "release", "t_start": 3.0, "t_end": 5.9},
            {"name": "rehold", "t_start": 6.0, "t_end": 8.9},
        ],
        "publisher_polls": [{"t": 0.0, "publishers": ["phoenix_wireless_estop"]}],
    }
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace))
    rc = preflight.main(["evaluate", "C", *_common(tmp_path), "--trace", str(trace_path)])
    record = json.loads((tmp_path / "stage_C.json").read_text())
    assert rc == 1 and record["verdict"] == "NO-GO"
    failing = {c["name"] for c in record["checks"] if not c["ok"]}
    assert "every stage before C counts (same commit, same lock)" in failing
    # The deadman trace itself was fine; only the ordering blocked it.
    assert not any("deadman" in name for name in failing)
    # Re-evaluating archives the previous record instead of overwriting it.
    preflight.main(["evaluate", "C", *_common(tmp_path), "--trace", str(trace_path)])
    assert len(list((tmp_path / "history").glob("stage_C_*.json"))) == 1


def test_payload_mode_needs_a_bundle(tmp_path, clean_identity) -> None:
    assert preflight.main(["A", *_common(tmp_path), "--payload"]) == 2


def test_safety_core_lists_name_real_files() -> None:
    for rel in preflight.SAFETY_CORE_FORMATTED:
        assert (REPO_ROOT / rel).is_file(), rel


def test_evaluate_with_missing_evidence_records_a_no_go(tmp_path, clean_identity) -> None:
    rc = preflight.main(
        ["evaluate", "B", *_common(tmp_path), "--run-dir", str(tmp_path / "halted_run")]
    )
    record = json.loads((tmp_path / "stage_B.json").read_text())
    assert rc == 1 and record["verdict"] == "NO-GO"
    assert any("bridge.jsonl" in m for m in record["missing_evidence"])


def test_rehearsal_may_advance_through_rehearsal_records_but_status_never_counts(
    tmp_path, clean_identity, monkeypatch, capsys
) -> None:
    _write_go(tmp_path, "A", "2026-09-12T10:00:00")
    _write_go(tmp_path, "B", "2026-09-12T10:01:00", rehearsal=True)
    assert preflight.main(["require", "C", *_common(tmp_path)]) == 1
    monkeypatch.setenv("PHOENIX_REHEARSAL", "1")
    assert preflight.main(["require", "C", *_common(tmp_path)]) == 0
    monkeypatch.delenv("PHOENIX_REHEARSAL")
    assert preflight.main(["status", *_common(tmp_path)]) == 1
