"""Incumbent-only FREE_FLOOR gate (scripts/free_floor_gate.py): identity, evidence, state machine.

Run logs here are built from VERBATIM real bridge tick records
(tests/fixtures/gantry_ab/real_policy_exits.json: a real deadman release and a real planned
end on the GO2), lengthened the same way tests/test_gantry_ab.py does. Stage records are
written by the production writer under the REAL stand lock; they are test fixtures, not
hardware evidence. Tests that need the archived model_2750 files skip without them (CI).
No robot, ROS or network. The process-level choreography is exercised by
scripts/free_floor_rehearsal.py (tests/test_free_floor_rehearsal.py runs it where ROS exists).
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "scripts/harness_preflight.sh"
WALK_CFG = REPO / "configs/sim2real/proposed/deploy_walk_iter2750.yaml"
WALK_LOCK = REPO / "configs/sim2real/locks/deploy_walk_iter2750.lock.yaml"
STAND_CFG = REPO / "configs/sim2real/deploy_walk_iter2750_stand.yaml"
STAND_LOCK = REPO / "configs/sim2real/locks/deploy_walk_iter2750_stand.lock.yaml"
S50_CFG = REPO / "configs/sim2real/proposed/deploy_walk_fric_s50_3749.yaml"
S50_LOCK = REPO / "configs/sim2real/locks/deploy_walk_fric_s50_3749.lock.yaml"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
ga = _load("gantry_ab_analyze", "scripts/gantry_ab_analyze.py")
ff = _load("free_floor_gate", "scripts/free_floor_gate.py")

from phoenix.sim2real import free_floor_state as st  # noqa: E402
from phoenix.sim2real import preflight  # noqa: E402
from phoenix.sim2real import preflight_eval as pe  # noqa: E402
from phoenix.sim2real.deploy_contract import (  # noqa: E402
    artifact_paths,
    live_environment_problems,
    load_lock,
)
from phoenix.sim2real.provenance import CodeIdentity, resolve_code_identity  # noqa: E402

FIX = json.loads((REPO / "tests/fixtures/gantry_ab/real_policy_exits.json").read_text())
REL = FIX["release"]["ticks"]
END = FIX["planned_end"]["ticks"]
SHA = "c" * 40
EXEC = "git-tree:" + "e" * 40
RT = {"code_sha": SHA, "execution_set_sha256": EXEC, "boot_id": "boot-1", "deadman": "wireless"}
CLEAN_ENV = {"PATH": "/usr/bin"}


def _ident(**over):
    base = dict(sha=SHA, source="git", branch="feat/phoenix-walk-v1-hw", dirty=False)
    base.update(over)
    return CodeIdentity(**base, execution_set_sha256=EXEC)


# ------------------------------------------------------------------ run-log builders
def _ticks(real, episode_s):
    real = copy.deepcopy(real)
    first, dt = real[0], real[1]["t_mono_ns"] - real[0]["t_mono_ns"]
    k_last = max(i for i, t in enumerate(real) if t["mode"] == "policy")
    span = real[k_last]["t_mono_ns"] - first["t_mono_ns"]
    n = max(0, round((episode_s * 1e9 - span) / dt))
    pre = []
    for j in range(n, 0, -1):
        t = copy.deepcopy(first)
        t["t_mono_ns"] = first["t_mono_ns"] - j * dt
        t["tick"] = first["tick"] - j
        pre.append(t)
    return pre + real


def _manifest(**over):
    m = {
        "record": "manifest",
        "live": True,
        "code_identity": {"sha": SHA, "source": "payload_sync"},
        "code_identity_problems": [],
        "startup_problems": [],
        "deploy_config": {"semantic_sha256": ff.INCUMBENT_WALK_CONFIG_SEMANTIC_SHA256},
        "artifacts": {r: {"sha256": ff.INCUMBENT[r]} for r in ff.MODEL_ROLES},
        "gate_params": {
            "walking_allowed": True,
            "deadman_required": True,
            "require_imu": True,
            "kp": 25.0,
            "kd": 0.5,
        },
    }
    m.update(over)
    return m


def _trace(ticks, e2e_s=0.55):
    """A remote_trace.jsonl consistent with the ticks of a release: L1 held until e2e_s before the
    bridge reads the release, then the estop flips just before the bridge reads it."""
    k = max(i for i, t in enumerate(ticks) if t["mode"] == "policy")
    t_obs = ticks[k + 1]["t_mono_ns"]
    t_last = t_obs - int(e2e_s * 1e9)
    rows = [{"t_mono_ns": t_last - 2_000_000_000, "topic": "start", "value": None}]
    t = t_last - 1_000_000_000
    while t <= t_last:
        rows.append({"t_mono_ns": t, "topic": "wireless", "value": True})
        t += 20_000_000
    rows.append({"t_mono_ns": t_obs - 300_000, "topic": "estop", "value": True})
    return rows


def _snap():
    return {"artifacts": dict(ff.INCUMBENT)}


def _run(
    session,
    name,
    ticks,
    *,
    kind=None,
    manifest=None,
    answer="y",
    parquet=True,
    walk=None,
    trace=True,
    attempt=True,
    snaps=True,
    alive=None,
    logs=True,
    reason="sigint",
    runtime=RT,
    started=True,
):
    kind = kind or ("drill" if "drill" in name else "walk")
    d = session / name
    d.mkdir(parents=True)
    faults = sorted({f for t in ticks for f in (t.get("faults") or [])} | {"bridge_shutdown"})
    rows = [manifest or _manifest(), *ticks, {"record": "end", "reason": reason, "faults": faults}]
    (d / "bridge.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if parquet:
        (d / "policy.parquet").write_bytes(b"PAR1 not a real parquet, size only")
    if answer is not None:
        (d / "operator_answer.txt").write_text(answer + "\n")
    if walk is not None:
        (d / "walk_cmd.jsonl").write_text(json.dumps(walk) + "\n")
    if attempt:
        (d / "attempt.json").write_text(
            json.dumps(
                {
                    "schema": ff.ATTEMPT_SCHEMA,
                    "kind": kind,
                    "run_dir": name,
                    "rehearsal": False,
                    "runtime_identity": runtime,
                }
            )
        )
    if snaps:
        for n in (ff.ARTIFACTS_START, ff.ARTIFACTS_END):
            (d / n).write_text(json.dumps(_snap()))
    (d / ff.ALIVE_AT_END).write_text(
        json.dumps(alive or {"deadman": True, "lowstate_bridge": True, "lowcmd_bridge": True})
    )
    if logs:
        for n in ("deadman", "lowstate_bridge", "lowcmd_bridge", "policy_node"):
            (d / f"{n}.log").write_text("log\n")
    if kind == "drill" and trace:
        (d / ff.REMOTE_TRACE).write_text(
            "\n".join(json.dumps(r) for r in (trace if isinstance(trace, list) else _trace(ticks)))
            + "\n"
        )
    if kind == "walk" and started and not (session / ff.STARTED).exists():
        ff._write_record(
            session,
            ff.STARTED,
            {
                "schema": st.STARTED_SCHEMA,
                "run_dir": name,
                "code_sha": SHA,
                "runtime_identity": runtime,
            },
        )
    return d


GOOD_WALK = {
    "record": "summary",
    "preset": "vx03_2s",
    "outcome": "complete",
    "n_published": 140,
    "max_abs": [0.3, 0.0, 0.0],
}
WALK_S = ff.profile_duration(ff.PRESETS["vx03_2s"]) + ga.WALK_AUTHORITY_MARGIN_S


@pytest.fixture
def session(tmp_path):
    s = tmp_path / "20261008_session"
    s.mkdir()
    return s


def _drill(session, name="FF_drill_001", **kw):
    d = _run(session, name, _ticks(REL, 2.0), **kw)
    return ff.drill_eval(d, session, SHA, RT)


def _walk(session, name="FF_walk_001", ticks=None, **kw):
    kw.setdefault("walk", GOOD_WALK)
    d = _run(session, name, ticks or _ticks(END, WALK_S), **kw)
    return ff.walk_eval(d, session, SHA, RT)


# ------------------------------------------------------------------ identity (pure)
def test_incumbent_hashes_match_the_gantry_analyzer_and_the_committed_lock():
    for role in ("checkpoint", "policy.onnx", "policy.onnx.data"):
        assert ff.INCUMBENT[role] == ga.ARMS["A"][role]
        assert ff.SEED50_HASHES[role] == ga.ARMS["B"][role]
    lock = load_lock(WALK_LOCK)
    for role, sha in ff.INCUMBENT.items():
        assert lock["artifacts"][role]["sha256"] == sha
    assert lock["deploy_config"]["semantic_sha256"] == ff.INCUMBENT_WALK_CONFIG_SEMANTIC_SHA256
    assert ff.INCUMBENT["checkpoint"].startswith("e03013d1")
    assert ff.INCUMBENT["policy.onnx"].startswith("6d36796f")


def test_correct_model_2750_hashes_are_accepted():
    assert ff.hash_identity_problems(ff.INCUMBENT, "t") == []


def test_seed_50_is_rejected_by_name():
    for role in ("checkpoint", "policy.onnx", "policy.onnx.data"):
        h = dict(ff.INCUMBENT) | {role: ff.SEED50_HASHES[role]}
        why = ff.hash_identity_problems(h, "t")
        assert why and "SEED 50" in why[0] and "gantry-only" in why[0]
    assert len(ff.hash_identity_problems(ff.SEED50_HASHES, "t")) == 3


def test_every_mixture_that_is_not_exactly_model_2750_is_refused():
    """Task 9: checkpoint/onnx/data each from the incumbent or seed 50: only all-incumbent passes."""
    import itertools

    roles = ("checkpoint", "policy.onnx", "policy.onnx.data")
    for pick in itertools.product((False, True), repeat=3):
        h = {
            r: (ff.SEED50_HASHES if s else ff.INCUMBENT)[r]
            for r, s in zip(roles, pick, strict=True)
        }
        why = ff.hash_identity_problems(h, "mix")
        assert (why == []) == (not any(pick)), (pick, why)
        assert len(why) == sum(pick)


def test_a_copied_file_name_cannot_launder_seed_50_bytes(tmp_path):
    """Identity is the hash of the bytes, never the file name: seed-50 bytes named model_2750.pt."""
    from phoenix.sim2real.deploy_contract import file_sha256

    p = tmp_path / "model_2750.pt"
    p.write_bytes(b"not the incumbent")
    h = dict(ff.INCUMBENT) | {"checkpoint": file_sha256(p)}
    assert "unknown artifact" in ff.hash_identity_problems(h, "t")[0]


def test_unknown_or_missing_artifact_is_rejected():
    h = dict(ff.INCUMBENT) | {"checkpoint": "0" * 64}
    assert "unknown artifact" in ff.hash_identity_problems(h, "t")[0]
    h = dict(ff.INCUMBENT) | {"policy.onnx": None}
    assert "missing" in ff.hash_identity_problems(h, "t")[0]
    assert len(ff.hash_identity_problems({}, "t")) == 3


def test_only_the_commissioning_command_exists():
    assert ff.FREE_FLOOR_PRESETS == ("vx03_2s",)
    p = ff.PRESETS["vx03_2s"]
    assert (p.vx, p.vy, p.wz, p.motion_s) == (0.3, 0.0, 0.0, 2.0)  # no yaw, 2 s
    assert ff.FREE_FLOOR_COMMAND == (0.3, 0.0, 0.0)
    for bad in ("yaw05_3s", "vx03_4s", "vx05_3s", "vxm03_2s", "", None, "nonsense"):
        why = ff.check(
            "walk",
            {},
            WALK_CFG,
            WALK_LOCK,
            Path("/nonexistent"),
            sessions=[],
            sha_now=SHA,
            preset=bad,
        )
        assert len(why) == 1 and "allows only" in why[0]


def test_the_cli_has_no_way_to_name_a_velocity_or_duration():
    """No flag carries a twist or a duration; the preset is a name from the single-entry allowlist."""
    src = (REPO / "scripts/free_floor_gate.py").read_text()
    for forbidden in ("--vx", "--vy", "--wz", "--duration", "--twist", "--velocity"):
        assert forbidden not in src
    harness = HARNESS.read_text()
    body = harness[harness.index("\nstage_free_floor() {") :]
    body = body[: body.index("\n}\n")]
    assert body.count("--preset") == 4  # check walk (gate, live), begin, the walk_command launch
    assert (
        "PHOENIX_WALK_AUTHORITY_MARGIN_S" not in body
        and "PHOENIX_STANDUP_S" not in body.replace("standup_s=2.0", "")
    )


def test_bundle_with_wrong_files_is_rejected(tmp_path):
    b = tmp_path / "bundle"
    b.mkdir()
    for name, data in (("policy.onnx", b"a"), ("policy.onnx.data", b"b"), ("m2750.pt", b"c")):
        (b / name).write_bytes(data)
    (b / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256((b / n).read_bytes()).hexdigest()}  {n}\n"
            for n in ("policy.onnx", "policy.onnx.data", "m2750.pt")
        )
    )
    why = ff.bundle_problems(b)  # self-consistent sums but not model_2750's bytes
    assert any("unknown artifact" in w for w in why)
    (b / "policy.onnx").write_bytes(b"tampered")
    assert any("differs from SHA256SUMS" in w for w in ff.bundle_problems(b))
    assert ff.bundle_problems(tmp_path / "nothing")[0].endswith("SHA256SUMS missing")


# ------------------------------------------------------------------ environment and session
def test_environment_that_could_change_a_run_is_refused():
    live = {"ROS_LOCALHOST_ONLY": "0"}
    assert ff.env_problems(live) == []
    for var, val, needle in (
        ("PHOENIX_WALK_AUTHORITY_MARGIN_S", "9.0", "PHOENIX_WALK_AUTHORITY_MARGIN_S"),
        ("PHOENIX_STANDUP_S", "0.5", "PHOENIX_STANDUP_S"),
        ("PHOENIX_MIN_BATTERY_SOC", "1", "battery"),
        ("PHOENIX_MIN_BATTERY_SOC", "garbage", "battery"),
        ("PHOENIX_DEADMAN", "keyboard", "PHOENIX_DEADMAN"),
        ("PHOENIX_EXPECT_BRANCH", "main", "PHOENIX_EXPECT_BRANCH"),
        ("PHOENIX_REHEARSAL", "1", "ROS_LOCALHOST_ONLY"),  # a rehearsal without localhost-only
        ("PHOENIX_HW_SESSIONS", "/tmp/x", "PHOENIX_HW_SESSIONS"),
        ("PHOENIX_HW_SESSIONS_TEST_ONLY", "1", "PHOENIX_HW_SESSIONS_TEST_ONLY"),
        ("PHOENIX_CONFIRM", "LIVE", "PHOENIX_CONFIRM"),
        ("PHOENIX_REHEARSAL_WALKED", "y", "PHOENIX_REHEARSAL_WALKED"),
        ("ROS_LOCALHOST_ONLY", "1", "localhost"),
    ):
        why = ff.env_problems(live | {var: val})
        assert why and any(needle in w for w in why), (var, why)
    assert (
        ff.env_problems(live | {"PHOENIX_STANDUP_S": "2.0"}) == []
    )  # the pinned value is harmless
    assert (
        ff.env_problems(live | {"PHOENIX_MIN_BATTERY_SOC": "30"}) == []
    )  # raising the floor is fine
    reh = {"PHOENIX_REHEARSAL": "1", "ROS_LOCALHOST_ONLY": "1"}
    assert ff.env_problems(reh) == []


def test_a_live_process_refuses_the_test_only_evidence_override(monkeypatch):
    monkeypatch.delenv("PHOENIX_REHEARSAL", raising=False)
    assert live_environment_problems() == []
    monkeypatch.setenv("PHOENIX_HW_SESSIONS", "/tmp/fake")
    monkeypatch.setenv("PHOENIX_HW_SESSIONS_TEST_ONLY", "1")
    assert len(live_environment_problems()) == 2
    monkeypatch.setenv("PHOENIX_REHEARSAL", "1")
    assert live_environment_problems() == []


def test_session_must_be_inside_the_session_root(tmp_path):
    root = tmp_path / "hw_sessions"
    (root / "s").mkdir(parents=True)
    assert ff.session_problems(root / "s", root) == []
    assert "not a direct child" in ff.session_problems(tmp_path / "elsewhere", root)[0]
    (tmp_path / "link").symlink_to(root / "s")
    assert "symlink" in ff.session_problems(tmp_path / "link", root)[0]


# ------------------------------------------------------------------ review / one run at a time
def test_started_run_blocks_everything_until_a_human_reviews(session):
    assert ff.unreviewed_runs([session]) == []
    ff._write_record(session, ff.STARTED, {"schema": st.STARTED_SCHEMA, "run_dir": "x"})
    assert "no review record" in ff.unreviewed_runs([session])[0]
    with pytest.raises(ValueError, match="no GO result"):
        ff.review(session, "accept", "yusuf", "looks fine")  # no result: cannot accept
    ff.review(session, "reject", "yusuf", "analysis failed, investigate")
    assert ff.unreviewed_runs([session]) == []
    with pytest.raises(FileExistsError):
        ff.review(session, "reject", "yusuf", "again")


def test_accept_needs_a_go_result_for_the_started_run_and_a_named_reviewer(session):
    ff._write_record(session, ff.STARTED, {"schema": st.STARTED_SCHEMA, "run_dir": "r"})
    ff._write_record(session, ff.RESULT, {"verdict": "NO-GO", "run_dir": "r"})
    with pytest.raises(ValueError, match="no GO result"):
        ff.review(session, "accept", "yusuf", "x")
    s2 = session.parent / "s2"
    s2.mkdir()
    ff._write_record(s2, ff.STARTED, {"schema": st.STARTED_SCHEMA, "run_dir": "r"})
    ff._write_record(s2, ff.RESULT, {"verdict": "GO", "run_dir": "other", "rehearsal": False})
    with pytest.raises(ValueError, match="different run"):
        ff.review(s2, "accept", "yusuf", "x")
    s3 = session.parent / "s3"
    s3.mkdir()
    ff._write_record(s3, ff.STARTED, {"schema": st.STARTED_SCHEMA, "run_dir": "r"})
    (s3 / "r").mkdir()
    (s3 / "r" / "bridge.jsonl").write_text("{}\n")
    ev = [{"path": "r/bridge.jsonl", "sha256": hashlib.sha256(b"{}\n").hexdigest()}]
    ff._write_record(
        s3, ff.RESULT, {"verdict": "GO", "run_dir": "r", "rehearsal": False, "evidence_files": ev}
    )
    with pytest.raises(ValueError, match="name and a note"):
        ff.review(s3, "accept", " ", "x")
    assert ff.review(s3, "accept", "yusuf", "ok")["decision"] == "accept"


def test_a_review_that_does_not_name_the_run_or_the_result_does_not_count(session):
    ff._write_record(session, ff.STARTED, {"schema": st.STARTED_SCHEMA, "run_dir": "r"})
    (session / ff.REVIEW).write_text("{}")  # a bare touch
    assert ff.unreviewed_runs([session])
    (session / ff.REVIEW).unlink()
    ff._write_record(session, ff.RESULT, {"verdict": "GO", "run_dir": "r", "rehearsal": False})
    ff.review(session, "reject", "yusuf", "no")
    assert ff.unreviewed_runs([session]) == []
    res = json.loads((session / ff.RESULT).read_text())
    res["verdict"] = "NO-GO"
    (session / ff.RESULT).write_text(json.dumps(res))  # edited after the review
    assert "changed after" in ff.unreviewed_runs([session])[0]


def test_a_review_copied_from_another_session_does_not_count(session, tmp_path):
    ff._write_record(session, ff.STARTED, {"schema": st.STARTED_SCHEMA, "run_dir": "r1"})
    ff.review(session, "reject", "a", "b")
    other = tmp_path / "other"
    other.mkdir()
    ff._write_record(other, ff.STARTED, {"schema": st.STARTED_SCHEMA, "run_dir": "r2"})
    (other / ff.REVIEW).write_text((session / ff.REVIEW).read_text())
    assert "different run" in ff.unreviewed_runs([other])[0]


def test_a_deleted_started_record_does_not_hide_a_walk_that_ran(session):
    (session / "FF_walk_1").mkdir()
    assert (
        st.unreviewed_runs([session]) == []
    )  # a refused start leaves an empty directory: harmless
    (session / "FF_walk_1" / "bridge.jsonl").write_text("{}\n")
    assert "deleted" in st.unreviewed_runs([session])[0]
    assert st.unreviewed_runs([session], current_run=session / "FF_walk_1") == []


def test_the_walk_that_is_starting_is_not_blocked_by_its_own_record(session):
    ff._write_record(session, ff.STARTED, {"schema": st.STARTED_SCHEMA, "run_dir": "FF_walk_1"})
    assert st.unreviewed_runs([session])
    assert st.unreviewed_runs([session], current_run=session / "FF_walk_1") == []
    assert st.unreviewed_runs([session], current_run=session / "FF_walk_2")  # any other run is not


# ------------------------------------------------------------------ drill analysis
def test_drill_go_from_a_real_release_while_standing(session):
    rec = _drill(session)
    assert rec["verdict"] == "GO", rec["problems"]
    assert rec["stop_latency_s"] <= ff.STOP_LATENCY_MAX_S
    assert rec["live"] is True and set(rec["artifact_sha256"]) == set(ff.MODEL_ROLES)
    assert rec["e2e_stop"]["measured"] and 0.5 < rec["e2e_stop"]["remote_last_held_to_zero_s"] < 0.6
    assert rec["runtime_identity"] == RT
    assert ff.drill_problems(session, _ident(), RT) == []


def test_drill_without_a_release_is_no_go(session):
    d = _run(session, "FF_drill_001", _ticks(END, 5.0145), trace=False)
    rec = ff.drill_eval(d, session, SHA, RT)
    assert rec["verdict"] == "NO-GO" and any(
        "needs a deadman_release" in p for p in rec["problems"]
    )
    assert ff.drill_problems(session, _ident(), RT)  # a NO-GO drill never satisfies the walk gate


@pytest.mark.parametrize(
    "case",
    [
        "seed50",
        "wrong_code",
        "rehearsal",
        "no_log",
        "stale_ls",
        "tilt",
        "clip",
        "no_answer",
        "bad_gains",
        "no_attempt",
        "identity_changed",
        "no_snapshot",
        "bridge_dead",
        "dirty_stop",
        "missing_process_log",
        "no_trace",
        "slow_stop",
        "early_release",
        "late_release",
    ],
)
def test_drill_fails_closed(session, case):
    ticks, man, kw, eps = _ticks(REL, 2.0), _manifest(), {}, 2.0
    runtime = RT
    if case == "seed50":
        man["artifacts"]["policy.onnx"]["sha256"] = ff.SEED50_HASHES["policy.onnx"]
    elif case == "wrong_code":
        man["code_identity"]["sha"] = "d" * 40
    elif case == "rehearsal":
        man["live"] = False
    elif case == "no_log":
        kw["parquet"] = False
    elif case == "stale_ls":
        for t in ticks:
            if t["mode"] == "policy":
                t["lowstate_fresh"] = False
    elif case == "tilt":
        for t in ticks:
            if t["mode"] == "policy":
                t["tilt_deg"] = 9.0
    elif case == "clip":
        for t in ticks:
            if t["mode"] == "policy":
                t["slew_clip"] = [True] + [False] * 11
    elif case == "no_answer":
        kw["answer"] = "n"
    elif case == "bad_gains":
        man["gate_params"]["kp"] = 60.0
    elif case == "no_attempt":
        kw["attempt"] = False
    elif case == "identity_changed":
        runtime = RT | {"boot_id": "boot-2"}  # evaluated under a different identity than it started
    elif case == "no_snapshot":
        kw["snaps"] = False
    elif case == "bridge_dead":
        kw["alive"] = {"deadman": True, "lowstate_bridge": True, "lowcmd_bridge": False}
        kw["reason"] = "exception"
    elif case == "dirty_stop":
        kw["reason"] = "exception"
    elif case == "missing_process_log":
        kw["logs"] = False
    elif case == "no_trace":
        kw["trace"] = False
    elif case == "slow_stop":
        kw["trace"] = _trace(ticks, e2e_s=1.6)
    elif case == "early_release":
        ticks, eps = _ticks(REL, 0.5), 0.5
    elif case == "late_release":
        ticks, eps = _ticks(REL, 4.6), 4.6
    d = _run(session, "FF_drill_001", ticks, manifest=man, **kw)
    rec = ff.drill_eval(d, session, SHA, runtime)
    assert rec["verdict"] == "NO-GO" and rec["problems"], (case, eps)


def test_stop_latency_gate_measures_resolvable_ticks(session):
    """A release window wider than the threshold means the log is too coarse to claim 0.0 s."""
    ticks = _ticks(REL, 2.0)
    k = max(i for i, t in enumerate(ticks) if t["mode"] == "policy")
    for t in ticks[k + 1 :]:
        t[
            "t_mono_ns"
        ] += 200_000_000  # a 0.2 s gap between the last policy tick and the release read
    d = _run(session, "FF_drill_001", ticks)
    rec = ff.drill_eval(d, session, SHA, RT)
    assert rec["verdict"] == "NO-GO" and any(
        "too coarse" in p or "spacing" in p for p in rec["problems"]
    )


def test_malformed_run_evidence_is_an_analyzer_failure_and_no_go(session):
    d = session / "FF_drill_001"
    d.mkdir()
    (d / "bridge.jsonl").write_text("not json\n")
    rec = ff.drill_eval(d, session, SHA, RT)
    assert rec["verdict"] == "NO-GO" and "analyzer failure" in rec["problems"][0]
    assert ff.drill_problems(session, _ident(), RT)
    d2 = session / "FF_walk_001"
    d2.mkdir()
    r2 = ff.walk_eval(d2, session, SHA, RT)  # no bridge.jsonl at all
    assert r2["verdict"] == "NO-GO" and "analyzer failure" in r2["problems"][0]


def test_drill_records_are_per_attempt_and_a_retry_is_analysed(session):
    """The first version wrote ONE write-once drill file: a NO-GO drill made the retry crash."""
    bad = _run(session, "FF_drill_001", _ticks(END, 5.0145), trace=False)
    assert ff.drill_eval(bad, session, SHA, RT)["verdict"] == "NO-GO"
    assert ff.drill_problems(session, _ident(), RT)
    rec = _drill(session, "FF_drill_002")
    assert rec["verdict"] == "GO"
    assert ff.drill_problems(session, _ident(), RT) == []  # the NEWEST attempt counts
    assert len(list(session.glob("free_floor_drill_FF_drill_*.json"))) == 2
    with pytest.raises(FileExistsError):
        ff.drill_eval(
            session / "FF_drill_002", session, SHA, RT
        )  # one record per run, never overwritten


def test_an_interrupted_newer_drill_withdraws_an_older_go(session):
    _drill(session, "FF_drill_001")
    assert ff.drill_problems(session, _ident(), RT) == []
    (session / "FF_drill_002").mkdir()  # started (or crashed) after: no record
    why = ff.drill_problems(session, _ident(), RT)
    assert why and "no analysed record" in why[0]


def test_stale_foreign_or_changed_runtime_drill_does_not_count(session):
    _drill(session)
    assert ff.drill_problems(session, _ident(sha="e" * 40), RT)  # different code
    for key, val in (
        ("boot_id", "boot-2"),  # the payload rebooted
        ("deadman", "joy"),  # another deadman source
        ("execution_set_sha256", "git-tree:other"),  # a file changed
        ("config_file_sha256", "abc"),
        ("bundle_sums_sha256", "abc"),
    ):
        why = ff.drill_problems(session, _ident(), RT | {key: val})
        assert why and any(key in w for w in why), key
    path = next(session.glob("free_floor_drill_FF_drill_*.json"))
    rec = json.loads(path.read_text())
    rec["utc"] = "2020-01-01T00:00:00Z"
    path.write_text(json.dumps(rec))
    assert any("stale" in p for p in ff.drill_problems(session, _ident(), RT))


def test_e2e_stop_is_measured_from_the_last_held_remote_message():
    ticks = _ticks(REL, 2.0)
    ex = ga.policy_exit(ticks, 5.0)
    out = ff.e2e_stop(_trace(ticks, 0.55), ex)
    assert out["measured"] and abs(out["remote_last_held_to_zero_s"] - 0.55) < 1e-6
    assert out["resolution_s"] == 0.02 and "finger" in out["resolution_note"]
    assert not ff.e2e_stop([], ex)["measured"]
    assert not ff.e2e_stop(_trace(ticks), {"t_zero_ns": None, "t_release_obs_ns": None})["measured"]


# ------------------------------------------------------------------ walk analysis
def test_walk_go_for_the_one_commissioning_run(session):
    rec = _walk(session)
    assert rec["verdict"] == "GO", rec["problems"]
    assert "STOP" in rec["next"] and "authorises nothing longer" in rec["next"]
    rep = rec["report"]
    assert rep["valid"] is True and rep["end_condition"] == "planned_end"
    assert rep["safety_event"] is False and rep["commanded_max_abs_vx_vy_wz"] == [0.3, 0.0, 0.0]
    assert rep["odometry"]["reliable"] is False
    assert rep["operator_measurement"]["measured"] is False  # nothing entered: said so, not guessed
    assert rep["estimated_pd_torque"]["estimate_only"] is True


def test_walk_report_carries_the_tape_measurement(session):
    d = _run(session, "FF_walk_001", _ticks(END, WALK_S), walk=GOOD_WALK)
    ff.measure(d, 41.0, -3.0, 5.0, "yusuf", "rear feet, both averaged")
    rec = ff.walk_eval(d, session, SHA, RT)
    assert rec["verdict"] == "GO", rec["problems"]
    assert rec["report"]["operator_measurement"]["forward_cm"] == 41.0
    assert any(e["path"].endswith(ff.MEASUREMENT) for e in rec["evidence_files"])


def test_measurement_is_structured_bounded_and_never_overwritten(tmp_path):
    with pytest.raises(ValueError, match="name"):
        ff.measure(tmp_path, 40.0, 0.0, None, " ", "")
    with pytest.raises(ValueError, match="units"):
        ff.measure(tmp_path, 4000.0, 0.0, None, "y", "")  # mm typed as cm
    ff.measure(tmp_path, 40.0, 0.0, None, "y", "")
    with pytest.raises(FileExistsError):
        ff.measure(tmp_path, 41.0, 0.0, None, "y", "")


@pytest.mark.parametrize(
    "case",
    [
        "yaw",
        "preset4s",
        "over_cmd",
        "incomplete",
        "release",
        "seed50",
        "clip",
        "no_summary",
        "short_window",
        "not_the_started_run",
        "identity_changed_since_start",
        "no_started_record",
    ],
)
def test_walk_fails_closed(session, case):
    ticks, man, walk = _ticks(END, WALK_S), _manifest(), dict(GOOD_WALK)
    kw = {}
    if case == "yaw":
        walk["max_abs"] = [0.3, 0.0, 0.5]
    elif case == "preset4s":
        walk["preset"] = "vx03_4s"
    elif case == "over_cmd":
        walk["max_abs"] = [0.5, 0.0, 0.0]
    elif case == "incomplete":
        walk["outcome"] = "deadman_released"
    elif case == "release":
        ticks = _ticks(REL, 2.0)
    elif case == "seed50":
        man["artifacts"]["checkpoint"]["sha256"] = ff.SEED50_HASHES["checkpoint"]
    elif case == "clip":
        for t in ticks:
            if t["mode"] == "policy":
                t["limit_clip"] = [True] + [False] * 11
    elif case == "no_summary":
        walk = None
    elif case == "short_window":
        ticks = _ticks(END, 5.0)
    elif case == "not_the_started_run":
        ff._write_record(session, ff.STARTED, {"schema": st.STARTED_SCHEMA, "run_dir": "other"})
        kw["started"] = False
    elif case == "identity_changed_since_start":
        ff._write_record(
            session,
            ff.STARTED,
            {
                "schema": st.STARTED_SCHEMA,
                "run_dir": "FF_walk_001",
                "runtime_identity": RT | {"boot_id": "x"},
            },
        )
        kw["started"] = False
    elif case == "no_started_record":
        kw["started"] = False
    rec = ff.walk_eval(
        _run(session, "FF_walk_001", ticks, manifest=man, walk=walk, **kw), session, SHA, RT
    )
    assert rec["verdict"] == "NO-GO" and rec["problems"], case


def test_a_result_is_never_overwritten(session):
    _walk(session, "FF_walk_001")
    with pytest.raises(FileExistsError):
        ff.walk_eval(
            _run(session, "FF_walk_002", _ticks(END, WALK_S), walk=GOOD_WALK, started=False),
            session,
            SHA,
            RT,
        )


# ------------------------------------------------------------------ gate on the real artifacts
_walk_cfg = yaml.safe_load(WALK_CFG.read_text())
HAVE_ARTIFACTS = all(p.is_file() for p in artifact_paths(_walk_cfg).values())
needs_artifacts = pytest.mark.skipif(
    not HAVE_ARTIFACTS, reason="archived model_2750 artifacts not on this machine"
)


def _stage(session, stage, *, lock=STAND_LOCK, cfg=STAND_CFG, ident=None, **extra):
    """A GO stage record through the production writer (a FIXTURE, not hardware evidence)."""
    run_dir = session / f"run_{stage}1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "bridge.jsonl").write_text('{"fixture": true}\n')
    ctx = types.SimpleNamespace(
        identity=ident or resolve_code_identity(REPO),
        id_problems=[],
        cfg_path=cfg,
        lock_path=lock,
        lock=load_lock(lock),
        lock_sha=preflight.file_sha256(lock),
        session=session,
        label="walk-iter2750-stand",
    )
    preflight.write_stage(
        ctx,
        stage,
        [pe.Check(name="fixture", ok=True, detail="test only")],
        {"run_dir": str(run_dir), **extra},
    )


@pytest.fixture
def ready(tmp_path, monkeypatch):
    monkeypatch.delenv("PHOENIX_REHEARSAL", raising=False)
    hw = tmp_path / "hw"
    prior = hw / "20261006_prior"
    prior.mkdir(parents=True)
    for stage in "CFGH":
        _stage(prior, stage)
    sess = hw / "20261008_now"
    sess.mkdir()
    ident = _ident()
    _stage(sess, "A", mode="payload", ident=ident)
    return types.SimpleNamespace(hw=hw, sess=sess, dirs=[prior, sess], ident=ident)


def _check(r, phase="drill", cfg=WALK_CFG, lock=WALK_LOCK, **kw):
    cfgd = yaml.safe_load(cfg.read_text())
    kw.setdefault("ident", r.ident)
    kw.setdefault("env", CLEAN_ENV)
    kw.setdefault("session_root", r.hw)
    return ff.check(phase, cfgd, cfg, lock, r.sess, sessions=r.dirs, preset="vx03_2s", **kw)


@needs_artifacts
def test_correct_model_2750_is_accepted_for_the_drill_and_the_walk_needs_the_drill(ready):
    assert _check(ready, "drill") == []
    why = _check(ready, "walk")
    assert any("no STOP/deadman drill" in w for w in why) and len(why) == 1


@needs_artifacts
def test_a_go_drill_opens_the_walk_and_the_walk_closes_the_session(ready):
    _drill(ready.sess)
    assert _check(ready, "walk", runtime=RT) == []
    ff._write_record(ready.sess, ff.STARTED, {"schema": st.STARTED_SCHEMA, "run_dir": "x"})
    why = _check(ready, "walk", runtime=RT)
    assert any("already started in this session" in w for w in why)
    assert any("not validly reviewed" in w for w in why)
    ff.review(ready.sess, "reject", "y", "n")
    why = _check(ready, "walk", runtime=RT)  # reviewed, but the session is closed to a second walk
    assert why and all("already started" in w for w in why)


@needs_artifacts
def test_an_unreviewed_walk_in_another_session_blocks_this_one(ready):
    ff._write_record(ready.dirs[0], ff.STARTED, {"schema": st.STARTED_SCHEMA, "run_dir": "x"})
    assert any("not validly reviewed" in w for w in _check(ready, "drill"))


@needs_artifacts
def test_drill_attempts_are_capped(ready):
    for i in range(ff.MAX_DRILL_ATTEMPTS):
        (ready.sess / f"FF_drill_00{i}").mkdir()
    assert any("drill attempts already" in w for w in _check(ready, "drill"))


@needs_artifacts
def test_seed_50_config_and_lock_are_rejected_even_with_everything_else_green(ready):
    why = _check(ready, "drill", cfg=S50_CFG, lock=S50_LOCK)
    assert why and any("SEED 50" in w for w in why)
    assert any("semantic hash" in w or "not the incumbent" in w for w in why)


@needs_artifacts
def test_seed_50_config_with_the_incumbent_lock_and_the_reverse_are_rejected(ready):
    for cfg, lock in ((S50_CFG, WALK_LOCK), (WALK_CFG, S50_LOCK)):
        why = _check(ready, "drill", cfg=cfg, lock=lock)
        assert why, (cfg.name, lock.name)


@needs_artifacts
def test_wrong_checkpoint_is_rejected(ready, tmp_path):
    cfg = yaml.safe_load(WALK_CFG.read_text())
    ck = tmp_path / "model_2750.pt"
    ck.write_bytes(b"seed 50 bytes under the incumbent's file name")
    cfg["policy"]["checkpoint_path"] = str(ck)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    why = ff.check(
        "drill",
        cfg,
        p,
        WALK_LOCK,
        ready.sess,
        sessions=ready.dirs,
        ident=ready.ident,
        env=CLEAN_ENV,
        session_root=ready.hw,
    )
    assert any("checkpoint" in w and "unknown artifact" in w for w in why)


@needs_artifacts
def test_wrong_lock_is_rejected(ready):
    why = _check(ready, "drill", lock=STAND_LOCK)
    assert why


@pytest.mark.parametrize("missing", ["C", "F", "G", "H"])
@needs_artifacts
def test_missing_stand_or_deadman_evidence_is_rejected(ready, missing):
    (ready.dirs[0] / f"stage_{missing}.json").unlink()
    assert any(f"stage {missing}" in w for w in _check(ready, "drill"))


@needs_artifacts
def test_rehearsal_stage_evidence_does_not_count_live_and_live_does_not_count_in_rehearsal(
    ready, monkeypatch
):
    monkeypatch.setenv("PHOENIX_REHEARSAL", "1")
    _stage(ready.dirs[0], "F", mode="x")  # moves the old record to history, writes a rehearsal one
    monkeypatch.delenv("PHOENIX_REHEARSAL")
    assert any("stage F" in w for w in _check(ready, "drill"))
    # stage A written live is refused by a rehearsal invocation, and vice versa
    monkeypatch.setenv("PHOENIX_REHEARSAL", "1")
    why = _check(ready, "drill", env={"PHOENIX_REHEARSAL": "1", "ROS_LOCALHOST_ONLY": "1"})
    assert any("rehearsal flag does not match" in w for w in why)


@needs_artifacts
def test_stage_a_must_cover_the_files_now_running_and_be_on_the_payload(ready):
    assert not any("stage A" in w for w in _check(ready, "drill"))
    other = _ident(sha="d" * 40)
    assert any("stage A is not for the code" in w for w in _check(ready, "drill", ident=other))
    # same commit string, different files (a tampered or re-synced tree): the digest differs
    tampered = CodeIdentity(
        sha=SHA,
        source="payload_sync",
        branch="feat/phoenix-walk-v1-hw",
        dirty=False,
        execution_set_sha256="payload:other",
    )
    why = _check(ready, "drill", ident=tampered)
    assert any("does not cover the files now running" in w for w in why)
    # an identity with problems (manifest mismatch) blocks regardless of the sha
    broken = _ident(problems=["1 synced file(s) no longer match PAYLOAD_SHA256SUMS"])
    assert any(w.startswith("code now:") for w in _check(ready, "drill", ident=broken))
    assert any("dirty" in w for w in _check(ready, "drill", ident=_ident(dirty=True)))
    assert any("branch" in w for w in _check(ready, "drill", ident=_ident(branch="main")))
    # a stage A that predates the digest (no execution_set_sha256) never counts
    rec = json.loads((ready.sess / "stage_A.json").read_text())
    rec["code_identity"].pop("execution_set_sha256")
    (ready.sess / "stage_A.json").write_text(json.dumps(rec))
    assert any("does not cover the files" in w for w in _check(ready, "drill"))
    rec["mode"] = "workstation"
    (ready.sess / "stage_A.json").write_text(json.dumps(rec))
    assert any("need the on-robot --payload run" in w for w in _check(ready, "drill"))


@needs_artifacts
def test_bundle_must_be_what_the_config_loads(ready, tmp_path):
    cfg = yaml.safe_load(WALK_CFG.read_text())
    paths = artifact_paths(cfg)
    b = tmp_path / "bundle"
    b.mkdir()
    for p in paths.values():
        (b / p.name).write_bytes(p.read_bytes())
    (b / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n" for f in sorted(b.iterdir())
        )
    )
    # the bundle is a verified copy, but the config still loads the ARCHIVE: not the same files
    why = ff.containment_problems(cfg, WALK_CFG, b)
    assert why and all("outside the bundle" in w for w in why)
    # a symlink among the loaded files is refused even when it points inside the bundle
    link = b / "linked.onnx"
    link.symlink_to(b / "policy.onnx")
    cfg["policy"]["onnx_path"] = str(link)
    assert any("symlink" in w for w in ff.containment_problems(cfg, WALK_CFG, b))


# ------------------------------------------------------------------ harness wiring (static + dry)
HARNESS_TEXT = HARNESS.read_text()


def _fn(name):
    start = HARNESS_TEXT.index(f"\n{name}() {{")
    return HARNESS_TEXT[start : HARNESS_TEXT.index("\n}\n", start)]


def test_generic_w_stays_the_gantry_command_and_cannot_reach_free_floor():
    walk = _fn("stage_walk")
    assert 'live_confirm "W $preset ON GANTRY"' in walk
    assert 'free_floor_gate.py" unreviewed' in walk  # but an unreviewed floor walk latches it
    assert "FREE_FLOOR" not in walk.replace("free-floor", "")
    assert "W) stage_walk" in HARNESS_TEXT and "FREE_FLOOR) stage_free_floor" in HARNESS_TEXT


def test_free_floor_runs_one_walk_command_once_and_never_chains():
    f = _fn("stage_free_floor")
    assert f.count("-m phoenix.sim2real.walk_command") == 1 and f.count("--start-deadline-s") == 1
    body = f.split("\n", 2)[2]  # without the "stage_free_floor() {" header
    assert "stage_walk" not in body and "stage_free_floor" not in body and "stage_stand" not in body
    assert "${sub}-eval" in f and "review accept|reject" in f
    confirm = _fn("free_floor_confirm")
    for needle in ("LEVEL", "NO gantry", "FALL AREA", "SPOTTER", "DEADMAN", "NO GANTRY", "0.6 s"):
        assert needle in confirm


def test_the_run_start_record_precedes_every_process_that_can_move_a_motor():
    f = _fn("stage_free_floor")
    begin = f.index('"$ff" begin --kind "$sub"')
    assert begin < f.index("launch deadman") < f.index("launch lowcmd_bridge")
    assert (
        f.index("operator_gate arm") < begin
    )  # the operator's arm comes first, then the final re-check
    assert f.index("free_floor_confirm") < f.index("new_run") < begin
    for launch in (
        "launch lowstate_bridge",
        "launch policy_node",
        "launch walk_cmd",
        "launch remote_trace",
    ):
        assert begin < f.index(launch)
    assert 'halt "free-floor $sub could not start' in f  # a refused begin stops the run


def test_free_floor_pins_the_values_a_stale_shell_could_change():
    f = _fn("stage_free_floor")
    assert "standup_s=2.0" in f and "PHOENIX_STANDUP_S" not in f
    assert "PHOENIX_WALK_AUTHORITY_MARGIN_S" not in f and "walk_dur + 4.0" in f
    assert '--lock "$DEPLOY_LOCK"' in f and '--authority-s "$authority"' in f
    assert "alive_at_end" in f and "snapshot --label start" in f and "snapshot --label end" in f


def _harness(args, env=None):
    import os

    e = dict(os.environ) | {
        "PHOENIX_SESSION": str(env["session"]),
        "PHOENIX_EXPECT_SHA": resolve_code_identity(REPO).sha,
    }
    return subprocess.run(
        ["bash", str(HARNESS), *args],
        capture_output=True,
        text=True,
        env=e,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )


def test_harness_refuses_free_floor_yaw_and_unknown_subcommands_without_motion(tmp_path):
    r = _harness(["FREE_FLOOR", "walk", "yaw05_3s"], {"session": tmp_path})
    assert r.returncode != 0 and "launched" not in r.stdout
    r = _harness(["FREE_FLOOR", "walk"], {"session": tmp_path})
    assert r.returncode != 0 and "typed explicitly" in r.stderr
    r = _harness(["FREE_FLOOR", "sprint"], {"session": tmp_path})
    assert r.returncode != 0 and "usage" in r.stderr
    r = _harness(
        ["FREE_FLOOR", "drill"], {"session": tmp_path}
    )  # default stand config: not the incumbent walking config
    assert r.returncode != 0 and "launched" not in r.stdout and "no motion" in r.stderr


# ------------------------------------------------------------------ desk readiness and lab commands
def test_lab_commands_are_resolved_complete_and_password_free():
    import free_floor_desk as desk

    text = desk.lab_commands("orin")
    sha = resolve_code_identity(REPO).sha
    assert sha in text and sha[:12] in text
    for needle in (
        "lablink doctor",
        "lablink status orin",
        "lablink resolve orin",
        "stage_payload_repo.sh",
        "stage_payload_bundle.sh",
        "derive_walk_bundle.sh",
        "A --payload",
        "FREE_FLOOR gate",
        "FREE_FLOOR drill",
        "FREE_FLOOR walk vx03_2s",
        "FREE_FLOOR review accept",
        "env -i",
    ):
        assert needle in text, needle
    assert "JETSON_PW=" not in text and "password" not in text.lower().replace("no passwords", "")
    assert "<" not in text.replace("<paste $UTC from step 2>", "").replace("<you>", "").replace(
        '"<what you saw>"', ""
    ).replace("<unresolved>", "").replace("< ", "").replace("<<", "")


def test_desk_checks_never_claim_robot_readiness(capsys):
    import free_floor_desk as desk

    assert desk.pending()
    assert any("DRILL" in k for k in desk.pending()) and any("PAYLOAD" in k for k in desk.pending())
    rc = desk.main_desk(types.SimpleNamespace(cmd="desk-ready", json=False))
    out = capsys.readouterr().out
    assert (
        "VERDICT:" in out and "NOT robot readiness" in out and "PAYLOAD VERIFICATION PENDING" in out
    )
    assert rc in (0, 1)
