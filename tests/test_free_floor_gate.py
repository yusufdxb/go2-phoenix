"""Incumbent-only FREE_FLOOR gate (scripts/free_floor_gate.py): identity, evidence, limits.

Run logs here are built from VERBATIM real bridge tick records
(tests/fixtures/gantry_ab/real_policy_exits.json: a real deadman release and a real planned
end on the GO2), lengthened the same way tests/test_gantry_ab.py does. Stage records are
written by the production writer under the REAL stand lock; they are test fixtures, not
hardware evidence. Tests that need the archived model_2750 files skip without them (CI).
No robot, ROS or network.
"""

from __future__ import annotations

import copy
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

from phoenix.sim2real import preflight  # noqa: E402
from phoenix.sim2real import preflight_eval as pe  # noqa: E402
from phoenix.sim2real.deploy_contract import (  # noqa: E402
    artifact_paths,
    load_lock,
)
from phoenix.sim2real.provenance import resolve_code_identity  # noqa: E402

FIX = json.loads((REPO / "tests/fixtures/gantry_ab/real_policy_exits.json").read_text())
REL = FIX["release"]["ticks"]
END = FIX["planned_end"]["ticks"]
SHA = "c" * 40


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
        "artifacts": {
            r: {"sha256": ff.INCUMBENT[r]}
            for r in ("checkpoint", "policy.onnx", "policy.onnx.data")
        },
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


def _run(session, name, ticks, *, manifest=None, answer="y", parquet=True, walk=None):
    d = session / name
    d.mkdir(parents=True)
    faults = sorted({f for t in ticks for f in (t.get("faults") or [])} | {"bridge_shutdown"})
    rows = [manifest or _manifest(), *ticks, {"record": "end", "faults": faults}]
    (d / "bridge.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if parquet:
        (d / "policy.parquet").write_bytes(b"PAR1 not a real parquet, size only")
    if answer is not None:
        (d / "operator_answer.txt").write_text(answer + "\n")
    if walk is not None:
        (d / "walk_cmd.jsonl").write_text(json.dumps(walk) + "\n")
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


def test_bundle_with_wrong_files_is_rejected(tmp_path):
    b = tmp_path / "bundle"
    b.mkdir()
    for name, data in (("policy.onnx", b"a"), ("policy.onnx.data", b"b"), ("m2750.pt", b"c")):
        (b / name).write_bytes(data)
    import hashlib

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


# ------------------------------------------------------------------ review / one run at a time
def test_started_run_blocks_everything_until_a_human_reviews(session):
    assert ff.unreviewed_runs([session]) == []
    ff._write_record(session, ff.STARTED, {"run_dir": "x"})
    assert "no human review" in ff.unreviewed_runs([session])[0]
    with pytest.raises(ValueError, match="no GO result"):
        ff.review(session, "accept", "yusuf", "looks fine")  # no result: cannot accept
    ff.review(session, "reject", "yusuf", "analysis failed, investigate")
    assert ff.unreviewed_runs([session]) == []
    with pytest.raises(FileExistsError):
        ff.review(session, "reject", "yusuf", "again")


def test_accept_needs_a_go_result_and_a_named_reviewer(session):
    ff._write_record(session, ff.STARTED, {})
    ff._write_record(session, ff.RESULT, {"verdict": "NO-GO"})
    with pytest.raises(ValueError, match="no GO result"):
        ff.review(session, "accept", "yusuf", "x")
    s2 = session.parent / "s2"
    s2.mkdir()
    ff._write_record(s2, ff.STARTED, {})
    ff._write_record(s2, ff.RESULT, {"verdict": "GO"})
    with pytest.raises(ValueError, match="name and a note"):
        ff.review(s2, "accept", " ", "x")
    assert ff.review(s2, "accept", "yusuf", "ok")["decision"] == "accept"


# ------------------------------------------------------------------ drill analysis
def test_drill_go_from_a_real_release_while_standing(session):
    d = _run(session, "FF_drill", _ticks(REL, 2.0))
    rec = ff.drill_eval(d, session, SHA)
    assert rec["verdict"] == "GO", rec["problems"]
    assert rec["stop_latency_s"] <= ff.STOP_LATENCY_MAX_S
    assert rec["live"] is True and set(rec["artifact_sha256"]) == {
        "checkpoint",
        "policy.onnx",
        "policy.onnx.data",
    }
    assert ff.drill_problems(session, SHA) == []


def test_drill_without_a_release_is_no_go(session):
    rec = ff.drill_eval(_run(session, "FF_drill", _ticks(END, 5.0145)), session, SHA)
    assert rec["verdict"] == "NO-GO" and any(
        "needs a deadman_release" in p for p in rec["problems"]
    )
    assert any("no STOP/deadman drill" not in p for p in rec["problems"])
    assert ff.drill_problems(session, SHA)  # a NO-GO drill never satisfies the walk gate


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
    ],
)
def test_drill_fails_closed(session, case):
    ticks, man, kw = _ticks(REL, 2.0), _manifest(), {}
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
    rec = ff.drill_eval(_run(session, "FF_drill", ticks, manifest=man, **kw), session, SHA)
    assert rec["verdict"] == "NO-GO" and rec["problems"], case


def test_malformed_run_evidence_is_an_analyzer_failure_and_no_go(session):
    d = session / "FF_drill"
    d.mkdir()
    (d / "bridge.jsonl").write_text("not json\n")
    rec = ff.drill_eval(d, session, SHA)
    assert rec["verdict"] == "NO-GO" and "analyzer failure" in rec["problems"][0]
    assert ff.drill_problems(session, SHA)
    d2 = session / "FF_walk"
    d2.mkdir()
    r2 = ff.walk_eval(d2, session, SHA)  # no bridge.jsonl at all
    assert r2["verdict"] == "NO-GO" and "analyzer failure" in r2["problems"][0]


def test_stale_or_foreign_drill_does_not_count(session):
    ff.drill_eval(_run(session, "FF_drill", _ticks(REL, 2.0)), session, SHA)
    assert ff.drill_problems(session, "e" * 40)  # different code
    rec = json.loads((session / ff.DRILL).read_text())
    rec["utc"] = "2020-01-01T00:00:00Z"
    (session / ff.DRILL).write_text(json.dumps(rec))
    assert any("stale" in p for p in ff.drill_problems(session, SHA))


# ------------------------------------------------------------------ walk analysis
def test_walk_go_for_the_one_commissioning_run(session):
    d = _run(session, "FF_walk", _ticks(END, WALK_S), walk=GOOD_WALK)
    rec = ff.walk_eval(d, session, SHA)
    assert rec["verdict"] == "GO", rec["problems"]
    assert "STOP" in rec["next"] and "authorises nothing longer" in rec["next"]


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
    ],
)
def test_walk_fails_closed(session, case):
    ticks, man, walk = _ticks(END, WALK_S), _manifest(), dict(GOOD_WALK)
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
    rec = ff.walk_eval(_run(session, "FF_walk", ticks, manifest=man, walk=walk), session, SHA)
    assert rec["verdict"] == "NO-GO" and rec["problems"], case


def test_a_result_is_never_overwritten(session):
    ff.walk_eval(_run(session, "a", _ticks(END, WALK_S), walk=GOOD_WALK), session, SHA)
    with pytest.raises(FileExistsError):
        ff.walk_eval(_run(session, "b", _ticks(END, WALK_S), walk=GOOD_WALK), session, SHA)


# ------------------------------------------------------------------ gate on the real artifacts
_walk_cfg = yaml.safe_load(WALK_CFG.read_text())
HAVE_ARTIFACTS = all(p.is_file() for p in artifact_paths(_walk_cfg).values())
needs_artifacts = pytest.mark.skipif(
    not HAVE_ARTIFACTS, reason="archived model_2750 artifacts not on this machine"
)


def _stage(session, stage, *, lock=STAND_LOCK, cfg=STAND_CFG, **extra):
    """A GO stage record through the production writer (a FIXTURE, not hardware evidence)."""
    run_dir = session / f"run_{stage}1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "bridge.jsonl").write_text('{"fixture": true}\n')
    ctx = types.SimpleNamespace(
        identity=resolve_code_identity(REPO),
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
    sha = resolve_code_identity(REPO).sha
    _stage(sess, "A", mode="payload")
    return types.SimpleNamespace(hw=hw, sess=sess, dirs=[prior, sess], sha=sha)


def _check(r, phase="drill", cfg=WALK_CFG, lock=WALK_LOCK, **kw):
    cfgd = yaml.safe_load(cfg.read_text())
    return ff.check(
        phase, cfgd, cfg, lock, r.sess, sessions=r.dirs, sha_now=r.sha, preset="vx03_2s", **kw
    )


@needs_artifacts
def test_correct_model_2750_is_accepted_for_the_drill_and_then_the_walk(ready):
    assert _check(ready, "drill") == []
    assert _check(ready, "walk"), "walk must need the STOP/deadman drill first"
    ff.drill_eval(
        _run(
            ready.sess,
            "FF_drill",
            _ticks(REL, 2.0),
            manifest=_manifest(code_identity={"sha": ready.sha}),
        ),
        ready.sess,
        ready.sha,
    )
    assert _check(ready, "walk") == []


@needs_artifacts
def test_seed_50_config_and_lock_are_rejected_even_with_everything_else_green(ready):
    if not S50_CFG.is_file():
        pytest.skip("seed 50 config absent")
    why = _check(ready, "drill", cfg=S50_CFG, lock=S50_LOCK)
    assert any("SEED 50" in w for w in why) or any(
        "unknown artifact" in w or "not the incumbent" in w for w in why
    )
    assert any("not the incumbent walking deploy config" in w for w in why)


@needs_artifacts
def test_wrong_checkpoint_is_rejected(ready, tmp_path):
    cfg = copy.deepcopy(_walk_cfg)
    fake = tmp_path / "other.pt"
    fake.write_bytes(b"not model_2750")
    cfg["policy"]["checkpoint_path"] = str(fake)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    why = _check(ready, "drill", cfg=p)
    assert any("checkpoint" in w and ("unknown artifact" in w or "lock records" in w) for w in why)


@needs_artifacts
def test_wrong_lock_is_rejected(ready):
    if not STAND_LOCK.is_file():
        pytest.skip()
    why = _check(ready, "drill", lock=STAND_LOCK)
    assert any("not the incumbent walking lock" in w or "lock:" in w for w in why)


@needs_artifacts
@pytest.mark.parametrize("missing", ["F", "G", "H", "C"])
def test_missing_stand_or_deadman_evidence_is_rejected(ready, missing):
    (ready.dirs[0] / f"stage_{missing}.json").unlink()
    why = _check(ready, "drill")
    assert any(f"stage {missing}" in w for w in why), why


@needs_artifacts
def test_rehearsal_stage_evidence_does_not_count(ready, monkeypatch):
    rec = json.loads((ready.dirs[0] / "stage_H.json").read_text())
    rec["rehearsal"] = True
    (ready.dirs[0] / "stage_H.json").write_text(json.dumps(rec))
    assert any("stage H" in w for w in _check(ready, "drill"))


@needs_artifacts
def test_stage_a_must_be_for_the_running_code_and_on_the_payload(ready):
    assert any(
        "stage A is not for the code now running" in w
        for w in ff.check(
            "drill",
            _walk_cfg,
            WALK_CFG,
            WALK_LOCK,
            ready.sess,
            sessions=ready.dirs,
            sha_now="f" * 40,
        )
    )
    (ready.sess / "stage_A.json").unlink()
    _stage(ready.sess, "A", mode="workstation")
    assert any("--payload" in w for w in _check(ready))
    (ready.sess / "stage_A.json").unlink()
    assert any("stage A record missing" in w for w in _check(ready))


@needs_artifacts
def test_an_unreviewed_free_floor_run_blocks_the_next_command(ready):
    assert _check(ready, "drill") == []
    ff._write_record(ready.dirs[0], ff.STARTED, {"run_dir": "FF_walk"})
    assert any("no human review" in w for w in _check(ready, "drill"))
    ff.review(ready.dirs[0], "reject", "yusuf", "NO-GO result, investigate")
    assert _check(ready, "drill") == []


# ------------------------------------------------------------------ harness wiring (static + dry)
HARNESS_TEXT = HARNESS.read_text()


def _fn(name):
    start = HARNESS_TEXT.index(f"\n{name}() {{")
    return HARNESS_TEXT[start : HARNESS_TEXT.index("\n}\n", start)]


def test_generic_w_stays_the_gantry_command_and_cannot_reach_free_floor():
    walk = _fn("stage_walk")
    assert 'live_confirm "W $preset ON GANTRY"' in walk
    assert "free_floor" not in walk.lower() and "FREE_FLOOR" not in walk
    assert "W) stage_walk" in HARNESS_TEXT and "FREE_FLOOR) stage_free_floor" in HARNESS_TEXT


def test_free_floor_runs_one_walk_command_once_and_never_chains():
    f = _fn("stage_free_floor")
    assert f.count("-m phoenix.sim2real.walk_command") == 1 and f.count("--start-deadline-s") == 1
    body = f.split("\n", 2)[2]  # without the "stage_free_floor() {" header
    assert "stage_walk" not in body and "stage_free_floor" not in body and "stage_stand" not in body
    assert "mark-started" in f and f.index("mark-started") < f.index("launch deadman")
    assert "${sub}-eval" in f and "review accept|reject" in f
    confirm = _fn("free_floor_confirm")
    for needle in ("LEVEL FLOOR", "NO gantry", "FALL AREA", "SPOTTER", "DEADMAN", "NO GANTRY"):
        assert needle in confirm


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
