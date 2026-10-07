#!/usr/bin/env python3
"""Incumbent-only FREE_FLOOR commissioning gate (model_2750, first walk with no gantry).

Why this exists. The gantry rule in ``harness_preflight.sh W`` is a typed phrase
("ON GANTRY") plus lab-card process; no code path inspects whether a rope exists.
The F/G hardware-evidence gate in ``deploy_contract`` is what physically blocks
walking, and it is bound to artifact hashes, not to a gantry. This script adds a
SEPARATE, explicit, incumbent-only path for the first free-floor walk and leaves
the gantry path (``W``, the A/B campaign) untouched.

Authorisation is for ONE policy only: model_2750 (checkpoint e03013d1...). Seed 50
(model_3749, fric_T_s50) is exploratory and stays gantry-only: its hashes are
listed here as an explicit denial and a different hash is "unknown artifact".
Everything fails closed: a missing or malformed record is a problem, not a default.

Software cannot see the floor, the rope or the spotters. The live step therefore
needs an operator's typed confirmation (harness_preflight.sh FREE_FLOOR); this
script only checks what the logs can prove.

    free_floor_gate.py check  --phase drill|walk --config C --lock L --session S [--bundle B] [--preset P]
    free_floor_gate.py drill-eval --run-dir R --session S     STOP/deadman drill while standing
    free_floor_gate.py walk-eval  --run-dir R --session S     the one vx03_2s run
    free_floor_gate.py review accept|reject --session S --by NAME --note TEXT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import gantry_ab_analyze as ga  # noqa: E402  (policy_exit, bridge_metrics, load_bridge)
import yaml  # noqa: E402

from phoenix.sim2real import deploy_contract as dc  # noqa: E402
from phoenix.sim2real.provenance import resolve_code_identity  # noqa: E402
from phoenix.sim2real.walk_command import PRESETS, profile_duration  # noqa: E402

# ---------------------------------------------------------------- identity
#: model_2750 = walk_vnext3c_seed47/2026-09-26_21-06-20/model_2750.pt, the hardware-tested incumbent.
INCUMBENT = {
    "checkpoint": "e03013d19ccc5df92a58762f8b76566a745c131534e8949ff1275739378198e4",
    "policy.onnx": "6d36796f857a490b3d446492581aa1952fa9afdb689bef88b09410db10512c32",
    "policy.onnx.data": "b6ec50ed6e295aecc0bb3b6d801d3a089c890af491956b1f4f1358c25079797b",
    "policy.manifest": "c6d4a32021ea064e7079ce08f21af5d16d1161b58095d2b271a3e7ac649d209a",
    "policy.pt": "371f5a442d5afeab56c40c58d7da23cbb37a0dc37acad030425ff731bce0f110",
}
#: Semantic hash of the incumbent's walking deploy config (the lock's deploy_config.semantic_sha256).
INCUMBENT_WALK_CONFIG_SEMANTIC_SHA256 = (
    "b3b61046aa5b5b6752b45707f9c2a341b93b45d7c40b8abb15a335082aa65e80"
)
INCUMBENT_GAINS = {"kp": 25.0, "kd": 0.5, "action_clip": 100.0, "action_scale": 0.25}
#: Seed 50 (fric_T_s50, model_3749): exploratory, GANTRY-ONLY. Named so the refusal says why.
SEED50_HASHES = {
    "checkpoint": "9948900b4568cb09bd0119b58c395f8aa0309179cf5467ac43bae0cd4945ba9a",
    "policy.onnx": "4b7063b47bac0b36d4696258da613da08a14cebe57c05f026d88aca73872034f",
    "policy.onnx.data": "13946b354f0dc7cfb92951164f4e88ccfa01b4ed13ccaa6bce243f65f394e00a",
}

# ---------------------------------------------------------------- commissioning limits
#: The ONLY walking command free-floor mode can ever request (0.3 m/s forward, 2 s, no yaw).
FREE_FLOOR_PRESET = "vx03_2s"
FREE_FLOOR_PRESETS = (FREE_FLOOR_PRESET,)
FREE_FLOOR_COMMAND = (0.3, 0.0, 0.0)
DRILL_AUTHORITY_S = 5.0
DRILL_RELEASE_WINDOW_S = (1.0, 4.0)  # the release must come inside the policy episode
STOP_LATENCY_MAX_S = 0.1  # measured 0.0 s (3.2 ms window) in the A/B analyzer replay of real logs
DRILL_MAX_AGE_S = 4 * 3600  # the drill must be this session day's, on this code
TILT_MAX_DEG = 6.0  # real incumbent stand/walk runs: <= 4.94 deg
CLIP_TICKS_MAX = 0  # real incumbent runs: 0 slew and 0 limit clip ticks under policy
CMD_TOL = 1e-9
STAGE_SCHEMA = "phoenix-preflight-stage/v1"
DRILL_SCHEMA = "phoenix-free-floor-drill/v1"
RESULT_SCHEMA = "phoenix-free-floor-result/v1"
REVIEW_SCHEMA = "phoenix-free-floor-review/v1"
STARTED = "free_floor_started.json"
DRILL = "free_floor_drill.json"
RESULT = "free_floor_result.json"
REVIEW = "free_floor_review.json"


def _rehearsal() -> bool:
    """True only for a localhost rehearsal (PHOENIX_REHEARSAL=1). A rehearsal invocation accepts
    only rehearsal records for the free-floor steps and a live one accepts only live records, so
    neither can stand in for the other."""
    return os.environ.get("PHOENIX_REHEARSAL") == "1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> tuple[Any, str | None]:
    if path.is_symlink():
        return None, f"{path.name} is a symlink"
    try:
        return json.loads(path.read_text()), None
    except (OSError, ValueError) as exc:
        return None, f"{path.name} unreadable: {exc}"


def _utc_age_s(stamp: Any) -> float | None:
    try:
        t = datetime.strptime(str(stamp), "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        try:
            t = datetime.strptime(str(stamp), "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None
    return (datetime.now(timezone.utc) - t.replace(tzinfo=timezone.utc)).total_seconds()


# ---------------------------------------------------------------- identity gate
def hash_identity_problems(hashes: Mapping[str, str | None], what: str) -> list[str]:
    """``hashes`` (role -> sha256) must be exactly model_2750; seed 50 and unknowns are named."""
    problems: list[str] = []
    for role in ("checkpoint", "policy.onnx", "policy.onnx.data"):
        got = hashes.get(role)
        if got is None:
            problems.append(f"{what}: {role} hash missing (identity cannot be established)")
        elif role in SEED50_HASHES and got == SEED50_HASHES[role]:
            problems.append(
                f"{what}: {role} is SEED 50 (fric_T_s50 / model_3749): exploratory, gantry-only, "
                "never authorised for free-floor"
            )
        elif got != INCUMBENT[role]:
            problems.append(
                f"{what}: {role} {got[:12]} is not model_2750 ({INCUMBENT[role][:12]}): unknown artifact"
            )
    return problems


def identity_problems(
    cfg: Mapping[str, Any], cfg_path: Path, lock_path: Path, bundle: Path | None
) -> list[str]:
    problems: list[str] = []
    try:
        lock = dc.load_lock(lock_path)
    except (OSError, ValueError) as exc:
        return [f"lock {lock_path} unreadable or malformed: {exc}"]
    lock_problems, observed = dc.verify_lock(
        lock, cfg, cfg_path, required_roles=("policy.onnx", "policy.onnx.data", "policy.manifest")
    )
    problems += [f"lock: {p}" for p in lock_problems]
    arts = observed["artifacts"]
    problems += hash_identity_problems(
        {r: (arts.get(r) or {}).get("sha256") for r in arts}, "config"
    )
    problems += hash_identity_problems(
        {r: (lock.get("artifacts") or {}).get(r, {}).get("sha256") for r in INCUMBENT}, "lock"
    )
    for role, want in INCUMBENT.items():
        got = (arts.get(role) or {}).get("sha256")
        if (
            got is not None
            and got != want
            and role not in ("checkpoint", "policy.onnx", "policy.onnx.data")
        ):
            problems.append(f"config: {role} {got[:12]} differs from model_2750's {want[:12]}")
    if observed["deploy_config"]["semantic_sha256"] != INCUMBENT_WALK_CONFIG_SEMANTIC_SHA256:
        problems.append("config is not the incumbent walking deploy config (semantic hash differs)")
    if (lock.get("deploy_config") or {}).get(
        "semantic_sha256"
    ) != INCUMBENT_WALK_CONFIG_SEMANTIC_SHA256:
        problems.append("lock is not the incumbent walking lock (deploy_config.semantic_sha256)")
    if (cfg.get("safety") or {}).get("stand_only") is not False:
        problems.append("free-floor needs the WALKING config (safety.stand_only: false)")
    manifest = dc._load_manifest(cfg)
    deploy = (manifest or {}).get("deploy") or {}
    if manifest is None:
        problems.append("policy manifest unreadable (gains/action scaling cannot be checked)")
    else:
        if manifest.get("checkpoint_sha256") != INCUMBENT["checkpoint"]:
            problems.append("manifest checkpoint_sha256 is not model_2750's")
        for k in ("kp", "kd", "action_clip"):
            if deploy.get(k) != INCUMBENT_GAINS[k]:
                problems.append(
                    f"manifest deploy.{k}={deploy.get(k)!r}, incumbent {INCUMBENT_GAINS[k]}"
                )
    if (cfg.get("control") or {}).get("action_scale") != INCUMBENT_GAINS["action_scale"]:
        problems.append("control.action_scale is not the incumbent's 0.25")
    # joint order, default pose, 45-D v2 observation schema, timeouts, and the F/G hardware
    # evidence for these exact artifacts are all in the deploy contract.
    problems += [f"contract: {p}" for p in dc.validate_deploy_contract(cfg)]
    if bundle is not None:
        problems += bundle_problems(bundle)
    return problems


def bundle_problems(bundle: Path) -> list[str]:
    """The deployment bundle: SHA256SUMS verifies and its model files are model_2750."""
    sums = bundle / "SHA256SUMS"
    if not sums.is_file():
        return [f"bundle {bundle}: SHA256SUMS missing"]
    problems: list[str] = []
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        sha, _, name = line.partition("  ")
        target = bundle / name.strip().lstrip("*")
        if not target.is_file() or _sha256(target) != sha:
            problems.append(
                f"bundle {bundle.name}: {name.strip()} missing or differs from SHA256SUMS"
            )
    files = {
        "policy.onnx": bundle / "policy.onnx",
        "policy.onnx.data": bundle / "policy.onnx.data",
    }
    cks = [p for p in bundle.glob("*.pt") if p.name != "policy.pt"]
    if len(cks) != 1:
        problems.append(
            f"bundle {bundle.name}: expected exactly one source checkpoint .pt, found {len(cks)}"
        )
    else:
        files["checkpoint"] = cks[0]
    hashes = {r: (_sha256(p) if p.is_file() else None) for r, p in files.items()}
    problems += hash_identity_problems(hashes, f"bundle {bundle.name}")
    return problems


# ---------------------------------------------------------------- evidence gate
def _stage_ok(
    stage: str, cfg: Mapping[str, Any], dirs: Sequence[Path], *, motors: str
) -> tuple[bool, list[str]]:
    observed = dc.observed_artifact_hashes(cfg, "<free-floor>")["artifacts"]
    want = {r: (observed.get(r) or {}).get("sha256") for r in dc._EVIDENCE_ROLES}
    if any(v is None for v in want.values()):
        return False, ["cannot hash the incumbent artifacts"]
    twin = dc.semantic_config_sha256(dc.stand_twin(cfg))
    seen: list[str] = []
    for d in dirs:
        path = d / f"stage_{stage}.json"
        if d.is_symlink() or not path.is_file():
            continue
        rec, err = _read_json(path)
        if err or not isinstance(rec, dict):
            seen.append(f"{d.name}: {err}")
            continue
        if motors == "live":
            why = dc._stage_record_rejections(rec, stage, twin, want, d)
        else:  # stage C: motors off by design (physical deadman, no bridge)
            why = []
            if rec.get("schema") != STAGE_SCHEMA or rec.get("stage") != stage:
                why.append("wrong schema or stage")
            if rec.get("verdict") != "GO":
                why.append(f"verdict {rec.get('verdict')!r}")
            if rec.get("rehearsal"):
                why.append("rehearsal")
            if rec.get("code_identity_problems") != []:
                why.append("code identity problems")
            if rec.get("lock_semantic_sha256") != twin:
                why.append("lock is not for the stand-phase twin")
            got = rec.get("lock_artifact_sha256") or {}
            why += [f"{r} hash differs" for r, s in want.items() if got.get(r) != s]
            # The existing stage-C writer binds no evidence_files (motors off, no telemetry);
            # the live STOP/deadman evidence is the drill, which is hash-bound.
        if not why:
            return True, []
        seen.append(f"{d.name}: " + ", ".join(why))
    return False, [
        f"stage {stage}: no GO record for model_2750's stand twin ({'; '.join(seen[:3]) or 'none found'})"
    ]


def _stage_a_problems(session: Path, sha_now: str, require_payload: bool) -> list[str]:
    rec, err = _read_json(session / "stage_A.json")
    if err or not isinstance(rec, dict):
        return [f"stage A record missing or unreadable in {session}: {err}"]
    ident = rec.get("code_identity") or {}
    problems = []
    if rec.get("schema") != STAGE_SCHEMA or rec.get("stage") != "A" or rec.get("verdict") != "GO":
        problems.append("stage A is not a GO record")
    if bool(rec.get("rehearsal")) != _rehearsal():
        problems.append("stage A rehearsal flag does not match this invocation (rehearsal evidence never counts live)")
    if ident.get("sha") != sha_now or rec.get("code_identity_problems") != []:
        problems.append("stage A is not for the code now running (or has code identity problems)")
    if require_payload and rec.get("mode") != "payload":
        problems.append(f"stage A mode is {rec.get('mode')!r}, need the on-robot --payload run")
    problems += hash_identity_problems(rec.get("lock_artifact_sha256") or {}, "stage A lock")
    return problems


def unreviewed_runs(dirs: Sequence[Path]) -> list[str]:
    """Free-floor runs that were started but have no human review record. They block everything."""
    out = []
    for d in dirs:
        if (d / STARTED).exists() and not (d / REVIEW).exists():
            out.append(
                f"{d.name}: a free-floor run was started and has no human review "
                f"(run: free_floor_gate.py review accept|reject --session {d})"
            )
    return out


def drill_problems(session: Path, sha_now: str) -> list[str]:
    rec, err = _read_json(session / DRILL)
    if err or not isinstance(rec, dict):
        return [f"no STOP/deadman drill record ({DRILL}): {err}"]
    problems = []
    if rec.get("schema") != DRILL_SCHEMA or rec.get("verdict") != "GO":
        problems.append("drill record is not a GO record")
    if bool(rec.get("rehearsal")) != _rehearsal() or rec.get("live") is not True:
        problems.append("drill record is not a live run of this kind (rehearsal flag mismatch)")
    if rec.get("code_sha") != sha_now:
        problems.append("drill was run on different code than is running now")
    problems += hash_identity_problems(rec.get("artifact_sha256") or {}, "drill")
    age = _utc_age_s(rec.get("utc"))
    if age is None or age < 0 or age > DRILL_MAX_AGE_S:
        problems.append(f"drill is stale or its timestamp is invalid (age {age})")
    problems += dc._evidence_file_rejections(rec, session)
    return problems


def check(
    phase: str,
    cfg: Mapping[str, Any],
    cfg_path: Path,
    lock_path: Path,
    session: Path,
    *,
    sessions: Sequence[Path] | None = None,
    sha_now: str,
    bundle: Path | None = None,
    preset: str | None = None,
    require_payload: bool = True,
) -> list[str]:
    """Every reason the ``phase`` ("drill" or "walk") must not run. Empty means allowed."""
    if phase not in ("drill", "walk"):
        return [f"unknown phase {phase!r}"]
    if phase == "walk" and preset not in FREE_FLOOR_PRESETS:
        return [
            f"free-floor allows only {list(FREE_FLOOR_PRESETS)}, got {preset!r}: "
            "no yaw, no longer or faster command without a new human-reviewed change"
        ]
    problems = identity_problems(cfg, cfg_path, lock_path, bundle)
    dirs = [Path(d) for d in (dc.default_session_dirs() if sessions is None else sessions)]
    for stage, motors in (("C", "off"), ("F", "live"), ("G", "live"), ("H", "live")):
        ok, why = _stage_ok(stage, cfg, dirs, motors=motors)
        if not ok:
            problems += why
    problems += _stage_a_problems(session, sha_now, require_payload)
    problems += unreviewed_runs(dirs)
    if phase == "walk":
        problems += drill_problems(session, sha_now)
    return problems


# ---------------------------------------------------------------- run evaluation
def _run_checks(
    run_dir: Path, session: Path, sha_now: str, authority_s: float
) -> tuple[list[str], dict[str, Any]]:
    """Facts common to the drill and the walk, from the bridge log. Returns (problems, facts)."""
    problems: list[str] = []
    manifest, ticks, end = ga.load_bridge(run_dir)
    facts: dict[str, Any] = {"live": manifest.get("live") is True}
    gp = manifest.get("gate_params") or {}
    if manifest.get("live") is not True:
        problems.append("bridge ran without --live (rehearsal)")
    if (manifest.get("code_identity") or {}).get("sha") != sha_now or manifest.get(
        "code_identity_problems"
    ) != []:
        problems.append("bridge ran on different code than now, or reported code identity problems")
    if manifest.get("startup_problems"):
        problems.append(f"bridge startup problems: {manifest['startup_problems']}")
    if (manifest.get("deploy_config") or {}).get(
        "semantic_sha256"
    ) != INCUMBENT_WALK_CONFIG_SEMANTIC_SHA256:
        problems.append("bridge ran a config other than the incumbent walking config")
    art = manifest.get("artifacts") or {}
    problems += hash_identity_problems(
        {
            r: (art.get(r) or {}).get("sha256")
            for r in ("checkpoint", "policy.onnx", "policy.onnx.data")
        },
        "bridge manifest",
    )
    if not (gp.get("walking_allowed") and gp.get("deadman_required") and gp.get("require_imu")):
        problems.append("bridge did not run with walking_allowed, deadman_required and require_imu")
    if gp.get("kp") != INCUMBENT_GAINS["kp"] or gp.get("kd") != INCUMBENT_GAINS["kd"]:
        problems.append(f"bridge gains kp={gp.get('kp')} kd={gp.get('kd')} are not the incumbent's")
    ex = ga.policy_exit(ticks, authority_s)
    facts["policy_exit"] = ex
    if ex["problems"]:
        problems.append(f"policy exit not established from the log: {ex['problems']}")
    bm = ga.bridge_metrics(ticks, end)
    facts["bridge_metrics"] = bm
    if bm["abnormal_faults"]:
        problems.append(f"abnormal faults: {bm['abnormal_faults']}")
    if not bm["one_policy_episode"] or not bm["n_policy_ticks"]:
        problems.append("not exactly one continuous policy episode")
    if not bm["lowstate_fresh_all_policy"]:
        problems.append("stale LowState on a policy tick")
    if (bm["slew_clip_ticks_policy"] or 0) > CLIP_TICKS_MAX or (
        bm["limit_clip_ticks_policy"] or 0
    ) > CLIP_TICKS_MAX:
        problems.append(
            f"command clipping under policy (slew {bm['slew_clip_ticks_policy']}, "
            f"limit {bm['limit_clip_ticks_policy']} ticks; incumbent baseline is 0)"
        )
    tilt = bm["max_tilt_deg_policy"]
    if tilt is None or tilt > TILT_MAX_DEG:
        problems.append(f"max tilt {tilt} deg exceeds {TILT_MAX_DEG} (or is missing)")
    if bm["max_cmd_age_s"] is None or bm["max_cmd_age_s"] > 0.2:
        problems.append(
            f"policy command age {bm['max_cmd_age_s']} s missing or above the bridge's 0.2 s"
        )
    pq = run_dir / "policy.parquet"
    if not pq.is_file() or pq.stat().st_size == 0:
        problems.append("policy.parquet missing or empty (logging not active)")
    ans = run_dir / "operator_answer.txt"
    facts["operator_answer"] = ans.read_text().strip() if ans.is_file() else None
    if facts["operator_answer"] != "y":
        problems.append(
            f"operator did not confirm a clean run (answer {facts['operator_answer']!r})"
        )
    facts["evidence_files"] = [
        {"path": str((run_dir / n).relative_to(session)), "sha256": _sha256(run_dir / n)}
        for n in ("bridge.jsonl", "policy.parquet", "operator_answer.txt")
        if (run_dir / n).is_file()
    ]
    return problems, facts


def _write_record(session: Path, name: str, rec: dict[str, Any]) -> None:
    path = session / name
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{path} already exists; records are never overwritten")
    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")


def _base_record(
    schema: str, verdict: str, problems: list[str], run_dir: Path, sha_now: str
) -> dict:
    return {
        "schema": schema,
        "verdict": verdict,
        "problems": problems,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_sha": sha_now,
        "run_dir": run_dir.name,
        "rehearsal": _rehearsal(),
    }


def drill_eval(run_dir: Path, session: Path, sha_now: str) -> dict[str, Any]:
    """STOP/deadman drill: standing on the floor under the incumbent at a zero command, the
    operator releases the deadman 1-4 s after the policy takes over; the bridge must go to a
    zero-motion command within STOP_LATENCY_MAX_S of observing the release."""
    try:
        problems, facts = _run_checks(run_dir, session, sha_now, DRILL_AUTHORITY_S)
        ex = facts["policy_exit"]
        if ex["exit"] != "deadman_release":
            problems.append(f"policy exit is {ex['exit']!r}, the drill needs a deadman_release")
        lat, win = ex.get("stop_latency_s"), ex.get("release_window_s")
        if lat is None or lat > STOP_LATENCY_MAX_S:
            problems.append(f"STOP latency {lat} s not shown within {STOP_LATENCY_MAX_S} s")
        ep = ex.get("policy_episode_s")
        if ep is None or not DRILL_RELEASE_WINDOW_S[0] <= ep <= DRILL_RELEASE_WINDOW_S[1]:
            problems.append(
                f"release came {ep} s into the policy episode, expected {DRILL_RELEASE_WINDOW_S}"
            )
        if win is None:
            problems.append("release window not established")
    except Exception as exc:  # analyzer failure is a NO-GO, never a pass
        problems, facts = [f"analyzer failure: {type(exc).__name__}: {exc}"], {}
    verdict = "NO-GO" if problems else "GO"
    rec = _base_record(DRILL_SCHEMA, verdict, problems, run_dir, sha_now)
    ex = facts.get("policy_exit") or {}
    rec.update(
        live=bool(facts.get("live")),
        artifact_sha256=(
            {r: INCUMBENT[r] for r in ("checkpoint", "policy.onnx", "policy.onnx.data")}
            if not problems
            else {}
        ),
        stop_latency_s=ex.get("stop_latency_s"),
        evidence_files=facts.get("evidence_files", []),
    )
    _write_record(session, DRILL, rec)
    return rec


def walk_eval(run_dir: Path, session: Path, sha_now: str) -> dict[str, Any]:
    """The one vx03_2s run. Always writes a result; NO-GO on any problem or analyzer failure."""
    try:
        authority = float(profile_duration(PRESETS[FREE_FLOOR_PRESET])) + ga.WALK_AUTHORITY_MARGIN_S
        problems, facts = _run_checks(run_dir, session, sha_now, authority)
        ex = facts["policy_exit"]
        if ex["exit"] != "planned_end":
            problems.append(f"policy exit is {ex['exit']!r}, expected the planned authority end")
        summary = ga.walk_summary(run_dir)
        if summary is None:
            problems.append("walk_cmd.jsonl has no summary record")
        else:
            mx = tuple(summary.get("max_abs") or ())
            if summary.get("preset") != FREE_FLOOR_PRESET or summary.get("outcome") != "complete":
                problems.append(
                    f"walk_cmd preset/outcome {summary.get('preset')}/{summary.get('outcome')}"
                )
            if len(mx) != 3 or any(
                abs(a - b) > CMD_TOL for a, b in zip(mx, FREE_FLOOR_COMMAND, strict=True)
            ):
                problems.append(
                    f"commanded max |v| {mx} is not exactly {FREE_FLOOR_COMMAND} (no yaw, no vy)"
                )
            if not summary.get("n_published"):
                problems.append("walk_command published nothing")
        facts["walk_cmd"] = summary
        facts["evidence_files"] += (
            [
                {
                    "path": str((run_dir / "walk_cmd.jsonl").relative_to(session)),
                    "sha256": _sha256(run_dir / "walk_cmd.jsonl"),
                }
            ]
            if (run_dir / "walk_cmd.jsonl").is_file()
            else []
        )
    except Exception as exc:
        problems, facts = [f"analyzer failure: {type(exc).__name__}: {exc}"], {}
    verdict = "NO-GO" if problems else "GO"
    rec = _base_record(RESULT_SCHEMA, verdict, problems, run_dir, sha_now)
    rec.update(preset=FREE_FLOOR_PRESET, evidence_files=facts.get("evidence_files", []))
    rec["next"] = (
        "STOP. Human review is required before any further free-floor command; "
        "this tool authorises nothing longer than vx03_2s."
    )
    _write_record(session, RESULT, rec)
    return rec


def review(session: Path, decision: str, by: str, note: str) -> dict[str, Any]:
    """The human act that unblocks the next free-floor command. Accept needs a GO result."""
    if not (session / STARTED).is_file():
        raise ValueError("no free-floor run was started in this session; nothing to review")
    if decision == "accept":
        res, err = _read_json(session / RESULT)
        if err or not isinstance(res, dict) or res.get("verdict") != "GO":
            raise ValueError(
                "cannot accept: no GO result record (analysis missing, failed or NO-GO)"
            )
    if not by.strip() or not note.strip():
        raise ValueError("a reviewer name and a note are required")
    rec = {
        "schema": REVIEW_SCHEMA,
        "decision": decision,
        "by": by,
        "note": note,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_record(session, REVIEW, rec)
    return rec


# ---------------------------------------------------------------- CLI
def _sha_now() -> str:
    return resolve_code_identity(str(REPO)).sha or ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--phase", choices=("drill", "walk"), required=True)
    c.add_argument("--config", type=Path, required=True)
    c.add_argument("--lock", type=Path, required=True)
    c.add_argument("--session", type=Path, required=True)
    c.add_argument("--bundle", type=Path)
    c.add_argument("--preset", default=FREE_FLOOR_PRESET)
    for name in ("drill-eval", "walk-eval"):
        e = sub.add_parser(name)
        e.add_argument("--run-dir", type=Path, required=True)
        e.add_argument("--session", type=Path, required=True)
    m = sub.add_parser("mark-started")
    m.add_argument("--session", type=Path, required=True)
    m.add_argument("--run-dir", type=Path, required=True)
    r = sub.add_parser("review")
    r.add_argument("decision", choices=("accept", "reject"))
    r.add_argument("--session", type=Path, required=True)
    r.add_argument("--by", required=True)
    r.add_argument("--note", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "check":
        cfg = yaml.safe_load(args.config.read_text())
        problems = check(
            args.phase,
            cfg,
            args.config,
            args.lock,
            args.session,
            sha_now=_sha_now(),
            bundle=args.bundle,
            preset=args.preset,
        )
        for p in problems:
            print(f"[free-floor] BLOCKED: {p}", file=sys.stderr)
        print(f"[free-floor] {args.phase}: {'NO-GO' if problems else 'GO (software gates only)'}")
        return 1 if problems else 0
    if args.cmd == "mark-started":
        _write_record(
            args.session,
            STARTED,
            {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "run_dir": args.run_dir.name,
            },
        )
        return 0
    if args.cmd == "review":
        try:
            print(json.dumps(review(args.session, args.decision, args.by, args.note), indent=2))
        except (ValueError, FileExistsError) as exc:
            print(f"[free-floor] review refused: {exc}", file=sys.stderr)
            return 1
        return 0
    fn = drill_eval if args.cmd == "drill-eval" else walk_eval
    rec = fn(args.run_dir, args.session, _sha_now())
    print(f"[free-floor] {args.cmd}: {rec['verdict']}")
    for p in rec["problems"]:
        print(f"[free-floor]   {p}", file=sys.stderr)
    return 0 if rec["verdict"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
