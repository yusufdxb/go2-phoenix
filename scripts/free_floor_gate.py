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

    free_floor_gate.py check   --phase drill|walk --config C --lock L --session S [--bundle B] [--preset P]
    free_floor_gate.py begin   --kind drill|walk --run-dir R ...   final re-check + run-start record (BEFORE any motor)
    free_floor_gate.py snapshot --run-dir R --label start|end ...  re-hash what the config loads
    free_floor_gate.py drill-eval --run-dir R --session S ...      STOP/deadman drill while standing
    free_floor_gate.py walk-eval  --run-dir R --session S ...      the one vx03_2s run
    free_floor_gate.py measure --run-dir R --forward-cm F ...      the tape-measured displacement
    free_floor_gate.py review accept|reject --session S --by NAME --note TEXT
    free_floor_gate.py unreviewed | desk-ready | lab-commands

State machine (per session directory; all records are write-once):

    stage_A.json (--payload, this execution set)
      -> FF_drill_<ts>/ + free_floor_drill_FF_drill_<ts>.json   (up to 3 attempts; the NEWEST must be GO)
      -> free_floor_started.json                                 (written BEFORE motors; one per session)
      -> FF_walk_<ts>/ + free_floor_result.json                  (analyzer verdict; never overwritten)
      -> free_floor_review.json                                  (human; names the run and the result hash)

Until the review exists every walking launch is refused (this script, the harness W, and the
LowCmd bridge itself). After a review (accept OR reject) the session is closed to further
free-floor walks: another walk needs a NEW session directory, i.e. a fresh stage A --payload and
a fresh drill. Accept does not unlock yaw, a longer duration or any other command: only vx03_2s
exists. The records are accident-proofing, not tamper-proofing.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
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
from phoenix.sim2real import free_floor_state as st  # noqa: E402
from phoenix.sim2real import walk_grant as wg  # noqa: E402
from phoenix.sim2real.provenance import (  # noqa: E402
    CodeIdentity,
    resolve_code_identity,
    resolve_expected_branch,
)
from phoenix.sim2real.provenance import identity_problems as code_identity_problems  # noqa: E402
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
MODEL_ROLES = ("checkpoint", "policy.onnx", "policy.onnx.data")

# ---------------------------------------------------------------- commissioning limits
#: The ONLY walking command free-floor mode can ever request (0.3 m/s forward, 2 s, no yaw).
FREE_FLOOR_PRESET = "vx03_2s"
FREE_FLOOR_PRESETS = (FREE_FLOOR_PRESET,)
FREE_FLOOR_COMMAND = (0.3, 0.0, 0.0)
DRILL_AUTHORITY_S = 5.0
DRILL_RELEASE_WINDOW_S = (1.0, 4.0)  # the release must come inside the policy episode
#: Bridge-hop latency: the bridge READING the release (first non-policy tick) to its first
#: zero-motion hold tick, and the tick spacing around it. NOT the operator's STOP latency: the
#: deadman node waits --joy-timeout-s (0.5 s) of remote silence and publishes at 10 Hz first.
STOP_LATENCY_MAX_S = 0.1
#: End to end: the last remote message with L1 held -> the first zero-motion hold tick, measured by
#: scripts/free_floor_remote_trace.py on the bridge's clock. Derived from the deadman node's own
#: design (0.5 s timeout + 0.1 s publish period + 0.4 s slack), not from robot data: the first
#: live drill is what measures it.
E2E_STOP_MAX_S = 1.0
E2E_STOP_DESIGN_S = 0.6
DRILL_MAX_AGE_S = 4 * 3600  # backstop; the drill is also bound to the runtime identity below
MAX_DRILL_ATTEMPTS = 3
TILT_MAX_DEG = 6.0  # real incumbent stand/walk runs: <= 4.94 deg
CLIP_TICKS_MAX = 0  # real incumbent runs: 0 slew and 0 limit clip ticks under policy
BATTERY_FLOOR_PCT = 20.0
STANDUP_S = 2.0
CMD_TOL = 1e-9
STAGE_SCHEMA = "phoenix-preflight-stage/v1"
DRILL_SCHEMA = "phoenix-free-floor-drill/v1"
RESULT_SCHEMA = "phoenix-free-floor-result/v1"
ATTEMPT_SCHEMA = "phoenix-free-floor-attempt/v1"
STARTED, RESULT, REVIEW = st.STARTED, st.RESULT, st.REVIEW
ATTEMPT = "attempt.json"
MEASUREMENT = "operator_measurement.json"
REMOTE_TRACE = "remote_trace.jsonl"
ARTIFACTS_START, ARTIFACTS_END = "runtime_artifacts_start.json", "runtime_artifacts_end.json"
ALIVE_AT_END = "alive_at_end.json"
#: Processes that must still be running when the harness ends the attempt.
ALIVE_REQUIRED = ("deadman", "lowstate_bridge", "lowcmd_bridge")
DRILL_GLOB = "FF_drill_*"


def drill_record_name(run_name: str) -> str:
    return f"free_floor_drill_{run_name}.json"


def _rehearsal(env: Mapping[str, str] | None = None) -> bool:
    """True only for a localhost rehearsal (PHOENIX_REHEARSAL=1). A rehearsal invocation accepts
    only rehearsal records for the free-floor steps and a live one accepts only live records, so
    neither can stand in for the other."""
    return (os.environ if env is None else env).get("PHOENIX_REHEARSAL") == "1"


_sha256 = st.file_sha256
_read_json = st.read_json


def _utc_age_s(stamp: Any) -> float | None:
    try:
        t = datetime.strptime(str(stamp), "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        try:
            t = datetime.strptime(str(stamp), "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None
    return (datetime.now(timezone.utc) - t.replace(tzinfo=timezone.utc)).total_seconds()


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------- environment
def env_problems(env: Mapping[str, str] | None = None) -> list[str]:
    """Variables that could silently change what a free-floor run means.

    Nothing here selects a policy (identity is hash-pinned), but several change the run itself or
    make fake evidence count, and a stale value left in a lab shell is how that happens."""
    env = os.environ if env is None else env
    rehearsal = _rehearsal(env)
    out: list[str] = []

    def set_(name: str) -> bool:
        return env.get(name) not in (None, "", "0")

    if env.get("PHOENIX_WALK_AUTHORITY_MARGIN_S"):
        out.append(
            "PHOENIX_WALK_AUTHORITY_MARGIN_S is set: it is a gantry W knob, free-floor pins 4.0 s "
            "and its analyzer checks exactly that (unset it so the evidence is unambiguous)"
        )
    if env.get("PHOENIX_STANDUP_S"):
        try:
            same = float(env["PHOENIX_STANDUP_S"]) == STANDUP_S
        except ValueError:
            same = False
        if not same:
            out.append(
                f"PHOENIX_STANDUP_S={env['PHOENIX_STANDUP_S']!r} is not the pinned {STANDUP_S}"
            )
    if env.get("PHOENIX_MIN_BATTERY_SOC"):
        try:
            low = float(env["PHOENIX_MIN_BATTERY_SOC"]) < BATTERY_FLOOR_PCT
        except ValueError:
            low = True
        if low:
            out.append(
                f"PHOENIX_MIN_BATTERY_SOC={env['PHOENIX_MIN_BATTERY_SOC']!r} lowers the "
                f"{BATTERY_FLOOR_PCT:.0f} % battery floor"
            )
    if env.get("PHOENIX_DEADMAN", "wireless") not in ("wireless", "joy"):
        out.append(f"PHOENIX_DEADMAN={env.get('PHOENIX_DEADMAN')!r} is not wireless or joy")
    branch = env.get("PHOENIX_EXPECT_BRANCH")
    if branch and branch != resolve_expected_branch(env={}):
        out.append(f"PHOENIX_EXPECT_BRANCH={branch!r} overrides the branch hardware work must use")
    if not rehearsal:
        for name in (
            "PHOENIX_REHEARSAL",
            "PHOENIX_REHEARSAL_WALKED",
            "PHOENIX_CONFIRM",
            dc.HW_SESSIONS_ENV,
            dc.HW_SESSIONS_TEST_FLAG,
        ):
            if set_(name):
                out.append(
                    f"{name} is set in a live invocation (rehearsal/test variable): unset it"
                )
        if env.get("ROS_LOCALHOST_ONLY") == "1":
            out.append("ROS_LOCALHOST_ONLY=1: localhost-only DDS cannot reach the robot")
    elif env.get("ROS_LOCALHOST_ONLY") != "1":
        out.append(
            "a rehearsal needs ROS_LOCALHOST_ONLY=1 (it must never share a network with a GO2)"
        )
    return out


def session_problems(session: Path, root: Path | None = None) -> list[str]:
    """The session must be a real directory directly under logs/hw_sessions, where every later
    check (the review latch, the A/B campaign, the bridge) looks for unreviewed runs."""
    root = (dc.session_root() if root is None else Path(root)).resolve()
    if session.is_symlink():
        return [f"session {session} is a symlink"]
    if session.resolve().parent != root:
        return [
            f"session {session} is not a direct child of {root}: runs outside it are invisible to "
            "the unreviewed-run latch (unset PHOENIX_SESSION or point it inside)"
        ]
    return []


# ---------------------------------------------------------------- identity gate
def hash_identity_problems(hashes: Mapping[str, str | None], what: str) -> list[str]:
    """``hashes`` (role -> sha256) must be exactly model_2750; seed 50 and unknowns are named."""
    problems: list[str] = []
    for role in MODEL_ROLES:
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


def _config_paths(node: Any, key: str | None = None) -> list[Path]:
    """Every path-valued config key the deploy path may open (the activation PATH_KEYS)."""
    out: list[Path] = []
    if isinstance(node, Mapping):
        for k, v in node.items():
            out += _config_paths(v, str(k))
    elif isinstance(node, list):
        for v in node:
            out += _config_paths(v)
    elif key in dc.PATH_KEYS and isinstance(node, str):
        out.append(Path(node))
    return out


def containment_problems(cfg: Mapping[str, Any], cfg_path: Path, bundle: Path) -> list[str]:
    """The files the policy node will open must BE the bundle that was verified: inside it, not
    reached through a symlink. Otherwise PHOENIX_BUNDLE proves nothing about what loads."""
    if bundle.is_symlink():
        return [f"bundle {bundle} is a symlink"]
    root = bundle.resolve()
    problems: list[str] = []
    for p in [cfg_path, *dc.artifact_paths(cfg).values(), *_config_paths(cfg)]:
        if p.is_symlink():
            problems.append(f"{p} is a symlink (the verified bytes are not necessarily what loads)")
        try:
            p.resolve().relative_to(root)
        except ValueError:
            problems.append(f"{p} is outside the bundle {root}: PHOENIX_BUNDLE is not what loads")
    return sorted(set(problems))


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
        if got is not None and got != want and role not in MODEL_ROLES:
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
        problems += containment_problems(cfg, cfg_path, bundle)
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
    pt = bundle / "policy.pt"
    if pt.is_file() and _sha256(pt) != INCUMBENT["policy.pt"]:
        problems.append(f"bundle {bundle.name}: policy.pt is not model_2750's TorchScript export")
    return problems


def runtime_identity(
    cfg: Mapping[str, Any],
    cfg_path: Path,
    lock_path: Path,
    bundle: Path | None,
    session: Path,
    ident: CodeIdentity,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Everything whose change between the drill and the walk could make the drill stop being
    evidence for the walk: code, files, payload boot, deadman source, DDS setup. Drill and walk
    must see an EQUAL dict; the age limit is only a backstop."""
    env = os.environ if env is None else env

    def maybe(path: Path | None) -> str | None:
        return _sha256(path) if path is not None and path.is_file() else None

    boot = None
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        pass
    dds = env.get("CYCLONEDDS_URI", "")
    arts = dc.observed_artifact_hashes(cfg, cfg_path)["artifacts"]
    return {
        "code_sha": ident.sha,
        "execution_set_sha256": ident.execution_set_sha256,
        "config_file_sha256": maybe(cfg_path),
        "lock_file_sha256": maybe(lock_path),
        "bundle_sums_sha256": maybe(bundle / "SHA256SUMS") if bundle else None,
        "stage_a_sha256": maybe(session / "stage_A.json"),
        "artifacts": {r: (arts.get(r) or {}).get("sha256") for r in sorted(arts)},
        "boot_id": boot,
        "deadman": env.get("PHOENIX_DEADMAN", "wireless"),
        "operator_remote": env.get("PHOENIX_OPERATOR_REMOTE", "0"),
        "rmw": env.get("RMW_IMPLEMENTATION"),
        "ros_localhost_only": env.get("ROS_LOCALHOST_ONLY", "0"),
        "cyclonedds_file_sha256": (
            maybe(Path(dds[len("file://") :])) if dds.startswith("file://") else None
        ),
        "standup_s": STANDUP_S,
    }


def identity_diff(then: Mapping[str, Any], now: Mapping[str, Any], what: str) -> list[str]:
    keys = sorted(set(then) | set(now))
    return [
        f"{what}: runtime identity changed: {k} {str(then.get(k))[:20]!r} -> {str(now.get(k))[:20]!r}"
        for k in keys
        if then.get(k) != now.get(k)
    ]


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


def _stage_a_problems(session: Path, ident: CodeIdentity, require_payload: bool) -> list[str]:
    rec, err = _read_json(session / "stage_A.json")
    if err or not isinstance(rec, dict):
        return [f"stage A record missing or unreadable in {session}: {err}"]
    rid = rec.get("code_identity") or {}
    problems = []
    if rec.get("schema") != STAGE_SCHEMA or rec.get("stage") != "A" or rec.get("verdict") != "GO":
        problems.append("stage A is not a GO record")
    if bool(rec.get("rehearsal")) != _rehearsal():
        problems.append(
            "stage A rehearsal flag does not match this invocation (rehearsal evidence never counts live)"
        )
    if rid.get("sha") != ident.sha or rec.get("code_identity_problems") != []:
        problems.append("stage A is not for the code now running (or has code identity problems)")
    elif not rid.get("execution_set_sha256") or rid.get("execution_set_sha256") != (
        ident.execution_set_sha256
    ):
        problems.append(
            "stage A does not cover the files now running (execution set digest missing or "
            "different): re-run stage A --payload on this exact tree"
        )
    if require_payload and rec.get("mode") != "payload":
        problems.append(f"stage A mode is {rec.get('mode')!r}, need the on-robot --payload run")
    problems += hash_identity_problems(rec.get("lock_artifact_sha256") or {}, "stage A lock")
    return problems


def unreviewed_runs(dirs: Sequence[Path]) -> list[str]:
    """Free-floor walks started and not validly reviewed. They block every later walking launch."""
    return st.unreviewed_runs(dirs)


def drill_runs(session: Path) -> list[Path]:
    return sorted(d for d in session.glob(DRILL_GLOB) if d.is_dir())


def drill_problems(
    session: Path, ident: CodeIdentity | str, runtime: Mapping[str, Any] | None = None
) -> list[str]:
    """The NEWEST drill attempt in this session must be a GO, on this code and runtime identity."""
    sha_now = ident if isinstance(ident, str) else ident.sha
    runs = drill_runs(session)
    if not runs:
        return [f"no STOP/deadman drill run in {session}"]
    newest = runs[-1].name
    rec, err = _read_json(session / drill_record_name(newest))
    if err or not isinstance(rec, dict):
        return [
            f"the latest drill attempt {newest} has no analysed record ({err}): an interrupted or "
            "unanalysed drill is not evidence; run a new drill"
        ]
    problems = []
    if rec.get("schema") != DRILL_SCHEMA or rec.get("verdict") != "GO":
        problems.append(f"the latest drill ({newest}) is not a GO record")
    if rec.get("run_dir") != newest:
        problems.append("drill record names a different run than the newest drill directory")
    if bool(rec.get("rehearsal")) != _rehearsal() or rec.get("live") is not True:
        problems.append("drill record is not a live run of this kind (rehearsal flag mismatch)")
    if rec.get("code_sha") != sha_now:
        problems.append("drill was run on different code than is running now")
    problems += hash_identity_problems(rec.get("artifact_sha256") or {}, "drill")
    age = _utc_age_s(rec.get("utc"))
    if age is None or age < 0 or age > DRILL_MAX_AGE_S:
        problems.append(f"drill is stale or its timestamp is invalid (age {age})")
    if runtime is not None:
        problems += identity_diff(rec.get("runtime_identity") or {}, runtime, "drill vs now")
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
    sha_now: str | None = None,
    ident: CodeIdentity | None = None,
    bundle: Path | None = None,
    preset: str | None = None,
    require_payload: bool = True,
    env: Mapping[str, str] | None = None,
    session_root: Path | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> list[str]:
    """Every reason the ``phase`` ("drill" or "walk") must not run. Empty means allowed."""
    if phase not in ("drill", "walk"):
        return [f"unknown phase {phase!r}"]
    if phase == "walk" and preset not in FREE_FLOOR_PRESETS:
        return [
            f"free-floor allows only {list(FREE_FLOOR_PRESETS)}, got {preset!r}: "
            "no yaw, no longer or faster command without a new human-reviewed change"
        ]
    if ident is None:
        ident = resolve_code_identity(REPO)
    if sha_now is not None and sha_now != ident.sha:
        ident = CodeIdentity(**{**ident.__dict__, "sha": sha_now})
    problems = env_problems(env)
    problems += session_problems(session, session_root)
    problems += [
        f"code now: {p}"
        for p in code_identity_problems(
            ident, expected_branch=resolve_expected_branch(env=env or {})
        )
    ]
    problems += identity_problems(cfg, cfg_path, lock_path, bundle)
    dirs = [Path(d) for d in (dc.default_session_dirs() if sessions is None else sessions)]
    for stage, motors in (("C", "off"), ("F", "live"), ("G", "live"), ("H", "live")):
        ok, why = _stage_ok(stage, cfg, dirs, motors=motors)
        if not ok:
            problems += why
    problems += _stage_a_problems(session, ident, require_payload)
    problems += unreviewed_runs(dirs)
    if phase == "drill" and len(drill_runs(session)) >= MAX_DRILL_ATTEMPTS:
        problems.append(
            f"{MAX_DRILL_ATTEMPTS} drill attempts already in this session: stop, read the records, "
            "and start a new session (fresh stage A) if you still want to proceed"
        )
    if phase == "walk":
        if (session / STARTED).exists():
            problems.append(
                "a free-floor walk was already started in this session; after its review a new "
                "session (new PHOENIX_SESSION, fresh stage A --payload and drill) is required"
            )
        problems += drill_problems(session, ident, runtime)
    return problems


# ---------------------------------------------------------------- run evaluation
def _load_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def e2e_stop(trace: Sequence[Mapping[str, Any]], ex: Mapping[str, Any]) -> dict[str, Any]:
    """Last remote message with L1 held -> the first zero-motion hold tick (bridge clock)."""
    t_zero, t_obs = ex.get("t_zero_ns"), ex.get("t_release_obs_ns")
    out: dict[str, Any] = {"measured": False}
    if t_zero is None or t_obs is None:
        out["why"] = "bridge zero-command tick not established"
        return out
    held = [
        r["t_mono_ns"] for r in trace if r.get("topic") == "wireless" and r.get("value") is True
    ]
    held = [t for t in held if t <= t_obs]
    est = [
        r["t_mono_ns"]
        for r in trace
        if r.get("topic") == "estop" and r.get("value") is True and r["t_mono_ns"] <= t_obs
    ]
    if not held:
        out["why"] = "no remote message with L1 held before the release in the trace"
        return out
    out.update(
        measured=True,
        remote_last_held_to_zero_s=(t_zero - held[-1]) / 1e9,
        remote_last_held_to_bridge_read_s=(t_obs - held[-1]) / 1e9,
        estop_true_to_bridge_read_s=((t_obs - est[0]) / 1e9) if est else None,
        design_expectation_s=E2E_STOP_DESIGN_S,
        resolution_s=0.02,
        resolution_note="remote message period (50 Hz); the human finger and the radio hop to the "
        "robot are not visible",
    )
    return out


def estimated_pd_torque(ticks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Peak |kp (target - q) - kd dq| per tick under policy, from logged bridge fields.

    An ESTIMATE of the commanded PD torque (no motor torque or current is logged); it is not a
    measured joint torque and ignores the motor's own limits."""
    peak, n = 0.0, 0
    for t in ticks:
        if t.get("mode") != "policy":
            continue
        q, dq, tgt = t.get("q_unitree"), t.get("dq_unitree"), t.get("final_target_unitree")
        kp, kd = t.get("kp_eff"), t.get("kd_eff")
        if not (q and dq and tgt and isinstance(kp, (int, float)) and isinstance(kd, (int, float))):
            continue
        n += 1
        peak = max(
            peak, max(abs(kp * (a - b) - kd * c) for a, b, c in zip(tgt, q, dq, strict=True))
        )
    return {"est_peak_pd_torque_nm": peak if n else None, "ticks": n, "estimate_only": True}


def _run_checks(
    run_dir: Path,
    session: Path,
    sha_now: str,
    authority_s: float,
    runtime: Mapping[str, Any],
    kind: str,
) -> tuple[list[str], dict[str, Any]]:
    """Facts common to the drill and the walk, from the run's evidence. Returns (problems, facts)."""
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
        {r: (art.get(r) or {}).get("sha256") for r in MODEL_ROLES}, "bridge manifest"
    )
    if not (gp.get("walking_allowed") and gp.get("deadman_required") and gp.get("require_imu")):
        problems.append("bridge did not run with walking_allowed, deadman_required and require_imu")
    if gp.get("kp") != INCUMBENT_GAINS["kp"] or gp.get("kd") != INCUMBENT_GAINS["kd"]:
        problems.append(f"bridge gains kp={gp.get('kp')} kd={gp.get('kd')} are not the incumbent's")
    # What the policy node loaded: the harness re-hashes the files the config names once the
    # policy is in control and again before the processes stop (the node's own hashes are only in
    # its log), so this is the recorded evidence that those bytes did not change under it.
    for name in (ARTIFACTS_START, ARTIFACTS_END):
        rec, err = _read_json(run_dir / name)
        if err or not isinstance(rec, dict):
            problems.append(f"runtime artifact snapshot {name} missing or unreadable: {err}")
        else:
            problems += hash_identity_problems(rec.get("artifacts") or {}, name)
            for role in ("policy.manifest", "policy.pt"):
                got = (rec.get("artifacts") or {}).get(role)
                if got is not None and got != INCUMBENT[role]:
                    problems.append(f"{name}: {role} {got[:12]} is not model_2750's")
    attempt, err = _read_json(run_dir / ATTEMPT)
    if err or not isinstance(attempt, dict):
        problems.append(f"run-start record {ATTEMPT} missing or unreadable: {err}")
    else:
        if attempt.get("schema") != ATTEMPT_SCHEMA or attempt.get("kind") != kind:
            problems.append(f"run-start record is not a {kind} attempt record")
        if attempt.get("run_dir") != run_dir.name:
            problems.append("run-start record names a different run directory")
        if bool(attempt.get("rehearsal")) != _rehearsal():
            problems.append("run-start record rehearsal flag mismatch")
        problems += identity_diff(
            attempt.get("runtime_identity") or {}, runtime, "run start vs end"
        )
    alive, err = _read_json(run_dir / ALIVE_AT_END)
    if err or not isinstance(alive, dict):
        problems.append(f"{ALIVE_AT_END} missing: {err}")
    else:
        dead = [n for n in ALIVE_REQUIRED if alive.get(n) is not True]
        if dead:
            problems.append(f"process(es) not running when the attempt ended: {dead}")
    for log in ("deadman", "lowstate_bridge", "lowcmd_bridge", "policy_node"):
        if not (run_dir / f"{log}.log").is_file():
            problems.append(f"{log}.log missing (process log not captured)")
    if (end or {}).get("reason") != "sigint":
        problems.append(
            f"bridge end record reason is {(end or {}).get('reason')!r}, expected a clean sigint stop"
        )
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
    facts["pd_torque"] = estimated_pd_torque(ticks)
    names = (
        "bridge.jsonl",
        "policy.parquet",
        "operator_answer.txt",
        ATTEMPT,
        ARTIFACTS_START,
        ARTIFACTS_END,
        ALIVE_AT_END,
        REMOTE_TRACE,
        MEASUREMENT,
    )
    facts["evidence_files"] = [
        {"path": str((run_dir / n).relative_to(session)), "sha256": _sha256(run_dir / n)}
        for n in names
        if (run_dir / n).is_file()
    ]
    return problems, facts


def _write_record(session: Path, name: str, rec: dict[str, Any]) -> None:
    path = session / name
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{path} already exists; records are never overwritten")
    path.write_text(json.dumps(rec, indent=2, sort_keys=True, default=str) + "\n")


def _base_record(
    schema: str, verdict: str, problems: list[str], run_dir: Path, sha_now: str
) -> dict:
    return {
        "schema": schema,
        "verdict": verdict,
        "problems": problems,
        "utc": _utc(),
        "code_sha": sha_now,
        "run_dir": run_dir.name,
        "rehearsal": _rehearsal(),
    }


def drill_eval(
    run_dir: Path, session: Path, sha_now: str, runtime: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """STOP/deadman drill: standing on the floor under the incumbent at a zero command, the
    operator releases the deadman 1-4 s after the policy takes over; the bridge must go to a
    zero-motion command within STOP_LATENCY_MAX_S of reading the release, and the whole path
    (last remote message with L1 held -> zero command) within E2E_STOP_MAX_S."""
    facts: dict[str, Any] = {}
    try:
        problems, facts = _run_checks(
            run_dir, session, sha_now, DRILL_AUTHORITY_S, runtime or {}, "drill"
        )
        if runtime is None:
            problems.append("runtime identity not supplied (cannot bind the drill)")
        ex = facts["policy_exit"]
        if ex["exit"] != "deadman_release":
            problems.append(f"policy exit is {ex['exit']!r}, the drill needs a deadman_release")
        lat, win = ex.get("stop_latency_s"), ex.get("release_window_s")
        rx0 = ex.get("release_rx_to_zero_s")
        if lat is None or lat > STOP_LATENCY_MAX_S:
            problems.append(f"STOP latency {lat} s not shown within {STOP_LATENCY_MAX_S} s")
        if rx0 is None or rx0 > STOP_LATENCY_MAX_S:
            problems.append(
                f"bridge receipt-to-zero {rx0} s not shown within {STOP_LATENCY_MAX_S} s"
            )
        if win is None or win > STOP_LATENCY_MAX_S:
            problems.append(
                f"release window {win} s (last policy tick -> release read) is not within "
                f"{STOP_LATENCY_MAX_S} s: the bridge log is too coarse around the release"
            )
        dt = (facts["bridge_metrics"].get("bridge_dt_s") or {}).get("max")
        if dt is None or dt > STOP_LATENCY_MAX_S:
            problems.append(f"bridge tick spacing max {dt} s: STOP timing is not resolvable")
        ep = ex.get("policy_episode_s")
        if ep is None or not DRILL_RELEASE_WINDOW_S[0] <= ep <= DRILL_RELEASE_WINDOW_S[1]:
            problems.append(
                f"release came {ep} s into the policy episode, expected {DRILL_RELEASE_WINDOW_S}"
            )
        trace_path = run_dir / REMOTE_TRACE
        if not trace_path.is_file():
            problems.append(f"{REMOTE_TRACE} missing: end-to-end STOP path not measured")
            facts["e2e_stop"] = {"measured": False, "why": "no trace"}
        else:
            facts["e2e_stop"] = e2e_stop(_load_trace(trace_path), ex)
            e2e = facts["e2e_stop"].get("remote_last_held_to_zero_s")
            if e2e is None or not 0.0 <= e2e <= E2E_STOP_MAX_S:
                problems.append(
                    f"end-to-end STOP (last remote message held -> zero command) {e2e} s is not "
                    f"within {E2E_STOP_MAX_S} s ({facts['e2e_stop'].get('why', 'measured')})"
                )
    except Exception as exc:  # analyzer failure is a NO-GO, never a pass
        problems, facts = [f"analyzer failure: {type(exc).__name__}: {exc}"], {}
    verdict = "NO-GO" if problems else "GO"
    rec = _base_record(DRILL_SCHEMA, verdict, problems, run_dir, sha_now)
    ex = facts.get("policy_exit") or {}
    rec.update(
        live=bool(facts.get("live")),
        artifact_sha256=({r: INCUMBENT[r] for r in MODEL_ROLES} if not problems else {}),
        runtime_identity=dict(runtime or {}),
        stop_latency_s=ex.get("stop_latency_s"),
        release_rx_to_zero_s=ex.get("release_rx_to_zero_s"),
        release_window_s=ex.get("release_window_s"),
        e2e_stop=facts.get("e2e_stop"),
        evidence_files=facts.get("evidence_files", []),
    )
    _write_record(session, drill_record_name(run_dir.name), rec)
    return rec


def _walk_report(
    run_dir: Path, facts: Mapping[str, Any], summary: Mapping[str, Any] | None, problems: list[str]
) -> dict[str, Any]:
    """The concise commissioning result. Everything here is read from evidence, never judged."""
    bm = facts.get("bridge_metrics") or {}
    ex = facts.get("policy_exit") or {}
    rep: dict[str, Any] = {
        "valid": not problems,
        "safety_event": bool(bm.get("abnormal_faults")) or ex.get("exit") != "planned_end",
        "end_condition": ex.get("exit"),
        "policy_episode_s": ex.get("policy_episode_s"),
        "authority_window_s": ex.get("authority_s"),
        "commanded_max_abs_vx_vy_wz": (summary or {}).get("max_abs"),
        "command_published_n": (summary or {}).get("n_published"),
        "max_tilt_deg": bm.get("max_tilt_deg_policy"),
        "slew_clip_ticks": bm.get("slew_clip_ticks_policy"),
        "limit_clip_ticks": bm.get("limit_clip_ticks_policy"),
        "faults": bm.get("faults"),
        "abnormal_faults": bm.get("abnormal_faults"),
        "lowstate_fresh_all_policy": bm.get("lowstate_fresh_all_policy"),
        "max_lowstate_age_s": bm.get("max_lowstate_age_s"),
        "max_cmd_age_s": bm.get("max_cmd_age_s"),
        "bridge_loop_dt_s": bm.get("bridge_dt_s"),
        "min_torque_margin_nm": bm.get("min_torque_margin_nm"),
        "estimated_pd_torque": facts.get("pd_torque"),
        "odometry": {
            "reliable": False,
            "note": "GO2 odometry is NOT recorded as valid by this pipeline (odom_valid is false on "
            "every row of the gantry walks); displacement comes ONLY from the tape measurement",
        },
    }
    tape, err = _read_json(run_dir / MEASUREMENT)
    rep["operator_measurement"] = (
        tape if isinstance(tape, dict) else {"measured": False, "why": err or "not entered"}
    )
    try:
        pol = ga.load_policy(run_dir)
        if pol is not None:
            tape = tape if isinstance(tape, dict) else {}
            fwd, lat = tape.get("forward_cm"), tape.get("lateral_cm")
            rep["policy_log"] = ga.walk_metrics(
                pol,
                FREE_FLOOR_PRESET,
                {
                    "tape_fwd_m": fwd / 100.0 if isinstance(fwd, (int, float)) else None,
                    "tape_lat_m": lat / 100.0 if isinstance(lat, (int, float)) else None,
                },
            )
    except Exception as exc:  # the report never turns a verdict into a crash
        rep["policy_log"] = {"error": f"{type(exc).__name__}: {exc}"}
    return rep


def walk_eval(
    run_dir: Path, session: Path, sha_now: str, runtime: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The one vx03_2s run. Always writes a result; NO-GO on any problem or analyzer failure."""
    facts: dict[str, Any] = {}
    summary = None
    try:
        authority = float(profile_duration(PRESETS[FREE_FLOOR_PRESET])) + ga.WALK_AUTHORITY_MARGIN_S
        problems, facts = _run_checks(run_dir, session, sha_now, authority, runtime or {}, "walk")
        if runtime is None:
            problems.append("runtime identity not supplied (cannot bind the walk)")
        started, err = _read_json(session / STARTED)
        if err or not isinstance(started, dict) or started.get("run_dir") != run_dir.name:
            problems.append("this run is not the one recorded in free_floor_started.json")
        else:
            problems += identity_diff(started.get("runtime_identity") or {}, runtime or {}, "start")
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
        if (run_dir / "walk_cmd.jsonl").is_file():
            facts["evidence_files"] = facts.get("evidence_files", []) + [
                {
                    "path": str((run_dir / "walk_cmd.jsonl").relative_to(session)),
                    "sha256": _sha256(run_dir / "walk_cmd.jsonl"),
                }
            ]
    except Exception as exc:
        problems, facts = [f"analyzer failure: {type(exc).__name__}: {exc}"], {}
    verdict = "NO-GO" if problems else "GO"
    rec = _base_record(RESULT_SCHEMA, verdict, problems, run_dir, sha_now)
    rec.update(
        preset=FREE_FLOOR_PRESET,
        runtime_identity=dict(runtime or {}),
        evidence_files=facts.get("evidence_files", []),
        report=_walk_report(run_dir, facts, summary, problems) if facts else {"valid": False},
    )
    rec["next"] = (
        "STOP. Human review is required before any further free-floor command; "
        "this tool authorises nothing longer than vx03_2s."
    )
    _write_record(session, RESULT, rec)
    return rec


def measure(
    run_dir: Path,
    forward_cm: float,
    lateral_cm: float,
    heading_deg: float | None,
    by: str,
    note: str,
) -> dict[str, Any]:
    """Structured tape measurement of the walk: the only displacement source."""
    if not by.strip():
        raise ValueError("a name is required for the measurement")
    for label, v in (("forward_cm", forward_cm), ("lateral_cm", lateral_cm)):
        if not -500.0 <= v <= 500.0:
            raise ValueError(f"{label} {v} is outside +-500 cm: check the units")
    rec = {
        "measured": True,
        "forward_cm": forward_cm,
        "lateral_cm": lateral_cm,
        "heading_deg": heading_deg,
        "by": by,
        "note": note,
        "utc": _utc(),
        "method": "tape between the floor marks under the robot's rear feet at start and at the "
        "end (both rear feet averaged), forward +, left +",
    }
    path = run_dir / MEASUREMENT
    if path.exists():
        raise FileExistsError(f"{path} already exists; measurements are never overwritten")
    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    return rec


def review(session: Path, decision: str, by: str, note: str) -> dict[str, Any]:
    """The human act that closes a free-floor walk. Accept needs a GO result for THAT run."""
    started, err = _read_json(session / STARTED)
    if err or not isinstance(started, dict):
        raise ValueError("no free-floor run was started in this session; nothing to review")
    if not by.strip() or not note.strip():
        raise ValueError("a reviewer name and a note are required")
    result_sha = None
    if (session / RESULT).is_file():
        res, rerr = _read_json(session / RESULT)
        result_sha = _sha256(session / RESULT)
        if decision == "accept":
            if rerr or not isinstance(res, dict) or res.get("verdict") != "GO":
                raise ValueError(
                    "cannot accept: no GO result record (analysis missing, failed or NO-GO)"
                )
            if res.get("run_dir") != started.get("run_dir"):
                raise ValueError(
                    "cannot accept: the result is for a different run than the one started"
                )
            if bool(res.get("rehearsal")) != _rehearsal():
                raise ValueError("cannot accept: rehearsal flag mismatch")
            bad = dc._evidence_file_rejections(res, session)
            if bad:
                raise ValueError(f"cannot accept: evidence no longer matches the result: {bad}")
    elif decision == "accept":
        raise ValueError("cannot accept: no GO result record (analysis missing, failed or NO-GO)")
    rec = {
        "schema": st.REVIEW_SCHEMA,
        "decision": decision,
        "by": by,
        "note": note,
        "utc": _utc(),
        "run_dir": started.get("run_dir"),
        "result_sha256": result_sha,
        "code_sha": started.get("code_sha"),
        "rehearsal": _rehearsal(),
        "consequence": "this session is closed to further free-floor walks; unlocks nothing "
        "longer than vx03_2s; a new walk needs a new session (fresh stage A --payload and drill)",
    }
    _write_record(session, REVIEW, rec)
    return rec


# ---------------------------------------------------------------- run start / snapshots
def begin(
    kind: str,
    cfg: Mapping[str, Any],
    cfg_path: Path,
    lock_path: Path,
    session: Path,
    run_dir: Path,
    *,
    bundle: Path | None,
    preset: str | None,
    ident: CodeIdentity | None = None,
    env: Mapping[str, str] | None = None,
    sessions: Sequence[Path] | None = None,
    session_root: Path | None = None,
) -> list[str]:
    """Final re-check, then the run-start record. The harness calls this after the operator's
    arm confirmation and BEFORE the first process that could move a motor, so a crash anywhere
    afterwards still leaves the walk recorded as started (and unreviewed)."""
    if kind not in ("drill", "walk"):
        return [f"unknown kind {kind!r}"]
    if bundle is None:
        return ["a live free-floor run needs the activated bundle (PHOENIX_BUNDLE)"]
    ident = resolve_code_identity(REPO) if ident is None else ident
    # One walking authority at a time: a live gantry (W) grant blocks this run BEFORE the
    # run-start record is written, so a refusal here never leaves a walk latched as started.
    live = wg.other_live_grants(
        session_root if session_root is not None else dc.session_root(), run_dir, time.time()
    )
    if live:
        return [f"another walking grant is live: {p}" for p in live]
    runtime = runtime_identity(cfg, cfg_path, lock_path, bundle, session, ident, env)
    problems = check(
        kind,
        cfg,
        cfg_path,
        lock_path,
        session,
        sessions=sessions,
        ident=ident,
        bundle=bundle,
        preset=preset,
        env=env,
        session_root=session_root,
        runtime=runtime,
    )
    if problems:
        return problems
    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "kind": kind,
        "run_dir": run_dir.name,
        "utc": _utc(),
        "rehearsal": _rehearsal(env),
        "preset": preset,
        "runtime_identity": runtime,
        "env": {
            k: v
            for k, v in sorted((os.environ if env is None else env).items())
            if k.startswith(("PHOENIX_", "ROS_", "RMW_", "CYCLONEDDS"))
        },
        "host": socket.gethostname(),
    }
    if kind == "walk":
        _write_record(
            session,
            STARTED,
            {
                "schema": st.STARTED_SCHEMA,
                "utc": attempt["utc"],
                "run_dir": run_dir.name,
                "code_sha": ident.sha,
                "preset": preset,
                "runtime_identity": runtime,
                "rehearsal": attempt["rehearsal"],
            },
        )
    _write_record(run_dir, ATTEMPT, attempt)
    return []


def banner(
    kind: str,
    cfg: Mapping[str, Any],
    cfg_path: Path,
    lock_path: Path,
    session: Path,
    bundle: Path | None,
    ident: CodeIdentity,
) -> str:
    """What the operator is about to run, in one block, before any typed confirmation."""
    arts = dc.observed_artifact_hashes(cfg, cfg_path)["artifacts"]
    short = lambda r: str((arts.get(r) or {}).get("sha256"))[:8]  # noqa: E731
    runs = drill_runs(session)
    drill = "none in this session"
    if runs:
        rec, _ = _read_json(session / drill_record_name(runs[-1].name))
        if isinstance(rec, dict):
            e2e = (rec.get("e2e_stop") or {}).get("remote_last_held_to_zero_s")
            drill = (
                f"{runs[-1].name}: {rec.get('verdict')}, age "
                f"{(_utc_age_s(rec.get('utc')) or 0) / 60:.0f} min, STOP end-to-end "
                f"{'n/a' if e2e is None else f'{e2e:.2f} s'}"
            )
        else:
            drill = f"{runs[-1].name}: NOT analysed"
    started, _ = _read_json(session / STARTED)
    nxt = {
        "drill": "after the drill: read free_floor_drill_*.json, then FREE_FLOOR gate, then the walk",
        "walk": "after the walk: measure with the tape, read the result, FREE_FLOOR review accept|reject",
    }[kind]
    sums = (bundle / "SHA256SUMS") if bundle else None
    lines = [
        f"FREE_FLOOR {kind}: {'REHEARSAL (no robot)' if _rehearsal() else 'LIVE'}",
        f"  code      : {str(ident.sha)[:12]} on {ident.branch} "
        f"({'DIRTY' if ident.dirty else 'clean'}; {str(ident.execution_set_sha256)[:24]})",
        f"  policy    : model_2750  checkpoint {short('checkpoint')}  onnx {short('policy.onnx')}  "
        f"data {short('policy.onnx.data')}  (hash-checked: bundle, lock, config, bridge, snapshots)",
        f"  bundle    : {bundle} (SHA256SUMS {_sha256(sums)[:12] if sums and sums.is_file() else 'MISSING'})",
        f"  session   : {session}",
        f"  command   : {'zero velocity, deadman release drill' if kind == 'drill' else 'vx 0.3 m/s forward, 2 s, no yaw, no lateral (the only command that exists)'}",
        f"  last drill: {drill}",
        f"  walk state: {'STARTED ' + str(started.get('run_dir')) if isinstance(started, dict) else 'none started in this session'}",
        f"  next      : {nxt}",
    ]
    return "\n".join(lines)


def snapshot(cfg: Mapping[str, Any], cfg_path: Path, run_dir: Path, label: str) -> dict[str, Any]:
    """Hash the files the config names, right now, into the run directory."""
    name = {"start": ARTIFACTS_START, "end": ARTIFACTS_END}[label]
    arts = dc.observed_artifact_hashes(cfg, cfg_path)
    rec = {
        "label": label,
        "utc": _utc(),
        "config_sha256": arts["deploy_config"]["sha256"],
        "artifacts": {r: (a or {}).get("sha256") for r, a in arts["artifacts"].items()},
        "resolved": {r: (a or {}).get("resolved") for r, a in arts["artifacts"].items()},
    }
    _write_record(run_dir, name, rec)
    return rec


# ---------------------------------------------------------------- CLI
def _common(sp: argparse.ArgumentParser, *, run: bool = True) -> None:
    sp.add_argument("--config", type=Path, required=True)
    sp.add_argument("--lock", type=Path, required=True)
    sp.add_argument("--session", type=Path, required=True)
    sp.add_argument("--bundle", type=Path)
    if run:
        sp.add_argument("--run-dir", type=Path, required=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--phase", choices=("drill", "walk"), required=True)
    _common(c, run=False)
    c.add_argument("--preset", default=FREE_FLOOR_PRESET)
    b = sub.add_parser("begin")
    b.add_argument("--kind", choices=("drill", "walk"), required=True)
    _common(b)
    b.add_argument("--preset")
    bn = sub.add_parser("banner")
    bn.add_argument("--kind", choices=("drill", "walk"), required=True)
    _common(bn, run=False)
    s = sub.add_parser("snapshot")
    s.add_argument("--label", choices=("start", "end"), required=True)
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--run-dir", type=Path, required=True)
    for name in ("drill-eval", "walk-eval"):
        _common(sub.add_parser(name))
    m = sub.add_parser("measure")
    m.add_argument("--run-dir", type=Path, required=True)
    m.add_argument("--forward-cm", type=float, required=True)
    m.add_argument("--lateral-cm", type=float, default=0.0)
    m.add_argument("--heading-deg", type=float)
    m.add_argument("--by", required=True)
    m.add_argument("--note", default="")
    r = sub.add_parser("review")
    r.add_argument("decision", choices=("accept", "reject"))
    r.add_argument("--session", type=Path, required=True)
    r.add_argument("--by", required=True)
    r.add_argument("--note", required=True)
    sub.add_parser("unreviewed")
    sub.add_parser("desk-ready").add_argument("--json", action="store_true")
    sub.add_parser("lab-commands").add_argument("--orin", default="orin")
    args = ap.parse_args(argv)

    if args.cmd in ("desk-ready", "lab-commands"):
        import free_floor_desk as desk  # noqa: E402  (read-only report, kept out of the gate)

        return desk.main_desk(args)
    if args.cmd == "unreviewed":
        why = unreviewed_runs(dc.default_session_dirs())
        for w in why:
            print(f"[free-floor] UNREVIEWED: {w}", file=sys.stderr)
        print(f"[free-floor] {'BLOCKED' if why else 'no unreviewed free-floor walk'}")
        return 1 if why else 0
    if args.cmd == "review":
        try:
            print(json.dumps(review(args.session, args.decision, args.by, args.note), indent=2))
        except (ValueError, FileExistsError) as exc:
            print(f"[free-floor] review refused: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.cmd == "measure":
        try:
            rec = measure(
                args.run_dir,
                args.forward_cm,
                args.lateral_cm,
                args.heading_deg,
                args.by,
                args.note,
            )
            print(json.dumps(rec, indent=2))
        except (ValueError, FileExistsError) as exc:
            print(f"[free-floor] measurement refused: {exc}", file=sys.stderr)
            return 1
        return 0
    cfg = yaml.safe_load(args.config.read_text())
    if args.cmd == "snapshot":
        snapshot(cfg, args.config, args.run_dir, args.label)
        return 0
    ident = resolve_code_identity(REPO)
    if args.cmd == "banner":
        print(banner(args.kind, cfg, args.config, args.lock, args.session, args.bundle, ident))
        return 0
    if args.cmd == "check":
        runtime = (
            runtime_identity(cfg, args.config, args.lock, args.bundle, args.session, ident)
            if args.phase == "walk" and args.bundle
            else None
        )
        problems = check(
            args.phase,
            cfg,
            args.config,
            args.lock,
            args.session,
            ident=ident,
            bundle=args.bundle,
            preset=args.preset,
            runtime=runtime,
        )
        for p in problems:
            print(f"[free-floor] BLOCKED: {p}", file=sys.stderr)
        bundle_note = "" if args.bundle else " (bundle NOT checked: no PHOENIX_BUNDLE)"
        print(
            f"[free-floor] {args.phase}: {'NO-GO' if problems else 'GO (software gates only)'}{bundle_note}"
        )
        return 1 if problems else 0
    if args.cmd == "begin":
        problems = begin(
            args.kind,
            cfg,
            args.config,
            args.lock,
            args.session,
            args.run_dir,
            bundle=args.bundle,
            preset=args.preset,
            ident=ident,
        )
        for p in problems:
            print(f"[free-floor] BLOCKED at run start: {p}", file=sys.stderr)
        return 1 if problems else 0
    # drill-eval / walk-eval
    runtime = runtime_identity(cfg, args.config, args.lock, args.bundle, args.session, ident)
    fn = drill_eval if args.cmd == "drill-eval" else walk_eval
    rec = fn(args.run_dir, args.session, ident.sha or "", runtime)
    print(f"[free-floor] {args.cmd}: {rec['verdict']}")
    for p in rec["problems"]:
        print(f"[free-floor]   {p}", file=sys.stderr)
    return 0 if rec["verdict"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
