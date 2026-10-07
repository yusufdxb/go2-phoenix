"""Read-only desk readiness report and the exact lab command sequence for FREE_FLOOR.

Imported by ``free_floor_gate.py desk-ready | lab-commands``. It reads the repo, the local archive
and the local evidence; it never touches a robot, the Jetson or the network, and it never writes.

``desk-ready`` answers ONE question: is everything that can be verified without the robot verified?
Its verdict is never robot readiness: payload verification and the live drill are listed as pending,
always, because this script cannot see them.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import free_floor_gate as ff
import yaml

from phoenix.sim2real import deploy_contract as dc
from phoenix.sim2real.provenance import (
    DEFAULT_EXPECTED_BRANCH,
    identity_problems,
    resolve_code_identity,
)

REPO = ff.REPO
WALK_CFG = REPO / "configs/sim2real/proposed/deploy_walk_iter2750.yaml"
WALK_LOCK = REPO / "configs/sim2real/locks/deploy_walk_iter2750.lock.yaml"
STAND_CFG = REPO / "configs/sim2real/deploy_walk_iter2750_stand.yaml"
STAND_LOCK = REPO / "configs/sim2real/locks/deploy_walk_iter2750_stand.lock.yaml"
S50_CFG = REPO / "configs/sim2real/proposed/deploy_walk_fric_s50_3749.yaml"
S50_LOCK = REPO / "configs/sim2real/locks/deploy_walk_fric_s50_3749.lock.yaml"
#: where the archived model_2750 export lives on this workstation (read-only archive)
ARCHIVE_EXPORT = Path.home() / "Archive/phoenix-deploy-candidates-2026-10-03/eval/iter2750_gate"
#: payload paths from the project notes (2026-10-06 session); verify with lablink exec, not assumed
PAYLOAD_HOME = "/home/unitree"
PAYLOAD_REPO = f"{PAYLOAD_HOME}/yusuf/go2-phoenix"
PAYLOAD_PHOENIX = f"{PAYLOAD_HOME}/phoenix"


def _git(*args: str) -> str:
    res = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=30, check=False
    )
    return res.stdout.strip()


def _workstation_stage_a(sha: str) -> str | None:
    for d in sorted(dc.default_session_dirs()):
        rec, err = ff._read_json(d / "stage_A.json")
        if err or not isinstance(rec, dict):
            continue
        ident = rec.get("code_identity") or {}
        if (
            rec.get("verdict") == "GO"
            and rec.get("mode") == "workstation"
            and not rec.get("rehearsal")
            and ident.get("sha") == sha
            and not ident.get("dirty")
            and rec.get("lock_path", "").endswith("deploy_walk_iter2750_stand.lock.yaml")
        ):
            return d.name
    return None


def import_origin() -> tuple[bool, str]:
    """Is ``phoenix`` imported from THIS checkout? An editable install elsewhere (the main checkout,
    in a worktree) silently wins over a missing PYTHONPATH, and the readiness answer is only about
    this tree."""
    import phoenix

    where = Path(phoenix.__file__).resolve()
    return where.is_relative_to(REPO), f"phoenix resolves to {where}, this checkout is {REPO}"


def desk_checks() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        out.append({"check": name, "ok": bool(ok), "detail": detail})

    from phoenix.sim2real import walk_grant as wg

    ok, detail = import_origin()
    add(
        "phoenix is imported from THIS checkout (not another worktree or the main checkout)",
        ok,
        detail,
    )
    add(
        "walk grant: FREE_FLOOR is bound to the incumbent ONNX, GANTRY and FREE_FLOOR presets are "
        "the registered ones",
        wg.FREE_FLOOR_ONNX_SHA256 == ff.INCUMBENT["policy.onnx"]
        and wg.ALLOWED_PRESETS[wg.MODE_FREE_FLOOR] == (wg.STAND_ZERO, ff.FREE_FLOOR_PRESET)
        and all(p in ff.PRESETS for p in wg.GANTRY_PRESETS),
    )
    ident = resolve_code_identity(REPO)
    idp = identity_problems(ident, expected_branch=DEFAULT_EXPECTED_BRANCH)
    add("code identity: branch, clean tracked tree", not idp, "; ".join(idp) or f"{ident.branch}")
    untracked = _git(
        "status", "--porcelain", "--untracked-files=all", "--", "src", "configs", "scripts", "docs"
    )
    add(
        "no untracked file in the payload sync set (only tracked files travel)",
        not untracked,
        untracked.replace("\n", "; ")[:300],
    )
    cfg = yaml.safe_load(WALK_CFG.read_text())
    paths = dc.artifact_paths(cfg)
    have = all(p.is_file() for p in paths.values())
    add("model_2750 artifacts present in the local archive", have, str(ARCHIVE_EXPORT))
    if have:
        ip = ff.identity_problems(cfg, WALK_CFG, WALK_LOCK, None)
        add(
            "identity: model_2750 hashes, walking config+lock, manifest gains, deploy contract "
            "(F/G hardware evidence), v2 observation schema",
            not ip,
            "; ".join(ip)[:600],
        )
        s50 = yaml.safe_load(S50_CFG.read_text())
        s50p = ff.identity_problems(s50, S50_CFG, S50_LOCK, None)
        add(
            "seed 50 config+lock is REFUSED by the free-floor identity gate",
            any("SEED 50" in p for p in s50p),
            f"{len(s50p)} problem(s)",
        )
    add(
        "seed 50 hashes alone are refused by name",
        len(ff.hash_identity_problems(ff.SEED50_HASHES, "t")) == 3,
    )
    stand = yaml.safe_load(STAND_CFG.read_text())
    for stage, motors in (("C", "off"), ("F", "live"), ("G", "live"), ("H", "live")):
        ok, why = ff._stage_ok(stage, cfg, dc.default_session_dirs(), motors=motors)
        add(
            f"hardware evidence: stage {stage} GO for model_2750's stand twin (2026-10-06)",
            ok,
            "; ".join(why)[:300],
        )
    del stand
    unrev = ff.unreviewed_runs(dc.default_session_dirs())
    add("no unreviewed free-floor walk in logs/hw_sessions", not unrev, "; ".join(unrev)[:300])
    sa = _workstation_stage_a(ident.sha or "")
    add(
        "workstation stage A GO at THIS commit on the stand lock (needed to stage the bundle)",
        sa is not None,
        sa
        or "run: DEPLOY_CFG=configs/sim2real/deploy_walk_iter2750_stand.yaml "
        "DEPLOY_LOCK=configs/sim2real/locks/deploy_walk_iter2750_stand.lock.yaml "
        "PHOENIX_EXPECT_SHA=$(git rev-parse HEAD) scripts/harness_preflight.sh A",
    )
    sh = subprocess.run(
        ["bash", "-n", str(REPO / "scripts/harness_preflight.sh")], capture_output=True
    )
    add("harness_preflight.sh parses", sh.returncode == 0, sh.stderr.decode()[:200])
    for rel in (
        "scripts/free_floor_gate.py",
        "scripts/free_floor_remote_trace.py",
        "scripts/derive_walk_bundle.sh",
        "scripts/stage_payload_repo.sh",
        "scripts/stage_payload_bundle.sh",
        "scripts/payload_prep.sh",
        "src/phoenix/sim2real/free_floor_state.py",
        "src/phoenix/sim2real/walk_grant.py",
    ):
        add(f"tracked: {rel}", bool(_git("ls-files", "--", rel)))
    real = REPO / "tests/fixtures/gantry_ab/real_policy_exits.json"
    add("real-tick fixtures for the STOP/deadman analysis present", real.is_file())
    return out


def pending() -> dict[str, list[str]]:
    return {
        "PAYLOAD VERIFICATION PENDING (cannot be seen from the desk)": [
            "code sync to the Orin and its manifest verification (stage_payload_repo.sh)",
            "stand bundle staged + walking bundle derived ON the Orin",
            "stage A --payload on the Orin at this commit",
            "come-here.service stopped, battery above 20 %, DDS NIC up (payload_prep.sh)",
        ],
        "HARDWARE DRILL PENDING (cannot be seen from the desk)": [
            "level, clean, dry floor with mu >= 0.4 and no mats; fall area clear; spotters outside it",
            "the live STOP/deadman drill and its measured end-to-end STOP time",
            "the walk itself, the tape measurement and the human review",
        ],
    }


def main_desk(args: Any) -> int:
    if args.cmd == "lab-commands":
        print(lab_commands(args.orin))
        return 0
    checks = desk_checks()
    ready = all(c["ok"] for c in checks)
    if args.json:
        print(json.dumps({"desk_ready": ready, "checks": checks, "pending": pending()}, indent=2))
        return 0 if ready else 1
    for c in checks:
        print(
            f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['check']}"
            + (f"\n         {c['detail']}" if c["detail"] and not c["ok"] else "")
        )
    print()
    print("VERDICT: " + ("FREE-FLOOR DESK READY" if ready else "DESK BLOCKERS REMAIN"))
    print("This is NOT robot readiness. Still pending, and not claimable from here:")
    for head, items in pending().items():
        print(f"  {head}")
        for i in items:
            print(f"    - {i}")
    return 0 if ready else 1


def lab_commands(orin: str) -> str:
    sha = resolve_code_identity(REPO).sha or "<unresolved>"
    s12 = sha[:12]
    stand = f"{PAYLOAD_PHOENIX}/walk2750-stand-{s12}"
    walk = f"{PAYLOAD_PHOENIX}/walk2750-{s12}"
    export = str(ARCHIVE_EXPORT)
    return f"""# FREE_FLOOR lab sequence for commit {sha}
# Values are resolved from this checkout. Payload paths come from the 2026-10-06 notes
# (user unitree, repo {PAYLOAD_REPO}); step 3 proves them before anything is copied.
# No passwords here. If scp/rsync asks for one, it is the Jetson login: export JETSON_PW yourself.
# Walking authority: every drill/walk issues its own short, run-bound grant (walk_grant.json in the run
# directory); a bridge or policy node started by hand has none and refuses to walk. Never launch them by hand.

## 0. WORKSTATION (mewtwo): verify the tree, then ask the desk checker
cd {REPO}
git rev-parse HEAD            # must print {sha}
git status --short            # tracked tree clean
scripts/free_floor_gate.py desk-ready    # needs: VERDICT: FREE-FLOOR DESK READY

## 1. LABLINK: the Orin must be reachable (stop on anything but green)
lablink doctor --json ; lablink status {orin} --json     # exit 0 = green

## 2. CLOCK (the payload has no RTC): set it from here, once
UTC="$(date -u +'%Y-%m-%d %H:%M:%S')"

## 3. PROVE THE PAYLOAD PATHS (read-only)
lablink exec {orin} 'ls -d {PAYLOAD_REPO} {PAYLOAD_PHOENIX}; echo HOME=$HOME; systemctl is-active come-here.service || true'

## 4. SYNC THE CODE (tracked files only; re-verifies the manifest on the payload)
ORIN="$(lablink resolve {orin})"
scripts/stage_payload_repo.sh "unitree@$ORIN:{PAYLOAD_REPO}"

## 5. STAGE THE STAND BUNDLE (workstation stage A at this commit is required: desk-ready checks it)
STAGE_A_SESSION="$(ls -d logs/hw_sessions/*_{s12} | head -1)" \\
DEPLOY_LOCK=configs/sim2real/locks/deploy_walk_iter2750_stand.lock.yaml \\
scripts/stage_payload_bundle.sh {export} \\
    configs/sim2real/deploy_walk_iter2750_stand.yaml "unitree@$ORIN:{stand}"

## 6. ON THE ORIN: prepare (stops come-here, checks DDS NIC, writes payload_env.sh), derive the walking bundle
lablink ssh {orin}
  cd {PAYLOAD_REPO}
  ./scripts/payload_prep.sh --utc "<paste $UTC from step 2>" --expect-sha {sha} --bundle {stand}
  scripts/derive_walk_bundle.sh {stand} configs/sim2real/proposed/deploy_walk_iter2750.yaml \\
      configs/sim2real/locks/deploy_walk_iter2750.lock.yaml {walk}

## 7. ON THE ORIN, FRESH SHELL: stage A --payload (STAND env)
env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/local/bin bash --noprofile --norc
  cd {PAYLOAD_REPO}
  source payload_env.sh
  export PHOENIX_EXPECT_SHA={sha} PHOENIX_BUNDLE={stand}
  export DEPLOY_CFG={stand}/deploy_walk_iter2750_stand.yaml DEPLOY_LOCK=configs/sim2real/locks/deploy_walk_iter2750_stand.lock.yaml
  SESS_FILE={PAYLOAD_PHOENIX}/session_{s12}.txt    # ONE session directory for stage A, the drill and the walk, even across UTC midnight
  [ -f "$SESS_FILE" ] || echo "logs/hw_sessions/$(date -u +%Y%m%d)_{s12}" > "$SESS_FILE"
  export PHOENIX_SESSION="$(cat "$SESS_FILE")"
  scripts/harness_preflight.sh A --payload                  # needs: STAGE A: GO (mode payload)

## 8. SAME SESSION, FRESH SHELL: the WALK env (never export anything else by hand)
env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/local/bin bash --noprofile --norc
  cd {PAYLOAD_REPO}
  source payload_env.sh
  export PHOENIX_EXPECT_SHA={sha} PHOENIX_BUNDLE={walk}
  export DEPLOY_CFG={walk}/deploy_walk_iter2750.yaml DEPLOY_LOCK=configs/sim2real/locks/deploy_walk_iter2750.lock.yaml
  SESS_FILE={PAYLOAD_PHOENIX}/session_{s12}.txt    # ONE session directory for stage A, the drill and the walk, even across UTC midnight
  [ -f "$SESS_FILE" ] || echo "logs/hw_sessions/$(date -u +%Y%m%d)_{s12}" > "$SESS_FILE"
  export PHOENIX_SESSION="$(cat "$SESS_FILE")"
  scripts/harness_preflight.sh FREE_FLOOR gate              # software gates only, motors off; shows the NEXT step
  scripts/harness_preflight.sh FREE_FLOOR drill             # LIVE zero-command stand; release the deadman ~2 s after policy starts
  cat "$PHOENIX_SESSION"/free_floor_drill_FF_drill_*.json   # read it: verdict, e2e_stop, problems
  scripts/harness_preflight.sh FREE_FLOOR gate              # now the walk phase must pass
  # physical setup on the walking floor (tape marks under both rear feet), then:
  scripts/harness_preflight.sh FREE_FLOOR walk vx03_2s      # LIVE; five YES + the typed phrase; tape measurement after it folds
  scripts/harness_preflight.sh FREE_FLOOR review accept --by <you> --note "<what you saw>"   # or reject

## 9. BACK ON THE WORKSTATION: pull the evidence (read-only, nothing is deleted on the Orin)
rsync -a "unitree@$ORIN:{PAYLOAD_REPO}/logs/hw_sessions/*_{s12}" logs/hw_sessions/
"""
