#!/usr/bin/env bash
# Hardware campaign runner for N arms (A, B, C, ...). It started as the exploratory 2-arm
# gantry A/B (docs/preregistration/2026-10-07-gantry-ab-exploratory.md, still the default arms
# file); the arms, block order and per-block slots now come from an arms file
# (configs/sim2real/campaigns/*.arms.json, written by scripts/make_campaign_arms.py).
#
# Wraps scripts/harness_preflight.sh; it never bypasses a harness gate. Every live stage
# still needs the operator's typed LIVE phrase at an interactive terminal, the deadman,
# the battery and contention checks, the per-arm ledger (A..E before F/G) and, for W, the
# hash-bound F/G walking evidence of that exact arm. For W the phrase says ON GANTRY or ON
# FLOOR; the setup typed is recorded in the grant and in campaign.jsonl. This script only
# fixes WHICH arm runs WHEN (the arms file's block order), keeps each arm's stage records in its
# own session directory, refuses bundles whose policy hashes are not the planned arm's, and
# records the operator's measurements in campaign.jsonl.
#
# Run ON THE PAYLOAD from the synced checkout (scripts/stage_payload_repo.sh):
#   scripts/hw_campaign.sh init [ARMS_FILE]       pin campaign id, commit, arms file, sessions
#   scripts/hw_campaign.sh verify                 bundles + locks + hashes of every arm (no motion)
#   scripts/hw_campaign.sh ladder <ARM> <stage> [args]   harness stage A(--payload)/B/C/D/E/F/G for that arm
#   scripts/hw_campaign.sh block <N>              the arms file's slot sequence for block N (+ STOP drill in each arm's last block)
#   scripts/hw_campaign.sh status                 what is done, what is next
# ARMS_FILE (init) defaults to $PHOENIX_CAMPAIGN_ARMS, then configs/sim2real/campaigns/ab_2026-10-07.arms.json;
# afterwards every command reads the copy pinned in the campaign directory (arms.json).
# scripts/gantry_ab_campaign.sh is a thin wrapper kept for the old name.
# On mewtwo after copying logs back:
#   python3 scripts/gantry_ab_analyze.py --campaign logs/gantry_ab/<id>/campaign.jsonl --out results/gantry_ab/<id>
#
# Bundles (ON THE PAYLOAD; staged per arm with stage_payload_bundle.sh, walking twin with
# derive_walk_bundle.sh): PHOENIX_AB_<ARM>_STAND and PHOENIX_AB_<ARM>_WALK, else the arms file's
# stand_bundle / walk_bundle, else ~/phoenix/ab<arm>-<stand|walk>-<sha8>.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
HARNESS="$REPO_ROOT/scripts/harness_preflight.sh"
DEFAULT_ARMS="configs/sim2real/campaigns/ab_2026-10-07.arms.json"
MIN_BLOCK_SOC=30

halt() { printf '\033[31mHALT: %s\033[0m\n' "$*" >&2; exit 1; }
SHA="$(git rev-parse HEAD)"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || halt "tracked files modified; the campaign runs only on a clean checkout"
ROOT="${PHOENIX_CAMPAIGN_ROOT:-logs/gantry_ab}"
CAMPAIGN_FILE_PTR="$ROOT/CURRENT"

cmd="${1:-status}"
[[ $# -gt 0 ]] && shift
# Arms file: init takes it from its argument; later commands use the copy pinned in the campaign.
if [[ "$cmd" == init ]]; then
    ARMS_FILE="${1:-${PHOENIX_CAMPAIGN_ARMS:-$DEFAULT_ARMS}}"
    [[ $# -eq 0 ]] || shift
elif [[ -n "${PHOENIX_CAMPAIGN_ARMS:-}" ]]; then
    ARMS_FILE="$PHOENIX_CAMPAIGN_ARMS"
elif [[ -f "$CAMPAIGN_FILE_PTR" && -f "$(cat "$CAMPAIGN_FILE_PTR")/arms.json" ]]; then
    ARMS_FILE="$(cat "$CAMPAIGN_FILE_PTR")/arms.json"
else
    ARMS_FILE="$DEFAULT_ARMS"
fi
[[ -f "$ARMS_FILE" ]] || halt "arms file $ARMS_FILE does not exist"
# Validated by make_campaign_arms.py (hashes are 64 hex, slots are presets W may authorize, ...);
# its output is shlex-quoted shell assignments: CAMPAIGN_NAME PREREG ARM_IDS ORDER SLOTS
# STOP_DRILL_BLOCKS and the per-arm tables ONNX_SHA DATA_SHA CKPT_SHA STAND_LOCK WALK_LOCK STAND_DIR WALK_DIR ARM_LABEL.
ARMS_SHELL="$(python3 "$REPO_ROOT/scripts/make_campaign_arms.py" --emit-shell "$ARMS_FILE")" || halt "arms file $ARMS_FILE is not valid"
eval "$ARMS_SHELL"
NBLOCKS=${#ORDER[@]}

campaign_dir() {
    [[ -f "$CAMPAIGN_FILE_PTR" ]] || halt "no campaign: run '$0 init' first"
    local d
    d="$(cat "$CAMPAIGN_FILE_PTR")"
    [[ -f "$d/campaign.jsonl" ]] || halt "campaign file $d/campaign.jsonl missing"
    echo "$d"
}

header_get() {  # field of the campaign header
    python3 - "$1/campaign.jsonl" "$2" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get("kind") == "header":
        print(r[sys.argv[2]]); break
PY
}

append() {  # append one JSON record (python dict literal fields via env) to campaign.jsonl
    python3 - "$1" "$2" <<'PY'
import json, sys, time
rec = json.loads(sys.argv[2])
rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
with open(sys.argv[1], "a") as f:
    f.write(json.dumps(rec) + "\n")
PY
}

bundle_dir() {  # arm kind -> bundle directory (env override, then the arms file, then the default)
    local arm="$1" kind="$2" var="PHOENIX_AB_${1}_${2^^}" cfg
    if [[ "$kind" == stand ]]; then cfg="${STAND_DIR[$arm]}"; else cfg="${WALK_DIR[$arm]}"; fi
    cfg="${cfg/#\~/$HOME}"
    echo "${!var:-${cfg:-$HOME/phoenix/ab${arm}-${kind}-${SHA:0:8}}}"
}

bundle_cfg() {
    python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import activation; print(activation.find_config(sys.argv[1]))' "$1"
}

verify_arm() {  # arm: both bundles verify, hashes are the planned arm's, locks verify
    local arm="$1" kind dir cfg lock
    for kind in stand walk; do
        dir="$(bundle_dir "$arm" "$kind")"
        [[ -d "$dir" ]] || halt "arm $arm $kind bundle $dir does not exist"
        ( cd "$dir" && sha256sum -c --quiet SHA256SUMS ) || halt "arm $arm $kind bundle SHA256SUMS does not verify"
        python3 "$dir/activation.py" verify --bundle "$dir" >/dev/null || halt "arm $arm $kind bundle activation does not verify"
        [[ "$(sha256sum "$dir/policy.onnx" | cut -c1-64)" == "${ONNX_SHA[$arm]}" ]] || halt "arm $arm $kind bundle policy.onnx is NOT arm $arm's"
        [[ "$(sha256sum "$dir/policy.onnx.data" | cut -c1-64)" == "${DATA_SHA[$arm]}" ]] || halt "arm $arm $kind bundle policy.onnx.data is NOT arm $arm's"
        local ck
        ck="$(ls "$dir"/*.pt 2>/dev/null | grep -v '/policy.pt$' | head -1 || true)"
        [[ -n "$ck" && "$(sha256sum "$ck" | cut -c1-64)" == "${CKPT_SHA[$arm]}" ]] || halt "arm $arm $kind bundle checkpoint is missing or NOT arm $arm's"
        cfg="$(bundle_cfg "$dir")"
        if [[ "$kind" == stand ]]; then lock="${STAND_LOCK[$arm]}"; else lock="${WALK_LOCK[$arm]}"; fi
        python3 - "$lock" "$cfg" "$kind" <<'PY' || halt "arm $arm $kind: lock does not verify against the bundle config"
import sys, yaml
from phoenix.sim2real.deploy_contract import is_stand_only, load_lock, verify_lock
cfg = yaml.safe_load(open(sys.argv[2]))
problems, _ = verify_lock(load_lock(sys.argv[1]), cfg, sys.argv[2])
if (sys.argv[3] == "stand") != is_stand_only(cfg):
    problems.append("stand_only flag does not match the bundle kind")
for p in problems:
    print("  " + p, file=sys.stderr)
sys.exit(1 if problems else 0)
PY
    done
    echo "[ab] arm $arm: stand + walk bundles verified (onnx ${ONNX_SHA[$arm]:0:12}, checkpoint ${CKPT_SHA[$arm]:0:12})"
}

arm_env() {  # arm kind: export the harness environment for that arm
    local arm="$1" kind="$2" d sess
    d="$(campaign_dir)"
    sess="$(header_get "$d" "session_$arm")"
    export PHOENIX_SESSION="$sess" PHOENIX_EXPECT_SHA="$SHA"
    if [[ "$kind" == stand ]]; then
        export DEPLOY_CFG="$(bundle_cfg "$(bundle_dir "$arm" stand)")" DEPLOY_LOCK="${STAND_LOCK[$arm]}"
        export PHOENIX_BUNDLE="$(bundle_dir "$arm" stand)"
    else
        export DEPLOY_CFG="$(bundle_cfg "$(bundle_dir "$arm" walk)")" DEPLOY_LOCK="${WALK_LOCK[$arm]}"
        unset PHOENIX_BUNDLE
    fi
}

run_harness() {  # arm kind log-tag harness-args... ; echoes the run dir the harness created
    local arm="$1" kind="$2" tag="$3" d log rc
    shift 3
    d="$(campaign_dir)"
    log="$d/harness_${tag}_$(date -u +%Y%m%dT%H%M%SZ).log"
    (
        arm_env "$arm" "$kind"
        echo "[ab] arm $arm ($kind) DEPLOY_CFG=$DEPLOY_CFG DEPLOY_LOCK=$DEPLOY_LOCK PHOENIX_SESSION=$PHOENIX_SESSION" >&2
        bash "$HARNESS" "$@" 2>&1 | tee "$log" >&2
        exit "${PIPESTATUS[0]}"
    ) && rc=0 || rc=$?
    local run
    run="$(grep -o 'evidence directory [^ ]*' "$log" | tail -1 | cut -d' ' -f3 || true)"
    echo "$rc ${run:-none} $log"
}

ask() {  # prompt -> answer on stdout (empty allowed)
    local a
    read -r -p "$1" a </dev/tty
    echo "$a"
}

yn() { local a; a="$(ask "$1 [y/n]: ")"; [[ "$a" == y ]]; }

battery_soc() {  # prints the lowest SoC seen; exit 0 only if every sample is above MIN_BLOCK_SOC
    local out rc
    out="$(mktemp)"
    (
        ros="${PHOENIX_ROS_SETUP:-/opt/ros/humble/setup.bash}"
        unitree="${PHOENIX_UNITREE_SETUP:-$HOME/unitree_ros2/cyclonedds_ws/install/setup.bash}"
        set +u
        # shellcheck disable=SC1090
        [[ -f "$ros" ]] && source "$ros"
        # shellcheck disable=SC1090
        [[ -f "$unitree" ]] && source "$unitree"
        set -u
        python3 -m phoenix.sim2real.hw_probe battery --min-soc "$MIN_BLOCK_SOC" --out "$out" >/dev/null 2>&1
    ) && rc=0 || rc=$?
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("soc_min", ""))' "$out" 2>/dev/null || true
    rm -f "$out"
    return "$rc"
}

record_run() {  # block arm slot attempt rc run log
    local block="$1" arm="$2" slot="$3" attempt="$4" rc="$5" run="$6" log="$7" d tape_f="" tape_l="" meas_yaw="" setup="" axes rope touched aborted note
    d="$(campaign_dir)"
    echo
    echo "[ab] block $block arm $arm $slot attempt $attempt: harness exit $rc, run dir $run"
    # Which measurements the slot needs: forward tape for a vx preset, yaw for a wz preset, both for a combined one.
    axes="$(python3 -c 'import sys; from phoenix.sim2real.walk_command import PRESETS as P; p = P.get(sys.argv[1]); print(("fwd " if p and p.vx else "") + ("yaw" if p and p.wz else ""))' "$slot")"
    if [[ "$axes" == *fwd* ]]; then
        tape_f="$(ask "  TAPE forward displacement in metres (start mark -> end mark along the initial heading; empty if not measured): ")"
        tape_l="$(ask "  TAPE lateral displacement in metres (+ left; empty if not measured): ")"
    fi
    if [[ "$axes" == *yaw* ]]; then
        meas_yaw="$(ask "  YAW change in degrees (start heading -> end heading, + counter-clockwise; empty if not measured): ")"
    fi
    # The setup the operator typed for W (gantry|floor), from this run's grant; with no rope on the floor the rope question is skipped.
    [[ -f "$run/walk_grant.json" ]] && setup="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("setup", ""))' "$run/walk_grant.json" 2>/dev/null || true)"
    if [[ "$setup" == floor ]]; then
        rope=na
    else
        rope="$(yn "  Did the rope carry weight at any moment?" && echo true || echo false)"
    fi
    touched="$(yn "  Did anyone touch the robot during the run?" && echo true || echo false)"
    aborted="$(yn "  Was the run ABORTED (deadman released early or halted)?" && echo true || echo false)"
    note="$(ask "  Note (free text, one line): ")"
    append "$d/campaign.jsonl" "$(python3 - "$block" "$arm" "$slot" "$attempt" "$rc" "$run" "$log" "$tape_f" "$tape_l" "$rope" "$touched" "$aborted" "$note" "$meas_yaw" "$setup" <<'PY'
import json, os, sys
a = sys.argv[1:]
num = lambda s: float(s) if s.strip() else None
print(json.dumps({"kind": "run", "block": int(a[0]), "arm": a[1], "slot": a[2], "attempt": int(a[3]),
                  "harness_rc": int(a[4]), "run_dir": a[5], "harness_log": a[6],
                  "tape_fwd_m": num(a[7]), "tape_lat_m": num(a[8]), "meas_yaw_deg": num(a[13]),
                  "setup": a[14] or None, "rope_loaded": None if a[9] == "na" else a[9] == "true",
                  "touched": a[10] == "true", "aborted": a[11] == "true", "note": a[12],
                  "rehearsal": os.environ.get("PHOENIX_REHEARSAL") == "1"}))
PY
)"
    # Prereg amendment 4: the operator's answer is not the only abort detector. The same
    # policy-exit classification the analysis uses is read from this run's bridge log now;
    # a run whose log does not show the expected exit is treated like a reported abort.
    local lc lc_ok=true
    lc="$(python3 scripts/gantry_ab_analyze.py --classify-run "$run" --slot "$slot" 2>&1)" || lc_ok=false
    append "$d/campaign.jsonl" "$(python3 - "$block" "$slot" "$attempt" "$lc_ok" "$lc" <<'PY'
import json, sys
a = sys.argv[1:]
try:
    c = json.loads(a[4].strip().splitlines()[-1])
except (ValueError, IndexError):
    c = {"error": a[4][-500:]}
print(json.dumps({"kind": "log_check", "block": int(a[0]), "slot": a[1], "attempt": int(a[2]),
                  "clean": a[3] == "true", "exit": c.get("exit"), "problems": c.get("problems"),
                  "stop_latency_s": c.get("stop_latency_s"), "error": c.get("error")}))
PY
)"
    if [[ "$lc_ok" == true ]]; then
        echo "[ab] log check: expected exit for $slot confirmed from bridge.jsonl"
    else
        echo "[ab] LOG CHECK: this run did NOT end as expected for $slot (from bridge.jsonl): $(tail -c 400 <<<"$lc")"
        [[ "$slot" != stop_drill && "$aborted" == false ]] \
            && echo "[ab] DISCREPANCY: operator answered 'not aborted'; the logs decide (prereg amendment 4)."
    fi
    [[ "$aborted" == false && "$rope" != true && "$touched" == false && "$rc" == 0 && "$lc_ok" == true ]]
}

slot_run() {  # block arm slot -> runs one slot with at most one repeat
    local block="$1" arm="$2" slot="$3" attempt out rc run log
    for attempt in 1 2; do
        case "$slot" in
            G) out="$(run_harness "$arm" stand "b${block}_G" G)" ;;
            stop_drill)
                echo "[ab] STOP DRILL (arm $arm): stage G; RELEASE L1 about 2 s after the policy takes over (announce it first)."
                out="$(run_harness "$arm" stand "b${block}_stopdrill" G)" ;;
            *) out="$(run_harness "$arm" walk "b${block}_${slot}" W "$slot")" ;;
        esac
        read -r rc run log <<<"$out"
        if record_run "$block" "$arm" "$slot" "$attempt" "$rc" "$run" "$log"; then
            return 0
        fi
        [[ "$slot" == stop_drill ]] && return 0  # the drill is never repeated
        if [[ "$attempt" == 1 ]] && yn "  Slot not clean. Repeat ONCE now (prereg allows one repeat)?"; then
            continue
        fi
        return 1
    done
    return 1
}

case "$cmd" in
    init)
        mkdir -p "$ROOT"
        id="$(date -u +%Y%m%d)_${SHA:0:12}"
        [[ "$CAMPAIGN_NAME" == "ab_2026-10-07" ]] || id="${id}_${CAMPAIGN_NAME}"
        d="$ROOT/$id"
        [[ -e "$d" ]] && halt "campaign $d already exists"
        mkdir -p "$d"
        cp "$ARMS_FILE" "$d/arms.json"
        # The header pins the arms (hashes and labels) so the analyzer needs no other source.
        append "$d/campaign.jsonl" "$(python3 - "$id" "$SHA" "$d/arms.json" <<'PY'
import hashlib, json, sys
cid, sha, path = sys.argv[1:4]
raw = open(path, "rb").read()
doc = json.loads(raw)
hdr = {"kind": "header", "campaign": cid, "commit": sha, "name": doc["name"], "order": doc["order"],
       "slots": doc["slots"], "prereg": doc.get("prereg"), "effort_bag": False,
       "arms_file_sha256": hashlib.sha256(raw).hexdigest(),
       "arms": {a: {"checkpoint": s["ckpt_sha256"], "policy.onnx": s["onnx_sha256"],
                    "policy.onnx.data": s["data_sha256"], "label": s.get("label")}
                for a, s in doc["arms"].items()}}
for a in doc["arms"]:
    hdr[f"session_{a}"] = f"logs/hw_sessions/{cid}_ab{a}"
print(json.dumps(hdr))
PY
)"
        echo "$d" > "$CAMPAIGN_FILE_PTR"
        echo "[ab] campaign $id at $SHA; arms ${ARM_IDS[*]}, order ${ORDER[*]}, slots ${SLOTS[*]}; sessions logs/hw_sessions/${id}_ab<arm>"
        ;;
    verify)
        for arm in "${ARM_IDS[@]}"; do verify_arm "$arm"; done
        ;;
    ladder)
        arm="${1:?arm id (${ARM_IDS[*]})}"; stage="${2:?stage}"; shift 2
        [[ -n "${ONNX_SHA[$arm]:-}" ]] || halt "arm must be one of ${ARM_IDS[*]}"
        [[ "$stage" =~ ^[ABCDEFG]$ ]] || halt "ladder stages are A B C D E F G (W runs only inside a block)"
        verify_arm "$arm"
        out="$(run_harness "$arm" stand "ladder_${arm}_${stage}" "$stage" "$@")"
        read -r rc run log <<<"$out"
        append "$(campaign_dir)/campaign.jsonl" "$(python3 -c 'import json,sys; print(json.dumps({"kind":"ladder","arm":sys.argv[1],"stage":sys.argv[2],"harness_rc":int(sys.argv[3]),"run_dir":sys.argv[4],"harness_log":sys.argv[5]}))' "$arm" "$stage" "$rc" "$run" "$log")"
        exit "$rc"
        ;;
    block)
        k="${1:?block number 1..$NBLOCKS}"
        [[ "$k" =~ ^[0-9]+$ ]] && (( k >= 1 && k <= NBLOCKS )) || halt "block must be 1..$NBLOCKS"
        d="$(campaign_dir)"
        arm="${ORDER[$((k - 1))]}"
        python3 - "$d/campaign.jsonl" "$k" <<'PY' || halt "blocks run in order; block $k is not next (see status)"
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1])]
done = {r["block"] for r in recs if r.get("kind") == "block_end"}
started = {r["block"] for r in recs if r.get("kind") == "block_start"}
stopped = any(r.get("kind") == "campaign_stop" for r in recs)
k = int(sys.argv[2])
if k in started and k not in done:
    print(f"block {k} was started and never finished; a block is never re-run (prereg amendment 2). "
          f"Close it with: hw_campaign.sh abandon {k} <reason>", file=sys.stderr)
sys.exit(0 if not stopped and k not in started and all(b in done for b in range(1, k)) else 1)
PY
        verify_arm "$arm"
        soc="$(battery_soc)" || halt "battery below ${MIN_BLOCK_SOC} % at block start (prereg: remaining blocks not run); record with: $0 stop battery"
        stop_block=false
        for b in "${STOP_DRILL_BLOCKS[@]}"; do [[ "$k" == "$b" ]] && stop_block=true; done
        echo "[ab] BLOCK $k = arm $arm, battery ${soc:-?} %. Sequence: ${SLOTS[*]}$( [[ $stop_block == true ]] && echo ', STOP drill')."
        append "$d/campaign.jsonl" "$(python3 -c 'import json,sys; print(json.dumps({"kind":"block_start","block":int(sys.argv[1]),"arm":sys.argv[2],"battery_soc":sys.argv[3]}))' "$k" "$arm" "$soc")"
        ok=true
        for slot in "${SLOTS[@]}"; do
            if ! slot_run "$k" "$arm" "$slot"; then
                ok=false
                echo "[ab] slot $slot of block $k ended without a clean run; per prereg the slot is missing."
                yn "  Did this trigger a campaign STOP condition (fault, fall, crash, limit clip, 2nd abort of this arm+preset)?" \
                    && { append "$d/campaign.jsonl" "{\"kind\": \"campaign_stop\", \"block\": $k, \"arm\": \"$arm\", \"slot\": \"$slot\"}"; halt "campaign STOP recorded"; }
                [[ "$slot" == G ]] && { echo "[ab] G stand not clean: this arm's walking stays blocked in code until a fresh G GO; skipping its W slots."; break; }
            fi
            sleep 1
        done
        if [[ "$stop_block" == true ]]; then
            slot_run "$k" "$arm" stop_drill || true
        fi
        append "$d/campaign.jsonl" "$(python3 -c 'import json,sys; print(json.dumps({"kind":"block_end","block":int(sys.argv[1]),"arm":sys.argv[2],"all_slots_clean":sys.argv[3]=="true"}))' "$k" "$arm" "$ok")"
        echo "[ab] block $k done. Wait at least 60 s before the next run."
        ;;
    abandon)
        k="${1:?block number}"; shift
        [[ "$k" =~ ^[0-9]+$ ]] && (( k >= 1 && k <= NBLOCKS )) || halt "block must be 1..$NBLOCKS"
        d="$(campaign_dir)"
        append "$d/campaign.jsonl" "$(python3 -c 'import json,sys; print(json.dumps({"kind":"block_end","block":int(sys.argv[1]),"abandoned":True,"reason":" ".join(sys.argv[2:])}))' "$k" "$@")"
        echo "[ab] block $k closed as abandoned; its missing slots stay missing"
        ;;
    stop)
        d="$(campaign_dir)"
        append "$d/campaign.jsonl" "$(python3 -c 'import json,sys; print(json.dumps({"kind":"campaign_stop","reason":" ".join(sys.argv[1:])}))' "$@")"
        echo "[ab] campaign STOP recorded"
        ;;
    status)
        d="$(campaign_dir)"
        python3 - "$d/campaign.jsonl" <<'PY'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1])]
h = next(r for r in recs if r["kind"] == "header")
print(f"campaign {h['campaign']} commit {h['commit'][:12]} order {' '.join(h['order'])}")
for r in recs:
    if r["kind"] in ("ladder", "block_start", "block_end", "campaign_stop") or r["kind"] == "run":
        print(" ", {k: v for k, v in r.items() if k in ("kind", "arm", "stage", "block", "slot", "attempt", "harness_rc", "aborted", "reason")})
done = {r["block"] for r in recs if r["kind"] == "block_end"}
nxt = next((b for b in range(1, len(h["order"]) + 1) if b not in done), None)
print("next block:", nxt, "(arm", h["order"][nxt - 1] + ")" if nxt else "none")
PY
        ;;
    *) halt "unknown command $cmd (init | verify | ladder | block | abandon | stop | status)" ;;
esac
