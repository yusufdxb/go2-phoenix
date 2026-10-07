#!/usr/bin/env bash
# harness_preflight.sh: the one fail-loud answer to "is Phoenix ready for a
# stand-only live GO2 test?", and the runner for every staged hardware gate.
#
#   scripts/harness_preflight.sh              stage A on THIS machine, then the ledger.
#                                             Exit 0 only if stages A..E all count for
#                                             the current commit and lock, i.e. the
#                                             first live stand (F) is permitted.
#   scripts/harness_preflight.sh status       the ledger only, same exit code.
#   scripts/harness_preflight.sh A            offline gates (add --payload on the robot)
#   scripts/harness_preflight.sh B [secs]     on-robot dryrun, bridge on /lowcmd_dry     motors OFF
#   scripts/harness_preflight.sh C            real physical deadman, interactive           motors OFF
#   scripts/harness_preflight.sh D [secs]     LowState / IMU / joint-state freshness       motors OFF
#   scripts/harness_preflight.sh E [secs]     LIVE bridge hold-current, no policy          motors LIVE
#   scripts/harness_preflight.sh F | G | H    LIVE cmd=0 stand of the active config (DEPLOY_LABEL), 2 s / 5 s / 10 s x 3   motors LIVE
#   scripts/harness_preflight.sh W <preset>   LIVE one walking attempt (presets: python3 -m phoenix.sim2real.walk_command --list)   motors LIVE
#                                             needs a WALKING DEPLOY_CFG/LOCK (safety.stand_only: false). NOT a ledger stage: no evaluator, evidence never counted.
#                                             rehearsal: PHOENIX_REHEARSAL=1 ROS_LOCALHOST_ONLY=1 PHOENIX_CONFIRM='LIVE W <preset> ON GANTRY', plus synthetic F/G records via
#                                             PHOENIX_HW_SESSIONS=<dir> PHOENIX_HW_SESSIONS_TEST_ONLY=1 (deploy_contract.walking_evidence_problems), else walking is refused.
#   scripts/harness_preflight.sh FP [joint ...]   LIVE active joint-friction probe, robot HUNG feet off the ground, released (no sport mode)   motors LIVE
#                                             needs a STAND-ONLY DEPLOY_CFG/LOCK; the bridge holds, then friction_probe_node is the command source. NOT a ledger stage (no pf evaluate).
#                                             rehearsal: PHOENIX_REHEARSAL=1 ROS_LOCALHOST_ONLY=1 PHOENIX_CONFIRM='LIVE FP HUNG FEET OFF GROUND'.
#
# Stages never chain. Each live stage is its own invocation, refuses unless every
# earlier stage counts, and refuses without an operator typing the stage phrase at
# a terminal. Nothing here runs git fetch, merge, pull or checkout: code reaches the
# payload only through scripts/stage_payload_repo.sh, and this script only reads it.
#
# Environment:
#   DEPLOY_CFG          deploy config. Default configs/sim2real/deploy_stand_h25.yaml.
#                       On the payload set it to the ACTIVATED bundle's pinned copy.
#   DEPLOY_LOCK         lock for that config. Default configs/sim2real/locks/deploy_stand_h25.lock.yaml
#   PHOENIX_EXPECT_SHA  commit the running code must be. REQUIRED for B..H and for A --payload.
#   PHOENIX_BUNDLE      activated bundle directory (A --payload).
#   PHOENIX_SESSION     evidence directory. Default logs/hw_sessions/<utc date>_<commit 12>.
#   PHOENIX_DEADMAN     wireless (GO2 remote, default) or joy.
#   PHOENIX_ROS_SETUP / PHOENIX_UNITREE_SETUP   setup scripts sourced for B..H.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Single-threaded BLAS for every node this script launches (the reliability shield's
# per-tick matvec). Measured on the GO2 Orin NX payload 2026-09-18: default threads gave
# shield ticks up to 13.1 ms (p99.9 4.0 ms, 0.23% over the 2 ms budget); one thread, max 0.10 ms.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

DEPLOY_CFG="${DEPLOY_CFG:-configs/sim2real/deploy_stand_h25.yaml}"
DEPLOY_LOCK="${DEPLOY_LOCK:-configs/sim2real/locks/deploy_stand_h25.lock.yaml}"
EXPECT="${PHOENIX_EXPECT_SHA:-}"
DEADMAN="${PHOENIX_DEADMAN:-wireless}"

c_red()   { printf '\033[31m%s\033[0m\n' "$*"; }
c_green() { printf '\033[32m%s\033[0m\n' "$*"; }
halt()    { c_red "HALT: $*" >&2; exit 1; }
ts()      { date -u +%Y%m%dT%H%M%SZ; }

[[ -f "$DEPLOY_CFG" ]] || halt "DEPLOY_CFG $DEPLOY_CFG does not exist"
[[ -f "$DEPLOY_LOCK" ]] || halt "DEPLOY_LOCK $DEPLOY_LOCK does not exist"
# Names the active config in operator prompts: the lock's name (h25-stand, walk-iter2750-stand, ...).
DEPLOY_LABEL="$(python3 -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1])).get("name") or "H25")' "$DEPLOY_LOCK")"

cfg_get() {
    python3 -c 'import sys, yaml; v = yaml.safe_load(open(sys.argv[1])); [v := v[k] for k in sys.argv[2].split(".")]; print(v)' "$DEPLOY_CFG" "$1"
}

current_sha() {
    python3 -c 'from phoenix.sim2real.provenance import resolve_code_identity; print(resolve_code_identity(".").sha or "")'
}

SHA_NOW="$(current_sha)"
[[ -n "$SHA_NOW" ]] || halt "cannot resolve the running code's commit (not a git work tree, and no PAYLOAD_SYNC.txt)"
SESSION="${PHOENIX_SESSION:-logs/hw_sessions/$(date -u +%Y%m%d)_${SHA_NOW:0:12}}"
mkdir -p "$SESSION"

pf() {
    local sub="$1"; shift
    python3 -m phoenix.sim2real.preflight "$sub" "$@" \
        --config "$DEPLOY_CFG" --lock "$DEPLOY_LOCK" --session "$SESSION" \
        ${EXPECT:+--expect-sha "$EXPECT"}
}

need_sha() {
    [[ -n "$EXPECT" ]] || halt "PHOENIX_EXPECT_SHA is required for this stage: set it to the commit scripts/stage_payload_repo.sh reported"
    [[ "$SHA_NOW" == "$EXPECT"* ]] || halt "running code is $SHA_NOW, not the expected $EXPECT"
}

ros_env() {
    local ros="${PHOENIX_ROS_SETUP:-/opt/ros/humble/setup.bash}"
    local unitree="${PHOENIX_UNITREE_SETUP:-$HOME/unitree_ros2/cyclonedds_ws/install/setup.bash}"
    # shellcheck disable=SC1090
    [[ -f "$ros" ]] && set +u && source "$ros" && set -u
    # shellcheck disable=SC1090
    [[ -f "$unitree" ]] && set +u && source "$unitree" && set -u
    command -v ros2 >/dev/null || halt "ros2 not on PATH (PHOENIX_ROS_SETUP=$ros)"
    python3 -c 'import rclpy, unitree_go.msg' 2>/dev/null || halt "rclpy or unitree_go messages not importable (PHOENIX_UNITREE_SETUP=$unitree)"
    [[ "${RMW_IMPLEMENTATION:-}" == "rmw_cyclonedds_cpp" ]] || halt "RMW_IMPLEMENTATION must be rmw_cyclonedds_cpp, got '${RMW_IMPLEMENTATION:-}'"
    if [[ "${ROS_LOCALHOST_ONLY:-0}" != "1" ]]; then
        [[ -n "${CYCLONEDDS_URI:-}" && -f "${CYCLONEDDS_URI#file://}" ]] || halt "CYCLONEDDS_URI must name an existing file (got '${CYCLONEDDS_URI:-}'); see docs/go2_field_notes.md section 1"
    fi
}

# ------------------------------------------------------------ process control
declare -A PIDS=()
RUN=""

is_alive() {
    local pid="$1"
    [[ -r "/proc/$pid/status" ]] && ! grep -q '^State:[[:space:]]*Z' "/proc/$pid/status"
}

launch() {
    local name="$1"; shift
    "$@" >"$RUN/$name.log" 2>&1 &
    PIDS[$name]=$!
    echo "[$(date +%T)] launched $name (pid ${PIDS[$name]}): $*"
}

require_alive() {
    local name
    for name in "$@"; do
        is_alive "${PIDS[$name]}" || halt "$name died (log: $RUN/$name.log)"
    done
}

stop_proc() {
    local name="$1" pid="${PIDS[$1]:-}"
    [[ -n "$pid" ]] || return 0
    if is_alive "$pid"; then
        kill -INT "$pid" 2>/dev/null || true   # already exiting is fine; liveness was checked before
        for _ in $(seq 50); do is_alive "$pid" || break; sleep 0.1; done
        is_alive "$pid" && kill -TERM "$pid" 2>/dev/null
    fi
    wait "$pid" 2>/dev/null || true             # reap; its exit status is not the verdict
    unset "PIDS[$name]"
}

stop_all() {
    local name
    for name in friction_probe walk_cmd policy_node lowcmd_bridge lowstate_bridge deadman; do
        [[ -n "${PIDS[$name]:-}" ]] && stop_proc "$name"
    done
    return 0
}
trap stop_all EXIT

new_run() {
    RUN="$SESSION/$1_$(ts)"
    [[ -e "$RUN" ]] && halt "run directory $RUN already exists"
    mkdir -p "$RUN"
    echo "[run] evidence directory $RUN"
}

deadman_cmd() {
    case "$DEADMAN" in
        wireless) echo "python3 -m phoenix.sim2real.wireless_estop_node" ;;
        joy)      echo "python3 -m phoenix.sim2real.deadman_joy_node" ;;
        *)        halt "PHOENIX_DEADMAN must be wireless or joy, got $DEADMAN" ;;
    esac
}

wait_for_telemetry() {
    # wait_for_telemetry <file> <python predicate on a tick dict> <timeout s> <what>
    local file="$1" predicate="$2" timeout="$3" what="$4"
    python3 - "$file" "$predicate" "$timeout" <<'PY' || halt "timed out after ${timeout}s waiting for: $what"
import json, sys, time
path, predicate, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
deadline = time.monotonic() + timeout
check = eval("lambda t: " + predicate)
while time.monotonic() < deadline:
    try:
        with open(path) as fh:
            for line in fh:
                try:
                    tick = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if tick.get("record") == "tick" and check(tick):
                    sys.exit(0)
    except FileNotFoundError:
        pass
    time.sleep(0.2)
sys.exit(1)
PY
}

live_confirm() {
    local stage="$1" what="$2" setup="${3:-ground}"
    if [[ "$setup" == "hung" ]]; then
        cat <<EOF

================================================================================
 LIVE MOTOR STAGE $stage: $what
 The robot must be HUNG on the gantry, FEET OFF THE GROUND, in low-level mode
 (sport service released; never SelectMode normal), a spotter at the robot, a
 clear 2 m radius, and the operator holding the physical deadman.
 HALT (release the deadman) on: joint snap or motor buzz, any joint that keeps
 moving after its segment ends, any node crash, Jetson reboot or brownout.
================================================================================
EOF
    else
    cat <<EOF

================================================================================
 LIVE MOTOR STAGE $stage: $what
 The robot must be FEET ON THE GROUND on the fall mat, tethered or spotted, in
 low-level mode (sport service released; never SelectMode normal), a spotter at
 the robot, a clear 2 m radius, and the operator holding the physical deadman.
 HALT (release the deadman) on: joint snap or motor buzz, base tipping or
 divergent oscillation, any node crash, Jetson reboot or brownout.
================================================================================
EOF
    fi
    local want="LIVE $stage"
    if [[ "${PHOENIX_REHEARSAL:-0}" == "1" ]]; then
        [[ "${ROS_LOCALHOST_ONLY:-0}" == "1" ]] || halt "rehearsal requires ROS_LOCALHOST_ONLY=1"
        [[ "${PHOENIX_CONFIRM:-}" == "$want" ]] || halt "rehearsal needs PHOENIX_CONFIRM='$want'"
        c_red "REHEARSAL (localhost only, no robot): evidence is recorded as rehearsal and never counts"
        return 0
    fi
    [[ -t 0 ]] || halt "live stage $stage requires an operator at an interactive terminal"
    local answer
    read -r -p "Type '$want' to proceed, anything else aborts: " answer
    [[ "$answer" == "$want" ]] || halt "operator did not confirm stage $stage"
}

bridge_args() {
    local stage="$1" telemetry="$2"
    echo --config "$DEPLOY_CFG" --lock "$DEPLOY_LOCK" --telemetry "$telemetry" --stage "$stage" --expect-sha "$EXPECT"
}

# --------------------------------------------------------------------- stages
# Nothing else may share the robot. come-here.service autostarts on Jetson boot,
# starves the payload and can command the motors, so every on-robot stage runs
# this first. The verdict logic is phoenix.sim2real.preflight_eval
# .contention_checks, which fails closed when the probe recorded nothing.
# Operator interaction for a live stage.
#
# Default is the terminal, unchanged. With PHOENIX_OPERATOR_REMOTE=1 the same
# question is asked through the GO2 remote instead, so the person holding L1 and
# watching the robot is the one who answers, rather than a second person relaying
# it to a keyboard. That relay is what src/phoenix/sim2real/operator_remote.py
# was written for, and until now nothing called it.
#
# Fails closed in every direction: a timeout, an ambiguous press (A and B
# together) or a released deadman all exit non-zero and HALT the stage. The
# remote can never turn a missing answer into a pass.
#
# Modes: arm (hold L1 + Start), judgement (A yes / B NO-GO), release (let go of L1).
operator_gate() {
    local mode="$1" label="$2" prompt="$3"
    if [[ "${PHOENIX_OPERATOR_REMOTE:-0}" != "1" ]]; then
        case "$mode" in
            judgement)
                local answer
                read -r -p "$prompt" answer
                printf '%s' "$answer"
                ;;
            *)
                read -r -p "$prompt" _
                ;;
        esac
        return 0
    fi
    local out="${RUN:-$SESSION}/remote_${label}_${mode}.json"
    mkdir -p "$(dirname "$out")"
    echo "[${label}] remote: $prompt" >&2
    local rc=0
    python3 -m phoenix.sim2real.operator_remote "$mode" \
        --timeout-s "${PHOENIX_OPERATOR_REMOTE_TIMEOUT_S:-120}" --out "$out" >&2 || rc=$?
    local result
    result="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['result'])" "$out" 2>/dev/null || echo no_answer)"
    if [[ $rc -ne 0 && "$mode" != "judgement" ]]; then
        halt "operator remote returned '$result' for $label $mode (exit $rc)"
    fi
    case "$mode" in
        judgement)
            case "$result" in
                yes) printf 'y' ;;
                no|halt) printf 'n' ;;
                *) halt "operator remote gave no usable judgement for $label (got '$result')" ;;
            esac
            ;;
    esac
    return 0
}

assert_no_contention() {
    # Runs BEFORE the stage's own run directory exists, so the evidence lands in
    # the session directory under the stage's name. One file per stage: a single
    # overwritten contention.json would let stage H inherit stage B's evidence.
    local stage="${1:-unknown}"
    local out="$SESSION/contention_${stage}.json"
    mkdir -p "$(dirname "$out")"
    # The probe exits 1 when it FINDS a competitor, which is evidence, not an
    # error, so its status is captured rather than gating. A probe that wrote
    # no evidence at all is a hard stop: evaluating a missing file, or a stale
    # one from an earlier stage, is how a gate goes green without looking.
    rm -f "$out"
    local probe_rc=0
    python3 -m phoenix.sim2real.hw_probe contention --out "$out" || probe_rc=$?
    [[ -f "$out" ]] || halt "contention probe wrote no evidence to $out (exit $probe_rc); \
it needs ROS 2 and systemctl on the payload"
    python3 - "$out" <<'CONTENTION_EOF' || halt "another system is sharing the robot; stop it and re-run this stage"
import json
import sys

from phoenix.sim2real.preflight_eval import contention_checks

probe = json.load(open(sys.argv[1]))
checks = contention_checks(probe)
bad = [c for c in checks if c.gating and not c.ok]
for c in checks:
    print(f"[contention] {'ok  ' if c.ok else 'FAIL'} {c.name}: {c.detail}")
sys.exit(1 if bad else 0)
CONTENTION_EOF
}

assert_battery() {
    # Every motors-live stage refuses to start unless the GO2 battery is ABOVE the floor
    # (operator rule 2026-10-06: above 20 %). Fails closed on no reading or a 0 % reading.
    local stage="${1:-unknown}"
    local out="$SESSION/battery_${stage}.json"
    mkdir -p "$(dirname "$out")"
    rm -f "$out"
    python3 -m phoenix.sim2real.hw_probe battery --min-soc "${PHOENIX_MIN_BATTERY_SOC:-20}" --out "$out" \
        || halt "battery is not above ${PHOENIX_MIN_BATTERY_SOC:-20} % (evidence: $out); charge the robot"
}

stage_A() {
    local rc=0
    if [[ "${1:-}" == "--payload" ]]; then
        need_sha
        [[ -n "${PHOENIX_BUNDLE:-}" ]] || halt "A --payload requires PHOENIX_BUNDLE (the activated bundle directory)"
        pf A --payload --bundle "$PHOENIX_BUNDLE" || rc=$?
    else
        pf A || rc=$?
    fi
    return "$rc"
}

stage_B() {
    need_sha; ros_env
    assert_no_contention B
    pf require B
    new_run B
    local rc=0
    bash "$REPO_ROOT/scripts/dryrun_pipeline.sh" \
        --config "$DEPLOY_CFG" --lock "$DEPLOY_LOCK" --run-dir "$RUN" \
        --expect-sha "$EXPECT" --duration "${1:-30}" || rc=$?
    local verdict_rc=0
    pf evaluate B --run-dir "$RUN" || verdict_rc=$?
    [[ $rc -eq 0 ]] || halt "dryrun pipeline exited $rc (evidence recorded above)"
    return "$verdict_rc"
}

stage_C() {
    need_sha; ros_env
    assert_no_contention C
    pf require C
    new_run C
    echo "[C] motors stay OFF: no lowcmd bridge is started in this stage."
    # shellcheck disable=SC2046
    launch deadman $(deadman_cmd)
    sleep 1
    require_alive deadman
    python3 -m phoenix.sim2real.hw_probe deadman \
        --estop-timeout-s "$(cfg_get safety.estop_timeout_s)" --out "$RUN/deadman_trace.json"
    require_alive deadman
    stop_proc deadman
    pf evaluate C --trace "$RUN/deadman_trace.json"
}

stage_D() {
    need_sha; ros_env
    assert_no_contention D
    pf require D
    new_run D
    launch lowstate_bridge python3 -m phoenix.sim2real.lowstate_bridge_node
    sleep 1
    require_alive lowstate_bridge
    python3 -m phoenix.sim2real.hw_probe rates --duration "${1:-10}" \
        --topics /lowstate /joint_states /imu/data --out "$RUN/probe.json"
    require_alive lowstate_bridge
    stop_proc lowstate_bridge
    pf evaluate D --probe "$RUN/probe.json"
}

stage_E() {
    local hold_s="${1:-10}"
    need_sha; ros_env
    assert_no_contention E
    assert_battery E
    pf require E
    live_confirm E "LowCmd bridge LIVE, holding measured posture for ${hold_s} s, no policy"
    new_run E
    local telemetry="$RUN/bridge.jsonl"
    # shellcheck disable=SC2046
    launch deadman $(deadman_cmd)
    launch lowstate_bridge python3 -m phoenix.sim2real.lowstate_bridge_node
    sleep 1
    require_alive deadman lowstate_bridge
    # shellcheck disable=SC2046
    launch lowcmd_bridge python3 -m phoenix.sim2real.lowcmd_bridge_node --live $(bridge_args E "$telemetry")
    wait_for_telemetry "$telemetry" 't.get("mode") == "hold" and t.get("deadman_source_ok")' \
        "$(cfg_get safety.first_message_timeout_s)" "bridge holding with the real deadman armed"
    require_alive deadman lowstate_bridge lowcmd_bridge
    local i
    for i in $(seq "$hold_s"); do sleep 1; require_alive deadman lowstate_bridge lowcmd_bridge; done
    echo "[E] hold complete; stopping the bridge (it sits the robot down on the way out)"
    stop_proc lowcmd_bridge
    stop_all
    pf evaluate E --telemetry "$telemetry" --duration "$hold_s"
}

stage_stand() {
    local stage="$1" authority attempts
    case "$stage" in F) authority=2; attempts=1 ;; G) authority=5; attempts=1 ;; H) authority=10; attempts=3 ;; esac
    need_sha; ros_env
    assert_no_contention "$stage"
    assert_battery "$stage"
    pf require "$stage"
    live_confirm "$stage" "${DEPLOY_LABEL} cmd=0 stand, ${authority} s of policy authority, ${attempts} attempt(s)"
    local first_msg max_rt telemetry_files=() answers=() k
    local standup_s="${PHOENIX_STANDUP_S:-2.0}"
    first_msg="$(cfg_get safety.first_message_timeout_s)"
    max_rt="$(python3 -c "print(float('$first_msg') + $authority)")"
    for k in $(seq "$attempts"); do
        new_run "${stage}${k}"
        local telemetry="$RUN/bridge.jsonl"
        if [[ "${PHOENIX_REHEARSAL:-0}" != "1" ]]; then
            operator_gate arm "${stage}${k}" \
                "[${stage}${k}] Robot folded on the mat, feet on the ground, spotter ready, deadman HELD. Press Enter to start: "
        fi
        # shellcheck disable=SC2046
        launch deadman $(deadman_cmd)
        launch lowstate_bridge python3 -m phoenix.sim2real.lowstate_bridge_node
        sleep 1
        require_alive deadman lowstate_bridge
        # shellcheck disable=SC2046
        # Stand-up ramp (2026-09-21): a folded start cannot rise under the policy's
        # measured-q slew clip and is outside its training distribution, so the
        # bridge first ramps to the training stance and the policy starts from there.
        launch lowcmd_bridge python3 -m phoenix.sim2real.lowcmd_bridge_node --live $(bridge_args "$stage" "$telemetry") \
            --standup-s "$standup_s"
        wait_for_telemetry "$telemetry" 't.get("mode") in ("hold", "standup") and t.get("deadman_source_ok")' \
            "$first_msg" "bridge armed with the real deadman"
        wait_for_telemetry "$telemetry" 't.get("standup_done")' \
            "$(python3 -c "print($standup_s + $first_msg)")" "stand-up ramp complete, holding the training stance"
        require_alive deadman lowstate_bridge lowcmd_bridge
        launch policy_node python3 -m phoenix.sim2real.ros2_policy_node --config "$DEPLOY_CFG" \
            --lock "$DEPLOY_LOCK" --authority-s "$authority" --max-runtime-s "$max_rt" \
            --log-parquet "$RUN/policy.parquet"
        # The window ends with the policy's single abort notice, which the bridge
        # latches as a fault; any other fault ends it too. A timeout is recorded,
        # not fatal here: the processes are still stopped (the bridge sits the robot down) and the
        # evaluator records the NO-GO.
        local wait_rc=0
        (wait_for_telemetry "$telemetry" 't.get("fault") is not None' \
            "$(python3 -c "print($max_rt + 2)")" "policy window end or a fault") || wait_rc=$?
        [[ $wait_rc -eq 0 ]] || c_red "[${stage}${k}] no end-of-window or fault seen in time; ending the attempt"
        if [[ $wait_rc -eq 0 ]]; then
            # The window end is latched as a policy abort, so the bridge runs its stop path:
            # hold for up to 1 s, then a controlled sit, while the deadman is still held.
            echo "[${stage}${k}] policy authority ended; the bridge holds for up to 1 s, then sits the robot down on its own."
        fi
        local answer="n"
        if [[ "${PHOENIX_REHEARSAL:-0}" == "1" ]]; then
            answer="${PHOENIX_REHEARSAL_STOOD:-n}"
        else
            answer="$(operator_gate judgement "${stage}${k}" \
                "[${stage}${k}] Did the robot stand on its feet and hold, with no collapse, oscillation or buzz? [y/n]: ")"
            operator_gate release "${stage}${k}" \
                "[${stage}${k}] Press Enter to end the attempt (the bridge sits the robot if it is not already sitting, folds the legs and exits): "
        fi
        stop_proc policy_node
        stop_proc lowcmd_bridge
        stop_all
        telemetry_files+=("$telemetry")
        answers+=("$answer")
    done
    pf evaluate "$stage" --telemetry "${telemetry_files[@]}" --operator-confirmed "${answers[@]}"
}

# W <preset>: one live walking attempt. NOT a ledger stage: there is no evaluator and
# nothing here is ever counted; it records evidence (bridge.jsonl, walk_cmd.jsonl,
# policy.parquet, the operator's answer) and prints a summary. The bridge records
# walking_allowed from the evidence-gated deploy contract; the policy node refuses to
# start a walking config whose contract (including F/G hardware evidence) fails.
# Rehearsal needs synthetic F/G records: see the help text for W.
stage_walk() {
    local preset="${1:-}" walk_dur authority margin
    [[ -n "$preset" ]] || halt "usage: harness_preflight.sh W <preset> (presets: python3 -m phoenix.sim2real.walk_command --list)"
    walk_dur="$(python3 -c 'import sys; from phoenix.sim2real.walk_command import PRESETS, profile_duration; print(profile_duration(PRESETS[sys.argv[1]]))' "$preset" 2>/dev/null)" \
        || halt "unknown walk preset '$preset' (python3 -m phoenix.sim2real.walk_command --list)"
    need_sha; ros_env
    assert_no_contention W
    assert_battery W
    local stand_only=""
    stand_only="$(cfg_get safety.stand_only 2>/dev/null)" || stand_only="unreadable"
    [[ "$stand_only" == "False" ]] \
        || halt "W needs a walking config: safety.stand_only must be explicitly false in $DEPLOY_CFG (set DEPLOY_CFG and DEPLOY_LOCK to the walking config)"
    echo "[W] walking config $DEPLOY_CFG (lock $DEPLOY_LOCK, ${DEPLOY_LABEL}). The bridge decides walking_allowed itself from the evidence-gated deploy contract (F and G hardware GO records for the stand-phase twin); this script does not."
    # The first walks are gantry-only (operator rule 2026-10-06): the phrase itself says so.
    live_confirm "W $preset ON GANTRY" "${DEPLOY_LABEL} WALKING attempt '$preset' with the robot ON THE GANTRY, profile ${walk_dur} s, deadman must stay held, 1 attempt"
    local first_msg max_rt telemetry standup_s="${PHOENIX_STANDUP_S:-2.0}"
    # Motion must never outlive the policy window (its end makes the bridge sit the robot).
    # Window end = profile_duration + margin = 5 + motion_s + margin after policy launch.
    # Worst-case motion end = walk_cmd start lag (<= 1.0 s, launched right after the policy)
    # + start deadline 3.0 s (a later deadman never starts: late_start) + lead-in 2.0 s
    # + motion_s + ramp-down (<= 1.0 s, vx 0.5) = 7 + motion_s. Requiring >= 2.0 s of zero
    # command before the window ends gives margin >= 9 - 5 = 4.0, hence the default.
    margin="${PHOENIX_WALK_AUTHORITY_MARGIN_S:-4.0}"
    first_msg="$(cfg_get safety.first_message_timeout_s)"
    authority="$(python3 -c "print($walk_dur + $margin)")"
    max_rt="$(python3 -c "print(float('$first_msg') + $authority)")"
    new_run "W_${preset}"
    telemetry="$RUN/bridge.jsonl"
    if [[ "${PHOENIX_REHEARSAL:-0}" != "1" ]]; then
        operator_gate arm "W_${preset}" \
            "[W $preset] Robot folded on the mat, feet on the ground, spotter ready, clear 2 m radius, deadman HELD (keep holding it for the whole walk). Press Enter to start: "
    fi
    # shellcheck disable=SC2046
    launch deadman $(deadman_cmd)
    launch lowstate_bridge python3 -m phoenix.sim2real.lowstate_bridge_node
    sleep 1
    require_alive deadman lowstate_bridge
    # --stage is a free-form label in lowcmd_bridge_node (default "unlabelled", never validated), so "W" is accepted as is.
    # shellcheck disable=SC2046
    launch lowcmd_bridge python3 -m phoenix.sim2real.lowcmd_bridge_node --live $(bridge_args W "$telemetry") \
        --standup-s "$standup_s"
    wait_for_telemetry "$telemetry" 't.get("mode") in ("hold", "standup") and t.get("deadman_source_ok")' \
        "$first_msg" "bridge armed with the real deadman"
    wait_for_telemetry "$telemetry" 't.get("standup_done")' \
        "$(python3 -c "print($standup_s + $first_msg)")" "stand-up ramp complete, holding the training stance"
    require_alive deadman lowstate_bridge lowcmd_bridge
    launch policy_node python3 -m phoenix.sim2real.ros2_policy_node --config "$DEPLOY_CFG" \
        --lock "$DEPLOY_LOCK" --authority-s "$authority" --max-runtime-s "$max_rt" \
        --log-parquet "$RUN/policy.parquet"
    # The authority window counts from the policy's first commanding tick, not from node
    # launch, and the policy node starts far slower than walk_command on the payload. Start
    # the profile only once the bridge is in policy mode, so the 2 s zero lead-in is really
    # fed to the policy and the margin above is measured from the same origin as the window.
    wait_for_telemetry "$telemetry" 't.get("mode") == "policy"' \
        "$first_msg" "policy in control before the walking profile starts"
    require_alive deadman lowstate_bridge lowcmd_bridge policy_node
    launch walk_cmd python3 -m phoenix.sim2real.walk_command --preset "$preset" --start-deadline-s 3.0 \
        --out "$RUN/walk_cmd.jsonl"
    local wait_rc=0
    (wait_for_telemetry "$telemetry" 't.get("fault") is not None' \
        "$(python3 -c "print($max_rt + 2)")" "policy window end or a fault") || wait_rc=$?
    [[ $wait_rc -eq 0 ]] || c_red "[W $preset] no end-of-window or fault seen in time; ending the attempt"
    # Let walk_command finish its own zero hold (bounded); stop_proc interrupts it if it is still running.
    for _ in $(seq 60); do is_alive "${PIDS[walk_cmd]}" || break; sleep 0.1; done
    echo "[W $preset] policy authority ended; the bridge holds for up to 1 s, then sits the robot down on its own."
    local answer="n"
    if [[ "${PHOENIX_REHEARSAL:-0}" == "1" ]]; then
        answer="${PHOENIX_REHEARSAL_WALKED:-n}"
    else
        answer="$(operator_gate judgement "W_${preset}" \
            "[W $preset] Did the robot walk the preset (start, move, stop) on its feet, with no collapse, oscillation, buzz or drift outside the spotter's reach? [y/n]: ")"
        operator_gate release "W_${preset}" \
            "[W $preset] Press Enter to end the attempt (the bridge sits the robot if it is not already sitting, folds the legs and exits): "
    fi
    stop_proc walk_cmd
    stop_proc policy_node
    stop_proc lowcmd_bridge
    stop_all
    printf '%s\n' "$answer" >"$RUN/operator_answer.txt"
    echo "[W $preset] NOT a ledger stage: no evaluator ran and this evidence is never counted. Operator answer: $answer"
    python3 - "$RUN/walk_cmd.jsonl" "$telemetry" <<'WALK_SUMMARY_EOF'
import json
import sys

summary = None
try:
    with open(sys.argv[1]) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record") == "summary":
                summary = rec
except FileNotFoundError:
    pass
faults = []
try:
    with open(sys.argv[2]) as fh:
        for line in fh:
            try:
                tick = json.loads(line)
            except json.JSONDecodeError:
                continue
            fault = tick.get("fault")
            if tick.get("record") == "tick" and fault is not None and fault not in faults:
                faults.append(fault)
except FileNotFoundError:
    pass
if summary is None:
    print("[W] walk_cmd: NO summary record (walk_command did not finish cleanly or never ran)")
else:
    vx, _vy, wz = summary["max_abs"]
    print(
        f"[W] walk_cmd: outcome={summary['outcome']} published={summary['n_published']} "
        f"max|vx|={vx:.3f} max|wz|={wz:.3f}"
    )
print(f"[W] bridge faults: {faults if faults else 'none'}")
sys.exit(0 if summary is not None and summary["outcome"] == "complete" else 1)
WALK_SUMMARY_EOF
}

# FP [joint ...]: active joint-friction probe on a HUNG robot (feet off the ground, released).
# NOT a ledger stage: no evaluator, evidence never counted. The bridge holds measured posture
# (stand-only config, no --standup-s, so there is no kp 60 -> 25 handover blend: its gains step
# from the hold gains to --kp/--kd at the first command) and friction_probe_node becomes the
# command source in place of the policy node, on the same topic and wire. Every bridge gate applies.
stage_fp() {
    local joints=("$@") fp_module="phoenix.sim2real.friction_probe"
    need_sha; ros_env
    assert_no_contention FP
    assert_battery FP
    local stand_only=""
    stand_only="$(cfg_get safety.stand_only 2>/dev/null)" || stand_only="unreadable"
    [[ "$stand_only" == "True" ]] \
        || halt "FP needs a STAND-ONLY config: safety.stand_only must be true in $DEPLOY_CFG (set DEPLOY_CFG and DEPLOY_LOCK to the walk-stand bundle)"
    local kp kd
    kp="$(python3 -c 'import importlib, sys; print(importlib.import_module(sys.argv[1]).KP)' "$fp_module" 2>/dev/null)" \
        || halt "cannot import KP from $fp_module (friction_probe protocol module missing?)"
    kd="$(python3 -c 'import importlib, sys; print(importlib.import_module(sys.argv[1]).KD)' "$fp_module" 2>/dev/null)" \
        || halt "cannot import KD from $fp_module"
    live_confirm "FP HUNG FEET OFF GROUND" \
        "friction probe on ${DEPLOY_LABEL}, joints: ${joints[*]:-all 12}, bridge kp=$kp kd=$kd, 1 attempt" hung
    local first_msg
    first_msg="$(cfg_get safety.first_message_timeout_s)"
    new_run FP
    local telemetry="$RUN/bridge.jsonl" trace="$RUN/fp_trace.jsonl" est_rc=0 sum_rc=0 node_rc=0
    if [[ "${PHOENIX_REHEARSAL:-0}" != "1" ]]; then
        operator_gate arm FP \
            "[FP] Robot HUNG, feet off the ground, released, spotter ready, deadman HELD (keep holding it for the whole probe). Press Enter to start: "
    fi
    # shellcheck disable=SC2046
    launch deadman $(deadman_cmd)
    launch lowstate_bridge python3 -m phoenix.sim2real.lowstate_bridge_node
    sleep 1
    require_alive deadman lowstate_bridge
    # No --standup-s: the bridge holds measured posture, like stage E.
    # shellcheck disable=SC2046
    launch lowcmd_bridge python3 -m phoenix.sim2real.lowcmd_bridge_node --live $(bridge_args FP "$telemetry") \
        --kp "$kp" --kd "$kd"
    wait_for_telemetry "$telemetry" 't.get("mode") == "hold" and t.get("deadman_source_ok")' \
        "$first_msg" "bridge holding with the real deadman armed"
    require_alive deadman lowstate_bridge lowcmd_bridge
    local node_args=(--config "$DEPLOY_CFG" --out "$trace")
    [[ ${#joints[@]} -gt 0 ]] && node_args+=(--joints "${joints[@]}")
    launch friction_probe python3 -m phoenix.sim2real.friction_probe_node "${node_args[@]}"
    local deadline=$((SECONDS + ${PHOENIX_FP_MAX_WAIT_S:-600}))
    while is_alive "${PIDS[friction_probe]}"; do
        require_alive deadman lowstate_bridge lowcmd_bridge
        sleep 0.5
        [[ $SECONDS -lt $deadline ]] || { c_red "[FP] probe node still running at the harness cap; stopping it"; break; }
    done
    if is_alive "${PIDS[friction_probe]}"; then
        stop_proc friction_probe
        node_rc=124
    else
        wait "${PIDS[friction_probe]}" 2>/dev/null || node_rc=$?
        unset "PIDS[friction_probe]"
    fi
    echo "[FP] probe node exit code $node_rc (0 complete, 2 aborted, 3 refused, 124 harness cap)"
    echo "[FP] the bridge latches the node's end-of-run notice, holds, then sits the robot down on its own."
    if [[ "${PHOENIX_REHEARSAL:-0}" != "1" ]]; then
        operator_gate release FP \
            "[FP] Press Enter to end the run (the bridge sits the robot if it is not already sitting, folds the legs and exits): "
    fi
    stop_proc lowcmd_bridge
    stop_all
    python3 - "$trace" "$telemetry" <<'FP_SUMMARY_EOF' || sum_rc=$?
import json
import sys

summary = None
try:
    with open(sys.argv[1]) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record") == "summary":
                summary = rec
except FileNotFoundError:
    pass
faults = []
modes = []
try:
    with open(sys.argv[2]) as fh:
        for line in fh:
            try:
                tick = json.loads(line)
            except json.JSONDecodeError:
                continue
            if tick.get("record") != "tick":
                continue
            if tick.get("mode") not in modes:
                modes.append(tick.get("mode"))
            fault = tick.get("fault")
            if fault is not None and fault not in faults:
                faults.append(fault)
except FileNotFoundError:
    pass
if summary is None:
    print("[FP] node: NO summary record (it did not finish cleanly or never ran)")
else:
    print(
        f"[FP] node: outcome={summary['outcome']} reason={summary['reason']} "
        f"ticks={summary['n_ticks']}"
    )
print(f"[FP] bridge modes seen: {modes}; faults: {faults if faults else 'none'}")
sys.exit(0 if summary is not None and summary["outcome"] == "complete" else 1)
FP_SUMMARY_EOF
    if [[ -f "${trace%.jsonl}.npz" ]]; then
        python3 -m phoenix.sim2real.friction_probe_node estimate --trace "${trace%.jsonl}.npz" --out "$RUN/fp_estimate.json" || est_rc=$?
        [[ $est_rc -eq 0 ]] || c_red "[FP] estimator exited $est_rc (trace kept at $trace)"
    else
        c_red "[FP] no .npz trace written (node never published?); nothing to estimate"
    fi
    echo "[FP] NOT a ledger stage: no evaluator ran and this evidence is never counted."
    return "$sum_rc"
}

# ----------------------------------------------------------------------- main
cmd="${1:-ready}"
[[ $# -gt 0 ]] && shift
case "$cmd" in
    ready)
        rc=0
        stage_A || rc=$?
        pf status || rc=$?
        exit "$rc"
        ;;
    status) pf status ;;
    A) stage_A "$@" ;;
    B) stage_B "$@" ;;
    C) stage_C ;;
    D) stage_D "$@" ;;
    E) stage_E "$@" ;;
    F|G|H) stage_stand "$cmd" ;;
    W) stage_walk "$@" ;;
    FP) stage_fp "$@" ;;
    -h|--help) sed -n '2,40p' "$0" ;;
    *) halt "unknown command '$cmd' (see --help)" ;;
esac
