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
#   scripts/harness_preflight.sh F | G | H    LIVE H25 cmd=0 stand, 2 s / 5 s / 10 s x 3   motors LIVE
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
    for name in policy_node lowcmd_bridge lowstate_bridge deadman; do
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
    local stage="$1" what="$2"
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
    echo "[E] hold complete; stopping the bridge (it damps on the way out)"
    stop_proc lowcmd_bridge
    stop_all
    pf evaluate E --telemetry "$telemetry" --duration "$hold_s"
}

stage_stand() {
    local stage="$1" authority attempts
    case "$stage" in F) authority=2; attempts=1 ;; G) authority=5; attempts=1 ;; H) authority=10; attempts=3 ;; esac
    need_sha; ros_env
    pf require "$stage"
    live_confirm "$stage" "H25 cmd=0 stand, ${authority} s of policy authority, ${attempts} attempt(s)"
    local first_msg max_rt telemetry_files=() answers=() k
    first_msg="$(cfg_get safety.first_message_timeout_s)"
    max_rt="$(python3 -c "print(float('$first_msg') + $authority)")"
    for k in $(seq "$attempts"); do
        new_run "${stage}${k}"
        local telemetry="$RUN/bridge.jsonl"
        if [[ "${PHOENIX_REHEARSAL:-0}" != "1" ]]; then
            read -r -p "[${stage}${k}] Robot folded on the mat, feet on the ground, spotter ready, deadman HELD. Press Enter to start: " _
        fi
        # shellcheck disable=SC2046
        launch deadman $(deadman_cmd)
        launch lowstate_bridge python3 -m phoenix.sim2real.lowstate_bridge_node
        sleep 1
        require_alive deadman lowstate_bridge
        # shellcheck disable=SC2046
        launch lowcmd_bridge python3 -m phoenix.sim2real.lowcmd_bridge_node --live $(bridge_args "$stage" "$telemetry")
        wait_for_telemetry "$telemetry" 't.get("mode") == "hold" and t.get("deadman_source_ok")' \
            "$first_msg" "bridge holding with the real deadman armed"
        require_alive deadman lowstate_bridge lowcmd_bridge
        launch policy_node python3 -m phoenix.sim2real.ros2_policy_node --config "$DEPLOY_CFG" \
            --lock "$DEPLOY_LOCK" --authority-s "$authority" --max-runtime-s "$max_rt" \
            --log-parquet "$RUN/policy.parquet"
        # The window ends with the policy's single abort notice, which the bridge
        # latches as a fault; any other fault ends it too. A timeout is recorded,
        # not fatal here: the processes are still stopped (the bridge damps) and the
        # evaluator records the NO-GO.
        local wait_rc=0
        (wait_for_telemetry "$telemetry" 't.get("fault") is not None' \
            "$(python3 -c "print($max_rt + 2)")" "policy window end or a fault") || wait_rc=$?
        [[ $wait_rc -eq 0 ]] || c_red "[${stage}${k}] no end-of-window or fault seen in time; ending the attempt"
        echo "[${stage}${k}] policy authority ended; bridge is HOLDING measured posture."
        local answer="n"
        if [[ "${PHOENIX_REHEARSAL:-0}" == "1" ]]; then
            answer="${PHOENIX_REHEARSAL_STOOD:-n}"
        else
            read -r -p "[${stage}${k}] Did the robot stand on its feet and hold, with no collapse, oscillation or buzz? [y/n]: " answer
            read -r -p "[${stage}${k}] Spotter ready: press Enter to DAMP (the robot sinks onto the mat) and end the attempt: " _
        fi
        stop_proc policy_node
        stop_proc lowcmd_bridge
        stop_all
        telemetry_files+=("$telemetry")
        answers+=("$answer")
    done
    pf evaluate "$stage" --telemetry "${telemetry_files[@]}" --operator-confirmed "${answers[@]}"
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
    -h|--help) sed -n '2,33p' "$0" ;;
    *) halt "unknown command '$cmd' (see --help)" ;;
esac
