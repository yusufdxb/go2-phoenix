#!/usr/bin/env bash
# Phoenix no-motor dryrun: the exact locked deploy config, every process checked.
#
# Brings up, in order, each logging into the run directory:
#   1. synthetic estop heartbeat   scripts/estop_publisher.sh (DRYRUN ONLY, not a deadman)
#   2. lowstate_bridge_node        /lowstate -> /joint_states + /imu/data
#   3. lowcmd_bridge_node          DRY: /lowcmd_dry only, --lock, telemetry to bridge.jsonl
#   4. ros2_policy_node            --config and --lock, never --onnx
#
# Then it waits for the bridge's first policy tick, checks every process is still
# alive, probes topic rates, publishers and content, runs out the duration while
# re-checking liveness every second, checks liveness again, and tears down. It writes
# processes.json and probe.json next to bridge.jsonl; scripts/harness_preflight.sh B
# turns them into the stage B verdict. Exit status is non-zero if any process died
# or policy mode was never reached. Nothing in here can publish /lowcmd.
#
# Usage:
#   scripts/dryrun_pipeline.sh --config CFG --lock LOCK --run-dir DIR --expect-sha SHA \
#                              [--duration SECONDS] [--probe-s SECONDS]
# The run directory must not exist yet: evidence is never overwritten.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="" LOCK="" RUN="" EXPECT="" DURATION=30 PROBE_S=10
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --lock) LOCK="$2"; shift 2 ;;
        --run-dir) RUN="$2"; shift 2 ;;
        --expect-sha) EXPECT="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --probe-s) PROBE_S="$2"; shift 2 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done

halt() { printf '\033[31mHALT: %s\033[0m\n' "$*" >&2; exit 1; }

[[ -n "$CONFIG" && -f "$CONFIG" ]] || halt "--config must name an existing deploy config"
[[ -n "$LOCK" && -f "$LOCK" ]] || halt "--lock must name an existing lock file"
[[ -n "$EXPECT" ]] || halt "--expect-sha is required"
[[ -n "$RUN" ]] || halt "--run-dir is required"
[[ -e "$RUN" && -n "$(ls -A "$RUN" 2>/dev/null)" ]] && halt "run directory $RUN is not empty"
mkdir -p "$RUN"
command -v ros2 >/dev/null || halt "ros2 not on PATH"
python3 -c 'import rclpy, unitree_go.msg' || halt "rclpy / unitree_go messages not importable"

FIRST_MSG="$(python3 -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["safety"]["first_message_timeout_s"])' "$CONFIG")"
(( $(python3 -c "print(int($DURATION > $PROBE_S))") )) || halt "--duration must exceed --probe-s"

declare -A PIDS=()
is_alive() { [[ -r "/proc/$1/status" ]] && ! grep -q '^State:[[:space:]]*Z' "/proc/$1/status"; }

launch() {
    local name="$1"; shift
    "$@" >"$RUN/$name.log" 2>&1 &
    PIDS[$name]=$!
    echo "[$(date +%T)] launched $name (pid ${PIDS[$name]}): $*"
}

snapshot() {
    # snapshot <phase>: record liveness of every launched process into processes.json
    local phase="$1" name args=()
    for name in "${!PIDS[@]}"; do
        if is_alive "${PIDS[$name]}"; then args+=("$name=1"); else args+=("$name=0"); fi
    done
    python3 - "$RUN/processes.json" "$phase" "${args[@]}" <<'PY'
import json, os, sys
path, phase, pairs = sys.argv[1], sys.argv[2], sys.argv[3:]
data = json.load(open(path)) if os.path.exists(path) else {}
data[phase] = {p.split("=")[0]: p.split("=")[1] == "1" for p in pairs}
json.dump(data, open(path, "w"), indent=2)
PY
}

all_alive() {
    local name
    for name in "${!PIDS[@]}"; do is_alive "${PIDS[$name]}" || { echo "DIED: $name (log $RUN/$name.log)" >&2; return 1; }; done
}

teardown() {
    local name pid
    for name in policy_node lowcmd_bridge lowstate_bridge estop_heartbeat; do
        pid="${PIDS[$name]:-}"
        [[ -n "$pid" ]] || continue
        if is_alive "$pid"; then
            kill -INT "$pid" 2>/dev/null || true
            for _ in $(seq 50); do is_alive "$pid" || break; sleep 0.1; done
            is_alive "$pid" && kill -TERM "$pid" 2>/dev/null
        fi
        wait "$pid" 2>/dev/null || true   # reap only; liveness was already recorded
    done
}
trap teardown EXIT

echo "[$(date +%T)] config $CONFIG  lock $LOCK  run $RUN"
launch estop_heartbeat env PHOENIX_DRYRUN_ONLY=1 bash "$REPO_ROOT/scripts/estop_publisher.sh"
launch lowstate_bridge python3 -m phoenix.sim2real.lowstate_bridge_node
launch lowcmd_bridge python3 -m phoenix.sim2real.lowcmd_bridge_node \
    --config "$CONFIG" --lock "$LOCK" --telemetry "$RUN/bridge.jsonl" --stage B --expect-sha "$EXPECT"
launch policy_node python3 -m phoenix.sim2real.ros2_policy_node \
    --config "$CONFIG" --lock "$LOCK" --log-parquet "$RUN/policy.parquet"

# Wait for the bridge to reach policy mode, bounded by the config's own startup window.
python3 - "$RUN/bridge.jsonl" "$FIRST_MSG" <<'PY' || { snapshot after_startup; halt "bridge never reached policy mode within first_message_timeout_s"; }
import json, sys, time
path, deadline = sys.argv[1], time.monotonic() + float(sys.argv[2]) + 5.0
while time.monotonic() < deadline:
    try:
        for line in open(path):
            try:
                if json.loads(line).get("mode") == "policy":
                    sys.exit(0)
            except json.JSONDecodeError:
                pass
    except FileNotFoundError:
        pass
    time.sleep(0.2)
sys.exit(1)
PY
snapshot after_startup
all_alive || halt "a process died during startup"

python3 -m phoenix.sim2real.hw_probe rates --duration "$PROBE_S" --out "$RUN/probe.json"
all_alive || { snapshot before_teardown; halt "a process died during the probe"; }

remaining="$(python3 -c "print(max(0, int(round($DURATION - $PROBE_S))))")"
for _ in $(seq "$remaining"); do
    sleep 1
    all_alive || { snapshot before_teardown; halt "a process died during the run"; }
done
snapshot before_teardown
all_alive || halt "a process died before teardown"
echo "[$(date +%T)] dryrun complete; evidence in $RUN"
