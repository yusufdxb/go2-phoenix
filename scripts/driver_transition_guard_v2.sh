#!/usr/bin/env bash
# One-shot: seed 45 is running under a driver process that loaded
# run_velocity_v2_seeds.sh into memory BEFORE seed 46 was changed from
# curriculum=off to curriculum=on + --joint-pos-limits-weight -10.0. Editing
# the script on disk does not change what that already-running instance does
# for seed 46 (bash reads a running for-loop as one unit).
#
# SAFETY RULES (from the 2026-09-25 incident where an earlier version of this
# exact script's own dummy-test verification killed the real production
# driver with an unscoped pkill -f):
#   1. Never act on anything until seed 45's OWN exit line is independently
#      confirmed in the log. Seed 45 is never touched, before or after.
#   2. No pkill -f, ever. Only kill ONE PID discovered by a narrow pgrep
#      match, re-verified with `ps -p PID -o cmd=` immediately before the
#      kill to confirm it is actually the expected stale process.
#   3. Proven on a dummy setup, with a UNIQUE marker string in the dummy
#      command line that cannot collide with anything real, before this
#      script is ever pointed at real paths.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRIVER_LOG="${DRIVER_LOG_OVERRIDE:-logs/run_velocity_v2_driver.log}"
WATCHDOG_LOG="${WATCHDOG_LOG_OVERRIDE:-logs/watchdog_v2.log}"
STALE_SEED_PATTERN="${STALE_SEED_PATTERN_OVERRIDE:-scripts/train_velocity.py.*walk_v2_seed46}"
EXIT_LINE_PATTERN="${EXIT_LINE_PATTERN_OVERRIDE:-seed 45 training exited with code}"
RUN_ROOT="${RUN_ROOT_OVERRIDE:-checkpoints/velocity/walk_v2_seed46}"
EXPECTED_ARM_LABEL="${EXPECTED_ARM_LABEL_OVERRIDE:-v2-curriculum-limitpen}"
CATCH_POLL_S=0.1
CATCH_POLLS=600

glog() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [driver-transition-guard-v2] $*" >> "$WATCHDOG_LOG"
}

# 1. Wait for seed 45's OWN exit line. Nothing below this point may run
#    before this is observed true.
until grep -q "$EXIT_LINE_PATTERN" "$DRIVER_LOG" 2>/dev/null; do
    sleep 2
done
glog "seed 45 finished, watching for a stale seed 46 (pattern: ${STALE_SEED_PATTERN})"

CAUGHT=0
POLL_I=0
while [[ "$POLL_I" -lt "$CATCH_POLLS" ]]; do
    PID="$(pgrep -f "$STALE_SEED_PATTERN" | head -n1)"
    if [[ -n "$PID" ]]; then
        CMDLINE="$(ps -p "$PID" -o cmd= 2>/dev/null || true)"
        if [[ "$CMDLINE" == *"$STALE_SEED_PATTERN"* ]] || echo "$CMDLINE" | grep -qE "$STALE_SEED_PATTERN"; then
            glog "caught stale seed 46 python pid ${PID} (cmd: ${CMDLINE:0:200}), killing it"
            kill "$PID" 2>/dev/null
            sleep 1
            if kill -0 "$PID" 2>/dev/null; then
                glog "pid ${PID} still alive after SIGTERM, sending SIGKILL"
                kill -9 "$PID" 2>/dev/null
            fi
            CAUGHT=1
        else
            glog "WARNING: pgrep matched pid ${PID} but its cmdline did not re-verify (cmd: ${CMDLINE:0:200}), NOT killing it"
        fi
        break
    fi
    sleep "$CATCH_POLL_S"
    POLL_I=$((POLL_I + 1))
done
if [[ "$CAUGHT" == "0" ]]; then
    glog "never caught a stale seed 46 process (driver may already be gone, e.g. its pane died with it, which is fine)"
fi

if [[ -d "$RUN_ROOT" ]]; then
    LABEL_FILE="${RUN_ROOT}/arm_label.txt"
    CURRENT_LABEL=""
    [[ -f "$LABEL_FILE" ]] && CURRENT_LABEL="$(cat "$LABEL_FILE")"
    if [[ -z "$CURRENT_LABEL" || "$CURRENT_LABEL" != "$EXPECTED_ARM_LABEL" ]]; then
        STAMP="$(date +%Y%m%d_%H%M%S)"
        MOVED="${RUN_ROOT}_stale_${STAMP}"
        mv "$RUN_ROOT" "$MOVED"
        glog "moved tainted checkpoint dir ${RUN_ROOT} (arm_label='${CURRENT_LABEL}', expected '${EXPECTED_ARM_LABEL}') -> ${MOVED}"
    else
        glog "${RUN_ROOT} already has the expected arm_label '${EXPECTED_ARM_LABEL}', leaving it"
    fi
fi

glog "done, exiting; watchdog should relaunch the fixed driver within its poll interval"
