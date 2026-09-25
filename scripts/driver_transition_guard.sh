#!/usr/bin/env bash
# One-shot: seed 42 is running under a driver process that loaded
# run_velocity_clip100_seeds.sh into memory BEFORE the --curriculum off fix
# was committed. Editing the script on disk does not change what that
# already-running instance does for its remaining seeds (bash reads a running
# for-loop as one unit), so left alone it would launch seed 43 with the OLD
# flags (curriculum ON) immediately after seed 42 exits.
#
# Empirically verified on this host (dummy dry run, 2026-09-25) before this
# script touched anything real:
#   - killing the driver's bash process ALSO kills its already-spawned
#     pipeline children (everything in a tmux pane shares one process group
#     here; plain "kill <driver_pid>" is not a safe way to leave a live child
#     running). So this script must NEVER act while seed 42 is still running.
#   - after seed 42 exits naturally, a newly-spawned stale seed 43 python
#     process CAN be caught and killed within ~10-50ms of it starting (well
#     before Isaac Sim/Kit finishes initializing, so no real training or
#     checkpoint happens under the wrong config).
#   - killing that stale child tends to tear down the whole stale tmux
#     session as a side effect (same shared-process-group behavior), which is
#     fine here: nothing valuable is left running under it once seed 42 is
#     done and the stale seed 43/44 attempt is killed.
#
# Strategy: wait for seed 42's exit line (never touch anything before that).
# Then tightly poll for a newly-spawned stale seed 43 or seed 44 python
# process and kill it the instant it appears. Clean up any checkpoint
# directory a stale attempt managed to create (identified by the ABSENCE of
# arm_label.txt, which the fixed train_velocity.py always writes) so the
# watchdog's relaunch of the fixed driver starts that seed fresh instead of
# "resuming" from a wrong-config checkpoint. Finally pkill the stale driver
# process itself (harmless no-op if it is already gone) so it cannot reach a
# premature postrun pass.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRIVER_LOG="logs/run_velocity_clip100_driver.log"
WATCHDOG_LOG="logs/watchdog.log"
STALE_SEEDS=(43 44)
CATCH_POLL_S=0.1
CATCH_POLLS=600         # 600 * 0.1s = 60s max watch window per stale seed

glog() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [driver-transition-guard] $*" >> "$WATCHDOG_LOG"
}

# 1. Wait for seed 42's OWN exit line. Nothing below this point may run before
#    this is observed true -- seed 42 must never be touched.
until grep -q "seed 42 training exited with code" "$DRIVER_LOG" 2>/dev/null; do
    sleep 2
done
glog "seed 42 finished, watching for the stale driver to spawn seed 43/44 with the pre-fix flags"

# 2. For each stale seed the old driver would launch next, tightly poll for
#    its python process and kill it the instant it appears.
for SEED in "${STALE_SEEDS[@]}"; do
    PATTERN="scripts/train_velocity.py.*walk_v1_clip100_seed${SEED}"
    CAUGHT=0
    POLL_I=0
    while [[ "$POLL_I" -lt "$CATCH_POLLS" ]]; do
        PID="$(pgrep -f "$PATTERN" | head -n1)"
        if [[ -n "$PID" ]]; then
            glog "caught stale seed ${SEED} python pid ${PID}, killing it"
            kill "$PID" 2>/dev/null
            sleep 1
            if kill -0 "$PID" 2>/dev/null; then
                glog "seed ${SEED} pid ${PID} still alive after SIGTERM, sending SIGKILL"
                kill -9 "$PID" 2>/dev/null
            fi
            CAUGHT=1
            break
        fi
        sleep "$CATCH_POLL_S"
        POLL_I=$((POLL_I + 1))
    done
    if [[ "$CAUGHT" == "0" ]]; then
        glog "never saw a stale seed ${SEED} process within the watch window (stale driver likely already gone, e.g. its pane died with seed 43 -- fine)"
    fi
done

# 3. Clean up any checkpoint dir a stale attempt created before being killed.
#    arm_label.txt is written by the FIXED train_velocity.py as one of its
#    first actions; its absence means the run_root was touched by the stale
#    (pre-fix) invocation and must not be treated as resumable progress.
for SEED in "${STALE_SEEDS[@]}"; do
    RUN_ROOT="checkpoints/velocity/walk_v1_clip100_seed${SEED}"
    if [[ -d "$RUN_ROOT" && ! -f "${RUN_ROOT}/arm_label.txt" ]]; then
        STAMP="$(date +%Y%m%d_%H%M%S)"
        MOVED="${RUN_ROOT}_stale_${STAMP}"
        mv "$RUN_ROOT" "$MOVED"
        glog "moved tainted (no arm_label.txt) checkpoint dir ${RUN_ROOT} -> ${MOVED}"
    fi
done

# 4. Belt and suspenders: kill the stale driver process itself if it is
#    somehow still alive (e.g. it did not reach seed 43 fast enough for step 2
#    to find anything, and is still sitting in its for-loop). No-op if gone.
if pkill -f "run_velocity_clip100_seeds.sh" 2>/dev/null; then
    glog "stale driver process was still alive, killed it"
fi

glog "done, exiting; watchdog should relaunch the fixed driver within its poll interval"
