#!/usr/bin/env bash
# Self-healing watchdog for the overnight clip_actions=100 run.
#
# Every ~3 minutes: if the driver (scripts/run_velocity_clip100_seeds.sh) is
# not running AND the chain has not finished, relaunch it. The driver itself
# is resume-aware (skips seeds already at the iteration target, resumes a
# partial seed from its highest checkpoint with --resume-continue-iterations),
# so relaunching it is always safe and idempotent -- this script's only job is
# noticing the driver died and starting it again, capped at 3 restarts per
# seed so a genuinely broken config does not loop forever.
#
# Singleton via flock: run this script itself (it re-execs under flock), do
# not source it.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOCK_FILE="logs/watchdog.lock"
LOG_FILE="logs/watchdog.log"
DRIVER_LOG="logs/run_velocity_clip100_driver.log"
POLL_INTERVAL_S=180
MAX_RESTARTS_PER_SEED=3
SEEDS=(42 43 44)
MAX_ITER=9000

mkdir -p logs

# Re-exec under flock -n so only one watchdog instance ever runs. If another
# instance already holds the lock, this exits immediately (0), which is the
# correct outcome for "make sure only one exists".
if [[ "${WATCHDOG_LOCKED:-0}" != "1" ]]; then
    exec env WATCHDOG_LOCKED=1 flock -n "$LOCK_FILE" "$0" "$@"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

chain_complete() {
    [[ -f "$DRIVER_LOG" ]] && grep -q "\[run_velocity_clip100_seeds\] DONE\." "$DRIVER_LOG"
}

driver_running() {
    pgrep -f "run_velocity_clip100_seeds.sh" >/dev/null 2>&1 \
        || pgrep -f "scripts/train_velocity.py.*walk_v1_clip100" >/dev/null 2>&1 \
        || pgrep -f "scripts/postrun_velocity.py" >/dev/null 2>&1
}

gpu_compute_apps_present() {
    local n
    n="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')"
    [[ "${n:-0}" -gt 0 ]]
}

latest_checkpoint_iter() {
    local run_root="$1"
    find "$run_root" -maxdepth 2 -name 'model_*.pt' 2>/dev/null \
        | sed -E 's#.*/model_([0-9]+)\.pt#\1#' \
        | sort -n \
        | tail -n1
}

first_incomplete_seed() {
    for SEED in "${SEEDS[@]}"; do
        local run_root="checkpoints/velocity/walk_v1_clip100_seed${SEED}"
        local done_iter
        done_iter="$(latest_checkpoint_iter "$run_root")"
        done_iter="${done_iter:-0}"
        if [[ "$done_iter" -lt "$MAX_ITER" ]]; then
            echo "$SEED"
            return 0
        fi
    done
    echo ""
}

restart_count_file() {
    echo "logs/watchdog_restart_count_seed${1}"
}

restart_count() {
    local f
    f="$(restart_count_file "$1")"
    [[ -f "$f" ]] && cat "$f" || echo 0
}

bump_restart_count() {
    local f n
    f="$(restart_count_file "$1")"
    n="$(restart_count "$1")"
    echo "$((n + 1))" > "$f"
}

log "watchdog armed, pid $$, polling every ${POLL_INTERVAL_S}s, cap ${MAX_RESTARTS_PER_SEED} restarts/seed"

while true; do
    if chain_complete; then
        log "chain complete (DONE. marker found in $DRIVER_LOG), watchdog exiting"
        exit 0
    fi

    if driver_running; then
        : # normal: driver or a training/postrun process is alive, nothing to do
    else
        if gpu_compute_apps_present; then
            log "driver not running but GPU still shows compute apps, skipping this cycle (letting it settle)"
        else
            SEED="$(first_incomplete_seed)"
            if [[ -z "$SEED" ]]; then
                # All seeds at target but chain not marked DONE: postrun likely
                # died. The driver script re-running the (now all-skip) seed
                # loop then re-running postrun is the correct recovery.
                log "all seeds at target iteration but chain not DONE; assuming postrun died, relaunching driver"
                COUNT=0
                SEED_LABEL="postrun"
            else
                COUNT="$(restart_count "$SEED")"
                SEED_LABEL="seed${SEED}"
                if [[ "$COUNT" -ge "$MAX_RESTARTS_PER_SEED" ]]; then
                    log "ALERT: seed ${SEED} has hit the ${MAX_RESTARTS_PER_SEED}-restart cap, NOT relaunching. Manual intervention needed."
                    log "watchdog exiting (will not retry this seed again)"
                    exit 1
                fi
                bump_restart_count "$SEED"
                COUNT="$((COUNT + 1))"
            fi
            log "driver not running, chain incomplete (${SEED_LABEL}), restart ${COUNT}/${MAX_RESTARTS_PER_SEED}: relaunching driver"
            # Empirically (2026-09-24/25) the WHOLE tmux session died with the
            # trainer, not just its window, twice tonight -- so the visible
            # window may need a fresh session, not just a fresh window in a
            # session that no longer exists.
            if tmux has-session -t phoenix-walk 2>/dev/null; then
                tmux new-window -t phoenix-walk -n main \
                    "cd '$REPO_ROOT' && ./scripts/run_velocity_clip100_seeds.sh 2>&1 | tee -a '$DRIVER_LOG'" \
                    2>>"$LOG_FILE" \
                    || log "WARNING: tmux new-window failed"
            else
                tmux new-session -d -s phoenix-walk -n main \
                    "cd '$REPO_ROOT' && ./scripts/run_velocity_clip100_seeds.sh 2>&1 | tee -a '$DRIVER_LOG'" \
                    2>>"$LOG_FILE" \
                    || log "WARNING: tmux new-session failed"
                log "phoenix-walk session was gone, created a fresh one"
            fi
        fi
    fi

    sleep "$POLL_INTERVAL_S"
done
