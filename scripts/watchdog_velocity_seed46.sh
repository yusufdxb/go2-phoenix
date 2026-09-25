#!/usr/bin/env bash
# Self-healing watchdog for seed 46 ONLY (2026-09-25: seed 45 abandoned by
# choice). No pkill or kill anywhere: only detects (pgrep, read-only) and
# relaunches the resume-aware, idempotent driver.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOCK_FILE="logs/watchdog_seed46.lock"
LOG_FILE="logs/watchdog_seed46.log"
DRIVER_LOG="logs/run_velocity_seed46_driver.log"
POLL_INTERVAL_S=180
MAX_RESTARTS=3
RUN_ROOT="checkpoints/velocity/walk_v2_seed46"
MAX_ITER=9000

mkdir -p logs

if [[ "${WATCHDOG_LOCKED:-0}" != "1" ]]; then
    exec env WATCHDOG_LOCKED=1 flock -n "$LOCK_FILE" "$0" "$@"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

chain_complete() {
    [[ -f "$DRIVER_LOG" ]] && grep -q "\[run_velocity_seed46_only\] DONE\." "$DRIVER_LOG"
}

driver_running() {
    pgrep -f "run_velocity_seed46_only.sh" >/dev/null 2>&1 \
        || pgrep -f "scripts/train_velocity.py.*walk_v2_seed46" >/dev/null 2>&1 \
        || pgrep -f "scripts/postrun_velocity_v2.py" >/dev/null 2>&1
}

gpu_compute_apps_present() {
    local n
    n="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')"
    [[ "${n:-0}" -gt 0 ]]
}

latest_checkpoint_iter() {
    find "$RUN_ROOT" -maxdepth 2 -name 'model_*.pt' 2>/dev/null \
        | sed -E 's#.*/model_([0-9]+)\.pt#\1#' \
        | sort -n \
        | tail -n1
}

restart_count_file() { echo "logs/watchdog_seed46_restart_count"; }
restart_count() {
    local f
    f="$(restart_count_file)"
    [[ -f "$f" ]] && cat "$f" || echo 0
}
bump_restart_count() {
    local f n
    f="$(restart_count_file)"
    n="$(restart_count)"
    echo "$((n + 1))" > "$f"
}

log "watchdog seed46 armed, pid $$, polling every ${POLL_INTERVAL_S}s, cap ${MAX_RESTARTS} restarts"

while true; do
    if chain_complete; then
        log "chain complete (DONE. marker found in $DRIVER_LOG), watchdog exiting"
        exit 0
    fi

    if driver_running; then
        :
    else
        if gpu_compute_apps_present; then
            log "driver not running but GPU still shows compute apps, skipping this cycle"
        else
            DONE_ITER="$(latest_checkpoint_iter)"
            DONE_ITER="${DONE_ITER:-0}"
            if [[ "$DONE_ITER" -ge "$MAX_ITER" ]]; then
                log "seed 46 at target iteration but chain not DONE; assuming postrun died, relaunching driver"
                COUNT=0
            else
                COUNT="$(restart_count)"
                if [[ "$COUNT" -ge "$MAX_RESTARTS" ]]; then
                    log "ALERT: seed 46 has hit the ${MAX_RESTARTS}-restart cap, NOT relaunching. Manual intervention needed."
                    exit 1
                fi
                bump_restart_count
                COUNT="$((COUNT + 1))"
            fi
            log "driver not running, chain incomplete, restart ${COUNT}/${MAX_RESTARTS}: relaunching driver"
            if tmux has-session -t phoenix-walk 2>/dev/null; then
                tmux new-window -t phoenix-walk -n main \
                    "cd '$REPO_ROOT' && ./scripts/run_velocity_seed46_only.sh 2>&1 | tee -a '$DRIVER_LOG'" \
                    2>>"$LOG_FILE" \
                    || log "WARNING: tmux new-window failed"
            else
                tmux new-session -d -s phoenix-walk -n main \
                    "cd '$REPO_ROOT' && ./scripts/run_velocity_seed46_only.sh 2>&1 | tee -a '$DRIVER_LOG'" \
                    2>>"$LOG_FILE" \
                    || log "WARNING: tmux new-session failed"
                log "phoenix-walk session was gone, created a fresh one"
            fi
        fi
    fi

    sleep "$POLL_INTERVAL_S"
done
