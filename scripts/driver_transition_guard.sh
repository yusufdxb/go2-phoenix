#!/usr/bin/env bash
# One-shot: wait for seed 42 (running under the OLD, already-in-memory driver
# process) to finish, then kill that stale process so the watchdog notices and
# relaunches using the CURRENT on-disk driver script (with --curriculum off
# for seeds 43/44). Needed because bash reads a running script's for-loop as a
# single unit at start, so editing run_velocity_clip100_seeds.sh on disk does
# not change what an already-running instance does for its remaining seeds.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRIVER_LOG="logs/run_velocity_clip100_driver.log"
WATCHDOG_LOG="logs/watchdog.log"

until grep -q "seed 42 training exited with code" "$DRIVER_LOG" 2>/dev/null; do
    sleep 10
done

echo "$(date '+%Y-%m-%d %H:%M:%S') [driver-transition-guard] seed 42 finished, killing stale driver process (loaded pre-curriculum-fix script) so the watchdog relaunches the FIXED script for seed 43/44" >> "$WATCHDOG_LOG"
pkill -f "run_velocity_clip100_seeds.sh" 2>/dev/null
echo "$(date '+%Y-%m-%d %H:%M:%S') [driver-transition-guard] done, exiting" >> "$WATCHDOG_LOG"
