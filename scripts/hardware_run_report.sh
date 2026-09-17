#!/usr/bin/env bash
# Post-run report for one GO2 hardware session, from recorded evidence only.
#
#     scripts/hardware_run_report.sh <session_dir> [--settle-window-s 1.0]
#
# Offline: reads stage_*.json and */bridge.jsonl that the session already wrote,
# computes nothing during the control loop, and writes
# <session_dir>/hardware_run_report.json next to the printed text.
#
# Exits non-zero unless the result is PASS, so it can gate a session wrap-up.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[[ $# -ge 1 ]] || { echo "usage: $0 <session_dir> [options]" >&2; exit 2; }
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m phoenix.sim2real.hardware_run_report "$@"
