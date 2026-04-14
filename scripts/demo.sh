#!/usr/bin/env bash
# Run the full demo pipeline: evaluate baseline + adapted policies,
# record rollouts, compose a side-by-side video.
#
# Usage:
#   ./scripts/demo.sh <baseline.pt> <adapted.pt> <real_clip.mp4>
#
# Any missing MP4 will be substituted with a placeholder.

set -euo pipefail

BASELINE="${1:-checkpoints/phoenix-base/latest.pt}"
ADAPTED="${2:-checkpoints/phoenix-adapt/latest.pt}"
REAL_CLIP="${3:-media/real_placeholder.mp4}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/_activate.sh"
cd "$REPO_ROOT"

mkdir -p media/renders

ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/IsaacLab}"
PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
"$ISAACLAB_PATH/isaaclab.sh" -p -m phoenix.demo.benchmark \
    --baseline "$BASELINE" \
    --adapted "$ADAPTED" \
    --render-dir media/renders

PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
python3 -m phoenix.demo.video_compose \
    --sim media/renders/sim_baseline.mp4 \
    --real "$REAL_CLIP" \
    --improved media/renders/sim_adapted.mp4 \
    --out media/side_by_side.mp4 \
    --labels "SIM (baseline)" "REAL (deployment)" "SIM (after Phoenix loop)"

echo "[demo.sh] Done -> media/side_by_side.mp4"
