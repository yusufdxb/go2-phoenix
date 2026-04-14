#!/usr/bin/env bash
# Reconstruct a real-world failure trajectory inside Isaac Sim.
#
# Usage:
#   ./scripts/replay.sh <trajectory.parquet> [--variations N] [--headless]

set -euo pipefail

TRAJ="${1:?trajectory parquet path required}"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/_activate.sh"
cd "$REPO_ROOT"

ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/IsaacLab}"
PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
"$ISAACLAB_PATH/isaaclab.sh" -p -m phoenix.replay.reconstruct \
    --trajectory "$TRAJ" \
    --variations-config configs/replay/variations.yaml \
    "$@"
