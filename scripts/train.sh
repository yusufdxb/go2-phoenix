#!/usr/bin/env bash
# Train the baseline PPO policy on Isaac Lab.
#
# Usage:
#   ./scripts/train.sh [config=configs/train/ppo.yaml] [--headless] [--num_envs N]
#
# Requires Isaac Lab. Set ISAACLAB_PATH to point at your IsaacLab install;
# defaults to ~/IsaacLab.

set -euo pipefail

CONFIG="${1:-configs/train/ppo.yaml}"
shift || true

ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/IsaacLab}"
if [[ ! -x "$ISAACLAB_PATH/isaaclab.sh" ]]; then
    echo "[train.sh] error: IsaacLab not found at $ISAACLAB_PATH" >&2
    echo "            set ISAACLAB_PATH=/path/to/IsaacLab" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/_activate.sh"

echo "[train.sh] Repo root       : $REPO_ROOT"
echo "[train.sh] Isaac Lab       : $ISAACLAB_PATH"
echo "[train.sh] Config          : $CONFIG"
echo "[train.sh] Extra args      : $*"

cd "$REPO_ROOT"

PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
"$ISAACLAB_PATH/isaaclab.sh" -p \
    -m phoenix.training.ppo_runner \
    --config "$CONFIG" \
    "$@"

