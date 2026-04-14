#!/usr/bin/env bash
# Fine-tune a baseline policy using failure-weighted curriculum.
#
# Usage:
#   ./scripts/adapt.sh [config=configs/train/adaptation.yaml]

set -euo pipefail

CONFIG="${1:-configs/train/adaptation.yaml}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/IsaacLab}"
PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
"$ISAACLAB_PATH/isaaclab.sh" -p -m phoenix.adaptation.fine_tune \
    --config "$CONFIG"
