#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/_activate.sh"
PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" python3 -m phoenix.training.evaluate \
    --checkpoint checkpoints/phoenix-stand-v3/latest.pt \
    --env-config configs/env/stand.yaml \
    --num-envs 16 \
    --num-episodes 32 \
    --metrics-out docs/pre_lab_stand_v3_rollout_2026-04-21.json \
    "$@"
