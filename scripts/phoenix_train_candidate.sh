#!/usr/bin/env bash
# TRAIN: fine-tune one experiment arm from the incumbent checkpoint, one seed.
#
# Usage:
#   scripts/phoenix_train_candidate.sh <arm> <env-overlay.yaml> <incumbent.pt> <seed>
#
# <arm> is a label only (continued | broad | phoenix | oracle); every arm uses
# configs/train/phoenix_finetune.yaml, so the env overlay is the ONLY difference
# between arms (same iterations, envs, PPO settings, warm start). Needs Isaac Lab
# and a free GPU; nothing in this script has been run for Phoenix v2 yet.
set -euo pipefail

ARM="${1:?arm}"
ENV_OVERLAY="${2:?env overlay yaml}"
INCUMBENT="${3:?incumbent checkpoint .pt}"
SEED="${4:?seed}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$ENV_OVERLAY" ]] || { echo "no overlay $ENV_OVERLAY" >&2; exit 2; }
[[ -f "$INCUMBENT" ]] || { echo "no checkpoint $INCUMBENT" >&2; exit 2; }

exec "$REPO_ROOT/scripts/train.sh" "$REPO_ROOT/configs/train/phoenix_finetune.yaml" \
    --env-config "$ENV_OVERLAY" \
    --resume "$INCUMBENT" \
    --seed "$SEED" \
    --run-name "phoenix-v2-${ARM}-s${SEED}"
