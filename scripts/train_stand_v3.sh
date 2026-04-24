#!/usr/bin/env bash
# Train phoenix-stand-v3 (Gate 7 retry after 2026-04-21).
#
# What this is:
#   Fine-tunes stand-v2 with 4x stronger action_rate penalty and 5x
#   stronger joint_acc penalty (see configs/env/stand_v3.yaml). Attacks
#   the out-of-distribution large-action failure mode observed on
#   hardware (FL_thigh 99.8% sat at mean|a|=0.90 on the stand). Does
#   NOT address the rear-thigh posture offset — a separate v4 pass
#   (stand-on-stand DR) or a floor-testing pivot is needed to take sat
#   below 5%.
#
# Prereqs:
#   - Isaac Sim venv at $ISAAC_VENV (default ~/isaac-sim-venv)
#   - IsaacLab at $ISAACLAB_PATH (default ~/IsaacLab)
#   - RTX-class GPU with ≥12 GB VRAM (mewtwo 5070 or lab-PC 5080 both fit)
#   - Repo checked out at audit-fixes-2026-04-16 (3c52b05 or newer)
#
# Usage (from the repo root):
#   ./scripts/train_stand_v3.sh               # default: 4096 envs, headless
#   ./scripts/train_stand_v3.sh --num_envs 8192
#
# Resume source (hardcoded intentionally — this is a v2→v3 fine-tune,
# not a from-scratch run): checkpoints/phoenix-stand-v2/2026-04-19_11-20-36/model_499.pt

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="configs/train/ppo_stand_v3.yaml"
RESUME="checkpoints/phoenix-stand-v2/2026-04-19_11-20-36/model_499.pt"

if [[ ! -f "$REPO_ROOT/$RESUME" ]]; then
    echo "[train_stand_v3] error: resume checkpoint missing at $RESUME" >&2
    echo "                  stand-v2 artifacts must be synced from T7 before training." >&2
    exit 1
fi

if [[ ! -f "$REPO_ROOT/$CONFIG" ]]; then
    echo "[train_stand_v3] error: config missing at $CONFIG" >&2
    exit 1
fi

# Delegate to the existing train.sh so we inherit its _activate.sh +
# IsaacLab invocation. Pass --resume and --headless by default.
exec "$REPO_ROOT/scripts/train.sh" "$CONFIG" --resume "$RESUME" --headless "$@"
