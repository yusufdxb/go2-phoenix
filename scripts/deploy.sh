#!/usr/bin/env bash
# Export a trained checkpoint to ONNX and launch the ROS 2 policy node.
#
# Usage:
#   ./scripts/deploy.sh <checkpoint.pt> [config=configs/sim2real/deploy.yaml]

set -euo pipefail

CKPT="${1:?checkpoint path required}"
CONFIG="${2:-configs/sim2real/deploy.yaml}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- 1) Export ONNX (must run in Isaac Lab python context so torch is present)
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/IsaacLab}"
PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
"$ISAACLAB_PATH/isaaclab.sh" -p -m phoenix.sim2real.export \
    --checkpoint "$CKPT" \
    --output "${CKPT%.*}.onnx" \
    --verify

# --- 2) Launch ROS 2 policy node (system python + rclpy)
if [[ -z "${ROS_DISTRO:-}" ]]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
fi

PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
python3 -m phoenix.sim2real.ros2_policy_node \
    --config "$CONFIG" \
    --onnx "${CKPT%.*}.onnx"
