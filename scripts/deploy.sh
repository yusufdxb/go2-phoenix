#!/usr/bin/env bash
# Export a trained checkpoint to ONNX and run the canonical-stand bench.
#
# This script does NOT bring up the robot. The ROS 2 deploy stack is a
# multi-node graph (real deadman + lowstate_bridge + lowcmd_bridge +
# policy node) that must be started deliberately on the Jetson, with the
# GO2 in low-level mode. Launching the policy node alone, with no bridges
# and no real deadman, is unsafe and useless. The bringup sequence is
# printed at the end; scripts/dryrun_pipeline.sh runs the no-motors path.
#
# Usage:
#   ./scripts/deploy.sh <checkpoint.pt> [config=configs/sim2real/deploy.yaml]

set -euo pipefail

CKPT="${1:?checkpoint path required}"
CONFIG="${2:-configs/sim2real/deploy.yaml}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/_activate.sh"
cd "$REPO_ROOT"

# --- 1) Export ONNX (must run in Isaac Lab python context so torch is present)
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/Sim/IsaacLab}"
PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
"$ISAACLAB_PATH/isaaclab.sh" -p -m phoenix.sim2real.export \
    --checkpoint "$CKPT" \
    --output "${CKPT%.*}.onnx" \
    --verify

# --- 2) Canonical-stand bench: fail fast on an under-trained export
#        Caught the 2026-04-16 under-trained flat-v0 export in <1s.
PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
python3 -m phoenix.sim2real.bench_export \
    --onnx       "${CKPT%.*}.onnx" \
    --deploy-cfg "$CONFIG"

# --- 3) Hand-off. The ROS 2 stack is brought up deliberately, not here:
#        a lone policy node with no bridges and no real deadman is unsafe.
cat <<EOF

ONNX exported and bench-passed: ${CKPT%.*}.onnx

Bring the deploy stack up on the Jetson (NOT from this script):
  * No-motors dry run:  ./scripts/dryrun_pipeline.sh
  * Live bringup, in this order:
      1. real deadman:  python3 -m phoenix.sim2real.deadman_joy_node
         (or wireless_estop_node). scripts/estop_publisher.sh is a bare
         heartbeat, NOT a deadman; do not use it for a live run.
      2. python3 -m phoenix.sim2real.lowstate_bridge_node
      3. python3 -m phoenix.sim2real.lowcmd_bridge_node --live
         (only after the GO2 is in low-level mode)
      4. python3 -m phoenix.sim2real.ros2_policy_node --config ${CONFIG} --onnx ${CKPT%.*}.onnx
EOF
