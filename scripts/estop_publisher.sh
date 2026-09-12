#!/usr/bin/env bash
# SYNTHETIC estop heartbeat for NO-MOTOR DRYRUNS ONLY. THIS IS NOT A DEADMAN.
#
# Publishes `std_msgs/Bool data: false` on /phoenix/estop at 10 Hz. Nothing a human
# does can make it publish True except killing it, so it proves nothing about the
# physical deadman. It exists so the no-motor dryrun (scripts/dryrun_pipeline.sh,
# stage B) can exercise the policy node and the DRY bridge.
#
# It can never arm a live run: the LowCmd bridge in --live mode requires the
# /phoenix/estop publisher set to be exactly one of the real deadman nodes
# (phoenix.sim2real.wireless_estop_node or deadman_joy_node), and `ros2 topic pub`
# is neither. For any live stage use the real deadman.
#
# It refuses to start unless PHOENIX_DRYRUN_ONLY=1 is set, so it cannot be launched
# by habit from an old runbook.
#
# Usage (dryrun only):
#     PHOENIX_DRYRUN_ONLY=1 ./scripts/estop_publisher.sh

set -euo pipefail

if [[ "${PHOENIX_DRYRUN_ONLY:-0}" != "1" ]]; then
    echo "REFUSING: scripts/estop_publisher.sh is a synthetic heartbeat, NOT A DEADMAN." >&2
    echo "  Set PHOENIX_DRYRUN_ONLY=1 for a no-motor dryrun. For anything live, run" >&2
    echo "  python3 -m phoenix.sim2real.wireless_estop_node (or deadman_joy_node)." >&2
    exit 1
fi

if ! command -v ros2 >/dev/null 2>&1; then
    echo "ERROR: ros2 not on PATH. Source /opt/ros/<distro>/setup.bash first." >&2
    exit 1
fi

TOPIC="${PHOENIX_ESTOP_TOPIC:-/phoenix/estop}"
RATE_HZ="${PHOENIX_ESTOP_RATE:-10}"

echo "[estop] SYNTHETIC DRYRUN HEARTBEAT, NOT A DEADMAN: publishing false on ${TOPIC} at ${RATE_HZ} Hz"
exec ros2 topic pub -r "${RATE_HZ}" "${TOPIC}" std_msgs/msg/Bool "{data: false}"
