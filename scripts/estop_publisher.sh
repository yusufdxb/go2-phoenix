#!/usr/bin/env bash
# Deadman's-switch publisher for the Phoenix policy node.
#
# Publishes `std_msgs/msg/Bool data: false` on /phoenix/estop at 10 Hz.
# The policy node refuses to start unless a publisher on /phoenix/estop is
# already live; this script satisfies that requirement.
#
# Flip to true on abort: Ctrl-C this script (it stops publishing False), or
# publish true from a second terminal with:
#     ros2 topic pub --once /phoenix/estop std_msgs/msg/Bool "{data: true}"
#
# Usage (run BEFORE launching the policy node):
#     source /opt/ros/humble/setup.bash
#     ./scripts/estop_publisher.sh
#
# Exit code:
#     0  on clean shutdown
#     1  if ROS 2 isn't sourced

set -euo pipefail

if ! command -v ros2 >/dev/null 2>&1; then
    echo "ERROR: ros2 not on PATH. Source /opt/ros/<distro>/setup.bash first." >&2
    exit 1
fi

TOPIC="${PHOENIX_ESTOP_TOPIC:-/phoenix/estop}"
RATE_HZ="${PHOENIX_ESTOP_RATE:-10}"

echo "[estop] publishing false on ${TOPIC} at ${RATE_HZ} Hz (Ctrl-C to stop)"
exec ros2 topic pub -r "${RATE_HZ}" "${TOPIC}" std_msgs/msg/Bool "{data: false}"
