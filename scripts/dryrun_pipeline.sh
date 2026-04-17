#!/usr/bin/env bash
# Phoenix dry-run pipeline harness.
#
# Brings up, in order:
#   1. estop heartbeat (scripts/estop_publisher.sh)
#   2. lowstate_bridge_node     (/lowstate → /joint_states + /imu/data)
#   3. lowcmd_bridge_node       DRY mode (publishes /lowcmd_dry only)
#   4. ros2_policy_node          --log-parquet
#
# Runs for $1 seconds (default 30), captures topic samples mid-run, then
# tears down cleanly. All per-stage logs land under /tmp/dryrun_*.log;
# samples land under /tmp/dryrun_samples.log; parquet path is printed.
#
# This script never touches /lowcmd. The lowcmd_bridge default
# publishes to /lowcmd_dry; the policy node only commands the bridge.
# To go live, run lowcmd_bridge separately with --live AFTER the GO2
# is in low-level mode and you have a real deadman publisher up.

DURATION_S="${1:-30}"
TS="$(date +%Y-%m-%d_%H-%M-%S)"
PARQUET="data/failures/dryrun_${TS}.parquet"
mkdir -p data/failures

source /opt/ros/humble/setup.bash
source ~/unitree_ros2/cyclonedds_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/unitree/unitree_ros2/cyclonedds_ws/src/cyclonedds.xml
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

cleanup() {
  echo "[$(date +%T)] cleanup"
  for P in $POLICY_PID $LCB_PID $LSB_PID $ESTOP_PID; do
    [ -n "$P" ] && kill -INT "$P" 2>/dev/null
  done
  sleep 2
  for P in $POLICY_PID $LCB_PID $LSB_PID $ESTOP_PID; do
    [ -n "$P" ] && kill -TERM "$P" 2>/dev/null
  done
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "[$(date +%T)] stage 1: estop heartbeat"
nohup bash scripts/estop_publisher.sh > /tmp/dryrun_estop.log 2>&1 &
ESTOP_PID=$!
sleep 1

echo "[$(date +%T)] stage 2: lowstate_bridge"
nohup python3 -m phoenix.sim2real.lowstate_bridge_node > /tmp/dryrun_lsb.log 2>&1 &
LSB_PID=$!
sleep 2

echo "[$(date +%T)] stage 3: lowcmd_bridge (DRY → /lowcmd_dry)"
nohup python3 -m phoenix.sim2real.lowcmd_bridge_node > /tmp/dryrun_lcb.log 2>&1 &
LCB_PID=$!
sleep 2

echo "[$(date +%T)] stage 4: policy node (parquet=${PARQUET})"
# Resolve the ONNX path from deploy.yaml so the harness stays in sync if
# the deployed policy switches (e.g. phoenix-base ↔ phoenix-flat).
ONNX_PATH=$(python3 -c "
import sys, yaml
with open('configs/sim2real/deploy.yaml') as fh:
    cfg = yaml.safe_load(fh)
print(cfg['policy']['onnx_path'])
")
nohup python3 -m phoenix.sim2real.ros2_policy_node \
    --config      configs/sim2real/deploy.yaml \
    --onnx        "${ONNX_PATH}" \
    --log-parquet "${PARQUET}" > /tmp/dryrun_policy.log 2>&1 &
POLICY_PID=$!
sleep 5

echo "[$(date +%T)] all stages up. estop=$ESTOP_PID lsb=$LSB_PID lcb=$LCB_PID policy=$POLICY_PID"

echo "[$(date +%T)] mid-run sampling..."
{
  # All Phoenix topics use BEST_EFFORT QoS (matches ros2_policy_node.py).
  # ros2 topic hz defaults to RELIABLE which is incompatible — override
  # reliability (Humble doesn't accept --qos-profile sensor_data here).
  echo "=== /joint_group_position_controller/command hz ==="
  timeout 4 ros2 topic hz --qos-reliability best_effort /joint_group_position_controller/command 2>&1 | tail -2
  echo "=== /lowcmd_dry hz ==="
  timeout 4 ros2 topic hz --qos-reliability best_effort /lowcmd_dry 2>&1 | tail -2
  echo "=== /joint_states hz ==="
  timeout 4 ros2 topic hz --qos-reliability best_effort /joint_states 2>&1 | tail -2
  echo "=== /imu/data hz ==="
  timeout 4 ros2 topic hz --qos-reliability best_effort /imu/data 2>&1 | tail -2
} > /tmp/dryrun_samples.log 2>&1

echo "[$(date +%T)] samples captured"
REMAINING=$((DURATION_S - 18))
[ $REMAINING -lt 0 ] && REMAINING=0
sleep $REMAINING
echo "[$(date +%T)] done"
echo "PARQUET_PATH=${PARQUET}"
