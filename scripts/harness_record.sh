#!/usr/bin/env bash
# harness_record.sh
# Wraps in-session logging: rosbag2 record + optional USB camera capture +
# a run manifest tagging block + scenario. One command per scenario.
#
# Usage:
#   bash scripts/harness_record.sh <block> <scenario> [duration_s]
#
# Examples:
#   bash scripts/harness_record.sh gate7 floor_hold_1
#   bash scripts/harness_record.sh gate8 g8a_step_0p3 60
#   bash scripts/harness_record.sh block3 slip_towel_1 30
#
# Output:
#   data/lab/<YYYY-MM-DD>/<block>_<scenario>_<ts>/
#     bag/                  rosbag2 store (sqlite3)
#     camera.mp4            ffmpeg capture from first /dev/video* (if present)
#     manifest.json         block, scenario, ts, topic list, camera device
#     record.log            combined stdout from all child processes
#
# Self-test:
#   bash scripts/harness_record.sh --mock gate7 selftest 3
#     skips ros2 + ffmpeg, writes a manifest only.

set -euo pipefail

MOCK=0
if [[ "${1:-}" == "--mock" ]]; then
    MOCK=1
    shift
fi

BLOCK="${1:?block label required (gate7|gate8|block3)}"
SCENARIO="${2:?scenario label required}"
DURATION_S="${3:-60}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATE_DIR=$(date +%Y-%m-%d)
TS=$(date +%H%M%S)
RUN_DIR="data/lab/${DATE_DIR}/${BLOCK}_${SCENARIO}_${TS}"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/record.log"

# Topic set. Justification (see LAB_CARD_NEXT.md "topic capture rationale"):
#   /lowstate                        : raw firmware state (full fidelity)
#   /lowcmd, /lowcmd_dry             : motor command (both live + dry modes)
#   /joint_states                    : lowstate_bridge_node publish
#   /imu/data                        : lowstate_bridge_node publish
#   /cmd_vel                         : operator command (Twist)
#   /phoenix/estop                   : deadman latch
#   /joint_group_position_controller/command : policy_node publish
# Optional (record if present at start):
#   /phoenix/policy_state            : (NOT currently published by policy_node;
#                                       reserved for future state-machine telemetry)
TOPICS=(
    /lowstate
    /lowcmd
    /lowcmd_dry
    /joint_states
    /imu/data
    /cmd_vel
    /phoenix/estop
    /joint_group_position_controller/command
)

# ---- helpers ----------------------------------------------------------------
require_bin() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: missing required binary: $1 ${2:-}" >&2
        exit 1
    fi
}

# ---- detect camera ----------------------------------------------------------
detect_camera() {
    if ! command -v v4l2-ctl >/dev/null 2>&1; then
        echo ""
        return
    fi
    local dev
    dev=$(v4l2-ctl --list-devices 2>/dev/null | awk '/\/dev\/video[0-9]+/{print $1; exit}')
    echo "${dev:-}"
}

# ---- binary checks ----------------------------------------------------------
if [[ $MOCK -eq 0 ]]; then
    require_bin ros2 "source /opt/ros/humble/setup.bash"
fi

CAMERA_DEV=""
if [[ $MOCK -eq 0 ]]; then
    CAMERA_DEV=$(detect_camera || true)
    if [[ -n "$CAMERA_DEV" ]] && ! command -v ffmpeg >/dev/null 2>&1; then
        echo "WARN: camera $CAMERA_DEV detected but ffmpeg missing. Skipping video." >&2
        CAMERA_DEV=""
    fi
fi

# ---- write manifest ---------------------------------------------------------
MANIFEST="$RUN_DIR/manifest.json"
python3 - "$MANIFEST" "$BLOCK" "$SCENARIO" "$DATE_DIR" "$TS" "$DURATION_S" "$CAMERA_DEV" "$MOCK" "${TOPICS[@]}" <<'PY'
import json
import sys
manifest_path, block, scenario, date_dir, ts, dur, cam, mock, *topics = sys.argv[1:]
data = {
    "block": block,
    "scenario": scenario,
    "date": date_dir,
    "timestamp": ts,
    "duration_s": int(dur),
    "topics": topics,
    "camera_device": cam or None,
    "mock": bool(int(mock)),
}
with open(manifest_path, "w") as fh:
    json.dump(data, fh, indent=2)
print(f"[record] manifest -> {manifest_path}")
PY

# ---- mock short circuit -----------------------------------------------------
if [[ $MOCK -eq 1 ]]; then
    echo "[record] mock mode: skipping ros2 + ffmpeg. Run dir: $RUN_DIR" | tee "$LOG_FILE"
    exit 0
fi

# ---- start rosbag2 ----------------------------------------------------------
BAG_DIR="$RUN_DIR/bag"
echo "[record] rosbag2 record -o $BAG_DIR (duration ${DURATION_S}s)" | tee -a "$LOG_FILE"

# Filter to topics that exist NOW (ros2 bag record fails on any nonexistent
# topic with strict mode; without strict it silently records partials. Be
# explicit: skip absent topics and log which ones were dropped).
LIVE_TOPICS=()
ALL_TOPICS_RAW=$(ros2 topic list 2>/dev/null || echo "")
for t in "${TOPICS[@]}"; do
    if grep -qx "$t" <<<"$ALL_TOPICS_RAW"; then
        LIVE_TOPICS+=("$t")
    else
        echo "[record] WARN: topic $t not present at record start, skipping" | tee -a "$LOG_FILE"
    fi
done

if [[ ${#LIVE_TOPICS[@]} -eq 0 ]]; then
    echo "ERROR: no topics from harness set are live. Bring up bridges first." >&2
    exit 2
fi

ros2 bag record -o "$BAG_DIR" "${LIVE_TOPICS[@]}" >>"$LOG_FILE" 2>&1 &
BAG_PID=$!

# ---- start camera ffmpeg if available ---------------------------------------
CAM_PID=""
if [[ -n "$CAMERA_DEV" ]]; then
    CAM_OUT="$RUN_DIR/camera.mp4"
    echo "[record] ffmpeg -> $CAM_OUT from $CAMERA_DEV" | tee -a "$LOG_FILE"
    ffmpeg -nostdin -loglevel warning \
        -f v4l2 -framerate 30 -video_size 640x480 -i "$CAMERA_DEV" \
        -t "$DURATION_S" -c:v libx264 -preset ultrafast -pix_fmt yuv420p \
        "$CAM_OUT" >>"$LOG_FILE" 2>&1 &
    CAM_PID=$!
fi

# ---- cleanup trap -----------------------------------------------------------
cleanup() {
    echo "[record] stopping (bag=$BAG_PID cam=${CAM_PID:-none})" | tee -a "$LOG_FILE"
    [[ -n "$BAG_PID" ]] && kill -INT "$BAG_PID" 2>/dev/null || true
    [[ -n "$CAM_PID" ]] && kill -INT "$CAM_PID" 2>/dev/null || true
    sleep 2
    [[ -n "$BAG_PID" ]] && kill -TERM "$BAG_PID" 2>/dev/null || true
    [[ -n "$CAM_PID" ]] && kill -TERM "$CAM_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep "$DURATION_S"
echo "[record] duration elapsed" | tee -a "$LOG_FILE"
cleanup
trap - EXIT

echo "[record] done. dir=$RUN_DIR"
echo "RUN_DIR=$RUN_DIR"
