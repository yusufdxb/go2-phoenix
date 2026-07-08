#!/usr/bin/env bash
# harness_preflight.sh
# Single-command preflight wrapping P1 through P7.
#
# Usage (Jetson-side, real run):
#   bash scripts/harness_preflight.sh
#
# Usage (self-test on the dev workstation, skips ROS2 and motor checks):
#   bash scripts/harness_preflight.sh --mock
#
# Optional env vars:
#   T7_MOUNT          path to the T7 checkpoint root (default: /media/external/go2-phoenix)
#   JETSON_REPO       repo root on Jetson (default: $HOME/workspace/go2-phoenix)
#   PHOENIX_BRANCH    branch to ff-only (default: main)
#   DEPLOY_CFG        deploy yaml (default: configs/sim2real/deploy_stand_v3.yaml)
#   CKPT_DIR          checkpoint dir to rsync (default: checkpoints/phoenix-stand-v3)
#
# Exits non-zero on any halt condition (P3 P4 P5 P6 failures).

set -euo pipefail

MOCK=0
if [[ "${1:-}" == "--mock" ]]; then
    MOCK=1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T7_MOUNT="${T7_MOUNT:-/media/external/go2-phoenix}"
JETSON_REPO="${JETSON_REPO:-$HOME/workspace/go2-phoenix}"
PHOENIX_BRANCH="${PHOENIX_BRANCH:-main}"
DEPLOY_CFG="${DEPLOY_CFG:-configs/sim2real/deploy_stand_v3.yaml}"
CKPT_DIR="${CKPT_DIR:-checkpoints/phoenix-stand-v3}"

LOG_DIR="/tmp/harness_preflight_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

c_red()   { printf "\033[31m%s\033[0m\n" "$*"; }
c_green() { printf "\033[32m%s\033[0m\n" "$*"; }
c_yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
step() { printf "\n=== %s ===\n" "$*"; }
halt() { c_red "HALT: $*"; exit 1; }

require_bin() {
    local bin="$1"
    local hint="${2:-}"
    if ! command -v "$bin" >/dev/null 2>&1; then
        halt "missing required binary: $bin ${hint:+($hint)}"
    fi
}

confirm_yes() {
    local prompt="$1"
    if [[ $MOCK -eq 1 ]]; then
        c_yellow "[mock] auto-confirming: $prompt"
        return 0
    fi
    read -r -p "$prompt [y/N] " ans
    if [[ "$ans" != "y" && "$ans" != "Y" ]]; then
        halt "operator declined: $prompt"
    fi
}

# ---- binary checks (always required, mock or not) ---------------------------
step "binary checks"
require_bin rsync
require_bin md5sum "install coreutils"
require_bin git
require_bin python3
if [[ $MOCK -eq 0 ]]; then
    require_bin ros2 "source /opt/ros/humble/setup.bash"
fi

# ---- P1: T7 -> Jetson rsync with md5 verification ---------------------------
step "P1: rsync checkpoint + deploy yaml from T7"
if [[ $MOCK -eq 1 ]]; then
    c_yellow "[mock] skipping T7 rsync (no external drive mounted for self-test)"
else
    if [[ ! -d "$T7_MOUNT" ]]; then
        halt "P1: T7 not mounted at $T7_MOUNT"
    fi
    SRC_CKPT="$T7_MOUNT/$CKPT_DIR/"
    DST_CKPT="$JETSON_REPO/$CKPT_DIR/"
    mkdir -p "$DST_CKPT"
    rsync -av --checksum "$SRC_CKPT" "$DST_CKPT" | tee "$LOG_DIR/p1_rsync.log"

    SRC_CFG="$T7_MOUNT/$DEPLOY_CFG"
    DST_CFG="$JETSON_REPO/$DEPLOY_CFG"
    if [[ -f "$SRC_CFG" ]]; then
        rsync -av --checksum "$SRC_CFG" "$DST_CFG"
    fi

    # md5 verify the ONNX + latest.pt
    for f in policy.onnx latest.pt policy.pt; do
        if [[ -f "$SRC_CKPT/$f" && -f "$DST_CKPT/$f" ]]; then
            s_md5=$(md5sum "$SRC_CKPT/$f" | awk '{print $1}')
            d_md5=$(md5sum "$DST_CKPT/$f" | awk '{print $1}')
            if [[ "$s_md5" != "$d_md5" ]]; then
                halt "P1: md5 mismatch on $f (T7=$s_md5 vs Jetson=$d_md5)"
            fi
            c_green "P1: md5 match $f $s_md5"
        fi
    done
fi

# ---- P2: git fetch + ff-only merge ------------------------------------------
step "P2: git fetch + ff-only merge $PHOENIX_BRANCH"
if [[ $MOCK -eq 1 ]]; then
    c_yellow "[mock] skipping git ff-only (would mutate the local repo)"
else
    cd "$JETSON_REPO"
    git fetch origin "$PHOENIX_BRANCH" | tee "$LOG_DIR/p2_fetch.log"
    git merge --ff-only "origin/$PHOENIX_BRANCH" | tee -a "$LOG_DIR/p2_fetch.log"
    HEAD_SHA=$(git rev-parse --short HEAD)
    c_green "P2: HEAD at $HEAD_SHA"
fi

cd "$REPO_ROOT"

# ---- P3: pytest gate --------------------------------------------------------
step "P3: pytest -x -q (no hardware markers)"
if [[ $MOCK -eq 1 ]]; then
    # In mock mode only run the harness self tests
    if ! pytest tests/test_harness_diversity.py tests/test_harness_record.py -x -q 2>&1 | tee "$LOG_DIR/p3_pytest.log"; then
        halt "P3: harness self-tests failed"
    fi
else
    if ! pytest tests/ -x -q -m "not sim and not ros" 2>&1 | tee "$LOG_DIR/p3_pytest.log"; then
        halt "P3: pytest failed"
    fi
fi
c_green "P3: pytest OK"

# ---- P4: parity gate --------------------------------------------------------
step "P4: phoenix.sim2real.verify_deploy parity gate"
if [[ $MOCK -eq 1 ]]; then
    c_yellow "[mock] skipping verify_deploy (needs torch + onnxruntime + parquet)"
else
    REF_PARQUET="${REF_PARQUET:-data/failures/dryrun_latest.parquet}"
    if [[ ! -f "$REF_PARQUET" ]]; then
        # Pick newest parquet under data/failures as fallback
        REF_PARQUET=$(ls -t data/failures/*.parquet 2>/dev/null | head -1 || true)
    fi
    if [[ -z "$REF_PARQUET" || ! -f "$REF_PARQUET" ]]; then
        halt "P4: no reference parquet for parity check (set REF_PARQUET env)"
    fi
    if ! PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" python3 -m phoenix.sim2real.verify_deploy \
        --parquet "$REF_PARQUET" --deploy-cfg "$DEPLOY_CFG" --tol 1e-4 2>&1 | tee "$LOG_DIR/p4_parity.log"; then
        halt "P4: parity gate FAIL"
    fi
    c_green "P4: parity OK"
fi

# ---- P5: three-bridge bringup + topic-hz check ------------------------------
step "P5: three-bridge bringup + /joint_states hz check"
if [[ $MOCK -eq 1 ]]; then
    c_yellow "[mock] skipping bridge bringup (requires ROS2 + cyclonedds + robot)"
else
    # Wrap the existing dryrun pipeline (already brings up the three bridges
    # and samples /joint_states hz). Re-use, do not reinvent.
    bash "$REPO_ROOT/scripts/dryrun_pipeline.sh" 25 2>&1 | tee "$LOG_DIR/p5_dryrun.log" || true
    if [[ -f /tmp/dryrun_samples.log ]]; then
        cp /tmp/dryrun_samples.log "$LOG_DIR/p5_samples.log"
        # Look for /joint_states average rate >= 200
        JS_HZ=$(grep -A1 "/joint_states hz" /tmp/dryrun_samples.log | tail -1 | awk -F'average rate: ' '{print $2}' | awk '{print $1}' | head -1)
        if [[ -z "$JS_HZ" ]]; then
            halt "P5: could not parse /joint_states hz from dryrun samples"
        fi
        # bash arithmetic is integer only; strip the decimal
        JS_HZ_INT=${JS_HZ%.*}
        if [[ "$JS_HZ_INT" -lt 200 ]]; then
            halt "P5: /joint_states hz $JS_HZ < 200"
        fi
        c_green "P5: /joint_states hz $JS_HZ OK"
    else
        halt "P5: dryrun samples log not produced"
    fi
fi

# ---- P6: E-stop verification (manual, prompted) -----------------------------
step "P6: E-stop press-latch -> release -> fresh-and-False verification"
cat <<'EOF'
  1. Press the wireless E-stop. Confirm `/phoenix/estop` latches True.
     ros2 topic echo --once /phoenix/estop
  2. Release E-stop. Confirm policy_node sees fresh-and-False before resuming.
EOF
confirm_yes "P6: E-stop behaves as expected?"

# ---- P7: physical check -----------------------------------------------------
step "P7: physical check"
cat <<'EOF'
  - Robot on fall mat
  - Low sit pose
  - Feet unloaded (suspended slightly or sitting flat without pre-load)
  - Tether clipped if Block 2 or Block 3 will run live
  - 2 m clear radius around robot
EOF
confirm_yes "P7: physical state OK?"

c_green ""
c_green "==============================================="
c_green "PREFLIGHT COMPLETE. Logs at $LOG_DIR"
c_green "==============================================="
