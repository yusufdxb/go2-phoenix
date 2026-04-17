#!/usr/bin/env bash
# Post-CUDA-12.8-install smoke test for the mewtwo toolchain.
# Walks the full phoenix-relevant layers in order. Halts on the first
# failure so you know exactly which layer broke.
#
# Run AFTER CUDA 12.8 install + any required reboot. Expected total
# runtime ~2 minutes.
#
# Usage:
#   ./scripts/cuda_128_smoke.sh
#
# Exits 0 on full pass, 1 + layer name on first failure.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

pass() { printf '[PASS] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n%s\n' "$1" "${2:-}" >&2; exit 1; }

# ---- Layer 1: NVIDIA driver loaded, GPU visible -----------------------------
echo "=== Layer 1: NVIDIA driver ==="
if ! nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi" "Driver not loaded. If a reboot was pending, do it now."
fi
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
pass "Driver $DRIVER on $GPU"

# ---- Layer 2: CUDA toolkit installed ----------------------------------------
echo "=== Layer 2: CUDA toolkit ==="
if ! command -v nvcc >/dev/null 2>&1; then
    fail "nvcc not in PATH" "Check /usr/local/cuda-12.8/bin is on PATH."
fi
NVCC_VER=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | head -1)
pass "nvcc $NVCC_VER"
case "$NVCC_VER" in
    "release 12.8") ;;
    *) fail "unexpected nvcc version" "Expected 12.8, got '$NVCC_VER'" ;;
esac

# ---- Layer 3: System python torch sees CUDA ---------------------------------
echo "=== Layer 3: system python torch CUDA ==="
SYS_TORCH_OUT=$(python3 -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)' 2>&1) \
    || fail "system python torch" "$SYS_TORCH_OUT"
pass "system torch: $SYS_TORCH_OUT"
case "$SYS_TORCH_OUT" in
    "True "*) ;;
    *) fail "system torch can't see CUDA" "Reinstall torch with --index-url https://download.pytorch.org/whl/cu128" ;;
esac

# ---- Layer 4: isaac-sim-venv torch sees CUDA --------------------------------
echo "=== Layer 4: isaac-sim-venv torch CUDA ==="
ISAAC_PY="${HOME}/isaac-sim-venv/bin/python"
if [[ ! -x "$ISAAC_PY" ]]; then
    fail "isaac-sim-venv missing" "Expected $ISAAC_PY"
fi
ISAAC_TORCH_OUT=$("$ISAAC_PY" -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.__version__)' 2>&1) \
    || fail "isaac-sim-venv torch" "$ISAAC_TORCH_OUT"
pass "isaac torch: $ISAAC_TORCH_OUT"
case "$ISAAC_TORCH_OUT" in
    "True "*) ;;
    *) fail "isaac-sim-venv torch can't see CUDA" "Reinstall: $ISAAC_PY -m pip install torch --index-url https://download.pytorch.org/whl/cu128" ;;
esac

# ---- Layer 5: Isaac Sim launch (headless, tiny) -----------------------------
echo "=== Layer 5: Isaac Sim launch ==="
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/IsaacLab}"
LAUNCH_OUT=$("$ISAACLAB_PATH/isaaclab.sh" -p -c 'print("isaac_ok")' 2>&1 | tail -20)
case "$LAUNCH_OUT" in
    *"isaac_ok"*) pass "Isaac Sim launches" ;;
    *) fail "Isaac Sim launch" "$LAUNCH_OUT" ;;
esac

# ---- Layer 6: phoenix unit tests (pure-python, no CUDA) ---------------------
echo "=== Layer 6: phoenix unit tests ==="
PYTEST_OUT=$(PYTHONPATH=src python3 -m pytest -q --ignore=tests/test_sim_integration.py 2>&1 | tail -3)
case "$PYTEST_OUT" in
    *"passed"*) pass "pytest: $(printf '%s' "$PYTEST_OUT" | tail -1)" ;;
    *) fail "phoenix unit tests" "$PYTEST_OUT" ;;
esac

# ---- Layer 7: ROS 2 rclpy import --------------------------------------------
echo "=== Layer 7: ROS 2 rclpy ==="
if [[ -f /opt/ros/humble/setup.bash ]]; then
    RCLPY_OUT=$(bash -c 'source /opt/ros/humble/setup.bash && python3 -c "import rclpy; print(rclpy.__name__)"' 2>&1)
    case "$RCLPY_OUT" in
        *"rclpy"*) pass "rclpy imports" ;;
        *) fail "rclpy import" "$RCLPY_OUT" ;;
    esac
else
    echo "[SKIP] /opt/ros/humble not present on this machine"
fi

# ---- Layer 8: phoenix export on an existing checkpoint ----------------------
echo "=== Layer 8: phoenix export (torch -> onnx, parity check) ==="
CKPT="checkpoints/phoenix-flat/latest.pt"
if [[ ! -e "$CKPT" ]]; then
    echo "[SKIP] $CKPT not present (no checkpoint to export against)"
else
    TMP_ONNX=$(mktemp --suffix=.onnx)
    EXPORT_OUT=$(PYTHONPATH="$REPO_ROOT/src" "$ISAAC_PY" -m phoenix.sim2real.export \
        --checkpoint "$CKPT" --output "$TMP_ONNX" --verify 2>&1 | tail -5)
    rm -f "$TMP_ONNX" "${TMP_ONNX%.onnx}.pt"
    case "$EXPORT_OUT" in
        *"Parity check passed"*) pass "export + parity" ;;
        *) fail "phoenix export" "$EXPORT_OUT" ;;
    esac
fi

echo
echo "=== All layers passed. Toolchain healthy. ==="
