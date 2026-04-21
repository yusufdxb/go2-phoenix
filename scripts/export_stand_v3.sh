#!/usr/bin/env bash
# Export phoenix-stand-v3 ONNX + TorchScript for Jetson deployment.
#
# Usage:
#   ./scripts/export_stand_v3.sh [checkpoint_dir]
#
# checkpoint_dir defaults to the newest sub-directory under
# checkpoints/phoenix-stand-v3/. Picks model_499.pt (or the highest
# model_N.pt if 499 isn't there) as the source.
#
# Produces, at checkpoints/phoenix-stand-v3/:
#   latest.pt        rsl_rl checkpoint copy
#   policy.onnx      inline-weights ONNX (obs=48, action=12)
#   policy.pt        TorchScript fallback
#   export_report.txt  per-file sha256 + parity report
#
# Runs phoenix.sim2real.export (torch-based, no IsaacLab required).
# On an air-gapped Jetson, onnx==1.16.2 is sufficient (see
# lab_findings_2026-04-20.md "onnx install path on the air-gapped Jetson").
# On the training machine onnx is already available via isaac-sim-venv.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/_activate.sh"

CKPT_DIR="${1:-}"
if [[ -z "$CKPT_DIR" ]]; then
    CKPT_DIR=$(ls -dt checkpoints/phoenix-stand-v3/[0-9]* 2>/dev/null | head -1 || true)
fi
if [[ -z "$CKPT_DIR" || ! -d "$CKPT_DIR" ]]; then
    echo "[export_stand_v3] error: no checkpoint_dir under checkpoints/phoenix-stand-v3/" >&2
    echo "                   train first via ./scripts/train_stand_v3.sh" >&2
    exit 1
fi

# Pick model_499.pt or the highest-numbered model_N.pt.
CKPT=""
if [[ -f "$CKPT_DIR/model_499.pt" ]]; then
    CKPT="$CKPT_DIR/model_499.pt"
else
    CKPT=$(ls "$CKPT_DIR"/model_[0-9]*.pt 2>/dev/null | sort -V | tail -1 || true)
fi
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "[export_stand_v3] error: no model_N.pt inside $CKPT_DIR" >&2
    exit 1
fi

OUT_DIR="checkpoints/phoenix-stand-v3"
mkdir -p "$OUT_DIR"
cp -f "$CKPT" "$OUT_DIR/latest.pt"

echo "[export_stand_v3] source checkpoint : $CKPT"
echo "[export_stand_v3] staged latest.pt  : $OUT_DIR/latest.pt"

PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
python3 -m phoenix.sim2real.export \
    --checkpoint "$OUT_DIR/latest.pt" \
    --output     "$OUT_DIR/policy.onnx" \
    --verify

REPORT="$OUT_DIR/export_report.txt"
{
    echo "stand-v3 export report"
    date -u +"%Y-%m-%dT%H:%M:%SZ"
    echo "source_checkpoint: $CKPT"
    echo
    echo "sha256:"
    sha256sum "$OUT_DIR/latest.pt" "$OUT_DIR/policy.onnx" "$OUT_DIR/policy.pt" 2>/dev/null || true
} > "$REPORT"

echo "[export_stand_v3] wrote $REPORT"
cat "$REPORT"
