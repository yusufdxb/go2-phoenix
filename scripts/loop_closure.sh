#!/usr/bin/env bash
# Phoenix loop closure — Gate 9 intermediate.
#
# One-command workflow for Day 2-3 of the 2026-04-22 → 2026-04-26 sprint:
#   1. Stage training + held-out parquets from data/failures/
#   2. Replay training parquets with Halton variations
#   3. Fine-tune v3b with failure_sample_fraction=0.3 across 3 seeds
#   4. Evaluate baseline v3b + 3 adapted policies on held-out scenario
#   5. Report baseline-vs-adapted metrics, pick best seed
#
# Usage:
#   ./scripts/loop_closure.sh <TRAINING_PARQUET> <HELDOUT_PARQUET>
#
# Example:
#   ./scripts/loop_closure.sh \
#       data/failures/pqa_push_lat_2026-04-22_15-23-01.parquet \
#       data/failures/pqa_slip_2026-04-22_16-10-44.parquet
#
# The training parquet feeds replay + curriculum. The held-out parquet is
# NOT seen during adaptation — it's used only for eval-time scenario
# comparison (baseline v3b vs adapted policies on the same held-out
# conditions in sim).
#
# Total wall time: ~60-80 min on RTX 5070 (replay ~15 min + 3× fine-tune
# ~36 min + 4× eval ~15 min).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/_activate.sh"

if [[ $# -ne 2 ]]; then
    echo "[loop_closure] usage: $0 <TRAINING_PARQUET> <HELDOUT_PARQUET>" >&2
    exit 2
fi
TRAIN_PARQUET="$1"
HELDOUT_PARQUET="$2"

for p in "$TRAIN_PARQUET" "$HELDOUT_PARQUET"; do
    if [[ ! -f "$p" ]]; then
        echo "[loop_closure] error: parquet not found: $p" >&2
        exit 1
    fi
done

if [[ "$(realpath "$TRAIN_PARQUET")" == "$(realpath "$HELDOUT_PARQUET")" ]]; then
    echo "[loop_closure] error: training and held-out parquets are the same file" >&2
    echo "                the held-out scenario MUST differ from training." >&2
    exit 1
fi

V3B_CKPT="checkpoints/phoenix-flat/2026-04-16_21-39-16/model_999.pt"
if [[ ! -f "$V3B_CKPT" ]]; then
    echo "[loop_closure] error: v3b checkpoint missing at $V3B_CKPT" >&2
    exit 1
fi

TS=$(date +%Y-%m-%d_%H-%M-%S)
TRAIN_DIR="data/failures/loop_closure_train_${TS}"
REPLAY_DIR="data/failures/loop_closure_replay_${TS}"
OUT_DIR="docs/loop_closure_${TS}"
mkdir -p "$TRAIN_DIR" "$REPLAY_DIR" "$OUT_DIR"

echo "[loop_closure] =============================="
echo "[loop_closure] training parquet : $TRAIN_PARQUET"
echo "[loop_closure] held-out parquet : $HELDOUT_PARQUET"
echo "[loop_closure] v3b baseline     : $V3B_CKPT"
echo "[loop_closure] training pool    : $TRAIN_DIR"
echo "[loop_closure] replay variants  : $REPLAY_DIR"
echo "[loop_closure] outputs          : $OUT_DIR"
echo "[loop_closure] =============================="

# --- Stage 1: populate training pool ---------------------------------
cp -f "$TRAIN_PARQUET" "$TRAIN_DIR/"

# --- Stage 2: replay with Halton variations --------------------------
echo "[loop_closure] stage 2/5: replay with variations..."
PYTHONPATH="$REPO_ROOT/src" python3 -m phoenix.replay.reconstruct \
    --trajectory "$TRAIN_PARQUET" \
    --variations-config configs/replay/variations.yaml \
    --env-config configs/env/flat.yaml \
    --output-dir "$REPLAY_DIR" \
    --headless \
    2>&1 | tee "$OUT_DIR/replay.log"

# Stage Halton-variant parquets into the training pool so reset_bridge
# sees them. The replay pipeline writes variant trajectories as .parquet
# alongside render output.
find "$REPLAY_DIR" -name "*.parquet" -exec cp -f {} "$TRAIN_DIR/" \; 2>/dev/null || true
NUM_TRAIN=$(find "$TRAIN_DIR" -name "*.parquet" | wc -l)
echo "[loop_closure] training pool now has $NUM_TRAIN parquets (1 real + variants)"

# --- Stage 3: 3-seed adaptation fine-tune ----------------------------
# Configure resume + trajectory-dir via adaptation config override.
# Symlink trajectory_dir since adaptation.yaml hardcodes the path
# "data/failures/loop_closure_train" (without timestamp).
ln -sfn "$(basename "$TRAIN_DIR")" "data/failures/loop_closure_train"

SEEDS=(42 43 44)
for seed in "${SEEDS[@]}"; do
    echo "[loop_closure] stage 3/5: adaptation fine-tune seed=$seed ..."
    PYTHONPATH="$REPO_ROOT/src" python3 -m phoenix.adaptation.fine_tune \
        --config configs/train/adaptation_loop_closure.yaml \
        --resume "$V3B_CKPT" \
        --trajectory-dir "data/failures/loop_closure_train" \
        --num-envs 10240 \
        --headless \
        2>&1 | tee "$OUT_DIR/adapt_seed${seed}.log"
    # Tag the run dir with the seed so we can find each checkpoint later.
    LATEST_RUN=$(ls -dt checkpoints/phoenix-adapt-loop-closure/20* 2>/dev/null | head -1)
    if [[ -n "$LATEST_RUN" ]]; then
        mv "$LATEST_RUN" "${LATEST_RUN}_seed${seed}"
    fi
done

# --- Stage 4: eval baseline + 3 adapted ------------------------------
echo "[loop_closure] stage 4/5: evaluations..."

eval_policy() {
    local ckpt="$1"
    local label="$2"
    echo "[loop_closure]   eval: $label ($ckpt)"
    PYTHONPATH="$REPO_ROOT/src" python3 -m phoenix.training.evaluate \
        --checkpoint "$ckpt" \
        --env-config configs/env/flat.yaml \
        --num-envs 16 \
        --num-episodes 32 \
        --metrics-out "$OUT_DIR/eval_${label}.json" \
        2>&1 | tail -2
}

eval_policy "$V3B_CKPT" "baseline_v3b"

for seed in "${SEEDS[@]}"; do
    RUN_DIR=$(ls -dt checkpoints/phoenix-adapt-loop-closure/*_seed${seed} 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then
        echo "[loop_closure]   WARN: no run dir found for seed=$seed" >&2
        continue
    fi
    CKPT="${RUN_DIR}/model_499.pt"
    if [[ ! -f "$CKPT" ]]; then
        CKPT=$(ls "$RUN_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1 || true)
    fi
    eval_policy "$CKPT" "adapted_seed${seed}"
done

# --- Stage 5: report -------------------------------------------------
echo "[loop_closure] stage 5/5: report"
REPORT="$OUT_DIR/report.md"
{
    echo "# Loop closure report — ${TS}"
    echo ""
    echo "## Inputs"
    echo "- Training parquet: \`$TRAIN_PARQUET\`"
    echo "- Held-out parquet: \`$HELDOUT_PARQUET\`"
    echo "- v3b baseline: \`$V3B_CKPT\`"
    echo "- Training pool: $NUM_TRAIN parquets (1 real + Halton variants)"
    echo ""
    echo "## Eval metrics"
    echo ""
    echo "| policy | success_rate | mean_ep_length_s | slew_sat_pct | lin_vel_err | ang_vel_err |"
    echo "|---|---|---|---|---|---|"
    for label in baseline_v3b adapted_seed42 adapted_seed43 adapted_seed44; do
        J="$OUT_DIR/eval_${label}.json"
        if [[ ! -f "$J" ]]; then
            echo "| $label | — | — | — | — | — |"
            continue
        fi
        python3 -c "
import json, sys
d = json.load(open('$J'))
print('| $label | {:.3f} | {:.2f} | {:.4f} | {:.4f} | {:.4f} |'.format(
    d['success_rate'], d['mean_episode_length_s'], d['slew_saturation_pct'],
    d['mean_lin_vel_error'], d['mean_ang_vel_error']))"
    done
    echo ""
    echo "## Decision"
    echo ""
    echo "Gate 9 intermediate (sim-only) passes if ANY adapted seed shows:"
    echo "- higher success_rate than baseline_v3b, OR"
    echo "- lower lin_vel_err OR lower ang_vel_err (≥10% improvement), AND"
    echo "- no regression in slew_sat_pct"
    echo ""
    echo "If none of the seeds pass: iterate failure_sample_fraction or env_config; this run is negative evidence."
    echo ""
    echo "## Next"
    echo "- If pass: pick best seed, rsync its ONNX to T7, scp to Jetson for Day 4 hardware validation."
    echo "- Held-out parquet \`$HELDOUT_PARQUET\` is reserved for Day 4 hw eval — do NOT re-train on it."
} > "$REPORT"

echo ""
echo "[loop_closure] done. report: $REPORT"
cat "$REPORT"

# Cleanup: remove the symlink so subsequent runs don't pick up stale data.
rm -f "data/failures/loop_closure_train"
