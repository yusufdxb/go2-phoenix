#!/usr/bin/env bash
# Train the clip_actions=100 arm for 3 seeds SEQUENTIALLY, then export+eval+gate every seed
# including the clip=1.0 control (seed 42 from run_velocity_seeds.sh).
#
# Sequential, not concurrent: measured 2026-09-25 on this GPU that 2 concurrent
# Isaac Sim/PhysX processes give ~98k combined env-steps/s (64k + 33k) versus
# ~140-157k solo -- concurrency LOSES ~35-40% of total throughput on this card
# (no MPS), so sequential at full num_envs is what actually maximizes total
# env-steps/s, not concurrent smaller runs.
#
# Idempotent / resume-aware: re-running this script (e.g. by the watchdog,
# scripts/watchdog_velocity.sh) skips any seed already at or past MAX_ITER and
# resumes a partially-trained seed from its highest-numbered checkpoint with
# --resume-continue-iterations, so the total per-seed iteration count is
# preserved (see train_velocity.py's --resume-continue-iterations for why the
# plain rsl_rl --resume semantics would otherwise run MAX_ITER extra iterations
# on top of what the checkpoint already has).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=/home/yusuf/Sim/isaac-sim-venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

SEEDS=(42 43 44)
# num_envs=8192 chosen over 4096/16384 by measured throughput (2026-09-25):
# 4096 envs: 0.62s/iter, ~157-162k env-steps/s
# 8192 envs: 0.92s/iter, ~213k env-steps/s (+35% over 4096)
# 16384 envs: 1.72s/iter, ~228k env-steps/s (+7% more, iteration time nearly
#   doubles; 24*16384=393k transitions/update is well past useful batch-size
#   returns for this problem, and PPO convergence tracks ITERATION count more
#   than raw env-steps, so 16384 would complete far fewer updates in the same
#   wall-clock budget -- worse for training quality despite higher raw steps/s)
MAX_ITER=9000
NUM_ENVS=8192
SAVE_INTERVAL=250
ACTION_CLIP=100.0
RUN_ROOTS=("checkpoints/velocity/walk_v1_seed42")  # clip=1.0 control, already trained

# Highest-numbered model_*.pt across ALL timestamped subdirs of a run root, or
# empty if none. Filenames carry the TRUE global rsl_rl iteration (see
# train_velocity.py --resume-continue-iterations), so this is correct even
# after a prior resume created a second subdir.
latest_checkpoint() {
    local run_root="$1"
    find "$run_root" -maxdepth 2 -name 'model_*.pt' 2>/dev/null \
        | sed -E 's#.*/model_([0-9]+)\.pt#\1 &#' \
        | sort -n -k1,1 \
        | tail -n1 \
        | cut -d' ' -f2-
}

latest_checkpoint_iter() {
    local run_root="$1"
    find "$run_root" -maxdepth 2 -name 'model_*.pt' 2>/dev/null \
        | sed -E 's#.*/model_([0-9]+)\.pt#\1#' \
        | sort -n \
        | tail -n1
}

for SEED in "${SEEDS[@]}"; do
    RUN_NAME="walk_v1_clip100_seed${SEED}"
    RUN_ROOT="checkpoints/velocity/${RUN_NAME}"
    RUN_ROOTS+=("$RUN_ROOT")

    DONE_ITER="$(latest_checkpoint_iter "$RUN_ROOT")"
    DONE_ITER="${DONE_ITER:-0}"

    if [[ "$DONE_ITER" -ge "$MAX_ITER" ]]; then
        echo "=================================================================="
        echo "[run_velocity_clip100_seeds] seed ${SEED} already at iteration ${DONE_ITER} >= ${MAX_ITER}, skipping"
        echo "=================================================================="
        continue
    fi

    RESUME_ARGS=()
    if [[ "$DONE_ITER" -gt 0 ]]; then
        RESUME_PATH="$(latest_checkpoint "$RUN_ROOT")"
        echo "=================================================================="
        echo "[run_velocity_clip100_seeds] seed ${SEED} resuming from iteration ${DONE_ITER} (${RESUME_PATH}) -> target ${MAX_ITER}"
        echo "=================================================================="
        RESUME_ARGS=(--resume "$RESUME_PATH" --resume-continue-iterations)
    else
        echo "=================================================================="
        echo "[run_velocity_clip100_seeds] starting seed ${SEED} (clip_actions=${ACTION_CLIP}) -> ${RUN_ROOT}"
        echo "=================================================================="
    fi

    "$PY" scripts/train_velocity.py \
        --run-name "$RUN_NAME" \
        --seed "$SEED" \
        --num-envs "$NUM_ENVS" \
        --max-iterations "$MAX_ITER" \
        --save-interval "$SAVE_INTERVAL" \
        --action-clip "$ACTION_CLIP" \
        "${RESUME_ARGS[@]}" \
        --headless \
        2>&1 | tee -a "logs/train_${RUN_NAME}.log"
    RC=${PIPESTATUS[0]}
    echo "[run_velocity_clip100_seeds] seed ${SEED} training exited with code ${RC}"
    if [[ $RC -ne 0 ]]; then
        echo "[run_velocity_clip100_seeds] seed ${SEED} FAILED, continuing to next seed anyway"
    fi
done

echo "=================================================================="
echo "[run_velocity_clip100_seeds] all seeds done, running postrun export+eval+gate chain"
echo "[run_velocity_clip100_seeds] run roots (control + 3 clip100 seeds): ${RUN_ROOTS[*]}"
echo "=================================================================="
"$PY" scripts/postrun_velocity.py \
    --run-roots "${RUN_ROOTS[@]}" \
    --python "$PY" \
    --summary-out "checkpoints/velocity/summary_$(date +%Y%m%d_%H%M%S).json" \
    2>&1 | tee -a logs/postrun_velocity_clip100.log

echo "[run_velocity_clip100_seeds] DONE. Summary written under checkpoints/velocity/."
