#!/usr/bin/env bash
# Seed 46 ONLY. 2026-09-25: seed 45 was stopped and abandoned by choice
# (Yusuf, via coordinator) to start seed 46 immediately with curriculum on +
# the action_l2 collapse fix + the joint_pos_limits weight fix for the calf
# hard-limit finding. Seed 45's checkpoints/log are kept but must NEVER be
# resumed by this or any other script; there is deliberately no SEEDS loop
# here, just one seed, so there is no path that could touch seed 45 again.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=/home/yusuf/Sim/isaac-sim-venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

SEED=46
RUN_NAME="walk_v2_seed46"
RUN_ROOT="checkpoints/velocity/${RUN_NAME}"
MAX_ITER=9000
NUM_ENVS=8192
SAVE_INTERVAL=250
ACTION_CLIP=100.0
CURRICULUM=on
ARM_LABEL=clip100-curriculum-limitpen
JOINT_POS_LIMITS_WEIGHT=-10.0

latest_checkpoint() {
    find "$RUN_ROOT" -maxdepth 2 -name 'model_*.pt' 2>/dev/null \
        | sed -E 's#.*/model_([0-9]+)\.pt#\1 &#' \
        | sort -n -k1,1 \
        | tail -n1 \
        | cut -d' ' -f2-
}

latest_checkpoint_iter() {
    find "$RUN_ROOT" -maxdepth 2 -name 'model_*.pt' 2>/dev/null \
        | sed -E 's#.*/model_([0-9]+)\.pt#\1#' \
        | sort -n \
        | tail -n1
}

DONE_ITER="$(latest_checkpoint_iter)"
DONE_ITER="${DONE_ITER:-0}"

if [[ "$DONE_ITER" -ge "$MAX_ITER" ]]; then
    echo "[run_velocity_seed46_only] seed 46 already at iteration ${DONE_ITER} >= ${MAX_ITER}, skipping training"
else
    RESUME_ARGS=()
    if [[ "$DONE_ITER" -gt 0 ]]; then
        RESUME_PATH="$(latest_checkpoint)"
        echo "[run_velocity_seed46_only] resuming seed 46 from iteration ${DONE_ITER} (${RESUME_PATH}) -> target ${MAX_ITER}"
        RESUME_ARGS=(--resume "$RESUME_PATH" --resume-continue-iterations)
    else
        echo "[run_velocity_seed46_only] starting seed 46 (curriculum=${CURRICULUM}, joint_pos_limits_weight=${JOINT_POS_LIMITS_WEIGHT}) -> ${RUN_ROOT}"
    fi

    "$PY" scripts/train_velocity.py \
        --run-name "$RUN_NAME" \
        --seed "$SEED" \
        --num-envs "$NUM_ENVS" \
        --max-iterations "$MAX_ITER" \
        --save-interval "$SAVE_INTERVAL" \
        --action-clip "$ACTION_CLIP" \
        --curriculum "$CURRICULUM" \
        --arm-label "$ARM_LABEL" \
        --joint-pos-limits-weight "$JOINT_POS_LIMITS_WEIGHT" \
        "${RESUME_ARGS[@]}" \
        --headless \
        2>&1 | tee -a "logs/train_${RUN_NAME}.log"
    RC=${PIPESTATUS[0]}
    echo "[run_velocity_seed46_only] seed 46 training exited with code ${RC}"
fi

echo "[run_velocity_seed46_only] running postrun checkpoint-sweep + gate chain (seed 46 only)"
"$PY" scripts/postrun_velocity_v2.py \
    --run-roots "$RUN_ROOT" \
    --python "$PY" \
    --summary-out "checkpoints/velocity/summary_seed46_$(date +%Y%m%d_%H%M%S).json" \
    2>&1 | tee -a logs/postrun_velocity_seed46.log

echo "[run_velocity_seed46_only] DONE. Summary written under checkpoints/velocity/."
