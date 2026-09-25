#!/usr/bin/env bash
# Verification + best-checkpoint run for the action_l2 collapse fix (commit
# 9909675): seed 45 (curriculum on, the arm that produced the best checkpoint
# in the sweep) and seed 46 (curriculum off), both with action_l2=-0.01 and
# entropy_coef=0.01, 8192 envs, 9000 iterations, sequential (see
# run_velocity_clip100_seeds.sh for the measured concurrency-loses-throughput
# evidence this still relies on).
#
# Idempotent / resume-aware, same pattern as run_velocity_clip100_seeds.sh:
# safe to relaunch (e.g. by the watchdog) at any point.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=/home/yusuf/Sim/isaac-sim-venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# 2026-09-25: sim2sim (feat/phoenix-walk-v1-sim2sim @ba7b437) found
# seed42@3000 drives FR/RL calves to their upper hard limit during
# counter-clockwise yaw (+0.6, worse at +0.8), targets 0.25-0.31 rad past the
# stop. Changed seed 46 from curriculum=off to curriculum=on (matching 45)
# PLUS --joint-pos-limits-weight -10.0 (legged_gym/unitree_rl_gym convention;
# Phoenix's soft_joint_pos_limit_factor is already 0.9, matching that
# convention, see UNITREE_GO2_CFG in ~/Sim/IsaacLab/source/isaaclab_assets/
# isaaclab_assets/robots/unitree.py:168 and spec.py's joint_pos_limits term),
# so 45 vs 46 isolates that ONE change. Seed 45 already running, untouched.
SEEDS=(45 46)
declare -A CURRICULUM_BY_SEED=([45]="on" [46]="on")
declare -A ARM_LABEL_BY_SEED=([45]="v2-curriculum" [46]="v2-curriculum-limitpen")
declare -A JOINT_POS_LIMITS_WEIGHT_BY_SEED=([45]="" [46]="-10.0")
MAX_ITER=9000
NUM_ENVS=8192
SAVE_INTERVAL=250
ACTION_CLIP=100.0
RUN_ROOTS=()

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
    RUN_NAME="walk_v2_seed${SEED}"
    RUN_ROOT="checkpoints/velocity/${RUN_NAME}"
    RUN_ROOTS+=("$RUN_ROOT")

    DONE_ITER="$(latest_checkpoint_iter "$RUN_ROOT")"
    DONE_ITER="${DONE_ITER:-0}"

    if [[ "$DONE_ITER" -ge "$MAX_ITER" ]]; then
        echo "=================================================================="
        echo "[run_velocity_v2_seeds] seed ${SEED} already at iteration ${DONE_ITER} >= ${MAX_ITER}, skipping"
        echo "=================================================================="
        continue
    fi

    RESUME_ARGS=()
    if [[ "$DONE_ITER" -gt 0 ]]; then
        RESUME_PATH="$(latest_checkpoint "$RUN_ROOT")"
        echo "=================================================================="
        echo "[run_velocity_v2_seeds] seed ${SEED} resuming from iteration ${DONE_ITER} (${RESUME_PATH}) -> target ${MAX_ITER}"
        echo "=================================================================="
        RESUME_ARGS=(--resume "$RESUME_PATH" --resume-continue-iterations)
    else
        echo "=================================================================="
        echo "[run_velocity_v2_seeds] starting seed ${SEED} (curriculum=${CURRICULUM_BY_SEED[$SEED]}) -> ${RUN_ROOT}"
        echo "=================================================================="
    fi

    JPL_ARGS=()
    if [[ -n "${JOINT_POS_LIMITS_WEIGHT_BY_SEED[$SEED]}" ]]; then
        JPL_ARGS=(--joint-pos-limits-weight "${JOINT_POS_LIMITS_WEIGHT_BY_SEED[$SEED]}")
    fi

    "$PY" scripts/train_velocity.py \
        --run-name "$RUN_NAME" \
        --seed "$SEED" \
        --num-envs "$NUM_ENVS" \
        --max-iterations "$MAX_ITER" \
        --save-interval "$SAVE_INTERVAL" \
        --action-clip "$ACTION_CLIP" \
        --curriculum "${CURRICULUM_BY_SEED[$SEED]}" \
        --arm-label "${ARM_LABEL_BY_SEED[$SEED]}" \
        "${JPL_ARGS[@]}" \
        "${RESUME_ARGS[@]}" \
        --headless \
        2>&1 | tee -a "logs/train_${RUN_NAME}.log"
    RC=${PIPESTATUS[0]}
    echo "[run_velocity_v2_seeds] seed ${SEED} training exited with code ${RC}"
    if [[ $RC -ne 0 ]]; then
        echo "[run_velocity_v2_seeds] seed ${SEED} FAILED, continuing to next seed anyway"
    fi
done

echo "=================================================================="
echo "[run_velocity_v2_seeds] all seeds done, running postrun checkpoint-sweep + gate chain"
echo "[run_velocity_v2_seeds] run roots: ${RUN_ROOTS[*]}"
echo "=================================================================="
"$PY" scripts/postrun_velocity_v2.py \
    --run-roots "${RUN_ROOTS[@]}" \
    --python "$PY" \
    --summary-out "checkpoints/velocity/summary_v2_$(date +%Y%m%d_%H%M%S).json" \
    2>&1 | tee -a logs/postrun_velocity_v2.log

echo "[run_velocity_v2_seeds] DONE. Summary written under checkpoints/velocity/."
