#!/usr/bin/env bash
# Train the clip_actions=100 arm for 3 seeds SEQUENTIALLY, then export+eval+gate every seed
# including the clip=1.0 control (seed 42 from run_velocity_seeds.sh).
#
# Sequential, not concurrent: measured 2026-09-25 on this GPU that 2 concurrent
# Isaac Sim/PhysX processes give ~98k combined env-steps/s (64k + 33k) versus
# ~140-157k solo -- concurrency LOSES ~35-40% of total throughput on this card
# (no MPS), so sequential at full num_envs is what actually maximizes total
# env-steps/s, not concurrent smaller runs.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=/home/yusuf/Sim/isaac-sim-venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

SEEDS=(42 43 44)
MAX_ITER=12000
NUM_ENVS=4096
SAVE_INTERVAL=500
ACTION_CLIP=100.0
RUN_ROOTS=("checkpoints/velocity/walk_v1_seed42")  # clip=1.0 control, already trained

for SEED in "${SEEDS[@]}"; do
    RUN_NAME="walk_v1_clip100_seed${SEED}"
    RUN_ROOT="checkpoints/velocity/${RUN_NAME}"
    RUN_ROOTS+=("$RUN_ROOT")
    echo "=================================================================="
    echo "[run_velocity_clip100_seeds] starting seed ${SEED} (clip_actions=${ACTION_CLIP}) -> ${RUN_ROOT}"
    echo "=================================================================="
    "$PY" scripts/train_velocity.py \
        --run-name "$RUN_NAME" \
        --seed "$SEED" \
        --num-envs "$NUM_ENVS" \
        --max-iterations "$MAX_ITER" \
        --save-interval "$SAVE_INTERVAL" \
        --action-clip "$ACTION_CLIP" \
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
