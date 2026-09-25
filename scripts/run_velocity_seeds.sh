#!/usr/bin/env bash
# Train PhoenixVelocity for 3 seeds sequentially, then export + eval every seed.
# Meant to run inside tmux session "phoenix-walk" (see feedback_show_training_output).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=/home/yusuf/Sim/isaac-sim-venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

SEEDS=(42 43 44)
MAX_ITER=6000
NUM_ENVS=4096
RUN_ROOTS=()

for SEED in "${SEEDS[@]}"; do
    RUN_NAME="walk_v1_seed${SEED}"
    RUN_ROOT="checkpoints/velocity/${RUN_NAME}"
    RUN_ROOTS+=("$RUN_ROOT")
    echo "=================================================================="
    echo "[run_velocity_seeds] starting seed ${SEED} -> ${RUN_ROOT}"
    echo "=================================================================="
    "$PY" scripts/train_velocity.py \
        --run-name "$RUN_NAME" \
        --seed "$SEED" \
        --num-envs "$NUM_ENVS" \
        --max-iterations "$MAX_ITER" \
        --headless \
        2>&1 | tee -a "logs/train_${RUN_NAME}.log"
    RC=${PIPESTATUS[0]}
    echo "[run_velocity_seeds] seed ${SEED} training exited with code ${RC}"
    if [[ $RC -ne 0 ]]; then
        echo "[run_velocity_seeds] seed ${SEED} FAILED, continuing to next seed anyway"
    fi
done

echo "=================================================================="
echo "[run_velocity_seeds] all seeds done, running postrun export+eval chain"
echo "=================================================================="
"$PY" scripts/postrun_velocity.py \
    --run-roots "${RUN_ROOTS[@]}" \
    --python "$PY" \
    --summary-out "checkpoints/velocity/summary_$(date +%Y%m%d_%H%M%S).json" \
    2>&1 | tee -a logs/postrun_velocity.log

echo "[run_velocity_seeds] DONE. Summary written under checkpoints/velocity/."
