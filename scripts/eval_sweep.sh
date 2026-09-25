#!/usr/bin/env bash
# Evaluate pre-collapse checkpoints for the 3 overnight clip-100 seeds.
# Sequential (proven best throughput, no concurrent Isaac Sim processes).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=/home/yusuf/Sim/isaac-sim-venv/bin/python
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

SEEDS=(42 43 44)
ITERS=(1000 2000 3000 4000)
OUT_DIR="logs/eval_sweep"
mkdir -p "$OUT_DIR"

for SEED in "${SEEDS[@]}"; do
    CKPT_DIR="checkpoints/velocity/walk_v1_clip100_seed${SEED}"
    for IT in "${ITERS[@]}"; do
        CKPT="$(find "$CKPT_DIR" -maxdepth 2 -name "model_${IT}.pt" | head -n1)"
        if [[ -z "$CKPT" ]]; then
            echo "SKIP seed${SEED} iter${IT}: checkpoint not found"
            continue
        fi
        OUT="${OUT_DIR}/seed${SEED}_iter${IT}.json"
        if [[ -f "$OUT" ]]; then
            echo "SKIP seed${SEED} iter${IT}: already evaluated"
            continue
        fi
        echo "=== eval seed${SEED} iter${IT} ($CKPT) ==="
        "$PY" scripts/eval_velocity.py \
            --checkpoint "$CKPT" \
            --num-envs 256 \
            --num-steps 1050 \
            --out "$OUT" \
            --headless \
            > "${OUT_DIR}/seed${SEED}_iter${IT}.log" 2>&1
        RC=$?
        echo "seed${SEED} iter${IT} exit code ${RC}"
    done
done
echo "SWEEP DONE"
