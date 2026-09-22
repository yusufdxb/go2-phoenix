#!/usr/bin/env bash
# Phase B/C development sweep: incumbent H25 checkpoint x limiter configs x {dr, nominal}.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/_activate.sh
export PYTHONUNBUFFERED=1
CKPT="${CKPT:-checkpoints/phoenix-stand-h25-lat-noise/2026-06-22_21-08-20/model_799.pt}"
OUT="${OUT:-results/phoenix_v2/sim_limiter}"
NUM_ENVS="${NUM_ENVS:-256}"
for cfg in configs/env/phoenix_v2/limiter_sweep/*.yaml; do
  name="$(basename "$cfg" .yaml)"
  case "$name" in dr_*) seed=1001 ;; *) seed=1002 ;; esac
  [ -f "$OUT/$name/summary.json" ] && { echo "skip $name"; continue; }
  echo "=== $name seed=$seed"
  PYTHONPATH=src python scripts/phoenix_v2_sim_stand.py --checkpoint "$CKPT" \
    --env-config "$cfg" --num-envs "$NUM_ENVS" --seed "$seed" --out "$OUT/$name" \
    --label "$name" 2>&1 | grep -v Warp | sed -n '/^{/,/^}/p'
done
