#!/usr/bin/env bash
# Phase D factorial: {trained action clamp on/off} x {measured-q clip / command-rate limit},
# the H25 policy through the simulated deployment path, nominal physics, same seed.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/_activate.sh
export PYTHONUNBUFFERED=1
OUT="${OUT:-results/phoenix_v2/sim2sim_factorial}"
N="${N:-64}"
for f in noclamp_measured_q clamp_measured_q noclamp_prev_command clamp_prev_command; do
  [ -f "$OUT/$f/summary.json" ] && { echo "skip $f"; continue; }
  echo "=== $f"
  PYTHONPATH=src python scripts/phoenix_v2_sim2sim_deploy.py \
    --deploy-config configs/sim2real/deploy_stand_h25.yaml --factorial "$f" \
    --env-config configs/env/phoenix_v2/sim2sim_nominal.yaml --num-envs "$N" --seed 2001 \
    --out "$OUT/$f" --label "$f" 2>&1 | grep -v Warp | sed -n '/^{/,/^}/p'
done
