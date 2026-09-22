#!/usr/bin/env bash
# Preregistered s_train pilot (EXPERIMENT.md phase 1): incumbent + frozen v2 deploy path,
# RR_thigh degraded by the deploy gate exactly as on hardware, nominal physics.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/_activate.sh
export PYTHONUNBUFFERED=1
OUT="${OUT:-results/phoenix_v2/strain_pilot}"
N="${N:-128}"
for s in 1.0 0.8 0.7 0.6 0.5; do
  name="rr_thigh_${s/./p}"
  [ -f "$OUT/$name/summary.json" ] && { echo "skip $name"; continue; }
  echo "=== $name"
  extra=()
  [ "$s" != "1.0" ] && extra=(--degrade "RR_thigh:$s" --allow-degradation)
  PYTHONPATH=src python scripts/phoenix_v2_sim2sim_deploy.py \
    --deploy-config configs/sim2real/deploy_stand_h25_v2.yaml \
    --env-config configs/env/phoenix_v2/sim2sim_nominal.yaml --num-envs "$N" --seed 3001 \
    --out "$OUT/$name" --label "$name" "${extra[@]}" 2>&1 | grep -v Warp | sed -n '/^{/,/^}/p'
done
