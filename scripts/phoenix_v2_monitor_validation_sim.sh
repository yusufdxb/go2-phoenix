#!/usr/bin/env bash
# Phase 1a (EXPERIMENT.md): monitor validation in simulation through the simulated
# deployment path. Usage: scripts/phoenix_v2_monitor_validation_sim.sh <s_train>
set -euo pipefail
S="${1:?s_train}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/_activate.sh
export PYTHONUNBUFFERED=1
OUT="${OUT:-results/phoenix_v2/monitor_validation_sim}"
run() {  # name n duration seed [degrade]
  local name=$1 n=$2 dur=$3 seed=$4 deg=${5:-}
  [ -f "$OUT/$name/summary.json" ] && { echo "skip $name"; return; }
  echo "=== $name"
  local extra=()
  [ -n "$deg" ] && extra=(--degrade "$deg" --allow-degradation)
  PYTHONPATH=src python scripts/phoenix_v2_sim2sim_deploy.py \
    --deploy-config configs/sim2real/deploy_stand_h25_v2.yaml \
    --env-config configs/env/phoenix_v2/sim2sim_nominal.yaml --num-envs "$n" \
    --duration-s "$dur" --seed "$seed" --out "$OUT/$name" --label "$name" "${extra[@]}" \
    2>&1 | grep -v Warp | sed -n '/^{/,/^}/p' | grep -E 'label|success_rate|hw_gate'
}
run calibration 10 60 5001
run false_positive 20 120 5002
run rr_thigh_s_train 10 60 5003 "RR_thigh:$S"
run rr_thigh_1p0 10 60 5004 "RR_thigh:1.0"
run fl_calf_s_train 10 60 5005 "FL_calf:$S"
run rl_hip_s_train 10 60 5006 "RL_hip:$S"
PYTHONPATH=src python scripts/phoenix_v2_monitor_validation.py --root "$OUT" --s-train "$S"
