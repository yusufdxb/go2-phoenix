#!/usr/bin/env bash
# Stage W s_train pilot (EXPERIMENT.md phase 1, amendment 11): the W2 walking baseline
# through the deploy path, RR_thigh degraded by the deploy gate exactly as on hardware.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/_activate.sh
export PYTHONUNBUFFERED=1
OUT="${OUT:-results/phoenix_v2/walk_strain_pilot}"
N="${N:-128}"
for s in 1.0 0.8 0.7 0.6 0.5; do
  name="rr_thigh_${s/./p}"
  [ -f "$OUT/$name/summary.json" ] && { echo "skip $name"; continue; }
  echo "=== $name"
  extra=()
  [ "$s" != "1.0" ] && extra=(--degrade "RR_thigh:$s" --allow-degradation)
  PYTHONPATH=src python scripts/phoenix_v2_sim2sim_deploy.py --walk \
    --deploy-config configs/sim2real/deploy_walk_w2_sim.yaml \
    --env-config configs/env/phoenix_v2/walk_deploy_a_nominal.yaml --num-envs "$N" \
    --seed 3101 --limiter-max-delta-override "${DQ:-0.6}" \
    --telemetry-envs "${TEL:-4}" --out "$OUT/$name" --label "$name" "${extra[@]}" 2>&1 | grep -v Warp | sed -n '/^{/,/^}/p' \
    | python3 -c "
import json,sys
s=json.load(sys.stdin)
print({k:(round(v,4) if isinstance(v,float) else v) for k,v in s.items() if k in
 ('label','walk_primary_score_mean','walk2_success_rate','success_rate','altered_fraction','gate_faults')})"
done
