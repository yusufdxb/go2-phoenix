#!/usr/bin/env bash
# Phase 7: walking limiter selection sweep (amendment 6.2 rule), development seeds only.
#   scripts/phoenix_v2_walk_limiter_sweep.sh <checkpoint>
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/_activate.sh
export PYTHONUNBUFFERED=1
CKPT="${1:?checkpoint}"
OUT="${OUT:-results/phoenix_v2/walk_limiter}"
for cfg in configs/env/phoenix_v2/walk_limiter_sweep/*.yaml; do
  name="$(basename "$cfg" .yaml)"
  case "$name" in dr_*) seed=${SEED_DR:-5101} ;; *) seed=${SEED_NOM:-5102} ;; esac
  [ -f "$OUT/$name/summary.json" ] && { echo "skip $name"; continue; }
  echo "=== $name seed=$seed"
  PYTHONPATH=src python scripts/phoenix_v2_sim_stand.py --checkpoint "$CKPT" --walk \
    --env-config "$cfg" --num-envs "${NUM_ENVS:-256}" --seed "$seed" --out "$OUT/$name" \
    --label "$name" 2>&1 | grep -v Warp | sed -n '/^{/,/^}/p' | python3 -c "
import json,sys
s=json.load(sys.stdin); print({k:(round(v,4) if isinstance(v,float) else v) for k,v in s.items() if k in ('label','walk2_success_rate','altered_fraction','fidelity_pass_rate','walk2_mean_lin_err_m_s')})"
done
