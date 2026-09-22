#!/usr/bin/env bash
# Phase D suite: v2 vs legacy deploy config through the simulated deployment path.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/_activate.sh
export PYTHONUNBUFFERED=1
OUT="${OUT:-results/phoenix_v2/sim2sim}"
N="${N:-64}"
for dep in v2:configs/sim2real/deploy_stand_h25_v2.yaml legacy:configs/sim2real/deploy_stand_h25.yaml; do
  tag=${dep%%:*}; cfg=${dep#*:}
  for cond in nominal:2001 dr:2002; do
    c=${cond%%:*}; seed=${cond#*:}
    name="${tag}_${c}"
    [ -f "$OUT/$name/summary.json" ] && { echo "skip $name"; continue; }
    echo "=== $name"
    PYTHONPATH=src python scripts/phoenix_v2_sim2sim_deploy.py --deploy-config "$cfg" \
      --env-config "configs/env/phoenix_v2/sim2sim_${c}.yaml" --num-envs "$N" --seed "$seed" \
      --out "$OUT/$name" --label "$name" --save-steps 2>&1 | grep -v Warp | sed -n '/^{/,/^}/p'
  done
done
