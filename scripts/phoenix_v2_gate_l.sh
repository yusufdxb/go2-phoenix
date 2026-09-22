#!/usr/bin/env bash
# Gate L (EXPERIMENT.md amendment 3) for one walking checkpoint.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/_activate.sh
export PYTHONUNBUFFERED=1
CKPT="${1:?checkpoint}"
OUT="${OUT:-results/phoenix_v2/gate_l}"
for c in nominal:4001 dr:4002 stand:4001; do
  n=${c%%:*}; seed=${c#*:}
  [ -f "$OUT/$n/summary.json" ] && continue
  echo "=== $n"
  PYTHONPATH=src python scripts/phoenix_v2_sim_stand.py --checkpoint "$CKPT" --walk \
    --env-config configs/env/phoenix_v2/walk_eval_$n.yaml --num-envs 256 --seed $seed \
    --out "$OUT/$n" --label "gate_l_$n" 2>&1 | grep -v Warp | sed -n '/^{/,/^}/p'
done
