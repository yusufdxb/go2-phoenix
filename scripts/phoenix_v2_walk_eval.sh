#!/usr/bin/env bash
# Recipe W evaluation of one walking checkpoint (amendment 6.4) on dev or final seeds.
#   scripts/phoenix_v2_walk_eval.sh <ckpt> <out_dir> dev|final [env_prefix]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/_activate.sh
export PYTHONUNBUFFERED=1
CKPT="${1:?checkpoint}"; OUT="${2:?out dir}"; SPLIT="${3:?dev|final}"
PREFIX="${4:-configs/env/phoenix_v2/walk_w_eval}"
case "$SPLIT" in dev) base=5000 ;; final) base=6000 ;; *) echo "split must be dev|final"; exit 2 ;; esac
EXTRA=()
[ -n "${WALK_THRESHOLDS:-}" ] && EXTRA+=(--walk-thresholds "$WALK_THRESHOLDS")
for c in nominal:1 dr:2 stand:3; do
  n=${c%%:*}; seed=$((base + ${c#*:}))
  [ -f "$OUT/$n/summary.json" ] && { echo "skip $n"; continue; }
  echo "=== $n seed=$seed"
  PYTHONPATH=src python scripts/phoenix_v2_sim_stand.py --checkpoint "$CKPT" --walk \
    --env-config "${PREFIX}_$n.yaml" --num-envs "${NUM_ENVS:-256}" --seed $seed \
    --out "$OUT/$n" --label "$(basename "$OUT")_$n" "${EXTRA[@]}" 2>&1 | grep -v Warp | sed -n '/^{/,/^}/p'
done
