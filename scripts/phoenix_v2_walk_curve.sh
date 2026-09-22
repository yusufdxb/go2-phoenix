#!/usr/bin/env bash
# Directional learning curve over EXISTING checkpoints of one run (diagnostic, dev seed 5001).
#   scripts/phoenix_v2_walk_curve.sh <run_dir> <out_dir> <iter>...
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/_activate.sh
export PYTHONUNBUFFERED=1
RUN="${1:?run dir}"; OUT="${2:?out}"; shift 2
for it in "$@"; do
  d="$OUT/it$(printf %04d "$it")"
  [ -f "$d/summary.json" ] && continue
  PYTHONPATH=src python scripts/phoenix_v2_sim_stand.py --checkpoint "$RUN/model_$it.pt" --walk \
    --env-config configs/env/phoenix_v2/walk_w_eval_nominal.yaml --num-envs "${NUM_ENVS:-128}" \
    --seed 5001 --save-steps --out "$d" --label "curve_it$it" > "$d.log" 2>&1 || echo "FAILED $it"
  PYTHONPATH=src python - "$d" <<'PY'
import json, sys
import numpy as np
from phoenix.monitor.walk_directional import directional_report
d = sys.argv[1]
z = dict(np.load(f"{d}/steps.npz"))
eps = [json.loads(l) for l in open(f"{d}/episodes.jsonl")]
rep = directional_report(z, eps, dt=0.02)
json.dump(rep, open(f"{d}/directional.json", "w"), indent=1)
print(d, {k: (v.get("n_segments"), round(v.get("vx_achieved", float("nan")), 3), round(v.get("success", float("nan")), 2)) for k, v in rep["bins"].items()}, flush=True)
PY
  rm -f "$d/steps.npz"
done
