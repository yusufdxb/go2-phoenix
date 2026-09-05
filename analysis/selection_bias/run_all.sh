#!/usr/bin/env bash
# Reproduce every selection-bias number quoted in paper/onset_residual_limitation.md.
#
#   ./run_all.sh            # rebuild the frame, run every test, tee to RESULTS.txt
#
# build_frame.py is read-only on the frozen v2 artifacts and rewrites blocks.csv /
# envs.csv deterministically; the rest read only those two files. Nothing here
# writes outside this directory.
set -euo pipefail
cd "$(dirname "$0")"
OUT=RESULTS.txt
{
  echo "# Phoenix selection-bias evidence, regenerated $(date -Is)"
  echo "# repo HEAD: $(git -C ../.. rev-parse HEAD)"
  echo "# registry:  reliability_eval/causal_viability_replication_v2/registry.json"
  echo
  python build_frame.py
  for s in tests_predicate.py tests_1_5.py tests_6.py tests_extrap.py \
           tests_perm_sens.py tests_final.py \
           adv_1_exposure.py adv_2_estimand.py adv_3_batch.py adv_4_inference.py; do
    [ -f "$s" ] || continue
    echo; echo "################ $s ################"
    python "$s"
  done
} 2>&1 | tee "$OUT"
echo "wrote $OUT"
