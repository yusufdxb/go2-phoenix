#!/usr/bin/env bash
# Preregistered friction-DR study (docs/preregistration/2026-10-07-joint-friction-dr.md):
# arms T (vnext3f) and C (vnext3), seeds 50-54, faithful resume of model_2750 to 3750.
# Interleaved T/C per seed so partial results stay paired. Run inside tmux.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY=/home/yusuf/Sim/isaac-sim-venv/bin/python
BASE=/home/yusuf/Archive/phoenix-deploy-candidates-2026-10-03/checkpoints/walk_vnext3c_seed47
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
mkdir -p logs/friction_dr
for SEED in ${SEEDS:-50 51 52 53 54}; do
  for ARM in T C; do
    SPEC=$([[ $ARM == T ]] && echo vnext3f || echo vnext3)
    RUN="fric_${ARM}_s${SEED}"
    if [[ -f "checkpoints/friction_dr/$RUN/latest.pt" ]] || ls checkpoints/friction_dr/$RUN/*/model_3750.pt >/dev/null 2>&1; then
      echo "[friction_dr] $RUN already done, skipping"; continue
    fi
    echo "================ $RUN ($SPEC) $(date -u +%FT%TZ) ================"
    "$PY" scripts/train_velocity.py --spec "$SPEC" --run-name "$RUN" --seed "$SEED" \
      --output-dir checkpoints/friction_dr --max-iterations 3750 \
      --resume "$BASE/2026-09-26_21-06-20/model_2750.pt" --resume-continue-iterations \
      --faithful-resume --resume-curriculum-state "$BASE/curriculum_final_state.json" \
      --arm-label "friction-dr-$ARM" --headless 2>&1 | tee -a "logs/friction_dr/$RUN.log"
    echo "[friction_dr] $RUN exit ${PIPESTATUS[0]} $(date -u +%FT%TZ)"
  done
done
echo "[friction_dr] ALL DONE $(date -u +%FT%TZ)"
