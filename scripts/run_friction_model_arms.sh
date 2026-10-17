#!/usr/bin/env bash
# Friction-model study arms: B (vnext3hw, hardware-identified joint friction) and C
# (vnext3broad, joint-agnostic broad friction DR). Same base (model_2750), same faithful
# resume + curriculum state, same +1000 iterations, same seeds: only the friction DR differs.
# Interleaved B/C per seed so partial results stay paired. Run inside tmux / nohup.
#   PY=... BASE=... OUT=... SEEDS="60 61 62" ARMS="B C" ITERS=3750 scripts/run_friction_model_arms.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY=${PY:-/home/yusuf/Sim/isaac-sim-venv/bin/python}
BASE=${BASE:-/home/yusuf/Archive/phoenix-deploy-candidates-2026-10-03/checkpoints/walk_vnext3c_seed47}
OUT=${OUT:-checkpoints/friction_model}
ITERS=${ITERS:-3750}
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
mkdir -p logs/friction_model "$OUT"
for SEED in ${SEEDS:-60 61 62}; do
  for ARM in ${ARMS:-B C}; do
    SPEC=$([[ $ARM == B ]] && echo vnext3hw || echo vnext3broad)
    RUN="fm_${ARM}_s${SEED}"
    if ls "$OUT"/$RUN/*/model_$((ITERS - 1)).pt >/dev/null 2>&1; then
      echo "[friction_model] $RUN already done, skipping"; continue
    fi
    echo "================ $RUN ($SPEC) $(date -u +%FT%TZ) host $(hostname) ================"
    "$PY" scripts/train_velocity.py --spec "$SPEC" --run-name "$RUN" --seed "$SEED" \
      --output-dir "$OUT" --max-iterations "$ITERS" \
      --resume "$BASE/2026-09-26_21-06-20/model_2750.pt" --resume-continue-iterations \
      --faithful-resume --resume-curriculum-state "$BASE/curriculum_final_state.json" \
      --arm-label "friction-model-$ARM" --headless 2>&1 | tee -a "logs/friction_model/$RUN.log"
    echo "[friction_model] $RUN exit ${PIPESTATUS[0]} $(date -u +%FT%TZ)"
  done
done
echo "[friction_model] ALL DONE $(date -u +%FT%TZ)"
