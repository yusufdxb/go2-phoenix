#!/usr/bin/env bash
# Drive the reliability-shield Phase 3 rollout grid.
#
# Genuinely held-out OOD: training DR randomizes friction [0.3,1.5] and mass
# +/-2 kg, so shifts are pinned BEYOND those ranges. motor_strength_scale is
# not in the training DR at all. Nominal uses the (randomized) training DR, so
# calibration reflects the true in-distribution operating spread.
#
# One condition/seed per Isaac process (crash isolation). Restartable: existing
# .npz are skipped. Flaky Blackwell startup segfaults are retried up to 3x.
set -u
cd "$(dirname "$0")/.."
export OMNI_KIT_ACCEPT_EULA=YES ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/Sim/IsaacLab}"

CKPT="${CKPT:-checkpoints/phoenix-flat/latest.pt}"
ENVCFG="${ENVCFG:-configs/env/flat.yaml}"
OUT="${OUT:-reliability_eval/raw}"
LOGS="${LOGS:-reliability_eval/logs}"
ENVS="${ENVS:-256}"
STEPS="${STEPS:-300}"
SEEDS="${SEEDS:-0 1 2}"
PY=~/Sim/isaac-sim-venv/bin/python
mkdir -p "$OUT" "$LOGS"

# Each condition pins ALL physics factors (single-factor control): a clean,
# deterministic nominal (friction 0.9, no added mass, no push) plus one
# held-out shift. Training DR randomized friction [0.3,1.5] and mass +/-2 kg,
# so these shifts are strictly outside it; motor_strength_scale was never
# randomized in training.
NOM='"friction_range":[0.9,0.9],"restitution_range":[0.0,0.0],"mass_offset_kg":[0.0,0.0],"push_velocity_xy":0.0,"push_velocity_yaw":0.0'
declare -A DR
DR[nominal]="{$NOM}"
DR[mass_moderate]="{$NOM,\"mass_offset_kg\":[4.0,4.0]}"
DR[mass_severe]="{$NOM,\"mass_offset_kg\":[7.0,7.0]}"
DR[friction_moderate]="{$NOM,\"friction_range\":[0.2,0.2]}"
DR[friction_severe]="{$NOM,\"friction_range\":[0.06,0.06]}"
DR[motor_moderate]="{$NOM,\"motor_strength_scale\":[0.6,0.6]}"
DR[motor_severe]="{$NOM,\"motor_strength_scale\":[0.45,0.45]}"

CONDS="${CONDS:-nominal mass_moderate mass_severe friction_moderate friction_severe motor_moderate motor_severe}"

run_one() {
  local cond=$1 seed=$2 dr=$3
  local out="$OUT/${cond}_seed${seed}.npz"
  # Skip only if the existing rollout was produced with the SHAPE WE ASKED FOR.
  # A bare -f test silently preserved a stale 64-env/300-step smoke rollout
  # across a grid re-run at 128/400, which then misrepresented the protocol in
  # every downstream number fit from it.
  if [ -f "$out" ]; then
    if [ -f "${out%.npz}.meta.json" ] && "$PY" -c "
import json,sys
m=json.load(open('${out%.npz}.meta.json'))
sys.exit(0 if m.get('num_envs')==$ENVS and m.get('max_steps')==$STEPS else 1)
" 2>/dev/null; then
      echo "[grid] skip existing $out"; return 0
    fi
    echo "[grid] STALE $out (shape != ${ENVS}x${STEPS}) - re-running"
    rm -f "$out" "${out%.npz}.meta.json"
  fi
  for attempt in 1 2 3; do
    echo "[grid] RUN $cond seed$seed attempt$attempt $(date +%H:%M:%S)"
    PYTHONPATH=src PYTHONUNBUFFERED=1 timeout 600 "$PY" scripts/reliability_rollout.py \
      --checkpoint "$CKPT" --env-config "$ENVCFG" --condition "$cond" \
      --dr-override "$dr" --out "$out" --num-envs "$ENVS" --max-steps "$STEPS" --seed "$seed" \
      > "$LOGS/${cond}_seed${seed}.log" 2>&1
    if [ -f "$out" ]; then echo "[grid] OK $out"; return 0; fi
    echo "[grid] FAIL $cond seed$seed attempt$attempt (tail: $(tail -1 "$LOGS/${cond}_seed${seed}.log"))"
  done
  echo "[grid] GIVEUP $cond seed$seed"; return 1
}

for cond in $CONDS; do
  for seed in $SEEDS; do
    run_one "$cond" "$seed" "${DR[$cond]}"
  done
done
echo "[grid] DONE $(date +%H:%M:%S)"
