#!/usr/bin/env bash
# Freeze and run the full causal-viability replication: 12 protocols, 24 arm runs.
#
# The v1 replication (2026-07-29) was driven by hand, so no committed script
# reproduced it from a clean clone. This is that script. It is also the v2 entry
# point: v2 differs from v1 only in --batched-blocks, which gives each scenario
# block its own environments so no simulator state can carry between blocks. See
# reliability_eval/negcontrol_probe_batched/PROBE_RESULT.md for why that was
# needed and what it fixed.
#
# Usage:
#   scripts/reliability_replication_run.sh <out-root> <study-id> [extra flags...]
#
# Example (v2):
#   scripts/reliability_replication_run.sh \
#       reliability_eval/causal_viability_replication_v2 \
#       phoenix_causal_viability_replication_v2 \
#       --batched-blocks
#
# Requires the Isaac Sim virtualenv python and OMNI_KIT_ACCEPT_EULA=YES.

set -euo pipefail

OUT_ROOT="${1:?usage: $0 <out-root> <study-id> [extra flags...]}"
STUDY_ID="${2:?usage: $0 <out-root> <study-id> [extra flags...]}"
shift 2
EXTRA=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PHOENIX_SIM_PYTHON:-$HOME/Sim/isaac-sim-venv/bin/python}"
CL="$REPO_ROOT/scripts/reliability_closed_loop.py"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Per-policy frozen inputs. The hashes of these files are what bundle.json pins,
# so changing any of them fails the arm run closed rather than silently drifting.
stand_args=(
  --artifact    deploy/shield_stand_v3.npz
  --onnx        deploy/stand_v3_latent.onnx
  --checkpoint  checkpoints/phoenix-stand-v3-h25-final/latest.pt
  --env-config  configs/env/stand_v3_h25.yaml
)
walk_args=(
  --artifact    deploy/shield_flat_v4.npz
  --onnx        deploy/flat_v4_latent.onnx
  --checkpoint  checkpoints/phoenix-flat-v4/latest.pt
  --env-config  configs/env/flat_v4.yaml
)

# cell | policy | disturbance | protocol_seed | process_seed | replicate_id
# Seeds are carried over verbatim from the v1 registry so a re-run under a design
# change is comparable to it block for block.
PROTOCOLS=(
  "stand_motor stand motor 2026074001 2026074101 process_01"
  "stand_obs   stand obs   2026074002 2026074101 process_01"
  "walk_motor  walk  motor 2026074003 2026074101 process_01"
  "walk_obs    walk  obs   2026074004 2026074101 process_01"
  "stand_motor stand motor 2026074005 2026074102 process_02"
  "stand_obs   stand obs   2026074006 2026074102 process_02"
  "walk_motor  walk  motor 2026074007 2026074102 process_02"
  "walk_obs    walk  obs   2026074008 2026074102 process_02"
  "stand_motor stand motor 2026074009 2026074103 process_03"
  "stand_obs   stand obs   2026074010 2026074103 process_03"
  "walk_motor  walk  motor 2026074011 2026074103 process_03"
  "walk_obs    walk  obs   2026074012 2026074103 process_03"
)

cd "$REPO_ROOT"
echo "[run] out root : $OUT_ROOT"
echo "[run] study id : $STUDY_ID"
echo "[run] extra    : ${EXTRA[*]:-<none>}"

for row in "${PROTOCOLS[@]}"; do
  read -r cell policy disturbance protocol_seed process_seed replicate <<<"$row"
  out="$OUT_ROOT/$replicate/$cell"
  if [[ "$policy" == "stand" ]]; then policy_args=("${stand_args[@]}"); else policy_args=("${walk_args[@]}"); fi

  common=(
    --out-dir "$out"
    --disturbance "$disturbance"
    --cell-id "$cell"
    --policy-name "$policy"
    --protocol-seed "$protocol_seed"
    --process-seed "$process_seed"
    --replicate-id "$replicate"
    --study-id "$STUDY_ID"
    --envs 16 --n-disturbed 32 --n-nominal 16 --horizon 500
    "${policy_args[@]}"
    "${EXTRA[@]}"
  )

  echo "=== [$replicate/$cell] freeze"
  "$PY" "$CL" --freeze "${common[@]}"
  for arm in unshielded oracle; do
    echo "=== [$replicate/$cell] arm $arm"
    "$PY" "$CL" --arm "$arm" "${common[@]}" </dev/null
  done
done

echo "[run] all 12 protocols complete under $OUT_ROOT"
