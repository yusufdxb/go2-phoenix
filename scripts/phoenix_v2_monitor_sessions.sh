#!/usr/bin/env bash
# Phase I/J monitor session sets: long nominal and degraded walking sessions, frozen W2
# through the exact deploy stack, one independent session per simulated robot.
#
#   scripts/phoenix_v2_monitor_sessions.sh <out_dir> <set_name> <seed_a> <seed_b> [degrade_spec]
#
# The screening's cells are 20 s, which gives about 19 monitor windows per session: far
# too thin to set a quantile threshold on. These are 120 s, about 119 windows each, which
# is the depth the preregistered monitor protocol assumed. `episode_length_s` is stretched
# to cover the duration by the deploy script, so a session has no mid-run reset and policy
# authority is continuous throughout.
#
# Seeds are passed in so the calibration / false-positive / degraded sets stay disjoint,
# which the evaluator enforces.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/_activate.sh
export PYTHONUNBUFFERED=1

OUT="${1:?out dir}"; SET="${2:?set name}"; SEED_A="${3:?seed a}"; SEED_B="${4:?seed b}"
SPEC="${5:-}"
CKPT_DIR="checkpoints/phoenix-walk-w2/2026-09-22_13-34-53"
DCFG="configs/sim2real/deploy_walk_w2_sim.yaml"
ECFG="configs/env/phoenix_v2/walk_deploy_a_nominal.yaml"
DURATION="${DURATION:-120}"
NUM_ENVS="${NUM_ENVS:-12}"
TELEMETRY_ENVS="${TELEMETRY_ENVS:-12}"
LIMITER_OVR="${LIMITER_OVR:-0.6}"
PIN_BAND="${PIN_BAND:-0.65}"   # amendment 14 walking band

for seed in "$SEED_A" "$SEED_B"; do
  dir="$OUT/$SET/seed$seed"
  [ -f "$dir/summary.json" ] && { echo "skip $SET seed$seed"; continue; }
  rm -rf "$dir"
  echo "=== $SET seed=$seed ${SPEC:+degrade=$SPEC} duration=${DURATION}s"
  PYTHONPATH=src python scripts/phoenix_v2_sim2sim_deploy.py --walk \
    --deploy-config "$DCFG" --env-config "$ECFG" \
    --onnx "$CKPT_DIR/policy.onnx" \
    --num-envs "$NUM_ENVS" --seed "$seed" --duration-s "$DURATION" \
    --limiter-max-delta-override "$LIMITER_OVR" \
    --telemetry-envs "$TELEMETRY_ENVS" \
    ${SPEC:+--degrade "$SPEC" --allow-degradation --degradation-pin-band "$PIN_BAND"} \
    --out "$dir" --label "$SET" 2>&1 \
    | grep -v Warp | sed -n '/^{/,/^}/p' | python3 -c "
import json,sys
s=json.load(sys.stdin)
print({k:(round(v,4) if isinstance(v,float) else v) for k,v in s.items() if k in
 ('label','walk2_success_rate','fidelity_pass_rate','safety_hold_episode_rate','gate_faults')})"
done
echo "$SET complete -> $OUT/$SET"
