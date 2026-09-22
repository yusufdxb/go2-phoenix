#!/usr/bin/env bash
# Stage W intervention screening (EXPERIMENT.md amendment 13,
# docs/research/INTERVENTION_SCREENING.md).
#
# Frozen W2 through the exact deploy stack, DR off, the deploy ActuatorGate applying a
# uniform kp/kd reduction to a named joint set. One cell = (family, severity) x seed.
#
#   scripts/phoenix_v2_intervention_screen.sh [out_dir]
#
# Re-runnable: a cell whose summary.json exists is skipped, so an interrupted screen
# resumes where it stopped. Nothing here is deleted.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/_activate.sh
export PYTHONUNBUFFERED=1

OUT="${1:-results/phoenix_v2/intervention_screen}"
CKPT_DIR="checkpoints/phoenix-walk-w2/2026-09-22_13-34-53"
DCFG="configs/sim2real/deploy_walk_w2_sim.yaml"
ECFG="configs/env/phoenix_v2/walk_deploy_a_nominal.yaml"
SEEDS=(7001 7002 7003)
NUM_ENVS="${NUM_ENVS:-128}"
DURATION="${DURATION:-20}"
LIMITER_OVR="${LIMITER_OVR:-0.6}"
TELEMETRY_ENVS="${TELEMETRY_ENVS:-4}"
# EXPERIMENT.md amendment 14: the saturation latch's standing band (0.175 rad) fires on a
# HEALTHY bang-bang walking policy, so the walking band is used for every walking cell.
PIN_BAND="${PIN_BAND:-0.65}"

# Amendment 13.1: the screening must load the frozen artifacts, nothing else.
EXPECT_ONNX_SHA=fd0d3f30873654530462fc40b8917267202cc779d85d69ee6eaa42782fc0fdbb
GOT=$(sha256sum "$CKPT_DIR/policy.onnx" | cut -d' ' -f1)
[ "$GOT" = "$EXPECT_ONNX_SHA" ] || { echo "W2 ONNX sha mismatch: $GOT" >&2; exit 1; }

# cell name : --degrade expression.  "nominal" is the shared s=1.0 reference and takes
# no --degrade at all (a 1.0 spec would still arm the gate's degradation bookkeeping).
CELLS=(
  "nominal:"
  "c1_rr_thigh_0p80:RR_thigh:0.80"  "c1_rr_thigh_0p70:RR_thigh:0.70"
  "c1_rr_thigh_0p60:RR_thigh:0.60"  "c1_rr_thigh_0p50:RR_thigh:0.50"
  "c2_leg_rr_0p90:leg_RR:0.90"  "c2_leg_rr_0p85:leg_RR:0.85"
  "c2_leg_rr_0p80:leg_RR:0.80"  "c2_leg_rr_0p75:leg_RR:0.75"
  "c2_leg_rr_0p70:leg_RR:0.70"
  "c3_rear_0p90:rear:0.90"  "c3_rear_0p85:rear:0.85"
  "c3_rear_0p80:rear:0.80"  "c3_rear_0p75:rear:0.75"
  "c3_rear_0p70:rear:0.70"
  "c4_all_0p90:all:0.90"  "c4_all_0p85:all:0.85"
  "c4_all_0p80:all:0.80"  "c4_all_0p75:all:0.75"
  "c4_all_0p70:all:0.70"
)

total=$(( ${#CELLS[@]} * ${#SEEDS[@]} )); done_n=0
echo "screening ${#CELLS[@]} cells x ${#SEEDS[@]} seeds = $total runs -> $OUT"
for entry in "${CELLS[@]}"; do
  name="${entry%%:*}"; spec="${entry#*:}"
  for seed in "${SEEDS[@]}"; do
    done_n=$((done_n+1))
    dir="$OUT/$name/seed$seed"
    [ -f "$dir/summary.json" ] && { echo "[$done_n/$total] skip $name seed$seed"; continue; }
    rm -rf "$dir"   # a half-written cell from an interrupted run; --out needs a fresh dir
    echo "=== [$done_n/$total] $name seed=$seed ${spec:+degrade=$spec}"
    PYTHONPATH=src python scripts/phoenix_v2_sim2sim_deploy.py --walk \
      --deploy-config "$DCFG" --env-config "$ECFG" \
      --onnx "$CKPT_DIR/policy.onnx" \
      --num-envs "$NUM_ENVS" --seed "$seed" --duration-s "$DURATION" \
      --limiter-max-delta-override "$LIMITER_OVR" \
      --telemetry-envs "$TELEMETRY_ENVS" \
      ${spec:+--degrade "$spec" --allow-degradation --degradation-pin-band "$PIN_BAND"} \
      --out "$dir" --label "$name" 2>&1 \
      | grep -v Warp | sed -n '/^{/,/^}/p' | python3 -c "
import json,sys
s=json.load(sys.stdin)
print({k:(round(v,4) if isinstance(v,float) else v) for k,v in s.items() if k in
 ('label','walk2_success_rate','walk_primary_score_mean','fidelity_pass_rate',
  'safety_hold_episode_rate','attitude_violation_episode_rate','gate_faults')})"
  done
done
echo "screening complete -> $OUT"
