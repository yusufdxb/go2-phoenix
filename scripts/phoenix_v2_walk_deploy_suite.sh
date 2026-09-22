#!/usr/bin/env bash
# Phase 9 for a walking candidate: the exact deploy stack (ONNX + deploy obs + action map
# + ActuatorGate) around Isaac Lab, conditions A to E of amendment 9.
#   scripts/phoenix_v2_walk_deploy_suite.sh <deploy_config> <out_dir> [limiter_override]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/_activate.sh
export PYTHONUNBUFFERED=1
DCFG="${1:?deploy config}"; OUT="${2:?out dir}"; OVR="${3:-}"
i=0
for c in a_nominal:6101 b_dr:6102 c_friction_low:6103 c_friction_high:6104 \
         d_actuator_weak:6105 d_actuator_strong:6106 e_command_corners:6107; do
  n=${c%%:*}; seed=${c#*:}; i=$((i+1))
  [ -f "$OUT/$n/summary.json" ] && { echo "skip $n"; continue; }
  echo "=== $n seed=$seed"
  PYTHONPATH=src python scripts/phoenix_v2_sim2sim_deploy.py --walk \
    --deploy-config "$DCFG" --env-config "configs/env/phoenix_v2/walk_deploy_$n.yaml" \
    --num-envs "${NUM_ENVS:-64}" --seed "$seed" --duration-s "${DURATION:-20}" \
    --out "$OUT/$n" --label "$n" ${OVR:+--limiter-max-delta-override "$OVR"} 2>&1 \
    | grep -v Warp | sed -n '/^{/,/^}/p' | python3 -c "
import json,sys
s=json.load(sys.stdin)
print({k:(round(v,4) if isinstance(v,float) else v) for k,v in s.items() if k in
 ('label','walk2_success_rate','success_rate','altered_fraction','fidelity_pass_rate',
  'hw_gate_pass_rate','safety_hold_episode_rate','gate_faults','walk2_mean_lin_err_m_s')})"
done
