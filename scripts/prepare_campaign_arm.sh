#!/usr/bin/env bash
# Prepare ONE policy as a hardware-campaign arm, the way arm B of the 2026-10-07 A/B was
# prepared (4d4c7ed): read-only archive copy, stand + walking deploy configs that differ from the
# incumbent's only in the four policy paths, deploy locks, E2 sim references. No motion.
#
#   scripts/prepare_campaign_arm.sh <tag> <checkpoint.pt> <run-root> <export-dir>
#
# <export-dir> is the gated export of that checkpoint (scripts/eval_friction_candidates.sh:
# results/friction_model/candidates/<label>/export with policy.onnx(.data), policy.pt,
# phoenix_manifest.json, parity_gate.json). <tag> names everything, e.g. fm_b60_3200.
# Writes configs/sim2real/deploy_walk_<tag>_stand.yaml, configs/sim2real/proposed/deploy_walk_<tag>.yaml,
# configs/sim2real/locks/deploy_walk_<tag>{,_stand}.lock.yaml, configs/sim2real/e2/walk_<tag>/.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PY:-/home/yusuf/Sim/isaac-sim-venv/bin/python}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$HOME/Archive/phoenix-friction-candidates-2026-10-17}"
TAG="$1"; CKPT="$(realpath "$2")"; RUN_ROOT="$(realpath "$3")"; EXPORT="$(realpath "$4")"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
halt() { printf 'HALT: %s\n' "$*" >&2; exit 1; }
python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$EXPORT/parity_gate.json" \
    || halt "$EXPORT/parity_gate.json is not a passed parity gate"

# 1. read-only archive: the configs point here, never at a results/ or checkpoints/ path
A="$ARCHIVE_ROOT/$TAG"
[[ ! -e "$A" ]] || halt "$A exists; archives are write-once"
mkdir -p "$A/checkpoint" "$A/eval"
cp "$CKPT" "$A/checkpoint/"
for f in spec.json curriculum_final_state.json arm_label.txt realized_dr_dump.json; do
    [[ -f "$RUN_ROOT/$f" ]] && cp "$RUN_ROOT/$f" "$A/checkpoint/"
done
cp "$(dirname "$CKPT")/spec.json" "$A/checkpoint/run_spec.json" 2>/dev/null || true
for f in policy.onnx policy.onnx.data policy.pt phoenix_manifest.json parity_gate.json result.json; do
    cp "$EXPORT/$f" "$A/eval/"
done
(cd "$A" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
chmod -R a-w "$A"
CK="$A/checkpoint/$(basename "$CKPT")"

# 2. deploy configs: the incumbent's with the four policy paths swapped, nothing else
swap() {  # <incumbent cfg> <out>
    { echo "# Friction-model campaign arm $TAG ($(basename "$CKPT")). Same as $(basename "$1") except the four policy paths."
      sed -e "s#^\(  onnx_path: \).*#\1\"$A/eval/policy.onnx\"#" \
          -e "s#^\(  manifest_path: \).*#\1\"$A/eval/phoenix_manifest.json\"#" \
          -e "s#^\(  torchscript_path: \).*#\1\"$A/eval/policy.pt\"#" \
          -e "s#^\(  checkpoint_path: \).*#\1\"$CK\"#" "$1"; } > "$2"
    local n
    n="$(diff "$1" "$2" | grep -c '^>' || true)"
    [[ "$n" == 5 ]] || halt "$2: expected the header + 4 path lines to differ, got $n"
}
STAND_CFG="configs/sim2real/deploy_walk_${TAG}_stand.yaml"
WALK_CFG="configs/sim2real/proposed/deploy_walk_${TAG}.yaml"
swap configs/sim2real/deploy_walk_iter2750_stand.yaml "$STAND_CFG"
swap configs/sim2real/proposed/deploy_walk_iter2750.yaml "$WALK_CFG"

# 3. locks (same commands as the 2026-10-07 arm B locks)
PAR=configs/sim2real/parity/walk_iter2750_v2_synthetic_obs.npz:npz:obs
STAND_LOCK="configs/sim2real/locks/deploy_walk_${TAG}_stand.lock.yaml"
WALK_LOCK="configs/sim2real/locks/deploy_walk_${TAG}.lock.yaml"
NAME="walk-${TAG//_/-}"
"$PY" -m phoenix.sim2real.deploy_contract lock --config "$STAND_CFG" --out "$STAND_LOCK" --name "$NAME-stand" \
    --checkpoint "$CK" --parity-input "$PAR" --canonical-bench
"$PY" -m phoenix.sim2real.deploy_contract lock --config "$WALK_CFG" --out "$WALK_LOCK" --name "$NAME" \
    --checkpoint "$CK" --parity-input "$PAR"

# 4. E2 sim references (nominal + the two boundary plants)
E2="configs/sim2real/e2/walk_${TAG}"
mkdir -p "$E2"
"$PY" scripts/actuator_check.py sim-reference --deploy-config "$STAND_CFG" --lock "$STAND_LOCK" --out "$E2/ref_nominal.json"
"$PY" scripts/actuator_check.py sim-reference --factor calf=0.8 --deploy-config "$STAND_CFG" --lock "$STAND_LOCK" --out "$E2/ref_calf_0.80.json"
"$PY" scripts/actuator_check.py sim-reference --factor all=0.75 --deploy-config "$STAND_CFG" --lock "$STAND_LOCK" --out "$E2/ref_all_0.75.json"
echo "arm $TAG prepared: $A, $STAND_CFG, $WALK_CFG, $STAND_LOCK, $WALK_LOCK, $E2"
