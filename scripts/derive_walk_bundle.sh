#!/usr/bin/env bash
# Derive the WALKING bundle from an already-staged, activated STAND bundle.
#
# Why this exists: scripts/stage_payload_bundle.sh needs a GO workstation stage A for
# the same lock, and stage A refuses any config that is not stand-only by design. The
# walking config differs from its stand-phase twin in exactly one key
# (safety.stand_only), and it reaches the same policy files, so the walking bundle is
# the verified stand bundle with the config swapped. This script proves both facts
# instead of assuming them, then pins and re-verifies the new bundle.
#
# Run it where the bundle will live (normally ON THE PAYLOAD, after
# scripts/stage_payload_repo.sh has synced this checkout there):
#
#   scripts/derive_walk_bundle.sh <stand-bundle-dir> <walk-config> <walk-lock> <dest-dir>
#
# Example on the Jetson:
#   scripts/derive_walk_bundle.sh ~/phoenix/walk2750-stand-<sha> \
#       configs/sim2real/proposed/deploy_walk_iter2750.yaml \
#       configs/sim2real/locks/deploy_walk_iter2750.lock.yaml \
#       ~/phoenix/walk2750-<sha>
#
# It does not decide whether walking is allowed: the bridge and the policy node still
# derive that from the evidence-gated deploy contract (stages F and G on hardware).

set -euo pipefail

STAND="${1:?usage: $0 <stand-bundle-dir> <walk-config> <walk-lock> <dest-dir>}"
WALK_CFG="${2:?usage: $0 <stand-bundle-dir> <walk-config> <walk-lock> <dest-dir>}"
WALK_LOCK="${3:?usage: $0 <stand-bundle-dir> <walk-config> <walk-lock> <dest-dir>}"
DEST="${4:?usage: $0 <stand-bundle-dir> <walk-config> <walk-lock> <dest-dir>}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Never write bytecode into a bundle directory (it would change what the bundle holds).
export PYTHONDONTWRITEBYTECODE=1

halt() { printf '\033[31mHALT: %s\033[0m\n' "$*" >&2; exit 1; }

[[ -d "$STAND" ]] || halt "stand bundle $STAND does not exist"
[[ -f "$WALK_CFG" ]] || halt "walking config $WALK_CFG does not exist"
[[ -f "$WALK_LOCK" ]] || halt "walking lock $WALK_LOCK does not exist"
[[ -e "$DEST" ]] && halt "destination $DEST already exists; refusing to overwrite a bundle"
STAND="$(cd "$STAND" && pwd)"

echo "[walk-bundle] 1/5 stand bundle bytes and activation"
( cd "$STAND" && sha256sum -c --quiet SHA256SUMS ) || halt "stand bundle SHA256SUMS does not verify"
python3 "$STAND/activation.py" verify --bundle "$STAND" >/dev/null || halt "stand bundle activation does not verify"
STAND_CFG="$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import activation; print(activation.find_config(sys.argv[1]))' "$STAND")"

echo "[walk-bundle] 2/5 walking config is the stand config with only stand_only flipped"
python3 - "$STAND_CFG" "$WALK_CFG" <<'PY' || halt "walking config is not the walking twin of the staged stand config"
import sys

import yaml

from phoenix.sim2real.deploy_contract import is_stand_only, semantic_config_sha256, stand_twin

stand = yaml.safe_load(open(sys.argv[1]))
walk = yaml.safe_load(open(sys.argv[2]))
problems = []
if not is_stand_only(stand):
    problems.append("the staged bundle's config is not stand-only")
if is_stand_only(walk):
    problems.append("the walking config is stand-only")
if semantic_config_sha256(stand_twin(walk)) != semantic_config_sha256(stand):
    problems.append("stand twin of the walking config differs from the staged stand config")
for p in problems:
    print(f"  {p}", file=sys.stderr)
sys.exit(1 if problems else 0)
PY

echo "[walk-bundle] 3/5 assemble and pin"
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
for f in "$STAND"/*; do
    [[ -f "$f" ]] || continue
    case "$(basename "$f")" in
        SHA256SUMS | *.yaml | *.yml) ;;
        *) cp -p "$f" "$DEST/" ;;
    esac
done
cp "$WALK_CFG" "$DEST/$(basename "$WALK_CFG")"
python3 "$DEST/activation.py" pin --bundle "$DEST" --target "$DEST"
( cd "$DEST" && sha256sum ./* > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS && sha256sum -c --quiet SHA256SUMS )
python3 "$DEST/activation.py" verify --bundle "$DEST" >/dev/null || halt "walking bundle activation does not verify"

echo "[walk-bundle] 4/5 walking lock against the pinned config and its files"
python3 - "$WALK_LOCK" "$DEST/$(basename "$WALK_CFG")" <<'PY' || halt "walking lock does not verify against the derived bundle"
import sys

import yaml

from phoenix.sim2real.deploy_contract import load_lock, verify_lock

problems, _ = verify_lock(load_lock(sys.argv[1]), yaml.safe_load(open(sys.argv[2])), sys.argv[2])
for p in problems:
    print(f"  {p}", file=sys.stderr)
sys.exit(1 if problems else 0)
PY

echo "[walk-bundle] 5/5 walking evidence (informational; the bridge enforces it)"
python3 - "$DEST/$(basename "$WALK_CFG")" <<'PY' || true
import sys

import yaml

from phoenix.sim2real.deploy_contract import walking_evidence_problems

problems = walking_evidence_problems(yaml.safe_load(open(sys.argv[1])))
print("  walking evidence: " + ("present (stages F and G)" if not problems else "MISSING"))
for p in problems:
    print(f"  {p}")
PY

echo
echo "[walk-bundle] derived $DEST from $STAND: bytes, activation and walking lock verified"
echo "  DEPLOY_CFG=$DEST/$(basename "$WALK_CFG") DEPLOY_LOCK=$WALK_LOCK"
