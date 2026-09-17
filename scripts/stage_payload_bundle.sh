#!/usr/bin/env bash
# Assemble the file set that must reach the GO2 payload, and refuse to assemble
# it unless the policy has passed the parity gate.
#
# There is no internet egress at the lab, so whatever is on the transport when
# it leaves the desk is what the session gets. Two failure modes this exists to
# prevent, both of which have a real cost of one lab day:
#
#   1. shipping policy.onnx without policy.onnx.data. The .onnx is a few KB of
#      graph; the sidecar carries every weight. onnxruntime resolves the sidecar
#      relative to the .onnx path, so a missing one fails at load on the payload.
#   2. shipping a policy nobody gated. scripts/parity_gate.py writes
#      parity_gate.json next to the export; this script reads it and stops if it
#      is missing or if passed is not true.
#
# Usage:
#   scripts/stage_payload_bundle.sh <checkpoint-dir> <deploy-cfg> <dest>
#
# <dest> is either a local directory or a remote [user@]host:/path, in which
# case the bundle is assembled locally, rsynced over, and the SHA256SUMS is
# re-verified ON THE PAYLOAD, which is the only check that proves the transfer
# rather than the copy. Host aliases jetson (wifi 192.168.0.70) and
# jetson-cable (192.168.123.18) are already in ~/.ssh/config; the password
# comes from JETSON_PW, default 123.
#
# Examples:
#   scripts/stage_payload_bundle.sh \
#       checkpoints/phoenix-stand-h25-lat-noise \
#       configs/sim2real/deploy_stand_h25.yaml \
#       deploy_staging/phoenix-stand-h25-lat-noise
#
#   scripts/stage_payload_bundle.sh \
#       checkpoints/phoenix-stand-h25-lat-noise \
#       configs/sim2real/deploy_stand_h25.yaml \
#       jetson:/home/unitree/phoenix/stand-h25-lat-noise

set -euo pipefail

CKPT_DIR="${1:?usage: $0 <checkpoint-dir> <deploy-cfg> <dest>}"
DEPLOY_CFG="${2:?usage: $0 <checkpoint-dir> <deploy-cfg> <dest>}"
DEST="${3:?usage: $0 <checkpoint-dir> <deploy-cfg> <dest>}"

# A dest of the form host:/path is remote. A leading / or ./ is always local,
# so an absolute path containing a colon is not mistaken for a host.
REMOTE=""
case "$DEST" in
  /*|./*|../*) ;;
  *:*) REMOTE="$DEST"; DEST="$(mktemp -d)" ;;
esac

GATE="$CKPT_DIR/parity_gate.json"
if [[ ! -f "$GATE" ]]; then
  echo "REFUSING TO STAGE: no $GATE. Run scripts/parity_gate.py first." >&2
  exit 1
fi
if ! grep -q '"passed": true' "$GATE"; then
  echo "REFUSING TO STAGE: $GATE does not record passed true." >&2
  exit 1
fi

# Workstation stage A evidence travels with the bundle. The payload's own stage A
# (scripts/harness_preflight.sh A --payload) refuses to pass without it: the
# checkpoint-side torch parity, the test suite, lint and type checks can only run on
# the workstation, and parity_golden.npz is what lets the payload check its own
# onnxruntime against those torch outputs.
STAGE_A_SESSION="${STAGE_A_SESSION:?set STAGE_A_SESSION to the workstation session directory holding a GO stage_A.json (scripts/harness_preflight.sh A)}"
STAGE_A_JSON="$STAGE_A_SESSION/stage_A.json"
GOLDEN="$STAGE_A_SESSION/parity_golden.npz"
if [[ ! -f "$STAGE_A_JSON" || ! -f "$GOLDEN" ]]; then
  echo "REFUSING TO STAGE: $STAGE_A_JSON or $GOLDEN missing" >&2
  exit 1
fi
# Stage A evidence is COMMIT-BOUND and LOCK-BOUND. Without these two checks a
# session directory from a dead SHA stages happily and is only caught later by
# `harness_preflight.sh A --payload` on the robot, which is the most expensive
# place to discover it. Amending a commit after running stage A is enough to
# produce exactly that (it happened on 2026-09-17), so compare here, at the desk.
_BUNDLE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEAD_SHA="$(git -C "$_BUNDLE_REPO" rev-parse HEAD)"
# Same env var and default the harness uses, so both read one lock.
_BUNDLE_LOCK="${DEPLOY_LOCK:-configs/sim2real/locks/deploy_stand_h25.lock.yaml}"
LOCK_SHA="$(sha256sum "$_BUNDLE_REPO/$_BUNDLE_LOCK" 2>/dev/null | cut -d' ' -f1 || true)"
if ! python3 - "$STAGE_A_JSON" "$HEAD_SHA" "$LOCK_SHA" <<'PY'
import json, sys

path, head_sha, lock_sha = sys.argv[1], sys.argv[2], sys.argv[3]
r = json.load(open(path))
problems = []
if r.get("stage") != "A":
    problems.append(f"stage is {r.get('stage')!r}, not 'A'")
if r.get("verdict") != "GO":
    problems.append(f"verdict is {r.get('verdict')!r}, not 'GO'")
if r.get("mode") != "workstation":
    problems.append(f"mode is {r.get('mode')!r}, not 'workstation'")
if r.get("rehearsal"):
    problems.append("evidence is flagged rehearsal and never counts")
recorded = (r.get("code_identity") or {}).get("sha")
if recorded != head_sha:
    problems.append(
        f"evidence is for commit {recorded}, but HEAD is {head_sha}. "
        "Re-run scripts/harness_preflight.sh A and use the NEW session directory."
    )
if (r.get("code_identity") or {}).get("dirty"):
    problems.append("evidence was recorded from a dirty tree")
if lock_sha and r.get("lock_file_sha256") and r["lock_file_sha256"] != lock_sha:
    problems.append(
        f"evidence is for lock {r['lock_file_sha256'][:12]}, current lock is {lock_sha[:12]}"
    )
for p in problems:
    print(f"  {p}", file=sys.stderr)
sys.exit(1 if problems else 0)
PY
then
  echo "REFUSING TO STAGE: $STAGE_A_JSON is not GO workstation stage A for THIS commit and lock" >&2
  exit 1
fi

# latest.pt is a symlink into the dated run directory; -L so the real weights
# travel, not a dangling link the payload cannot resolve.
REQUIRED=(policy.onnx policy.onnx.data policy.pt latest.pt)
for f in "${REQUIRED[@]}"; do
  if [[ ! -e "$CKPT_DIR/$f" ]]; then
    echo "REFUSING TO STAGE: missing $CKPT_DIR/$f" >&2
    exit 1
  fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$DEST"
for f in "${REQUIRED[@]}"; do
  cp -Lf "$CKPT_DIR/$f" "$DEST/$f"
done
cp -f "$GATE" "$DEST/parity_gate.json"
cp -f "$STAGE_A_JSON" "$DEST/workstation_stage_A.json"
cp -f "$GOLDEN" "$DEST/parity_golden.npz"
cp -f "$DEPLOY_CFG" "$DEST/$(basename "$DEPLOY_CFG")"
[[ -f "$CKPT_DIR/export_report.txt" ]] && cp -f "$CKPT_DIR/export_report.txt" "$DEST/"

# Transfer is not activation. The deploy configs ship workstation-relative paths
# ("checkpoints/<run>/policy.onnx") and ros2_policy_node resolves them with a
# bare Path() against the payload's working directory, so a bundle whose bytes
# verify perfectly can still sit untouched while the node loads whatever the
# payload's own checkpoints/ tree holds. activation.py rewrites the bundle's own
# copy of the config so every path the node opens is absolute and in-bundle.
# It is stdlib + PyYAML only and travels with the bundle so it can re-verify on
# the payload with no repo checkout and no PYTHONPATH.
cp -f "$REPO_ROOT/src/phoenix/sim2real/activation.py" "$DEST/activation.py"

# The pin target is where the bundle will LIVE, which for a remote push is the
# payload path, not the local staging temp dir.
if [[ -n "$REMOTE" ]]; then
  PIN_TARGET="${REMOTE#*:}"
else
  PIN_TARGET="$(cd "$DEST" && pwd)"
fi

# Pin BEFORE the manifest is written: pinning rewrites the config, so a manifest
# taken first would be invalidated by the very step that activates the bundle.
python3 "$DEST/activation.py" pin --bundle "$DEST" --target "$PIN_TARGET"

( cd "$DEST" && sha256sum ./* > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS )
( cd "$DEST" && sha256sum -c SHA256SUMS )

if [[ -z "$REMOTE" ]]; then
  # Local dest: the bundle already lives at its pin target, so activation is
  # checkable here and now.
  python3 "$DEST/activation.py" verify --bundle "$DEST"
  echo
  echo "[stage] staged $CKPT_DIR -> $DEST"
  exit 0
fi

# ------------------------------------------------------------------ remote push
HOST="${REMOTE%%:*}"
RPATH="${REMOTE#*:}"
PW="${JETSON_PW:-123}"
SSH=(sshpass -p "$PW" ssh "$HOST")

echo
echo "[stage] pushing to $HOST:$RPATH"
"${SSH[@]}" "mkdir -p '$RPATH'"
sshpass -p "$PW" rsync -a --delete -e "sshpass -p $PW ssh" "$DEST"/ "$HOST:$RPATH"/

# The local sha256sum -c above only proves the local copy. This one proves the
# bytes that actually reached the payload.
echo "[stage] verifying on $HOST"
"${SSH[@]}" "cd '$RPATH' && sha256sum -c SHA256SUMS"

# sha256sum -c proves the bytes arrived. This proves the config that arrived
# names those exact bytes, so the node cannot silently load an older export.
echo "[stage] verifying activation on $HOST"
"${SSH[@]}" "cd '$RPATH' && python3 activation.py verify --bundle '$RPATH'"

rm -rf "$DEST"
echo
echo "[stage] staged $CKPT_DIR -> $HOST:$RPATH, transfer AND activation verified"
