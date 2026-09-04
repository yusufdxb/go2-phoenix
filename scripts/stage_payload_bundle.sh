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

# latest.pt is a symlink into the dated run directory; -L so the real weights
# travel, not a dangling link the payload cannot resolve.
REQUIRED=(policy.onnx policy.onnx.data policy.pt latest.pt)
for f in "${REQUIRED[@]}"; do
  if [[ ! -e "$CKPT_DIR/$f" ]]; then
    echo "REFUSING TO STAGE: missing $CKPT_DIR/$f" >&2
    exit 1
  fi
done

mkdir -p "$DEST"
for f in "${REQUIRED[@]}"; do
  cp -Lf "$CKPT_DIR/$f" "$DEST/$f"
done
cp -f "$GATE" "$DEST/parity_gate.json"
cp -f "$DEPLOY_CFG" "$DEST/$(basename "$DEPLOY_CFG")"
[[ -f "$CKPT_DIR/export_report.txt" ]] && cp -f "$CKPT_DIR/export_report.txt" "$DEST/"

( cd "$DEST" && sha256sum ./* > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS )
( cd "$DEST" && sha256sum -c SHA256SUMS )

if [[ -z "$REMOTE" ]]; then
  echo
  echo "[stage] staged $CKPT_DIR -> $DEST"
  echo "[stage] verify on the payload with: sha256sum -c SHA256SUMS"
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

rm -rf "$DEST"
echo
echo "[stage] staged $CKPT_DIR -> $HOST:$RPATH, verified on the payload"
