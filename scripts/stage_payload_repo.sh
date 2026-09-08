#!/usr/bin/env bash
# Put the CURRENT repo code on the GO2 payload without internet.
#
# The lab network has no egress, so `git fetch origin` on the payload (what
# scripts/harness_preflight.sh P2 does) cannot work there. The payload's
# checkout is whatever the last session left, which for Phoenix is the April
# stand-v3 era: no activation.py, no first_message_timeout_s, an older
# ros2_policy_node. A verified bundle (scripts/stage_payload_bundle.sh) run by
# stale node code is still a stale session.
#
# This script rsyncs the code paths the payload actually executes, writes a
# sha256 manifest of exactly the tracked files it sent, re-verifies that
# manifest ON THE PAYLOAD, and then checks that the payload's python3 imports
# `phoenix` from the synced checkout (not from a stale site-packages copy).
# It never touches checkpoints/, data/ or logs/ on the payload.
#
# Usage:
#   scripts/stage_payload_repo.sh <dest>
#
# <dest> is the payload's repo root: a local directory (self-test) or
# [user@]host:/path. Refuses to run from a dirty working tree unless
# ALLOW_DIRTY=1, because PAYLOAD_SYNC.txt records the commit and a dirty tree
# makes that record a lie. JETSON_PW defaults to 123 (factory default).
#
# Examples:
#   scripts/stage_payload_repo.sh /tmp/phoenix-repo-selftest
#   scripts/stage_payload_repo.sh jetson-cable:/home/unitree/go2-phoenix

set -euo pipefail

DEST="${1:?usage: $0 <dest>   (local dir or [user@]host:/path to the payload repo root)}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# What the payload runs. Anything else (reliability_eval, paper, media, the
# 11 MB of study artifacts) stays on the workstation.
SYNC_DIRS=(src configs scripts docs)
SYNC_FILES=(pyproject.toml README.md)

REMOTE=""
case "$DEST" in
  /*|./*|../*) ;;
  *:*) REMOTE="$DEST" ;;
esac

# Untracked files inside the sync set count as dirty too: only tracked files
# travel, so an uncommitted new script or config would be silently left
# behind while PAYLOAD_SYNC.txt reports a clean tree.
UNTRACKED="$(git ls-files --others --exclude-standard -- "${SYNC_DIRS[@]}" "${SYNC_FILES[@]}")"
if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$UNTRACKED" ]]; then
  if [[ "${ALLOW_DIRTY:-0}" != "1" ]]; then
    echo "REFUSING TO STAGE: working tree is dirty; commit first or set ALLOW_DIRTY=1." >&2
    echo "  (untracked files in the sync set would NOT travel; they are tracked-only)" >&2
    git status --short >&2
    exit 1
  fi
  DIRTY="true"
else
  DIRTY="false"
fi

COMMIT="$(git rev-parse HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# Manifest of exactly the tracked files being sent. Untracked files in the
# synced dirs are excluded from rsync too, so the manifest and the transfer
# describe the same set.
git ls-files -- "${SYNC_DIRS[@]}" "${SYNC_FILES[@]}" > "$STAGE/filelist"
if [[ ! -s "$STAGE/filelist" ]]; then
  echo "REFUSING TO STAGE: git ls-files returned nothing for the sync set." >&2
  exit 1
fi
N_FILES="$(wc -l < "$STAGE/filelist")"
sha256sum $(cat "$STAGE/filelist") > "$STAGE/PAYLOAD_SHA256SUMS"

cat > "$STAGE/PAYLOAD_SYNC.txt" <<EOF
commit:  $COMMIT
branch:  $BRANCH
dirty:   $DIRTY
synced:  $(date -u +%Y-%m-%dT%H:%M:%SZ) from $(hostname -s)
files:   $N_FILES tracked files under: ${SYNC_DIRS[*]} ${SYNC_FILES[*]}
verify:  sha256sum -c PAYLOAD_SHA256SUMS   (run from this directory)
EOF

# rsync from the tracked-file list, so nothing untracked (caches, .venv,
# scratch outputs) travels. --delete-missing-args is not what we want;
# --files-from with --delete is scoped to the listed dirs by rsync itself.
RSYNC_COMMON=(-a --files-from="$STAGE/filelist" --relative)

sync_to() {
  local target="$1"
  rsync "${RSYNC_COMMON[@]}" ./ "$target"/
  rsync -a "$STAGE/PAYLOAD_SHA256SUMS" "$STAGE/PAYLOAD_SYNC.txt" "$target"/
}

if [[ -z "$REMOTE" ]]; then
  mkdir -p "$DEST"
  sync_to "$DEST"
  ( cd "$DEST" && sha256sum -c --quiet PAYLOAD_SHA256SUMS )
  echo "[repo] $N_FILES files verified at $DEST"
  echo "[repo] $(sed -n 1p "$DEST/PAYLOAD_SYNC.txt")"
  exit 0
fi

# ------------------------------------------------------------------ remote push
HOST="${REMOTE%%:*}"
RPATH="${REMOTE#*:}"
PW="${JETSON_PW:-123}"
SSH=(sshpass -p "$PW" ssh "$HOST")
export RSYNC_RSH="sshpass -p $PW ssh"

echo "[repo] pushing $N_FILES tracked files @ ${COMMIT:0:7} (dirty=$DIRTY) -> $HOST:$RPATH"
"${SSH[@]}" "mkdir -p '$RPATH'"
sync_to "$HOST:$RPATH"

# The local manifest proves nothing about the payload. This does.
echo "[repo] verifying manifest on $HOST"
"${SSH[@]}" "cd '$RPATH' && sha256sum -c --quiet PAYLOAD_SHA256SUMS && echo '[repo] manifest OK on payload'"

# A non-editable `pip install .` from a previous session would shadow the
# synced source with a frozen copy in site-packages, and the node would run
# old code while every file on disk is current.
echo "[repo] checking which phoenix the payload's python3 imports"
IMPORTED="$("${SSH[@]}" "cd '$RPATH' && python3 -c 'import os, phoenix; print(os.path.dirname(os.path.abspath(phoenix.__file__)))'" 2>&1 || true)"
EXPECTED="$RPATH/src/phoenix"
if [[ "$IMPORTED" == "$EXPECTED" ]]; then
  echo "[repo] import OK: $IMPORTED"
else
  echo "[repo] IMPORT MISMATCH: payload python3 imports phoenix from:" >&2
  echo "         $IMPORTED" >&2
  echo "       expected:" >&2
  echo "         $EXPECTED" >&2
  echo "       Fix on the payload (no internet needed):" >&2
  echo "         cd '$RPATH' && python3 -m pip install --no-deps --no-build-isolation -e ." >&2
  echo "       then re-run this script." >&2
  exit 1
fi

echo
echo "[repo] staged repo @ ${COMMIT:0:7} -> $HOST:$RPATH, manifest AND import verified"
