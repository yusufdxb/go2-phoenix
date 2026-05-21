#!/usr/bin/env bash
# harness_eod.sh
# End-of-day rsync wrapper: Jetson -> T7 for parquets, rosbags, camera video.
# Prints a per-file checksum summary.
#
# Usage:
#   bash scripts/harness_eod.sh
#   bash scripts/harness_eod.sh --mock        # writes to /tmp/t7_mock instead
#
# Optional env vars:
#   T7_MOUNT      T7 mount root (default: /media/yusuf/T7 Storage/go2-phoenix)
#   LAB_DATE      date subdir under data/lab to sync (default: today)
#
# Exit non-zero on rsync error or any checksum mismatch.

set -euo pipefail

MOCK=0
if [[ "${1:-}" == "--mock" ]]; then
    MOCK=1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LAB_DATE="${LAB_DATE:-$(date +%Y-%m-%d)}"
# Default T7 mount; --mock forces /tmp/t7_mock UNLESS T7_MOUNT is already set
# in the environment (the env override always wins so tests can point at a
# hermetic tmp dir).
DEFAULT_T7="/media/yusuf/T7 Storage/go2-phoenix"
if [[ $MOCK -eq 1 && -z "${T7_MOUNT:-}" ]]; then
    T7_MOUNT="/tmp/t7_mock"
fi
T7_MOUNT="${T7_MOUNT:-$DEFAULT_T7}"
if [[ $MOCK -eq 1 ]]; then
    mkdir -p "$T7_MOUNT"
fi

require_bin() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: missing required binary: $1 ${2:-}" >&2
        exit 1
    fi
}
require_bin rsync
require_bin md5sum

if [[ $MOCK -eq 0 && ! -d "$T7_MOUNT" ]]; then
    echo "ERROR: T7 not mounted at $T7_MOUNT" >&2
    exit 1
fi

echo "[eod] LAB_DATE=$LAB_DATE"
echo "[eod] T7_MOUNT=$T7_MOUNT"

SRC_PARQUETS="data/failures/"
SRC_LAB="data/lab/${LAB_DATE}/"

DST_PARQUETS="$T7_MOUNT/data/failures/"
DST_LAB="$T7_MOUNT/data/lab/${LAB_DATE}/"
mkdir -p "$DST_PARQUETS" "$DST_LAB"

# ---- rsync parquets ---------------------------------------------------------
echo "[eod] rsync $SRC_PARQUETS -> $DST_PARQUETS"
if [[ -d "$SRC_PARQUETS" ]]; then
    rsync -av --checksum --include="*.parquet" --exclude="*" \
        "$SRC_PARQUETS" "$DST_PARQUETS" | tee /tmp/harness_eod_parquets.log
else
    echo "[eod] (no $SRC_PARQUETS, skipping)"
fi

# ---- rsync today's lab dir (bags + camera + manifest) -----------------------
echo "[eod] rsync $SRC_LAB -> $DST_LAB"
if [[ -d "$SRC_LAB" ]]; then
    rsync -av --checksum "$SRC_LAB" "$DST_LAB" | tee /tmp/harness_eod_lab.log
else
    echo "[eod] (no $SRC_LAB, skipping)"
fi

# ---- per-file checksum verification ----------------------------------------
echo ""
echo "[eod] checksum verification..."
MISMATCHES=0
TRANSFERRED=0

verify_tree() {
    local src="$1"
    local dst="$2"
    local glob="${3:-}"
    if [[ ! -d "$src" ]]; then return 0; fi
    # Iterate only regular files. Optional glob filter mirrors the rsync
    # --include pattern so we don't false-flag files that were intentionally
    # excluded from transfer (e.g. scratch.txt next to parquets).
    while IFS= read -r -d '' f; do
        if [[ -n "$glob" ]]; then
            local base="${f##*/}"
            case "$base" in
                $glob) ;;
                *) continue ;;
            esac
        fi
        local rel="${f#$src}"
        local src_md5 dst_md5
        src_md5=$(md5sum "$f" | awk '{print $1}')
        if [[ ! -f "$dst$rel" ]]; then
            echo "  MISSING on dst: $rel"
            MISMATCHES=$((MISMATCHES + 1))
            continue
        fi
        dst_md5=$(md5sum "$dst$rel" | awk '{print $1}')
        if [[ "$src_md5" != "$dst_md5" ]]; then
            echo "  MISMATCH $rel src=$src_md5 dst=$dst_md5"
            MISMATCHES=$((MISMATCHES + 1))
        else
            TRANSFERRED=$((TRANSFERRED + 1))
        fi
    done < <(find "$src" -type f -print0)
}

verify_tree "$SRC_PARQUETS" "$DST_PARQUETS" "*.parquet"
verify_tree "$SRC_LAB" "$DST_LAB"

echo ""
echo "[eod] summary:"
echo "  files verified: $TRANSFERRED"
echo "  mismatches:     $MISMATCHES"
echo "  dst root:       $T7_MOUNT"

if [[ $MISMATCHES -gt 0 ]]; then
    echo "[eod] FAIL: checksum mismatches detected"
    exit 1
fi

echo "[eod] OK"
