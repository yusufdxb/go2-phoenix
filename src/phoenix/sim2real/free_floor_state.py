"""State files of the incumbent-only FREE_FLOOR commissioning path (stdlib only).

``scripts/free_floor_gate.py`` writes and reads these; the LowCmd bridge and the gantry ``W``
runner import :func:`unreviewed_runs` so that a free-floor walk that nobody has reviewed latches
EVERY later walking launch, not only the next ``FREE_FLOOR`` command.

This is accident-proofing, not tamper-proofing: the files are plain JSON in a session directory.
A review only counts if it names the run it reviews, and, when an analyzer result exists, the
sha256 of that result, so a bare ``touch free_floor_review.json`` or a review copied from another
session does not unlatch anything.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

STARTED = "free_floor_started.json"
RESULT = "free_floor_result.json"
REVIEW = "free_floor_review.json"
STARTED_SCHEMA = "phoenix-free-floor-started/v1"
REVIEW_SCHEMA = "phoenix-free-floor-review/v1"
REVIEW_DECISIONS = ("accept", "reject")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> tuple[Any, str | None]:
    """(parsed, None) or (None, why). A symlink is refused: state files are written, not linked."""
    if path.is_symlink():
        return None, f"{path.name} is a symlink"
    try:
        return json.loads(path.read_text()), None
    except (OSError, ValueError) as exc:
        return None, f"{path.name} unreadable: {exc}"


def review_problems(session: Path) -> list[str]:
    """Why the review record in ``session`` does not validly close the run started there."""
    started, err = read_json(session / STARTED)
    if err or not isinstance(started, dict):
        return [f"started record unreadable: {err}"]
    rec, err = read_json(session / REVIEW)
    if err or not isinstance(rec, dict):
        return [f"no valid review record: {err}"]
    problems: list[str] = []
    if rec.get("schema") != REVIEW_SCHEMA:
        problems.append("review has the wrong schema")
    if rec.get("decision") not in REVIEW_DECISIONS:
        problems.append(f"review decision {rec.get('decision')!r} is not accept or reject")
    for key in ("by", "note"):
        if not isinstance(rec.get(key), str) or not rec[key].strip():
            problems.append(f"review has no {key}")
    if rec.get("run_dir") != started.get("run_dir"):
        problems.append("review names a different run than the one started in this session")
    result = session / RESULT
    if result.is_file() and not result.is_symlink():
        if rec.get("result_sha256") != file_sha256(result):
            problems.append("the result record changed after (or was never named by) this review")
    elif rec.get("result_sha256") is not None:
        problems.append("review names a result record that does not exist")
    return problems


def unreviewed_runs(dirs: Sequence[Path], *, current_run: Path | None = None) -> list[str]:
    """Free-floor walks that were started and have no valid human review. They block walking.

    ``current_run`` is the run directory of the process asking (the bridge's telemetry directory):
    the walk that is starting right now has, by design, a started record and no review yet."""
    out = []
    for d in dirs:
        # A walk run that produced bridge telemetry but has no started record: the record was
        # deleted (or never written). Motors may have moved, so it latches like an unreviewed walk.
        for run in sorted(d.glob("FF_walk_*")):
            if (run / "bridge.jsonl").exists() and not (d / STARTED).exists():
                if current_run is None or run.resolve() != current_run.resolve():
                    out.append(
                        f"{d.name}: {run.name} has bridge telemetry but the session has no "
                        f"{STARTED} (deleted?); treat it as an unreviewed walk"
                    )
        if not (d / STARTED).exists():
            continue
        if current_run is not None:
            started, _ = read_json(d / STARTED)
            name = started.get("run_dir") if isinstance(started, dict) else None
            if name and (d / name).resolve() == current_run.resolve():
                continue
        if (d / REVIEW).exists() and not review_problems(d):
            continue
        why = review_problems(d) if (d / REVIEW).exists() else ["no review record"]
        out.append(
            f"{d.name}: a free-floor walk was started and is not validly reviewed ({'; '.join(why)}); "
            f"run: scripts/harness_preflight.sh FREE_FLOOR review accept|reject (session {d})"
        )
    return out
