#!/usr/bin/env python3
"""harness_diversity.py

Post-block diversity validator. Walks a directory of failure parquets, runs
the rule-based :class:`phoenix.real_world.FailureDetector` on each, and prints
a coverage matrix grouped by filename prefix mode label. Returns non-zero if
the Block-3 diversity gate fails:

  * fewer than 4 distinct mode prefixes, OR
  * any chosen mode has fewer than 2 replicate parquets, OR
  * any parquet has rows <= 200 (aborted-too-early), OR
  * any parquet yields zero detector events (silent / no failure).

Usage:
    python3 scripts/harness_diversity.py data/failures/
    python3 scripts/harness_diversity.py --min-modes 4 --min-replicates 2 \
        --min-rows 200 data/failures/
    python3 scripts/harness_diversity.py --self-test
        runs against a tmp dir of synthetic in-memory parquets and exits 0.

The prefix convention: parquets are named like
``pqa_<mode>_<ts>.parquet`` and the mode token is the second underscore-
separated segment (``slip``, ``push_lat``, ``ramp_dn`` ...). Gate7/Gate8
parquets (``gate7_floor_*.parquet``) are excluded from diversity counting
since they are not Block-3 collection samples.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# torch-free: the harness must import on a no-sim machine.


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dir", nargs="?", type=Path, help="directory of parquets")
    p.add_argument("--min-modes", type=int, default=4)
    p.add_argument("--min-replicates", type=int, default=2)
    p.add_argument("--min-rows", type=int, default=200)
    p.add_argument(
        "--include-prefix",
        action="append",
        default=None,
        help="only count parquets starting with this prefix (default: pqa_)",
    )
    p.add_argument("--self-test", action="store_true", help="run built-in self-test and exit")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON summary")
    return p.parse_args(argv)


def _mode_from_name(stem: str) -> str | None:
    """Extract mode label from a parquet filename stem.

    Convention: ``pqa_<mode>_<ts>.parquet`` -> mode is the second underscore-
    separated segment. Mode tokens may themselves contain one underscore
    (e.g. ``push_lat``, ``ramp_dn``, ``stop_sudden``, ``deadman_freeze``).
    Heuristic: take everything between the first underscore and the last
    underscore-separated timestamp-looking segment.
    """
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    if parts[0] not in ("pqa", "pqb"):
        return None
    # Last segment is timestamp; mode is the middle.
    mode_parts = parts[1:-1]
    # If the last segment isn't digits-only, treat the whole tail as mode
    # (some lab-day runs may not append a timestamp).
    if parts[-1].isdigit() or any(c.isdigit() for c in parts[-1]):
        return "_".join(mode_parts) or None
    return "_".join(parts[1:]) or None


def _count_parquet_rows_and_events(path: Path) -> tuple[int, int]:
    """Return ``(rows, n_failure_events)`` for a trajectory parquet.

    Reads via pyarrow (already a phoenix dep). Runs the detector across the
    rows; counts FailureEvent emissions. If the parquet already carries
    ``failure_flag`` / ``failure_mode`` columns from the recorder, also
    counts those as a sanity baseline (use whichever is higher).
    """
    import numpy as np
    import pyarrow.parquet as pq

    from phoenix.real_world.failure_detector import FailureDetector

    table = pq.read_table(path)
    n_rows = table.num_rows

    # Cheap path: trust recorder-side flags if present
    recorder_events = 0
    if "failure_flag" in table.column_names:
        recorder_events = int(sum(1 for v in table["failure_flag"].to_pylist() if v))

    # Re-run detector for an independent count
    try:
        ts = table["timestamp_s"].to_pylist()
        base_quat = [np.asarray(v, dtype=np.float32) for v in table["base_quat"].to_pylist()]
        base_pos = [np.asarray(v, dtype=np.float32) for v in table["base_pos"].to_pylist()]
        cmd_vel = [np.asarray(v, dtype=np.float32) for v in table["command_vel"].to_pylist()]
        lin_vel_b = [
            np.asarray(v, dtype=np.float32) for v in table["base_lin_vel_body"].to_pylist()
        ]
    except KeyError:
        return n_rows, recorder_events

    det = FailureDetector()
    detector_events = 0
    for i in range(n_rows):
        q = base_quat[i]
        # xyzw -> roll, pitch
        x, y, z, w = (float(v) for v in q)
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        height = float(base_pos[i][2])
        evt = det.step(
            timestamp_s=float(ts[i]),
            pitch_rad=float(pitch),
            roll_rad=float(roll),
            base_height_m=height,
            cmd_lin_vel=cmd_vel[i][:2],
            actual_lin_vel=lin_vel_b[i][:2],
        )
        if evt is not None:
            detector_events += 1

    return n_rows, max(recorder_events, detector_events)


def scan(
    root: Path,
    include_prefixes: list[str],
    min_rows: int,
) -> dict:
    """Return per-file results + per-mode aggregation."""
    files = sorted(p for p in root.glob("*.parquet") if any(p.name.startswith(pref) for pref in include_prefixes))
    per_file: list[dict] = []
    per_mode: dict[str, list[dict]] = defaultdict(list)

    for p in files:
        mode = _mode_from_name(p.stem)
        try:
            rows, events = _count_parquet_rows_and_events(p)
            err = None
        except Exception as exc:
            rows, events, err = 0, 0, str(exc)
        row = {
            "path": str(p),
            "mode": mode,
            "rows": rows,
            "events": events,
            "rows_ok": rows > min_rows,
            "events_ok": events > 0,
            "error": err,
        }
        per_file.append(row)
        if mode:
            per_mode[mode].append(row)

    return {"files": per_file, "modes": per_mode}


def evaluate(
    summary: dict,
    *,
    min_modes: int,
    min_replicates: int,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    modes = summary["modes"]
    n_modes = len(modes)
    if n_modes < min_modes:
        failures.append(f"fewer than {min_modes} distinct modes (got {n_modes}: {sorted(modes)})")
    for m, rows in modes.items():
        n_good = sum(1 for r in rows if r["rows_ok"] and r["events_ok"] and not r["error"])
        if n_good < min_replicates:
            failures.append(
                f"mode '{m}' has {n_good} good replicates (need >= {min_replicates})"
            )
    for r in summary["files"]:
        if r["error"]:
            failures.append(f"{Path(r['path']).name}: read error {r['error']}")
        elif not r["rows_ok"]:
            failures.append(f"{Path(r['path']).name}: rows {r['rows']} (need > {min_rows_for(r)})")
        elif not r["events_ok"]:
            failures.append(f"{Path(r['path']).name}: no detector events (silent parquet)")
    return (len(failures) == 0, failures)


def min_rows_for(row: dict) -> int:
    # placeholder so the message reads naturally; threshold is uniform.
    return 200


def print_matrix(summary: dict) -> None:
    modes = summary["modes"]
    print(f"\n{'MODE':<20} {'#FILES':<8} {'#GOOD':<8} REPLICATES")
    print("-" * 70)
    for m in sorted(modes):
        rows = modes[m]
        n_good = sum(1 for r in rows if r["rows_ok"] and r["events_ok"] and not r["error"])
        names = ", ".join(Path(r["path"]).stem for r in rows)
        print(f"{m:<20} {len(rows):<8} {n_good:<8} {names}")
    print()


# ---------------- self-test --------------------------------------------------


def _write_synth_parquet(path: Path, *, rows: int, force_failure: bool, mode: str) -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    def vec(n):
        return [np.zeros(n, dtype=np.float32).tolist() for _ in range(rows)]

    base_quat = []
    base_pos = []
    cmd_vel = []
    lin_vel = []
    for i in range(rows):
        # neutral quat
        q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        pos = np.array([0.0, 0.0, 0.35], dtype=np.float32)
        if force_failure and i > rows // 2:
            # tip pitch over the 0.8 rad threshold
            half = 1.0  # ~57 deg pitch
            q = np.array([0.0, np.sin(half / 2), 0.0, np.cos(half / 2)], dtype=np.float32)
        base_quat.append(q.tolist())
        base_pos.append(pos.tolist())
        cmd_vel.append([0.0, 0.0, 0.0])
        lin_vel.append([0.0, 0.0, 0.0])

    table = pa.table(
        {
            "step": list(range(rows)),
            "timestamp_s": [i * 0.02 for i in range(rows)],
            "base_pos": base_pos,
            "base_quat": base_quat,
            "base_lin_vel_body": lin_vel,
            "base_ang_vel_body": vec(3),
            "joint_pos": vec(12),
            "joint_vel": vec(12),
            "command_vel": cmd_vel,
            "action": vec(12),
            "contact_forces": vec(4),
            "failure_flag": [False] * rows,
            "failure_mode": [None] * rows,
        }
    )
    pq.write_table(table, path)


def self_test() -> int:
    """Build a synthetic data/failures with 4 modes x 2 reps + run validator."""
    import shutil

    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        modes = ["slip", "push_lat", "ramp_dn", "yaw_shove"]
        for m in modes:
            for rep in range(2):
                pp = tmp_p / f"pqa_{m}_{rep:04d}.parquet"
                _write_synth_parquet(pp, rows=400, force_failure=True, mode=m)

        summary = scan(tmp_p, include_prefixes=["pqa_"], min_rows=200)
        print_matrix(summary)
        ok, failures = evaluate(summary, min_modes=4, min_replicates=2)
        if not ok:
            print("[self-test] FAIL")
            for f in failures:
                print("  -", f)
            return 1
        print("[self-test] PASS")
        return 0
    # unreachable
    shutil.rmtree(tmp, ignore_errors=True)  # noqa


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if args.dir is None:
        print("ERROR: directory positional required (or pass --self-test)", file=sys.stderr)
        return 2
    if not args.dir.is_dir():
        print(f"ERROR: not a directory: {args.dir}", file=sys.stderr)
        return 2

    include_prefixes = args.include_prefix or ["pqa_"]
    summary = scan(args.dir, include_prefixes=include_prefixes, min_rows=args.min_rows)
    if args.json:
        # Strip per-mode duplicates for serializability
        out = {
            "files": summary["files"],
            "modes": {m: len(v) for m, v in summary["modes"].items()},
        }
        print(json.dumps(out, indent=2))
    else:
        print_matrix(summary)

    ok, failures = evaluate(
        summary, min_modes=args.min_modes, min_replicates=args.min_replicates
    )
    if not ok:
        print("DIVERSITY GATE: FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("DIVERSITY GATE: PASS")
    return 0


if __name__ == "__main__":
    # Make `phoenix` importable when run from repo root without install.
    repo_src = Path(__file__).resolve().parent.parent / "src"
    if repo_src.is_dir() and str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    sys.exit(main())
