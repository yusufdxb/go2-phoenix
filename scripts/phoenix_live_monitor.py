#!/usr/bin/env python3
"""Live health vector: tail a bridge.jsonl while the robot runs and print the table.

  PYTHONPATH=src python3 scripts/phoenix_live_monitor.py <bridge.jsonl> \
      --baseline baseline.json [--out health_stream.jsonl] [--once]

Read-only: it follows the file the LowCmd bridge is writing and never talks to the
robot. ``--once`` processes what is in the file and exits (for re-analysis and tests).
Runs on the payload or on a workstation reading the file over the network.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from phoenix.monitor.health import format_report
from phoenix.monitor.live import LiveMonitor
from phoenix.monitor.residual import Baseline


def follow(path: Path, once: bool, poll_s: float = 0.1):
    """Yield complete JSON records as they are appended; a partial last line waits."""
    while not path.exists():
        if once:
            return
        time.sleep(poll_s)
    with path.open("r") as fh:
        pending = ""
        while True:
            chunk = fh.readline()
            if not chunk:
                if once:
                    return
                time.sleep(poll_s)
                continue
            pending += chunk
            if not pending.endswith("\n"):
                continue
            line, pending = pending.strip(), ""
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield rec
            if rec.get("record") == "end":
                return


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("telemetry", type=Path)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--out", type=Path, help="append one JSON health report per window")
    p.add_argument("--once", action="store_true", help="process the file as it is and exit")
    p.add_argument("--quiet", action="store_true", help="do not print the table")
    a = p.parse_args(argv)

    mon = LiveMonitor(Baseline.from_dict(json.loads(a.baseline.read_text())))
    out = a.out.open("a") if a.out else None
    n = 0
    try:
        for rec in follow(a.telemetry, a.once):
            rep = mon.feed(rec)
            if rep is None:
                continue
            n += 1
            if out is not None:
                out.write(json.dumps({"window": n, "health": [h.to_dict() for h in rep]}) + "\n")
                out.flush()
            if not a.quiet:
                print(f"\nwindow {n}\n{format_report(rep)}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if out is not None:
            out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
