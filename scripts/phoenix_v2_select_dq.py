#!/usr/bin/env python3
"""Apply the amendment 6.2 dq_max selection rule to a sweep directory.

Expects ``<sweep>/{dr,nominal}_{hard_only_0,prev_command_<d>}/summary.json`` (``p`` for the
decimal point, as written by the sweep configs). Metric: ``mean_primary_score`` (stand)
or ``walk2_success_rate`` (walking, ``--metric walk``). Prints the table and both the
one-sided (amendment 6, binding) and two-sided (amendment 2 reading, reported) choices,
and writes ``<sweep>/selection.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GRID = (0.02, 0.035, 0.05, 0.075, 0.10, 0.175)
MAX_ALT, MAX_ALT_J, MARGIN = 0.01, 0.05, 0.02


def _tag(d: float) -> str:
    return f"{d:g}".replace(".", "p")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sweep", type=Path)
    ap.add_argument("--metric", choices=("stand", "walk"), default="stand")
    a = ap.parse_args(argv)
    key = "mean_primary_score" if a.metric == "stand" else "walk2_success_rate"

    def load(name):
        return json.loads((a.sweep / name / "summary.json").read_text())

    ref = {c: load(f"{c}_hard_only_0")[key] for c in ("dr", "nominal")}
    rows, one, two = [], None, None
    for d in GRID:
        row = {"dq_max": d}
        ok1 = ok2 = True
        for c in ("dr", "nominal"):
            s = load(f"{c}_prev_command_{_tag(d)}")
            alt = s["altered_fraction"]
            alt_j = max(v["altered_fraction"] for v in s["per_joint"].values())
            m = s[key]
            row[c] = {"altered": alt, "worst_joint_altered": alt_j, key: m,
                      "delta_vs_hard_only": m - ref[c], "seed": s["seed"]}
            base = alt <= MAX_ALT and alt_j <= MAX_ALT_J
            ok1 &= base and m >= ref[c] - MARGIN
            ok2 &= base and abs(m - ref[c]) <= MARGIN
        row["passes_one_sided"], row["passes_two_sided"] = ok1, ok2
        rows.append(row)
        if ok1 and one is None:
            one = d
        if ok2 and two is None:
            two = d
    out = {"rule": "amendment 6.2", "metric": key, "hard_only_reference": ref, "rows": rows,
           "selected_one_sided": one, "selected_two_sided_reported": two}
    (a.sweep / "selection.json").write_text(json.dumps(out, indent=1))
    print(f"hard-only {key}: {ref}")
    for r in rows:
        print(f"{r['dq_max']:>6}  " + "  ".join(
            f"{c}: alt {r[c]['altered']*100:5.2f}% worst {r[c]['worst_joint_altered']*100:5.2f}% "
            f"{r[c][key]:.4f} ({r[c]['delta_vs_hard_only']:+.4f})" for c in ("dr", "nominal"))
            + f"  one-sided {'PASS' if r['passes_one_sided'] else 'fail'}"
              f"  two-sided {'PASS' if r['passes_two_sided'] else 'fail'}")
    print(f"SELECTED (one-sided, binding): {one}   two-sided (reported): {two}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
