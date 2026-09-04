#!/usr/bin/env python3
"""Render the causal-viability replication result table from combined_summary.json.

The v1 table was transcribed by hand. Every number in it is quoted in the paper,
so this renders it from the analysis artifact instead, and the artifact is the
only place a number can come from.

Pooling rules, stated here because the table quotes pooled numbers that the
summary only stores per process:

* fall counts are summed across the three processes;
* rates, task completion, return-until-first-fall and oracle dose are weighted
  by that process-cell's jointly eligible environment pair count, which is the
  denominator those rates were computed against;
* the latency column reports the min of the per-process minima, the median of
  the per-process medians and the max of the per-process maxima. It is a range
  across processes, not a pooled distribution.

Usage:
    scripts/reliability_result_table.py <combined_summary.json> <out.md>
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

CELL_ORDER = ("stand_motor", "stand_obs", "walk_motor", "walk_obs")
CELL_LABEL = {
    "stand_motor": ("Standing", "Motor degradation"),
    "stand_obs": ("Standing", "Observation corruption"),
    "walk_motor": ("Walking", "Motor degradation"),
    "walk_obs": ("Walking", "Observation corruption"),
}
FAMILY_LABEL = {"motor": "Motor degradation", "obs": "Observation corruption"}


def pp(x: float) -> str:
    return f"{x * 100:+.2f}"


def ci(d: dict) -> str:
    return f"[{pp(d['ci_low'])}, {pp(d['ci_high'])}]"


def weighted(rows: list[dict], weights: list[int], pick) -> float:
    total = sum(weights)
    return sum(pick(r) * w for r, w in zip(rows, weights, strict=True)) / total


def render(summary: dict) -> str:
    cells = summary["pooled_cells"]
    by_cell: dict[str, list[dict]] = {c: [] for c in CELL_ORDER}
    for r in summary["process_results"]:
        by_cell[r["cell_id"]].append(r)

    out: list[str] = []
    a = out.append
    a("# Causal Viability Replication Result")
    a("")
    a(summary["sign_convention"])
    a("")
    verdict = "PASSED" if summary["gate_passed"] else "FAILED"
    a(f"Formal gate: **{verdict}**.")
    a("")
    a("| Check | Result |")
    a("|---|---|")
    for k, v in sorted(summary["gate_checks"].items()):
        a(f"| {k.replace('_', ' ')} | {'pass' if v else 'FAIL'} |")
    a("")
    a(f"Registry hash: `{summary['registry_hash']}`")
    a("")

    a("## Primary outcome")
    a("")
    a(
        "| Policy | Fault | Independent disturbed blocks | Eligible pairs | "
        "Unshielded falls | Oracle falls | Unshielded rate | Oracle rate | "
        "Block-paired effect, pp | 95% block-bootstrap CI, pp | Process effects, pp |"
    )
    a("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for cid in CELL_ORDER:
        c = cells[cid]
        policy, fault = CELL_LABEL[cid]
        procs = ", ".join(pp(e) for e in c["process_effects"])
        a(
            f"| {policy} | {fault} | {c['independent_disturbed_blocks']} | "
            f"{c['jointly_eligible_environment_pairs']} | {c['unshielded_fall_count']} | "
            f"{c['oracle_fall_count']} | {c['unshielded_fall_rate'] * 100:.2f}% | "
            f"{c['oracle_fall_rate'] * 100:.2f}% | {pp(c['mean_difference'])} | "
            f"{ci(c)} | {procs} |"
        )
    a("")

    a("## Leave-one-process-out effects")
    a("")
    a("| Cell | Leave out process 1, pp | Leave out process 2, pp | Leave out process 3, pp |")
    a("|---|---:|---:|---:|")
    for cid in CELL_ORDER:
        lopo = cells[cid]["leave_one_process_out"]
        policy, fault = CELL_LABEL[cid]
        vals = [
            f"{pp(lopo[k]['mean_difference'])} {ci(lopo[k])}"
            for k in ("process_01", "process_02", "process_03")
        ]
        a(f"| {policy}, {fault.split()[0].lower()} | " + " | ".join(vals) + " |")
    a("")

    a("## Fault-family interaction")
    a("")
    a("| Quantity | Independent blocks | Effect, pp | 95% block-bootstrap CI, pp |")
    a("|---|---:|---:|---:|")
    acct = summary["independent_block_accounting"]
    for fam in ("motor", "obs"):
        f = summary["pooled_fault_families"][fam]
        a(
            f"| {FAMILY_LABEL[fam]} | {f['independent_disturbed_blocks']} | "
            f"{pp(f['mean_difference'])} | {ci(f)} |"
        )
    inter = summary["fault_by_treatment_interaction_obs_minus_motor"]
    a(
        f"| Observation-minus-motor interaction | {acct['disturbed_total']} | "
        f"{pp(inter['mean_difference'])} | {ci(inter)} |"
    )
    a("")

    a("## Secondary outcomes")
    a("")
    a(
        "| Cell | Task completion U/O | Return until first fall U/O | Oracle dose | "
        "Treated falls | Latency ticks min/median/max | Nominal false handoffs |"
    )
    a("|---|---|---|---:|---:|---|---:|")
    for cid in CELL_ORDER:
        rows = by_cell[cid]
        w = [r["jointly_eligible_environment_pairs"] for r in rows]
        policy, fault = CELL_LABEL[cid]
        tc = {
            arm: weighted(rows, w, lambda r, a=arm: r["task_completion_rate_jointly_eligible"][a])
            for arm in ("unshielded", "oracle")
        }
        rt = {
            arm: weighted(
                rows, w, lambda r, a=arm: r["return_until_first_fall_mean_jointly_eligible"][a]
            )
            for arm in ("unshielded", "oracle")
        }
        dose = weighted(rows, w, lambda r: r["oracle_fallback_dose_mean"])
        treated = sum(r["oracle_treated_falls"] for r in rows)
        lat_min = min(r["oracle_intervention_latency_ticks"]["min"] for r in rows)
        lat_med = statistics.median(r["oracle_intervention_latency_ticks"]["median"] for r in rows)
        lat_max = max(r["oracle_intervention_latency_ticks"]["max"] for r in rows)
        handoffs = sum(r["nominal_false_handoff_count"] for r in rows)
        a(
            f"| {policy}, {fault.split()[0].lower()} | "
            f"{tc['unshielded'] * 100:.2f}% / {tc['oracle'] * 100:.2f}% | "
            f"{rt['unshielded']:.3f} / {rt['oracle']:.3f} | {dose:.3f} | {treated} | "
            f"{lat_min:g} / {lat_med:g} / {lat_max:g} | {handoffs} |"
        )
    a("")

    a("## Pre-onset negative control")
    a("")
    a("| Cell | Pre-onset U-minus-O effect, pp | 95% block-bootstrap CI, pp |")
    a("|---|---:|---:|")
    for cid in CELL_ORDER:
        n = summary["pooled_pre_onset_negative_control_cells"][cid]
        policy, fault = CELL_LABEL[cid]
        a(f"| {policy}, {fault.split()[0].lower()} | {pp(n['mean_difference'])} | {ci(n)} |")
    a("")
    fams = summary["pooled_pre_onset_negative_control_fault_families"]
    a(
        f"The motor-family pre-onset effect was {pp(fams['motor']['mean_difference'])} pp "
        f"{ci(fams['motor'])}. The observation-family pre-onset effect was "
        f"{pp(fams['obs']['mean_difference'])} pp {ci(fams['obs'])}."
    )
    a("")
    return "\n".join(out)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    summary = json.loads(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(render(summary))
    print(f"[table] wrote {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
