#!/usr/bin/env python3
"""Audit the batched-block harness's pre-onset onset-observation residual.

Reads the same frozen registry the registered analysis reads and writes a JSON audit
plus a short human-readable summary. It does not touch the frozen gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phoenix.reliability.onset_residual import audit_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-boot", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_registry(args.registry, n_boot=args.n_boot, seed=args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"FAIL CLOSED: refusing to overwrite {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    summary = result["summary"]
    print(
        f"[onset-residual] threshold separation in "
        f"{summary['replicates_with_onset_threshold_separation']}/{summary['replicates']} "
        f"replicates, log10 P(chance)={summary['log10_joint_probability_of_separation_by_chance']:.1f}"
    )
    print(
        f"[onset-residual] pre-onset fall differences: "
        f"{summary['pre_onset_fall_difference_environments']} environments"
    )
    for cell, values in result["cells"].items():
        registered = values["registered"]
        clean = values["contamination_free"]
        print(
            f"[onset-residual] {cell:12s} "
            f"registered n={registered['blocks']:3d} "
            f"{registered['mean_difference'] * 100:+7.2f} pp "
            f"[{registered['ci_low'] * 100:+6.2f}, {registered['ci_high'] * 100:+6.2f}] | "
            f"contamination-free n={clean['blocks']:3d} "
            f"{clean['mean_difference'] * 100:+7.2f} pp "
            f"[{clean['ci_low'] * 100:+6.2f}, {clean['ci_high'] * 100:+6.2f}]"
        )
    print(
        "[onset-residual] all cells reproduce contamination-free: "
        f"{summary['all_cells_reproduce_contamination_free']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
