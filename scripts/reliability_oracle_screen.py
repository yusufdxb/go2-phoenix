#!/usr/bin/env python3
"""Analyze the perfect-onset fallback screen from raw study arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phoenix.reliability.oracle_screen import screen_oracle_fallback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    result = screen_oracle_fallback(args.out_dir, n_boot=args.n_boot, seed=args.seed)
    output = Path(args.output_json) if args.output_json else Path(args.out_dir) / "oracle_screen.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    effect = result["oracle_effect_unshielded_minus_oracle"]
    print(
        f"[ORACLE-SCREEN] disturbance={result['disturbance_kind']} "
        f"unshielded={result['disturbed_fall_rate']['unshielded']:.4f} "
        f"oracle={result['disturbed_fall_rate']['oracle']:.4f} "
        f"effect={effect['mean_difference']:+.4f} "
        f"95%CI=[{effect['ci_low']:+.4f}, {effect['ci_high']:+.4f}] "
        f"verdict={result['verdict']}"
    )
    print(f"[ORACLE-SCREEN] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

