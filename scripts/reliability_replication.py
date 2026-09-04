#!/usr/bin/env python3
"""Freeze or analyze the Phoenix causal-viability replication registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phoenix.reliability.replication import (
    analyze_registry,
    build_registry,
    validate_preflight_pair,
    validate_process_preflights,
    validate_replicate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    registry = sub.add_parser("registry")
    registry.add_argument("--root", required=True)
    registry.add_argument("--output", required=True)
    registry.add_argument("--exploratory-protocol", action="append", default=[])
    registry.add_argument(
        "--study-id",
        default=None,
        help="Study id every protocol under --root must declare. Defaults to the v1 "
        "replication id so existing artifacts validate unchanged.",
    )

    validate = sub.add_parser("validate")
    validate.add_argument("--out-dir", required=True)
    validate.add_argument("--output", default=None)
    validate.add_argument("--n-boot", type=int, default=10_000)
    validate.add_argument("--seed", type=int, default=20260730)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--paired-out-dir", required=True)
    preflight.add_argument("--process-out-dir", action="append", required=True)
    preflight.add_argument("--output", required=True)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--registry", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--n-boot", type=int, default=20_000)
    analyze.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"FAIL CLOSED: refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if args.command == "registry":
        kwargs = {} if args.study_id is None else {"study_id": args.study_id}
        result = build_registry(
            args.root,
            exploratory_protocols=args.exploratory_protocol,
            **kwargs,
        )
        write_json(args.output, result)
        print(
            f"[registry] protocols={result['independent_protocols']} "
            f"blocks={result['independent_blocks']} hash={result['registry_hash']}"
        )
        return 0
    if args.command == "validate":
        result = validate_replicate(args.out_dir, n_boot=args.n_boot, seed=args.seed)
        if args.output:
            write_json(args.output, result)
        effect = result["effect_unshielded_minus_oracle"]
        print(
            f"[validate] {result['replicate_id']} {result['cell_id']} "
            f"effect={effect['mean_difference']:+.6f} "
            f"95%CI=[{effect['ci_low']:+.6f}, {effect['ci_high']:+.6f}]"
        )
        return 0
    if args.command == "preflight":
        result = {
            "paired_contract": validate_preflight_pair(args.paired_out_dir),
            "process_independence": validate_process_preflights(args.process_out_dir),
        }
        write_json(args.output, result)
        print("[preflight] paired contract and process independence passed")
        return 0

    result = analyze_registry(
        args.registry,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    write_json(args.output, result)
    interaction = result["fault_by_treatment_interaction_obs_minus_motor"]
    print(
        f"[analyze] interaction={interaction['mean_difference']:+.6f} "
        f"95%CI=[{interaction['ci_low']:+.6f}, {interaction['ci_high']:+.6f}] "
        f"gate_passed={result['gate_passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
