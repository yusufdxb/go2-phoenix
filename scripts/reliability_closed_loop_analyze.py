"""Analyse the paired closed-loop intervention study.

Reads the three arm outputs, verifies they came from the same frozen protocol,
and computes the pre-registered estimands: per-block fall rates, the paired
block-level differences with block-bootstrap CIs, and the intervention's effect
on nominal (undisturbed) blocks. Blocks are the unit of analysis throughout.

Prints an honest verdict. The primary question is causal: does enabling the
shield reduce the fall rate relative to the unshielded policy? The secondary
question isolates the monitor's contribution: does the shield beat a sham that
switches to the same fallback on an information-free schedule?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phoenix.reliability.study import paired_difference, read_protocol


def load_arm(out_dir: Path, arm: str) -> dict:
    data = np.load(out_dir / f"arm_{arm}.npz")
    meta = json.loads((out_dir / f"arm_{arm}.meta.json").read_text())
    return {"block_id": data["block_id"], "fell": data["fell"], "engaged": data["engaged"], "meta": meta}


def block_fall_rates(arm: dict, mask: np.ndarray) -> np.ndarray:
    """Per-block fall rate over the selected blocks, ordered by block_id."""
    order = np.argsort(arm["block_id"])
    fell = arm["fell"][order][mask]
    return fell.mean(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reliability_eval/closed_loop")
    ap.add_argument("--n-disturbed", type=int, default=32)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    _, protocol = read_protocol(out_dir / "protocol.json")
    arms = {a: load_arm(out_dir, a) for a in ("unshielded", "shielded", "sham")}

    # Every arm must be the same protocol and bundle, or the pairing is a lie.
    proto_hash = protocol.get("protocol_hash")
    bundle_id = protocol.get("bundle_id")
    for name, arm in arms.items():
        if arm["meta"].get("protocol_hash") != proto_hash:
            raise SystemExit(f"FAIL: arm '{name}' protocol_hash != frozen protocol")
        if arm["meta"].get("bundle_id") != bundle_id:
            raise SystemExit(f"FAIL: arm '{name}' bundle_id != frozen bundle")
        if arm["meta"].get("pilot"):
            raise SystemExit(f"FAIL: arm '{name}' is a pilot; not part of the study")

    ref_ids = np.sort(arms["unshielded"]["block_id"])
    for name, arm in arms.items():
        if not np.array_equal(np.sort(arm["block_id"]), ref_ids):
            raise SystemExit(f"FAIL: arm '{name}' does not cover the same blocks")

    disturbed = np.sort(ref_ids) < args.n_disturbed
    nominal = ~disturbed
    envs = arms["unshielded"]["fell"].shape[1]

    rates = {a: block_fall_rates(arms[a], disturbed) for a in arms}
    nom_rates = {a: block_fall_rates(arms[a], nominal) for a in arms}
    engaged = {a: block_fall_rates({"block_id": arms[a]["block_id"], "fell": arms[a]["engaged"]}, disturbed) for a in arms}

    print("=" * 72)
    print("PAIRED CLOSED-LOOP INTERVENTION STUDY")
    print(f"bundle={bundle_id}  protocol={str(proto_hash)[:16]}  "
          f"{args.n_disturbed} disturbed + {int(nominal.sum())} nominal blocks x {envs} envs")
    print("=" * 72)
    print(f"{'arm':<12}{'disturbed fall':>16}{'nominal fall':>14}{'engaged':>10}")
    for a in ("unshielded", "shielded", "sham"):
        print(f"{a:<12}{rates[a].mean():>16.3f}{nom_rates[a].mean():>14.3f}{engaged[a].mean():>10.3f}")

    # Primary: unshielded - shielded (positive => shield helps).
    primary = paired_difference(rates["unshielded"], rates["shielded"], seed=0)
    # Secondary: sham - shielded (positive => the monitor's timing helps beyond the act of switching).
    secondary = paired_difference(rates["sham"], rates["shielded"], seed=1)
    # Cost on nominal: shielded - unshielded (positive => the shield causes falls where none occurred).
    nominal_cost = paired_difference(nom_rates["shielded"], nom_rates["unshielded"], seed=2)

    def show(label, res, helps_if_positive):
        d = res["mean_difference"]
        verdict = "HELPS" if (d > 0 and res["excludes_zero"]) else (
            "HARMS" if (d < 0 and res["excludes_zero"]) else "NO EFFECT")
        print(f"\n{label}")
        print(f"  mean difference {d:+.3f}  95% CI [{res['ci_low']:+.3f}, {res['ci_high']:+.3f}]  "
              f"(n={res['n_blocks']} blocks)")
        print(f"  discordant blocks: {res['blocks_a_worse']} vs {res['blocks_b_worse']}, "
              f"{res['blocks_tied']} tied  ->  {verdict}")

    print("\n" + "-" * 72)
    show("PRIMARY  unshielded - shielded  (positive = shield prevents falls)", primary, True)
    show("SECONDARY  sham - shielded  (positive = monitor timing beats blind switching)", secondary, True)
    show("NOMINAL COST  shielded - unshielded on undisturbed blocks", nominal_cost, False)

    summary = {
        "bundle_id": bundle_id,
        "protocol_hash": proto_hash,
        "n_disturbed_blocks": int(disturbed.sum()),
        "n_nominal_blocks": int(nominal.sum()),
        "envs_per_block": envs,
        "disturbed_fall_rate": {a: float(rates[a].mean()) for a in arms},
        "nominal_fall_rate": {a: float(nom_rates[a].mean()) for a in arms},
        "engagement_rate": {a: float(engaged[a].mean()) for a in arms},
        "primary_unshielded_minus_shielded": primary,
        "secondary_sham_minus_shielded": secondary,
        "nominal_cost_shielded_minus_unshielded": nominal_cost,
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
