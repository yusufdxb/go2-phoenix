"""Screen a fallback with perfect disturbance-onset timing.

This is the cheapest causal test in the reliability study. It measures whether a
fallback helps when engaged exactly at the true disturbance onset, before monitor
fitting or threshold tuning. It is not an optimal-timing upper bound: preemptive
or delayed intervention may have a different effect.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phoenix.reliability.study import paired_difference, read_protocol


def _disturbance_kind(blocks: list, protocol: dict) -> str | None:
    registered = protocol["params"].get("disturbance_kind")
    if registered is not None:
        return registered

    inferred = set()
    fields = {
        "motor": "motor_scale",
        "obs": "obs_noise",
        "command": "command_speed",
    }
    for block in blocks:
        if not block.disturbed:
            continue
        for kind, field in fields.items():
            if getattr(block, field) is not None:
                inferred.add(kind)
    return inferred.pop() if len(inferred) == 1 else None


def _load_arm(out_dir: Path, arm: str) -> dict:
    data = np.load(out_dir / f"arm_{arm}.npz")
    meta = json.loads((out_dir / f"arm_{arm}.meta.json").read_text())
    return {
        "block_id": np.asarray(data["block_id"]),
        "fell": np.asarray(data["fell"], dtype=bool),
        "engaged": np.asarray(data["engaged"], dtype=bool),
        "meta": meta,
    }


def _ordered_rows(arm: dict, block_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = {int(block_id): row for row, block_id in enumerate(arm["block_id"])}
    if set(index) != set(map(int, block_ids)):
        raise ValueError("arm does not cover exactly the frozen protocol blocks")
    rows = np.asarray([index[int(block_id)] for block_id in block_ids])
    return arm["fell"][rows], arm["engaged"][rows]


def screen_oracle_fallback(
    out_dir: str | Path,
    *,
    n_boot: int = 10_000,
    seed: int = 20260729,
) -> dict:
    """Return the block-level causal effect of perfect-onset fallback use.

    The reported effect is ``unshielded fall rate - oracle fall rate``. Positive
    values mean the fallback prevents falls. Negative values mean it causes
    additional falls even with perfect disturbance timing.
    """

    out_dir = Path(out_dir)
    blocks, protocol = read_protocol(out_dir / "protocol.json")
    arms = {name: _load_arm(out_dir, name) for name in ("unshielded", "oracle")}
    protocol_hash = protocol["protocol_hash"]
    bundle_id = protocol["bundle_id"]

    for name, arm in arms.items():
        if arm["meta"].get("protocol_hash") != protocol_hash:
            raise ValueError(f"{name} arm protocol hash does not match the frozen protocol")
        if arm["meta"].get("bundle_id") != bundle_id:
            raise ValueError(f"{name} arm bundle does not match the frozen protocol")
        if arm["meta"].get("pilot"):
            raise ValueError(f"{name} arm is marked as a pilot")

    block_ids = np.asarray([block.block_id for block in blocks], dtype=np.int64)
    disturbed = np.asarray([block.disturbed for block in blocks], dtype=bool)
    if not disturbed.any() or disturbed.all():
        raise ValueError("oracle screen requires both disturbed and nominal blocks")

    ordered = {
        name: _ordered_rows(arm, block_ids)
        for name, arm in arms.items()
    }
    fall_rates = {
        name: values[0].mean(axis=1)
        for name, values in ordered.items()
    }
    engagement_rates = {
        name: values[1].mean(axis=1)
        for name, values in ordered.items()
    }

    oracle_disturbed_engagement = float(engagement_rates["oracle"][disturbed].mean())
    oracle_nominal_engagement = float(engagement_rates["oracle"][~disturbed].mean())
    if oracle_disturbed_engagement != 1.0 or oracle_nominal_engagement != 0.0:
        raise ValueError(
            "oracle treatment contract violated: expected full disturbed engagement "
            "and zero nominal engagement"
        )

    effect = paired_difference(
        fall_rates["unshielded"][disturbed],
        fall_rates["oracle"][disturbed],
        n_boot=n_boot,
        seed=seed,
    )
    nominal_noise = paired_difference(
        fall_rates["oracle"][~disturbed],
        fall_rates["unshielded"][~disturbed],
        n_boot=n_boot,
        seed=seed + 1,
    )

    if not effect["excludes_zero"]:
        verdict = "inconclusive"
    elif effect["mean_difference"] > 0:
        verdict = "fallback_helps"
    else:
        verdict = "fallback_harms"

    envs_per_block = int(ordered["unshielded"][0].shape[1])
    return {
        "schema_version": 1,
        "bundle_id": bundle_id,
        "protocol_hash": protocol_hash,
        "disturbance_kind": _disturbance_kind(blocks, protocol),
        "n_disturbed_blocks": int(disturbed.sum()),
        "n_nominal_blocks": int((~disturbed).sum()),
        "envs_per_block": envs_per_block,
        "disturbed_fall_rate": {
            name: float(rates[disturbed].mean())
            for name, rates in fall_rates.items()
        },
        "disturbed_fall_count": {
            name: int(ordered[name][0][disturbed].sum())
            for name in arms
        },
        "oracle_engagement": {
            "disturbed": oracle_disturbed_engagement,
            "nominal": oracle_nominal_engagement,
        },
        "oracle_effect_unshielded_minus_oracle": effect,
        "nominal_identical_treatment_noise_oracle_minus_unshielded": nominal_noise,
        "verdict": verdict,
        "interpretation": (
            "Positive effects mean a perfect-onset fallback prevents falls. "
            "Negative effects mean the fallback causes falls even with perfect timing."
        ),
        "limitation": (
            "This screen evaluates intervention exactly at disturbance onset. "
            "It does not exclude benefit from preemptive or delayed intervention."
        ),
    }
