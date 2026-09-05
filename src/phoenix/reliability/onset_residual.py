"""Audit the pre-onset onset-observation residual left by the batched-block harness.

The batched-block harness (``--batched-blocks``) removes temporal carryover between
blocks by construction: block ``i`` owns environments ``i * envs .. i * envs + envs - 1``
and lives for exactly one block, so no block has a predecessor whose simulator state it
could inherit. What it does not remove is *within-tick* coupling: all 768 environments
are advanced by one shared GPU PhysX batch, so a numerical perturbation introduced when
the earliest-onset environments are treated can reach environments that have not yet
reached their own onset tick.

This module measures that residual instead of inferring it. It makes one falsifiable
prediction and checks it: if the channel is within-tick spatial coupling seeded at the
earliest onset in the batch, then whether a block's onset observation diverges between
the paired arms must be decided purely by that block's onset *tick*, with a single
threshold per arm pair and no dependence on block index, disturbance status, or
environment index. Temporal carryover across blocks would instead order divergence by
block index, which the batched harness does not even define.

It also computes a contamination-free sensitivity analysis: the registered primary
estimand restricted to the blocks whose onset observations are bit-identical across
arms. That subset is untouched by the residual, so it bounds how much the residual could
be doing to the headline.

Nothing here changes the frozen gate. ``analyze_registry`` is the registered analysis and
is left alone; this is a separate audit that reads the same artifacts.
"""

from __future__ import annotations

from math import comb, log10
from pathlib import Path

import numpy as np

from phoenix.reliability.replication import (
    DEFAULT_STUDY_ID,
    EXPECTED_CELLS,
    _bootstrap_mean,
    _ordered_arm,
    _read_arm,
    read_registry,
)
from phoenix.reliability.study import read_protocol


def _load_pair(out_dir: Path) -> dict:
    """Load one paired process-cell directory into block-ordered arrays."""

    blocks, protocol = read_protocol(out_dir / "protocol.json")
    params = protocol["params"]
    n_blocks = len(blocks)
    envs = int(params["envs_per_block"])
    block_ids = np.asarray([block.block_id for block in blocks], dtype=np.int64)

    arms = {}
    for arm in ("unshielded", "oracle"):
        raw, _ = _read_arm(out_dir, arm)
        arms[arm] = _ordered_arm(
            raw,
            arm=arm,
            block_ids=block_ids,
            n_blocks=n_blocks,
            envs=envs,
        )
    return {
        "arms": arms,
        "n_blocks": n_blocks,
        "envs": envs,
        "cell_id": params["cell_id"],
        "replicate_id": params["replicate_id"],
        "disturbed": np.asarray([block.disturbed for block in blocks], dtype=bool),
        "onset": np.asarray([block.onset_tick for block in blocks], dtype=np.int64),
    }


def _block_effects(pair: dict) -> np.ndarray:
    """Recompute the registered per-block paired effect over disturbed blocks.

    This mirrors ``validate_replicate``: the rate is taken over the environment pairs
    that are jointly onset-eligible, meaning neither arm fell before onset.
    """

    unshielded = pair["arms"]["unshielded"]
    oracle = pair["arms"]["oracle"]
    disturbed = pair["disturbed"]
    n_blocks = pair["n_blocks"]

    joint_eligible = disturbed[:, None] & ~unshielded["pre_onset_fall"] & ~oracle["pre_onset_fall"]
    eligible_per_block = joint_eligible.sum(axis=1)

    def rate(values: np.ndarray) -> np.ndarray:
        return np.divide(
            (values & joint_eligible).sum(axis=1),
            eligible_per_block,
            out=np.full(n_blocks, np.nan, dtype=np.float64),
            where=eligible_per_block > 0,
        )

    effects = rate(unshielded["post_onset_fall"]) - rate(oracle["post_onset_fall"])
    return effects[disturbed]


def _divergent_blocks(pair: dict) -> np.ndarray:
    """Blocks whose onset observation is not bit-identical between the paired arms."""

    difference = np.abs(
        pair["arms"]["unshielded"]["onset_obs"] - pair["arms"]["oracle"]["onset_obs"]
    )
    return difference.reshape(pair["n_blocks"], -1).max(axis=1) > 0


def audit_replicate(out_dir: str | Path) -> dict:
    """Measure the onset residual for one paired process-cell directory."""

    pair = _load_pair(Path(out_dir))
    unshielded = pair["arms"]["unshielded"]
    oracle = pair["arms"]["oracle"]
    onset = pair["onset"]
    divergent = _divergent_blocks(pair)

    clean_onsets = onset[~divergent]
    divergent_onsets = onset[divergent]
    first_divergent = int(divergent_onsets.min()) if divergent.any() else None
    last_clean = int(clean_onsets.max()) if (~divergent).any() else None

    # The prediction: a single onset-tick threshold separates divergent from clean.
    if first_divergent is None or last_clean is None:
        separated = True
    else:
        separated = first_divergent > last_clean

    earliest = int(onset.min())
    delay_bracket = (
        None
        if not separated or first_divergent is None or last_clean is None
        else [last_clean - earliest, first_divergent - earliest]
    )

    n_divergent = int(divergent.sum())
    n_blocks = pair["n_blocks"]
    # Probability that a uniformly random subset of this size is exactly the
    # top-|subset| blocks by onset tick. Ties in onset ticks make this conservative
    # in the anti-conservative direction only if they straddle the threshold, which
    # perfect separation rules out.
    chance = 1.0 / comb(n_blocks, n_divergent) if 0 < n_divergent < n_blocks else 1.0

    pre_onset_difference = unshielded["pre_onset_fall"].astype(np.int64) - oracle[
        "pre_onset_fall"
    ].astype(np.int64)

    return {
        "cell_id": pair["cell_id"],
        "replicate_id": pair["replicate_id"],
        "blocks": n_blocks,
        "reset_state_max_abs_difference": float(
            np.max(np.abs(unshielded["reset_state"] - oracle["reset_state"]))
        ),
        "initial_observation_max_abs_difference": float(
            np.max(np.abs(unshielded["initial_obs"] - oracle["initial_obs"]))
        ),
        "onset_observation_max_abs_difference": float(
            np.max(np.abs(unshielded["onset_obs"] - oracle["onset_obs"]))
        ),
        "onset_divergent_blocks": n_divergent,
        "earliest_onset_tick": earliest,
        "last_clean_onset_tick": last_clean,
        "first_divergent_onset_tick": first_divergent,
        "onset_threshold_separation": bool(separated),
        "propagation_delay_bracket_ticks": delay_bracket,
        "log10_probability_of_separation_by_chance": float(log10(chance)),
        "pre_onset_fall_difference_environments": int(np.abs(pre_onset_difference).sum()),
        "pre_onset_fall_difference_blocks": int((pre_onset_difference != 0).any(axis=1).sum()),
        "block_effects_all": _block_effects(pair).tolist(),
        "block_effects_contamination_free": _block_effects(pair)[
            (~divergent)[pair["disturbed"]]
        ].tolist(),
        "pre_onset_block_differences": (
            unshielded["pre_onset_fall"][pair["disturbed"]].mean(axis=1)
            - oracle["pre_onset_fall"][pair["disturbed"]].mean(axis=1)
        ).tolist(),
    }


def audit_registry(
    registry_path: str | Path,
    *,
    n_boot: int = 20_000,
    seed: int = 20260730,
) -> dict:
    """Audit the onset residual across every replicate in a frozen registry."""

    registry = read_registry(registry_path)
    registry_root = Path(registry_path).parent
    if registry.get("study_id", DEFAULT_STUDY_ID) is None:
        raise ValueError("registry does not declare a study id")

    replicates = [
        audit_replicate(registry_root / entry["out_dir"]) for entry in registry["entries"]
    ]

    by_cell: dict[str, list[dict]] = {
        cell: [item for item in replicates if item["cell_id"] == cell] for cell in EXPECTED_CELLS
    }

    cells = {}
    for cell, items in sorted(by_cell.items()):
        if not items:
            continue
        registered = [np.asarray(item["block_effects_all"]) for item in items]
        clean = [np.asarray(item["block_effects_contamination_free"]) for item in items]
        pre = [np.asarray(item["pre_onset_block_differences"]) for item in items]
        if any(group.size == 0 for group in clean):
            raise ValueError(f"{cell} has a replicate with no contamination-free blocks")

        # Same bootstrap seeding as the registered pooled-cell analysis, so the
        # registered column here is directly comparable to combined_summary.json.
        point, low, high = _bootstrap_mean(registered, n_boot=n_boot, seed=seed + 100)
        c_point, c_low, c_high = _bootstrap_mean(clean, n_boot=n_boot, seed=seed + 100)
        p_point, p_low, p_high = _bootstrap_mean(pre, n_boot=n_boot, seed=seed + 250)

        cells[cell] = {
            "registered": {
                "blocks": int(sum(group.size for group in registered)),
                "mean_difference": point,
                "ci_low": low,
                "ci_high": high,
            },
            "contamination_free": {
                "blocks": int(sum(group.size for group in clean)),
                "mean_difference": c_point,
                "ci_low": c_low,
                "ci_high": c_high,
            },
            "pre_onset_negative_control": {
                "mean_difference": p_point,
                "ci_low": p_low,
                "ci_high": p_high,
            },
            "sign_agrees": bool(np.sign(point) == np.sign(c_point)),
            "contamination_free_interval_excludes_zero": bool(c_low * c_high > 0),
            "registered_point_inside_contamination_free_interval": bool(c_low <= point <= c_high),
        }

    separated = sum(1 for item in replicates if item["onset_threshold_separation"])
    return {
        "replicates": replicates,
        "cells": cells,
        "summary": {
            "replicates": len(replicates),
            "replicates_with_onset_threshold_separation": separated,
            "log10_joint_probability_of_separation_by_chance": float(
                sum(item["log10_probability_of_separation_by_chance"] for item in replicates)
            ),
            "reset_states_bit_identical_in_every_replicate": all(
                item["reset_state_max_abs_difference"] == 0.0 for item in replicates
            ),
            "initial_observations_bit_identical_in_every_replicate": all(
                item["initial_observation_max_abs_difference"] == 0.0 for item in replicates
            ),
            "pre_onset_fall_difference_environments": int(
                sum(item["pre_onset_fall_difference_environments"] for item in replicates)
            ),
            "all_cells_reproduce_contamination_free": all(
                cell["sign_agrees"] and cell["contamination_free_interval_excludes_zero"]
                for cell in cells.values()
            ),
        },
    }
