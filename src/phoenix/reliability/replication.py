"""Fail-closed validation and analysis for the Phoenix 2x2 replication gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from phoenix.reliability.bundle import file_sha256, value_sha256
from phoenix.reliability.study import paired_difference, read_protocol

EXPECTED_CELLS = {
    "stand_motor": ("stand", "motor"),
    "stand_obs": ("stand", "obs"),
    "walk_motor": ("walk", "motor"),
    "walk_obs": ("walk", "obs"),
}
EXPECTED_REPLICATES = ("process_01", "process_02", "process_03")

#: Study identifier the v1 replication froze. Kept as the default so every
#: existing artifact validates unchanged; a re-run under a changed design
#: passes its own id rather than overwriting v1.
DEFAULT_STUDY_ID = "phoenix_causal_viability_replication_v1"
REQUIRED_ARRAYS = {
    "block_id",
    "fell",
    "pre_onset_fall",
    "post_onset_fall",
    "switch_tick",
    "full_fallback_tick",
    "engaged",
    "fall_tick",
    "return_until_first_fall",
    "active_ticks",
    "blend_sum",
    "reset_count",
    "fault_injected",
    "reset_state",
    "initial_obs",
    "onset_obs",
    "task_complete",
}
ENV_ARRAYS = {
    "fell",
    "pre_onset_fall",
    "post_onset_fall",
    "switch_tick",
    "full_fallback_tick",
    "engaged",
    "fall_tick",
    "return_until_first_fall",
    "active_ticks",
    "blend_sum",
    "reset_count",
    "task_complete",
}


def _scenario_fingerprint(block) -> str:
    return value_sha256(block.to_dict())


def _scenario_set_fingerprint(blocks) -> str:
    return value_sha256([block.to_dict() for block in blocks])


def _read_arm(
    out_dir: Path,
    arm: str,
    *,
    suffix: str = "",
) -> tuple[dict[str, np.ndarray], dict]:
    raw_path = out_dir / f"arm_{arm}{suffix}.npz"
    meta_path = out_dir / f"arm_{arm}{suffix}.meta.json"
    if not raw_path.is_file() or not meta_path.is_file():
        raise ValueError(f"missing {arm} raw output or metadata in {out_dir}")
    with np.load(raw_path, allow_pickle=False) as loaded:
        missing = REQUIRED_ARRAYS - set(loaded.files)
        if missing:
            raise ValueError(f"{arm} raw output is missing arrays: {sorted(missing)}")
        data = {key: np.asarray(loaded[key]) for key in loaded.files}
    meta = json.loads(meta_path.read_text())
    if meta.get("raw_output_sha256") != file_sha256(raw_path):
        raise ValueError(f"{arm} raw-output hash mismatch")
    return data, meta


def validate_preflight_pair(out_dir: str | Path) -> dict:
    """Validate the A-B-N-N-A contract subset for one paired process seed."""

    out_dir = Path(out_dir)
    blocks, protocol = read_protocol(out_dir / "protocol.json")
    params = protocol["params"]
    selected = blocks[:2] + blocks[-2:] + blocks[:1]
    expected_ids = np.asarray([block.block_id for block in selected])
    loaded = {}
    metas = {}
    for arm in ("unshielded", "oracle"):
        data, meta = _read_arm(out_dir, arm, suffix="_preflight")
        if data["block_id"].shape != (5,) or not np.array_equal(data["block_id"], expected_ids):
            raise ValueError(f"{arm} preflight does not contain the registered A-B-N-N-A rows")
        if not meta.get("pilot") or not meta.get("preflight_subset"):
            raise ValueError(f"{arm} preflight metadata is not marked non-study")
        if meta.get("protocol_hash") != protocol["protocol_hash"]:
            raise ValueError(f"{arm} preflight protocol hash mismatch")
        if meta.get("process_seed") != params["process_seed"]:
            raise ValueError(f"{arm} preflight process seed mismatch")
        loaded[arm], metas[arm] = data, meta
    if metas["unshielded"]["process_uuid"] == metas["oracle"]["process_uuid"]:
        raise ValueError("preflight arms did not run in separate simulator processes")

    unshielded, oracle = loaded["unshielded"], loaded["oracle"]
    for arm, data in loaded.items():
        for key in ENV_ARRAYS:
            if data[key].shape != (5, 16):
                raise ValueError(f"{arm} preflight {key} has invalid shape {data[key].shape}")
        if not np.all(data["fault_injected"]):
            raise ValueError(f"{arm} preflight contains a failed fault injection")
        if not np.allclose(data["reset_state"][0], data["reset_state"][4], rtol=0.0, atol=1e-6):
            raise ValueError(f"{arm} A-B-N-N-A reset state did not reproduce")

    if not np.allclose(
        unshielded["reset_state"],
        oracle["reset_state"],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("paired preflight arms do not share registered reset states")
    if unshielded["engaged"].any() or np.any(unshielded["blend_sum"] != 0):
        raise ValueError("unshielded preflight received fallback treatment")

    disturbed_rows = np.asarray([True, True, False, False, True])
    onset = np.asarray([block.onset_tick for block in selected])[:, None]
    expected_onset = np.broadcast_to(onset, oracle["switch_tick"].shape)
    oracle_eligible = disturbed_rows[:, None] & ~oracle["pre_onset_fall"]
    if np.any(oracle["switch_tick"][oracle_eligible] != expected_onset[oracle_eligible]):
        raise ValueError("oracle preflight did not engage at registered onset")
    oracle_ineligible = disturbed_rows[:, None] & oracle["pre_onset_fall"]
    if np.any(oracle["switch_tick"][oracle_ineligible] != -1):
        raise ValueError("oracle preflight treated a replacement episode")
    if oracle["engaged"][~disturbed_rows].any() or np.any(
        oracle["switch_tick"][~disturbed_rows] != -1
    ):
        raise ValueError("oracle preflight engaged on nominal trials")
    return {
        "protocol_seed": params["protocol_seed"],
        "process_seed": params["process_seed"],
        "protocol_hash": protocol["protocol_hash"],
        "scenario_set_sha256": _scenario_set_fingerprint(blocks),
        "trajectory_sha256": {
            arm: metas[arm]["trajectory_sha256"] for arm in ("unshielded", "oracle")
        },
        "process_uuid": {
            arm: metas[arm]["process_uuid"] for arm in ("unshielded", "oracle")
        },
        "reset_replay_passed": True,
        "paired_reset_state_passed": True,
        "pre_treatment_nondeterminism": {
            "initial_observation_max_abs_difference": float(
                np.max(np.abs(unshielded["initial_obs"] - oracle["initial_obs"]))
            ),
            "onset_observation_max_abs_difference": float(
                np.max(np.abs(unshielded["onset_obs"] - oracle["onset_obs"]))
            ),
            "pre_onset_fall_count": {
                "unshielded": int(unshielded["pre_onset_fall"].sum()),
                "oracle": int(oracle["pre_onset_fall"].sum()),
            },
        },
        "oracle_onset_passed": True,
        "nominal_non_engagement_passed": True,
    }


def validate_process_preflights(out_dirs: list[str | Path]) -> dict:
    """Prove same scenarios and distinct process trajectories across three pilots."""

    if len(out_dirs) != 3:
        raise ValueError("process preflight requires exactly three directories")
    entries = []
    for value in out_dirs:
        out_dir = Path(value)
        blocks, protocol = read_protocol(out_dir / "protocol.json")
        data, meta = _read_arm(out_dir, "unshielded", suffix="_preflight")
        entries.append(
            {
                "protocol_seed": protocol["params"]["protocol_seed"],
                "process_seed": protocol["params"]["process_seed"],
                "scenario_set_sha256": _scenario_set_fingerprint(blocks),
                "process_uuid": meta["process_uuid"],
                "trajectory_sha256": meta["trajectory_sha256"],
                "initial_obs_sha256": hashlib.sha256(data["initial_obs"].tobytes()).hexdigest(),
            }
        )
    if len({entry["protocol_seed"] for entry in entries}) != 1:
        raise ValueError("process preflight must hold protocol seed fixed")
    if len({entry["scenario_set_sha256"] for entry in entries}) != 1:
        raise ValueError("identical protocol seeds did not reproduce scenario specifications")
    for key in ("process_seed", "process_uuid", "trajectory_sha256", "initial_obs_sha256"):
        if len({entry[key] for entry in entries}) != 3:
            raise ValueError(f"three process preflights are not distinct by {key}")
    return {
        "processes": entries,
        "identical_protocol_specification_passed": True,
        "distinct_process_trajectories_passed": True,
    }


def _ordered_arm(
    data: dict[str, np.ndarray],
    *,
    arm: str,
    block_ids: np.ndarray,
    n_blocks: int,
    envs: int,
) -> dict[str, np.ndarray]:
    ids = np.asarray(data["block_id"])
    if ids.shape != (n_blocks,):
        raise ValueError(f"{arm} block_id has shape {ids.shape}, expected {(n_blocks,)}")
    if len(np.unique(ids)) != n_blocks:
        raise ValueError(f"{arm} contains duplicate block IDs")
    if set(map(int, ids)) != set(map(int, block_ids)):
        raise ValueError(f"{arm} does not cover exactly the frozen protocol blocks")
    order = np.asarray([int(np.flatnonzero(ids == block_id)[0]) for block_id in block_ids])

    ordered = {}
    for key, value in data.items():
        if value.shape and value.shape[0] == n_blocks:
            ordered[key] = value[order]
        else:
            ordered[key] = value
    for key in ENV_ARRAYS:
        if ordered[key].shape != (n_blocks, envs):
            raise ValueError(
                f"{arm} {key} has shape {ordered[key].shape}, expected {(n_blocks, envs)}"
            )
    if ordered["fault_injected"].shape != (n_blocks,):
        raise ValueError(f"{arm} fault_injected must have one value per block")
    for key in ("reset_state", "initial_obs", "onset_obs"):
        if ordered[key].ndim != 3 or ordered[key].shape[:2] != (n_blocks, envs):
            raise ValueError(f"{arm} {key} has invalid shape {ordered[key].shape}")
    return ordered


def validate_replicate(
    out_dir: str | Path,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    study_id: str = DEFAULT_STUDY_ID,
) -> dict:
    """Validate one paired process-cell directory and return block-level effects."""

    out_dir = Path(out_dir)
    blocks, protocol = read_protocol(out_dir / "protocol.json")
    params = protocol["params"]
    if protocol.get("arms") != ["unshielded", "oracle"]:
        raise ValueError("replication protocol must freeze exactly unshielded and oracle")
    if params.get("study_id") != study_id:
        raise ValueError(
            f"unexpected or missing replication study ID: {params.get('study_id')!r}, "
            f"expected {study_id!r}"
        )
    n_blocks = len(blocks)
    envs = int(params["envs_per_block"])
    if n_blocks != 48 or params.get("n_disturbed") != 32 or params.get("n_nominal") != 16:
        raise ValueError("replication requires exactly 32 disturbed and 16 nominal blocks")
    if envs != 16 or params.get("horizon_ticks") != 500:
        raise ValueError("replication requires 16 envs per block and a 500-tick horizon")

    block_ids = np.asarray([block.block_id for block in blocks], dtype=np.int64)
    if len(np.unique(block_ids)) != n_blocks:
        raise ValueError("protocol contains duplicate block IDs")
    block_seeds = np.asarray([block.seed for block in blocks], dtype=np.int64)
    if len(np.unique(block_seeds)) != n_blocks:
        raise ValueError("protocol contains duplicate block seeds")

    loaded = {}
    metas = {}
    for arm in ("unshielded", "oracle"):
        raw, meta = _read_arm(out_dir, arm)
        loaded[arm] = _ordered_arm(
            raw,
            arm=arm,
            block_ids=block_ids,
            n_blocks=n_blocks,
            envs=envs,
        )
        metas[arm] = meta

    exact_meta = {
        "study_id": params["study_id"],
        "cell_id": params["cell_id"],
        "replicate_id": params["replicate_id"],
        "policy_name": params["policy_name"],
        "bundle_id": protocol["bundle_id"],
        "protocol_hash": protocol["protocol_hash"],
        "protocol_seed": params["protocol_seed"],
        "process_seed": params["process_seed"],
        "blocks": n_blocks,
        "envs_per_block": envs,
        "pilot": False,
        "source_snapshot_sha256": params["source_snapshot_sha256"],
        "resolved_env_config_sha256": params["resolved_env_config_sha256"],
    }
    for arm, meta in metas.items():
        for key, expected in exact_meta.items():
            if meta.get(key) != expected:
                raise ValueError(
                    f"{arm} metadata {key}={meta.get(key)!r}, expected {expected!r}"
                )
        if meta.get("arm") != arm:
            raise ValueError(f"{arm} metadata identifies arm {meta.get('arm')!r}")
        if not meta.get("process_uuid"):
            raise ValueError(f"{arm} metadata lacks a process UUID")
    if metas["unshielded"]["process_uuid"] == metas["oracle"]["process_uuid"]:
        raise ValueError("paired arms must run in separate simulator processes")
    for key in (
        "runtime_versions",
        "bundle_file_hashes",
        "fallback_contract_sha256",
        "source_snapshot_sha256",
        "resolved_env_config_sha256",
    ):
        if metas["unshielded"].get(key) != metas["oracle"].get(key):
            raise ValueError(f"paired-arm metadata mismatch for {key}")

    unshielded = loaded["unshielded"]
    oracle = loaded["oracle"]
    disturbed = np.asarray([block.disturbed for block in blocks], dtype=bool)
    onset = np.asarray([block.onset_tick for block in blocks], dtype=np.int32)[:, None]

    for arm, values in loaded.items():
        if not np.array_equal(values["fell"], values["pre_onset_fall"] | values["post_onset_fall"]):
            raise ValueError(f"{arm} fall partition is inconsistent")
        if not np.array_equal(values["fell"], values["fall_tick"] >= 0):
            raise ValueError(f"{arm} fall labels disagree with fall ticks")
        if not np.array_equal(values["task_complete"], ~values["fell"]):
            raise ValueError(f"{arm} task-completion labels disagree with falls")
        if np.any(values["pre_onset_fall"] & (values["fall_tick"] >= onset)):
            raise ValueError(f"{arm} pre-onset fall ordering is invalid")
        if np.any(values["post_onset_fall"] & (values["fall_tick"] < onset)):
            raise ValueError(f"{arm} post-onset fall ordering is invalid")
        if not np.all(values["fault_injected"]):
            raise ValueError(f"{arm} contains a failed or missing fault injection")

    if unshielded["engaged"].any() or np.any(unshielded["switch_tick"] != -1):
        raise ValueError("unshielded arm received fallback treatment")
    if np.any(unshielded["blend_sum"] != 0):
        raise ValueError("unshielded arm has nonzero fallback dose")

    nominal = ~disturbed
    if oracle["engaged"][nominal].any() or np.any(oracle["switch_tick"][nominal] != -1):
        raise ValueError("oracle engaged on nominal trials")
    if np.any(oracle["blend_sum"][nominal] != 0):
        raise ValueError("oracle nominal trials have nonzero fallback dose")
    oracle_eligible = disturbed[:, None] & ~oracle["pre_onset_fall"]
    expected_onset = np.broadcast_to(onset, oracle["switch_tick"].shape)
    if np.any(oracle["switch_tick"][oracle_eligible] != expected_onset[oracle_eligible]):
        raise ValueError("oracle did not first engage at the registered onset")
    oracle_ineligible = disturbed[:, None] & oracle["pre_onset_fall"]
    if np.any(oracle["switch_tick"][oracle_ineligible] != -1):
        raise ValueError("oracle treated a replacement episode after a pre-onset fall")

    reset_max_abs = float(np.max(np.abs(unshielded["reset_state"] - oracle["reset_state"])))
    initial_obs_max_abs = float(
        np.max(np.abs(unshielded["initial_obs"] - oracle["initial_obs"]))
    )
    onset_max_abs = float(np.max(np.abs(unshielded["onset_obs"] - oracle["onset_obs"])))
    if not np.allclose(
        unshielded["reset_state"],
        oracle["reset_state"],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"paired arms do not share matched reset states; max abs diff {reset_max_abs}"
        )

    joint_eligible = (
        disturbed[:, None]
        & ~unshielded["pre_onset_fall"]
        & ~oracle["pre_onset_fall"]
    )
    eligible_per_block = joint_eligible.sum(axis=1)
    if np.any(eligible_per_block[disturbed] == 0):
        raise ValueError("a disturbed block has no jointly onset-eligible environment pair")

    def eligible_rate(values: np.ndarray) -> np.ndarray:
        numerator = (values & joint_eligible).sum(axis=1)
        return np.divide(
            numerator,
            eligible_per_block,
            out=np.full(n_blocks, np.nan, dtype=np.float64),
            where=eligible_per_block > 0,
        )

    u_post = eligible_rate(unshielded["post_onset_fall"])
    o_post = eligible_rate(oracle["post_onset_fall"])
    effect = paired_difference(
        u_post[disturbed],
        o_post[disturbed],
        n_boot=n_boot,
        seed=seed,
    )
    pre_effect = paired_difference(
        unshielded["pre_onset_fall"][disturbed].mean(axis=1),
        oracle["pre_onset_fall"][disturbed].mean(axis=1),
        n_boot=n_boot,
        seed=seed + 1,
    )

    disturbed_joint = joint_eligible[disturbed]
    u_post_disturbed = unshielded["post_onset_fall"][disturbed]
    o_post_disturbed = oracle["post_onset_fall"][disturbed]
    u_return = unshielded["return_until_first_fall"][disturbed]
    o_return = oracle["return_until_first_fall"][disturbed]
    oracle_switch = oracle["switch_tick"][disturbed]
    disturbed_onset = onset[disturbed]
    latency = oracle_switch - disturbed_onset
    valid_latency = disturbed_joint & (oracle_switch >= 0)
    dose_denominator = oracle["active_ticks"][disturbed]
    normalized_dose = np.divide(
        oracle["blend_sum"][disturbed],
        dose_denominator,
        out=np.zeros_like(oracle["blend_sum"][disturbed], dtype=np.float64),
        where=dose_denominator > 0,
    )

    block_differences = u_post[disturbed] - o_post[disturbed]
    return {
        "schema_version": 1,
        "out_dir": out_dir.as_posix(),
        "cell_id": params["cell_id"],
        "replicate_id": params["replicate_id"],
        "policy_name": params["policy_name"],
        "fault_family": params["disturbance_kind"],
        "protocol_hash": protocol["protocol_hash"],
        "protocol_seed": params["protocol_seed"],
        "process_seed": params["process_seed"],
        "independent_disturbed_blocks": int(disturbed.sum()),
        "independent_nominal_blocks": int(nominal.sum()),
        "within_block_environments": envs,
        "jointly_eligible_environment_pairs": int(disturbed_joint.sum()),
        "pre_onset_fall_count": {
            "unshielded": int(unshielded["pre_onset_fall"][disturbed].sum()),
            "oracle": int(oracle["pre_onset_fall"][disturbed].sum()),
        },
        "post_onset_fall_count_jointly_eligible": {
            "unshielded": int((u_post_disturbed & disturbed_joint).sum()),
            "oracle": int((o_post_disturbed & disturbed_joint).sum()),
        },
        "post_onset_fall_rate_jointly_eligible": {
            "unshielded": float((u_post_disturbed & disturbed_joint).sum() / disturbed_joint.sum()),
            "oracle": float((o_post_disturbed & disturbed_joint).sum() / disturbed_joint.sum()),
        },
        "task_completion_rate_jointly_eligible": {
            "unshielded": float(1.0 - (u_post_disturbed & disturbed_joint).sum() / disturbed_joint.sum()),
            "oracle": float(1.0 - (o_post_disturbed & disturbed_joint).sum() / disturbed_joint.sum()),
        },
        "return_until_first_fall_mean_jointly_eligible": {
            "unshielded": float(u_return[disturbed_joint].mean()),
            "oracle": float(o_return[disturbed_joint].mean()),
        },
        "oracle_fallback_dose_mean": float(normalized_dose[disturbed_joint].mean()),
        "oracle_intervention_latency_ticks": {
            "min": int(latency[valid_latency].min()),
            "median": float(np.median(latency[valid_latency])),
            "max": int(latency[valid_latency].max()),
        },
        "oracle_treated_falls": int(
            (o_post_disturbed & disturbed_joint & oracle["engaged"][disturbed]).sum()
        ),
        "nominal_false_handoff_count": int(oracle["engaged"][nominal].sum()),
        "effect_unshielded_minus_oracle": effect,
        "pre_onset_negative_control_unshielded_minus_oracle": pre_effect,
        "block_differences": block_differences.tolist(),
        "pre_onset_block_differences": (
            unshielded["pre_onset_fall"][disturbed].mean(axis=1)
            - oracle["pre_onset_fall"][disturbed].mean(axis=1)
        ).tolist(),
        "reset_state_max_abs_difference": reset_max_abs,
        "initial_observation_max_abs_difference": initial_obs_max_abs,
        "onset_observation_max_abs_difference": onset_max_abs,
        "trajectory_sha256": {
            arm: metas[arm]["trajectory_sha256"] for arm in ("unshielded", "oracle")
        },
        "process_uuid": {
            arm: metas[arm]["process_uuid"] for arm in ("unshielded", "oracle")
        },
    }


def build_registry(
    root: str | Path,
    *,
    exploratory_protocols: list[str | Path] | None = None,
    study_id: str = DEFAULT_STUDY_ID,
) -> dict:
    """Validate all frozen protocols before any arm runs and return a registry."""

    root = Path(root)
    entries = []
    seen_protocol_seeds: set[int] = set()
    seen_block_seeds: set[int] = set()
    seen_scenarios: set[str] = set()
    exploratory_seeds: set[int] = set()
    for path in exploratory_protocols or []:
        blocks, _ = read_protocol(path)
        exploratory_seeds.update(block.seed for block in blocks)

    for replicate_id in EXPECTED_REPLICATES:
        for cell_id, (policy, fault) in EXPECTED_CELLS.items():
            out_dir = root / replicate_id / cell_id
            blocks, protocol = read_protocol(out_dir / "protocol.json")
            params = protocol["params"]
            if protocol.get("arms") != ["unshielded", "oracle"]:
                raise ValueError(f"{out_dir} did not freeze exactly the required arms")
            expected = {
                "study_id": study_id,
                "replicate_id": replicate_id,
                "cell_id": cell_id,
                "policy_name": policy,
                "disturbance_kind": fault,
                "n_disturbed": 32,
                "n_nominal": 16,
                "envs_per_block": 16,
                "horizon_ticks": 500,
            }
            for key, value in expected.items():
                if params.get(key) != value:
                    raise ValueError(f"{out_dir} parameter {key}={params.get(key)!r}, expected {value!r}")
            protocol_seed = int(params["protocol_seed"])
            if protocol_seed in seen_protocol_seeds:
                raise ValueError(f"protocol seed {protocol_seed} is reused")
            seen_protocol_seeds.add(protocol_seed)

            block_seeds = {block.seed for block in blocks}
            reused = block_seeds & seen_block_seeds
            if reused:
                raise ValueError(f"{out_dir} reuses block seeds: {sorted(reused)[:5]}")
            old_reuse = block_seeds & exploratory_seeds
            if old_reuse:
                raise ValueError(f"{out_dir} reuses exploratory block seeds: {sorted(old_reuse)[:5]}")
            seen_block_seeds.update(block_seeds)

            fingerprints = {_scenario_fingerprint(block) for block in blocks}
            duplicated = fingerprints & seen_scenarios
            if duplicated:
                raise ValueError(f"{out_dir} reuses complete scenario definitions")
            seen_scenarios.update(fingerprints)
            entries.append(
                {
                    "replicate_id": replicate_id,
                    "cell_id": cell_id,
                    "policy_name": policy,
                    "fault_family": fault,
                    "out_dir": out_dir.relative_to(root).as_posix(),
                    "protocol_seed": protocol_seed,
                    "process_seed": int(params["process_seed"]),
                    "protocol_hash": protocol["protocol_hash"],
                    "scenario_set_sha256": _scenario_set_fingerprint(blocks),
                    "source_snapshot_sha256": params["source_snapshot_sha256"],
                    "bundle_id": protocol["bundle_id"],
                }
            )

    process_seeds = {
        replicate_id: {
            entry["process_seed"]
            for entry in entries
            if entry["replicate_id"] == replicate_id
        }
        for replicate_id in EXPECTED_REPLICATES
    }
    if any(len(seeds) != 1 for seeds in process_seeds.values()):
        raise ValueError("all four cells in one replicate must share one registered process seed")
    if len({next(iter(seeds)) for seeds in process_seeds.values()}) != 3:
        raise ValueError("the three replicate process seeds must be distinct")
    source_hashes = {entry["source_snapshot_sha256"] for entry in entries}
    if len(source_hashes) != 1:
        raise ValueError("all protocols must freeze the same experimental source snapshot")

    payload = {
        "schema_version": 1,
        "study_id": study_id,
        "expected_cells": sorted(EXPECTED_CELLS),
        "expected_replicates": list(EXPECTED_REPLICATES),
        "entries": entries,
        "independent_protocols": len(entries),
        "independent_blocks": len(seen_block_seeds),
        "exploratory_protocols_excluded": [Path(p).as_posix() for p in exploratory_protocols or []],
    }
    payload["registry_hash"] = value_sha256(payload)
    return payload


def read_registry(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text())
    stated = payload.pop("registry_hash", None)
    recomputed = value_sha256(payload)
    if stated != recomputed:
        raise ValueError("replication registry was modified after it was frozen")
    payload["registry_hash"] = stated
    if len(payload.get("entries", [])) != 12:
        raise ValueError("replication registry must contain exactly 12 process-cell entries")
    return payload


def _bootstrap_mean(
    groups: list[np.ndarray],
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = float(np.mean(np.concatenate(groups)))
    boots = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        resampled = [
            group[rng.integers(0, len(group), size=len(group))]
            for group in groups
        ]
        boots[index] = np.mean(np.concatenate(resampled))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def analyze_registry(
    registry_path: str | Path,
    *,
    n_boot: int = 20_000,
    seed: int = 20260730,
) -> dict:
    """Analyze all 12 registered process-cell pairs from raw arm outputs.

    Registry entries record their output directory relative to the replication
    root, so ``registry_path`` must sit at that root. Resolving against the
    registry's own directory is what keeps the study reproducible from a fresh
    clone on any machine. Joining an absolute recorded path is a no-op, so a
    registry frozen before that convention still analyzes on the machine that
    produced it.
    """

    registry = read_registry(registry_path)
    registry_root = Path(registry_path).parent
    # Validate every protocol against the id the registry itself declares, so a
    # registry and the artifacts under it cannot silently belong to different
    # studies.
    registry_study_id = registry.get("study_id", DEFAULT_STUDY_ID)
    process_results = [
        validate_replicate(
            registry_root / entry["out_dir"],
            n_boot=n_boot,
            seed=seed + index,
            study_id=registry_study_id,
        )
        for index, entry in enumerate(registry["entries"])
    ]
    process_uuids = [
        process_uuid
        for result in process_results
        for process_uuid in result["process_uuid"].values()
    ]
    if len(set(process_uuids)) != 24:
        raise ValueError("full replication does not contain 24 unique arm-process UUIDs")
    by_cell: dict[str, list[dict]] = {
        cell: [result for result in process_results if result["cell_id"] == cell]
        for cell in EXPECTED_CELLS
    }
    if any(len(results) != 3 for results in by_cell.values()):
        raise ValueError("each cell must contain exactly three process results")

    pooled_cells = {}
    pooled_pre_onset_cells = {}
    for cell, results in by_cell.items():
        groups = [np.asarray(result["block_differences"]) for result in results]
        point, lo, hi = _bootstrap_mean(groups, n_boot=n_boot, seed=seed + 100)
        process_effects = [
            result["effect_unshielded_minus_oracle"]["mean_difference"]
            for result in results
        ]
        loo = {}
        for omitted in EXPECTED_REPLICATES:
            kept = [
                np.asarray(result["block_differences"])
                for result in results
                if result["replicate_id"] != omitted
            ]
            estimate, ci_low, ci_high = _bootstrap_mean(
                kept,
                n_boot=n_boot,
                seed=seed + 200 + EXPECTED_REPLICATES.index(omitted),
            )
            loo[omitted] = {
                "mean_difference": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        pooled_cells[cell] = {
            "independent_disturbed_blocks": int(sum(len(group) for group in groups)),
            "mean_difference": point,
            "ci_low": lo,
            "ci_high": hi,
            "process_effects": process_effects,
            "process_effect_sd": float(np.std(process_effects, ddof=1)),
            "process_effect_range": [float(min(process_effects)), float(max(process_effects))],
            "leave_one_process_out": loo,
            "unshielded_fall_count": int(
                sum(result["post_onset_fall_count_jointly_eligible"]["unshielded"] for result in results)
            ),
            "oracle_fall_count": int(
                sum(result["post_onset_fall_count_jointly_eligible"]["oracle"] for result in results)
            ),
            "jointly_eligible_environment_pairs": int(
                sum(result["jointly_eligible_environment_pairs"] for result in results)
            ),
        }
        denominator = pooled_cells[cell]["jointly_eligible_environment_pairs"]
        pooled_cells[cell]["unshielded_fall_rate"] = (
            pooled_cells[cell]["unshielded_fall_count"] / denominator
        )
        pooled_cells[cell]["oracle_fall_rate"] = (
            pooled_cells[cell]["oracle_fall_count"] / denominator
        )
        pre_groups = [
            np.asarray(result["pre_onset_block_differences"])
            for result in results
        ]
        pre_point, pre_lo, pre_hi = _bootstrap_mean(
            pre_groups,
            n_boot=n_boot,
            seed=seed + 250,
        )
        pooled_pre_onset_cells[cell] = {
            "mean_difference": pre_point,
            "ci_low": pre_lo,
            "ci_high": pre_hi,
        }

    pooled_faults = {}
    pooled_pre_onset_faults = {}
    for fault in ("motor", "obs"):
        results = [result for result in process_results if result["fault_family"] == fault]
        groups = [np.asarray(result["block_differences"]) for result in results]
        point, lo, hi = _bootstrap_mean(groups, n_boot=n_boot, seed=seed + 300)
        pooled_faults[fault] = {
            "independent_disturbed_blocks": int(sum(len(group) for group in groups)),
            "mean_difference": point,
            "ci_low": lo,
            "ci_high": hi,
        }
        pre_groups = [
            np.asarray(result["pre_onset_block_differences"])
            for result in results
        ]
        pre_point, pre_lo, pre_hi = _bootstrap_mean(
            pre_groups,
            n_boot=n_boot,
            seed=seed + 350,
        )
        pooled_pre_onset_faults[fault] = {
            "mean_difference": pre_point,
            "ci_low": pre_lo,
            "ci_high": pre_hi,
        }

    obs_groups = [
        np.asarray(result["block_differences"])
        for result in process_results
        if result["fault_family"] == "obs"
    ]
    motor_groups = [
        np.asarray(result["block_differences"])
        for result in process_results
        if result["fault_family"] == "motor"
    ]
    rng = np.random.default_rng(seed + 400)
    interaction_point = float(
        np.mean(np.concatenate(obs_groups)) - np.mean(np.concatenate(motor_groups))
    )
    interaction_boot = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        obs = np.concatenate(
            [group[rng.integers(0, len(group), len(group))] for group in obs_groups]
        )
        motor = np.concatenate(
            [group[rng.integers(0, len(group), len(group))] for group in motor_groups]
        )
        interaction_boot[index] = obs.mean() - motor.mean()
    interaction_lo, interaction_hi = np.percentile(interaction_boot, [2.5, 97.5])

    expected_direction = {
        "motor": lambda value: value < 0,
        "obs": lambda value: value > 0,
    }
    direction_by_process = all(
        expected_direction[result["fault_family"]](
            result["effect_unshielded_minus_oracle"]["mean_difference"]
        )
        for result in process_results
    )
    fault_intervals_pass = (
        pooled_faults["motor"]["ci_high"] < 0
        and pooled_faults["obs"]["ci_low"] > 0
    )
    loo_direction_pass = all(
        expected_direction[EXPECTED_CELLS[cell][1]](entry["mean_difference"])
        for cell, summary in pooled_cells.items()
        for entry in summary["leave_one_process_out"].values()
    )
    pre_onset_controls_pass = all(
        entry["ci_low"] <= 0 <= entry["ci_high"]
        for entry in (
            list(pooled_pre_onset_cells.values())
            + list(pooled_pre_onset_faults.values())
        )
    )

    return {
        "schema_version": 1,
        "registry_hash": registry["registry_hash"],
        "sign_convention": (
            "positive unshielded-minus-oracle means fallback reduces post-onset falls; "
            "negative means fallback increases post-onset falls"
        ),
        "process_results": process_results,
        "independent_block_accounting": {
            "disturbed_per_process_cell": 32,
            "disturbed_per_pooled_cell": 96,
            "disturbed_per_fault_family": 192,
            "disturbed_total": 384,
            "nominal_total": 192,
        },
        "pooled_cells": pooled_cells,
        "pooled_fault_families": pooled_faults,
        "pooled_pre_onset_negative_control_cells": pooled_pre_onset_cells,
        "pooled_pre_onset_negative_control_fault_families": pooled_pre_onset_faults,
        "fault_by_treatment_interaction_obs_minus_motor": {
            "mean_difference": interaction_point,
            "ci_low": float(interaction_lo),
            "ci_high": float(interaction_hi),
        },
        "gate_checks": {
            "direction_reproduces_in_all_process_cells": direction_by_process,
            "pooled_fault_intervals_exclude_zero": fault_intervals_pass,
            "leave_one_process_out_preserves_direction": loo_direction_pass,
            "pre_onset_negative_controls_include_zero": pre_onset_controls_pass,
        },
        "gate_passed": bool(
            direction_by_process
            and fault_intervals_pass
            and loo_direction_pass
            and pre_onset_controls_pass
        ),
    }


def stable_json_sha256(value) -> str:
    """Public helper used by tests and command-line artifact writers."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
