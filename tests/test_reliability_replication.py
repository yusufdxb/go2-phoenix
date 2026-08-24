from __future__ import annotations

import json

import numpy as np
import pytest

from phoenix.reliability.bundle import file_sha256
from phoenix.reliability.replication import (
    analyze_registry,
    build_registry,
    validate_replicate,
)
from phoenix.reliability.study import (
    REPLICATION_ARMS,
    generate_blocks,
    read_protocol,
    write_protocol,
)


def _params(*, cell_id="stand_obs", replicate_id="process_01", protocol_seed=101, process_seed=201):
    policy, fault = cell_id.split("_")
    return {
        "study_id": "phoenix_causal_viability_replication_v1",
        "replicate_id": replicate_id,
        "cell_id": cell_id,
        "policy_name": policy,
        "protocol_seed": protocol_seed,
        "process_seed": process_seed,
        "envs_per_block": 16,
        "n_disturbed": 32,
        "n_nominal": 16,
        "horizon_ticks": 500,
        "disturbance_kind": fault,
        "motor_scale_range": [0.3, 0.55],
        "obs_noise_range": [1.0, 3.0],
        "onset_range": [100, 200],
        "source_snapshot_sha256": "source",
        "resolved_env_config_sha256": "config",
    }


def _write_raw(path, arm, blocks, *, helpful=True):
    n_blocks, envs = len(blocks), 16
    disturbed = np.asarray([block.disturbed for block in blocks])
    onset = np.asarray([block.onset_tick for block in blocks])[:, None]
    fell = np.zeros((n_blocks, envs), dtype=bool)
    post = np.zeros_like(fell)
    if helpful:
        width = 8 if arm == "unshielded" else 4
    else:
        width = 4 if arm == "unshielded" else 8
    post[disturbed, :width] = True
    fell |= post
    fall_tick = np.full((n_blocks, envs), -1, dtype=np.int32)
    fall_tick[post] = np.broadcast_to(onset + 20, fall_tick.shape)[post]
    engaged = np.zeros_like(fell)
    switch_tick = np.full((n_blocks, envs), -1, dtype=np.int32)
    full_tick = np.full((n_blocks, envs), -1, dtype=np.int32)
    blend_sum = np.zeros((n_blocks, envs), dtype=np.float64)
    if arm == "oracle":
        engaged[disturbed] = True
        switch_tick[disturbed] = onset[disturbed]
        full_tick[disturbed] = onset[disturbed] + 9
        blend_sum[disturbed] = 50.0
    arrays = {
        "block_id": np.asarray([block.block_id for block in blocks]),
        "fell": fell,
        "pre_onset_fall": np.zeros_like(fell),
        "post_onset_fall": post,
        "switch_tick": switch_tick,
        "full_fallback_tick": full_tick,
        "engaged": engaged,
        "fall_tick": fall_tick,
        "return_until_first_fall": np.ones((n_blocks, envs)),
        "active_ticks": np.full((n_blocks, envs), 500, dtype=np.int32),
        "blend_sum": blend_sum,
        "reset_count": fell.astype(np.int32),
        "fault_injected": np.ones(n_blocks, dtype=bool),
        "reset_state": np.zeros((n_blocks, envs, 37), dtype=np.float32),
        "initial_obs": np.zeros((n_blocks, envs, 2), dtype=np.float32),
        "onset_obs": np.ones((n_blocks, envs, 2), dtype=np.float32),
        "task_complete": ~fell,
    }
    raw_path = path / f"arm_{arm}.npz"
    np.savez(raw_path, **arrays)
    return arrays, raw_path


def _write_meta(path, arm, protocol, params, raw_path):
    meta = {
        "arm": arm,
        "study_id": params["study_id"],
        "cell_id": params["cell_id"],
        "replicate_id": params["replicate_id"],
        "policy_name": params["policy_name"],
        "bundle_id": protocol["bundle_id"],
        "protocol_hash": protocol["protocol_hash"],
        "protocol_seed": params["protocol_seed"],
        "process_seed": params["process_seed"],
        "process_uuid": f"uuid-{path.parent.name}-{path.name}-{arm}",
        "blocks": 48,
        "envs_per_block": 16,
        "pilot": False,
        "source_snapshot_sha256": params["source_snapshot_sha256"],
        "resolved_env_config_sha256": params["resolved_env_config_sha256"],
        "runtime_versions": {"isaaclab": "test"},
        "bundle_file_hashes": {"policy_checkpoint": "policy", "shield_artifact": "fallback"},
        "fallback_contract_sha256": "fallback-contract",
        "trajectory_sha256": f"trajectory-{arm}",
        "raw_output_sha256": file_sha256(raw_path),
    }
    (path / f"arm_{arm}.meta.json").write_text(json.dumps(meta))


def _valid_replicate(tmp_path, *, helpful=True):
    params = _params()
    blocks = generate_blocks(
        n_disturbed=32,
        n_nominal=16,
        disturbance="obs",
        seed=params["protocol_seed"],
    )
    write_protocol(
        tmp_path / "protocol.json",
        blocks,
        bundle_id="bundle",
        params=params,
        arms=REPLICATION_ARMS,
    )
    _, protocol = read_protocol(tmp_path / "protocol.json")
    for arm in REPLICATION_ARMS:
        _, raw_path = _write_raw(tmp_path, arm, blocks, helpful=helpful)
        _write_meta(tmp_path, arm, protocol, params, raw_path)
    return blocks, protocol


def _rewrite_raw_and_hash(path, arm, arrays):
    raw_path = path / f"arm_{arm}.npz"
    np.savez(raw_path, **arrays)
    meta_path = path / f"arm_{arm}.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["raw_output_sha256"] = file_sha256(raw_path)
    meta_path.write_text(json.dumps(meta))


def test_replication_sign_convention_positive_means_fallback_helps(tmp_path):
    _valid_replicate(tmp_path, helpful=True)
    result = validate_replicate(tmp_path, n_boot=100)
    assert result["effect_unshielded_minus_oracle"]["mean_difference"] == pytest.approx(0.25)
    assert result["post_onset_fall_rate_jointly_eligible"] == {
        "unshielded": 0.5,
        "oracle": 0.25,
    }


def test_replication_sign_convention_negative_means_fallback_harms(tmp_path):
    _valid_replicate(tmp_path, helpful=False)
    result = validate_replicate(tmp_path, n_boot=100)
    assert result["effect_unshielded_minus_oracle"]["mean_difference"] == pytest.approx(-0.25)


def test_replication_rejects_duplicate_arm_block_ids(tmp_path):
    _valid_replicate(tmp_path)
    with np.load(tmp_path / "arm_oracle.npz") as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    arrays["block_id"][1] = arrays["block_id"][0]
    _rewrite_raw_and_hash(tmp_path, "oracle", arrays)
    with pytest.raises(ValueError, match="duplicate block IDs"):
        validate_replicate(tmp_path, n_boot=10)


def test_replication_rejects_wrong_oracle_onset(tmp_path):
    _valid_replicate(tmp_path)
    with np.load(tmp_path / "arm_oracle.npz") as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    arrays["switch_tick"][0, 0] += 1
    _rewrite_raw_and_hash(tmp_path, "oracle", arrays)
    with pytest.raises(ValueError, match="registered onset"):
        validate_replicate(tmp_path, n_boot=10)


def test_replication_rejects_nominal_oracle_engagement(tmp_path):
    _valid_replicate(tmp_path)
    with np.load(tmp_path / "arm_oracle.npz") as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    arrays["engaged"][40, 0] = True
    arrays["switch_tick"][40, 0] = 150
    arrays["blend_sum"][40, 0] = 1.0
    _rewrite_raw_and_hash(tmp_path, "oracle", arrays)
    with pytest.raises(ValueError, match="nominal"):
        validate_replicate(tmp_path, n_boot=10)


def test_replication_rejects_process_metadata_mismatch(tmp_path):
    _valid_replicate(tmp_path)
    meta_path = tmp_path / "arm_oracle.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["process_seed"] += 1
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="process_seed"):
        validate_replicate(tmp_path, n_boot=10)


def test_replication_rejects_missing_raw_array(tmp_path):
    _valid_replicate(tmp_path)
    with np.load(tmp_path / "arm_oracle.npz") as loaded:
        arrays = {
            key: np.asarray(loaded[key])
            for key in loaded.files
            if key != "blend_sum"
        }
    _rewrite_raw_and_hash(tmp_path, "oracle", arrays)
    with pytest.raises(ValueError, match="missing arrays"):
        validate_replicate(tmp_path, n_boot=10)


def test_registry_accepts_disjoint_protocols_and_rejects_seed_reuse(tmp_path):
    root = tmp_path / "replication"
    protocol_seed = 1000
    process_seeds = {
        "process_01": 2001,
        "process_02": 2002,
        "process_03": 2003,
    }
    for replicate_id in process_seeds:
        for cell_id in ("stand_motor", "stand_obs", "walk_motor", "walk_obs"):
            params = _params(
                cell_id=cell_id,
                replicate_id=replicate_id,
                protocol_seed=protocol_seed,
                process_seed=process_seeds[replicate_id],
            )
            protocol_seed += 1
            out_dir = root / replicate_id / cell_id
            out_dir.mkdir(parents=True)
            blocks = generate_blocks(
                n_disturbed=32,
                n_nominal=16,
                disturbance=params["disturbance_kind"],
                seed=params["protocol_seed"],
            )
            write_protocol(
                out_dir / "protocol.json",
                blocks,
                bundle_id=f"bundle-{cell_id}",
                params=params,
                arms=REPLICATION_ARMS,
            )
    registry = build_registry(root)
    assert registry["independent_protocols"] == 12
    assert registry["independent_blocks"] == 576

    duplicate = root / "process_03" / "walk_obs" / "protocol.json"
    payload = json.loads(duplicate.read_text())
    payload["params"]["protocol_seed"] = 1000
    payload_without_hash = {key: value for key, value in payload.items() if key != "protocol_hash"}
    from phoenix.reliability.bundle import value_sha256

    payload["protocol_hash"] = value_sha256(payload_without_hash)
    duplicate.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="protocol seed"):
        build_registry(root)


def test_combined_analysis_recovers_fault_interaction(tmp_path):
    root = tmp_path / "replication"
    protocol_seed = 3000
    process_seeds = {
        "process_01": 4001,
        "process_02": 4002,
        "process_03": 4003,
    }
    for replicate_id in process_seeds:
        for cell_id in ("stand_motor", "stand_obs", "walk_motor", "walk_obs"):
            params = _params(
                cell_id=cell_id,
                replicate_id=replicate_id,
                protocol_seed=protocol_seed,
                process_seed=process_seeds[replicate_id],
            )
            protocol_seed += 1
            out_dir = root / replicate_id / cell_id
            out_dir.mkdir(parents=True)
            blocks = generate_blocks(
                n_disturbed=32,
                n_nominal=16,
                disturbance=params["disturbance_kind"],
                seed=params["protocol_seed"],
            )
            write_protocol(
                out_dir / "protocol.json",
                blocks,
                bundle_id=f"bundle-{cell_id}",
                params=params,
                arms=REPLICATION_ARMS,
            )
            _, protocol = read_protocol(out_dir / "protocol.json")
            for arm in REPLICATION_ARMS:
                _, raw_path = _write_raw(
                    out_dir,
                    arm,
                    blocks,
                    helpful=params["disturbance_kind"] == "obs",
                )
                _write_meta(out_dir, arm, protocol, params, raw_path)

    registry = build_registry(root)
    registry_path = root / "registry.json"
    registry_path.write_text(json.dumps(registry))
    result = analyze_registry(registry_path, n_boot=100)
    assert result["gate_passed"]
    assert result["pooled_fault_families"]["motor"]["mean_difference"] == pytest.approx(-0.25)
    assert result["pooled_fault_families"]["obs"]["mean_difference"] == pytest.approx(0.25)
    interaction = result["fault_by_treatment_interaction_obs_minus_motor"]
    assert interaction["mean_difference"] == pytest.approx(0.5)
    assert result["independent_block_accounting"]["disturbed_total"] == 384
