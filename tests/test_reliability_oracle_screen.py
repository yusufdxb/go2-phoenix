from __future__ import annotations

import json

import numpy as np
import pytest

from phoenix.reliability.oracle_screen import screen_oracle_fallback
from phoenix.reliability.study import generate_blocks, write_protocol


def _write_arm(path, arm, *, protocol_hash, bundle_id, fell, engaged):
    block_ids = np.arange(fell.shape[0])
    np.savez(
        path / f"arm_{arm}.npz",
        block_id=block_ids,
        fell=fell,
        engaged=engaged,
        switch_tick=np.full_like(fell, -1, dtype=np.int32),
        fall_tick=np.full_like(fell, -1, dtype=np.int32),
    )
    (path / f"arm_{arm}.meta.json").write_text(
        json.dumps(
            {
                "protocol_hash": protocol_hash,
                "bundle_id": bundle_id,
                "pilot": False,
            }
        )
    )


def _study(tmp_path, oracle_disturbed_falls, *, protocol_params=None):
    blocks = generate_blocks(n_disturbed=4, n_nominal=2, seed=7)
    protocol_hash = write_protocol(
        tmp_path / "protocol.json",
        blocks,
        bundle_id="bundle",
        params={"disturbance_kind": "obs"} if protocol_params is None else protocol_params,
    )
    unshielded = np.zeros((6, 8), dtype=bool)
    unshielded[:4, :4] = True
    oracle = np.zeros((6, 8), dtype=bool)
    oracle[:4, :oracle_disturbed_falls] = True
    unshielded_engaged = np.zeros_like(unshielded)
    oracle_engaged = np.zeros_like(oracle)
    oracle_engaged[:4] = True
    _write_arm(
        tmp_path,
        "unshielded",
        protocol_hash=protocol_hash,
        bundle_id="bundle",
        fell=unshielded,
        engaged=unshielded_engaged,
    )
    _write_arm(
        tmp_path,
        "oracle",
        protocol_hash=protocol_hash,
        bundle_id="bundle",
        fell=oracle,
        engaged=oracle_engaged,
    )


def test_oracle_screen_reports_helpful_fallback(tmp_path):
    _study(tmp_path, oracle_disturbed_falls=0)
    result = screen_oracle_fallback(tmp_path, n_boot=500)
    assert result["verdict"] == "fallback_helps"
    assert result["disturbed_fall_count"] == {"unshielded": 16, "oracle": 0}
    assert result["oracle_effect_unshielded_minus_oracle"]["mean_difference"] == 0.5


def test_oracle_screen_reports_harmful_fallback(tmp_path):
    _study(tmp_path, oracle_disturbed_falls=7)
    result = screen_oracle_fallback(tmp_path, n_boot=500)
    assert result["verdict"] == "fallback_harms"
    assert result["oracle_effect_unshielded_minus_oracle"]["mean_difference"] == -0.375


def test_oracle_screen_infers_legacy_disturbance_kind(tmp_path):
    _study(tmp_path, oracle_disturbed_falls=0, protocol_params={})
    result = screen_oracle_fallback(tmp_path, n_boot=100)
    assert result["disturbance_kind"] == "motor"


def test_oracle_screen_fails_closed_on_treatment_contract_violation(tmp_path):
    _study(tmp_path, oracle_disturbed_falls=0)
    path = tmp_path / "arm_oracle.npz"
    data = np.load(path)
    engaged = data["engaged"].copy()
    engaged[0, 0] = False
    np.savez(
        path,
        block_id=data["block_id"],
        fell=data["fell"],
        engaged=engaged,
        switch_tick=data["switch_tick"],
        fall_tick=data["fall_tick"],
    )
    with pytest.raises(ValueError, match="oracle treatment contract"):
        screen_oracle_fallback(tmp_path, n_boot=100)
