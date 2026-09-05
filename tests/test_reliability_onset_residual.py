from __future__ import annotations

import numpy as np

from phoenix.reliability.onset_residual import audit_replicate
from phoenix.reliability.study import REPLICATION_ARMS, read_protocol
from tests.test_reliability_replication import _rewrite_raw_and_hash, _valid_replicate


def _load_arrays(tmp_path, arm):
    with np.load(tmp_path / f"arm_{arm}.npz") as data:
        return {key: data[key] for key in data.files}


def _stamp_onset_obs(tmp_path, blocks, *, threshold):
    """Make the oracle arm's onset observation differ exactly on late-onset blocks."""

    onset = np.asarray([block.onset_tick for block in blocks])
    arrays = _load_arrays(tmp_path, "oracle")
    onset_obs = arrays["onset_obs"].copy()
    onset_obs[onset >= threshold] += 0.5
    arrays["onset_obs"] = onset_obs
    _rewrite_raw_and_hash(tmp_path, "oracle", arrays)
    return onset


def test_audit_detects_perfect_onset_threshold_separation(tmp_path):
    blocks, _ = _valid_replicate(tmp_path)
    onset = _stamp_onset_obs(tmp_path, blocks, threshold=160)

    result = audit_replicate(tmp_path)

    expected_divergent = int((onset >= 160).sum())
    assert result["onset_divergent_blocks"] == expected_divergent
    assert result["onset_threshold_separation"] is True
    assert result["first_divergent_onset_tick"] > result["last_clean_onset_tick"]
    assert result["onset_observation_max_abs_difference"] == 0.5
    # A perfectly separated split of 48 blocks is astronomically unlikely by chance.
    assert result["log10_probability_of_separation_by_chance"] < -5.0


def test_audit_reports_no_separation_when_divergence_ignores_onset(tmp_path):
    blocks, _ = _valid_replicate(tmp_path)
    arrays = _load_arrays(tmp_path, "oracle")
    onset_obs = arrays["onset_obs"].copy()
    # Diverge on alternating block indices, which cuts across the onset ordering.
    onset_obs[::2] += 0.5
    arrays["onset_obs"] = onset_obs
    _rewrite_raw_and_hash(tmp_path, "oracle", arrays)

    result = audit_replicate(tmp_path)

    assert result["onset_divergent_blocks"] == 24
    assert result["onset_threshold_separation"] is False
    assert result["propagation_delay_bracket_ticks"] is None


def test_audit_bit_identical_arms_report_zero_residual(tmp_path):
    _valid_replicate(tmp_path)

    result = audit_replicate(tmp_path)

    assert result["onset_divergent_blocks"] == 0
    assert result["onset_observation_max_abs_difference"] == 0.0
    assert result["reset_state_max_abs_difference"] == 0.0
    assert result["initial_observation_max_abs_difference"] == 0.0
    assert result["pre_onset_fall_difference_environments"] == 0
    # With nothing contaminated the two estimands are the same set of blocks.
    assert result["block_effects_contamination_free"] == result["block_effects_all"]


def test_contamination_free_subset_is_the_early_onset_disturbed_blocks(tmp_path):
    blocks, _ = _valid_replicate(tmp_path)
    onset = _stamp_onset_obs(tmp_path, blocks, threshold=160)
    disturbed = np.asarray([block.disturbed for block in blocks])

    result = audit_replicate(tmp_path)

    expected = int((disturbed & (onset < 160)).sum())
    assert len(result["block_effects_contamination_free"]) == expected
    assert len(result["block_effects_all"]) == int(disturbed.sum())


def test_registered_block_effects_match_validate_replicate(tmp_path):
    """The audit recomputes the registered estimand rather than trusting a summary."""

    from phoenix.reliability.replication import validate_replicate

    _valid_replicate(tmp_path)
    registered = validate_replicate(tmp_path, n_boot=200, seed=7)
    audited = audit_replicate(tmp_path)

    assert audited["block_effects_all"] == registered["block_differences"]
    assert audited["pre_onset_block_differences"] == registered["pre_onset_block_differences"]


def test_audit_reads_both_arms_from_the_frozen_protocol(tmp_path):
    _valid_replicate(tmp_path)
    _, protocol = read_protocol(tmp_path / "protocol.json")

    result = audit_replicate(tmp_path)

    assert protocol["arms"] == list(REPLICATION_ARMS)
    assert result["cell_id"] == protocol["params"]["cell_id"]
    assert result["replicate_id"] == protocol["params"]["replicate_id"]
    assert result["blocks"] == 48
