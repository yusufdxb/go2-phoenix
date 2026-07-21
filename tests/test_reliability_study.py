"""Tests for the paired closed-loop intervention study machinery."""

from __future__ import annotations

import json

import numpy as np
import pytest

from phoenix.reliability.deploy import (
    ArbiterTimings,
    OperatingPoint,
    build_shield,
    save_artifact,
    whitener_from_cholesky,
)
from phoenix.reliability.ood_monitor import MahalanobisScorer
from phoenix.reliability.study import (
    ARMS,
    MIN_ONSET_TICK,
    VectorShield,
    generate_blocks,
    paired_difference,
    read_protocol,
    sham_schedule,
    write_protocol,
)


@pytest.fixture
def nominal() -> np.ndarray:
    rng = np.random.default_rng(0)
    base = rng.standard_normal((1500, 6))
    return base @ rng.standard_normal((6, 6))


@pytest.fixture
def artifact(tmp_path, nominal):
    scorer = MahalanobisScorer.fit(nominal)
    op = OperatingPoint(
        trip_threshold=40.0,
        clear_threshold=10.0,
        trip_persistence=3,
        arming_ticks=15,
        nominal_episode_fpr=0.04,
        falls_warned=1.0,
        median_lead_s=0.64,
        median_full_fallback_lead_s=0.44,
    )
    path = tmp_path / "shield.npz"
    save_artifact(
        path,
        mean=scorer.mean,
        whitener=whitener_from_cholesky(scorer._chol),
        operating_point=op,
        provenance={"control_dt_s": 0.02},
        timings=ArbiterTimings(handoff_ticks=5, latch=True),
    )
    return path


# --- VectorShield ------------------------------------------------------------


def test_vector_shield_envs_are_independent(artifact, nominal):
    """One environment tripping must not drag any other with it."""
    vec = VectorShield(artifact, num_envs=4)
    ood = nominal[0] + 500.0
    for _ in range(40):
        latents = np.stack([ood, nominal[1], nominal[2], nominal[3]])
        blend, _, _ = vec.step(latents)
    assert blend[0] == 1.0
    assert np.all(blend[1:] == 0.0)


def test_vector_shield_respects_arming(artifact, nominal):
    vec = VectorShield(artifact, num_envs=2)
    ood = nominal[0] + 500.0
    arming = vec.operating_point.arming_ticks
    for _ in range(arming):
        blend, score, armed = vec.step(np.stack([ood, ood]))
        assert np.all(blend == 0.0)
        assert not np.any(armed)
        assert np.all(score > vec.operating_point.trip_threshold)
    _, _, armed = vec.step(np.stack([ood, ood]))
    assert np.all(armed)


def test_nominal_blocks_use_none_not_nan():
    """NaN is not valid JSON and would break the protocol hash round-trip."""
    blocks = generate_blocks(n_disturbed=2, n_nominal=2)
    assert all(b.motor_scale is None for b in blocks if not b.disturbed)
    assert all(isinstance(b.motor_scale, float) for b in blocks if b.disturbed)


def test_vector_shield_resets_only_selected_envs(artifact, nominal):
    vec = VectorShield(artifact, num_envs=3)
    ood = nominal[0] + 500.0
    for _ in range(40):
        blend, _, _ = vec.step(np.stack([ood, ood, ood]))
    assert np.all(blend == 1.0)
    vec.reset([1])
    blend, _, armed = vec.step(np.stack([ood, ood, ood]))
    assert blend[1] == 0.0 and not armed[1]
    assert blend[0] == 1.0 and blend[2] == 1.0


def test_vector_shield_rejects_wrong_batch(artifact, nominal):
    vec = VectorShield(artifact, num_envs=3)
    with pytest.raises(ValueError, match="expected 3"):
        vec.step(np.stack([nominal[0], nominal[1]]))


def test_vector_shield_matches_the_scalar_deploy_shield(artifact, nominal):
    """Runtime equivalence: the study must exercise the shipped code path.

    A per-environment trace from VectorShield has to agree tick-for-tick with an
    independently constructed DeployShield fed the same sequence. If these ever
    diverge, the closed-loop study is measuring something the robot will not do.
    """
    rng = np.random.default_rng(3)
    sequence = np.concatenate(
        [nominal[:30], nominal[:25] + 400.0, nominal[:30], nominal[:20] + 900.0]
    )
    rng.shuffle(sequence[:0])  # no-op; ordering is deliberate

    vec = VectorShield(artifact, num_envs=2)
    scalar, _, _ = build_shield(artifact)

    for row in sequence:
        blend, score, armed = vec.step(np.stack([row, row]))
        # Read before stepping, to match VectorShield's convention: `armed`
        # describes the state the tick's decision was made under.
        scalar_armed = scalar.armed
        decision = scalar.step(row)
        assert blend[0] == pytest.approx(decision.blend)
        assert blend[0] == blend[1]
        assert score[0] == pytest.approx(decision.raw_score)
        assert bool(armed[0]) == scalar_armed


# --- scenario blocks + protocol freeze ---------------------------------------


def test_generate_blocks_counts_and_kinds():
    blocks = generate_blocks(n_disturbed=8, n_nominal=4)
    assert len(blocks) == 12
    assert sum(b.disturbed for b in blocks) == 8
    assert len({b.block_id for b in blocks}) == 12


def test_generate_blocks_is_deterministic():
    a = generate_blocks(n_disturbed=6, n_nominal=2, seed=7)
    b = generate_blocks(n_disturbed=6, n_nominal=2, seed=7)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]


def test_generate_blocks_onset_is_after_arming():
    """The disturbance must never land during the arming window."""
    blocks = generate_blocks(n_disturbed=32, n_nominal=16)
    assert all(b.onset_tick >= MIN_ONSET_TICK for b in blocks)


def test_generate_blocks_rejects_early_onset():
    with pytest.raises(ValueError, match="after arming"):
        generate_blocks(onset_range=(10, 50))


def test_generate_blocks_motor_scale_is_continuous():
    """Discrete always-on levels made Phase 3 partly an env-classification task."""
    blocks = [b for b in generate_blocks(n_disturbed=32, n_nominal=0) if b.disturbed]
    assert len({b.motor_scale for b in blocks}) > 20


def test_protocol_roundtrip(tmp_path):
    blocks = generate_blocks(n_disturbed=4, n_nominal=2)
    path = tmp_path / "protocol.json"
    digest = write_protocol(path, blocks, bundle_id="abc123", params={"envs": 16})
    loaded, payload = read_protocol(path)
    assert [b.to_dict() for b in loaded] == [b.to_dict() for b in blocks]
    assert payload["bundle_id"] == "abc123"
    assert payload["arms"] == list(ARMS)
    assert len(digest) == 64


def test_protocol_detects_post_hoc_editing(tmp_path):
    """A protocol changed after seeing an outcome must not pass as registered."""
    blocks = generate_blocks(n_disturbed=4, n_nominal=2)
    path = tmp_path / "protocol.json"
    write_protocol(path, blocks, bundle_id="abc123", params={})
    payload = json.loads(path.read_text())
    payload["blocks"][0]["motor_scale"] = 0.99
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="modified since it was frozen"):
        read_protocol(path)


# --- sham schedule -----------------------------------------------------------


def test_sham_preserves_the_switching_marginals():
    realised = {0: [10, None, 30], 1: [None, None, 5], 2: [7, 8, None], 3: [1, None, None]}
    sham = sham_schedule(realised, seed=0)
    assert sorted(map(str, sum(sham.values(), []))) == sorted(map(str, sum(realised.values(), [])))
    assert set(sham) == set(realised)


def test_sham_never_gives_a_block_its_own_schedule():
    """Otherwise the sham arm would silently be the real shield."""
    realised = {i: [i, None] for i in range(12)}
    for seed in range(25):
        sham = sham_schedule(realised, seed=seed)
        assert all(sham[i] != realised[i] for i in realised)


# --- paired analysis ---------------------------------------------------------


def test_paired_difference_detects_a_real_effect():
    unshielded = np.full(40, 0.5)
    shielded = np.full(40, 0.2)
    out = paired_difference(unshielded, shielded, seed=0)
    assert out["mean_difference"] == pytest.approx(0.3)
    assert out["excludes_zero"]
    assert out["n_blocks"] == 40


def test_paired_difference_reports_no_effect_honestly():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, 40)
    out = paired_difference(a, a.copy(), seed=0)
    assert out["mean_difference"] == pytest.approx(0.0)
    assert not out["excludes_zero"]
    assert out["blocks_tied"] == 40


def test_paired_difference_counts_discordance():
    a = np.array([1.0, 0.0, 1.0, 1.0])
    b = np.array([0.0, 1.0, 1.0, 0.0])
    out = paired_difference(a, b, seed=0)
    assert out["blocks_a_worse"] == 2
    assert out["blocks_b_worse"] == 1
    assert out["blocks_tied"] == 1


def test_paired_difference_requires_matched_blocks():
    with pytest.raises(ValueError, match="same blocks"):
        paired_difference(np.zeros(4), np.zeros(5))


def test_paired_difference_rejects_empty():
    with pytest.raises(ValueError, match="at least one block"):
        paired_difference(np.array([]), np.array([]))
