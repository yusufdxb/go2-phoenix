"""Tests for the Orin-deployable shield artifact and runtime."""

from __future__ import annotations

import json

import numpy as np
import pytest

from phoenix.reliability.arbiter import ShieldState
from phoenix.reliability.deploy import (
    ARTIFACT_VERSION,
    DeployMonitor,
    OperatingPoint,
    build_shield,
    load_artifact,
    parity_report,
    save_artifact,
    whitener_from_cholesky,
)
from phoenix.reliability.ood_monitor import MahalanobisScorer


@pytest.fixture
def nominal() -> np.ndarray:
    rng = np.random.default_rng(0)
    # Correlated, anisotropic cloud — an identity covariance would let a broken
    # whitener pass.
    base = rng.standard_normal((2000, 8))
    mix = rng.standard_normal((8, 8))
    return base @ mix + np.arange(8, dtype=np.float64)


@pytest.fixture
def scorer(nominal: np.ndarray) -> MahalanobisScorer:
    return MahalanobisScorer.fit(nominal)


@pytest.fixture
def op() -> OperatingPoint:
    return OperatingPoint(
        trip_threshold=40.0,
        clear_threshold=10.0,
        trip_persistence=3,
        nominal_episode_fpr=0.02,
        falls_warned=1.0,
        median_lead_s=0.68,
    )


def test_whitener_inverts_cholesky(scorer: MahalanobisScorer) -> None:
    w = whitener_from_cholesky(scorer._chol)
    assert np.allclose(w @ scorer._chol, np.eye(w.shape[0]), atol=1e-9)


def test_whitener_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="square"):
        whitener_from_cholesky(np.zeros((3, 4)))


def test_deploy_monitor_matches_float64_scorer(
    scorer: MahalanobisScorer, nominal: np.ndarray
) -> None:
    monitor = DeployMonitor.from_scorer(scorer)
    ref = scorer.score(nominal[:200])
    got = monitor.score(nominal[:200])
    # float32 deploy path, so agreement is to single precision, not exact.
    assert np.allclose(got, ref, rtol=1e-4)


def test_deploy_monitor_scores_ood_higher(scorer: MahalanobisScorer, nominal: np.ndarray) -> None:
    monitor = DeployMonitor.from_scorer(scorer)
    shifted = nominal[:200] + 25.0
    assert monitor.score(shifted).mean() > monitor.score(nominal[:200]).mean() * 10


def test_deploy_monitor_fails_toward_safe_on_non_finite(scorer: MahalanobisScorer) -> None:
    monitor = DeployMonitor.from_scorer(scorer)
    bad = np.zeros(monitor.dim)
    bad[3] = np.nan
    assert monitor.score_one(bad) == float("inf")
    bad[3] = np.inf
    assert monitor.score_one(bad) == float("inf")


def test_deploy_monitor_reuses_buffers(scorer: MahalanobisScorer, nominal: np.ndarray) -> None:
    """The control loop must not allocate per tick: buffers are identity-stable."""
    monitor = DeployMonitor.from_scorer(scorer)
    diff_id, proj_id = id(monitor._diff), id(monitor._proj)
    first = monitor.score_one(nominal[0])
    for row in nominal[:50]:
        monitor.score_one(row)
    assert id(monitor._diff) == diff_id
    assert id(monitor._proj) == proj_id
    # And scoring is stateless: the same input still gives the same answer.
    assert monitor.score_one(nominal[0]) == pytest.approx(first)


def test_deploy_monitor_rejects_wrong_dimension(scorer: MahalanobisScorer) -> None:
    monitor = DeployMonitor.from_scorer(scorer)
    with pytest.raises(ValueError, match="expected"):
        monitor.score_one(np.zeros(monitor.dim + 1))


def test_deploy_monitor_rejects_non_finite_constants() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        DeployMonitor(np.array([0.0, np.nan]), np.eye(2))


def test_deploy_monitor_rejects_mismatched_whitener() -> None:
    with pytest.raises(ValueError, match="does not match"):
        DeployMonitor(np.zeros(4), np.eye(3))


def test_artifact_roundtrip(tmp_path, scorer: MahalanobisScorer, op: OperatingPoint) -> None:
    path = tmp_path / "shield.npz"
    save_artifact(
        path,
        mean=scorer.mean,
        whitener=whitener_from_cholesky(scorer._chol),
        operating_point=op,
        provenance={"checkpoint_sha256": "deadbeef"},
    )
    monitor, loaded_op, meta = load_artifact(path)
    assert monitor.dim == scorer.mean.size
    assert loaded_op == op
    assert meta["provenance"]["checkpoint_sha256"] == "deadbeef"
    assert meta["artifact_version"] == ARTIFACT_VERSION


def test_load_artifact_rejects_dimension_mismatch(
    tmp_path, scorer: MahalanobisScorer, op: OperatingPoint
) -> None:
    """A monitor fit on different taps must not silently deploy."""
    path = tmp_path / "shield.npz"
    save_artifact(
        path,
        mean=scorer.mean,
        whitener=whitener_from_cholesky(scorer._chol),
        operating_point=op,
        provenance={},
    )
    with pytest.raises(ValueError, match="different tap"):
        load_artifact(path, expected_dim=scorer.mean.size + 1)


def test_load_artifact_rejects_unknown_version(
    tmp_path, scorer: MahalanobisScorer, op: OperatingPoint
) -> None:
    path = tmp_path / "shield.npz"
    save_artifact(
        path,
        mean=scorer.mean,
        whitener=whitener_from_cholesky(scorer._chol),
        operating_point=op,
        provenance={},
    )
    with np.load(path, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files}
    meta = json.loads(str(payload["meta"]))
    meta["artifact_version"] = 99
    payload["meta"] = np.array(json.dumps(meta))
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="unsupported artifact version"):
        load_artifact(path)


def test_build_shield_pins_the_calibrated_operating_point(
    tmp_path, scorer: MahalanobisScorer, op: OperatingPoint
) -> None:
    path = tmp_path / "shield.npz"
    save_artifact(
        path,
        mean=scorer.mean,
        whitener=whitener_from_cholesky(scorer._chol),
        operating_point=op,
        provenance={},
    )
    shield, loaded_op, _ = build_shield(path)
    cfg = shield.arbiter.cfg
    assert cfg.trip_threshold == op.trip_threshold
    assert cfg.clear_threshold == op.clear_threshold
    assert cfg.trip_persistence == op.trip_persistence
    assert loaded_op == op


def test_shield_stays_nominal_then_engages(
    tmp_path, scorer: MahalanobisScorer, nominal: np.ndarray, op: OperatingPoint
) -> None:
    path = tmp_path / "shield.npz"
    save_artifact(
        path,
        mean=scorer.mean,
        whitener=whitener_from_cholesky(scorer._chol),
        operating_point=op,
        provenance={},
    )
    shield, _, _ = build_shield(path, handoff_ticks=4)

    for row in nominal[:100]:
        decision = shield.step(row)
        assert decision.blend == 0.0
        assert decision.state is ShieldState.NOMINAL

    ood = nominal[0] + 50.0
    for _ in range(op.trip_persistence + 4):
        decision = shield.step(ood)
    assert decision.blend == 1.0
    assert decision.state is ShieldState.FALLBACK


def test_shield_engages_on_non_finite_latent(
    tmp_path, scorer: MahalanobisScorer, op: OperatingPoint
) -> None:
    path = tmp_path / "shield.npz"
    save_artifact(
        path,
        mean=scorer.mean,
        whitener=whitener_from_cholesky(scorer._chol),
        operating_point=op,
        provenance={},
    )
    shield, _, _ = build_shield(path, handoff_ticks=2)
    bad = np.full(scorer.mean.size, np.nan)
    for _ in range(op.trip_persistence + 2):
        decision = shield.step(bad)
    assert decision.blend == 1.0


def test_parity_report_agrees_on_decisions(
    scorer: MahalanobisScorer, nominal: np.ndarray
) -> None:
    monitor = DeployMonitor.from_scorer(scorer)
    samples = np.concatenate([nominal[:500], nominal[:100] + 30.0])
    report = parity_report(scorer, monitor, samples, trip_threshold=40.0)
    assert report["decision_disagreement"] == 0
    assert report["max_rel_err"] < 1e-4
    assert report["n_samples"] == 600
