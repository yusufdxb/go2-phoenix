"""Tests for the Orin-deployable shield artifact and runtime."""

from __future__ import annotations

import json

import numpy as np
import pytest

from phoenix.reliability.arbiter import ShieldState
from phoenix.reliability.deploy import (
    ARTIFACT_VERSION,
    ArbiterTimings,
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
    # Correlated, anisotropic cloud, an identity covariance would let a broken
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
        arming_ticks=0,
        nominal_episode_fpr=0.02,
        falls_warned=1.0,
        median_lead_s=0.68,
        median_full_fallback_lead_s=0.48,
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
    shield, _, _ = build_shield(path)

    for row in nominal[:100]:
        decision = shield.step(row)
        assert decision.blend == 0.0
        assert decision.state is ShieldState.NOMINAL

    ood = nominal[0] + 50.0
    for _ in range(op.trip_persistence + 20):
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
    shield, _, _ = build_shield(path)
    bad = np.full(scorer.mean.size, np.nan)
    for _ in range(op.trip_persistence + 20):
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


# --- arming window -----------------------------------------------------------
#
# Regression tests for the defect that made the shipped v1 artifact unusable:
# calibration discarded the first 15 post-reset ticks while the runtime armed
# immediately. On the real stand-v3 rollouts the median nominal score at tick 0
# was ~1.1e6 against a trip threshold of ~9.2e3, so all 320 nominal
# environments engaged the fallback at startup on a perfectly healthy robot.


def _write(tmp_path, scorer, op, timings=None):
    path = tmp_path / "shield.npz"
    save_artifact(
        path,
        mean=scorer.mean,
        whitener=whitener_from_cholesky(scorer._chol),
        operating_point=op,
        provenance={},
        timings=timings,
    )
    return path


def _armed_op(**kw) -> OperatingPoint:
    base = dict(
        trip_threshold=40.0,
        clear_threshold=10.0,
        trip_persistence=3,
        arming_ticks=15,
        nominal_episode_fpr=0.02,
        falls_warned=1.0,
        median_lead_s=0.64,
        median_full_fallback_lead_s=0.44,
    )
    base.update(kw)
    return OperatingPoint(**base)


def test_shield_cannot_engage_during_arming(tmp_path, scorer, nominal) -> None:
    """A wildly OOD latent during the arming window must not engage the shield."""
    path = _write(tmp_path, scorer, _armed_op())
    shield, op, _ = build_shield(path)
    ood = nominal[0] + 500.0

    for tick in range(op.arming_ticks):
        assert not shield.armed
        decision = shield.step(ood)
        assert decision.blend == 0.0, f"engaged at tick {tick} during arming"
        assert decision.state is ShieldState.NOMINAL
        # The score is still reported truthfully, only the arbiter is held.
        assert decision.raw_score > op.trip_threshold

    assert shield.armed
    for _ in range(op.trip_persistence + 20):
        decision = shield.step(ood)
    assert decision.blend == 1.0


def test_reset_re_arms_the_shield(tmp_path, scorer, nominal) -> None:
    path = _write(tmp_path, scorer, _armed_op())
    shield, op, _ = build_shield(path)
    for _ in range(op.arming_ticks):
        shield.step(nominal[0])
    assert shield.armed
    shield.reset()
    assert not shield.armed
    assert shield.step(nominal[0] + 500.0).blend == 0.0


def test_arming_ticks_zero_arms_immediately(tmp_path, scorer, op, nominal) -> None:
    path = _write(tmp_path, scorer, op)
    shield, _, _ = build_shield(path)
    assert shield.armed
    for _ in range(op.trip_persistence + 20):
        decision = shield.step(nominal[0] + 500.0)
    assert decision.blend == 1.0


def test_operating_point_rejects_negative_arming() -> None:
    with pytest.raises(ValueError, match="arming_ticks"):
        _armed_op(arming_ticks=-1)


def test_operating_point_rejects_inverted_hysteresis() -> None:
    with pytest.raises(ValueError, match="clear_threshold"):
        _armed_op(clear_threshold=100.0)


# --- frozen bundle -----------------------------------------------------------


def test_build_shield_uses_artifact_timings_not_defaults(tmp_path, scorer) -> None:
    """Ramp and release timings ship with the artifact, not the launch file."""
    timings = ArbiterTimings(
        handoff_ticks=7,
        recover_ticks=33,
        clear_persistence=4,
        min_fallback_ticks=9,
        latch=True,
    )
    path = _write(tmp_path, scorer, _armed_op(), timings=timings)
    shield, _, meta = build_shield(path)
    cfg = shield.arbiter.cfg
    assert cfg.handoff_ticks == 7
    assert cfg.recover_ticks == 33
    assert cfg.clear_persistence == 4
    assert cfg.min_fallback_ticks == 9
    assert cfg.latch is True
    assert meta["timings"]["handoff_ticks"] == 7


def test_load_artifact_rejects_internally_inconsistent_dim(tmp_path, scorer) -> None:
    """A truthful-looking meta width must not mask constants of another size."""
    path = _write(tmp_path, scorer, _armed_op())
    with np.load(path, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files}
    meta = json.loads(str(payload["meta"]))
    meta["latent_dim"] = int(scorer.mean.size) + 3
    payload["meta"] = np.array(json.dumps(meta))
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="internally inconsistent"):
        load_artifact(path)
