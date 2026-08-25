"""Orin-deployable form of the reliability shield.

Phase 3 established *that* policy-latent OOD scoring warns before the robot
falls. This module is what actually runs on the GO2's Jetson at 50 Hz. Three
things separate it from the offline scorers in
:mod:`phoenix.reliability.ood_monitor`:

**Fit in float64, deploy in float32.** The covariance estimate, its Cholesky
factor, and the inversion are numerically delicate; they happen once, offline,
in double precision. What ships is a single dense whitener ``W = L^-1`` in
float32. The deploy-time score is then ``||W (x - mu)||^2``, one matrix-vector
product, no solve, no allocation. :func:`parity_report` is the gate that proves
the float32 path agrees with the float64 one, both numerically and (the part
that actually matters) in its *trip decisions*.

**Fixed allocation.** :class:`DeployMonitor` preallocates its two working
buffers at construction and writes into them with ``out=`` every tick. A 50 Hz
control loop that allocates is a 50 Hz control loop that eventually meets the
garbage collector at the wrong moment.

**Episode-level calibration, baked in.** The single most important Phase 3
finding was that a naive per-frame FPR target produces a shield that engages on
essentially every nominal episode, because latent scores are strongly
autocorrelated in time. The deployed artifact therefore carries the
*episode-calibrated* threshold and the persistence count ``K`` that were
validated together, so the robot cannot be run at an operating point nobody
measured.

The artifact is a single ``.npz``: constants plus a JSON provenance blob
recording which checkpoint, which rollouts, and which measured operating point
produced it. Loading refuses artifacts whose latent dimension disagrees with the
policy, and refuses non-finite constants, fail closed, as everywhere else in
this layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phoenix.reliability.arbiter import ShieldState, SimplexArbiter, SimplexArbiterCfg
from phoenix.reliability.runtime import ShieldDecision

ARTIFACT_VERSION = 2


@dataclass(frozen=True)
class OperatingPoint:
    """The measured operating point a deployed shield is pinned to.

    ``trip_threshold`` / ``clear_threshold`` are on the squared-Mahalanobis
    scale. ``trip_persistence`` is the ``K`` consecutive over-threshold ticks
    required to engage, jointly calibrated with the threshold, never chosen
    independently.

    ``arming_ticks`` is the number of ticks after a start or reset during which
    the shield scores but cannot engage. It exists because the calibration
    discards exactly that many post-reset ticks, and a runtime that armed
    immediately would be operating in a regime nobody measured. Concretely, on
    the stand-v3 rollouts the median nominal score at tick 0 is ~1.1e6 against a
    trip threshold of ~9.2e3: without arming, every single nominal episode
    engages the fallback at startup. Carrying the value in the artifact is what
    keeps calibration and runtime from drifting apart again.

    ``lead`` fields are the *measured* consequences of the operating point,
    carried along so a deployment can be audited without re-reading the study.
    ``median_lead_s`` is the margin to the **decision** tick (the end of the
    K-run, which is when the arbiter actually trips), not to the start of the
    run; ``median_full_fallback_lead_s`` further subtracts the handoff ramp and
    is the margin that physically matters.
    """

    trip_threshold: float
    clear_threshold: float
    trip_persistence: int
    arming_ticks: int
    nominal_episode_fpr: float
    falls_warned: float
    median_lead_s: float
    median_full_fallback_lead_s: float = float("nan")

    def __post_init__(self) -> None:
        if self.arming_ticks < 0:
            raise ValueError("arming_ticks must be >= 0")
        if not self.clear_threshold < self.trip_threshold:
            raise ValueError("clear_threshold must be < trip_threshold")
        for name in ("trip_threshold", "clear_threshold"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")

    def to_dict(self) -> dict:
        return {
            "trip_threshold": float(self.trip_threshold),
            "clear_threshold": float(self.clear_threshold),
            "trip_persistence": int(self.trip_persistence),
            "arming_ticks": int(self.arming_ticks),
            "nominal_episode_fpr": float(self.nominal_episode_fpr),
            "falls_warned": float(self.falls_warned),
            "median_lead_s": float(self.median_lead_s),
            "median_full_fallback_lead_s": float(self.median_full_fallback_lead_s),
        }

    @classmethod
    def from_dict(cls, d: dict) -> OperatingPoint:
        return cls(
            trip_threshold=float(d["trip_threshold"]),
            clear_threshold=float(d["clear_threshold"]),
            trip_persistence=int(d["trip_persistence"]),
            arming_ticks=int(d["arming_ticks"]),
            nominal_episode_fpr=float(d.get("nominal_episode_fpr", float("nan"))),
            falls_warned=float(d.get("falls_warned", float("nan"))),
            median_lead_s=float(d.get("median_lead_s", float("nan"))),
            median_full_fallback_lead_s=float(
                d.get("median_full_fallback_lead_s", float("nan"))
            ),
        )


@dataclass(frozen=True)
class ArbiterTimings:
    """Ramp and release timings, part of the frozen bundle, not launch config.

    These affect outcomes as much as the threshold does: the handoff ramp sets
    how long it takes to actually reach the fallback (and therefore the real
    safety margin), and the release policy decides whether the robot hands
    control back to a policy that just went out of distribution. Freezing them
    in the artifact alongside the threshold means the configuration that was
    measured is the configuration that runs.
    """

    handoff_ticks: int = 10
    recover_ticks: int = 25
    clear_persistence: int = 10
    min_fallback_ticks: int = 20
    latch: bool = False

    def to_dict(self) -> dict:
        return {
            "handoff_ticks": int(self.handoff_ticks),
            "recover_ticks": int(self.recover_ticks),
            "clear_persistence": int(self.clear_persistence),
            "min_fallback_ticks": int(self.min_fallback_ticks),
            "latch": bool(self.latch),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ArbiterTimings:
        return cls(
            handoff_ticks=int(d["handoff_ticks"]),
            recover_ticks=int(d["recover_ticks"]),
            clear_persistence=int(d["clear_persistence"]),
            min_fallback_ticks=int(d["min_fallback_ticks"]),
            latch=bool(d["latch"]),
        )


def whitener_from_cholesky(chol: np.ndarray) -> np.ndarray:
    """Return ``W = L^-1`` for lower-Cholesky ``L``, in float64.

    Inverting the triangular factor once, offline, turns the per-tick cost from
    a triangular solve into a single matrix-vector product, the same arithmetic
    (``||W z||^2 == z^T (L L^T)^-1 z``) with a fixed, allocation-free shape.
    """
    chol = np.asarray(chol, dtype=np.float64)
    if chol.ndim != 2 or chol.shape[0] != chol.shape[1]:
        raise ValueError(f"expected a square Cholesky factor, got {chol.shape}")
    identity = np.eye(chol.shape[0], dtype=np.float64)
    # Lower-triangular solve: L W = I.
    return np.linalg.solve(chol, identity)


class DeployMonitor:
    """Allocation-free float32 Mahalanobis scorer for the control loop.

    Holds the whitener and mean as float32 and scores a single latent vector per
    call into preallocated buffers. Any non-finite element in the input scores
    ``+inf``, matching the offline convention: a garbage frame must push the
    arbiter *toward* the fallback, never silently past it.
    """

    def __init__(self, mean: np.ndarray, whitener: np.ndarray) -> None:
        mean = np.ascontiguousarray(mean, dtype=np.float32).reshape(-1)
        whitener = np.ascontiguousarray(whitener, dtype=np.float32)
        if whitener.shape != (mean.size, mean.size):
            raise ValueError(f"whitener {whitener.shape} does not match mean dim {mean.size}")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(whitener)):
            raise ValueError("monitor constants contain non-finite values")
        self._mean = mean
        self._whitener = whitener
        self._diff = np.zeros(mean.size, dtype=np.float32)
        self._proj = np.zeros(mean.size, dtype=np.float32)

    @property
    def dim(self) -> int:
        return int(self._mean.size)

    @classmethod
    def from_scorer(cls, scorer) -> DeployMonitor:
        """Build the deploy form from a fitted :class:`MahalanobisScorer`."""
        return cls(scorer.mean, whitener_from_cholesky(scorer._chol))

    def score_one(self, feature: np.ndarray) -> float:
        """Squared Mahalanobis distance for one latent vector."""
        feat = np.asarray(feature)
        if feat.size != self._mean.size:
            raise ValueError(f"expected {self._mean.size} features, got {feat.size}")
        np.subtract(feat.reshape(-1), self._mean, out=self._diff, casting="unsafe")
        if not np.all(np.isfinite(self._diff)):
            return float("inf")
        np.dot(self._whitener, self._diff, out=self._proj)
        return float(np.dot(self._proj, self._proj))

    def score(self, features: np.ndarray) -> np.ndarray:
        """Batch convenience wrapper (offline / test use, not the control loop)."""
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return np.array([self.score_one(row) for row in arr], dtype=np.float64)


class DeployShield:
    """The full on-robot shield: fixed-alloc monitor plus the Simplex arbiter.

    One call per control tick. ``step`` returns the same
    :class:`~phoenix.reliability.runtime.ShieldDecision` the offline runtime
    produces, so the sim study and the robot share a decision type.

    For the first ``arming_ticks`` ticks after construction or :meth:`reset`,
    the shield is *disarmed*: it scores every tick (so the telemetry and any
    downstream logging see the true value) but does not advance the arbiter, so
    it cannot engage. This mirrors, exactly, the post-reset warmup the
    calibration discards. Without it the shield engages on startup on a
    perfectly healthy robot, because the immediate post-reset latent is
    enormously far from the steady-state nominal cloud it was fit on.
    """

    def __init__(
        self, monitor: DeployMonitor, arbiter: SimplexArbiter, *, arming_ticks: int = 0
    ) -> None:
        if arming_ticks < 0:
            raise ValueError("arming_ticks must be >= 0")
        self.monitor = monitor
        self.arbiter = arbiter
        self.arming_ticks = int(arming_ticks)
        self._ticks = 0

    @property
    def dim(self) -> int:
        return self.monitor.dim

    @property
    def armed(self) -> bool:
        return self._ticks >= self.arming_ticks

    def reset(self) -> None:
        self.arbiter.reset()
        self._ticks = 0

    def step(self, latent: np.ndarray) -> ShieldDecision:
        raw = self.monitor.score_one(latent)
        if not self.armed:
            self._ticks += 1
            # Disarmed: report the true score, but hold the arbiter at NOMINAL.
            return ShieldDecision(
                blend=0.0, state=ShieldState.NOMINAL, raw_score=raw, filtered_score=raw
            )
        self._ticks += 1
        out = self.arbiter.update(raw)
        return ShieldDecision(
            blend=out.blend, state=out.state, raw_score=raw, filtered_score=raw
        )


def save_artifact(
    path: str | Path,
    *,
    mean: np.ndarray,
    whitener: np.ndarray,
    operating_point: OperatingPoint,
    provenance: dict,
    timings: ArbiterTimings | None = None,
) -> Path:
    """Write the deploy artifact (constants + provenance) to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "artifact_version": ARTIFACT_VERSION,
        "latent_dim": int(np.asarray(mean).size),
        "operating_point": operating_point.to_dict(),
        "timings": (timings or ArbiterTimings()).to_dict(),
        "provenance": provenance,
    }
    np.savez(
        path,
        mean=np.asarray(mean, dtype=np.float32),
        whitener=np.asarray(whitener, dtype=np.float32),
        meta=np.array(json.dumps(meta, indent=2)),
    )
    return path


def load_artifact(path: str | Path, *, expected_dim: int | None = None):
    """Load a deploy artifact into ``(DeployMonitor, OperatingPoint, meta)``.

    Fails closed: an unknown artifact version, a latent-dimension mismatch
    against ``expected_dim``, or non-finite constants all raise rather than
    quietly deploying a shield that scores the wrong thing.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        mean = data["mean"]
        whitener = data["whitener"]
        meta = json.loads(str(data["meta"]))
    version = int(meta.get("artifact_version", -1))
    if version != ARTIFACT_VERSION:
        raise ValueError(f"unsupported artifact version {version} (expected {ARTIFACT_VERSION})")
    # The metadata's declared width is what every downstream compatibility check
    # is made against, so it has to agree with the constants actually stored
    # here, otherwise a truthful-looking dimension check passes while the
    # monitor scores something else entirely.
    if int(meta["latent_dim"]) != int(np.asarray(mean).size):
        raise ValueError(
            f"artifact is internally inconsistent: meta latent_dim={meta['latent_dim']} "
            f"but stored mean has {np.asarray(mean).size} elements"
        )
    if expected_dim is not None and int(meta["latent_dim"]) != int(expected_dim):
        raise ValueError(
            f"artifact latent_dim={meta['latent_dim']} but policy emits {expected_dim}; "
            "the monitor was fit on a different tap or a different policy"
        )
    monitor = DeployMonitor(mean, whitener)
    op = OperatingPoint.from_dict(meta["operating_point"])  # validates in __post_init__
    return monitor, op, meta


def build_shield(
    path: str | Path,
    *,
    expected_dim: int | None = None,
) -> tuple[DeployShield, OperatingPoint, dict]:
    """Load an artifact and construct the shield exactly as it was calibrated.

    Nothing about the shield's behaviour is settable here. The threshold, the
    persistence ``K``, the arming window, the ramps and the release policy all
    come from the artifact, because they were measured together and they all
    change the outcome: the handoff ramp sets the real safety margin, and the
    release policy decides whether the robot hands control back to a policy that
    just went out of distribution. A launch file that can retune any one of them
    is a launch file that can quietly invalidate the whole study.

    To change the configuration, refit the artifact.
    """
    monitor, op, meta = load_artifact(path, expected_dim=expected_dim)
    timings = ArbiterTimings.from_dict(meta["timings"])
    cfg = SimplexArbiterCfg(
        trip_threshold=op.trip_threshold,
        clear_threshold=op.clear_threshold,
        trip_persistence=op.trip_persistence,
        clear_persistence=timings.clear_persistence,
        handoff_ticks=timings.handoff_ticks,
        recover_ticks=timings.recover_ticks,
        min_fallback_ticks=timings.min_fallback_ticks,
        latch=timings.latch,
    )
    shield = DeployShield(monitor, SimplexArbiter(cfg), arming_ticks=op.arming_ticks)
    return shield, op, meta


def parity_report(
    scorer,
    monitor: DeployMonitor,
    samples: np.ndarray,
    *,
    trip_threshold: float,
) -> dict:
    """Compare the float64 fit path against the float32 deploy path.

    Numerical agreement alone is not the interesting question, a shield is a
    *decision*, so the report also counts how often the two paths disagree about
    whether a frame is above the trip threshold. ``decision_disagreement`` is the
    number that gates a deployment; ``max_rel_err`` is diagnostic.
    """
    samples = np.asarray(samples, dtype=np.float64)
    ref = np.asarray(scorer.score(samples), dtype=np.float64)
    got = monitor.score(samples)
    finite = np.isfinite(ref) & np.isfinite(got)
    denom = np.maximum(np.abs(ref[finite]), 1e-12)
    rel = np.abs(got[finite] - ref[finite]) / denom
    disagree = int(np.sum((ref > trip_threshold) != (got > trip_threshold)))
    return {
        "n_samples": int(samples.shape[0]),
        "max_rel_err": float(np.max(rel)) if rel.size else 0.0,
        "median_rel_err": float(np.median(rel)) if rel.size else 0.0,
        "max_abs_err": float(np.max(np.abs(got[finite] - ref[finite]))) if rel.size else 0.0,
        "decision_disagreement": disagree,
        "decision_disagreement_rate": disagree / max(samples.shape[0], 1),
        "trip_threshold": float(trip_threshold),
    }
