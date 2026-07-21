"""Machinery for the paired closed-loop intervention study.

Phase 3 and 4 established that the monitor *warns* before a fall. They did not
establish that the shield *prevents* one: warnings were scored against what the
unshielded policy did, and the fallback was never actually engaged. This module
is the apparatus for the experiment that closes that gap.

Three design commitments, each of which exists to defeat a specific way this
study could produce a meaningless number:

**Blocks, not frames, not environments.** A *scenario block* is a fully specified
disturbance schedule, pre-generated and hashed before any arm runs, then replayed
identically in every arm. The block is the unit of analysis. Sixteen environments
inside one block share its disturbance and are not sixteen independent replicates;
treating them as such would shrink every confidence interval by a factor of four
for free.

**A sham arm.** Given that the shield engages on 100% of episodes under some
shifts that produce zero falls, a two-arm study showing fewer falls with the
shield could just mean "switching to a stand pose under changed physics helps",
with the monitor contributing nothing. The sham arm switches to the same fallback
with the same switching frequency and timing distribution, but on a schedule
permuted across blocks so it cannot know anything about the episode it is in.
Shielded-minus-sham is the comparison that isolates the monitor's *timing*.

**Per-environment shield state.** Each environment gets its own
:class:`~phoenix.reliability.deploy.DeployShield`, constructed from the same
artifact and reset on that environment's own reset, so the closed-loop harness
exercises literally the same code path as the robot rather than a vectorised
re-implementation that might quietly disagree with it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from phoenix.reliability.bundle import value_sha256
from phoenix.reliability.deploy import DeployShield, build_shield

ARM_UNSHIELDED = "unshielded"
ARM_SHIELDED = "shielded"
ARM_SHAM = "sham"
ARMS = (ARM_UNSHIELDED, ARM_SHIELDED, ARM_SHAM)


class VectorShield:
    """One independent :class:`DeployShield` per environment.

    Deliberately a list of the real deploy object rather than a batched
    reimplementation: the whole point of the closed-loop study is that the thing
    being measured is the thing that ships. At 16 environments and ~8 us per
    scalar step this costs well under a millisecond per tick, which is
    irrelevant offline.
    """

    def __init__(self, artifact: str | Path, num_envs: int) -> None:
        self._shields: list[DeployShield] = []
        self.operating_point = None
        self.meta = None
        for _ in range(num_envs):
            shield, op, meta = build_shield(artifact)
            self._shields.append(shield)
            self.operating_point, self.meta = op, meta
        self.num_envs = num_envs

    def reset(self, env_ids=None) -> None:
        """Re-arm the given environments (all of them when ``env_ids`` is None)."""
        ids = range(self.num_envs) if env_ids is None else env_ids
        for i in ids:
            self._shields[i].reset()

    def step(self, latents: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance every environment one tick.

        Returns ``(blend, score, armed)`` arrays of length ``num_envs``.
        """
        latents = np.asarray(latents)
        if latents.shape[0] != self.num_envs:
            raise ValueError(f"expected {self.num_envs} latents, got {latents.shape[0]}")
        blend = np.empty(self.num_envs, dtype=np.float64)
        score = np.empty(self.num_envs, dtype=np.float64)
        armed = np.empty(self.num_envs, dtype=bool)
        for i, shield in enumerate(self._shields):
            # Sampled before the step: `armed` describes the state this tick's
            # decision was made under, not the state it left behind.
            armed[i] = shield.armed
            decision = shield.step(latents[i])
            blend[i] = decision.blend
            score[i] = decision.raw_score
        return blend, score, armed


@dataclass(frozen=True)
class ScenarioBlock:
    """One pre-registered, replayable experimental condition.

    ``onset_tick`` is when the disturbance is applied. It is bounded below by
    ``MIN_ONSET_TICK`` so that the disturbance always lands well after the
    shield has armed and the robot has stabilised — otherwise the study would be
    partly measuring the startup transient rather than the intervention, and the
    arming window would confound the treatment.
    """

    block_id: int
    seed: int
    disturbed: bool
    # Motor-strength scale applied at ``onset_tick``: actuator stiffness and
    # damping multiplied by this factor. None for nominal blocks (not NaN —
    # NaN is not valid JSON and would break the protocol hash round-trip).
    #
    # Motor degradation rather than friction because it is the disturbance that
    # can actually be injected mid-episode: Isaac Lab's friction event term
    # caches a fixed bucket pool at startup and cannot be retargeted later,
    # whereas actuator gains are writable per environment at any tick. It is
    # also the stronger held-out shift — motor strength was never randomised
    # during training at all.
    motor_scale: float | None
    onset_tick: int
    horizon_ticks: int

    def to_dict(self) -> dict:
        return asdict(self)


MIN_ONSET_TICK = 100  # 2.0 s at 50 Hz: long after arming (15) and stabilisation


def generate_blocks(
    *,
    n_disturbed: int = 32,
    n_nominal: int = 16,
    motor_scale_range: tuple[float, float] = (0.30, 0.55),
    onset_range: tuple[int, int] = (100, 200),
    horizon_ticks: int = 500,
    seed: int = 20260720,
) -> list[ScenarioBlock]:
    """Pre-generate the scenario blocks from the registered distribution.

    The motor scale is drawn from a continuous range rather than the two pinned
    values used in Phase 3 (0.45 and 0.6, which produced 19.5% and 0% falls).
    Discrete, always-on shift levels made the detection task partly an
    environment-classification problem; a continuous distribution applied
    mid-episode does not.
    """
    if onset_range[0] < MIN_ONSET_TICK:
        raise ValueError(
            f"onset must be >= {MIN_ONSET_TICK} ticks so the disturbance lands after "
            "arming and stabilisation, not during them"
        )
    rng = np.random.default_rng(seed)
    blocks: list[ScenarioBlock] = []
    for i in range(n_disturbed):
        blocks.append(
            ScenarioBlock(
                block_id=i,
                seed=int(rng.integers(0, 2**31 - 1)),
                disturbed=True,
                motor_scale=float(rng.uniform(*motor_scale_range)),
                onset_tick=int(rng.integers(onset_range[0], onset_range[1] + 1)),
                horizon_ticks=horizon_ticks,
            )
        )
    for j in range(n_nominal):
        blocks.append(
            ScenarioBlock(
                block_id=n_disturbed + j,
                seed=int(rng.integers(0, 2**31 - 1)),
                disturbed=False,
                motor_scale=None,
                onset_tick=int(rng.integers(onset_range[0], onset_range[1] + 1)),
                horizon_ticks=horizon_ticks,
            )
        )
    return blocks


def write_protocol(
    path: str | Path,
    blocks: list[ScenarioBlock],
    *,
    bundle_id: str,
    params: dict,
) -> str:
    """Freeze the protocol to disk and return its hash.

    Written before any arm runs. The returned hash is recorded with the results,
    so a protocol edited after seeing an outcome cannot masquerade as the one
    that was registered.
    """
    payload = {
        "bundle_id": bundle_id,
        "params": params,
        "arms": list(ARMS),
        "blocks": [b.to_dict() for b in blocks],
    }
    protocol_hash = value_sha256(payload)
    payload["protocol_hash"] = protocol_hash
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return protocol_hash


def read_protocol(path: str | Path) -> tuple[list[ScenarioBlock], dict]:
    """Load a frozen protocol, verifying it has not been edited since freezing."""
    payload = json.loads(Path(path).read_text())
    stated = payload.pop("protocol_hash", None)
    recomputed = value_sha256(payload)
    if stated != recomputed:
        raise ValueError(
            f"protocol at {path} has been modified since it was frozen "
            f"(hash {recomputed[:12]} != recorded {str(stated)[:12]})"
        )
    payload["protocol_hash"] = stated  # restored so callers can record it
    blocks = [ScenarioBlock(**b) for b in payload["blocks"]]
    return blocks, payload


def sham_schedule(
    shielded_switch_ticks: dict[int, list[int | None]],
    *,
    seed: int,
) -> dict[int, list[int | None]]:
    """Build the sham arm's switch schedule from the shielded arm's realised one.

    Takes ``{block_id: [switch tick or None per env]}`` as the shield actually
    behaved, and permutes those schedules **across blocks**. The result has the
    same marginal switching frequency and the same timing distribution, but
    within any given episode the switch time carries no information about that
    episode. Any fall reduction the sham arm achieves is therefore attributable
    to the act of standing, not to the monitor.
    """
    rng = np.random.default_rng(seed)
    block_ids = sorted(shielded_switch_ticks)
    donors = rng.permutation(len(block_ids))
    # A block must not donate to itself, or the "sham" would be the real thing.
    for i, d in enumerate(donors):
        if d == i:
            j = (i + 1) % len(block_ids)
            donors[i], donors[j] = donors[j], donors[i]
    return {
        block_ids[i]: list(shielded_switch_ticks[block_ids[int(d)]])
        for i, d in enumerate(donors)
    }


def paired_difference(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Paired block-level difference ``mean(a - b)`` with a block-bootstrap CI.

    ``a`` and ``b`` are per-block fall rates for two arms over the *same* blocks.
    Resampling whole blocks (not environments, not frames) is what keeps the
    interval honest: environments inside a block share a disturbance and are
    positively correlated, so resampling them would understate the variance.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"arms must cover the same blocks, got {a.shape} and {b.shape}")
    if a.size == 0:
        raise ValueError("need at least one block")
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "n_blocks": int(diff.size),
        "mean_difference": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "alpha": alpha,
        # Discordance: blocks where exactly one arm did better. The paired
        # signal lives here; concordant blocks carry no information about the
        # difference.
        "blocks_a_worse": int(np.sum(diff > 0)),
        "blocks_b_worse": int(np.sum(diff < 0)),
        "blocks_tied": int(np.sum(diff == 0)),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }
