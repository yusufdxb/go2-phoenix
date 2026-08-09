#!/usr/bin/env python3
"""Generate golden-vector fixtures pinning the C++ runtime to the Python.

The Python deploy path is the oracle. This script runs it over a deterministic
pseudo-random set of inputs and records every input and output, so the native
runtime can replay the identical inputs and be checked against the identical
outputs.

Two design choices matter:

*Hex float encoding.* Every float is written with ``float.hex()`` and parsed in
C++ with ``strtod``/``strtof``. Decimal text does not round-trip IEEE-754
exactly at every precision, and the declared tolerance for several of these
stages is *bit-exact*, so a decimal fixture would inject an error larger than
the thing being measured.

*Tolerances are declared before the comparison runs*, in
``docs/NATIVE_RUNTIME_PLAN.md`` and in the C++ test, derived from the dtype and
operation chain. They are not chosen after looking at a diff.

Covered here:

* ``projected_gravity`` (audit risk R9: a shipped version had gx/gy flipped)
* ``slew_clip`` (audit risk R13: np.clip is minimum(maximum(a, lo), hi), so it
  propagates NaN but CLAMPS infinities, and std::clamp on NaN is undefined
  behaviour. The NaN/Inf asymmetry was found by these fixtures, not by reading
  the code.)
* the gate ladder decision (precedence and the abort cause)

* ONNX policy inference (``action`` and ``latent``), when onnxruntime is
  importable. Both sides must run the SAME onnxruntime version with a single
  intra-op thread and sequential execution, or the comparison measures the
  runtime configuration rather than the port. The version actually used is
  written into the fixture header so a later mismatch is visible rather than
  silent.

* the reliability shield (DeployMonitor score, SimplexArbiter state and blend),
  driven by a score sequence built to walk the arbiter through every state
  transition including the non-finite case that audit risk R17 turns on.

Usage::

    python scripts/generate_parity_fixtures.py \
        runtime/phoenix_core/test/fixtures/parity_v1.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phoenix.sim2real.gate import (  # noqa: E402
    GateConfig,
    Outcome,
    SensorSnapshot,
    evaluate_gates,
)
from phoenix.sim2real.ros2_policy_node import (  # noqa: E402
    _projected_gravity_from_quat,
    _rpy_from_quat_xyzw,
)
from phoenix.sim2real.safety import (  # noqa: E402
    MAX_DELTA_PER_STEP_RAD,
    per_step_clip_array,
)

FORMAT_VERSION = 1
SEC = 1_000_000_000

# Stable integer encoding of the outcome, mirrored by the C++ enum. Pinned
# explicitly rather than relying on declaration order in either language.
OUTCOME_CODE = {
    Outcome.SILENT: 0,
    Outcome.LATCH_SILENT: 1,
    Outcome.PUBLISH_DEFAULT: 2,
    Outcome.LATCH_AND_PUBLISH_DEFAULT: 3,
    Outcome.RUN_POLICY: 4,
}

# Abort reasons are free-form strings in Python and an enum in C++. Map the
# stable prefix, since the attitude reason embeds formatted angles and the
# first-message reason embeds a CSV of missing topics.
# Shield state wire encoding. Pinned to the tuple order the Python telemetry
# uses, independent of either language's enum declaration order (risk R22).
STATE_CODE = {
    "nominal": 0,
    "handoff": 1,
    "fallback": 2,
    "recovering": 3,
}

REASON_CODE = [
    (None, 0),
    ("max_runtime", 1),
    ("external_estop", 2),
    ("estop_heartbeat_stale", 3),
    ("estop_publisher_missing", 4),
    ("sensor_missing", 5),
    ("sensor_stale", 6),
    ("first_message_timeout", 7),
    ("nan_in_joint_state", 8),
    ("nan_in_imu", 9),
    ("attitude", 10),
    ("unknown_safety_gate", 11),
]


def reason_code(reason: str | None) -> int:
    if reason is None:
        return 0
    for prefix, code in REASON_CODE:
        if prefix is not None and reason.startswith(prefix):
            return code
    raise ValueError(f"unmapped abort reason: {reason!r}")


def hx(v) -> str:
    """Hex-encode a float so it round-trips bit-exactly through text."""
    return float(v).hex()


def gravity_records(rng: np.random.Generator, n: int) -> list[str]:
    out = []
    # A fixed set of structurally interesting quaternions first, then random
    # unit quaternions, then deliberately non-finite ones (the gate ladder
    # depends on NaN propagating rather than being clamped).
    fixed = [
        (0.0, 0.0, 0.0, 1.0),
        (0.2588190, 0.0, 0.0, 0.9659258),
        (0.0, 0.3826834, 0.0, 0.9238795),
        (0.0, 0.0, 0.7071068, 0.7071068),
        (0.5, 0.5, 0.5, 0.5),
        (float("nan"), 0.0, 0.0, 1.0),
        (0.0, float("inf"), 0.0, 1.0),
    ]
    quats = list(fixed)
    for _ in range(n):
        v = rng.normal(size=4)
        v /= np.linalg.norm(v)
        quats.append(tuple(float(c) for c in v))

    for x, y, z, w in quats:
        g = _projected_gravity_from_quat(x, y, z, w)
        roll, pitch, _ = _rpy_from_quat_xyzw(x, y, z, w)
        out.append(
            "G "
            + " ".join(hx(c) for c in (x, y, z, w))
            + " "
            + " ".join(hx(c) for c in g)
            + " "
            + hx(roll)
            + " "
            + hx(pitch)
        )
    return out


def slew_records(rng: np.random.Generator, n: int) -> list[str]:
    out = []
    for i in range(n):
        current = rng.normal(scale=0.6, size=12).astype(np.float32)
        # Mix magnitudes so some elements clip and some do not.
        delta = rng.normal(scale=0.4, size=12).astype(np.float32)
        target = (current + delta).astype(np.float32)

        # Salt roughly one record in eight with a non-finite element.
        if i % 8 == 3:
            target[i % 12] = [np.nan, np.inf, -np.inf][i % 3]
        if i % 8 == 7:
            current[i % 12] = np.nan

        clipped = per_step_clip_array(target, current, MAX_DELTA_PER_STEP_RAD)
        clipped = np.asarray(clipped, dtype=np.float32)
        out.append(
            "S "
            + " ".join(hx(v) for v in target)
            + " "
            + " ".join(hx(v) for v in current)
            + " "
            + " ".join(hx(v) for v in clipped)
        )
    return out


def gate_records(rng: np.random.Generator, n: int) -> list[str]:
    config = GateConfig(
        max_runtime_s=120.0,
        estop_timeout_s=0.5,
        sensor_timeout_s=0.2,
        first_message_timeout_s=15.0,
        pitch_rad=0.8,
        roll_rad=0.6,
    )
    out = []
    now = 10 * SEC

    for _i in range(n):
        # Bias generation toward the boundaries: uniformly random inputs almost
        # never trip a gate, so a naive sweep would test the nominal path a
        # thousand times and every abort path zero times.
        latched = bool(rng.random() < 0.1)
        elapsed = float(rng.choice([10.0, 119.999, 120.0, 120.001, 500.0]))
        seen = [bool(rng.random() > 0.15) for _ in range(3)]
        started = now - int(rng.choice([1, 14, 15, 16, 40]) * SEC)

        estop_known = bool(rng.random() > 0.1)
        estop_last = now - int(rng.choice([0.0, 0.4, 0.5, 0.6, 2.0]) * SEC)
        if rng.random() < 0.1:
            estop_last = -1
        estop_value = bool(rng.random() < 0.2)

        imu_last = now - int(rng.choice([0.0, 0.15, 0.2, 0.25, 1.0]) * SEC)
        js_last = now - int(rng.choice([0.0, 0.15, 0.2, 0.25, 1.0]) * SEC)
        if rng.random() < 0.05:
            imu_last = -1
        if rng.random() < 0.05:
            js_last = -1

        q = rng.normal(scale=0.5, size=12).astype(np.float32)
        qd = rng.normal(scale=0.5, size=12).astype(np.float32)
        if rng.random() < 0.1:
            q[int(rng.integers(12))] = np.nan
        if rng.random() < 0.1:
            qd[int(rng.integers(12))] = np.inf

        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        quat = tuple(float(c) for c in quat)
        ang = tuple(float(c) for c in rng.normal(scale=0.5, size=3))
        if rng.random() < 0.08:
            quat = (np.nan, quat[1], quat[2], quat[3])
        if rng.random() < 0.08:
            ang = (np.nan, ang[1], ang[2])

        roll, pitch, _ = _rpy_from_quat_xyzw(*quat)
        # Sometimes force an attitude violation regardless of the quaternion,
        # because random unit quaternions rarely land past the limits.
        if rng.random() < 0.15:
            pitch = float(rng.choice([0.79, 0.8, 0.81, 1.5]))
        if rng.random() < 0.15:
            roll = float(rng.choice([0.59, 0.6, 0.61, -1.2]))

        snap = SensorSnapshot(
            now_ns=now,
            elapsed_s=elapsed,
            node_started_ns=started,
            seen_estop=seen[0],
            seen_imu=seen[1],
            seen_joint_state=seen[2],
            estop_last_ns=None if estop_last < 0 else estop_last,
            estop_value=estop_value if estop_known else None,
            imu_last_ns=None if imu_last < 0 else imu_last,
            joint_state_last_ns=None if js_last < 0 else js_last,
            joint_pos=q,
            joint_vel=qd,
            quat_xyzw=quat,
            ang_vel=ang,
            roll=roll,
            pitch=pitch,
        )
        d = evaluate_gates(snap, config, already_latched=latched)

        fields = [
            "L",
            str(int(latched)),
            hx(elapsed),
            str(now),
            str(started),
            str(int(seen[0])),
            str(int(seen[1])),
            str(int(seen[2])),
            str(estop_last),
            str(int(estop_value)),
            str(int(estop_known)),
            str(imu_last),
            str(js_last),
        ]
        fields += [hx(v) for v in q]
        fields += [hx(v) for v in qd]
        fields += [hx(v) for v in quat]
        fields += [hx(v) for v in ang]
        fields += [hx(roll), hx(pitch)]
        fields += [str(OUTCOME_CODE[d.outcome]), str(reason_code(d.reason))]
        out.append(" ".join(fields))
    return out


def onnx_records(rng: np.random.Generator, n: int, model: Path) -> tuple[list[str], str]:
    """Replay observations through the exported graph via onnxruntime.

    Determinism is enforced, not assumed: one intra-op thread, one inter-op
    thread, sequential execution. Thread count changes reduction order, which
    changes results at the bit level, so an unpinned comparison would be
    measuring thread scheduling.

    No normalization is applied here, and none must be applied in C++ either.
    For some checkpoints it is baked into the graph and for others it is
    absent, so a runtime that normalizes by config flag double-normalizes one
    of them (audit risk R1).
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return [], ""

    if not model.exists():
        return [], ""

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])

    out = []
    for i in range(n):
        if i == 0:
            obs = np.zeros(48, dtype=np.float32)
        elif i == 1:
            obs = np.ones(48, dtype=np.float32)
        else:
            obs = rng.normal(scale=0.8, size=48).astype(np.float32)

        action, latent = sess.run(["action", "latent"], {"obs": obs.reshape(1, -1)})
        out.append(
            "O "
            + " ".join(hx(v) for v in obs)
            + " "
            + " ".join(hx(v) for v in action[0])
            + " "
            + " ".join(hx(v) for v in latent[0])
        )
    return out, ort.__version__


def shield_records(rng: np.random.Generator, n: int) -> list[str]:
    """Drive the real DeployShield through a scripted score trajectory.

    The sequence is designed, not random: a uniform sample would sit in NOMINAL
    forever and prove nothing about handoff, dwell, recovery, re-trip, or the
    non-finite convention. Constants are small and synthetic so the fixture
    stays readable and does not depend on a shipped artifact.
    """
    from phoenix.reliability.arbiter import SimplexArbiter, SimplexArbiterCfg
    from phoenix.reliability.deploy import DeployMonitor, DeployShield

    dim = 8
    mean = rng.normal(size=dim).astype(np.float32)
    # A well-conditioned whitener: identity plus a small perturbation.
    w = (np.eye(dim) + 0.15 * rng.normal(size=(dim, dim))).astype(np.float32)

    cfg = SimplexArbiterCfg(
        trip_threshold=12.0,
        clear_threshold=3.0,
        trip_persistence=3,
        clear_persistence=4,
        handoff_ticks=5,
        recover_ticks=6,
        min_fallback_ticks=4,
        latch=False,
    )
    arming = 5
    shield = DeployShield(
        DeployMonitor(mean, w), SimplexArbiter(cfg), arming_ticks=arming
    )

    out = [
        "H "
        + str(dim)
        + " "
        + hx(cfg.trip_threshold)
        + " "
        + hx(cfg.clear_threshold)
        + " "
        + " ".join(
            str(v)
            for v in (
                cfg.trip_persistence,
                cfg.clear_persistence,
                cfg.handoff_ticks,
                cfg.recover_ticks,
                cfg.min_fallback_ticks,
                int(cfg.latch),
                arming,
            )
        )
        + " "
        + " ".join(hx(v) for v in mean)
        + " "
        + " ".join(hx(v) for v in w.reshape(-1))
    ]

    for i in range(n):
        phase = i % 90
        if phase < 12:
            scale = 0.05          # deep in-distribution: NOMINAL
        elif phase < 40:
            scale = 3.0           # far out: trip, handoff, fallback
        elif phase < 70:
            scale = 0.05          # back in: dwell, clear, recover
        else:
            scale = 1.0           # ambiguous band
        latent = (mean + scale * rng.normal(size=dim)).astype(np.float32)

        # Salt with non-finite latents: R17 says these must count as ABOVE
        # trip and never toward clear. Placed inside the recovery window on
        # purpose, where a wrong sign flips the decision.
        if phase in (60, 61, 62):
            latent[i % dim] = [np.nan, np.inf, -np.inf][i % 3]

        d = shield.step(latent)
        out.append(
            "P "
            + " ".join(hx(v) for v in latent)
            + " "
            + hx(d.raw_score)
            + " "
            + str(STATE_CODE[d.state])
            + " "
            + hx(d.blend)
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path)
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--gravity", type=int, default=500)
    ap.add_argument("--slew", type=int, default=500)
    ap.add_argument("--gate", type=int, default=4000)
    ap.add_argument("--onnx", type=int, default=300)
    ap.add_argument("--shield", type=int, default=900)
    ap.add_argument(
        "--model", type=Path, default=REPO_ROOT / "deploy" / "stand_v3_latent.onnx"
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    lines = [
        f"# phoenix parity fixtures v{FORMAT_VERSION}",
        f"# seed {args.seed}",
        "# generated by scripts/generate_parity_fixtures.py from the Python oracle",
        "# floats are hex-encoded (float.hex) so they round-trip bit-exactly",
        f"# slew max_delta {hx(MAX_DELTA_PER_STEP_RAD)}",
    ]
    onnx_lines, ort_version = onnx_records(rng, args.onnx, args.model)
    if onnx_lines:
        lines.append(f"# onnxruntime {ort_version} model {args.model.name}")
        lines.append("# onnx session: intra_op=1 inter_op=1 SEQUENTIAL CPUExecutionProvider")
    else:
        lines.append("# onnxruntime unavailable: no inference fixtures in this file")

    lines += gravity_records(rng, args.gravity)
    lines += slew_records(rng, args.slew)
    lines += gate_records(rng, args.gate)
    lines += shield_records(rng, args.shield)
    lines += onnx_lines

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")

    n = len(lines) - 5
    print(f"wrote {n} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
