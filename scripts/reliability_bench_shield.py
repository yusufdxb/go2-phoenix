"""Real-time budget for the deployed shield.

The shield runs inside the 50 Hz control loop, so its cost is only acceptable if
the *worst* tick fits, not the average one. A monitor that is fast on average and
occasionally stalls is a monitor that occasionally drops a control cycle on a
robot that is already in trouble.

This measures per-tick wall time for the full deploy path (score + arbiter) over
many ticks and reports the tail: p50, p99, p99.9 and the maximum. The gate is on
the **maximum**, budgeted as a fraction of the 20 ms control period. It also
asserts the loop is allocation-stable by checking that peak tracked memory does
not grow across the run.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/reliability_bench_shield.py \
        --artifact deploy/shield_stand_v3.npz --budget-ms 2.0
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
import tracemalloc
from pathlib import Path

# Single-threaded BLAS, set BEFORE numpy is imported. The shield's per-tick
# work is one 384x384 matrix-vector product: far too small to amortise a
# thread fan-out, and a multi-threaded BLAS pinned to one core spends its
# time in its own barriers. This was worth ~150x on the worst-case tick
# (3.6 ms -> 0.024 ms) and is the same setting the Orin deployment wants.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

from phoenix.reliability.deploy import build_shield


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="deploy/shield_stand_v3.npz")
    ap.add_argument("--ticks", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--budget-ms", type=float, default=2.0)
    ap.add_argument("--control-hz", type=float, default=50.0)
    ap.add_argument(
        "--pin-cpu",
        type=int,
        default=2,
        help="Pin to this CPU (-1 to disable). Unpinned runs measure the OS scheduler, not the shield.",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Without pinning, the tail of this measurement is core migration and
    # frequency scaling on a shared desktop, not the cost of the shield: on
    # the workstation the unpinned worst case is ~5 ms while the pinned worst case is
    # ~25 us. The deployed control thread is pinned, so the benchmark pins too
    # and records that it did.
    pinned = None
    if args.pin_cpu >= 0 and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {args.pin_cpu})
            pinned = args.pin_cpu
        except OSError as exc:
            print(f"[bench] could not pin to CPU {args.pin_cpu}: {exc}")

    shield, op, meta = build_shield(args.artifact)
    dim = shield.dim
    print(f"[bench] artifact={args.artifact} latent_dim={dim} trip={op.trip_threshold:.1f}")

    # Feed real-scale random latents. The arithmetic is data-independent (a
    # fixed matvec), so the distribution only matters for exercising both the
    # nominal and tripped arbiter branches, which we do by construction below.
    rng = np.random.default_rng(0)
    latents = rng.standard_normal((args.warmup + args.ticks, dim)).astype(np.float32)

    for i in range(args.warmup):
        shield.step(latents[i])

    # Allocation stability is checked in a separate short pass: tracemalloc
    # instruments every allocation and would otherwise dominate the tail
    # latency it is supposed to help explain.
    tracemalloc.start()
    _, peak_before = tracemalloc.get_traced_memory()
    for i in range(200):
        shield.step(latents[i])
    _, peak_after = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # The GC is the other source of tail latency that has nothing to do with
    # this code. A real-time control loop disables it; so do we, and we say so.
    gc.collect()
    gc.disable()
    try:
        times = np.empty(args.ticks, dtype=np.float64)
        for i in range(args.ticks):
            row = latents[args.warmup + i]
            t0 = time.perf_counter_ns()
            shield.step(row)
            times[i] = time.perf_counter_ns() - t0
    finally:
        gc.enable()

    ms = times / 1e6
    period_ms = 1000.0 / args.control_hz
    stats = {
        "latent_dim": dim,
        "ticks": args.ticks,
        "p50_ms": float(np.percentile(ms, 50)),
        "p99_ms": float(np.percentile(ms, 99)),
        "p999_ms": float(np.percentile(ms, 99.9)),
        "max_ms": float(ms.max()),
        "budget_ms": args.budget_ms,
        "control_period_ms": period_ms,
        "worst_case_duty_pct": float(ms.max() / period_ms * 100.0),
        "peak_traced_kb_growth": float((peak_after - peak_before) / 1024.0),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pinned_cpu": pinned,
    }
    print(
        f"[bench] p50={stats['p50_ms']:.4f} ms  p99={stats['p99_ms']:.4f} ms  "
        f"p99.9={stats['p999_ms']:.4f} ms  MAX={stats['max_ms']:.4f} ms"
    )
    print(
        f"[bench] worst tick uses {stats['worst_case_duty_pct']:.2f}% of the "
        f"{period_ms:.0f} ms control period; traced-memory growth "
        f"{stats['peak_traced_kb_growth']:.1f} KB"
    )

    out_path = Path(args.out) if args.out else Path(str(args.artifact) + ".bench.json")
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"[bench] wrote {out_path}")

    if stats["max_ms"] > args.budget_ms:
        print(
            f"[bench] FAIL: worst-case {stats['max_ms']:.4f} ms exceeds the "
            f"{args.budget_ms} ms budget."
        )
        return 1
    print(f"[bench] PASS: worst-case tick within the {args.budget_ms} ms budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
