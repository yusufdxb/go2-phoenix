"""Re-run the FP friction estimator on raw probe traces and compare with the stored estimates.

Read-only: walks ``<root>/**/FP_*/fp_trace.npz``, re-estimates with
:func:`phoenix.sim2real.friction_probe.estimate` (same loader as the node's ``estimate``
subcommand) and diffs every numeric field against the sibling ``fp_estimate.json``
written on the robot. Nothing under ``root`` is modified.

    PYTHONPATH=src python scripts/reproduce_fp_estimates.py --root <hw_sessions dir> [--out X.json]

Exit 0 when every stored value is reproduced within ``--tol``; 1 otherwise; 2 when no
traces are found.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from phoenix.sim2real import friction_probe as fp
from phoenix.sim2real.friction_probe_node import load_trace_npz

FIELDS = ("fs_nm", "bias_nm", "kinetic_nm", "tau_plus_nm", "tau_minus_nm", "fs_tau_est_nm", "bias_tau_est_nm")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_run(npz: Path, tol: float) -> dict[str, Any]:
    new = {j.name: j for j in fp.estimate(load_trace_npz(npz)).joints}
    stored_path = npz.parent / "fp_estimate.json"
    out: dict[str, Any] = {"run": npz.parent.name, "trace": str(npz), "trace_sha256": _sha256(npz)}
    if not stored_path.exists():
        out["status"] = "no_stored_estimate"
        return out
    out["estimate_sha256"] = _sha256(stored_path)
    mismatches: list[str] = []
    worst = 0.0
    joints: list[dict[str, Any]] = []
    for s in json.loads(stored_path.read_text())["joints"]:
        n = new[s["name"]]
        for k in FIELDS:
            a, b = s.get(k), getattr(n, k, None)
            if (a is None) != (b is None):
                mismatches.append(f"{s['name']}.{k}: stored={a} reproduced={b}")
            elif a is not None:
                worst = max(worst, abs(a - b))
                if abs(a - b) > tol:
                    mismatches.append(f"{s['name']}.{k}: stored={a} reproduced={b}")
        if n.fs_nm is not None:
            joints.append({k: getattr(n, k, None) for k in ("name", *FIELDS)} | {"flags": list(n.flags)})
    out |= {"status": "ok" if not mismatches else "mismatch", "max_abs_diff": worst,
            "mismatches": mismatches, "two_sided_joints": joints}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    traces = sorted(a.root.glob("**/FP_*/fp_trace.npz"))
    if not traces:
        print(f"no fp_trace.npz under {a.root}", file=sys.stderr)
        return 2
    runs = [compare_run(t, a.tol) for t in traces]
    for r in runs:
        print(f"{r['run']}: {r['status']}" + (f" max|d|={r['max_abs_diff']:.3g}" if "max_abs_diff" in r else ""))
        for m in r.get("mismatches", []):
            print(f"  {m}")
        for j in r.get("two_sided_joints", []):
            print(f"  {j['name']:15s} Fs={j['fs_nm']:.3f} bias={j['bias_nm']:+.3f} kinetic={j['kinetic_nm']:.3f}")
    if a.out:
        a.out.write_text(json.dumps({"root": str(a.root), "tol": a.tol, "runs": runs}, indent=1))
    return 0 if all(r["status"] != "mismatch" for r in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
