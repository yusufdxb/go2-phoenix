#!/usr/bin/env python3
"""Phoenix loop CLI: RUN -> DETECT -> CONDITION -> TRAIN -> VERIFY -> REDEPLOY, offline parts.

Every subcommand reads files and writes one JSON (or YAML) artifact; nothing here
talks to the robot or starts a simulator.

  fidelity   bridge.jsonl                          -> deploy-fidelity report (PASS/FAIL)
  calibrate  nominal bridge.jsonl files            -> baseline.json for one command regime
  assess     bridge.jsonl + baseline.json          -> health report (table + JSON)
  condition  health.json + parent env              -> targeted env overlay YAML
  gate       candidate / incumbent eval JSONs      -> PROMOTE / REJECT decision
  promote    decision.json + deploy lock           -> refuse unless PROMOTE for that checkpoint

Run with the system python and PYTHONPATH=src (see docs/runbooks/TRAINING.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from phoenix.condition.distribution import ConditionError, condition
from phoenix.monitor.fidelity import fidelity_report
from phoenix.monitor.health import HealthMonitor, JointHealth, PersistenceConfig, format_report
from phoenix.monitor.layers import read_bridge_telemetry, tracking_pairs
from phoenix.monitor.residual import (
    Baseline,
    WindowConfig,
    authority_ratio,
    calibrate,
    window_stats,
)
from phoenix.validate.candidate_gate import candidate_gate
from phoenix.validate.promotion import promotion_problems


def _write(path: Path | None, obj) -> None:
    text = json.dumps(obj, indent=2, default=float)
    if path is None:
        print(text)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
        print(f"wrote {path}", file=sys.stderr)


def cmd_fidelity(a: argparse.Namespace) -> int:
    rep = fidelity_report(read_bridge_telemetry(a.telemetry))
    rep["telemetry"] = str(a.telemetry)
    _write(a.out, rep)
    return 0 if rep["verdict"] == "PASS" else 1


def cmd_calibrate(a: argparse.Namespace) -> int:
    cfg = WindowConfig()
    stats = []
    for path in a.telemetry:
        fid = fidelity_report(read_bridge_telemetry(path))
        if fid["verdict"] != "PASS" and not a.allow_low_fidelity:
            print(f"REFUSING {path}: deploy fidelity {fid['reasons']}", file=sys.stderr)
            return 2
        stats.append(window_stats(tracking_pairs(read_bridge_telemetry(path)), cfg))
    try:
        base = calibrate(stats, regime=a.regime, cfg=cfg, source=[str(p) for p in a.telemetry])
    except ValueError as exc:
        print(f"REFUSING calibration: {exc}", file=sys.stderr)
        return 2
    _write(a.out, base.to_dict())
    return 0


def cmd_assess(a: argparse.Namespace) -> int:
    base = Baseline.from_dict(json.loads(a.baseline.read_text()))
    if base.regime != a.regime:
        print(
            f"REFUSING: baseline regime {base.regime!r} != run regime {a.regime!r}", file=sys.stderr
        )
        return 2
    layers = read_bridge_telemetry(a.telemetry)
    fid = fidelity_report(layers)
    cfg = WindowConfig(**base.window) if base.window else WindowConfig()
    st = window_stats(tracking_pairs(layers), cfg)
    s = authority_ratio(st, base, cfg)
    mon = HealthMonitor(base, PersistenceConfig())
    report: list[JointHealth] = mon.report()
    for i in range(st.n_windows):
        report = mon.update(s[i], st.torque_gain_ratio[i])
    print(format_report(report), file=sys.stderr)
    _write(
        a.out,
        {
            "schema": "phoenix-health/v1",
            "telemetry": str(a.telemetry),
            "regime": a.regime,
            "windows": st.n_windows,
            "fidelity": fid,
            "health": [h.to_dict() for h in report],
        },
    )
    return 0


def cmd_condition(a: argparse.Namespace) -> int:
    doc = json.loads(a.health.read_text())
    if doc.get("fidelity", {}).get("verdict") != "PASS":
        print("REFUSING: the assessed run failed the deploy-fidelity gate", file=sys.stderr)
        return 2
    report = [JointHealth(**h) for h in doc["health"]]
    base = json.loads(a.baseline.read_text()) if a.baseline else None
    try:
        res = condition(
            report,
            parent_env=a.parent_env,
            telemetry_paths=[doc["telemetry"]],
            baseline=base,
            extra={"health_file": str(a.health)},
        )
    except ConditionError as exc:
        print(f"NO TARGETED DISTRIBUTION: {exc}", file=sys.stderr)
        return 3
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(yaml.safe_dump(res.overlay(), sort_keys=False))
    print(f"wrote {a.out}: {res.spec.to_yaml_block()}", file=sys.stderr)
    return 0


def _episodes(path: Path) -> tuple[dict[str, list[float]], dict[str, str]]:
    """``{"checkpoint_sha256": ..., "conditions": {cond: {"episodes": [values]}}}``."""
    doc = json.loads(path.read_text())
    vals = {c: list(map(float, v["episodes"])) for c, v in doc["conditions"].items()}
    shas = {
        c: v.get("checkpoint_sha256", doc.get("checkpoint_sha256"))
        for c, v in doc["conditions"].items()
    }
    return vals, shas


def cmd_gate(a: argparse.Namespace) -> int:
    cand, cand_sha = _episodes(a.candidate)
    inc, _ = _episodes(a.incumbent)
    parity = json.loads(a.parity.read_text()) if a.parity else None
    dec = candidate_gate(cand, inc, a.candidate_sha256, cand_sha, parity)
    _write(a.out, dec)
    return 0 if dec["decision"] == "PROMOTE" else 1


def cmd_promote(a: argparse.Namespace) -> int:
    problems = promotion_problems(
        json.loads(a.decision.read_text()), yaml.safe_load(a.lock.read_text())
    )
    for p in problems:
        print(f"REFUSING: {p}", file=sys.stderr)
    return 0 if not problems else 2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fidelity")
    s.add_argument("telemetry", type=Path)
    s.add_argument("--out", type=Path)
    s.set_defaults(fn=cmd_fidelity)

    s = sub.add_parser("calibrate")
    s.add_argument("telemetry", type=Path, nargs="+")
    s.add_argument("--regime", required=True, help="command regime label, e.g. stand or walk_0.5")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--allow-low-fidelity", action="store_true", help="development only")
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("assess")
    s.add_argument("telemetry", type=Path)
    s.add_argument("--baseline", type=Path, required=True)
    s.add_argument("--regime", required=True)
    s.add_argument("--out", type=Path)
    s.set_defaults(fn=cmd_assess)

    s = sub.add_parser("condition")
    s.add_argument("health", type=Path)
    s.add_argument("--parent-env", required=True, help="defaults entry relative to --out's dir")
    s.add_argument("--baseline", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(fn=cmd_condition)

    s = sub.add_parser("gate")
    s.add_argument("--candidate", type=Path, required=True)
    s.add_argument("--incumbent", type=Path, required=True)
    s.add_argument("--candidate-sha256", required=True)
    s.add_argument("--parity", type=Path)
    s.add_argument("--out", type=Path)
    s.set_defaults(fn=cmd_gate)

    s = sub.add_parser("promote")
    s.add_argument("--decision", type=Path, required=True)
    s.add_argument("--lock", type=Path, required=True)
    s.set_defaults(fn=cmd_promote)

    a = p.parse_args(argv)
    np.seterr(all="ignore")
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())
