#!/usr/bin/env python3
"""E2 actuator-effectiveness check (see src/phoenix/sim2real/actuator_check.py).

  actuator_check.py sim-reference --factor calf=0.8 --seconds 10 --out ref_calf_0.80.json
  actuator_check.py estimate --reference ref_nominal.json --boundary-calf ref_calf_0.80.json \
      --boundary-all ref_all_0.75.json --sweep ref_calf_*.json --bridge F.jsonl G.jsonl --out e2.json
  (same with --sim-as-hw ref_X.json in place of --bridge)

Primary verdict: hardware calf err vs simulated boundary plants (calf x0.80, all x0.75).
The ratio g estimates are uncalibrated and reported only.
Run sim-reference with the MuJoCo interpreter (``~/Sim/isaac-sim-venv/bin/python``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from phoenix.sim2real import actuator_check as ac  # noqa: E402

DEFAULT_DEPLOY = _REPO / "configs/sim2real/deploy_walk_iter2750_stand.yaml"
GATE_CFG = _REPO / "configs/sim2sim/gate_v3.yaml"


def cmd_sim_reference(args: argparse.Namespace) -> int:
    import numpy as np
    import yaml
    from sim2sim_robustness import pattern_factors

    from phoenix.sim2sim.deploy_spec import load_deploy_spec
    from phoenix.sim2sim.gate import GateScenario, load_gate_config
    from phoenix.sim2sim.gate_runner import JOINT_ORDER, OnnxPolicy, RunOptions, run_gate_scenario

    deploy = yaml.safe_load(Path(args.deploy_config).read_text())["policy"]
    cfg = load_gate_config(GATE_CFG)
    spec = load_deploy_spec(deploy["manifest_path"], required_envelope=cfg.required_envelope)
    fk = np.asarray(
        ac.apply_factors(
            pattern_factors(args.pattern, spec.joint_order).tolist(), spec.joint_order, args.factor
        )
    )
    kp_cmd = dict(zip(spec.joint_order, spec.kp.tolist(), strict=True))
    kd_cmd = dict(zip(spec.joint_order, spec.kd.tolist(), strict=True))
    spec = dataclasses.replace(spec, kp=spec.kp * fk, kd=spec.kd * fk)
    policy = OnnxPolicy(deploy["onnx_path"], spec.obs_dim)
    scen = GateScenario(
        "e2_stand_zero_cmd", "nominal", float(args.seconds), ((0.0, (0.0, 0.0, 0.0)),)
    )
    trace: list = []
    opts = RunOptions(latency_ms=float(cfg.physics["latency_ms"]), trace=trace)
    res = run_gate_scenario(spec, policy, scen, cfg, opts)
    fell = bool(res["metrics"].get("fell"))
    ref = ac.summarize_trace(
        [r["t"] for r in trace],
        np.asarray([r["q"] for r in trace]),
        np.asarray([r["target"] for r in trace]),
        JOINT_ORDER,
        kp_cmd,
        kd_cmd,
    )
    g_true = dict(zip(spec.joint_order, [float(x) for x in fk], strict=True))
    ref.update(
        pattern=args.pattern,
        factor_specs=list(args.factor),
        fell=fell,
        seconds=float(args.seconds),
        g_true={n: g_true[n] for n in ac.POLICY_JOINT_ORDER},
        deploy_config=str(args.deploy_config),
        onnx=deploy["onnx_path"],
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(ref, indent=2))
    print(
        f"{args.pattern} {args.factor}: fell={fell} n={ref['n']} "
        f"kp(commanded)={sorted(set(kp_cmd.values()))} kd={sorted(set(kd_cmd.values()))} "
        f"calf_err={ac.group_mean_err(ref, 'calf'):.4f}"
    )
    return 0


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def cmd_estimate(args: argparse.Namespace) -> int:
    ref = _load(args.reference)
    hw = _load(args.sim_as_hw) if args.sim_as_hw else ac.summarize_hw(ac.load_ticks(args.bridge))
    decision = ac.decide(hw, _load(args.boundary_calf), _load(args.boundary_all), ref)
    calibrated: dict = {"sweep": None}
    if args.sweep:
        pts = ac.sweep_points([_load(p) for p in args.sweep])
        g, status = ac.invert_sweep(pts, decision["calf_err_hw"])
        calibrated = {"sweep": pts, "g_calf_est": g, "status": status}
    ratio = ac.estimate_effectiveness(ref, hw, min_ref_err=args.min_ref_err)
    out = {
        "verdict": decision["verdict"],
        "decision": decision,
        "calibrated": calibrated,
        "ratio_uncalibrated": ratio,
        "g_true": hw.get("g_true"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    d = decision
    print(
        f"calf_err_hw {d['calf_err_hw']:.4f} | calf boundary (calf=0.80) {d['calf_boundary_err']:.4f}"
        f" -> {'GO' if d['calf_test_go'] else 'STOP'} | global boundary (all=0.75) "
        f"{d['global_boundary_err']:.4f} -> {'GO' if d['global_test_go'] else 'STOP'}"
    )
    t = d["thigh_informational"]
    print(
        f"thigh err hw {t['thigh_err_hw']:.4f} vs boundary refs "
        f"{t['thigh_err_calf_boundary_ref']:.4f} / {t['thigh_err_all_boundary_ref']:.4f} (informational)"
    )
    if args.sweep:
        print(f"g_calf_est {calibrated['g_calf_est']:.3f} ({calibrated['status']})")
    gt = hw.get("g_true")
    if gt:
        calves = sorted({gt[n] for n in ac.POLICY_JOINT_ORDER if ac.joint_group(n) == "calf"})
        print(f"g_true calves {calves}")
    print(
        f"ratio (uncalibrated, reported only): calf mean {ratio['calf_mean']} "
        f"global mean {ratio['global_mean']} ratio-verdict {ratio['ratio_verdict_uncalibrated']}"
    )
    print(f"verdict: {out['verdict']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim-reference")
    s.add_argument("--pattern", default="nominal")
    s.add_argument(
        "--factor",
        action="append",
        default=[],
        metavar="GROUP=VALUE",
        help="repeatable; GROUP in calf/thigh/hip/all; applied after --pattern, later wins",
    )
    s.add_argument("--seconds", type=float, default=10.0)
    s.add_argument("--deploy-config", default=str(DEFAULT_DEPLOY))
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_sim_reference)
    e = sub.add_parser("estimate")
    e.add_argument("--reference", required=True, help="nominal reference")
    e.add_argument("--boundary-calf", required=True, help="reference with calf=0.80")
    e.add_argument("--boundary-all", required=True, help="reference with all=0.75")
    e.add_argument("--sweep", nargs="+", help="calf-only references for the calibrated estimate")
    grp = e.add_mutually_exclusive_group(required=True)
    grp.add_argument("--bridge", nargs="+")
    grp.add_argument("--sim-as-hw")
    e.add_argument("--min-ref-err", type=float, default=ac.MIN_REF_ERR_RAD)
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_estimate)
    args = ap.parse_args(argv)
    for spec_s in getattr(args, "factor", []):
        ac.parse_factor(spec_s)  # fail early on a malformed --factor
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
