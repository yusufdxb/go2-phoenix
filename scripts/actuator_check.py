#!/usr/bin/env python3
"""E2 actuator-effectiveness check (see src/phoenix/sim2real/actuator_check.py).

  actuator_check.py sim-reference --factor calf=0.8 --seconds 10 --out ref_calf_0.80.json
  actuator_check.py estimate --reference ref_nominal.json --boundary-calf ref_calf_0.80.json \
      --boundary-all ref_all_0.75.json --sweep ref_calf_*.json --session logs/hw_sessions/<date>_<sha> --out e2.json
  (same with --sim-as-hw ref_X.json in place of --bridge)

The check refuses (exit 2, verdict REFUSED, never GO) on mismatched artifact hashes,
wrong boundary plants, gain mismatch, rehearsal telemetry or a low-quality window.

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
DEFAULT_LOCK = _REPO / "configs/sim2real/locks/deploy_walk_iter2750_stand.lock.yaml"
GATE_CFG = _REPO / "configs/sim2sim/gate_v3.yaml"


def cmd_sim_reference(args: argparse.Namespace) -> int:
    import numpy as np
    import yaml
    from sim2sim_robustness import pattern_factors

    from phoenix.sim2real import deploy_contract as dc
    from phoenix.sim2sim.deploy_spec import load_deploy_spec
    from phoenix.sim2sim.gate import GateScenario, load_gate_config
    from phoenix.sim2sim.gate_runner import JOINT_ORDER, OnnxPolicy, RunOptions, run_gate_scenario

    deploy_cfg = yaml.safe_load(Path(args.deploy_config).read_text())
    lock = dc.load_lock(args.lock)
    problems, observed = dc.verify_lock(
        lock, deploy_cfg, args.deploy_config, required_roles=ac.ARTIFACT_ROLES
    )
    if problems:
        print("REFUSED: artifacts do not match the lock, not writing a reference:")
        for prob in problems:
            print(f"  - {prob}")
        return 2
    identity = {role: observed["artifacts"][role]["sha256"] for role in ac.ARTIFACT_ROLES}
    identity["config_semantic_sha256"] = observed["deploy_config"]["semantic_sha256"]
    deploy = deploy_cfg["policy"]
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
        identity=identity,
        lock={"path": str(Path(args.lock).name), "name": lock.get("name")},
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
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ref = _load(args.reference)
        if args.sim_as_hw:
            hw = _load(args.sim_as_hw)
            if hw.get("source") != "sim" or not hw.get("g_true"):
                raise ac.Refusal(
                    f"{args.sim_as_hw}: --sim-as-hw takes a simulator reference "
                    "(source 'sim' with g_true), not a hardware summary"
                )
            for untrusted in ("allow_rehearsal", "live", "ledger"):
                hw.pop(untrusted, None)
            mode = "sim-as-hw"
        elif args.session:
            files, ledger = ac.ledger_telemetry(args.session, ac.ref_identity(ref, "nominal"))
            hw = ac.summarize_telemetry(
                ac.load_telemetry(files), ac.ref_identity(ref, "nominal"), allow_rehearsal=False
            )
            hw["ledger"] = ledger
            mode = "session"
        else:
            if not args.allow_rehearsal:
                raise ac.Refusal(
                    "--bridge files are not bound to a hardware ledger record; use --session "
                    "(--bridge is for --allow-rehearsal desk checks only)"
                )
            hw = ac.summarize_telemetry(
                ac.load_telemetry(args.bridge),
                ac.ref_identity(ref, "nominal"),
                allow_rehearsal=True,
            )
            mode = "bridge-desk"
        b_calf, b_all = _load(args.boundary_calf), _load(args.boundary_all)
        decision = ac.decide(hw, b_calf, b_all, ref)
        sweep_refs = [_load(p) for p in (args.sweep or [])]
        for p, r in zip(args.sweep or [], sweep_refs, strict=True):
            ac.check_ref_consistent(p, r, ref)
        calibrated: dict = {"sweep": None}
        if sweep_refs:
            pts = ac.sweep_points(sweep_refs)
            g, status = ac.invert_sweep(pts, decision["calf_err_hw"])
            calibrated = {"sweep": pts, "g_calf_est": g, "status": status}
        ratio = ac.estimate_effectiveness(ref, hw, min_ref_err=args.min_ref_err)
    except ac.Refusal as exc:
        out_path.write_text(json.dumps({"verdict": "REFUSED", "reason": str(exc)}, indent=2))
        print(f"REFUSED: {exc}")
        print("verdict: REFUSED (no GO/STOP issued)")
        return 2
    hardware_evidence = mode == "session"
    out = {
        "verdict": decision["verdict"],
        "input_mode": mode,
        "valid_for_walk_decision": hardware_evidence,
        "ledger": hw.get("ledger"),
        "decision": decision,
        "calibrated": calibrated,
        "ratio_uncalibrated": ratio,
        "g_true": hw.get("g_true"),
        "hardware": {k: hw[k] for k in ("telemetry", "window_gates", "live") if k in hw},
    }
    d = decision
    for info in hw.get("telemetry", []):
        print(
            f"telemetry {info['path']}: stage {info['stage']} live {info['live']} "
            f"commit {str(info['commit'])[:10]} dirty {info['dirty']}"
        )
    for g in hw.get("window_gates", []):
        print(
            f"window {g['label']}: {g['window_s']:.2f} s ({g['n_window']} ticks) dq_rms_max "
            f"{g['dq_rms_max_rad_s']:.4f} tilt_max {g['tilt_max_deg']:.2f} gaps<={g['max_tick_gap_s']:.3f}"
        )
    print(
        f"calf_err_hw {d['calf_err_hw']:.4f} | calf boundary ({d['calf_boundary_label']}) "
        f"{d['calf_boundary_err']:.4f} -> {'GO' if d['calf_test_go'] else 'STOP'} | global boundary "
        f"({d['global_boundary_label']}) {d['global_boundary_err']:.4f} -> "
        f"{'GO' if d['global_test_go'] else 'STOP'}"
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
    if hardware_evidence:
        print(f"verdict: {out['verdict']}  [input: hardware ledger session]")
    else:
        print(
            f"verdict: {out['verdict']}  [input: {mode}; NOT hardware evidence, not a walk decision]"
        )
    # Written last: a crash while printing must never leave a verdict file behind.
    out_path.write_text(json.dumps(out, indent=2))
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
    s.add_argument("--lock", default=str(DEFAULT_LOCK), help="deploy lock the artifacts must match")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_sim_reference)
    e = sub.add_parser("estimate")
    e.add_argument("--reference", required=True, help="nominal reference")
    e.add_argument("--boundary-calf", required=True, help="reference with calf=0.80")
    e.add_argument("--boundary-all", required=True, help="reference with all=0.75")
    e.add_argument("--sweep", nargs="+", help="calf-only references for the calibrated estimate")
    grp = e.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--session", help="hardware session dir; telemetry comes from its stage G and H records"
    )
    grp.add_argument("--bridge", nargs="+", help="desk/tests only, requires --allow-rehearsal")
    grp.add_argument("--sim-as-hw")
    e.add_argument(
        "--allow-rehearsal", action="store_true", help="accept non-live telemetry (tests only)"
    )
    e.add_argument("--min-ref-err", type=float, default=ac.MIN_REF_ERR_RAD)
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_estimate)
    args = ap.parse_args(argv)
    for spec_s in getattr(args, "factor", []):
        ac.parse_factor(spec_s)  # fail early on a malformed --factor
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
