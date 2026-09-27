"""Gate v3 + cold-start cases under perturbed actuator PD gains (MuJoCo, diagnostic).

The deploy spec's per-joint kp / kd (the DC-motor PD gains, the same quantity
Isaac's motor-strength randomization scales) are multiplied by a per-joint
factor before running the unchanged gate v3 scenarios and the
sim2sim_cold_start.py cases:

  --pattern NAME     fixed factors (deterministic): nominal, weak_rear (RL/RR x0.75),
                     weak_calf (calves x0.75), lr_asym (left x0.85, right x1.15), all_0.7
  --range LO HI --draw K   per-joint factors ~ U(LO, HI), kp and kd drawn independently,
                     numpy seed K (reproducible)

The gate's own thresholds are unchanged; the report states the perturbation.

    PYTHONPATH=src python scripts/sim2sim_robustness.py --onnx P --manifest M --pattern weak_rear --out o.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))


def pattern_factors(name: str, joint_order) -> np.ndarray:
    f = np.ones(len(joint_order))
    for i, j in enumerate(joint_order):
        leg = j[:2]
        if name == "nominal":
            pass
        elif name == "weak_rear" and leg in ("RL", "RR"):
            f[i] = 0.75
        elif name == "weak_calf" and j.endswith("calf_joint"):
            f[i] = 0.75
        elif name == "lr_asym":
            f[i] = 0.85 if leg in ("FL", "RL") else 1.15
        elif name == "all_0.7":
            f[i] = 0.7
        elif name not in ("weak_rear", "weak_calf"):
            raise SystemExit(f"unknown pattern {name}")
    return f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--range", nargs=2, type=float, default=None)
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from sim2sim_cold_start import CASES, COMMAND_S, SETTLE_S

    from phoenix.sim2sim.deploy_spec import load_deploy_spec
    from phoenix.sim2sim.gate import GateScenario, load_gate_config
    from phoenix.sim2sim.gate_runner import OnnxPolicy, RunOptions, run_gate, run_gate_scenario

    cfg = load_gate_config(_REPO / "configs/sim2sim/gate_v3.yaml")
    spec = load_deploy_spec(args.manifest, required_envelope=cfg.required_envelope)
    if args.range is not None:
        rng = np.random.default_rng(args.draw)
        fk = rng.uniform(args.range[0], args.range[1], size=len(spec.joint_order))
        fd = rng.uniform(args.range[0], args.range[1], size=len(spec.joint_order))
        cond = f"range_{args.range[0]:g}_{args.range[1]:g}_draw{args.draw}"
    else:
        fk = fd = pattern_factors(args.pattern or "nominal", spec.joint_order)
        cond = f"pattern_{args.pattern or 'nominal'}"
    spec = dataclasses.replace(spec, kp=spec.kp * fk, kd=spec.kd * fd)
    policy = OnnxPolicy(args.onnx, spec.obs_dim)

    report = run_gate(spec, policy, cfg, policy_info={"onnx": args.onnx, "perturbation": cond})
    scen = {}
    for name, r in report["scenarios"].items():
        m = r["metrics"]
        scen[name] = {
            "safety_pass": r["pass"],
            "fell": m.get("fell"),
            "lin_vel_rmse_mps": m.get("lin_vel_rmse_mps"),
            "yaw_rate_rmse_radps": m.get("yaw_rate_rmse_radps"),
            "hard_limit_violation_steps": m.get("hard_limit_violation_steps"),
            "near_limit_fraction": m.get("near_limit_fraction"),
            "torque_saturation_fraction": m.get("torque_saturation_fraction"),
            "failed_safety_checks": [c["check"] for c in r["checks"] if c["tier"] == "safety" and not c["pass"]],
        }
    opts = RunOptions(latency_ms=float(cfg.physics["latency_ms"]))
    cold = {}
    for name, segs in CASES.items():
        shifted = [(0.0, (0.0, 0.0, 0.0))] + [(SETTLE_S + t, c) for t, c in segs]
        sc = GateScenario(name, "nominal", SETTLE_S + segs[-1][0] + COMMAND_S, tuple(shifted))
        m = run_gate_scenario(spec, policy, sc, cfg, opts)["metrics"]
        cmd = segs[-1][1]
        axis = max(range(3), key=lambda i: abs(cmd[i]))
        ach = [m.get("tracked_mean_vx"), m.get("tracked_mean_vy"), m.get("tracked_mean_wz")]
        ratio = None if ach[axis] is None else ach[axis] / cmd[axis]
        cold[name] = {"tracking_ratio": ratio, "fell": bool(m.get("fell")),
                      "started": bool(ratio is not None and ratio >= 0.5 and not m.get("fell"))}
    out = {
        "condition": cond,
        "kp_factors": fk.tolist(),
        "kd_factors": fd.tolist(),
        "joint_order": list(spec.joint_order),
        "gate_verdict": report["verdict"],
        "gate_would_be_verdict": report["would_be_verdict"],
        "gate_failures": report["failures"],
        "gate_performance_failures": report.get("performance_failures", []),
        "scenarios": scen,
        "cold": cold,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"{cond}: gate {report['verdict']} fails={report['failures']} cold=" +
          " ".join(f"{k.replace('cold_to_', '')}={(v['tracking_ratio'] or 0):.2f}" for k, v in cold.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
