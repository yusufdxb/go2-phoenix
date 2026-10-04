"""Cold-start cases in MuJoCo: settled default stance -> command (deployment start state).

Same physics, deploy spec and runner as the sim2sim gate
(phoenix.sim2sim.gate_runner.run_gate_scenario, gate_v3.yaml physics), but each
case holds a zero command for SETTLE_S before its command, so what is measured
is the transition out of a still stance. 2026-09-25: walk_vnext_seed47 stood
perfectly still at forward 0.3 from the settled stance (gate v3 forward_0p3)
yet walked 0.5 and held 0.3 when coming down from 0.5.

Mirrors the Isaac cases in scripts/eval_velocity.py (_cold_start_sweep) of the
train branch. Diagnostic only: it has no verdict; thresholds live in the run's
preregistration.

    PYTHONPATH=src python scripts/sim2sim_cold_start.py --onnx P --manifest M --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

SETTLE_S = 1.0
COMMAND_S = 6.0
CASES: dict[str, list[tuple[float, tuple[float, float, float]]]] = {
    "cold_to_vx_p0.3": [(0.0, (0.3, 0.0, 0.0))],
    "cold_to_vx_m0.3": [(0.0, (-0.3, 0.0, 0.0))],
    "cold_to_vy_p0.3": [(0.0, (0.0, 0.3, 0.0))],
    "cold_to_vy_m0.3": [(0.0, (0.0, -0.3, 0.0))],
    "cold_to_yaw_p0.8": [(0.0, (0.0, 0.0, 0.8))],
    "cold_to_yaw_m0.8": [(0.0, (0.0, 0.0, -0.8))],
    "moving_0.5_to_0.3": [(0.0, (0.5, 0.0, 0.0)), (4.0, (0.3, 0.0, 0.0))],
    "cold_to_vx_p0.2": [(0.0, (0.2, 0.0, 0.0))],
    "cold_to_vx_m0.2": [(0.0, (-0.2, 0.0, 0.0))],
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gate-config", default=str(_REPO / "configs/sim2sim/gate_v3.yaml"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from phoenix.sim2sim.deploy_spec import load_deploy_spec
    from phoenix.sim2sim.gate import GateScenario, load_gate_config
    from phoenix.sim2sim.gate_runner import OnnxPolicy, RunOptions, run_gate_scenario

    cfg = load_gate_config(args.gate_config)
    spec = load_deploy_spec(args.manifest, required_envelope=None)
    policy = OnnxPolicy(args.onnx, spec.obs_dim)
    opts = RunOptions(latency_ms=float(cfg.physics["latency_ms"]))
    out: dict = {
        "onnx": args.onnx,
        "manifest": args.manifest,
        "obs_schema": spec.obs_schema,
        "manifest_problems": list(spec.manifest_problems),
        "settle_s": SETTLE_S,
        "cases": {},
    }
    for name, segs in CASES.items():
        shifted = [(0.0, (0.0, 0.0, 0.0))] + [(SETTLE_S + t, c) for t, c in segs]
        duration = SETTLE_S + segs[-1][0] + COMMAND_S
        scenario = GateScenario(name, "nominal", duration, tuple(shifted))
        m = run_gate_scenario(spec, policy, scenario, cfg, opts)["metrics"]
        cmd = segs[-1][1]
        axis = max(range(3), key=lambda i: abs(cmd[i]))
        achieved = [m.get("tracked_mean_vx"), m.get("tracked_mean_vy"), m.get("tracked_mean_wz")]
        ratio = None if achieved[axis] is None else achieved[axis] / cmd[axis]
        out["cases"][name] = {
            "command": list(cmd),
            "fell": bool(m.get("fell")),
            "achieved_mean": achieved,
            "tracking_ratio": ratio,
            "started": bool(ratio is not None and ratio >= 0.5 and not m.get("fell")),
            "lin_vel_rmse_mps": m.get("lin_vel_rmse_mps"),
            "yaw_rate_rmse_radps": m.get("yaw_rate_rmse_radps"),
        }
        r = "-" if ratio is None else f"{ratio:.2f}"
        print(f"{name:18s} fell={bool(m.get('fell'))} ratio={r} achieved={achieved}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
