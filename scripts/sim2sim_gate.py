#!/usr/bin/env python3
"""Phoenix sim2sim gate: one command, one JSON verdict, one MP4 per scenario.

    PYTHONPATH=src python scripts/sim2sim_gate.py --onnx <policy.onnx> --manifest <manifest.json>

See docs/sim2sim_gate.md. Exit code: 0 PASS, 1 FAIL, 3 DIAGNOSTIC (a gate
setting was overridden, so no verdict is issued), 2 usage/setup error.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

_MESA_EGL = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"


def _configure_headless_gl() -> None:
    """CPU (Mesa llvmpipe) EGL by default so video never touches the GPU."""
    if "MUJOCO_GL" in os.environ:
        return
    os.environ["MUJOCO_GL"] = "egl"
    if Path(_MESA_EGL).is_file():
        os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _MESA_EGL)
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")


def _fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "YES" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--onnx", required=True, help="exported actor ONNX (obs -> action)")
    ap.add_argument(
        "--manifest",
        required=True,
        help="phoenix_manifest.json, the legacy H25 manifest, or a sim2sim deploy spec",
    )
    ap.add_argument(
        "--out", default=None, help="output directory (default runs/sim2sim_gate/<name>_<utc>)"
    )
    ap.add_argument(
        "--gate",
        choices=("v1", "v2", "v3"),
        default="v3",
        help="gate version (default v3: v2 plus blocking responsiveness; see docs/sim2sim_gate.md)",
    )
    ap.add_argument("--gate-config", default=None, help="explicit gate yaml; overrides --gate")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument(
        "--scenarios", nargs="*", default=None, help="subset (makes the run DIAGNOSTIC)"
    )
    ap.add_argument(
        "--latency-ms", type=float, default=None, help="override (makes the run DIAGNOSTIC)"
    )
    ap.add_argument(
        "--action-clip",
        default=None,
        help="override the deploy clip, a float or 'none' (makes the run DIAGNOSTIC)",
    )
    args = ap.parse_args(argv)

    _configure_headless_gl()
    from phoenix.sim2sim.deploy_spec import DeploySpecError, load_deploy_spec
    from phoenix.sim2sim.gate import load_gate_config
    from phoenix.sim2sim.gate_runner import OnnxPolicy, run_gate, sha256_file, write_gate_report

    cfg = load_gate_config(args.gate_config or args.gate)
    try:
        spec = load_deploy_spec(args.manifest, required_envelope=cfg.required_envelope)
    except (DeploySpecError, KeyError, ValueError) as exc:
        print(f"ERROR: manifest {args.manifest}: {exc}", file=sys.stderr)
        return 2
    diag: list[str] = []
    if args.action_clip is not None:
        clip = None if args.action_clip.lower() == "none" else float(args.action_clip)
        diag.append(f"action_clip overridden {spec.action_clip} -> {clip}")
        spec = spec.with_action_clip(clip)
    try:
        policy = OnnxPolicy(args.onnx, spec.obs_dim)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else _REPO / "runs" / "sim2sim_gate" / f"{spec.name}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"gate {cfg.path}\nspec {spec.name} ({spec.kind}, obs {spec.obs_dim}-D, clip {spec.action_clip})\n"
        f"out  {out}"
    )

    blocking = "SAFETY" if cfg.version >= 2 else "GATE"

    def progress(name, r):
        m = r["metrics"]
        perf = "" if cfg.version < 2 else f" perf={'PASS' if r['performance_pass'] else 'FAIL'}"
        print(
            f"  {name:14s} {blocking} {'PASS' if r['pass'] else 'FAIL'}{perf}  fell={_fmt(m['fell'])} "
            f"{m['fall_reason'] or ''} lin_rmse={_fmt(m['lin_vel_rmse_mps'])} "
            f"yaw_rmse={_fmt(m['yaw_rate_rmse_radps'])} h={_fmt(m['mean_base_height_m'])} "
            f"preclip={_fmt(m['pre_clip_saturation_rate'])} "
            f"tq_sat={ {g: round(v, 3) for g, v in m['torque_saturation_fraction'].items() if v is not None} }",
            flush=True,
        )

    report = run_gate(
        spec,
        policy,
        cfg,
        policy_info={
            "onnx": str(Path(args.onnx).resolve()),
            "onnx_sha256": sha256_file(args.onnx),
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": sha256_file(args.manifest),
        },
        latency_ms=args.latency_ms,
        video_dir=None if args.no_video else out / "videos",
        scenario_names=args.scenarios,
        diagnostic_reasons=diag,
        progress=progress,
    )
    path = write_gate_report(report, out / "gate_report.json")
    vids = [r.get("video_error") for r in report["scenarios"].values() if r.get("video_error")]
    if vids:
        print(f"video skipped: {vids[0]}")
    print(
        f"\n{report['gate']['name']}: VERDICT {report['verdict']} "
        f"({blocking} would-be {report['would_be_verdict']})"
    )
    for f in report["failures"][:40]:
        print(f"  - {f}")
    if cfg.version >= 2:
        perf = report["tiers"]["performance"]
        print(f"PERFORMANCE (non-blocking): {perf['verdict']}")
        for label, row in perf["named"].items():
            if row:
                print(
                    f"  {label:16s} {row['metric']} {_fmt(row['value'])} <= {row['threshold']} "
                    f"{'PASS' if row['pass'] else 'FAIL'}"
                )
        for f in report["performance_failures"]:
            print(f"  - {f}")
    print(f"report {path}")
    return {"PASS": 0, "FAIL": 1}.get(report["verdict"], 3)


if __name__ == "__main__":
    sys.exit(main())
