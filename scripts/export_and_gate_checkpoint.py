"""Export ONNX + build manifest + run the sim2sim gate for ONE arbitrary checkpoint.

Unlike postrun_velocity.py (which always uses a run's latest.pt), this takes any
checkpoint path directly, for ranking/selecting a pre-collapse checkpoint that is
not the run's final one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SIM2SIM_WORKTREE = Path("/home/yusuf/workspace/go2-phoenix/.claude/worktrees/agent-a6f2a9725e191de1f")
SIM2SIM_PYTHON = Path("/home/yusuf/Projects/go2-positive-control/.venv/bin/python")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha(repo_root: Path) -> tuple[str, bool]:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True).stdout.strip())
    return sha, dirty


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, required=True, help="checkpoints/velocity/<run_name> (for spec.json)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--python", type=Path, default=Path("/home/yusuf/Sim/isaac-sim-venv/bin/python"))
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from phoenix.velocity.contract import MANIFEST_NAME, build_manifest
    from phoenix.velocity.env_cfg import ACTUATOR_DAMPING, ACTUATOR_STIFFNESS
    from phoenix.velocity.spec import default_spec, spec_from_dict

    args.out_dir.mkdir(parents=True, exist_ok=True)

    specs = sorted(args.run_root.glob("*/spec.json"))
    spec = spec_from_dict(json.loads(specs[0].read_text())) if specs else default_spec()

    git_sha, git_dirty = _git_sha(REPO_ROOT)
    ckpt_sha = _sha256(args.checkpoint.resolve())

    # Match postrun_velocity.py's manifest shape exactly (gate_v2's manifest
    # validator rejects extra fields, verified empirically: adding
    # "widest_ranges_reached_at_end_of_run"/"note" to curriculum, and a
    # top-level "extra" block, made every one of these 6 gate runs fail on
    # "spec:manifest_valid" even though phoenix's OWN manifest validation
    # (locomotion_capable) passed cleanly and the commands field was byte
    # identical to postrun's. This checkpoint-specific caveat (that
    # curriculum_final_state.json reflects the run's END state, not
    # necessarily this earlier checkpoint) is recorded in this script's
    # docstring and the accompanying report instead.
    cur_state_path = args.run_root / "curriculum_final_state.json"
    curriculum_ranges = spec.commands.final_ranges()
    cur_dict = {"num_levels": spec.curriculum.num_levels, "enabled": spec.curriculum.enabled}
    if cur_state_path.exists():
        from phoenix.velocity.contract import CommandRanges

        cur_full = json.loads(cur_state_path.read_text())
        curriculum_ranges = CommandRanges.from_mapping(cur_full["widest_ranges_reached"])

    manifest = build_manifest(
        checkpoint_sha256=ckpt_sha,
        commands=curriculum_ranges,
        git_sha=git_sha,
        git_dirty=git_dirty,
        seed=int(spec.seed),
        task="phoenix-velocity-flat-go2",
        simulator="isaac-lab-physx",
        reward_scales=spec.reward_scales(),
        domain_randomization=json.loads(json.dumps(spec.domain_randomization.__dict__, default=str)),
        curriculum=cur_dict,
        obs_schema_version=spec.obs_schema,
        deploy={"kp": ACTUATOR_STIFFNESS, "kd": ACTUATOR_DAMPING, "action_clip": spec.action.clip_actions},
    )
    manifest_path = args.out_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2))

    result: dict = {"checkpoint": str(args.checkpoint), "manifest": str(manifest_path)}

    onnx_path = args.out_dir / "policy.onnx"
    export_cmd = [
        str(args.python), "-m", "phoenix.sim2real.export",
        "--checkpoint", str(args.checkpoint.resolve()),
        "--output", str(onnx_path),
        "--verify",
    ]
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + ":" + env.get("PYTHONPATH", "")
    r = subprocess.run(export_cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300)
    result["export_returncode"] = r.returncode
    result["export_stdout_tail"] = r.stdout[-2000:]
    result["export_stderr_tail"] = r.stderr[-2000:]
    if r.returncode == 0:
        result["onnx"] = str(onnx_path)

        gate_out = args.out_dir / "sim2sim_gate"
        gate_cmd = [
            str(SIM2SIM_PYTHON), str(SIM2SIM_WORKTREE / "scripts" / "sim2sim_gate.py"),
            "--onnx", str(onnx_path.resolve()),
            "--manifest", str(manifest_path.resolve()),
            "--out", str(gate_out),
            "--no-video",
        ]
        gate_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        r2 = subprocess.run(gate_cmd, cwd=SIM2SIM_WORKTREE, env=gate_env, capture_output=True, text=True, timeout=900)
        result["gate_returncode"] = r2.returncode
        result["gate_verdict"] = "PASS" if r2.returncode == 0 else "FAIL" if r2.returncode == 1 else "DIAGNOSTIC_OR_ERROR"
        result["gate_stdout_tail"] = r2.stdout[-3000:]
        result["gate_stderr_tail"] = r2.stderr[-1500:]
        report = gate_out / "gate_report.json"
        if report.exists():
            result["gate_report"] = json.loads(report.read_text())

    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("_tail")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
