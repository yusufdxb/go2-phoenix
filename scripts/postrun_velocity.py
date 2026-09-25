"""After all PhoenixVelocity training runs finish: manifest + ONNX export + eval, per seed.

Meant to be chained after the tmux training sequence (see scripts/run_velocity_seeds.sh):
for each seed's log_root/latest.pt it
  1. writes phoenix_manifest.json (phoenix.velocity.contract.build_manifest), including
     the "deploy" block {"kp","kd","action_clip"} read from the RUN'S OWN spec.json
     (not a fresh default_spec()), so a checkpoint trained with a non-default
     ActionSpec.clip_actions gets its own value recorded, not the current default.
  2. exports ONNX + verifies parity (phoenix.sim2real.export, deploy-owned, called not edited)
  3. runs scripts/eval_velocity.py for held-out metrics with DR on
  4. runs the sim2sim gate (feat/phoenix-walk-v1-sim2sim worktree, CPU, --no-video)
     against the exported ONNX + manifest, records the verdict
  5. aggregates everything into a summary JSON

Runs OUTSIDE Isaac Sim for steps 1-2 (pure torch/hashing), and re-launches an
Isaac Sim subprocess per seed for step 3 (eval needs the live env). Step 4 uses
a completely separate venv/worktree (MuJoCo-based, no Isaac Sim).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("phoenix.velocity.postrun")

#: The sim2sim gate lives in a sibling worktree on a different branch (not ours
#: to edit): feat/phoenix-walk-v1-sim2sim.
SIM2SIM_WORKTREE = Path("/home/yusuf/workspace/go2-phoenix/.claude/worktrees/agent-a6f2a9725e191de1f")
SIM2SIM_PYTHON = Path("/home/yusuf/Projects/go2-positive-control/.venv/bin/python")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-run export + eval chain for PhoenixVelocity seeds.")
    p.add_argument("--run-roots", type=Path, nargs="+", required=True, help="checkpoints/velocity/<run_name> dirs")
    p.add_argument("--python", type=Path, default=Path("/home/yusuf/Sim/isaac-sim-venv/bin/python"))
    p.add_argument("--eval-num-envs", type=int, default=256)
    p.add_argument("--eval-num-steps", type=int, default=1000)
    p.add_argument("--skip-gate", action="store_true", help="skip the sim2sim gate step")
    p.add_argument("--summary-out", type=Path, default=Path("checkpoints/velocity/summary.json"))
    return p.parse_args(argv)


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


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))

    from phoenix.velocity.contract import MANIFEST_NAME, build_manifest
    from phoenix.velocity.env_cfg import ACTUATOR_DAMPING, ACTUATOR_STIFFNESS
    from phoenix.velocity.spec import default_spec, spec_from_dict

    fallback_spec = default_spec()
    git_sha, git_dirty = _git_sha(repo_root)

    summary = {"seeds": []}
    for run_root in args.run_roots:
        latest = run_root / "latest.pt"
        seed_entry: dict = {"run_root": str(run_root)}
        if not latest.exists():
            seed_entry["error"] = "no latest.pt"
            summary["seeds"].append(seed_entry)
            logger.warning("%s: no latest.pt, skipping", run_root)
            continue

        spec_path = run_root / "spec.json"
        if spec_path.exists():
            # THIS run's actual spec (seed, action.clip_actions, ...), not a fresh
            # default_spec() -- runs launched before the 2026-09-25 clip fix, or with
            # a CLI override, must keep their own recorded action_clip in the manifest.
            spec = spec_from_dict(json.loads(spec_path.read_text()))
        else:
            spec = fallback_spec
            logger.warning("%s: no spec.json, using default_spec() (seed/action_clip may be wrong)", run_root)
        seed = spec.seed
        ckpt_sha = _sha256(latest.resolve())

        curriculum_ranges = spec.commands.final_ranges()
        deploy_block = {
            "kp": ACTUATOR_STIFFNESS,
            "kd": ACTUATOR_DAMPING,
            "action_clip": spec.action.clip_actions,
        }
        manifest = build_manifest(
            checkpoint_sha256=ckpt_sha,
            commands=curriculum_ranges,
            git_sha=git_sha,
            git_dirty=git_dirty,
            seed=int(seed),
            task="phoenix-velocity-flat-go2",
            simulator="isaac-lab-physx",
            reward_scales=spec.reward_scales(),
            domain_randomization=json.loads(json.dumps(spec.domain_randomization.__dict__, default=str)),
            curriculum={"num_levels": spec.curriculum.num_levels, "enabled": spec.curriculum.enabled},
            deploy=deploy_block,
        )
        manifest_path = run_root / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2))
        seed_entry["manifest"] = str(manifest_path)
        seed_entry["checkpoint_sha256"] = ckpt_sha
        seed_entry["deploy"] = deploy_block

        onnx_path = run_root / "policy.onnx"
        export_cmd = [
            str(args.python), "-m", "phoenix.sim2real.export",
            "--checkpoint", str(latest.resolve()),
            "--output", str(onnx_path),
            "--verify",
        ]
        r = subprocess.run(export_cmd, cwd=repo_root, env=_env(repo_root), capture_output=True, text=True, timeout=600)
        seed_entry["export_returncode"] = r.returncode
        seed_entry["export_stdout_tail"] = r.stdout[-2000:]
        seed_entry["export_stderr_tail"] = r.stderr[-2000:]
        if r.returncode == 0:
            seed_entry["onnx"] = str(onnx_path)
        else:
            logger.error("export FAILED for %s:\n%s", run_root, r.stderr[-2000:])

        eval_out = run_root / "eval_metrics.json"
        eval_cmd = [
            str(args.python), "scripts/eval_velocity.py",
            "--checkpoint", str(latest.resolve()),
            "--num-envs", str(args.eval_num_envs),
            "--num-steps", str(args.eval_num_steps),
            "--out", str(eval_out),
            "--headless",
        ]
        r2 = subprocess.run(eval_cmd, cwd=repo_root, env=_env(repo_root), capture_output=True, text=True, timeout=1800)
        seed_entry["eval_returncode"] = r2.returncode
        seed_entry["eval_stderr_tail"] = r2.stderr[-2000:]
        if r2.returncode == 0 and eval_out.exists():
            seed_entry["eval_metrics"] = json.loads(eval_out.read_text())
        else:
            logger.error("eval FAILED for %s:\n%s", run_root, r2.stderr[-2000:])

        if not args.skip_gate and seed_entry.get("onnx"):
            gate_out = run_root / "sim2sim_gate"
            gate_cmd = [
                str(SIM2SIM_PYTHON),
                str(SIM2SIM_WORKTREE / "scripts" / "sim2sim_gate.py"),
                "--onnx", str(onnx_path.resolve()),
                "--manifest", str(manifest_path.resolve()),
                "--out", str(gate_out),
                "--no-video",
            ]
            import os as _os

            gate_env = {k: v for k, v in _os.environ.items() if k != "PYTHONPATH"}
            r3 = subprocess.run(
                gate_cmd, cwd=SIM2SIM_WORKTREE, env=gate_env, capture_output=True, text=True, timeout=900
            )
            seed_entry["gate_returncode"] = r3.returncode
            seed_entry["gate_verdict"] = (
                "PASS" if r3.returncode == 0 else "FAIL" if r3.returncode == 1 else "DIAGNOSTIC_OR_ERROR"
            )
            seed_entry["gate_stdout_tail"] = r3.stdout[-3000:]
            seed_entry["gate_stderr_tail"] = r3.stderr[-1500:]
            report = gate_out / "gate_report.json"
            if report.exists():
                seed_entry["gate_report"] = json.loads(report.read_text())
            if r3.returncode not in (0, 1):
                logger.error("sim2sim gate ERRORED for %s:\n%s", run_root, r3.stderr[-1500:])
        elif not args.skip_gate:
            seed_entry["gate_verdict"] = "SKIPPED_NO_ONNX"

        summary["seeds"].append(seed_entry)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2))
    logger.info("summary written -> %s", args.summary_out)
    return 0


def _env(repo_root: Path) -> dict[str, str]:
    import os

    e = dict(os.environ)
    e["PYTHONPATH"] = str(repo_root / "src") + ":" + e.get("PYTHONPATH", "")
    e["OMNI_KIT_ACCEPT_EULA"] = "YES"
    e["PYTHONUNBUFFERED"] = "1"
    return e


if __name__ == "__main__":
    sys.exit(main())
