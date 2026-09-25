"""After all PhoenixVelocity training runs finish: manifest + ONNX export + eval, per seed.

Meant to be chained after the tmux training sequence (see scripts/run_velocity_seeds.sh):
for each seed's log_root/latest.pt it
  1. writes phoenix_manifest.json (phoenix.velocity.contract.build_manifest)
  2. exports ONNX + verifies parity (phoenix.sim2real.export, deploy-owned, called not edited)
  3. runs scripts/eval_velocity.py for held-out metrics with DR on
  4. aggregates everything into a summary JSON

Runs OUTSIDE Isaac Sim for steps 1-2 (pure torch/hashing), and re-launches an
Isaac Sim subprocess per seed for step 3 (eval needs the live env).
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-run export + eval chain for PhoenixVelocity seeds.")
    p.add_argument("--run-roots", type=Path, nargs="+", required=True, help="checkpoints/velocity/<run_name> dirs")
    p.add_argument("--python", type=Path, default=Path("/home/yusuf/Sim/isaac-sim-venv/bin/python"))
    p.add_argument("--eval-num-envs", type=int, default=256)
    p.add_argument("--eval-num-steps", type=int, default=1000)
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

    from phoenix.velocity.contract import CommandRanges, MANIFEST_NAME, build_manifest
    from phoenix.velocity.spec import default_spec

    spec = default_spec()
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
        seed = json.loads(spec_path.read_text())["seed"] if spec_path.exists() else spec.seed
        ckpt_sha = _sha256(latest.resolve())

        curriculum_ranges = spec.commands.final_ranges()
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
        )
        manifest_path = run_root / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2))
        seed_entry["manifest"] = str(manifest_path)
        seed_entry["checkpoint_sha256"] = ckpt_sha

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
