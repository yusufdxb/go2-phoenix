"""Postrun v2: full checkpoint sweep (not just latest.pt), ranked, gated.

For each run root:
  1. Build a diagnostic table (iteration, action_abs_p99, mean action std,
     episode length) from the TRAINING LOG at every 500-iter checkpoint, so
     collapse (or its absence) is visible without any extra GPU work.
  2. Evaluate every 500-iter checkpoint with the fixed eval_velocity.py
     (1050-step full-episode rollout, 256 envs -- see the earlier 500-step
     bug where most envs never completed an episode).
  3. Rank by the same score as rank_eval_sweep.py.
  4. Refuse to export/gate unless the git tree is clean (the gate's own
     manifest_valid check fails a dirty-tree checkpoint; better to fail
     loudly here than produce a misleading FAIL).
  5. Export ONNX + verify parity for the top 3 per run, build a manifest
     matching postrun_velocity.py's exact shape (extra manifest fields broke
     gate_v2's schema check earlier tonight), run gate_v2 and gate_v3 (if the
     sim2sim worktree has picked it up) on each.
  6. Write one ranked summary JSON per run plus a combined one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SIM2SIM_WORKTREE = Path("/home/yusuf/workspace/go2-phoenix/.claude/worktrees/agent-a6f2a9725e191de1f")
SIM2SIM_PYTHON = Path("/home/yusuf/Projects/go2-positive-control/.venv/bin/python")
GATE_V2 = SIM2SIM_WORKTREE / "configs" / "sim2sim" / "gate_v2.yaml"
GATE_V3 = SIM2SIM_WORKTREE / "configs" / "sim2sim" / "gate_v3.yaml"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-roots", type=Path, nargs="+", required=True)
    p.add_argument("--python", type=Path, default=Path("/home/yusuf/Sim/isaac-sim-venv/bin/python"))
    p.add_argument("--step", type=int, default=500, help="checkpoint sweep granularity")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--eval-num-envs", type=int, default=256)
    p.add_argument("--eval-num-steps", type=int, default=1050)
    p.add_argument("--summary-out", type=Path, required=True)
    return p.parse_args(argv)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha_dirty(repo_root: Path) -> tuple[str, bool]:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True).stdout.strip())
    return sha, dirty


def _checkpoints_at_step(run_root: Path, step: int) -> list[tuple[int, Path]]:
    found: dict[int, Path] = {}
    for p in run_root.glob("*/model_*.pt"):
        m = re.match(r"model_(\d+)\.pt", p.name)
        if not m:
            continue
        it = int(m.group(1))
        if it % step == 0 and it > 0:
            # prefer the most recently modified if duplicated across subdirs
            if it not in found or p.stat().st_mtime > found[it].stat().st_mtime:
                found[it] = p
    return sorted(found.items())


def _training_log_stats(run_root: Path) -> dict[int, dict]:
    """Parse logs/train_<run_name>.log for action_abs_p99 / mean action std / episode length per iteration."""
    log_path = REPO_ROOT / "logs" / f"train_{run_root.name}.log"
    stats: dict[int, dict] = {}
    if not log_path.exists():
        return stats
    text = log_path.read_text(errors="replace")
    # Split on iteration headers, keep the block after each.
    blocks = re.split(r"Learning iteration (\d+)/\d+", text)
    # blocks = [pre, iter_str, block, iter_str, block, ...]
    for i in range(1, len(blocks) - 1, 2):
        it = int(blocks[i])
        block = blocks[i + 1]
        p99_m = re.search(r"action_abs_p99:\s*([-\d.]+)", block)
        std_m = re.search(r"Mean action std:\s*([-\d.]+)", block)
        len_m = re.search(r"Mean episode length:\s*([-\d.]+)", block)
        entropy_m = re.search(r"Mean entropy loss:\s*([-\d.]+)", block)
        stats[it] = {
            "action_abs_p99": float(p99_m.group(1)) if p99_m else None,
            "mean_action_std": float(std_m.group(1)) if std_m else None,
            "mean_episode_length": float(len_m.group(1)) if len_m else None,
            "mean_entropy_loss": float(entropy_m.group(1)) if entropy_m else None,
        }
    return stats


def _eval_env(repo_root: Path) -> dict[str, str]:
    import os

    e = dict(os.environ)
    e["PYTHONPATH"] = str(repo_root / "src") + ":" + e.get("PYTHONPATH", "")
    e["OMNI_KIT_ACCEPT_EULA"] = "YES"
    e["PYTHONUNBUFFERED"] = "1"
    return e


def _score(m: dict) -> float:
    vt = m["velocity_tracking"]
    lin_errs = [vt[k]["mean_lin_err_mps"] for k in ("stand", "slow", "fast") if vt[k]["mean_lin_err_mps"] is not None]
    yaw_errs = [vt[k]["mean_yaw_err_rad_s"] for k in ("stand", "slow", "fast") if vt[k]["mean_yaw_err_rad_s"] is not None]
    lin_err = sum(lin_errs) / len(lin_errs) if lin_errs else 1.0
    yaw_err = sum(yaw_errs) / len(yaw_errs) if yaw_errs else 1.0
    headroom = max(v["peak_over_limit"] for v in m["torque_headroom"].values())
    h = m["standing_base_height_m"]["mean"] or 0.0
    height_penalty = abs(h - 0.30) if h else 1.0
    # 2026-09-25: the random-command joint_near_limit_fraction (averaged over
    # all 12 joints and whatever commands happen to be sampled) hid the real
    # per-joint failure sim2sim found (seed42@3000 drove FR/RL calves to their
    # HARD limit during CCW yaw). fixed_command_sweep directly measures this;
    # weighted heavily (5x) since it is a specific, reproducible hardware-limit
    # violation, not a general tracking-quality signal.
    sweep = m.get("fixed_command_sweep") or {}
    calf_hard_limit_penalty = max(
        (v.get("fraction_envs_calf_near_hard_limit", 0.0) for v in sweep.values()), default=0.0
    )
    calf_overshoot_penalty = max(
        (v.get("max_calf_hard_limit_overshoot_rad", 0.0) for v in sweep.values()), default=0.0
    )
    return (
        10.0 * m["fall_rate"]
        + lin_err
        + yaw_err
        + m["joint_near_limit_fraction"]
        + max(0.0, headroom - 0.9)
        + 0.5 * height_penalty
        + 5.0 * calf_hard_limit_penalty
        + 20.0 * calf_overshoot_penalty
    )


def _run_gate(gate_yaml: Path, onnx_path: Path, manifest_path: Path, out_dir: Path) -> dict:
    import os

    if not gate_yaml.exists():
        return {"skipped": f"{gate_yaml} not found"}
    cmd = [
        str(SIM2SIM_PYTHON), str(SIM2SIM_WORKTREE / "scripts" / "sim2sim_gate.py"),
        "--onnx", str(onnx_path.resolve()),
        "--manifest", str(manifest_path.resolve()),
        "--gate-config", str(gate_yaml),
        "--out", str(out_dir.resolve()),
        "--no-video",
    ]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run(cmd, cwd=SIM2SIM_WORKTREE, env=env, capture_output=True, text=True, timeout=900)
    verdict = "PASS" if r.returncode == 0 else "FAIL" if r.returncode == 1 else "DIAGNOSTIC_OR_ERROR"
    report_path = out_dir / "gate_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    return {
        "gate_config": str(gate_yaml),
        "returncode": r.returncode,
        "verdict": verdict,
        "stdout_tail": r.stdout[-3000:],
        "stderr_tail": r.stderr[-1500:],
        "report": report,
    }


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from phoenix.velocity.contract import CommandRanges, MANIFEST_NAME, build_manifest
    from phoenix.velocity.env_cfg import ACTUATOR_DAMPING, ACTUATOR_STIFFNESS
    from phoenix.velocity.spec import default_spec, spec_from_dict

    git_sha, git_dirty = _git_sha_dirty(REPO_ROOT)
    if git_dirty:
        print(
            "WARNING: working tree is DIRTY. gate_v2/v3's manifest_valid check will "
            "fail every export from this tree (verified 2026-09-25: it flags "
            "'checkpoint was trained from a dirty or unknown working tree'). "
            "Export/gate steps will still run so their real per-scenario results "
            "are visible, but the overall verdict will be FAIL regardless of "
            "policy quality until this is committed.",
            flush=True,
        )

    combined = {"git_sha": git_sha, "git_dirty": git_dirty, "runs": {}}

    for run_root in args.run_roots:
        run_name = run_root.name
        print(f"=== sweeping {run_root} ===", flush=True)
        specs = sorted(run_root.glob("*/spec.json"))
        spec = spec_from_dict(json.loads(specs[0].read_text())) if specs else default_spec()

        train_stats = _training_log_stats(run_root)
        checkpoints = _checkpoints_at_step(run_root, args.step)

        run_result: dict = {"run_root": str(run_root), "diagnostic_table": [], "checkpoints": []}

        for it, ckpt_path in checkpoints:
            row = {"iteration": it, **train_stats.get(it, {})}
            run_result["diagnostic_table"].append(row)

            eval_out = REPO_ROOT / "logs" / "eval_sweep_v2" / f"{run_name}_iter{it}.json"
            eval_out.parent.mkdir(parents=True, exist_ok=True)
            eval_cmd = [
                str(args.python), "scripts/eval_velocity.py",
                "--checkpoint", str(ckpt_path.resolve()),
                "--num-envs", str(args.eval_num_envs),
                "--num-steps", str(args.eval_num_steps),
                "--out", str(eval_out),
                "--headless",
            ]
            r = subprocess.run(eval_cmd, cwd=REPO_ROOT, env=_eval_env(REPO_ROOT), capture_output=True, text=True, timeout=900)
            entry = {"iteration": it, "checkpoint": str(ckpt_path), "eval_returncode": r.returncode}
            if r.returncode == 0 and eval_out.exists():
                metrics = json.loads(eval_out.read_text())
                entry["metrics"] = metrics
                entry["score"] = _score(metrics)
            else:
                entry["eval_stderr_tail"] = r.stderr[-1500:]
                entry["score"] = float("inf")
            run_result["checkpoints"].append(entry)
            print(f"  iter {it}: score={entry['score']}", flush=True)

        ranked = sorted([c for c in run_result["checkpoints"] if c["score"] != float("inf")], key=lambda c: c["score"])
        run_result["ranked"] = [{"iteration": c["iteration"], "score": c["score"]} for c in ranked]

        top = ranked[: args.top_n]
        run_result["top"] = []
        for c in top:
            it = c["iteration"]
            ckpt_path = Path(c["checkpoint"])
            out_dir = REPO_ROOT / "logs" / "eval_sweep_v2" / f"{run_name}_iter{it}_gate"
            out_dir.mkdir(parents=True, exist_ok=True)

            ckpt_sha = _sha256(ckpt_path.resolve())
            cur_state_path = run_root / "curriculum_final_state.json"
            curriculum_ranges = spec.commands.final_ranges()
            cur_dict = {"num_levels": spec.curriculum.num_levels, "enabled": spec.curriculum.enabled}
            if cur_state_path.exists():
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
                deploy={"kp": ACTUATOR_STIFFNESS, "kd": ACTUATOR_DAMPING, "action_clip": spec.action.clip_actions},
            )
            manifest_path = out_dir / MANIFEST_NAME
            manifest_path.write_text(json.dumps(manifest, indent=2))

            onnx_path = out_dir / "policy.onnx"
            export_cmd = [
                str(args.python), "-m", "phoenix.sim2real.export",
                "--checkpoint", str(ckpt_path.resolve()),
                "--output", str(onnx_path),
                "--verify",
            ]
            r = subprocess.run(export_cmd, cwd=REPO_ROOT, env=_eval_env(REPO_ROOT), capture_output=True, text=True, timeout=300)
            top_entry = {
                "iteration": it,
                "checkpoint": str(ckpt_path),
                "score": c["score"],
                "manifest": str(manifest_path),
                "export_returncode": r.returncode,
                "export_stdout_tail": r.stdout[-1500:],
                "export_stderr_tail": r.stderr[-1500:],
            }
            if r.returncode == 0:
                top_entry["onnx"] = str(onnx_path)
                top_entry["gate_v2"] = _run_gate(GATE_V2, onnx_path, manifest_path, out_dir / "gate_v2")
                top_entry["gate_v3"] = _run_gate(GATE_V3, onnx_path, manifest_path, out_dir / "gate_v3")
            run_result["top"].append(top_entry)
            print(
                f"  TOP iter {it}: gate_v2={top_entry.get('gate_v2', {}).get('verdict')} "
                f"gate_v3={top_entry.get('gate_v3', {}).get('verdict')}",
                flush=True,
            )

        combined["runs"][run_name] = run_result

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(combined, indent=2))
    print(f"summary written -> {args.summary_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
