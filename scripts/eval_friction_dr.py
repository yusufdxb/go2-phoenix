"""Evaluate the preregistered friction-DR study (docs/preregistration/2026-10-07-joint-friction-dr.md).

SIM ONLY. Per checkpoint (incumbent model_2750, and model_3750 of every T/C run):
  1. export ONNX with phoenix.sim2real.export --verify (the SAME exporter for every arm,
     the incumbent included, so export differences cannot favour an arm);
  2. write phoenix_manifest.json exactly as scripts/postrun_velocity.py does;
  3. write a deploy config = the model_2750 walk config with the four policy paths swapped;
  4. run scripts/desk/sim_replay_walk.py on nominal, fmeas=1.0, fmeas=1.25, fmeas=1.5.
Then `--analyze` computes the preregistered endpoints and tests.

    ~/Sim/isaac-sim-venv/bin/python scripts/eval_friction_dr.py --label I     # incumbent
    ~/Sim/isaac-sim-venv/bin/python scripts/eval_friction_dr.py --label fric_T_s50
    ~/Sim/isaac-sim-venv/bin/python scripts/eval_friction_dr.py --analyze
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PY = "/home/yusuf/Sim/isaac-sim-venv/bin/python"
BASE = Path(
    "/home/yusuf/Archive/phoenix-deploy-candidates-2026-10-03/checkpoints/walk_vnext3c_seed47"
)
INCUMBENT = BASE / "2026-09-26_21-06-20" / "model_2750.pt"
TEMPLATE = REPO / "configs/sim2real/proposed/deploy_walk_iter2750.yaml"
RUNS = REPO / "checkpoints/friction_dr"
OUT = REPO / "results/friction_dr"
EVAL_ITER = 3749  # final checkpoint of a 1000-iteration resume from 2750 (amendment 1)
CONDITIONS = "nominal;fmeas=1.0;fmeas=1.25;fmeas=1.5"
SEEDS = (50, 51, 52, 53, 54)
MIN_EFFECT = {"E1": 0.15, "E2": 0.10}
ALPHA = 0.05


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(label: str) -> tuple[Path, Path, Path]:
    """(checkpoint, spec.json, curriculum_final_state.json) for a label."""
    if label == "I":
        return INCUMBENT, INCUMBENT.parent / "spec.json", BASE / "curriculum_final_state.json"
    root = RUNS / label
    ckpts = sorted(root.glob(f"*/model_{EVAL_ITER}.pt"))
    if len(ckpts) != 1:
        raise SystemExit(f"{label}: expected one model_{EVAL_ITER}.pt, found {ckpts}")
    return ckpts[0], ckpts[0].parent / "spec.json", root / "curriculum_final_state.json"


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO / 'src'}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def evaluate(label: str, workers: int) -> None:
    from phoenix.velocity.contract import MANIFEST_NAME, CommandRanges, build_manifest
    from phoenix.velocity.env_cfg import ACTUATOR_DAMPING, ACTUATOR_STIFFNESS
    from phoenix.velocity.spec import spec_from_dict

    ckpt, spec_path, cur_path = _resolve(label)
    out = OUT / label
    out.mkdir(parents=True, exist_ok=True)
    onnx = out / "policy.onnx"
    r = subprocess.run(
        [PY, "-m", "phoenix.sim2real.export", "--checkpoint", str(ckpt), "--output", str(onnx), "--verify"],
        cwd=REPO, env=_env(), capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    (out / "export.log").write_text(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"{label}: export failed, see {out / 'export.log'}")

    spec = spec_from_dict(json.loads(spec_path.read_text()))
    cur = json.loads(cur_path.read_text())
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip()
    )
    manifest = build_manifest(
        checkpoint_sha256=_sha256(ckpt),
        commands=CommandRanges.from_mapping(cur["widest_ranges_reached"]),
        git_sha=git_sha,
        git_dirty=dirty,
        seed=int(spec.seed),
        task="phoenix-velocity-flat-go2",
        simulator="isaac-lab-physx",
        reward_scales=spec.reward_scales(),
        domain_randomization=json.loads(
            json.dumps(spec.domain_randomization.__dict__, default=str)
        ),
        curriculum={"num_levels": spec.curriculum.num_levels, "enabled": spec.curriculum.enabled},
        obs_schema_version=spec.obs_schema,
        deploy={
            "kp": ACTUATOR_STIFFNESS,
            "kd": ACTUATOR_DAMPING,
            "action_clip": spec.action.clip_actions,
        },
    )
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

    import yaml

    deploy = yaml.safe_load(TEMPLATE.read_text())
    deploy["policy"].update(
        onnx_path=str(onnx),
        manifest_path=str(out / MANIFEST_NAME),
        torchscript_path=str(out / "policy.pt"),
        checkpoint_path=str(ckpt),
    )
    deploy_path = out / "deploy.yaml"
    deploy_path.write_text(yaml.safe_dump(deploy, sort_keys=False))

    r = subprocess.run(
        [PY, str(REPO / "scripts/desk/sim_replay_walk.py"), "--out-dir", str(out / "replay"),
         "--deploy-config", str(deploy_path), "--conditions", CONDITIONS,
         "--workers", str(workers), "--tag", "fric"],
        cwd=REPO, env=_env(), capture_output=True, text=True, timeout=3600,
    )  # fmt: skip
    (out / "replay.log").write_text(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"{label}: replay failed, see {out / 'replay.log'}")
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "label": label,
                "checkpoint": str(ckpt),
                "checkpoint_sha256": _sha256(ckpt),
                "onnx_sha256": _sha256(onnx),
                "replay_script_sha256": _sha256(REPO / "scripts/desk/sim_replay_walk.py"),
                "git_sha": git_sha,
                "git_dirty": dirty,
            },
            indent=2,
        )
    )
    print(f"{label}: done -> {out}", flush=True)


def endpoints(label: str) -> dict:
    rows = json.loads((OUT / label / "replay" / "fric_results.json").read_text())
    by = {(r["label"], r["preset"]): r for r in rows}
    res: dict = {"fell_any": any(r["fell"] for r in rows)}
    for cond in ("nominal", "fmeas=1.0", "fmeas=1.25", "fmeas=1.5"):
        yaw = by[(cond, "yaw05_3s")]
        fwd = by[(cond, "vx03_4s")]
        res[cond] = {"yaw": yaw["yaw_deg"] / 86.0, "fwd": fwd["fwd_m"] / 1.2}
    res["E1"] = res["fmeas=1.25"]["yaw"]
    res["E2"] = res["fmeas=1.25"]["fwd"]
    return res


def _signflip_p(d: np.ndarray) -> float:
    """One-sided exact sign-flip p for mean(d) > 0."""
    obs = d.mean()
    flips = [np.mean(d * np.array(s)) for s in itertools.product((1, -1), repeat=len(d))]
    return float(np.mean([f >= obs - 1e-12 for f in flips]))


def _boot_ci(d: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def analyze() -> int:
    report: dict = {"incumbent": endpoints("I"), "seeds": {}, "tests": {}}
    d = {"E1": [], "E2": []}
    for s in SEEDS:
        t, c = endpoints(f"fric_T_s{s}"), endpoints(f"fric_C_s{s}")
        report["seeds"][s] = {"T": t, "C": c}
        for k in d:
            d[k].append(t[k] - c[k])
    passed_prev = True
    for k in ("E1", "E2"):
        arr = np.asarray(d[k])
        p = _signflip_p(arr)
        ok = passed_prev and p <= ALPHA and arr.mean() >= MIN_EFFECT[k]
        report["tests"][k] = {
            "d": arr.tolist(),
            "mean_d": float(arr.mean()),
            "ci95_boot": _boot_ci(arr),
            "p_signflip_one_sided": p,
            "min_effect": MIN_EFFECT[k],
            "tested": passed_prev,
            "pass": bool(ok),
        }
        passed_prev = bool(ok)
    report["P2_incumbent_yaw_at_1p25"] = report["incumbent"]["E1"]
    report["P2_pass"] = report["incumbent"]["E1"] <= 0.60
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["tests"], indent=2))
    print(
        "P2 incumbent yaw@1.25:",
        round(report["incumbent"]["E1"], 3),
        "pass" if report["P2_pass"] else "FAIL",
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    for label in args.label:
        evaluate(label, args.workers)
    return analyze() if args.analyze else 0


if __name__ == "__main__":
    sys.exit(main())
