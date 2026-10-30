#!/usr/bin/env python3
"""Train-only grid fit of model B joint Coulomb friction, as pre-registered in
configs/sim2real/actuator_model_B.yaml. Runs scripts/actuator_replay.py --splits train
for each grid point (in parallel), never reads the held-out sessions, and writes the
fit table plus a frozen-parameter YAML.

    PYTHONPATH=src python scripts/fit_model_b.py --data-root <hw_sessions> --scene <scene.xml> \
        --cache logs/actuator_s2r/baseline_A/cache --out logs/actuator_s2r/model_B/fit \
        --frozen configs/sim2real/actuator_model_B_frozen.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from phoenix.sim2real.actuator_replay import JOINT_TYPES

REPO = Path(__file__).resolve().parents[1]
SPLIT = REPO / "configs/sim2real/actuator_replay_split.yaml"
PREREG = REPO / "configs/sim2real/actuator_model_B.yaml"


def objective(metrics_path: Path) -> dict[str, float]:
    """Per joint type: mean over training sessions of the session type-mean pos RMSE [rad]."""
    m = json.loads(metrics_path.read_text())
    assert all(r["split"] == "train" for r in m["sessions"].values()), "held-out session in a fit run"
    return {jt: float(np.mean([r["by_type"]["pos_rmse"][jt] for r in m["sessions"].values()
                               if jt in r["by_type"]["pos_rmse"]])) for jt in JOINT_TYPES}


def run_point(args, f: dict[str, float], out: Path) -> dict[str, float]:
    tag = "_".join(f"{f[jt]:.4f}" for jt in JOINT_TYPES)
    d = out / "runs" / tag
    if not (d / "metrics.json").exists():
        cmd = [sys.executable, str(REPO / "scripts/actuator_replay.py"), "--split", str(SPLIT),
               "--data-root", args.data_root, "--scene", args.scene, "--dt", "0.001", "--reset-s", "2.0",
               "--delay-s", "0.0", "--cache", args.cache, "--no-plots", "--splits", "train",
               "--model-b", str(PREREG), "--frictionloss", ",".join(f"{f[jt]:.6f}" for jt in JOINT_TYPES),
               "--label", f"B fit point {tag}", "--out", str(d)]
        env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)
    return objective(d / "metrics.json")


def sweep(args, grids: dict[str, np.ndarray], out: Path) -> list[dict]:
    n = len(next(iter(grids.values())))
    pts = [{jt: float(grids[jt][i]) for jt in JOINT_TYPES} for i in range(n)]
    with ThreadPoolExecutor(args.workers) as ex:
        objs = list(ex.map(lambda p: run_point(args, p, out), pts))
    return [{"friction_nm": p, "pos_rmse": o} for p, o in zip(pts, objs, strict=True)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--workers", type=int, default=11)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(PREREG.read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    b = {jt: tuple(cfg["bounds_nm"][jt]) for jt in JOINT_TYPES}

    ref = sweep(args, {jt: np.array([0.0]) for jt in JOINT_TYPES}, out)[0]  # F = 0, must equal model A
    g1 = {jt: np.linspace(*b[jt], 11) for jt in JOINT_TYPES}
    s1 = sweep(args, g1, out)
    best1 = {jt: min(s1, key=lambda r, jt=jt: r["pos_rmse"][jt])["friction_nm"][jt] for jt in JOINT_TYPES}
    step = {jt: g1[jt][1] - g1[jt][0] for jt in JOINT_TYPES}
    g2 = {jt: np.clip(np.linspace(best1[jt] - step[jt], best1[jt] + step[jt], 5), *b[jt]) for jt in JOINT_TYPES}
    s2 = sweep(args, g2, out)
    allpts = s1 + s2
    best = {jt: min(allpts, key=lambda r, jt=jt: r["pos_rmse"][jt]) for jt in JOINT_TYPES}
    frozen = {jt: best[jt]["friction_nm"][jt] for jt in JOINT_TYPES}
    combo = sweep(args, {jt: np.array([frozen[jt]]) for jt in JOINT_TYPES}, out)[0]
    coupling = {jt: combo["pos_rmse"][jt] / best[jt]["pos_rmse"][jt] - 1 for jt in JOINT_TYPES}
    at_bound = {jt: bool(np.isclose(frozen[jt], b[jt][0]) or np.isclose(frozen[jt], b[jt][1])) for jt in JOINT_TYPES}

    res = {"objective": cfg["fit"]["objective"], "bounds_nm": b, "F0_reference": ref, "stage1": s1,
           "stage2": s2, "frozen_nm": frozen, "combined_check": combo, "coupling_rel_change": coupling,
           "frozen_at_bound": at_bound,
           "train_gain_vs_F0": {jt: 1 - combo["pos_rmse"][jt] / ref["pos_rmse"][jt] for jt in JOINT_TYPES}}
    (out / "fit_results.json").write_text(json.dumps(res, indent=1))
    print("F=0 (model A) train pos RMSE:", {k: round(v, 5) for k, v in ref["pos_rmse"].items()})
    for r in s1 + s2:
        print("F", {k: round(v, 3) for k, v in r["friction_nm"].items()},
              "pos", {k: round(v, 5) for k, v in r["pos_rmse"].items()})
    print("frozen", frozen, "combined", {k: round(v, 5) for k, v in combo["pos_rmse"].items()},
          "coupling", {k: round(v, 4) for k, v in coupling.items()}, "at_bound", at_bound)
    Path(args.frozen).write_text(
        "# Model B frozen parameters: written by scripts/fit_model_b.py from TRAINING sessions only,\n"
        "# before any held-out evaluation. Do not edit after the held-out run.\n"
        + yaml.safe_dump({"structure": cfg["structure"], "frozen_nm": frozen, "constraint": cfg["constraint"],
                          "fit_objective_train_pos_rmse_rad": combo["pos_rmse"],
                          "model_A_train_pos_rmse_rad": ref["pos_rmse"]}, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
