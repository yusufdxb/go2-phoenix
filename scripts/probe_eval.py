"""Replay v2 friction-probe recordings through frozen actuator models and score them.

Implements the replay and the analysis of ``docs/preregistration/2026-10-30-actuator-probe-v2.md``
with every threshold read from ``configs/sim2real/actuator_probe_v2_prereg.yaml``. Each run
directory holds ``fp_trace.npz`` and, on hardware, ``bridge.jsonl`` (the command stream and IMU);
without a bridge log the probe rows are the command stream (the simulated probe).

Each complete excursion is replayed on its own: all 12 joints initialised from the measured
q, dq at its first row, 1 ms MuJoCo steps, the logged command held between ticks, no reset,
sampled at the probe row times. Models: A (no added friction), B, Bm, Bm_fam, plus any
``--model NAME=yaml``. ``--grid`` adds the oracle constant-Coulomb sweep (in-sample by
construction) and ``--fit-c1`` writes Model C1 fitted on these runs (use on CAL only).

    PYTHONPATH=src python scripts/probe_eval.py RUN_DIR [RUN_DIR ...] --scene <scene.xml> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actuator_replay import (  # noqa: E402
    actuator,
    build_fixed_base,
    set_joint_friction,
    tilt_gravity,
)

from phoenix.sim2real import friction_probe as fp  # noqa: E402
from phoenix.sim2real import probe_metrics as pm  # noqa: E402
from phoenix.sim2real.actuator_replay import JOINT_TYPES, joint_type, load_session  # noqa: E402
from phoenix.sim2real.go2_model import UNITREE_MOTOR_ORDER  # noqa: E402

ORDER = list(UNITREE_MOTOR_ORDER)
PREREG = "configs/sim2real/actuator_probe_v2_prereg.yaml"
MAX_GAP_S = 0.10  # an excursion with a larger internal gap is dropped for every model
PLOT_MODELS = ("A", "B", "Bm", "Bm_fam")
_G: dict = {}  # scene, profile, dt and runs, inherited by forked workers


def _sensors(bridge: Path) -> dict:
    """Run summary of the bridge ``sensors`` block: mean roll, pitch [deg] from ``rpy``
    (ASSUMPTION: radians; level if absent), foot_force per leg (median, max; a rise on a hung
    robot means ground contact), power_v (first, last) and max motor_temp_c."""
    rp, ff, pv, mt = [], [], [], []
    with bridge.open() as fh:
        for line in fh:
            if '"sensors"' not in line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = r.get("sensors") or {}
            if r.get("record") != "tick":
                continue
            if v.get("rpy") is not None and len(v["rpy"]) >= 2:
                rp.append(v["rpy"][:2])
            if v.get("foot_force") is not None and len(v["foot_force"]) == 4:
                ff.append(v["foot_force"])
            if v.get("power_v") is not None:
                pv.append(float(v["power_v"]))
            if v.get("motor_temp_c") is not None:
                mt.append(max(v["motor_temp_c"]))
    out = {"tilt": (0.0, 0.0) if not rp else tuple(float(x) for x in np.degrees(np.mean(rp, axis=0))),
           "tilt_src": "rpy" if rp else "none (level)"}
    if ff:
        f = np.asarray(ff, dtype=np.float64)
        out["foot_force_median"], out["foot_force_max"] = np.median(f, 0).tolist(), f.max(0).tolist()
    if pv:
        out["power_v_first_last"] = (pv[0], pv[-1])
    if mt:
        out["motor_temp_max_c"] = max(mt)
    return out


def load_run(d: Path) -> dict:
    z = np.load(d / "fp_trace.npz")
    if [str(n) for n in z["joint_order"]] != ORDER:
        raise ValueError(f"{d}: joint_order is not the Unitree motor order")
    n = len(z["t"])

    def rows12(x):
        x = np.asarray(x, dtype=np.float64)
        return np.broadcast_to(x[:, None] if x.ndim == 1 else x, (n, 12)).copy()

    run = {
        "name": d.name, "q": z["q"], "dq": z["dq"], "target": z["target"],
        "kp": rows12(z["kp"]), "kd": rows12(z["kd"]), "active": z["active"].astype(str),
        "phase": z["phase"].astype(str), "q0": np.asarray(z["q0"], dtype=np.float64),
        "rate": z["rate"] if "rate" in z.files else np.full(n, fp.RATE_RAD_S),
    }
    if (d / "bridge.jsonl").exists():
        s = load_session(d)
        run["t"], run["cmd"] = s.probe["t"], (s.t, s.target, s.kp, s.kd)
        run["sensors"] = _sensors(d / "bridge.jsonl")
        run["tilt"], run["cmd_src"] = run["sensors"]["tilt"], "bridge"
    else:
        run["t"] = z["t"].astype(np.float64)
        run["cmd"] = (run["t"], run["target"], run["kp"], run["kd"])
        run["tilt"], run["cmd_src"], run["sensors"] = (0.0, 0.0), "probe_rows", {}
    ex = pm.excursions(run["active"], run["phase"], run["rate"], run["target"], run["q0"], ORDER)
    keep = [e for e in ex if np.max(np.diff(run["t"][e.rows])) <= MAX_GAP_S]
    run["ex"], run["n_ex_dropped"] = keep, len(ex) - len(keep)
    return run


def yaml_spec(path: str | Path) -> dict:
    return dict_spec(yaml.safe_load(Path(path).read_text()))


def dict_spec(cfg: dict) -> dict:
    c = cfg.get("constraint", {})
    return {"fl": {k: float(v) for k, v in cfg["frozen_nm"].items()},
            "solref": c.get("solref"), "solimp": c.get("solimp")}


def _replay_ex(mujoco, m, d, qadr, dadr, calf, run: dict, ex) -> tuple[np.ndarray, np.ndarray]:
    r, j, dt = ex.rows, ex.joint, _G["dt"]
    tk = run["t"][r]
    tc, tg, kpc, kdc = run["cmd"]
    mujoco.mj_resetData(m, d)
    d.qpos[qadr], d.qvel[dadr] = run["q"][r[0]], run["dq"][r[0]]
    mujoco.mj_forward(m, d)
    qs, dqs = np.empty(len(r)), np.empty(len(r))
    qs[0], dqs[0] = d.qpos[qadr[j]], d.qvel[dadr[j]]
    n = 0
    for i in range(1, len(r)):
        n_i = int(round((tk[i] - tk[0]) / dt))
        while n < n_i:
            c = max(int(np.searchsorted(tc, tk[0] + n * dt + 1e-9, "right")) - 1, 0)
            _, app = actuator(tg[c], d.qpos[qadr], d.qvel[dadr], kpc[c], kdc[c], calf)
            d.qfrc_applied[:] = 0
            d.qfrc_applied[dadr] = app
            mujoco.mj_step(m, d)
            n += 1
        qs[i], dqs[i] = d.qpos[qadr[j]], d.qvel[dadr[j]]
    return qs, dqs


def _replay_model(item: tuple[str, dict]) -> tuple[str, list]:
    import mujoco

    name, spec = item
    m, qadr, dadr, _ = build_fixed_base(_G["scene"], _G["profile"], _G["dt"])
    if spec["fl"] is not None:
        set_joint_friction(m, dadr, spec["fl"], spec["solref"], spec["solimp"])
    d = mujoco.MjData(m)
    calf = np.array([joint_type(n) == "calf" for n in ORDER])
    out = []
    for run in _G["runs"]:
        tilt_gravity(m, *run["tilt"])
        out.append([_replay_ex(mujoco, m, d, qadr, dadr, calf, run, ex) for ex in run["ex"]])
    return name, out


def replay_all(specs: dict[str, dict], workers: int) -> dict[str, list]:
    with mp.get_context("fork").Pool(min(workers, len(specs))) as pool:
        return dict(pool.map(_replay_model, list(specs.items())))


def score(traces: list, runs: list[dict], cfg: dict, thr: float | None = None):
    exc, wins = [], []
    for run, tr in zip(runs, traces, strict=True):
        for ex, (q_m, dq_m) in zip(run["ex"], tr, strict=True):
            e, w = pm.score_excursion(ex, run["t"], run["phase"], run["target"], run["kp"], run["q"],
                                      run["dq"], q_m, dq_m, run["q0"], cfg, thr)
            e["run"] = run["name"]
            for x in w:
                x["run"] = run["name"]
            exc.append(e)
            wins += w
    return exc, wins


def residual_wins(wins: list[dict]) -> list[dict]:
    """Windows with thr_hw replaced by thr_hw - thr_m, so h_vel tests the rate shift a constant
    Coulomb model fails to reproduce (kd v and detection lag shift any plant's raw threshold)."""
    return [{**x, "thr_hw": x["thr_hw"] - x["thr_m"]} for x in wins
            if np.isfinite(x["thr_hw"]) and np.isfinite(x["thr_m"])]


def joint_objective(wins: list[dict]) -> dict[str, float]:
    """Per joint median |threshold error|, a missed or false breakaway counting as infinite."""
    out = {}
    for j, n in enumerate(ORDER):
        v = [abs(x["thr_err"]) if np.isfinite(x["thr_err"]) else np.inf
             for x in wins if x["joint"] == j and (x["hw_broke"] or x["false"])]
        if v:
            out[n] = float(np.median(v))
    return out


def closed_form_f(wins_hw: list[dict]) -> dict[str, float]:
    """F_j = (median + thr_hw + median - thr_hw) / 2: gravity load cancels between directions."""
    out = {}
    for j, n in enumerate(ORDER):
        p = [x["thr_hw"] for x in wins_hw if x["joint"] == j and x["dir"] > 0 and np.isfinite(x["thr_hw"])]
        m = [x["thr_hw"] for x in wins_hw if x["joint"] == j and x["dir"] < 0 and np.isfinite(x["thr_hw"])]
        if p and m:
            out[n] = float((np.median(p) + np.median(m)) / 2.0)
    return out


def oracle(runs, cfg, base: dict, workers: int) -> dict:
    """Constant per-joint Coulomb sweep: uniform 0-1.5 N m by 0.05, then +-0.04 by 0.01."""
    grid = {f"U{f:.2f}": {**base, "fl": {t: float(f) for t in JOINT_TYPES}}
            for f in np.round(np.arange(0.0, 1.5001, 0.05), 2)}
    obj = {k: joint_objective(score(v, runs, cfg)[1]) for k, v in replay_all(grid, workers).items()}
    best = {}
    for n in ORDER:
        cand = [(obj[k][n], grid[k]["fl"]["hip"]) for k in grid if n in obj[k]]
        if cand:
            best[n] = min(cand)[1]
    deltas = np.round(np.arange(-0.04, 0.0401, 0.01), 2)
    ref = {f"R{dl:+.2f}": {**base, "fl": {n: max(best[n] + dl, 0.0) for n in best}} for dl in deltas}
    obj2 = {k: joint_objective(score(v, runs, cfg)[1]) for k, v in replay_all(ref, workers).items()}
    for n in best:
        cand = [(obj2[k][n], ref[k]["fl"][n]) for k in ref if n in obj2[k]]
        best[n] = round(min(cand)[1], 3)
    return {"per_joint_nm": best, "objective": "median |thr_err|, missed/false = inf",
            "label": "oracle: fitted on the scored runs themselves, never a validation result"}


def plot_overlays(runs, traces: dict[str, list], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for ri, run in enumerate(runs):
        for j, n in enumerate(ORDER):
            p = _plot_joint(plt, run, traces, ri, j, n, out, tag=len(runs) > 1)
            if p:
                paths.append(p)
    return paths


def _plot_joint(plt, run, traces, ri: int, j: int, n: str, out: Path, tag: bool) -> str | None:
    idx = [i for i, ex in enumerate(run["ex"]) if ex.joint == j][:8]
    if not idx:
        return None
    fig, axs = plt.subplots(2, 4, figsize=(16, 6), squeeze=False)
    for a, i in zip(axs.flat, idx, strict=False):
        ex = run["ex"][i]
        tt = run["t"][ex.rows] - run["t"][ex.rows[0]]
        q0 = run["q0"][j]
        a.plot(tt, (run["target"][ex.rows, j] - q0) * 1e3, "--", c="0.6", lw=1, label="target")
        a.plot(tt, (run["q"][ex.rows, j] - q0) * 1e3, c="k", lw=1.6, label="hw")
        for mname in PLOT_MODELS:
            if mname in traces:
                a.plot(tt, (traces[mname][ri][i][0] - q0) * 1e3, lw=1, label=mname)
        a.set_title(f"k={ex.k} dir={ex.direction:+d} rate={ex.rate:.2f}", fontsize=9)
        a.set_xlabel("s")
        a.set_ylabel("q - q0 [mrad]")
    axs.flat[0].legend(fontsize=7)
    fig.suptitle(f"{run['name']} {n}")
    fig.tight_layout()
    p = out / (f"overlay_{run['name']}_{n}.png" if tag else f"overlay_{n}.png")
    fig.savefig(p, dpi=90)
    plt.close(fig)
    return p.name


def _fmt(x) -> str:
    return "nan" if not np.isfinite(x) else f"{x:.3f}"


def write_md(res: dict, path: Path) -> None:
    keys = ("abs_thr_err", "thr_hw", "abs_t_err", "missed_frac", "false_frac", "pos_rmse",
            "vel_rmse", "hold_err_mrad", "slip_rate_hw", "slip_rate_m", "slip_recall")
    lines = [f"# probe_eval: {res['label']}", "", f"Runs: {', '.join(res['runs'])}", ""]
    for f in JOINT_TYPES:
        lines += [f"## {f}", "", "| model | " + " | ".join(keys) + " |",
                  "|---" * (len(keys) + 1) + "|"]
        for mname, s in res["summary"].items():
            lines.append(f"| {mname} | " + " | ".join(_fmt(s["per_family"][f].get(k, np.nan))
                                                    for k in keys) + " |")
        lines.append("")
    lines += ["## per joint median |threshold error| [N m]", "",
              "| joint | " + " | ".join(res["summary"]) + " |", "|---" * (len(res["summary"]) + 1) + "|"]
    for n in ORDER:
        lines.append(f"| {n} | " + " | ".join(_fmt(s["per_joint"].get(n, {}).get("abs_thr_err", np.nan))
                                              for s in res["summary"].values()) + " |")
    lines += ["", "## pooled", ""]
    lines += [f"- {mn}: " + ", ".join(f"{k} {_fmt(v)}" for k, v in s["pooled"].items())
              for mn, s in res["summary"].items()]
    lines += ["", "## decisions", "", "```", json.dumps(res["decisions"], indent=1, default=float), "```"]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="run directories (fp_trace.npz [+ bridge.jsonl])")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--profile", default="isaac_matched")
    ap.add_argument("--prereg", default=PREREG)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="", help="e.g. 'P1 positive control', 'VAL', 'in-sample 10-07'")
    ap.add_argument("--model", action="append", default=[], help="extra arm NAME=yaml")
    ap.add_argument("--grid", action="store_true", help="oracle constant-Coulomb sweep (in-sample)")
    ap.add_argument("--fit-c1", default="", help="write Model C1 fitted on these runs to this yaml")
    ap.add_argument("--exclude", action="append", default=[],
                    help="RUN:JOINT excursions dropped for every model (source-documented exclusions)")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.prereg).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [load_run(Path(r)) for r in args.runs]
    drop = {tuple(x.split(":", 1)) for x in args.exclude}
    for run in runs:
        keep = [e for e in run["ex"] if (run["name"], ORDER[e.joint]) not in drop]
        run["n_ex_excluded"], run["ex"] = len(run["ex"]) - len(keep), keep
    _G.update(scene=args.scene, profile=args.profile, dt=cfg["replay"]["dt_s"], runs=runs)
    mods = cfg["models"]
    specs = {"A": {"fl": None, "solref": None, "solimp": None},
             "B": yaml_spec(mods["B"]["config"]), "Bm": yaml_spec(mods["Bm"]["config"]),
             "Bm_fam": dict_spec(mods["Bm_fam"])}
    for kv in args.model:
        k, v = kv.split("=", 1)
        specs[k] = yaml_spec(v)
    traces = replay_all(specs, args.workers)
    sc = {k: score(traces[k], runs, cfg) for k in specs}
    summ = {k: pm.summarize(*sc[k], ORDER) for k in specs}
    alt = cfg["metrics"]["breakaway_thr_alt_rad"]
    summ_alt = {k: pm.summarize(*score(traces[k], runs, cfg, alt), ORDER) for k in specs}

    wins_hw, exc_hw = sc["B"][1], sc["B"][0]
    floor = {}
    for run in runs:
        for n, v in pm.noise_floor(run["t"], run["q"], run["active"], run["phase"], ORDER, cfg).items():
            floor.setdefault(n, []).append(v)
    floor_per_s = float(np.mean([np.mean(v) for v in floor.values()])) if floor else float("nan")
    ex_s = sum(float(run["t"][ex.rows[-1]] - run["t"][ex.rows[0]]) for run in runs for ex in run["ex"])
    hw_rate_per_s = sum(e["n_slip_hw"] for e in exc_hw) / ex_s if ex_s else float("nan")
    hw_med_fam = {f: summ["B"]["per_family"][f].get("thr_hw", np.nan) for f in JOINT_TYPES}
    pooled = summ["B"]["pooled"]
    dec = {
        "usable_thresholds": bool(pooled["hw_breakaway_frac"] >= cfg["decision"]["min_hw_breakaway_frac"]),
        "hw_breakaway_frac": pooled["hw_breakaway_frac"], "A1_onset_frac": pooled["onset_frac"],
        "H_mag": pm.h_mag(summ["B"], summ["Bm"], ORDER, cfg),
        "H_mag_Bm_fam": pm.h_mag(summ["B"], summ["Bm_fam"], ORDER, cfg),
        "H_mag_alt_thr": pm.h_mag(summ_alt["B"], summ_alt["Bm"], ORDER, cfg),
        "H_vel_raw_hw": pm.h_vel(wins_hw, ORDER, cfg),
        "H_vel_resid": {k: pm.h_vel(residual_wins(sc[k][1]), ORDER, cfg) for k in ("B", "Bm")},
        "H_dir": {k: pm.h_dir(sc[k][1], hw_med_fam, ORDER, cfg) for k in ("B", "Bm")},
        "H_inad": pm.h_inad(summ, floor_per_s, hw_rate_per_s, cfg),
        "noise_floor_per_s": floor, "model_C_rule": {
            k: pm.model_c_accept(summ[k], summ["B"], summ["Bm"], cfg) for k in specs},
        "closed_form_F_nm": closed_form_f(wins_hw),
    }
    dec["H_vel"] = {"supported": all(v["supported"] for v in dec["H_vel_resid"].values()),
                    "rule": "amendment 1: hw fast-slow shift not reproduced by B nor Bm"}
    if args.grid or args.fit_c1:
        dec["oracle"] = oracle(runs, cfg, specs["B"], args.workers)
    if args.fit_c1:
        c1 = {"structure": "B0", "frozen_nm": dec["oracle"]["per_joint_nm"],
              "constraint": {"solref": specs["B"]["solref"], "solimp": specs["B"]["solimp"]},
              "fitted_on": [r["name"] for r in runs], "method": dec["oracle"]["objective"],
              "closed_form_crosscheck_nm": dec["closed_form_F_nm"]}
        Path(args.fit_c1).write_text(yaml.safe_dump(c1, sort_keys=False))
    res = {
        "label": args.label, "runs": [r["name"] for r in runs], "scene": args.scene,
        "profile": args.profile, "prereg": args.prereg,
        "run_info": [{"name": r["name"], "cmd_src": r["cmd_src"], "tilt_deg": r["tilt"],
                      "n_excursions": len(r["ex"]), "n_dropped_gap": r["n_ex_dropped"],
                      "n_excluded": r["n_ex_excluded"], "sensors": r["sensors"]} for r in runs],
        "exclude": sorted(args.exclude),
        "summary": summ, "summary_alt_thr": summ_alt, "decisions": dec,
        "windows": {k: sc[k][1] for k in specs}, "excursions": {k: sc[k][0] for k in specs},
    }
    res["overlays"] = plot_overlays(runs, traces, out)
    (out / "eval.json").write_text(json.dumps(res, indent=1, default=float))
    write_md(res, out / "summary.md")
    print(json.dumps({"label": args.label, "run_info": res["run_info"],
                      "usable": dec["usable_thresholds"], "H_mag": dec["H_mag"],
                      "Bm_fam": dec["H_mag_Bm_fam"]["supported"], "H_inad": dec["H_inad"]["status"],
                      "H_vel": dec["H_vel"]["supported"]}, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
