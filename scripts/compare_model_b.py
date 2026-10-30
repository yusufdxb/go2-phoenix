#!/usr/bin/env python3
"""A vs B comparison of the frozen model-B replay runs, plus the pre-registered verdict.

Reads the eval runs written by scripts/actuator_replay.py (A_level, B_level, LOCK_level,
{A,B}_{pitch,roll}{+,-}, {A,B,LOCK}_reset10), writes compare.json / compare.md and
hw / A / B / target overlays. Never fits anything.

    PYTHONPATH=src python scripts/compare_model_b.py --eval logs/actuator_s2r/model_B/eval \
        --cache logs/actuator_s2r/baseline_A/cache --config configs/sim2real/actuator_model_B.yaml \
        --out logs/actuator_s2r/model_B/compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from phoenix.sim2real.actuator_replay import (
    JOINT_TYPES,
    breakaway,
    joint_type,
    load_cache,
    probe_ramps,
)
from phoenix.sim2real.go2_model import UNITREE_MOTOR_ORDER

TILTS = ("pitch+", "pitch-", "roll+", "roll-")
JOINT_KEYS = ("pos_rmse", "vel_rmse", "sse_diff", "stuck_agree", "dir_resid_split", "breakaway_abs_err_s",
              "err_at_break_hw", "err_at_break_sim")


def load(ev: Path, run: str) -> dict:
    return json.loads((ev / run / "metrics.json").read_text())["sessions"]


def type_mean(sess: dict, split: str, key: str) -> dict[str, float]:
    """Per type: mean over sessions of the session type mean (excluded joints dropped)."""
    out = {}
    for jt in JOINT_TYPES:
        v = [r["by_type"][key][jt] for r in sess.values() if r["split"] == split and jt in r["by_type"][key]]
        if v:
            out[jt] = float(np.mean(v))
    return out


def joint_rows(sess: dict, split: str) -> list[tuple[str, str, dict]]:
    rows = []
    for name, r in sess.items():
        if r["split"] != split:
            continue
        pj = r["metrics"]["per_joint"]
        for n in UNITREE_MOTOR_ORDER:
            if "excluded" not in pj[n]:
                rows.append((name, n, pj[n]))
    return rows


def pooled(rows, key: str, jt: str | None = None) -> float:
    v = [m[key] for _, n, m in rows if key in m and np.isfinite(m[key]) and (jt is None or joint_type(n) == jt)]
    return float(np.mean(v)) if v else float("nan")


def tau_resid(sess: dict, split: str) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for r in sess.values():
        if r["split"] != split:
            continue
        for n, p in r["probe_torque"].items():
            if "excluded" not in p:
                out.setdefault(joint_type(n), []).append(p["tau_sim_minus_tau_est_rmse_nm"])
    return {k: float(np.mean(v)) for k, v in out.items()}


def pct(a: float, b: float) -> float:
    return 100.0 * (b - a) / abs(a) if a else float("nan")


def verdict(ma: dict, mb: dict, ra, rb, orient: dict, acc: dict) -> dict:
    """Apply the pre-registered rule. 'pooled' = mean over probed held-out joints (literal
    reading); 'per_type' additionally requires every probed type to pass."""
    imp = {jt: -pct(ma["pos_rmse"][jt], mb["pos_rmse"][jt]) / 100 for jt in JOINT_TYPES}
    pos_reg = {jt: pct(ma["pos_rmse"][jt], mb["pos_rmse"][jt]) / 100 for jt in JOINT_TYPES}
    vel_reg = {jt: pct(ma["vel_rmse"][jt], mb["vel_rmse"][jt]) / 100 for jt in JOINT_TYPES}
    n_imp = sum(v >= acc["improve_frac"] for v in imp.values())
    no_reg = all(pos_reg[jt] <= 0.05 and vel_reg[jt] <= 0.10 for jt in JOINT_TYPES)
    timing = {"A": pooled(ra, "breakaway_abs_err_s"), "B": pooled(rb, "breakaway_abs_err_s")}
    timing["drop"] = 1 - timing["B"] / timing["A"]
    eab = {"hw": pooled(rb, "err_at_break_hw"), "B": pooled(rb, "err_at_break_sim")}
    eab["rel"] = eab["B"] / eab["hw"] - 1
    per_type = {}
    for jt in JOINT_TYPES:
        ta, tb = pooled(ra, "breakaway_abs_err_s", jt), pooled(rb, "breakaway_abs_err_s", jt)
        hw, sb = pooled(rb, "err_at_break_hw", jt), pooled(rb, "err_at_break_sim", jt)
        if np.isfinite(ta) and np.isfinite(tb):
            per_type[jt] = {"timing_drop": 1 - tb / ta, "err_at_break_rel": sb / hw - 1,
                            "pass": bool(1 - tb / ta >= 0.30 and abs(sb / hw - 1) <= 0.50)}
    stick_pooled = timing["drop"] >= 0.30 and abs(eab["rel"]) <= 0.50
    stick_types = stick_pooled and all(p["pass"] for p in per_type.values())
    orient_conf = any(orient[jt]["max_abs_change"] >= orient[jt]["a_to_b_abs"]
                      for jt in JOINT_TYPES if imp[jt] >= acc["improve_frac"])
    reject = n_imp == 0 or any(v > 0.10 for v in pos_reg.values())

    def call(stick: bool) -> str:
        if reject:
            return "REJECT"
        if n_imp >= 2 and no_reg and stick and not orient_conf:
            return "ACCEPT"
        return "INCONCLUSIVE"

    return {"pos_improvement": imp, "pos_change": pos_reg, "vel_change": vel_reg, "n_types_improved": n_imp,
            "no_regression": no_reg, "breakaway_timing": timing, "err_at_break": eab, "per_type_stick": per_type,
            "orientation_confounds": orient_conf, "verdict_pooled": call(stick_pooled),
            "verdict_per_type": call(stick_types)}


def ramp_table(ev: Path, cache: Path, sess: dict, run: str = "level") -> dict:
    """Post-hoc diagnostic (not pre-registered): per probe ramp, did hw / A / B break away
    at all? The registered timing metric only scores ramps where both hw and sim broke."""
    rows, counts = [], {}
    for name, r in sess.items():
        s = load_cache(cache / f"{name}.npz")
        q = {k: np.load(ev / f"{k}_{run}" / f"{name}_sim.npz")["q_sim"] for k in "AB"}
        excl = r["metrics"]["per_joint"]
        for j, ph, t0, t1, d in probe_ramps(s):
            n = UNITREE_MOTOR_ORDER[j]
            if "excluded" in excl[n]:
                continue
            br = {"hw": breakaway(s.q[:, j], s.target[:, j], s.t, t0, t1, d)}
            br.update({k: breakaway(q[k][:, j], s.target[:, j], s.t, t0, t1, d) for k in "AB"})
            rows.append({"session": name, "split": r["split"], "joint": n, "phase": ph,
                         **{f"{k}_t": v[0] for k, v in br.items()}, **{f"{k}_err": v[1] for k, v in br.items()}})
            c = counts.setdefault(f"{r['split']}/{joint_type(n)}/{ph}", {"ramps": 0, "hw": 0, "A": 0, "B": 0,
                                                                          "A_agree": 0, "B_agree": 0})
            c["ramps"] += 1
            hw = np.isfinite(br["hw"][0])
            c["hw"] += int(hw)
            for k in "AB":
                c[k] += int(np.isfinite(br[k][0]))
                c[f"{k}_agree"] += int(np.isfinite(br[k][0]) == hw)
    return {"rows": rows, "counts": dict(sorted(counts.items()))}


def overlays(ev: Path, cache: Path, out: Path, names: list[str]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    for name in names:
        s = load_cache(cache / f"{name}.npz")
        qa = np.load(ev / "A_level" / f"{name}_sim.npz")["q_sim"]
        qb = np.load(ev / "B_level" / f"{name}_sim.npz")["q_sim"]
        ramps: dict[int, list] = {}
        for j, ph, t0, t1, _ in probe_ramps(s):
            ramps.setdefault(j, []).append((ph, t0, t1))
        if not ramps:
            continue
        fig, axes = plt.subplots(len(ramps), 2, figsize=(14, 2.6 * len(ramps)), squeeze=False)
        for r, (j, rr) in enumerate(sorted(ramps.items())):
            t0, t1 = rr[0][1] - 0.5, rr[-1][2] + 0.5
            w = (s.t >= t0) & (s.t <= t1)
            n = UNITREE_MOTOR_ORDER[j]
            ax = axes[r, 0]
            ax.plot(s.t[w], s.target[w, j], color="0.55", lw=0.9, label="target (cmd)")
            ax.plot(s.t[w], s.q[w, j], color="#1f4e8c", lw=1.2, label="hw")
            ax.plot(s.t[w], qa[w, j], color="#c0502b", lw=0.9, ls="--", label="A")
            ax.plot(s.t[w], qb[w, j], color="#2e8b57", lw=0.9, ls="-.", label="B")
            ax.set_title(f"{name} {n}: q [rad]", fontsize=9)
            ax = axes[r, 1]
            ax.plot(s.t[w], s.target[w, j] - s.q[w, j], color="#1f4e8c", lw=1.2, label="hw")
            ax.plot(s.t[w], s.target[w, j] - qa[w, j], color="#c0502b", lw=0.9, ls="--", label="A")
            ax.plot(s.t[w], s.target[w, j] - qb[w, j], color="#2e8b57", lw=0.9, ls="-.", label="B")
            ax.set_title(f"{n}: target - q [rad] (probe ramps)", fontsize=9)
            for ph, a, b in rr:
                for x in (axes[r, 0], axes[r, 1]):
                    x.axvspan(a, b, color="#f2c14e" if ph == "rise" else "#9ecae1", alpha=0.15, lw=0)
            for x in axes[r]:
                x.tick_params(labelsize=7)
        axes[0, 0].legend(fontsize=7, loc="best")
        axes[0, 1].legend(fontsize=7, loc="best")
        fig.tight_layout()
        p = out / f"{name}_probe_hw_A_B.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        made.append(p.name)
    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    ev, out = Path(args.eval), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    acc = yaml.safe_load(Path(args.config).read_text())["acceptance"]
    runs = {k: load(ev, k) for k in ("A_level", "B_level", "LOCK_level", "A_reset10", "B_reset10", "LOCK_reset10")}
    for m in "AB":
        for tl in TILTS:
            runs[f"{m}_{tl}"] = load(ev, f"{m}_{tl}")
    res: dict = {"by_type": {}, "tau_resid": {}}
    for sp in ("train", "heldout"):
        res["by_type"][sp] = {k: {key: type_mean(runs[k], sp, key) for key in
                                  ("pos_rmse", "vel_rmse", "sse_diff", "stuck_agree", "dir_resid_split")}
                              for k in runs}
        res["tau_resid"][sp] = {k: tau_resid(runs[k], sp) for k in ("A_level", "B_level", "LOCK_level")}
    held = res["by_type"]["heldout"]
    ra, rb = joint_rows(runs["A_level"], "heldout"), joint_rows(runs["B_level"], "heldout")
    res["per_joint_heldout"] = [
        {"session": s, "joint": n, **{f"A_{k}": a.get(k) for k in JOINT_KEYS},
         **{f"B_{k}": b.get(k) for k in JOINT_KEYS}}
        for (s, n, a), (_, _, b) in zip(ra, rb, strict=True)]
    orient = {}
    for jt in JOINT_TYPES:
        ch = [abs(held[f"{m}_{tl}"]["pos_rmse"][jt] - held[f"{m}_level"]["pos_rmse"][jt]) for m in "AB" for tl in TILTS]
        orient[jt] = {"max_abs_change": float(max(ch)),
                      "a_to_b_abs": abs(held["B_level"]["pos_rmse"][jt] - held["A_level"]["pos_rmse"][jt])}
    res["orientation_heldout"] = orient
    res["acceptance"] = verdict(held["A_level"], held["B_level"], ra, rb, orient, acc)
    res["ramps_posthoc"] = ramp_table(ev, Path(args.cache), runs["A_level"])
    if (ev / "B_reset10_sim").is_dir():  # same replay, state reset every 10 s instead of 2 s
        res["ramps_posthoc_reset10"] = ramp_table(ev, Path(args.cache), runs["A_level"], "reset10_sim")
    res["overlays"] = overlays(ev, Path(args.cache), out,
                               [n for n, r in runs["A_level"].items() if r["split"] == "heldout"])
    (out / "compare.json").write_text(json.dumps(res, indent=1, default=float))

    lines = ["# Model B vs A (frozen B, no refit)", ""]
    for sp in ("heldout", "train"):
        b = res["by_type"][sp]
        lines += [f"## {sp}: per type (mean of session type means)", "",
              "| metric | type | A | B | B-A | % | LOCK | A reset10 | B reset10 | LOCK reset10 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for key in ("pos_rmse", "vel_rmse", "sse_diff", "stuck_agree", "dir_resid_split"):
            for jt in JOINT_TYPES:
                a, bb = b["A_level"][key].get(jt, np.nan), b["B_level"][key].get(jt, np.nan)
                lines.append(f"| {key} | {jt} | {a:.4f} | {bb:.4f} | {bb - a:+.4f} | {pct(a, bb):+.0f} | "
                         + " | ".join(f"{b[k][key].get(jt, np.nan):.4f}" for k in
                                      ("LOCK_level", "A_reset10", "B_reset10", "LOCK_reset10")) + " |")
        t = res["tau_resid"][sp]
        lines += ["", "tau_sim - tau_est RMSE on probe ramps [N m] (tau_est is the motor's current-based estimate):",
              "", "| type | A | B | LOCK |", "|---|---|---|---|"]
        lines += [f"| {jt} | {t['A_level'].get(jt, np.nan):.3f} | {t['B_level'].get(jt, np.nan):.3f} | "
              f"{t['LOCK_level'].get(jt, np.nan):.3f} |" for jt in JOINT_TYPES]
        lines.append("")
    lines += ["## held-out per joint", "",
          "| session | joint | pos A | pos B | vel A | vel B | break err A [s] | break err B [s] | "
          "err@break hw | A | B |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["per_joint_heldout"]:
        f = [r[k] if r[k] is not None else np.nan for k in
             ("A_pos_rmse", "B_pos_rmse", "A_vel_rmse", "B_vel_rmse", "A_breakaway_abs_err_s",
              "B_breakaway_abs_err_s", "B_err_at_break_hw", "A_err_at_break_sim", "B_err_at_break_sim")]
        lines.append(f"| {r['session'][-7:-1]} | {r['joint']} | " + " | ".join(f"{x:.4f}" for x in f) + " |")
    lines += ["", "## orientation sensitivity (held-out pos RMSE, rad)", "",
          "| type | max abs change over tilts (A and B) | abs A->B change |", "|---|---|---|"]
    lines += [f"| {jt} | {o['max_abs_change']:.5f} | {o['a_to_b_abs']:.5f} |" for jt, o in orient.items()]
    a = res["acceptance"]
    lines += ["", "## pre-registered acceptance (held out)", "",
          f"- pos RMSE change per type: {', '.join(f'{k} {100 * v:+.0f}%' for k, v in a['pos_change'].items())}",
          f"- vel RMSE change per type: {', '.join(f'{k} {100 * v:+.0f}%' for k, v in a['vel_change'].items())}",
          f"- types improved >= 10 %: {a['n_types_improved']}/3; no regression: {a['no_regression']}",
          f"- breakaway timing error, pooled over probed joints: A {a['breakaway_timing']['A']:.3f} s, "
          f"B {a['breakaway_timing']['B']:.3f} s, drop {100 * a['breakaway_timing']['drop']:.0f}% (need >= 30%)",
          f"- error at breakaway, pooled: hw {a['err_at_break']['hw']:.4f} rad, B {a['err_at_break']['B']:.4f} rad, "
          f"B/hw - 1 = {100 * a['err_at_break']['rel']:+.0f}% (need within 50%)"]
    lines += [f"- {jt}: timing drop {100 * p['timing_drop']:+.0f}%, err@break B/hw - 1 {100 * p['err_at_break_rel']:+.0f}%"
          f" -> {'pass' if p['pass'] else 'FAIL'}" for jt, p in a["per_type_stick"].items()]
    lines += [f"- orientation confounds an improved type: {a['orientation_confounds']}",
          f"- verdict, pooled reading: **{a['verdict_pooled']}**; per-type reading: **{a['verdict_per_type']}**",
          "", "## post-hoc diagnostic: did the joint break away on each probe ramp (not pre-registered)", "",
          "| split/type/phase | ramps | hw broke | A broke | B broke | A agrees w/ hw | B agrees w/ hw |",
          "|---|---|---|---|---|---|---|"]
    lines += [f"| {k} | {c['ramps']} | {c['hw']} | {c['A']} | {c['B']} | {c['A_agree']} | {c['B_agree']} |"
          for k, c in res["ramps_posthoc"]["counts"].items()]
    if "ramps_posthoc_reset10" in res:
        lines += ["", "same table with the state reset every 10 s (fewer resets inside probe ramps):", "",
              "| split/type/phase | ramps | hw broke | A broke | B broke | A agrees w/ hw | B agrees w/ hw |",
              "|---|---|---|---|---|---|---|"]
        lines += [f"| {k} | {c['ramps']} | {c['hw']} | {c['A']} | {c['B']} | {c['A_agree']} | {c['B_agree']} |"
              for k, c in res["ramps_posthoc_reset10"]["counts"].items()]
    lines += ["", "held-out ramps: breakaway time [s] / signed error at breakaway [rad]", "",
          "| session | joint | phase | hw | A | B |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['session'][-7:-1]} | {r['joint']} | {r['phase']} | "
          + " | ".join(f"{r[f'{k}_t']:.3f} / {r[f'{k}_err']:+.4f}" for k in ("hw", "A", "B")) + " |"
          for r in res["ramps_posthoc"]["rows"] if r["split"] == "heldout"]
    lines += ["", "overlays: " + ", ".join(res["overlays"])]
    (out / "compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-14:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
