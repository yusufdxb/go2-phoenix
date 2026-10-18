"""Per-arm comparison and plots for an N-arm walking campaign, from gantry_ab_analyze.py output.

    python3 scripts/gantry_ab_analyze.py --campaign <campaign.jsonl> --out <dir> [--remap ...]
    python3 scripts/abc_compare.py --runs <dir>/runs.json --out <dir>/compare [--slot vx03_4s]

Uses only the analyzer's VALID first attempts (its validity rules are the preregistered
ones). Tracking errors per trial, all from recorded fields:

* ``F`` = tape forward / commanded forward (the primary measure; 1.0 = tracked), and
  ``fwd_err_m`` = commanded - tape (m).
* ``lat_tape_m``: tape lateral drift; ``yaw_drift_deg``: IMU yaw change on a straight command.
* ``F_odom``, ``odom_leg_speed_rmse_mps``: kinematic leg-odometry ESTIMATES, shown beside the
  tape and never in its place; ``odom_over_tape`` checks the estimate on every taped trial.

Arms are compared with A (the incumbent) by the difference of means and an exact two-sided
permutation p-value over all relabelings (with 4 trials per arm the smallest possible p is
2/70 = 0.029). The campaign is exploratory; the p-value describes these trials only.

Writes comparison.json, comparison.md and forward.png, drift.png, trials.png, odom_vs_tape.png.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

# Categorical slots 1-3 (validated: CVD dE 9.2, normal-vision 27.6; C is below 3:1 on the
# surface, so every arm is also direct-labelled and has its own marker).
ARM_STYLE = {"A": ("#2a78d6", "o"), "B": ("#eb6834", "s"), "C": ("#1baf7a", "D")}
EXTRA_STYLE = ("#eda100", "^")
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
METRICS = (
    "F",
    "fwd_err_m",
    "lat_tape_m",
    "yaw_drift_deg",
    "F_odom",
    "odom_leg_lat_m",
    "odom_leg_speed_rmse_mps",
    "odom_over_tape",
)


def trial_rows(runs: list[dict], slot: str) -> list[dict]:
    """One row per valid first attempt of ``slot`` (the analyzer's first_valid rule)."""
    seen, rows = set(), []
    for r in sorted(runs, key=lambda r: (r["block"], r.get("physical_attempt") or r.get("attempt") or 1)):
        if r.get("slot") != slot or not r.get("valid") or r["block"] in seen:
            continue
        seen.add(r["block"])
        w = r.get("walk") or {}
        cmd = w.get("commanded_fwd_m")
        tape = None if w.get("F") is None or cmd is None else w["F"] * cmd
        odom = w.get("odom_leg_fwd_m")
        rows.append(
            {
                "block": r["block"],
                "arm": r.get("measured_arm"),
                "run_dir": r.get("run_dir"),
                "answers_entered_by": r.get("answers_entered_by"),
                "soc_pre_pct": (r.get("battery") or {}).get("soc_pre_pct"),
                "power_v_median": (r.get("battery") or {}).get("power_v_median"),
                "commanded_fwd_m": cmd,
                "tape_fwd_m": tape,
                "F": w.get("F"),
                "fwd_err_m": None if tape is None else cmd - tape,
                "lat_tape_m": w.get("tape_lat_m"),
                "yaw_drift_deg": w.get("yaw_drift_deg"),
                "odom_leg_fwd_m": odom,
                "F_odom": w.get("F_odom"),
                "odom_leg_lat_m": w.get("odom_leg_lat_m"),
                "odom_leg_speed_rmse_mps": w.get("odom_leg_speed_rmse_mps"),
                "odom_over_tape": (odom / tape) if odom is not None and tape else None,
            }
        )
    return rows


def _spread(x: list[float]) -> dict[str, Any] | None:
    if not x:
        return None
    a = np.asarray(x, float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else None,
        "min": float(a.min()),
        "max": float(a.max()),
    }


def permutation_p(a: list[float], b: list[float]) -> float | None:
    """Exact two-sided permutation p-value for the difference of means (all relabelings)."""
    if len(a) < 2 or len(b) < 2:
        return None
    pool = np.asarray(a + b, float)
    obs = abs(np.mean(b) - np.mean(a))
    n, k = len(pool), len(a)
    hits = total = 0
    for idx in itertools.combinations(range(n), k):
        m = np.zeros(n, bool)
        m[list(idx)] = True
        total += 1
        hits += abs(pool[~m].mean() - pool[m].mean()) >= obs - 1e-12
    return hits / total


def compare(rows: list[dict], reference: str = "A") -> dict[str, Any]:
    arms = sorted({r["arm"] for r in rows if r["arm"]})
    vals = {a: {m: [r[m] for r in rows if r["arm"] == a and r[m] is not None] for m in METRICS} for a in arms}
    out: dict[str, Any] = {
        "per_arm": {a: {m: _spread(v) for m, v in ms.items()} for a, ms in vals.items()},
        "vs_reference": {},
    }
    if reference in vals:
        for a in arms:
            if a == reference:
                continue
            out["vs_reference"][a] = {
                m: {
                    "mean_diff": (float(np.mean(vals[a][m]) - np.mean(vals[reference][m]))
                                  if vals[a][m] and vals[reference][m] else None),
                    "perm_p_two_sided": permutation_p(vals[reference][m], vals[a][m]),
                    "n": [len(vals[reference][m]), len(vals[a][m])],
                }
                for m in ("F", "fwd_err_m", "lat_tape_m", "yaw_drift_deg", "F_odom")
            }
    return out


def _style(arm: str) -> tuple[str, str]:
    return ARM_STYLE.get(arm, EXTRA_STYLE)


def _axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def _strip(ax, rows: list[dict], arms: list[str], key: str, hollow: bool = False, dx: float = 0.0) -> None:
    for i, a in enumerate(arms):
        v = [r[key] for r in rows if r["arm"] == a and r[key] is not None]
        if not v:
            continue
        c, mk = _style(a)
        x = i + dx + np.linspace(-0.08, 0.08, len(v)) if len(v) > 1 else np.array([i + dx])
        ax.scatter(x, v, s=64, marker=mk, facecolor="none" if hollow else c, edgecolor=c if hollow else "white",
                   linewidths=1.6 if hollow else 1.5, zorder=3)
        if not hollow:
            ax.hlines(np.mean(v), i + dx - 0.2, i + dx + 0.2, color=INK, lw=2, zorder=4)


def make_plots(rows: list[dict], out: Path, slot: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    arms = sorted({r["arm"] for r in rows if r["arm"]})
    written = []
    names = [f"{a}\n(n={sum(1 for r in rows if r['arm'] == a and r['F'] is not None)} taped)" for a in arms]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    _strip(ax, rows, arms, "F", dx=-0.15)
    _strip(ax, rows, arms, "F_odom", hollow=True, dx=0.15)
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.text(len(arms) - 0.5, 1.0, " commanded", va="bottom", ha="right", color=MUTED, fontsize=8)
    ax.set_xticks(range(len(arms)), names)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_ylim(0, max([1.15] + [(r["F"] or 0) + 0.1 for r in rows] + [(r["F_odom"] or 0) + 0.1 for r in rows]))
    _axes(ax, f"Forward tracking, {slot}: distance walked / distance commanded", "fraction of commanded")
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=MUTED, label="tape (measured)"),
            Line2D([], [], marker="o", ls="", mfc="none", color=MUTED, label="leg odometry (estimate)"),
            Line2D([], [], color=INK, lw=2, label="mean of taped trials"),
        ],
        frameon=False, fontsize=8, loc="upper left",
    )
    fig.tight_layout()
    p = out / "forward.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, key, title, unit in (
        (axs[0], "lat_tape_m", "Lateral drift (tape)", "m, + left"),
        (axs[1], "yaw_drift_deg", "Heading drift (IMU)", "deg, + counter-clockwise"),
    ):
        _strip(ax, rows, arms, key)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(range(len(arms)), arms)
        ax.set_xlim(-0.6, len(arms) - 0.4)
        _axes(ax, title, unit)
    fig.tight_layout()
    p = out / "drift.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    fig, ax = plt.subplots(figsize=(8, 3.6))
    for r in rows:
        c, mk = _style(r["arm"])
        if r["soc_pre_pct"] is not None:
            ax.scatter(r["block"], r["soc_pre_pct"], s=64, marker=mk, color=c, edgecolor="white", zorder=3)
            ax.annotate(r["arm"], (r["block"], r["soc_pre_pct"]), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
    _axes(ax, "Trial order and battery before each walk", "state of charge, %")
    ax.set_xlabel("block (trial order)", color=MUTED, fontsize=9)
    fig.tight_layout()
    p = out / "trials.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    pairs = [r for r in rows if r["tape_fwd_m"] is not None and r["odom_leg_fwd_m"] is not None]
    if pairs:
        fig, ax = plt.subplots(figsize=(4.8, 4.4))
        hi = max(max(r["tape_fwd_m"] for r in pairs), max(r["odom_leg_fwd_m"] for r in pairs)) * 1.15
        ax.plot([0, hi], [0, hi], color=MUTED, lw=1, ls="--")
        for a in arms:
            pr = [r for r in pairs if r["arm"] == a]
            if pr:
                c, mk = _style(a)
                ax.scatter([r["tape_fwd_m"] for r in pr], [r["odom_leg_fwd_m"] for r in pr], s=64, marker=mk,
                           color=c, edgecolor="white", label=a, zorder=3)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_xlabel("tape forward, m", color=MUTED, fontsize=9)
        _axes(ax, "Leg odometry vs tape", "leg odometry forward, m")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        p = out / "odom_vs_tape.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(str(p))
    return written


def report(rows: list[dict], cmp: dict, slot: str, src: str) -> str:
    f = lambda v, d=3: "" if v is None else f"{v:.{d}f}"  # noqa: E731
    lines = [f"# Arm comparison, {slot} ({src})", "",
             "Tape = measured. Leg odometry = estimate (no-slip kinematics), never used in place of tape.", "",
             "| block | arm | SoC % | entered by | cmd m | tape m | F | fwd err m | lat m | yaw drift deg | odom m | F_odom | odom/tape |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['block']} | {r['arm']} | {f(r['soc_pre_pct'], 0)} | {r['answers_entered_by']} | {f(r['commanded_fwd_m'])} | "
            f"{f(r['tape_fwd_m'])} | {f(r['F'])} | {f(r['fwd_err_m'])} | {f(r['lat_tape_m'])} | {f(r['yaw_drift_deg'], 1)} | "
            f"{f(r['odom_leg_fwd_m'])} | {f(r['F_odom'])} | {f(r['odom_over_tape'])} |"
        )
    lines += ["", "## Per arm (mean, std, n)", "", "| arm | metric | mean | std | n |", "|---|---|---|---|---|"]
    for a, ms in cmp["per_arm"].items():
        for m, sp in ms.items():
            if sp:
                lines.append(f"| {a} | {m} | {f(sp['mean'])} | {f(sp['std'])} | {sp['n']} |")
    if cmp["vs_reference"]:
        lines += ["", "## Versus A (difference of means, exact permutation p)", "",
                  "| arm | metric | diff | p | n (A, arm) |", "|---|---|---|---|---|"]
        for a, ms in cmp["vs_reference"].items():
            for m, d in ms.items():
                lines.append(f"| {a} | {m} | {f(d['mean_diff'])} | {f(d['perm_p_two_sided'])} | {d['n']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, required=True, help="runs.json from gantry_ab_analyze.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slot", default="vx03_4s")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    runs = json.loads(args.runs.read_text())
    rows = trial_rows(runs, args.slot)
    cmp = compare(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    doc = {"slot": args.slot, "source": str(args.runs), "trials": rows, **cmp}
    if not args.no_plots and rows:
        doc["plots"] = make_plots(rows, args.out, args.slot)
    (args.out / "comparison.json").write_text(json.dumps(doc, indent=2, default=str))
    md = report(rows, cmp, args.slot, str(args.runs))
    (args.out / "comparison.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
