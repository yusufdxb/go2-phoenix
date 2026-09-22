#!/usr/bin/env python3
"""Apply the frozen intervention-screening rules to the screening artifacts.

EXPERIMENT.md amendment 13 / docs/research/INTERVENTION_SCREENING.md section 5. Every
threshold here is quoted from that preregistration and none is computed from the data:
this script decides, it does not choose.

    PYTHONPATH=src python scripts/phoenix_v2_screen_analyze.py [screen_dir]

Writes ``screening.json`` (machine-readable verdict), ``screening.md`` (the reported
table) and ``dose_response.png`` next to the cells.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# --- the preregistered constants, section 5. Do not tune. ---------------------
MIN_DROP = 0.15  # qualification rule 1 and 2
MIN_REMAINING = 0.40  # rule 3, adaptation headroom
MAX_SAFETY_HOLD = 0.15  # rule 4
MAX_ATTITUDE = 0.30
MAX_ABORT_BAND = 0.0
MIN_FIDELITY = 0.90
MONOTONE_TOL = 0.05  # rule 5, allowed reversal between adjacent severities

FAMILIES = {
    "c1": ("C1 single joint", "RR_thigh", 1),
    "c2": ("C2 one leg", "leg_RR", 3),
    "c3": ("C3 both rear legs", "rear", 6),
    "c4": ("C4 global", "all", 12),
}


def _cell_key(name: str) -> tuple[str, float]:
    """``c2_leg_rr_0p85`` -> ``("c2", 0.85)``; the shared reference -> ``("nominal", 1.0)``."""
    if name == "nominal":
        return "nominal", 1.0
    fam = name.split("_", 1)[0]
    sev = name.rsplit("_", 1)[1]
    return fam, float(sev.replace("p", "."))


def load(screen: Path) -> dict[str, dict[int, dict[str, Any]]]:
    cells: dict[str, dict[int, dict[str, Any]]] = {}
    for d in sorted(screen.iterdir()):
        if not d.is_dir():
            continue
        for sd in sorted(d.glob("seed*")):
            f = sd / "summary.json"
            if f.is_file():
                cells.setdefault(d.name, {})[int(sd.name[4:])] = json.loads(f.read_text())
    return cells


def pooled(seeds: dict[int, dict[str, Any]], key: str) -> float:
    """Episode-weighted mean, so a short cell cannot outvote a long one."""
    num = sum(s[key] * s["num_robots"] for s in seeds.values())
    den = sum(s["num_robots"] for s in seeds.values())
    return num / den


def summarise(name: str, seeds: dict[int, dict[str, Any]], nominal: dict[int, dict[str, Any]]):
    fam, sev = _cell_key(name)
    any_s = next(iter(seeds.values()))
    deg = any_s.get("controlled_degradation") or {}
    succ = pooled(seeds, "walk2_success_rate")
    nom = pooled(nominal, "walk2_success_rate")
    per_seed = {
        sd: {
            "success": s["walk2_success_rate"],
            "drop": nominal[sd]["walk2_success_rate"] - s["walk2_success_rate"],
        }
        for sd, s in sorted(seeds.items())
        if sd in nominal
    }
    return {
        "cell": name,
        "family": fam,
        "family_label": FAMILIES[fam][0] if fam in FAMILIES else fam,
        "target": deg.get("joint") if deg else "nominal",
        "joints": deg.get("joints", []),
        "n_joints": len(deg.get("joints", [])) or (0 if fam == "nominal" else None),
        "severity": sev,
        "floor_applied": deg.get("min_scale"),
        "seeds": sorted(seeds),
        "episodes": sum(s["num_robots"] for s in seeds.values()),
        "walking_success": succ,
        "drop_vs_nominal": nom - succ,
        "per_seed": per_seed,
        "min_seed_drop": min((v["drop"] for v in per_seed.values()), default=float("nan")),
        "primary_score": pooled(seeds, "walk_primary_score_mean"),
        "fidelity_pass_rate": pooled(seeds, "fidelity_pass_rate"),
        "safety_hold_rate": pooled(seeds, "safety_hold_episode_rate"),
        "attitude_violation_rate": pooled(seeds, "attitude_violation_episode_rate"),
        "abort_band_rate": pooled(seeds, "abort_band_episode_rate"),
        "mean_lin_err_m_s": pooled(seeds, "walk2_mean_lin_err_m_s"),
        "mean_yaw_err_rad_s": pooled(seeds, "walk2_mean_yaw_err_rad_s"),
        "median_progress_ratio": pooled(seeds, "walk2_median_progress_ratio"),
        "p05_min_base_height_m": pooled(seeds, "walk2_p05_min_base_height_m"),
        "p95_max_joint_speed_rad_s": pooled(seeds, "walk2_p95_max_joint_speed_rad_s"),
        "failures_by_check": {
            k: sum(s["walk2_failures_by_check"][k] for s in seeds.values())
            for k in any_s["walk2_failures_by_check"]
        },
        "gate_faults": sorted({f for s in seeds.values() for f in (s.get("gate_faults") or [])}),
    }


def monotone(cells: list[dict[str, Any]]) -> tuple[bool, float]:
    """Rule 5, on one family: success must not RISE as severity deepens, beyond tol."""
    ordered = sorted(cells, key=lambda c: -c["severity"])  # mild -> severe
    worst = 0.0
    for a, b in zip(ordered, ordered[1:], strict=False):
        worst = max(worst, b["walking_success"] - a["walking_success"])
    return worst <= MONOTONE_TOL, worst


def qualify(c: dict[str, Any], family_ok: bool, worst: float) -> dict[str, Any]:
    rules = {
        "magnitude_pooled_drop_ge_0.15": c["drop_vs_nominal"] >= MIN_DROP,
        "reproducible_every_seed_drop_ge_0.15": c["min_seed_drop"] >= MIN_DROP,
        "headroom_success_ge_0.40": c["walking_success"] >= MIN_REMAINING,
        "not_catastrophic": (
            c["safety_hold_rate"] <= MAX_SAFETY_HOLD
            and c["attitude_violation_rate"] <= MAX_ATTITUDE
            and c["abort_band_rate"] <= MAX_ABORT_BAND
            and c["fidelity_pass_rate"] >= MIN_FIDELITY
        ),
        "family_monotone_within_0.05": family_ok,
    }
    return {**c, "rules": rules, "qualifies": all(rules.values()), "family_worst_reversal": worst}


def select(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Fewest joints, then least severe (largest scale), then smallest reversal."""
    ok = [r for r in rows if r["qualifies"]]
    if not ok:
        return None
    return sorted(ok, key=lambda r: (r["n_joints"], -r["severity"], r["family_worst_reversal"]))[0]


def plot(rows: list[dict[str, Any]], nom: float, out: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    for fam, (label, _t, nj) in FAMILIES.items():
        pts = sorted((r for r in rows if r["family"] == fam), key=lambda r: -r["severity"])
        if not pts:
            continue
        ax.plot(
            [p["severity"] for p in pts],
            [p["walking_success"] for p in pts],
            marker="o",
            label=f"{label} ({nj} joint{'s' if nj > 1 else ''})",
        )
    ax.axhline(nom, ls="--", lw=1, color="0.4")
    ax.axhline(nom - MIN_DROP, ls=":", lw=1.2, color="crimson")
    ax.text(
        0.505,
        nom - MIN_DROP + 0.012,
        f"sensitivity bar (drop {MIN_DROP})",
        color="crimson",
        fontsize=8,
    )
    ax.text(0.505, nom + 0.012, f"nominal {nom:.3f}", color="0.4", fontsize=8)
    ax.axhline(MIN_REMAINING, ls=":", lw=1.2, color="darkorange")
    ax.text(
        0.505,
        MIN_REMAINING + 0.012,
        f"headroom floor ({MIN_REMAINING})",
        color="darkorange",
        fontsize=8,
    )
    ax.invert_xaxis()
    ax.set_xlabel("actuator gain scale applied to the joint set (severity increases to the right)")
    ax.set_ylabel("walking success (score_walk_v2)")
    ax.set_title("Stage W intervention screening: dose-response, frozen W2")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out)
    return True


def main() -> int:
    screen = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phoenix_v2/intervention_screen")
    cells = load(screen)
    if "nominal" not in cells:
        print("no nominal cell; cannot compute a drop", file=sys.stderr)
        return 1
    nominal = cells.pop("nominal")
    nom_row = summarise("nominal", nominal, nominal)
    rows = [summarise(n, s, nominal) for n, s in sorted(cells.items())]

    fam_mono = {}
    for fam in FAMILIES:
        pts = [r for r in rows if r["family"] == fam]
        fam_mono[fam] = monotone(pts) if len(pts) > 1 else (True, 0.0)
    rows = [qualify(r, *fam_mono[r["family"]]) for r in rows]
    chosen = select(rows)

    held_out = []
    if chosen:
        sevs = sorted(
            {r["severity"] for r in rows if r["family"] == chosen["family"]}, reverse=True
        )
        i = sevs.index(chosen["severity"])
        held_out = [
            s
            for s in (sevs[i - 1] if i > 0 else None, sevs[i + 1] if i + 1 < len(sevs) else None)
            if s
        ]

    out = {
        "schema": "phoenix-v2-intervention-screen/v1",
        "prereg": "docs/research/INTERVENTION_SCREENING.md (EXPERIMENT.md amendment 13)",
        "thresholds": {
            "min_drop": MIN_DROP,
            "min_remaining": MIN_REMAINING,
            "max_safety_hold": MAX_SAFETY_HOLD,
            "max_attitude": MAX_ATTITUDE,
            "max_abort_band": MAX_ABORT_BAND,
            "min_fidelity": MIN_FIDELITY,
            "monotone_tol": MONOTONE_TOL,
        },
        "nominal": nom_row,
        "family_monotonicity": {
            k: {"ok": v[0], "worst_reversal": v[1]} for k, v in fam_mono.items()
        },
        "cells": rows,
        "selected": chosen,
        "held_out_severities": held_out,
        "secondary_global_qualifies": any(r["qualifies"] and r["family"] == "c4" for r in rows),
        "stop": chosen is None,
    }
    (screen / "screening.json").write_text(json.dumps(out, indent=1))

    nom = nom_row["walking_success"]
    lines = [
        "# Stage W intervention screening, result",
        "",
        f"Frozen W2, exact deploy stack, DR off, {nom_row['episodes']} nominal episodes "
        f"over seeds {nom_row['seeds']}. Preregistration: "
        "`docs/research/INTERVENTION_SCREENING.md` (amendment 13). Every threshold below "
        "was frozen before any cell ran.",
        "",
        f"**Nominal reference: walking success {nom:.4f}**, primary score "
        f"{nom_row['primary_score']:.4f}, fidelity {nom_row['fidelity_pass_rate']:.3f}.",
        "",
        "| family | joints | s | walk success | drop | min seed drop | prim score | fid | hold | att | prog | h05 | qualifies |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["family"], -r["severity"])):
        lines.append(
            f"| {r['family_label']} | {r['n_joints']} | {r['severity']:.2f} | "
            f"{r['walking_success']:.4f} | {r['drop_vs_nominal']:+.4f} | {r['min_seed_drop']:+.4f} | "
            f"{r['primary_score']:.4f} | {r['fidelity_pass_rate']:.3f} | {r['safety_hold_rate']:.3f} | "
            f"{r['attitude_violation_rate']:.3f} | {r['median_progress_ratio']:.3f} | "
            f"{r['p05_min_base_height_m']:.3f} | "
            + (
                "**YES**"
                if r["qualifies"]
                else ", ".join(k for k, v in r["rules"].items() if not v)
            )
            + " |"
        )
    lines += ["", "## Family monotonicity (rule 5, tolerance 0.05)", ""]
    for fam, (ok, worst) in fam_mono.items():
        if any(r["family"] == fam for r in rows):
            lines.append(
                f"* {FAMILIES[fam][0]}: worst reversal {worst:+.4f}, {'PASS' if ok else 'FAIL'}"
            )
    lines += ["", "## Selection", ""]
    if chosen:
        lines += [
            f"**{chosen['family_label']} at s = {chosen['severity']:.2f}** "
            f"({chosen['n_joints']} joints: {', '.join(chosen['joints'])}), floor "
            f"{chosen['floor_applied']}.",
            "",
            f"Walking success {chosen['walking_success']:.4f}, a drop of "
            f"{chosen['drop_vs_nominal']:.4f} from nominal (every seed at least "
            f"{chosen['min_seed_drop']:.4f}), with {chosen['walking_success']:.1%} of "
            "episodes still succeeding.",
            "",
            "Selected by the frozen rule: fewest affected joints among qualifying cells, "
            "then the least severe qualifying severity.",
            "",
            f"Held-out severities, reserved and not used for development: {held_out}.",
        ]
        if out["secondary_global_qualifies"] and chosen["family"] != "c4":
            lines += [
                "",
                "The global family also qualifies, so it is carried as the "
                "preregistered secondary intervention (section 5).",
            ]
    else:
        lines += [
            "**No cell qualifies. The Phoenix adaptation experiment stops.**",
            "",
            "Finding: no safe actuator intervention within the tested envelope produced "
            "the required measurable degradation. By the stop rule no floor is lowered, "
            "no family added, and the endpoint is not changed again.",
        ]
    (screen / "screening.md").write_text("\n".join(lines) + "\n")
    made = plot(rows, nom, screen / "dose_response.png")
    print("\n".join(lines))
    print(
        f"\nwrote {screen}/screening.json, screening.md" + (", dose_response.png" if made else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
