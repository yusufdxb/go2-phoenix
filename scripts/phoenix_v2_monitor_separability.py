#!/usr/bin/env python3
"""Can the residual estimator see this intervention at all, in the WALKING regime?

A DIAGNOSTIC, not a gate. It runs before the Phase H monitor design so that the design
is chosen against measured separability rather than assumed. It changes no thresholds and
makes no DEGRADED/NOMINAL decision.

The existing estimator (``phoenix.monitor.residual``) assumes that, under identical task
and load, the steady tracking error of the PD loop scales as ``1/s``. Standing satisfies
that. Walking may not: the commanded velocity is resampled, gait phase varies, and W2
commands bang-bang joint targets, so the nominal residual is partly the plant low-passing
a fast command rather than a shortfall of authority. This script measures, per joint and
aggregated over joints, how much of the nominal spread that costs and whether a degraded
session is still separable.

    PYTHONPATH=src python scripts/phoenix_v2_monitor_separability.py \
        --nominal results/phoenix_v2/intervention_screen/nominal \
        --degraded results/phoenix_v2/intervention_screen/c2_leg_rr_0p80 \
        --out results/phoenix_v2/monitor_separability

Calibration and evaluation nominal sessions are disjoint (the calibration half is held
out of the nominal evaluation set), so the nominal numbers are not self-fitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phoenix.monitor.layers import read_bridge_telemetry, tracking_pairs
from phoenix.monitor.residual import authority_ratio, calibrate, window_stats
from phoenix.sim2real.go2_model import UNITREE_MOTOR_ORDER

np.seterr(all="ignore")


def sessions(root: Path) -> list[Path]:
    return sorted(root.glob("seed*/bridge/robot*.jsonl"))


def stats_for(paths: list[Path]):
    return [window_stats(tracking_pairs(read_bridge_telemetry(p))) for p in paths]


#: The candidate hypothesis space for "which joints changed". It is exactly the set of
#: physically meaningful groups the intervention itself is drawn from
#: (:data:`phoenix.sim2real.degradation.JOINT_GROUPS`) plus the twelve singletons. It is
#: small, fixed in advance and interpretable: the monitor is NOT searching 2**12 subsets
#: for whichever one looks worst.
def _candidate_groups() -> dict[str, tuple[int, ...]]:
    from phoenix.sim2real.degradation import JOINT_GROUPS

    idx = {n: i for i, n in enumerate(UNITREE_MOTOR_ORDER)}
    groups = {name: tuple(idx[j] for j in js) for name, js in JOINT_GROUPS.items()}
    groups.update({n: (idx[n],) for n in UNITREE_MOTOR_ORDER})
    return groups


CANDIDATE_GROUPS = _candidate_groups()

#: Which candidate group each screening family actually degraded. Fixed by the screening
#: design, not chosen from the data.
TRUE_GROUP = {"c1": "RR_thigh_joint", "c2": "leg_RR", "c3": "rear", "c4": "all"}


def group_shift(s: np.ndarray, members: tuple[int, ...]) -> float:
    """Median over windows of the median over the group's joints of ``s_hat``.

    Averaging across a group is the whole point: a single joint's window-to-window
    spread in the walking regime is far larger than the effect being measured, because
    the commanded velocity and gait phase change the load. A group of size m cuts that
    noise roughly as 1/sqrt(m) while leaving a uniform reduction untouched.
    """
    return float(np.nanmedian(np.nanmedian(s[:, list(members)], axis=1)))


def session_scores(st, baseline) -> dict[str, object]:
    """Per-session summary of the per-joint ratio and its across-joint aggregate.

    ``global_shift`` is the median over the twelve joints of each window's ``s_hat``,
    then the median over windows. Under a uniform reduction every joint moves together,
    so the aggregate is an estimate of the applied scale with about 1/sqrt(12) the
    per-joint noise; under a sparse change it stays near 1 and the per-joint values
    carry the signal. That is the statistic the Phase H design has to live or die on.
    """
    s = authority_ratio(st, baseline)  # (windows, 12)
    per_joint = np.nanmedian(s, axis=0)
    per_window_global = np.nanmedian(s, axis=1)
    return {
        "windows": int(s.shape[0]),
        "per_joint_median": [float(v) for v in per_joint],
        "global_shift": float(np.nanmedian(per_window_global)),
        "global_p2p5": float(np.nanpercentile(per_window_global, 2.5)),
        "global_p97p5": float(np.nanpercentile(per_window_global, 97.5)),
        "min_joint_median": float(np.nanmin(per_joint)),
        "argmin_joint": UNITREE_MOTOR_ORDER[int(np.nanargmin(per_joint))],
        "group_shift": {g: group_shift(s, m) for g, m in CANDIDATE_GROUPS.items()},
    }


def separation(a: list[float], b: list[float]) -> dict[str, float]:
    """Rank separation of two session-level score sets, threshold-free.

    AUROC for "degraded scores lower than nominal", plus the plain overlap.
    """
    nom, deg = np.asarray(a, float), np.asarray(b, float)
    wins = (deg[:, None] < nom[None, :]).sum() + 0.5 * (deg[:, None] == nom[None, :]).sum()
    return {
        "auroc_degraded_below_nominal": float(wins / (len(nom) * len(deg))),
        "nominal_min": float(nom.min()),
        "degraded_max": float(deg.max()),
        "separable_without_overlap": bool(deg.max() < nom.min()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominal", type=Path, required=True)
    ap.add_argument("--degraded", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calib-fraction", type=float, default=0.5)
    a = ap.parse_args()

    nom = sessions(a.nominal)
    if len(nom) < 4:
        print(f"need at least 4 nominal sessions, found {len(nom)}")
        return 1
    k = max(2, int(round(len(nom) * a.calib_fraction)))
    calib_paths, eval_paths = nom[:k], nom[k:]
    base = calibrate(stats_for(calib_paths), regime="walk")

    nom_rows = [session_scores(s, base) for s in stats_for(eval_paths)]
    out: dict[str, object] = {
        "schema": "phoenix-v2-monitor-separability/v1",
        "note": "DIAGNOSTIC ONLY. No gate, no threshold, no decision.",
        "regime": "walk",
        "calibration_sessions": [str(p) for p in calib_paths],
        "nominal_eval_sessions": [str(p) for p in eval_paths],
        "baseline_reference_rms": [float(v) for v in base.reference_rms],
        "nominal": nom_rows,
        "nominal_global_shift": [r["global_shift"] for r in nom_rows],
        "conditions": {},
    }
    ng = [float(r["global_shift"]) for r in nom_rows]
    print(f"calibrated on {len(calib_paths)} sessions, evaluating {len(eval_paths)} nominal")
    print(f"nominal global shift: median {np.median(ng):.3f} range {min(ng):.3f}-{max(ng):.3f}")

    for d in a.degraded:
        rows = [session_scores(s, base) for s in stats_for(sessions(d))]
        if not rows:
            print(f"{d.name}: no telemetry, skipped")
            continue
        dg = [float(r["global_shift"]) for r in rows]
        sep = separation(ng, dg)
        # Which candidate group separates this condition best, and does the ranking
        # actually point at the joints that were degraded?
        by_group = {}
        for g in CANDIDATE_GROUPS:
            gn = [float(r["group_shift"][g]) for r in nom_rows]
            gd = [float(r["group_shift"][g]) for r in rows]
            by_group[g] = {
                **separation(gn, gd),
                "nominal_median": float(np.median(gn)),
                "degraded_median": float(np.median(gd)),
            }
        # The honest number is the PRE-SPECIFIED group, the one that was actually
        # degraded. The argmax over ~24 candidates is reported too, but only to show
        # how often a wrong group wins by chance: selecting it on this same data is
        # optimistic and is never a detection result.
        true_g = TRUE_GROUP.get(d.name.split("_", 1)[0])
        best = max(by_group, key=lambda g: by_group[g]["auroc_degraded_below_nominal"])
        out["conditions"][d.name] = {
            "sessions": rows,
            "global_shift": dg,
            "separation": sep,
            "by_group": by_group,
            "true_group": true_g,
            "argmax_group": best,
            "argmax_is_true_group": best == true_g,
        }
        t = by_group.get(true_g, {})
        print(
            f"{d.name:22s} global {np.median(dg):.3f} (AUROC {sep['auroc_degraded_below_nominal']:.3f})"
            f" | TRUE {true_g:14s} {t.get('nominal_median', float('nan')):.3f}->"
            f"{t.get('degraded_median', float('nan')):.3f} "
            f"AUROC {t.get('auroc_degraded_below_nominal', float('nan')):.3f} "
            f"clean {t.get('separable_without_overlap')}"
            f" | argmax {best}{'' if best == true_g else ' (WRONG)'}"
        )

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "separability.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}/separability.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
