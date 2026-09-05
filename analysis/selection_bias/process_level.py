"""Process-level (mean-of-process-means) inference for the Phoenix replication.

The frozen bootstrap in ``replication._bootstrap_mean`` resamples BLOCKS within
each process and never resamples PROCESSES, so every interval it reports is
conditional on the process seeds actually run. Blocks inside one process share a
policy load, a physics batch and a process RNG, so they are not independent
replicates of the process. This script reports the other level: the process is
the unit, the estimate is the mean of the per-process means, and the interval
comes from the between-process spread.

Two interval flavours, both reported so nobody has to take one on faith:
  - Student t on the n process means (what the paper quotes; exact under
    normality of the process means, conservative at small n)
  - BCa-free percentile bootstrap RESAMPLING PROCESSES (distribution-free, but
    at n<=5 its percentiles are coarse; reported as a cross-check only)

Everything is a pure function of blocks.csv / envs.csv, which build_frame.py
regenerates from the frozen arm artifacts. Nothing here reads the audit JSONs,
so it is an independent recomputation of the registered quantities, not a
re-print of them.

Writes process_level.json next to this file. The paper renders from that file.
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
W_COMMON = 300  # 500 - max onset 200: the longest window every block supplies in full
N_BOOT = 20_000
SEED = 20260905


def _t_interval(process_means: np.ndarray) -> dict:
    """Student t interval on the process means. n is the number of PROCESSES."""
    x = np.asarray(process_means, dtype=float)
    n = len(x)
    point = float(x.mean())
    if n < 2:
        return {"n_processes": n, "point_pp": point * 100, "lo_pp": None, "hi_pp": None,
                "excludes_zero": None, "process_effects_pp": (x * 100).tolist()}
    se = float(x.std(ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.975, n - 1)) * se
    lo, hi = point - half, point + half
    return {
        "n_processes": n,
        "point_pp": point * 100,
        "lo_pp": lo * 100,
        "hi_pp": hi * 100,
        "excludes_zero": bool(lo * hi > 0),
        "process_effects_pp": (x * 100).tolist(),
    }


def _process_bootstrap(process_means: np.ndarray, seed: int = SEED) -> dict:
    x = np.asarray(process_means, dtype=float)
    n = len(x)
    if n < 2:
        return {"lo_pp": None, "hi_pp": None, "excludes_zero": None}
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"lo_pp": float(lo) * 100, "hi_pp": float(hi) * 100,
            "excludes_zero": bool(lo * hi > 0)}


def _block_bootstrap(groups: list[np.ndarray], seed: int = SEED) -> dict:
    """The frozen convention: resample blocks within process, never processes.

    Reproduced here only so the two levels sit in one table and the difference
    between them is visible rather than asserted.
    """
    groups = [np.asarray(g, dtype=float) for g in groups if len(g)]
    if not groups:
        return {"point_pp": None, "lo_pp": None, "hi_pp": None, "excludes_zero": None}
    rng = np.random.default_rng(seed)
    point = float(np.concatenate(groups).mean())
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        draws[i] = np.concatenate(
            [g[rng.integers(0, len(g), len(g))] for g in groups]
        ).mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point_pp": point * 100, "lo_pp": float(lo) * 100, "hi_pp": float(hi) * 100,
            "excludes_zero": bool(lo * hi > 0)}


def _summarize(frame: pd.DataFrame, column: str = "effect") -> dict:
    """Both inferential levels for one group of blocks."""
    per_process = frame.groupby("replicate")[column].mean().sort_index()
    out = {"n_blocks": int(frame[column].notna().sum())}
    out.update(_t_interval(per_process.to_numpy()))
    out["process_bootstrap"] = _process_bootstrap(per_process.to_numpy())
    out["block_bootstrap"] = _block_bootstrap(
        [g[column].dropna().to_numpy() for _, g in frame.groupby("replicate")]
    )
    out["process_ids"] = per_process.index.tolist()
    return out


def _paired_contrast(frame: pd.DataFrame, left: str, right: str, key: str) -> dict:
    """obs-minus-motor style contrast, paired WITHIN process so the contrast has
    the same n as the cells it is built from."""
    a = frame[frame[key] == left].groupby("replicate")["effect"].mean()
    b = frame[frame[key] == right].groupby("replicate")["effect"].mean()
    joined = pd.concat([a, b], axis=1, keys=["l", "r"]).dropna()
    diff = (joined["l"] - joined["r"]).sort_index()
    out = _t_interval(diff.to_numpy())
    out["process_bootstrap"] = _process_bootstrap(diff.to_numpy())
    out["contrast"] = f"{left} minus {right}"
    return out


def _effects_at_window(envs: pd.DataFrame, W: int) -> pd.DataFrame:
    """Recompute the registered estimand on a common-length post-onset window.

    Identical eligibility rule (drop a pair if either arm fell pre-onset) and
    identical unit (the block); only the outcome window is truncated, so the
    comparison isolates exposure length.
    """
    x = envs[envs.disturbed & envs.eligible].copy()
    u = (x.u_falltick >= x.onset) & (x.u_falltick < x.onset + W) & (x.u_falltick >= 0)
    o = (x.o_falltick >= x.onset) & (x.o_falltick < x.onset + W) & (x.o_falltick >= 0)
    x["uW"], x["oW"] = u, o
    g = (x.groupby(["cell", "replicate", "block_id", "leakfree"])
           .agg(uW=("uW", "mean"), oW=("oW", "mean")).reset_index())
    g["effect"] = g.uW - g.oW
    g["family"] = np.where(g.cell.str.endswith("_obs"), "obs", "motor")
    return g


def _onset_sensitivity(disturbed: pd.DataFrame) -> dict:
    """Effect modification by onset tick, per cell, at the process level.

    Selection into the leak-free subset is exactly 1{onset <= per-pair
    threshold}, so effect modification by onset is the ONLY route by which a
    deterministic pre-treatment selection rule can bias the subset estimate.
    Slope is fit within process and averaged over processes, so between-process
    structure cannot masquerade as a within-process trend.
    """
    out = {}
    for cell, g in disturbed.groupby("cell"):
        slopes = []
        for _, gg in g.groupby("replicate"):
            gg = gg.dropna(subset=["effect"])
            if len(gg) < 3 or gg.onset.nunique() < 2:
                continue
            slopes.append(np.polyfit(gg.onset.to_numpy(), gg.effect.to_numpy(), 1)[0])
        slopes = np.asarray(slopes, dtype=float)
        iv = _t_interval(slopes)  # units: effect-fraction per tick
        lf, dv = g[g.leakfree], g[~g.leakfree]
        shift = float(dv.onset.mean() - lf.onset.mean())
        out[cell] = {
            "slope_pp_per_tick": iv["point_pp"],
            "slope_lo_pp_per_tick": iv["lo_pp"],
            "slope_hi_pp_per_tick": iv["hi_pp"],
            "n_processes": iv["n_processes"],
            "mean_onset_shift_divergent_minus_leakfree_ticks": shift,
            "implied_extrapolation_bias_pp": None if iv["point_pp"] is None
                else iv["point_pp"] * shift,
            "implied_bias_bound_pp": None if iv["lo_pp"] is None else [
                min(iv["lo_pp"] * shift, iv["hi_pp"] * shift),
                max(iv["lo_pp"] * shift, iv["hi_pp"] * shift),
            ],
        }
    return out


def _threshold_sweep(disturbed: pd.DataFrame) -> dict:
    """Ignore the fitted per-pair threshold; sweep an arbitrary onset cut.

    If the leak-free result is an artifact of where the threshold happened to
    land, the answer should move as the cut moves.
    """
    out = {}
    for cell, g in disturbed.groupby("cell"):
        row = {}
        for q in range(105, 205, 10):
            s = g[g.onset <= q]
            per = s.groupby("replicate").effect.mean()
            row[str(q)] = {
                "effect_pp": float(per.mean() * 100) if len(per) else None,
                "n_blocks": int(len(s)),
                "n_processes": int(per.notna().sum()),
            }
        out[cell] = row
    return out


def _residual_counts(envs: pd.DataFrame) -> dict:
    """Pre-onset fall-status discrepancies: the residual contamination budget.

    Reported on the disturbed env pairs the registered estimand actually uses,
    and on all pairs, because quoting one for the other is exactly the error
    this file exists to stop.
    """
    diff = envs.u_pre != envs.o_pre
    dist = envs[envs.disturbed]
    dist_diff = dist.u_pre != dist.o_pre
    lf_leak = envs[envs.leakfree & envs.disturbed]
    return {
        "env_pairs_all": int(len(envs)),
        "env_pairs_disturbed": int(len(dist)),
        "pre_onset_fall_status_differs_all": int(diff.sum()),
        "pre_onset_fall_status_differs_disturbed": int(dist_diff.sum()),
        "blocks_affected_disturbed": int(
            dist[dist_diff].groupby(["replicate", "cell", "block_id"]).ngroups
        ),
        "leakfree_disturbed_pairs_with_discrepancy": int(
            (lf_leak.u_pre != lf_leak.o_pre).sum()
        ),
        "by_cell_disturbed": dist[dist_diff].cell.value_counts().to_dict(),
    }


def main() -> None:
    blocks = pd.read_csv(HERE / "blocks.csv")
    envs = pd.read_csv(HERE / "envs.csv")
    blocks["family"] = np.where(blocks.cell.str.endswith("_obs"), "obs", "motor")
    d = blocks[blocks.disturbed].copy()
    processes = sorted(d.replicate.unique())

    result: dict = {
        "n_processes": len(processes),
        "processes": processes,
        "n_blocks_total": int(len(blocks)),
        "n_blocks_disturbed": int(len(d)),
        "common_window_ticks": W_COMMON,
        "note": (
            "point_pp / lo_pp / hi_pp are the PROCESS-level (mean-of-process-means, "
            "Student t on n processes) estimate and interval. block_bootstrap is the "
            "frozen within-process block bootstrap, reproduced for comparison only. "
            "The two answer different questions and must not be mixed in one table."
        ),
    }

    # --- primary: four cells, full sample and leak-free subset -----------------
    result["cells_full"] = {c: _summarize(g) for c, g in d.groupby("cell")}
    result["cells_leakfree"] = {
        c: _summarize(g[g.leakfree]) for c, g in d.groupby("cell")
    }
    result["cells_divergent"] = {
        c: _summarize(g[~g.leakfree]) for c, g in d.groupby("cell")
    }

    # --- registered gate level: fault family ----------------------------------
    result["families_full"] = {f: _summarize(g) for f, g in d.groupby("family")}
    result["families_leakfree"] = {
        f: _summarize(g[g.leakfree]) for f, g in d.groupby("family")
    }
    result["interaction_full"] = _paired_contrast(d, "obs", "motor", "family")
    result["interaction_leakfree"] = _paired_contrast(
        d[d.leakfree], "obs", "motor", "family"
    )

    # --- pre-onset negative control -------------------------------------------
    result["pre_onset_control_cells"] = {
        c: _summarize(g, column="pre_effect") for c, g in d.groupby("cell")
    }

    # --- common-window W = 300 -------------------------------------------------
    gW = _effects_at_window(envs, W_COMMON)
    result["cells_full_W300"] = {c: _summarize(g) for c, g in gW.groupby("cell")}
    result["cells_leakfree_W300"] = {
        c: _summarize(g[g.leakfree]) for c, g in gW.groupby("cell")
    }
    result["families_full_W300"] = {f: _summarize(g) for f, g in gW.groupby("family")}

    # --- selection diagnostics -------------------------------------------------
    result["onset_sensitivity"] = _onset_sensitivity(d)
    result["threshold_sweep"] = _threshold_sweep(d)
    result["leakfree_counts"] = {
        c: {"leakfree": int(g.leakfree.sum()), "total": int(len(g))}
        for c, g in d.groupby("cell")
    }
    result["residual"] = _residual_counts(envs)

    (HERE / "process_level.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    # --- human-readable echo ---------------------------------------------------
    def line(name: str, s: dict) -> str:
        if s.get("lo_pp") is None:
            return "  %-26s %+7.2f  [process interval undefined at n=%d]" % (
                name, s["point_pp"], s["n_processes"])
        bb = s.get("block_bootstrap", {})
        blk = ("   block-boot %+7.2f [%+7.2f,%+7.2f]"
               % (bb["point_pp"], bb["lo_pp"], bb["hi_pp"])) if bb.get("lo_pp") is not None else ""
        return "  %-26s %+7.2f [%+7.2f,%+7.2f] n_proc=%d n_blk=%-3d %s%s" % (
            name, s["point_pp"], s["lo_pp"], s["hi_pp"], s["n_processes"],
            s.get("n_blocks", -1), "EXCL 0" if s["excludes_zero"] else "*** INCLUDES 0 ***", blk)

    print("processes: %d %s" % (len(processes), processes))
    print("\n=== PRIMARY, four cells, FULL sample (process level) ===")
    for c, s in result["cells_full"].items():
        print(line(c, s))
    print("\n=== FAULT FAMILY (the level the registered gate criterion is defined on) ===")
    for f, s in result["families_full"].items():
        print(line(f, s))
    print(line("interaction obs-motor", result["interaction_full"]))
    print("\n=== LEAK-FREE SUBSET, four cells (process level) ===")
    for c, s in result["cells_leakfree"].items():
        print(line(c, s))
    print("\n=== LEAK-FREE SUBSET, fault family ===")
    for f, s in result["families_leakfree"].items():
        print(line(f, s))
    print(line("interaction obs-motor", result["interaction_leakfree"]))
    print("\n=== PRE-ONSET NEGATIVE CONTROL (must include zero) ===")
    for c, s in result["pre_onset_control_cells"].items():
        print(line(c, s))
    print("\n=== COMMON WINDOW W=%d, full sample ===" % W_COMMON)
    for c, s in result["cells_full_W300"].items():
        print(line(c, s))
    print("\n=== COMMON WINDOW W=%d, fault family ===" % W_COMMON)
    for f, s in result["families_full_W300"].items():
        print(line(f, s))
    print("\n=== ONSET SENSITIVITY (effect modification by onset tick) ===")
    for c, s in result["onset_sensitivity"].items():
        b = s["implied_bias_bound_pp"]
        print("  %-12s slope=%+.4f pp/tick [%s]  onset shift=%+.1f ticks  implied bias=%+.2f pp%s"
              % (c, s["slope_pp_per_tick"],
                 "n/a" if s["slope_lo_pp_per_tick"] is None
                 else "%+.4f,%+.4f" % (s["slope_lo_pp_per_tick"], s["slope_hi_pp_per_tick"]),
                 s["mean_onset_shift_divergent_minus_leakfree_ticks"],
                 s["implied_extrapolation_bias_pp"],
                 "" if b is None else "  bound [%+.2f,%+.2f]" % (b[0], b[1])))
    print("\n=== RESIDUAL CONTAMINATION ===")
    for k, v in result["residual"].items():
        print("  %-46s %s" % (k, v))
    print("\nwrote %s" % (HERE / "process_level.json"))


if __name__ == "__main__":
    main()
