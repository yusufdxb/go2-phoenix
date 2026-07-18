"""Offline evaluation of the reliability shield on Phase 3 rollouts.

Answers the real question, not the plumbing one: does a policy-latent OOD
score give useful, statistically defensible early warning of behavioral
failure, and does it beat simple baselines?

Pipeline (pure numpy + the phoenix.reliability package; no Isaac):
  1. Load rollout .npz per condition/seed. The failure oracle is the env's
     own termination that is NOT a timeout (fall = done & ~time_out) --
     independent of any monitor score.
  2. Segment each parallel env's timeline into episodes at done frames;
     drop the first WARMUP frames of each episode (spawn-drop transient).
  3. Fit every detector on NOMINAL data only (a fit split); calibrate its
     operating threshold on a held-out nominal split (nominal-only FPR).
  4. Detectors compared: latent-Mahalanobis (the shield), obs-Mahalanobis,
     observation magnitude, critic value signal, action saturation, and a
     random baseline.
  5. Per OOD condition/severity/seed: detection AUROC/AUPRC + FPR@op,
     detection lead time before fall, intervention success (shield engages
     before the fall), and unnecessary-fallback rate on nominal. Seed
     spread -> confidence intervals.
  6. Save results.json, a markdown report, and plots. Fails closed on
     missing/empty inputs; never fabricates a positive result.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from phoenix.reliability.arbiter import SimplexArbiter, SimplexArbiterCfg
from phoenix.reliability.metrics import average_precision, roc_auc
from phoenix.reliability.ood_monitor import MahalanobisScorer
from phoenix.reliability.runtime import ShieldRuntime, calibrate_arbiter_thresholds

DT = 0.02  # control period (s): flat env decimation 4 * sim dt 0.005 = 50 Hz
WARMUP = 15  # frames dropped after each reset (spawn-drop transient)
TARGET_FPR = 0.05  # nominal-only operating point
SEVERITY_ORDER = {"moderate": 0, "severe": 1}


# --------------------------------------------------------------------------
# Loading + episode segmentation
# --------------------------------------------------------------------------
def _cond_of(path: str) -> tuple[str, int]:
    base = Path(path).stem
    m = re.match(r"(.+)_seed(\d+)$", base)
    return m.group(1), int(m.group(2))


def load_grid(raw_dir: str) -> dict:
    """Return {condition: {seed: npz-dict}} for all rollouts found."""
    out: dict = defaultdict(dict)
    files = sorted(glob.glob(f"{raw_dir}/*.npz"))
    for f in files:
        if Path(f).stem in ("smoke_nominal", "nohooks", "clean"):
            continue
        cond, seed = _cond_of(f)
        out[cond][seed] = dict(np.load(f))
    return out


def episode_spans(done_col: np.ndarray):
    """Yield (start, end_inclusive) index spans between resets for one env."""
    start = 0
    T = len(done_col)
    for t in range(T):
        if done_col[t]:
            yield (start, t)
            start = t + 1
    if start < T:
        yield (start, T - 1)


def steady_mask(d: dict) -> np.ndarray:
    """(T, N) bool mask: True on frames >= WARMUP steps into their episode."""
    done = d["done"]
    T, N = done.shape
    mask = np.zeros((T, N), dtype=bool)
    for n in range(N):
        for s, e in episode_spans(done[:, n]):
            mask[min(s + WARMUP, e + 1) : e + 1, n] = True
    return mask


# --------------------------------------------------------------------------
# Per-frame detector scores (higher = more out-of-distribution / trouble)
# --------------------------------------------------------------------------
def detector_scores(d: dict, scorers: dict) -> dict:
    """Return {method: (T, N) score array} for one rollout."""
    T, N = d["done"].shape
    lat = d["latent"].reshape(T * N, -1)
    obs = d["obs"].reshape(T * N, -1)
    out = {}
    out["latent_maha"] = scorers["latent"].score(lat).reshape(T, N)
    out["obs_maha"] = scorers["obs"].score(obs).reshape(T, N)
    out["obs_magnitude"] = np.linalg.norm(obs, axis=1).reshape(T, N)
    val = d["value"]
    out["value_signal"] = (-val).reshape(T, N)  # low value = trouble
    act = d["action"]
    out["action_sat"] = np.mean(np.abs(act) > 0.98, axis=2)  # fraction saturated
    rng = np.random.default_rng(0)
    out["random"] = rng.standard_normal((T, N))
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def fall_frames(d: dict) -> np.ndarray:
    return d["done"] & (~d["time_out"])


def lead_times(score_col, fall_col, done_col, thr):
    """Per failing episode: seconds between first threshold cross and the fall.

    Positive = warned early; 0/negative = warned at/after onset; None-miss =
    never crossed before the fall. Returns (lead_list, n_fail, n_warned).
    """
    leads, n_fail, n_warned = [], 0, 0
    for s, e in episode_spans(done_col):
        seg_fall = fall_col[s : e + 1]
        if not seg_fall.any():
            continue
        onset = int(np.argmax(seg_fall))  # first fall frame in segment
        n_fail += 1
        seg_score = score_col[s : e + 1]
        crossed = np.flatnonzero(seg_score[: onset + 1] > thr)
        if crossed.size and crossed[0] < onset:
            leads.append((onset - crossed[0]) * DT)
            n_warned += 1
    return leads, n_fail, n_warned


def unnecessary_fallback_rate(d: dict, scorer, arb_cfg, feat_key, mask) -> float:
    """Fraction of nominal steady episodes where the shield engages (false alarm)."""
    rt = ShieldRuntime(scorer, SimplexArbiter(arb_cfg), feature_key=feat_key)
    done = d["done"]
    T, N = done.shape
    feat = d["latent"] if feat_key == "latent" else d["obs"]
    engaged_eps, total_eps = 0, 0
    for n in range(N):
        for s, e in episode_spans(done[:, n]):
            if e - s < WARMUP:
                continue
            rt.reset()
            eng = False
            for t in range(s, e + 1):
                dec = rt.step(feat[t, n])
                if t >= s + WARMUP and dec.engaged:
                    eng = True
            total_eps += 1
            engaged_eps += int(eng)
    return engaged_eps / max(total_eps, 1)


def intervention_success(d: dict, scorer, arb_cfg, feat_key) -> tuple[int, int]:
    """Falling episodes where the shield engages strictly before the fall onset."""
    rt = ShieldRuntime(scorer, SimplexArbiter(arb_cfg), feature_key=feat_key)
    done, fall = d["done"], fall_frames(d)
    T, N = done.shape
    feat = d["latent"] if feat_key == "latent" else d["obs"]
    n_fail, n_saved = 0, 0
    for n in range(N):
        for s, e in episode_spans(done[:, n]):
            seg_fall = fall[s : e + 1]
            if not seg_fall.any():
                continue
            onset = s + int(np.argmax(seg_fall))
            n_fail += 1
            rt.reset()
            engaged_before = False
            for t in range(s, e + 1):
                dec = rt.step(feat[t, n])
                if t < onset and dec.engaged:
                    engaged_before = True
                    break
            n_saved += int(engaged_before)
    return n_saved, n_fail


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="reliability_eval/raw")
    ap.add_argument("--out-dir", default="reliability_eval/results")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    grid = load_grid(args.raw_dir)
    if "nominal" not in grid or not grid["nominal"]:
        raise SystemExit("FAIL-CLOSED: no nominal rollouts found; cannot calibrate.")
    ood_conds = sorted(c for c in grid if c != "nominal")
    if not ood_conds:
        raise SystemExit("FAIL-CLOSED: no OOD rollouts found.")
    seeds = sorted(grid["nominal"].keys())
    print(f"[eval] conditions={list(grid)} seeds={seeds}")

    methods = ["latent_maha", "obs_maha", "obs_magnitude", "value_signal", "action_sat", "random"]
    per_seed: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    fall_rates: dict = defaultdict(dict)

    for seed in seeds:
        nom = grid["nominal"][seed]
        nmask = steady_mask(nom)
        # 50/50 split of nominal steady frames for fit vs calibration/test.
        T, N = nom["done"].shape
        lat_all = nom["latent"].reshape(T * N, -1)[nmask.reshape(-1)]
        obs_all = nom["obs"].reshape(T * N, -1)[nmask.reshape(-1)]
        rs = np.random.default_rng(seed)
        perm = rs.permutation(len(lat_all))
        half = len(perm) // 2
        fit_idx, test_idx = perm[:half], perm[half:]
        scorers = {
            "latent": MahalanobisScorer.fit(lat_all[fit_idx]),
            "obs": MahalanobisScorer.fit(obs_all[fit_idx]),
        }
        # arbiter thresholds calibrated on nominal-fit latent scores
        trip, clear = calibrate_arbiter_thresholds(
            scorers["latent"], lat_all[fit_idx], trip_fpr=0.01, clear_fpr=0.15
        )
        arb_cfg = SimplexArbiterCfg(
            trip_threshold=trip, clear_threshold=clear, trip_persistence=3,
            clear_persistence=10, handoff_ticks=8, recover_ticks=20, min_fallback_ticks=10,
        )

        # nominal-test scores per method -> thresholds at TARGET_FPR + negatives for AUROC
        nom_scores = detector_scores(nom, scorers)
        nmask_flat = nmask.reshape(-1)
        neg = {m: nom_scores[m].reshape(-1)[nmask_flat][test_idx] for m in methods}
        thr = {m: float(np.quantile(neg[m][np.isfinite(neg[m])], 1 - TARGET_FPR)) for m in methods}

        fall_rates["nominal"][seed] = float(fall_frames(nom).any(0).mean())
        per_seed[seed]["nominal"]["unnec_fallback_latent"] = unnecessary_fallback_rate(
            nom, scorers["latent"], arb_cfg, "latent", nmask
        )

        for cond in ood_conds:
            if seed not in grid[cond]:
                continue
            dd = grid[cond][seed]
            smask = steady_mask(dd).reshape(-1)
            sc = detector_scores(dd, scorers)
            fall_rates[cond][seed] = float(fall_frames(dd).any(0).mean())
            rec = per_seed[seed][cond]
            for m in methods:
                pos = sc[m].reshape(-1)[smask]
                labels = np.concatenate([np.zeros(len(neg[m])), np.ones(len(pos))])
                scores = np.concatenate([neg[m], pos])
                good = np.isfinite(scores)
                rec[f"auroc_{m}"] = roc_auc(scores[good], labels[good])
                rec[f"auprc_{m}"] = average_precision(scores[good], labels[good])
                rec[f"fpr_at_op_{m}"] = float(np.mean(neg[m] > thr[m]))
                # lead time per method (pool over envs)
                all_leads, nf, nw = [], 0, 0
                for n in range(dd["done"].shape[1]):
                    ld, f_, w_ = lead_times(sc[m][:, n], fall_frames(dd)[:, n], dd["done"][:, n], thr[m])
                    all_leads += ld
                    nf += f_
                    nw += w_
                rec[f"n_fail_{m}"] = nf
                rec[f"warn_rate_{m}"] = (nw / nf) if nf else float("nan")
                rec[f"lead_median_{m}"] = float(np.median(all_leads)) if all_leads else float("nan")
                rec[f"lead_p25_{m}"] = float(np.percentile(all_leads, 25)) if all_leads else float("nan")
            # shield intervention (latent) on this OOD condition
            saved, nfail = intervention_success(dd, scorers["latent"], arb_cfg, "latent")
            rec["intervention_saved"] = saved
            rec["intervention_nfail"] = nfail
            rec["intervention_rate"] = (saved / nfail) if nfail else float("nan")

    # ---- aggregate across seeds (mean + 95% normal CI) --------------------
    def agg(vals):
        v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
        if v.size == 0:
            return {"mean": None, "lo": None, "hi": None, "n": 0}
        m, sd = float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0
        ci = 1.96 * sd / np.sqrt(v.size) if v.size > 1 else 0.0
        return {"mean": m, "lo": m - ci, "hi": m + ci, "n": int(v.size)}

    summary: dict = {"config": {"dt": DT, "warmup": WARMUP, "target_fpr": TARGET_FPR}, "conditions": {}}
    summary["fall_rates"] = {c: agg(list(fall_rates[c].values())) for c in fall_rates}
    summary["unnecessary_fallback_latent_nominal"] = agg(
        [per_seed[s]["nominal"].get("unnec_fallback_latent") for s in seeds]
    )
    for cond in ood_conds:
        block: dict = {}
        for m in methods:
            block[m] = {
                "auroc": agg([per_seed[s][cond].get(f"auroc_{m}") for s in seeds]),
                "auprc": agg([per_seed[s][cond].get(f"auprc_{m}") for s in seeds]),
                "lead_median_s": agg([per_seed[s][cond].get(f"lead_median_{m}") for s in seeds]),
                "warn_rate": agg([per_seed[s][cond].get(f"warn_rate_{m}") for s in seeds]),
            }
        block["shield_intervention_rate"] = agg([per_seed[s][cond].get("intervention_rate") for s in seeds])
        block["n_fail_latent"] = agg([per_seed[s][cond].get("n_fail_latent") for s in seeds])
        summary["conditions"][cond] = block

    (out / "results.json").write_text(json.dumps({"summary": summary, "per_seed": _to_jsonable(per_seed)}, indent=2))
    _write_report(out, summary, ood_conds, methods)
    _plots(out, summary, ood_conds)
    print(f"[eval] wrote {out}/results.json, report.md, plots")
    return 0


def _to_jsonable(dd):
    if isinstance(dd, dict):
        return {str(k): _to_jsonable(v) for k, v in dd.items()}
    if isinstance(dd, (np.floating, np.integer)):
        return float(dd)
    return dd


def _write_report(out, summary, ood_conds, methods):
    L = ["# Reliability Shield — Phase 3 (Isaac twin) Results\n"]
    c = summary["config"]
    L.append(f"dt={c['dt']}s, warmup={c['warmup']} frames dropped/episode, operating point = {c['target_fpr']:.0%} nominal FPR.\n")
    L.append("## Behavioral fall rate (env base-contact oracle)\n")
    L.append("| condition | fall rate (env-ever-fell) |")
    L.append("|---|---|")
    for cond, a in summary["fall_rates"].items():
        L.append(f"| {cond} | {_fmt(a)} |")
    uf = summary["unnecessary_fallback_latent_nominal"]
    L.append(f"\nUnnecessary-fallback rate on nominal (latent shield): **{_fmt(uf)}** per episode.\n")
    L.append("## Detection: latent-Mahalanobis vs baselines (AUROC, mean [95% CI] over seeds)\n")
    header = "| condition | " + " | ".join(methods) + " |"
    L.append(header)
    L.append("|" + "---|" * (len(methods) + 1))
    for cond in ood_conds:
        row = [cond] + [_fmt(summary["conditions"][cond][m]["auroc"]) for m in methods]
        L.append("| " + " | ".join(row) + " |")
    L.append("\n## Lead time before fall (seconds, latent shield) and warning rate\n")
    L.append("| condition | n_fail | warn_rate | lead median (s) | shield intervention rate |")
    L.append("|---|---|---|---|---|")
    for cond in ood_conds:
        b = summary["conditions"][cond]
        L.append(
            f"| {cond} | {_fmt(b['n_fail_latent'])} | {_fmt(b['latent_maha']['warn_rate'])} | "
            f"{_fmt(b['latent_maha']['lead_median_s'])} | {_fmt(b['shield_intervention_rate'])} |"
        )
    (out / "report.md").write_text("\n".join(L) + "\n")


def _fmt(a):
    if not a or a.get("mean") is None:
        return "n/a"
    if a.get("n", 0) > 1 and a.get("lo") is not None:
        return f"{a['mean']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]"
    return f"{a['mean']:.3f}"


def _plots(out, summary, ood_conds):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("[eval] matplotlib unavailable; skipping plots")
        return
    methods = ["latent_maha", "obs_maha", "obs_magnitude", "value_signal", "action_sat", "random"]
    x = np.arange(len(ood_conds))
    w = 0.13
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, m in enumerate(methods):
        vals = [summary["conditions"][c][m]["auroc"]["mean"] or 0 for c in ood_conds]
        ax.bar(x + i * w, vals, w, label=m)
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set_xticks(x + w * 2.5)
    ax.set_xticklabels(ood_conds, rotation=20, ha="right")
    ax.set_ylabel("Detection AUROC")
    ax.set_title("OOD detection: policy-latent vs baselines")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "auroc_by_condition.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
