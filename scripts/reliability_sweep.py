"""Operating-point sweep: is there an episode-quiet point with real lead time?

The main eval calibrated the shield at a 1% frame FPR, which -- because latent
scores are temporally autocorrelated -- engages on ~100% of nominal EPISODES.
This sweep recalibrates at the EPISODE level: for a grid of (threshold
percentile, persistence K), it measures the nominal per-episode engagement
rate and, on the failure-inducing shift (friction_severe), the fraction of
falls warned before onset and the median lead time. Vectorized run detection,
so it is fast. Answers: does a principled operating point exist?
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np

from phoenix.reliability.ood_monitor import MahalanobisScorer

DT = 0.02
WARMUP = 15


def load(raw="reliability_eval/raw"):
    g = {}
    for f in sorted(glob.glob(f"{raw}/*.npz")):
        stem = Path(f).stem
        if stem in ("smoke_nominal", "nohooks", "clean"):
            continue
        m = re.match(r"(.+)_seed(\d+)$", stem)
        g.setdefault(m.group(1), {})[int(m.group(2))] = dict(np.load(f))
    return g


def spans(done_col):
    start = 0
    for t in range(len(done_col)):
        if done_col[t]:
            yield start, t
            start = t + 1
    if start < len(done_col):
        yield start, len(done_col) - 1


def first_krun_before(exceed, K, limit):
    """First index i (< limit) that completes... actually STARTS a K-run of True.

    Returns the run-start index, or None. Only runs fully before `limit`.
    """
    if K <= 0 or len(exceed) < K:
        return None
    conv = np.convolve(exceed.astype(int), np.ones(K, int), "valid")  # len T-K+1
    hits = np.flatnonzero(conv == K)  # window start indices with K consecutive
    for i in hits:
        if i + K - 1 < limit:  # run completes before the onset/limit
            return int(i)
    return None


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="reliability_eval/raw")
    ap.add_argument("--out-dir", default="reliability_eval/results")
    ap.add_argument("--faller", default=None, help="OOD condition to measure lead on; default = highest fall rate")
    args = ap.parse_args()

    g = load(args.raw_dir)
    seeds = sorted(g["nominal"].keys())
    grid_p = [99.0, 99.5, 99.9, 99.95, 99.99]
    grid_K = [3, 5, 10, 20]
    # Pick the failure-inducing shift automatically (a standing policy faults
    # under different shifts than a walker).
    if args.faller:
        faller = args.faller
    else:
        rates = {}
        for c in g:
            if c == "nominal":
                continue
            fr = [float((d["done"] & (~d["time_out"])).any(0).mean()) for d in g[c].values()]
            rates[c] = float(np.mean(fr))
        faller = max(rates, key=rates.get)
        print(f"[sweep] auto-selected faller={faller} (fall rates: { {k: round(v,3) for k,v in rates.items()} })")

    rows = {}
    for seed in seeds:
        nom = g["nominal"][seed]
        Tn, Nn = nom["done"].shape
        lat = nom["latent"].reshape(Tn * Nn, -1)
        # steady mask
        sm = np.zeros((Tn, Nn), bool)
        for n in range(Nn):
            for s, e in spans(nom["done"][:, n]):
                sm[min(s + WARMUP, e + 1) : e + 1, n] = True
        smf = sm.reshape(-1)
        rs = np.random.default_rng(seed)
        idx = rs.permutation(int(smf.sum()))
        steady = lat[smf]
        half = len(idx) // 2
        scorer = MahalanobisScorer.fit(steady[idx[:half]])
        nom_fit_scores = scorer.score(steady[idx[:half]])
        nom_score_col = scorer.score(lat).reshape(Tn, Nn)  # for nominal-test episodes

        fs = g[faller][seed]
        Tf, Nf = fs["done"].shape
        fs_scores = scorer.score(fs["latent"].reshape(Tf * Nf, -1)).reshape(Tf, Nf)
        fs_fall = fs["done"] & (~fs["time_out"])

        for p in grid_p:
            thr = float(np.percentile(nom_fit_scores, p))
            for K in grid_K:
                # nominal per-episode engagement (test half of envs' steady episodes)
                nom_eps, nom_eng = 0, 0
                for n in range(Nn):
                    for s, e in spans(nom["done"][:, n]):
                        if e - s < WARMUP + K:
                            continue
                        seg = nom_score_col[s + WARMUP : e + 1, n]
                        nom_eps += 1
                        nom_eng += int(first_krun_before(seg > thr, K, len(seg) + 1) is not None)
                # friction_severe: warn before fall onset
                nf, nw, leads = 0, 0, []
                for n in range(Nf):
                    for s, e in spans(fs["done"][:, n]):
                        seg_fall = fs_fall[s : e + 1, n]
                        if not seg_fall.any():
                            continue
                        onset = int(np.argmax(seg_fall))
                        nf += 1
                        seg = fs_scores[s : e + 1, n]
                        st = first_krun_before(seg > thr, K, onset)
                        if st is not None:
                            nw += 1
                            leads.append((onset - st) * DT)
                rows.setdefault((p, K), []).append(
                    {
                        "nom_engage": nom_eng / max(nom_eps, 1),
                        "warn": nw / max(nf, 1),
                        "lead": float(np.median(leads)) if leads else float("nan"),
                    }
                )

    print(f"{'p%':>7} {'K':>3} | {'nom_ep_FPR':>10} | {'fricS_warn':>10} | {'lead_med_s':>10}")
    print("-" * 52)
    best = None
    out = []
    for (p, K), lst in sorted(rows.items()):
        ne = np.mean([r["nom_engage"] for r in lst])
        wr = np.mean([r["warn"] for r in lst])
        ld = np.nanmean([r["lead"] for r in lst])
        out.append({"p": p, "K": K, "nom_ep_fpr": ne, "warn": wr, "lead_med_s": ld})
        print(f"{p:>7} {K:>3} | {ne:>10.3f} | {wr:>10.3f} | {ld:>10.3f}")
        if ne <= 0.10 and wr >= 0.5 and (best is None or wr > best["warn"]):
            best = {"p": p, "K": K, "nom_ep_fpr": ne, "warn": wr, "lead_med_s": ld}
    print("\nBEST actionable point (nominal episode FPR <= 0.10 and warn >= 0.5):")
    print(json.dumps(best, indent=2) if best else "  NONE FOUND -- no quiet operating point warns in time.")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{args.out_dir}/sweep.json").write_text(
        json.dumps({"faller": faller, "grid": out, "best": best}, indent=2)
    )


if __name__ == "__main__":
    main()
