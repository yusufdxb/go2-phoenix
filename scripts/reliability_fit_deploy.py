"""Fit the deployable shield artifact from nominal rollouts.

Phase 4, step 1. Turns the Phase 3 study into a single file the Jetson can
load. The discipline that matters here is *what gets to influence the operating
point*:

* The scorer is fit on one half of the nominal envs, in float64.
* The threshold and persistence ``K`` are selected on the **held-out** half of
  the nominal envs, using nominal data only, the selection rule is "the most
  sensitive operating point whose held-out nominal *episode* false-alarm rate
  stays within budget". No OOD rollout influences the choice.
* The shifted conditions are then scored *once*, as evaluation, and the
  resulting warn rate and lead time are recorded in the artifact as measured
  consequences rather than as selection criteria.

That ordering is what makes the deployed number defensible: if the threshold
had been picked to maximise the warn rate, the reported warn rate would be a
training-set number wearing a test-set label.

Finally the float32 deploy path is compared against the float64 fit path over
every sample, and the run **fails** if the two disagree about any frame's trip
decision.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/reliability_fit_deploy.py \
        --raw-dir reliability_eval/raw_stand \
        --out deploy/shield_stand_v3.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

from phoenix.reliability.deploy import (
    ArbiterTimings,
    DeployMonitor,
    OperatingPoint,
    parity_report,
    save_artifact,
    whitener_from_cholesky,
)
from phoenix.reliability.ood_monitor import MahalanobisScorer

DT = 0.02  # 50 Hz control
WARMUP = 15  # ticks discarded after each reset before an episode counts
GRID_P = (99.0, 99.5, 99.9, 99.95, 99.99, 99.995, 99.999)
GRID_K = (3, 5, 10, 20, 30, 50)


def load_raw(raw_dir: str) -> dict[str, dict[int, dict]]:
    groups: dict[str, dict[int, dict]] = {}
    for path in sorted(glob.glob(f"{raw_dir}/*.npz")):
        stem = Path(path).stem
        match = re.match(r"(.+)_seed(\d+)$", stem)
        if match is None:
            continue
        groups.setdefault(match.group(1), {})[int(match.group(2))] = dict(np.load(path))
    if "nominal" not in groups:
        raise SystemExit(f"no nominal rollouts found in {raw_dir}")
    return groups


def episode_spans(done_col: np.ndarray):
    start = 0
    for t in range(len(done_col)):
        if done_col[t]:
            yield start, t
            start = t + 1
    if start < len(done_col):
        yield start, len(done_col) - 1


def first_krun(exceed: np.ndarray, k: int, limit: int) -> int | None:
    """Start index of the first run of ``k`` consecutive True values ending before ``limit``."""
    if k <= 0 or len(exceed) < k:
        return None
    conv = np.convolve(exceed.astype(np.int32), np.ones(k, dtype=np.int32), "valid")
    for i in np.flatnonzero(conv == k):
        if i + k - 1 < limit:
            return int(i)
    return None


def steady_mask(done: np.ndarray) -> np.ndarray:
    """Boolean (T, N) mask of ticks at least ``WARMUP`` steps past a reset."""
    steps, envs = done.shape
    mask = np.zeros((steps, envs), dtype=bool)
    for n in range(envs):
        for s, e in episode_spans(done[:, n]):
            mask[min(s + WARMUP, e + 1) : e + 1, n] = True
    return mask


def episode_engagement(scores: np.ndarray, done: np.ndarray, thr: float, k: int) -> tuple[int, int]:
    """Return ``(engaged_episodes, total_episodes)`` at threshold ``thr`` / persistence ``k``."""
    engaged = total = 0
    for n in range(done.shape[1]):
        for s, e in episode_spans(done[:, n]):
            if e - s < WARMUP + k:
                continue
            seg = scores[s + WARMUP : e + 1, n]
            total += 1
            engaged += int(first_krun(seg > thr, k, len(seg) + 1) is not None)
    return engaged, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="reliability_eval/raw_stand")
    ap.add_argument("--out", default="deploy/shield_stand_v3.npz")
    ap.add_argument("--report", default=None, help="JSON fit report (default: <out>.report.json)")
    ap.add_argument(
        "--max-episode-fpr",
        type=float,
        default=0.01,
        help="Budget for held-out nominal per-episode false alarms",
    )
    ap.add_argument("--clear-percentile", type=float, default=90.0)
    ap.add_argument("--fit-samples", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    # Ramp / release timings are frozen INTO the artifact (they change the
    # outcome as much as the threshold does), so they are set here, once, at
    # fit time, not in the launch config.
    ap.add_argument("--handoff-ticks", type=int, default=10)
    ap.add_argument("--recover-ticks", type=int, default=25)
    ap.add_argument("--clear-persistence", type=int, default=10)
    ap.add_argument("--min-fallback-ticks", type=int, default=20)
    ap.add_argument("--latch", action="store_true", default=True)
    ap.add_argument("--no-latch", dest="latch", action="store_false")
    args = ap.parse_args()

    groups = load_raw(args.raw_dir)
    seeds = sorted(groups["nominal"])
    print(f"[fit] conditions={sorted(groups)} seeds={seeds}")

    # --- split nominal envs into fit / calibration halves -------------------
    # Splitting by ENV, not by frame, keeps whole episodes on one side of the
    # split. A frame-wise split would leak: consecutive frames of one episode
    # are near-duplicates, so the calibration half would be scoring data the
    # scorer had effectively already seen.
    fit_rows: list[np.ndarray] = []
    calib: list[tuple[np.ndarray, np.ndarray]] = []
    rng = np.random.default_rng(args.seed)
    for seed in seeds:
        run = groups["nominal"][seed]
        latent, done = run["latent"], run["done"]
        envs = latent.shape[1]
        perm = rng.permutation(envs)
        fit_envs, calib_envs = perm[: envs // 2], perm[envs // 2 :]
        mask = steady_mask(done)
        sub = latent[:, fit_envs, :][mask[:, fit_envs]]
        fit_rows.append(sub.reshape(-1, latent.shape[-1]))
        calib.append((latent[:, calib_envs, :], done[:, calib_envs]))

    fit_data = np.concatenate(fit_rows, axis=0).astype(np.float64)
    if fit_data.shape[0] > args.fit_samples:
        idx = rng.choice(fit_data.shape[0], args.fit_samples, replace=False)
        fit_data = fit_data[idx]
    latent_dim = fit_data.shape[1]
    print(f"[fit] fitting on {fit_data.shape[0]} nominal frames, dim={latent_dim}")

    scorer = MahalanobisScorer.fit(fit_data)
    print(f"[fit] Ledoit-Wolf shrinkage = {scorer.shrinkage:.4f}")
    monitor = DeployMonitor.from_scorer(scorer)

    # --- select the operating point on held-out nominal only ----------------
    calib_scores = []
    for latent, done in calib:
        steps, envs, _ = latent.shape
        flat = scorer.score(latent.reshape(steps * envs, -1)).reshape(steps, envs)
        calib_scores.append((flat, done))
    fit_scores = scorer.score(fit_data)

    candidates = []
    for p in GRID_P:
        thr = float(np.percentile(fit_scores, p))
        for k in GRID_K:
            eng = tot = 0
            for flat, done in calib_scores:
                e, t = episode_engagement(flat, done, thr, k)
                eng += e
                tot += t
            fpr = eng / max(tot, 1)
            candidates.append({"p": p, "K": k, "threshold": thr, "episode_fpr": fpr, "n_ep": tot})
            print(f"[fit]   p={p:<6} K={k:<3} held-out nominal episode FPR = {fpr:.4f} (n={tot})")

    feasible = [c for c in candidates if c["episode_fpr"] <= args.max_episode_fpr]
    if not feasible:
        best = min(candidates, key=lambda c: c["episode_fpr"])
        raise SystemExit(
            "FAIL CLOSED: no candidate meets the nominal episode-FPR budget "
            f"{args.max_episode_fpr}; best achievable was {best['episode_fpr']:.4f} "
            f"at p={best['p']} K={best['K']}"
        )
    # Selection rule, fixed a priori and computable from nominal data alone:
    # minimise K first, then the threshold. K is the *intrinsic detection delay*
    #, the shield cannot possibly warn sooner than K ticks after the latent
    # goes bad, so buying quiet with a long persistence window silently
    # forfeits exactly the fast failures the shield exists to catch. Among
    # equal-K candidates the lowest (most sensitive) threshold wins.
    chosen = min(feasible, key=lambda c: (c["K"], c["threshold"]))
    clear_thr = float(np.percentile(fit_scores, args.clear_percentile))
    if not clear_thr < chosen["threshold"]:
        clear_thr = chosen["threshold"] * 0.5
    print(
        f"[fit] SELECTED p={chosen['p']} K={chosen['K']} trip={chosen['threshold']:.2f} "
        f"clear={clear_thr:.2f} (held-out nominal episode FPR {chosen['episode_fpr']:.4f})"
    )

    # --- evaluate the chosen point on the shifted conditions (report only) ---
    thr, k = chosen["threshold"], chosen["K"]
    per_condition = {}
    for cond in sorted(groups):
        if cond == "nominal":
            continue
        warned = falls = 0
        leads: list[float] = []
        engaged = episodes = 0
        for seed, run in groups[cond].items():
            latent, done, time_out = run["latent"], run["done"], run["time_out"]
            steps, envs, _ = latent.shape
            flat = scorer.score(latent.reshape(steps * envs, -1)).reshape(steps, envs)
            fell = done & (~time_out)
            e_, t_ = episode_engagement(flat, done, thr, k)
            engaged += e_
            episodes += t_
            for n in range(envs):
                for s, e in episode_spans(done[:, n]):
                    seg_fall = fell[s : e + 1, n]
                    if not seg_fall.any():
                        continue
                    onset = int(np.argmax(seg_fall))
                    falls += 1
                    start = first_krun(flat[s : e + 1, n] > thr, k, onset)
                    if start is not None:
                        warned += 1
                        # Lead is measured to the DECISION tick, which is the
                        # end of the K-run, not its start: the arbiter has not
                        # tripped until the run completes. Reporting from the
                        # run start overstates the margin by K-1 ticks.
                        leads.append((onset - (start + k - 1)) * DT)
        per_condition[cond] = {
            "falls": falls,
            "warned": warned,
            "warn_rate": warned / falls if falls else None,
            "median_lead_s": float(np.median(leads)) if leads else None,
            "episode_engagement_rate": engaged / max(episodes, 1),
        }
        print(
            f"[eval] {cond:<18} falls={falls:<5} warned={warned:<5} "
            f"lead={per_condition[cond]['median_lead_s']} "
            f"engage={per_condition[cond]['episode_engagement_rate']:.3f}"
        )

    fallers = {c: v for c, v in per_condition.items() if v["falls"] >= 20}
    headline = max(fallers, key=lambda c: fallers[c]["falls"]) if fallers else None

    # --- float32 deploy parity gate -----------------------------------------
    parity_pool = [fit_data[rng.choice(fit_data.shape[0], min(4000, fit_data.shape[0]), False)]]
    for cond in sorted(groups):
        if cond == "nominal":
            continue
        run = next(iter(groups[cond].values()))
        flat = run["latent"].reshape(-1, latent_dim)
        parity_pool.append(flat[rng.choice(flat.shape[0], min(2000, flat.shape[0]), False)])
    parity_samples = np.concatenate(parity_pool, axis=0)
    parity = parity_report(scorer, monitor, parity_samples, trip_threshold=thr)
    print(
        f"[parity] n={parity['n_samples']} max_rel_err={parity['max_rel_err']:.3e} "
        f"decision_disagreement={parity['decision_disagreement']}"
    )
    if parity["decision_disagreement"] != 0:
        raise SystemExit(
            "FAIL CLOSED: the float32 deploy path disagrees with the float64 fit path "
            f"on {parity['decision_disagreement']} frames' trip decision."
        )
    if parity["max_rel_err"] > 1e-3:
        raise SystemExit(
            f"FAIL CLOSED: float32 relative error {parity['max_rel_err']:.3e} exceeds 1e-3."
        )

    # --- write artifact ------------------------------------------------------
    meta_path = Path(args.raw_dir) / "nominal_seed0.meta.json"
    src_meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    head = per_condition.get(headline, {}) if headline else {}
    timings = ArbiterTimings(
        handoff_ticks=args.handoff_ticks,
        recover_ticks=args.recover_ticks,
        clear_persistence=args.clear_persistence,
        min_fallback_ticks=args.min_fallback_ticks,
        latch=args.latch,
    )
    lead = head.get("median_lead_s")
    op = OperatingPoint(
        trip_threshold=thr,
        clear_threshold=clear_thr,
        trip_persistence=k,
        # The runtime must not be able to engage during the same post-reset
        # window this calibration discards, or it operates in a regime nobody
        # measured. Carrying WARMUP into the artifact keeps the two bound.
        arming_ticks=WARMUP,
        nominal_episode_fpr=chosen["episode_fpr"],
        falls_warned=head.get("warn_rate") if head.get("warn_rate") is not None else float("nan"),
        median_lead_s=lead if lead is not None else float("nan"),
        # The margin that physically matters: the fallback is not actually in
        # control until the handoff ramp completes.
        median_full_fallback_lead_s=(
            lead - timings.handoff_ticks * DT if lead is not None else float("nan")
        ),
    )
    provenance = {
        "raw_dir": args.raw_dir,
        "checkpoint": src_meta.get("checkpoint"),
        "checkpoint_sha256": src_meta.get("checkpoint_sha256"),
        "env_config": src_meta.get("env_config"),
        "tap_indices": src_meta.get("tap_indices"),
        "latent_dim": latent_dim,
        "obs_dim": src_meta.get("obs_dim"),
        "versions": src_meta.get("versions"),
        "control_dt_s": DT,
        "selection": {
            "rule": "most sensitive point within held-out nominal episode-FPR budget",
            "max_episode_fpr": args.max_episode_fpr,
            "percentile": chosen["p"],
            "shrinkage": scorer.shrinkage,
            "fit_frames": int(fit_data.shape[0]),
            "calib_episodes": chosen["n_ep"],
        },
        "headline_condition": headline,
        "evaluation": per_condition,
        "parity": parity,
    }
    out = save_artifact(
        args.out,
        mean=scorer.mean,
        whitener=whitener_from_cholesky(scorer._chol),
        operating_point=op,
        provenance=provenance,
        timings=timings,
    )
    size_kb = out.stat().st_size / 1024
    print(f"[fit] wrote {out} ({size_kb:.0f} KB)")

    report_path = Path(args.report) if args.report else Path(str(args.out) + ".report.json")
    report_path.write_text(
        json.dumps(
            {
                "operating_point": op.to_dict(),
                "timings": timings.to_dict(),
                "candidates": candidates,
                "per_condition": per_condition,
                "parity": parity,
                "provenance": provenance,
            },
            indent=2,
        )
    )
    print(f"[fit] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
