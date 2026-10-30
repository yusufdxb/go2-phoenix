"""Metrics for the v2 friction-probe replay study.

Definitions: ``docs/preregistration/2026-10-30-actuator-probe-v2.md``; thresholds:
``configs/sim2real/actuator_probe_v2_prereg.yaml`` (the defaults here only make the
functions callable in tests). Pure numpy; the MuJoCo replay is ``scripts/probe_eval.py``.

Units: q [rad] sampled at the probe's 50 Hz rows, thresholds [N m] = kp * tracking error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from phoenix.sim2real.actuator_replay import JOINT_TYPES, breakaway, joint_type

#: A breakaway window is a ramp phase plus the dwell after it, so a model that breaks
#: late (after the ramp stopped) is scored as late, not as missed.
WINDOW_PHASES = {"rise": ("rise", "dwell_peak"), "fall": ("fall", "dwell_zero", "rest")}
ZERO_DWELL = ("dwell_zero", "rest")
#: Probe rows that are not part of an excursion.
NON_EXCURSION = ("settle", "settle_hold", "abort", "done")


@dataclass(frozen=True)
class Excursion:
    joint: int  # index in the probe joint order
    k: int  # excursion number within this joint's block (v2: schedule index)
    rows: np.ndarray  # probe row indices, rise start to end of the zero dwell
    direction: int  # sign of the rise
    rate: float  # commanded ramp rate [rad/s]


def excursions(
    active: np.ndarray, phase: np.ndarray, rate: np.ndarray, target: np.ndarray,
    q0: np.ndarray, order: Sequence[str],
) -> list[Excursion]:
    """Split probe rows into complete excursions (rise, fall and a zero dwell present).

    A new excursion starts at every row entering ``rise``; it ends before the next
    rise, a change of active joint, or a non-excursion row. Incomplete excursions
    (an aborted tail) are dropped, for every model alike.
    """
    out: list[Excursion] = []
    counts: dict[int, int] = {}
    cur: list[int] = []

    def close() -> None:
        if not cur:
            return
        ph = set(phase[cur].tolist())
        if "rise" in ph and "fall" in ph and ph & set(ZERO_DWELL):
            j = list(order).index(str(active[cur[0]]))
            r = np.asarray(cur)
            rise = r[phase[r] == "rise"]
            d = int(np.sign(np.mean(target[rise, j] - q0[j])))
            k = counts.get(j, 0)
            counts[j] = k + 1
            out.append(Excursion(j, k, r, d, float(rate[rise[0]])))
        cur.clear()

    for i in range(len(phase)):
        p = str(phase[i])
        if p in NON_EXCURSION:
            close()
            continue
        starts = p == "rise" and (i == 0 or str(phase[i - 1]) != "rise")
        if starts or (cur and active[i] != active[cur[0]]):
            close()
        if cur or p == "rise":
            cur.append(i)
    close()
    return out


def windows(ex: Excursion, phase: np.ndarray) -> list[tuple[str, int, np.ndarray]]:
    """(name, motion direction, row indices) of the rise and the return (fall) window."""
    out = []
    for name, sgn in (("rise", 1), ("fall", -1)):
        r = ex.rows[np.isin(phase[ex.rows], WINDOW_PHASES[name])]
        if len(r) >= 2:
            out.append((name, sgn * ex.direction, r))
    return out


def slip_events(
    q: np.ndarray, t: np.ndarray, moving: float = 5e-4, still: int = 3,
) -> list[tuple[float, float]]:
    """Motion events on a 50 Hz trace: maximal same-sign runs of moving ticks
    (|q[k+1] - q[k]| >= ``moving``) with >= ``still`` still ticks on both sides.
    Returns (start time, displacement) per event."""
    d = np.diff(np.asarray(q, dtype=np.float64))
    mv = np.abs(d) >= moving
    sg = np.sign(d)
    n, i, out = len(d), 0, []
    while i < n:
        if not mv[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mv[j + 1] and sg[j + 1] == sg[i]:
            j += 1
        left = i >= still and not mv[i - still : i].any()
        right = j + 1 + still <= n and not mv[j + 1 : j + 1 + still].any()
        if left and right:
            out.append((float(t[i]), float(d[i : j + 1].sum())))
        i = j + 1
    return out


def micro_slips(events, lo: float = 0.002, hi: float = 0.020) -> list[tuple[float, float]]:
    return [e for e in events if lo <= abs(e[1]) < hi]


def matched(hw: list, model: list, window_s: float = 0.10) -> int:
    """hw events with a model event of the same sign starting within ``window_s``."""
    return sum(
        any(np.sign(m[1]) == np.sign(h[1]) and abs(m[0] - h[0]) <= window_s for m in model)
        for h in hw
    )


def _slip_cfg(cfg: dict) -> dict:
    s = cfg["metrics"]["slip"]
    return {"moving": s["moving_tick_rad"], "still": s["still_ticks"]}


def score_excursion(
    ex: Excursion, t: np.ndarray, phase: np.ndarray, target: np.ndarray, kp: np.ndarray,
    q_hw: np.ndarray, dq_hw: np.ndarray, q_m: np.ndarray, dq_m: np.ndarray, q0: np.ndarray,
    cfg: dict, thr: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Metrics 1-9 for one excursion of one model. Arrays are full-run (rows x 12) for
    the hw side; ``q_m``/``dq_m`` are the model's probed-joint trace on ``ex.rows``."""
    j, r = ex.joint, ex.rows
    met = cfg["metrics"]
    thr = met["breakaway_thr_rad"] if thr is None else thr
    sl = met["slip"]
    qh, tt = q_hw[r, j], t[r]
    loc = {int(i): n for n, i in enumerate(r)}
    exc = {
        "joint": j, "k": ex.k, "dir": ex.direction, "rate": ex.rate,
        "pos_rmse": float(np.sqrt(np.mean((q_m - qh) ** 2))),
        "vel_rmse": float(np.sqrt(np.mean((dq_m - dq_hw[r, j]) ** 2))),
    }
    ev_h = micro_slips(slip_events(qh, tt, **_slip_cfg(cfg)), sl["min_event_rad"], sl["max_event_rad"])
    ev_m = micro_slips(slip_events(q_m, tt, **_slip_cfg(cfg)), sl["min_event_rad"], sl["max_event_rad"])
    exc.update(
        n_slip_hw=len(ev_h), n_slip_m=len(ev_m), n_slip_matched=matched(ev_h, ev_m, sl["match_window_s"]),
        slip_disp_hw=float(np.mean([abs(e[1]) for e in ev_h])) if ev_h else float("nan"),
        slip_disp_m=float(np.mean([abs(e[1]) for e in ev_m])) if ev_m else float("nan"),
    )
    z = r[np.isin(phase[r], ZERO_DWELL)]
    hold = z[t[z] >= t[z[-1]] - met["hold_window_s"] + 1e-9] if len(z) else z
    hl = [loc[int(i)] for i in hold]
    exc["hold_err_mrad"] = float(abs(np.mean(q_m[hl]) - np.mean(q_hw[hold, j])) * 1e3) if hl else float("nan")
    exc["hold_off_hw_mrad"] = float((np.mean(q_hw[hold, j]) - q0[j]) * 1e3) if hl else float("nan")
    wins = []
    for name, d, w in windows(ex, phase):
        t0, t1 = float(t[w[0]]), float(t[w[-1]])
        lw = [loc[int(i)] for i in w]
        th, eh = breakaway(q_hw[w, j], target[w, j], t[w], t0, t1, d, thr)
        tm, em = breakaway(q_m[lw], target[w, j], t[w], t0, t1, d, thr)
        k_p = float(kp[w[0], j]) if np.ndim(kp) == 2 else float(kp[w[0]])
        hw_b, m_b = np.isfinite(th), np.isfinite(tm)
        dt_tick = float(np.median(np.diff(t[w])))
        wins.append({
            "joint": j, "k": ex.k, "rep": ex.k // 4, "rate": ex.rate, "win": name, "dir": d,
            "thr_hw": k_p * eh, "thr_m": k_p * em,
            "thr_err": k_p * (em - eh) if hw_b and m_b else float("nan"),
            "t_err": tm - th if hw_b and m_b else float("nan"),
            "missed": bool(hw_b and not m_b), "false": bool(m_b and not hw_b),
            "hw_broke": bool(hw_b), "onset_hw": bool(hw_b and th <= 2 * dt_tick + 1e-9),
        })
    return exc, wins


def noise_floor(
    t: np.ndarray, q_hw: np.ndarray, active: np.ndarray, phase: np.ndarray,
    order: Sequence[str], cfg: dict,
) -> dict[str, float]:
    """Micro-slip events per second on each probed joint's own settle phase (hw)."""
    sl = cfg["metrics"]["slip"]
    out = {}
    for j, n in enumerate(order):
        r = np.flatnonzero((active.astype(str) == n) & (phase.astype(str) == "settle"))
        if len(r) < 10:
            continue
        ev = micro_slips(slip_events(q_hw[r, j], t[r], **_slip_cfg(cfg)), sl["min_event_rad"],
                         sl["max_event_rad"])
        out[n] = len(ev) / float(t[r[-1]] - t[r[0]])
    return out


def sign_flip_p(d: Sequence[float]) -> float:
    """Exact two-sided paired sign-flip p-value of mean(d) (2^n enumeration)."""
    d = np.asarray([x for x in d if np.isfinite(x)], dtype=np.float64)
    if len(d) == 0:
        return float("nan")
    obs = abs(d.mean())
    signs = np.asarray(list(product((1.0, -1.0), repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= obs - 1e-12))


def _fam(order: Sequence[str]) -> dict[str, list[int]]:
    return {f: [j for j, n in enumerate(order) if joint_type(n) == f] for f in JOINT_TYPES}


def _nanmed(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def summarize(exc: list[dict], wins: list[dict], order: Sequence[str]) -> dict[str, Any]:
    """Per joint (median over windows/excursions) and per family (median of joints)."""
    pj: dict[str, dict[str, float]] = {}
    for j, n in enumerate(order):
        w = [x for x in wins if x["joint"] == j]
        e = [x for x in exc if x["joint"] == j]
        if not w:
            continue
        pj[n] = {
            "abs_thr_err": _nanmed([abs(x["thr_err"]) for x in w]),
            "abs_t_err": _nanmed([abs(x["t_err"]) for x in w]),
            "thr_hw": _nanmed([x["thr_hw"] for x in w]),
            "missed_frac": float(np.mean([x["missed"] for x in w])),
            "false_frac": float(np.mean([x["false"] for x in w])),
            "dir_err": _nanmed([x["thr_err"] for x in w if x["dir"] > 0])
            - _nanmed([x["thr_err"] for x in w if x["dir"] < 0]),
            "pos_rmse": _nanmed([x["pos_rmse"] for x in e]),
            "vel_rmse": _nanmed([x["vel_rmse"] for x in e]),
            "hold_err_mrad": _nanmed([x["hold_err_mrad"] for x in e]),
            "slip_rate_hw": float(np.mean([x["n_slip_hw"] for x in e])),
            "slip_rate_m": float(np.mean([x["n_slip_m"] for x in e])),
            "slip_recall": (sum(x["n_slip_matched"] for x in e) / sum(x["n_slip_hw"] for x in e))
            if sum(x["n_slip_hw"] for x in e) else float("nan"),
        }
    fam = {}
    for f, js in _fam(order).items():
        names = [order[j] for j in js if order[j] in pj]
        fam[f] = {k: _nanmed([pj[n][k] for n in names]) for k in next(iter(pj.values()))} if names else {}
    nh = sum(x["n_slip_hw"] for x in exc)
    pooled = {
        "n_windows": len(wins),
        "hw_breakaway_frac": float(np.mean([x["hw_broke"] for x in wins])) if wins else float("nan"),
        "onset_frac": float(np.mean([x["onset_hw"] for x in wins if x["hw_broke"]]))
        if any(x["hw_broke"] for x in wins) else float("nan"),
        "missed_false_frac": float(np.mean([x["missed"] or x["false"] for x in wins])) if wins else float("nan"),
        "slip_recall": sum(x["n_slip_matched"] for x in exc) / nh if nh else float("nan"),
        "slip_rate_hw": nh / len(exc) if exc else float("nan"),
    }
    return {"per_joint": pj, "per_family": fam, "pooled": pooled}


def h_mag(sb: dict, sbm: dict, order: Sequence[str], cfg: dict) -> dict[str, Any]:
    """Primary: Bm vs B median |threshold error|, family cuts and the 12-joint sign flip."""
    c = cfg["decision"]["H_mag"]
    names = [n for n in order if n in sb["per_joint"] and n in sbm["per_joint"]]
    d = [sb["per_joint"][n]["abs_thr_err"] - sbm["per_joint"][n]["abs_thr_err"] for n in names]
    cut = {}
    for f in JOINT_TYPES:
        b, m = sb["per_family"][f].get("abs_thr_err"), sbm["per_family"][f].get("abs_thr_err")
        cut[f] = 1.0 - m / b if b and np.isfinite(b) and np.isfinite(m) else float("nan")
    p = sign_flip_p(d)
    n_fam = sum(v >= c["min_rel_reduction"] for v in cut.values() if np.isfinite(v))
    return {
        "family_rel_reduction": cut, "n_families_cut": int(n_fam), "sign_flip_p": p,
        "mean_d_nm": float(np.nanmean(d)) if d else float("nan"), "n_joints": len(names),
        "supported": bool(n_fam >= c["min_families"] and p < c["alpha"] and np.nanmean(d) > 0),
    }


def h_vel(wins_hw: list[dict], order: Sequence[str], cfg: dict) -> dict[str, Any]:
    """hw fast vs slow breakaway threshold per family (cells: joint x motion direction)."""
    c = cfg["decision"]["H_vel"]
    rates = sorted({x["rate"] for x in wins_hw})
    if len(rates) < 2:
        return {"supported": False, "note": "one rate only"}
    slow, fast = rates[0], rates[-1]
    out, n_sup = {}, 0
    for f, js in _fam(order).items():
        diffs, sds, slows = [], [], []
        for j, d in product(js, (1, -1)):
            cell = {rt: [x["thr_hw"] for x in wins_hw if x["joint"] == j and x["dir"] == d
                         and x["rate"] == rt and np.isfinite(x["thr_hw"])] for rt in (slow, fast)}
            if not cell[slow] or not cell[fast]:
                continue
            diffs.append(np.median(cell[fast]) - np.median(cell[slow]))
            slows.append(np.median(cell[slow]))
            sds += [np.std(v, ddof=1) for v in cell.values() if len(v) > 1]
        if not diffs:
            out[f] = {"note": "no paired cells"}
            continue
        diff, base = float(np.median(diffs)), float(np.median(slows))
        sd = float(np.median(sds)) if sds else float("nan")
        bar = max(c["min_rel_diff"] * abs(base), c["min_sd_multiple"] * (sd if np.isfinite(sd) else 0.0),
                  c["min_abs_nm"])
        sup = abs(diff) > bar
        n_sup += sup
        out[f] = {"fast_minus_slow_nm": diff, "slow_nm": base, "repeat_sd_nm": sd, "bar_nm": bar,
                  "exceeds": bool(sup)}
    return {"per_family": out, "supported": bool(n_sup >= c["min_families"])}


def h_dir(wins: list[dict], hw_med_fam: dict[str, float], order: Sequence[str],
          cfg: dict) -> dict[str, Any]:
    """+ vs - signed model threshold error per family; flagged only if consistent in
    every (repeat, rate) subset."""
    c = cfg["decision"]["H_dir"]
    out = {}
    for f, js in _fam(order).items():
        w = [x for x in wins if x["joint"] in js and np.isfinite(x["thr_err"])]
        sub = []
        for rep, rt in {(x["rep"], x["rate"]) for x in w}:
            p = [x["thr_err"] for x in w if x["rep"] == rep and x["rate"] == rt and x["dir"] > 0]
            m = [x["thr_err"] for x in w if x["rep"] == rep and x["rate"] == rt and x["dir"] < 0]
            if p and m:
                sub.append(np.mean(p) - np.mean(m))
        if not sub:
            out[f] = {"note": "no subsets"}
            continue
        allw = np.mean([x["thr_err"] for x in w if x["dir"] > 0]) - np.mean(
            [x["thr_err"] for x in w if x["dir"] < 0])
        rel = abs(allw) / abs(hw_med_fam[f]) if hw_med_fam.get(f) else float("nan")
        consistent = all(np.sign(s) == np.sign(allw) for s in sub)
        out[f] = {"plus_minus_nm": float(allw), "rel": float(rel), "consistent": bool(consistent),
                  "flag": bool(rel > c["min_rel_diff"] and consistent)}
    return out


def h_inad(sums: dict[str, dict], floor_per_s: float, hw_rate_per_s: float, cfg: dict,
           models: Sequence[str] = ("B", "Bm")) -> dict[str, Any]:
    """Both constant-friction models miss hw micro-slips, and hw slips are above floor."""
    c = cfg["decision"]
    resolvable = bool(hw_rate_per_s >= 2.0 * floor_per_s and hw_rate_per_s > 0)
    rec = {m: sums[m]["pooled"]["slip_recall"] for m in models if m in sums}
    if not resolvable:
        return {"status": "undetermined: no resolvable micro-slip", "recall": rec,
                "hw_rate_per_s": hw_rate_per_s, "floor_per_s": floor_per_s}
    low = all(np.isfinite(v) and v < c["H_inad"]["max_recall"] for v in rec.values())
    return {"status": "supported" if low else "not supported", "recall": rec,
            "hw_rate_per_s": hw_rate_per_s, "floor_per_s": floor_per_s}


def model_c_accept(s: dict, sb: dict, sbm: dict, cfg: dict) -> dict[str, Any]:
    """The registered acceptance rule, applied to any candidate summary ``s``."""
    c = cfg["decision"]["model_C_accept"]
    checks = {}
    for f in JOINT_TYPES:
        sf = s["per_family"][f]
        hw = abs(sf["thr_hw"])
        best = min(sb["per_family"][f]["pos_rmse"], sbm["per_family"][f]["pos_rmse"])
        checks[f"{f}_thr"] = bool(sf["abs_thr_err"] <= c["max_threshold_err_frac"] * hw)
        checks[f"{f}_pos"] = bool(sf["pos_rmse"] <= c["max_pos_rmse_vs_best"] * best)
    checks["missed_false"] = bool(s["pooled"]["missed_false_frac"] <= c["max_missed_false_frac"])
    if s["pooled"]["slip_rate_hw"] >= 0.1:
        checks["slip_recall"] = bool(s["pooled"]["slip_recall"] >= c["min_slip_recall"])
    return {"checks": checks, "accept": all(checks.values())}


__all__ = [
    "Excursion", "WINDOW_PHASES", "excursions", "h_dir", "h_inad", "h_mag", "h_vel", "matched",
    "micro_slips", "model_c_accept", "noise_floor", "score_excursion", "sign_flip_p",
    "slip_events", "summarize", "windows",
]
