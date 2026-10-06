"""E2: per-joint actuator-effectiveness estimate from a standing robot.

Effectiveness ``g_j`` is actual PD stiffness over nominal, the quantity
``scripts/sim2sim_robustness.py --pattern weak_calf`` scales (calves x0.75).

Method (fixed). While the robot stands still under the deploy policy at a zero
velocity command, a loaded joint satisfies ``g*kp*(target - q) - g*kd*dq = load``.
With ``dq ~ 0`` and the same stance and load in simulation and on hardware::

    g_hw / g_sim = (kp_sim * err_sim) / (kp_hw * err_hw)        err = median(target - q)

``g_sim = 1`` for the nominal simulator reference, so the ratio is ``g_hw``. The
``kp`` factors normalise the case where the simulator and the hardware were
commanded different gains. Torque estimates derived from motor current are
deliberately not used: they apply the nominal torque constant, so they cannot see
the gain loss being measured.

This module is pure analysis: numpy only, no ROS and no MuJoCo import. The
simulator reference is produced by ``scripts/actuator_check.py sim-reference``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .go2_model import POLICY_JOINT_ORDER, UNITREE_MOTOR_ORDER

SCHEMA = "phoenix-actuator-ref/v1"
#: Hardware window: skip this long after the first policy tick (handover kp blend is 1 s).
HW_SKIP_S = 1.5
#: Simulator window: skip this long after the start of the run.
SIM_SKIP_S = 2.0
#: Hips carry little load; below this |err_sim| the ratio has no SNR.
MIN_REF_ERR_RAD = 0.02
CALF_MEAN_MIN = 0.80
GLOBAL_MEAN_MIN = 0.75
VERDICT_GO = "GO model_2750"
VERDICT_STOP = "STOP: re-plan"
#: Joint groups the decision rule is defined over.
RULE_GROUPS = ("thigh", "calf")


def joint_group(name: str) -> str:
    """``FL_thigh_joint`` -> ``thigh``."""
    return name.split("_")[1]


def unitree_to_policy(values: Sequence[float]) -> dict[str, float]:
    """Map a 12-vector in Unitree motor order to ``{joint name: value}``, by name."""
    if len(values) != len(UNITREE_MOTOR_ORDER):
        raise ValueError(f"expected {len(UNITREE_MOTOR_ORDER)} values, got {len(values)}")
    return {name: float(v) for name, v in zip(UNITREE_MOTOR_ORDER, values, strict=True)}


def _per_joint(value: Any) -> dict[str, float]:
    """A scalar (every joint) or a 12-list in Unitree order, as ``{name: value}``."""
    if isinstance(value, (list, tuple)):
        return unitree_to_policy(value)
    return {name: float(value) for name in UNITREE_MOTOR_ORDER}


def load_ticks(paths: Iterable[str | Path]) -> list[list[dict[str, Any]]]:
    """Bridge telemetry JSONL files -> one list of ``record == "tick"`` dicts per file."""
    out: list[list[dict[str, Any]]] = []
    for p in paths:
        ticks: list[dict[str, Any]] = []
        with Path(p).open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("record") == "tick":
                    ticks.append(rec)
        out.append(ticks)
    return out


def select_settled_window(
    ticks: Sequence[Mapping[str, Any]], skip_s: float = HW_SKIP_S
) -> list[Mapping[str, Any]]:
    """Policy-mode ticks from ``skip_s`` after the first policy tick to the last one,
    keeping only those whose ``kp_eff`` equals its value on the last policy tick."""
    pol = [t for t in ticks if t.get("mode") == "policy"]
    if not pol:
        return []
    t0 = int(pol[0]["t_mono_ns"]) + int(round(skip_s * 1e9))
    final_kp = _per_joint(pol[-1]["kp_eff"])
    out = []
    for t in pol:
        if int(t["t_mono_ns"]) < t0:
            continue
        kp = _per_joint(t["kp_eff"])
        if all(math.isclose(kp[n], final_kp[n], rel_tol=0.0, abs_tol=1e-9) for n in kp):
            out.append(t)
    return out


def summarize_hw(
    tick_sets: Sequence[Sequence[Mapping[str, Any]]], skip_s: float = HW_SKIP_S
) -> dict[str, Any]:
    """Pool the settled windows of several bridge files into one per-joint summary
    (same shape as a simulator reference, minus ``g_true``)."""
    err: dict[str, list[float]] = {n: [] for n in POLICY_JOINT_ORDER}
    q: dict[str, list[float]] = {n: [] for n in POLICY_JOINT_ORDER}
    tgt: dict[str, list[float]] = {n: [] for n in POLICY_JOINT_ORDER}
    dq: dict[str, list[float]] = {n: [] for n in POLICY_JOINT_ORDER}
    kp: dict[str, list[float]] = {n: [] for n in POLICY_JOINT_ORDER}
    kd: dict[str, list[float]] = {n: [] for n in POLICY_JOINT_ORDER}
    n_ticks = 0
    for ticks in tick_sets:
        for t in select_settled_window(ticks, skip_s):
            n_ticks += 1
            qm = unitree_to_policy(t["q_unitree"])
            tm = unitree_to_policy(t["final_target_unitree"])
            dm = unitree_to_policy(t["dq_unitree"])
            kpm = _per_joint(t["kp_eff"])
            kdm = _per_joint(t["kd_eff"])
            for n in POLICY_JOINT_ORDER:
                q[n].append(qm[n])
                tgt[n].append(tm[n])
                err[n].append(tm[n] - qm[n])
                dq[n].append(dm[n])
                kp[n].append(kpm[n])
                kd[n].append(kdm[n])
    if n_ticks == 0:
        raise ValueError("no settled policy-mode ticks in the supplied telemetry")
    med = {
        "err": err,
        "q": q,
        "target": tgt,
        "kp": kp,
        "kd": kd,
    }
    out: dict[str, Any] = {"schema": SCHEMA, "source": "hardware", "n": n_ticks, "skip_s": skip_s}
    for key, d in med.items():
        out[key] = {n: float(np.median(v)) for n, v in d.items()}
    out["dq_rms"] = {n: float(np.sqrt(np.mean(np.square(v)))) for n, v in dq.items()}
    out["joint_order"] = list(POLICY_JOINT_ORDER)
    return out


def summarize_trace(
    times: Sequence[float],
    q: np.ndarray,
    target: np.ndarray,
    joint_names: Sequence[str],
    kp: Mapping[str, float],
    kd: Mapping[str, float],
    skip_s: float = SIM_SKIP_S,
) -> dict[str, Any]:
    """Simulator trace (rows = control ticks, columns = ``joint_names``) -> per-joint summary."""
    t = np.asarray(times, dtype=float)
    keep = t >= skip_s
    if not np.any(keep):
        raise ValueError(
            f"trace ends at {float(t[-1]) if len(t) else 0.0:.2f} s, before the {skip_s} s skip"
        )
    qw = np.asarray(q, dtype=float)[keep]
    tw = np.asarray(target, dtype=float)[keep]
    idx = {n: i for i, n in enumerate(joint_names)}
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "source": "sim",
        "n": int(keep.sum()),
        "skip_s": skip_s,
    }
    out["err"] = {n: float(np.median(tw[:, idx[n]] - qw[:, idx[n]])) for n in POLICY_JOINT_ORDER}
    out["q"] = {n: float(np.median(qw[:, idx[n]])) for n in POLICY_JOINT_ORDER}
    out["target"] = {n: float(np.median(tw[:, idx[n]])) for n in POLICY_JOINT_ORDER}
    out["q_std"] = {n: float(np.std(qw[:, idx[n]])) for n in POLICY_JOINT_ORDER}
    out["kp"] = {n: float(kp[n]) for n in POLICY_JOINT_ORDER}
    out["kd"] = {n: float(kd[n]) for n in POLICY_JOINT_ORDER}
    out["joint_order"] = list(POLICY_JOINT_ORDER)
    return out


def verdict(calf_mean: float | None, global_mean: float | None) -> str:
    """GO only if calf mean > 0.80 AND global (thigh+calf) mean > 0.75, both strict."""
    if calf_mean is None or global_mean is None:
        return VERDICT_STOP
    if calf_mean > CALF_MEAN_MIN and global_mean > GLOBAL_MEAN_MIN:
        return VERDICT_GO
    return VERDICT_STOP


def estimate_effectiveness(
    ref: Mapping[str, Any], hw: Mapping[str, Any], min_ref_err: float = MIN_REF_ERR_RAD
) -> dict[str, Any]:
    """Per-joint ``g = kp_ref*err_ref / (kp_hw*err_hw)``: UNCALIBRATED, reported only.

    The ratio assumes an unchanged stance, which fails when gains drop (see
    :func:`decide` for the primary rule). Also returns the ratio-based verdict.

    ``ref`` is the nominal simulator reference (``g_sim = 1``); ``hw`` is a hardware
    summary or a simulator reference used as stand-in hardware. A joint is excluded
    from the rule when ``|err_ref| < min_ref_err`` (no SNR) or when ``g`` is not finite;
    hips are reported but never part of the rule (it is defined over thighs and calves).
    """
    g: dict[str, float | None] = {}
    excluded: dict[str, str] = {}
    for n in POLICY_JOINT_ORDER:
        e_ref, e_hw = float(ref["err"][n]), float(hw["err"][n])
        kp_ref, kp_hw = float(ref["kp"][n]), float(hw["kp"][n])
        if abs(e_hw) < 1e-9 or kp_hw <= 0.0:
            g[n] = None
            excluded[n] = "err_hw ~ 0 or kp_hw <= 0: ratio undefined"
            continue
        g[n] = kp_ref * e_ref / (kp_hw * e_hw)
        if joint_group(n) not in RULE_GROUPS:
            excluded[n] = "hip: not in the decision rule"
        if abs(e_ref) < min_ref_err:
            excluded[n] = f"|err_sim| {abs(e_ref):.4f} < {min_ref_err} rad: low SNR"
    calf = [
        v for n, v in g.items() if v is not None and n not in excluded and joint_group(n) == "calf"
    ]
    glob = [
        v
        for n, v in g.items()
        if v is not None and n not in excluded and joint_group(n) in RULE_GROUPS
    ]
    calf_mean = float(np.mean(calf)) if calf else None
    global_mean = float(np.mean(glob)) if glob else None
    normalised = any(
        not math.isclose(float(ref["kp"][n]), float(hw["kp"][n]), rel_tol=1e-6)
        for n in POLICY_JOINT_ORDER
    )
    out: dict[str, Any] = {
        "g": g,
        "calf_mean": calf_mean,
        "global_mean": global_mean,
        "n_calf_used": len(calf),
        "n_rule_used": len(glob),
        "excluded": excluded,
        "ratio_verdict_uncalibrated": verdict(calf_mean, global_mean),
        "rule": f"GO iff calf_mean > {CALF_MEAN_MIN} and global(thigh+calf)_mean > {GLOBAL_MEAN_MIN}",
        "kp_ref": dict(ref["kp"]),
        "kp_hw": dict(hw["kp"]),
        "kd_ref": dict(ref["kd"]),
        "kd_hw": dict(hw["kd"]),
        "kp_normalised": normalised,
        "n_ref": ref.get("n"),
        "n_hw": hw.get("n"),
    }
    g_true = hw.get("g_true")
    if g_true:
        out["g_true"] = dict(g_true)
    return out


# ---------------------------------------------------------------------------
# Primary decision: compare hardware against simulated BOUNDARY plants. The
# boundary references were run with the same policy, so they already contain the
# stance shift that biases the ratio estimator.
# ---------------------------------------------------------------------------

FACTOR_GROUPS = ("calf", "thigh", "hip", "all")
#: Boundary plants for the decision: calf-only 0.80, and every joint 0.75.
BOUNDARY_CALF_G = 0.80
BOUNDARY_ALL_G = 0.75


def parse_factor(spec: str) -> tuple[str, float]:
    """``"calf=0.85"`` -> ``("calf", 0.85)``; group must be calf, thigh, hip or all."""
    group, sep, val = spec.partition("=")
    if not sep or group not in FACTOR_GROUPS:
        raise ValueError(f"bad --factor {spec!r}: expected GROUP=VALUE, GROUP in {FACTOR_GROUPS}")
    v = float(val)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"bad --factor {spec!r}: value must be finite and > 0")
    return group, v


def apply_factors(
    base: Sequence[float], joint_order: Sequence[str], specs: Sequence[str]
) -> list[float]:
    """Per-joint factors: ``base`` overridden by each ``GROUP=VALUE`` in order (later wins)."""
    f = [float(x) for x in base]
    for spec in specs:
        group, v = parse_factor(spec)
        for i, n in enumerate(joint_order):
            if group == "all" or joint_group(n) == group:
                f[i] = v
    return f


def group_mean_err(summary: Mapping[str, Any], group: str) -> float:
    """Mean over the joints of ``group`` of the median ``target - q`` (positive under sag)."""
    vals = [float(summary["err"][n]) for n in POLICY_JOINT_ORDER if joint_group(n) == group]
    return float(np.mean(vals))


def decide(
    hw: Mapping[str, Any],
    ref_calf_boundary: Mapping[str, Any],
    ref_all_boundary: Mapping[str, Any],
    ref_nominal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """GO iff ``calf_err_hw`` is below BOTH boundary calf errors (strict).

    CALF test: boundary = calf err of the plant with calves x0.80. GLOBAL test:
    boundary = calf err of the plant with every joint x0.75. A weaker calf sags more,
    so a smaller error means a stronger calf. Thigh errors are reported only.
    """
    calf_hw = group_mean_err(hw, "calf")
    b_calf = group_mean_err(ref_calf_boundary, "calf")
    b_all = group_mean_err(ref_all_boundary, "calf")
    if ref_nominal is not None:
        nom = group_mean_err(ref_nominal, "calf")
        if not (nom > 0.0 and b_calf > nom and b_all > nom):
            raise ValueError(
                f"sign/monotonicity check failed: calf err nominal {nom:.4f}, "
                f"calf-0.80 {b_calf:.4f}, all-0.75 {b_all:.4f} (weaker plants must sag more)"
            )
    calf_test = calf_hw < b_calf
    global_test = calf_hw < b_all
    th_hw = group_mean_err(hw, "thigh")
    th_calf = group_mean_err(ref_calf_boundary, "thigh")
    th_all = group_mean_err(ref_all_boundary, "thigh")
    return {
        "calf_err_hw": calf_hw,
        "calf_boundary_err": b_calf,
        "calf_test_go": calf_test,
        "global_boundary_err": b_all,
        "global_test_go": global_test,
        "verdict": VERDICT_GO if (calf_test and global_test) else VERDICT_STOP,
        "thigh_informational": {
            "thigh_err_hw": th_hw,
            "thigh_err_calf_boundary_ref": th_calf,
            "thigh_err_all_boundary_ref": th_all,
            "note": "thigh loads are ~1 N m; not used in the verdict",
        },
    }


def sweep_points(refs: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
    """``(g_calf, calf_err)`` per calf-only sweep reference (thigh/hip g_true must be 1)."""
    pts = []
    for r in refs:
        gt = r["g_true"]
        others = [gt[n] for n in POLICY_JOINT_ORDER if joint_group(n) != "calf"]
        if any(not math.isclose(float(v), 1.0) for v in others):
            raise ValueError("sweep references must scale calves only")
        calves = {float(gt[n]) for n in POLICY_JOINT_ORDER if joint_group(n) == "calf"}
        if len(calves) != 1:
            raise ValueError("sweep reference has unequal calf factors")
        pts.append((calves.pop(), group_mean_err(r, "calf")))
    return sorted(pts)


def invert_sweep(points: Sequence[tuple[float, float]], err_hw: float) -> tuple[float, str]:
    """Piecewise-linear inverse of ``calf_err(g)`` at ``err_hw``; clips outside the sweep."""
    pts = sorted(points)
    gs = [p[0] for p in pts]
    errs = [p[1] for p in pts]
    if len(pts) < 2 or any(b >= a for a, b in zip(errs, errs[1:], strict=False)):
        raise ValueError(f"calf err must fall strictly with g across the sweep, got {pts}")
    if err_hw > errs[0]:
        return gs[0], f"outside sweep (g < {gs[0]:g}), clipped"
    if err_hw < errs[-1]:
        return gs[-1], f"outside sweep (g > {gs[-1]:g}), clipped"
    g = float(np.interp(err_hw, errs[::-1], gs[::-1]))
    return g, "in sweep"
