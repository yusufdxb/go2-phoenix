"""Analyze the EXPLORATORY gantry A/B campaign (docs/preregistration/2026-10-07-gantry-ab-exploratory.md).

Reads only raw run logs (bridge.jsonl, policy.parquet, walk_cmd.jsonl) plus the operator
records the campaign runner wrote (campaign.jsonl). The arm of every run is taken from the
artifact hashes in its own bridge manifest, never from a label; a run whose hashes do not
match its planned arm is INVALID. Validity, metrics and the decision rule are the
preregistered ones; nothing is excluded by hand.

    python3 scripts/gantry_ab_analyze.py --campaign logs/gantry_ab/<id>/campaign.jsonl --out <dir>

Writes <dir>/runs.json (every attempt, every metric), <dir>/summary.json and <dir>/report.md.
Exit 0 always when the analysis ran; the verdict is in the files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from phoenix.sim2real import actuator_check as ac  # noqa: E402

ARMS = {
    "A": {
        "checkpoint": "e03013d19ccc5df92a58762f8b76566a745c131534e8949ff1275739378198e4",
        "policy.onnx": "6d36796f857a490b3d446492581aa1952fa9afdb689bef88b09410db10512c32",
        "policy.onnx.data": "b6ec50ed6e295aecc0bb3b6d801d3a089c890af491956b1f4f1358c25079797b",
        "sim_label": "I",  # results/friction_dr/I = the same ONNX, byte-identical export
        "sim_sag": "configs/sim2real/e2/walk_iter2750/ref_nominal.json",
    },
    "B": {
        "checkpoint": "9948900b4568cb09bd0119b58c395f8aa0309179cf5467ac43bae0cd4945ba9a",
        "policy.onnx": "4b7063b47bac0b36d4696258da613da08a14cebe57c05f026d88aca73872034f",
        "policy.onnx.data": "13946b354f0dc7cfb92951164f4e88ccfa01b4ed13ccaa6bce243f65f394e00a",
        "sim_label": "fric_T_s50",
        "sim_sag": "configs/sim2real/e2/walk_fric_s50_3749/ref_nominal.json",
    },
}
NORMAL_END_FAULTS = {"policy_abort:authority_window_complete", "estop_asserted", "bridge_shutdown"}
PRESETS = ("vx03_4s", "yaw05_3s")
SETTLE_AFTER_CMD_S = 1.0
YAW_SETTLE_RAD_S = 0.1
YAW_SETTLE_HOLD_S = 0.5
OSC_BAND_HZ = (4.0, 15.0)
LOOP_GAP_S = 0.010
POLICY_GAP_S = 0.030
MIN_VALID = 2
SEPARATION_MIN = 0.15
OVERSHOOT_MAX = 1.3
TILT_MARGIN_DEG = 2.0
DQ_RATIO_MAX = 1.5


# ------------------------------------------------------------------ loading


def load_bridge(run_dir: Path) -> tuple[dict, list[dict], dict | None]:
    manifest, ticks, end = None, [], None
    with (run_dir / "bridge.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            kind = r.get("record")
            if kind == "manifest":
                manifest = r
            elif kind == "tick":
                ticks.append(r)
            elif kind == "end":
                end = r
    if manifest is None:
        raise ValueError(f"{run_dir}: bridge.jsonl has no manifest")
    return manifest, ticks, end


def arm_of(manifest: dict) -> str | None:
    art = manifest.get("artifacts") or {}
    got = {
        k: (art.get(k) or {}).get("sha256")
        for k in ("checkpoint", "policy.onnx", "policy.onnx.data")
    }
    for arm, want in ARMS.items():
        if all(got[k] == want[k] for k in got):
            return arm
    return None


def quat_yaw(q: np.ndarray) -> np.ndarray:
    """Yaw of quaternions stored x, y, z, w (policy.parquet base_quat; checked by unit norm
    and by reproducing the 2026-10-07 IMU yaw in the self-check)."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def load_policy(run_dir: Path) -> dict[str, np.ndarray] | None:
    p = run_dir / "policy.parquet"
    if not p.is_file():
        return None
    import pandas as pd

    df = pd.read_parquet(p)
    return {
        "t": df["timestamp_s"].to_numpy(dtype=float),
        "quat": np.stack(df["base_quat"].to_numpy()).astype(float),
        "gyro": np.stack(df["base_ang_vel_body"].to_numpy()).astype(float),
        "cmd": np.stack(df["command_vel"].to_numpy()).astype(float),
        "odom_valid": df["odom_valid"].to_numpy(dtype=bool),
        "base_pos": np.stack(df["base_pos"].to_numpy()).astype(float),
    }


def walk_summary(run_dir: Path) -> dict | None:
    p = run_dir / "walk_cmd.jsonl"
    if not p.is_file():
        return None
    out = None
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r.get("record") == "summary":
            out = r
    return out


# ------------------------------------------------------------------ metrics


def _t(ticks: list[dict]) -> np.ndarray:
    return np.array([t["t_mono_ns"] for t in ticks], dtype=float) / 1e9


def _pol(vals: Any) -> np.ndarray:
    m = ac.unitree_to_policy(vals)
    return np.array([m[n] for n in ac.POLICY_JOINT_ORDER])


def _group_idx(group: str) -> list[int]:
    return [i for i, n in enumerate(ac.POLICY_JOINT_ORDER) if ac.joint_group(n) == group]


def bridge_metrics(ticks: list[dict], end: dict | None) -> dict:
    live = [t for t in ticks if t["mode"] != "silent"]
    pol_i = [i for i, t in enumerate(ticks) if t["mode"] == "policy"]
    out: dict[str, Any] = {"n_ticks": len(ticks), "n_policy_ticks": len(pol_i)}
    out["one_policy_episode"] = bool(pol_i) and pol_i[-1] - pol_i[0] + 1 == len(pol_i)
    pol = [ticks[i] for i in pol_i]
    out["lowstate_fresh_all_policy"] = (
        all(bool(t.get("lowstate_fresh")) for t in pol) if pol else False
    )
    faults = set()
    for t in ticks:
        faults.update(t.get("faults") or [])
    if end:
        faults.update(end.get("faults") or [])
    out["faults"] = sorted(faults)
    out["abnormal_faults"] = sorted(f for f in faults if f not in NORMAL_END_FAULTS)
    out["end_reason"] = (end or {}).get("reason")
    out["slew_clip_ticks_policy"] = int(sum(any(t.get("slew_clip") or []) for t in pol))
    out["limit_clip_ticks_policy"] = int(sum(any(t.get("limit_clip") or []) for t in pol))
    tm = [min(t["torque_margin_nm"]) for t in pol if t.get("torque_margin_nm")]
    out["min_torque_margin_nm"] = float(min(tm)) if tm else None
    tilts = [abs(t["tilt_deg"]) for t in pol if t.get("tilt_deg") is not None]
    out["max_tilt_deg_policy"] = float(max(tilts)) if tilts else None
    dt = np.diff(_t(live)) if len(live) > 1 else np.array([])
    out["bridge_dt_s"] = (
        {
            "p50": float(np.median(dt)),
            "p99": float(np.percentile(dt, 99)),
            "max": float(dt.max()),
            "n_gt_10ms": int((dt > LOOP_GAP_S).sum()),
        }
        if dt.size
        else None
    )
    for k in ("lowstate_age_s", "cmd_age_s", "imu_age_s"):
        v = [t[k] for t in pol if t.get(k) is not None]
        out[f"max_{k}"] = float(max(v)) if v else None
    modes = [t["mode"] for t in ticks]
    tt = _t(ticks)
    # STOP: deadman released (estop asserted) while the policy had authority
    est = [i for i in pol_i if ticks[i].get("estop_value")]
    if est:
        j = next((k for k in range(est[0], len(ticks)) if modes[k] != "policy"), None)
        out["stop_latency_s"] = float(tt[j] - tt[est[0]]) if j is not None else None
    else:
        out["stop_latency_s"] = None
    if pol_i:
        k_sit = next((k for k in range(pol_i[-1], len(ticks)) if modes[k] == "sit"), None)
        out["window_end_to_sit_s"] = float(tt[k_sit] - tt[pol_i[-1]]) if k_sit is not None else None
    return out


def motion_window(pol: dict[str, np.ndarray]) -> tuple[int, int, int] | None:
    """(i0, i1, i_end) on the policy record: last zero row before motion, last moving row,
    first row >= 1.0 s after the command returned to zero (clipped to the record)."""
    moving = np.abs(pol["cmd"]).sum(axis=1) > 0
    if not moving.any():
        return None
    first = int(np.argmax(moving))
    last = int(len(moving) - 1 - np.argmax(moving[::-1]))
    i0 = max(first - 1, 0)
    t_end = pol["t"][last] + SETTLE_AFTER_CMD_S
    i_end = int(min(np.searchsorted(pol["t"], t_end), len(pol["t"]) - 1))
    return i0, last, i_end


def walk_metrics(pol: dict[str, np.ndarray], preset: str, tape: dict) -> dict:
    out: dict[str, Any] = {}
    w = motion_window(pol)
    if w is None:
        return {"error": "no non-zero command in policy.parquet"}
    i0, i1, ie = w
    t, cmd, gz = pol["t"], pol["cmd"], pol["gyro"][:, 2]
    dt = np.diff(t, prepend=t[0])
    norms = np.linalg.norm(pol["quat"], axis=1)
    out["quat_norm_minmax"] = [float(norms.min()), float(norms.max())]
    yaw = quat_yaw(pol["quat"])
    cmd_yaw = float(np.sum(cmd[i0 + 1 : i1 + 1, 2] * dt[i0 + 1 : i1 + 1]))
    cmd_fwd = float(np.sum(cmd[i0 + 1 : i1 + 1, 0] * dt[i0 + 1 : i1 + 1]))
    out["commanded_yaw_deg"] = math.degrees(cmd_yaw)
    out["commanded_fwd_m"] = cmd_fwd
    out["imu_yaw_deg"] = math.degrees(float(yaw[ie] - yaw[i0]))
    out["gyro_yaw_deg"] = math.degrees(float(np.sum(gz[i0 + 1 : ie + 1] * dt[i0 + 1 : ie + 1])))
    out["policy_dt_s"] = {
        "p99": float(np.percentile(np.diff(t), 99)),
        "max": float(np.diff(t).max()),
        "n_gt_30ms": int((np.diff(t) > POLICY_GAP_S).sum()),
    }
    if preset.startswith("yaw"):
        out["Y"] = (yaw[ie] - yaw[i0]) / cmd_yaw if abs(cmd_yaw) > 1e-6 else None
        k = max(1, int(round(0.1 / max(np.median(np.diff(t)), 1e-3))))
        ma = np.convolve(gz, np.ones(k) / k, mode="same")
        wz_cmd = float(np.max(np.abs(cmd[i0 + 1 : i1 + 1, 2])))
        out["yaw_rate_overshoot"] = (
            float(np.max(np.abs(ma[i0 + 1 : i1 + 1])) / wz_cmd) if wz_cmd > 0 else None
        )
        settle = None
        for j in range(i1 + 1, len(t)):
            span = (t >= t[j]) & (t <= t[j] + YAW_SETTLE_HOLD_S)
            if t[-1] - t[j] < YAW_SETTLE_HOLD_S:
                break
            if np.all(np.abs(gz[span]) < YAW_SETTLE_RAD_S):
                settle = float(t[j] - t[i1])
                break
        out["yaw_settling_s"] = settle
    else:
        out["yaw_drift_deg"] = out["imu_yaw_deg"]
        tf = tape.get("tape_fwd_m")
        out["F"] = (float(tf) / cmd_fwd) if tf is not None and cmd_fwd > 1e-6 else None
        out["tape_lat_m"] = tape.get("tape_lat_m")
        if pol["odom_valid"][i0] and pol["odom_valid"][ie]:
            out["odom_fwd_m"] = float(
                np.linalg.norm(pol["base_pos"][ie, :2] - pol["base_pos"][i0, :2])
            )
        else:
            out["odom_fwd_m"] = None  # not valid on this run; never substituted for tape
    return out


def joint_tracking(ticks: list[dict]) -> dict:
    """During motion (non-zero velocity fed to the policy): RMS(final target - q) per group
    and swing ratio per joint (p95-p5 of q / of target), as in the MuJoCo replay."""
    mv = [
        t
        for t in ticks
        if t["mode"] == "policy"
        and t.get("policy")
        and any(abs(v or 0.0) > 0 for v in (t["policy"].get("velocity_command_fed") or []))
    ]
    if len(mv) < 10:
        return {"n": len(mv)}
    q = np.stack([_pol(t["q_unitree"]) for t in mv])
    tg = np.stack([_pol(t["final_target_unitree"]) for t in mv])
    rms = np.sqrt(np.mean((tg - q) ** 2, axis=0))
    sw = lambda x: np.percentile(x, 95, axis=0) - np.percentile(x, 5, axis=0)  # noqa: E731
    ratio = sw(q) / np.maximum(sw(tg), 1e-6)
    out = {"n": len(mv)}
    for g in ("hip", "thigh", "calf"):
        idx = _group_idx(g)
        out[f"rms_err_{g}"] = float(np.mean(rms[idx]))
        out[f"swing_ratio_{g}"] = float(np.mean(ratio[idx]))
    return out


def osc_fraction(t: np.ndarray, dq: np.ndarray) -> float | None:
    """Max over joints of the dq power fraction in OSC_BAND_HZ (uniform 400 Hz resample)."""
    if len(t) < 64 or t[-1] - t[0] < 1.0:
        return None
    tu = np.arange(t[0], t[-1], 1 / 400.0)
    best = 0.0
    for j in range(dq.shape[1]):
        x = np.interp(tu, t, dq[:, j])
        x = x - x.mean()
        p = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / 400.0)
        tot = p[1:].sum()
        if tot > 0:
            best = max(best, float(p[(f >= OSC_BAND_HZ[0]) & (f <= OSC_BAND_HZ[1])].sum() / tot))
    return best


def stand_metrics(ticks: list[dict]) -> dict:
    """E2-definition calf sag (same function family as the sim reference) + posture."""
    try:
        window, report = ac.settle_window(ticks, label="G")
    except ac.Refusal as exc:
        return {"settle_window_ok": False, "refusal": str(exc)}
    summ = ac.summarize_hw([ticks], labels=["G"])
    calf = [n for n in ac.POLICY_JOINT_ORDER if ac.joint_group(n) == "calf"]
    q = np.stack([_pol(t["q_unitree"]) for t in window])
    tg = np.stack([_pol(t["final_target_unitree"]) for t in window])
    dq = np.stack([_pol(t["dq_unitree"]) for t in window])
    ci = _group_idx("calf")
    return {
        "settle_window_ok": True,
        "window_s": report.get("window_s"),
        "calf_sag_median": float(np.mean([summ["err"][n] for n in calf])),
        "calf_sag_mean": float(np.mean(tg[:, ci] - q[:, ci])),
        "calf_sag_per_joint": {n: summ["err"][n] for n in calf},
        "dq_rms_max_calf": float(max(summ["dq_rms"][n] for n in calf)),
        "dq_rms_max_all": float(max(summ["dq_rms"].values())),
        "max_tilt_deg": float(max(abs(t["tilt_deg"]) for t in window)),
        "osc_frac_4_15hz": osc_fraction(_t(window), dq),
    }


# ------------------------------------------------------------------ campaign


def analyze_run(rec: dict) -> dict:
    run_dir = Path(rec["run_dir"])
    slot = rec["slot"]
    out: dict[str, Any] = {
        k: rec.get(k) for k in ("block", "arm", "slot", "attempt", "physical_attempt", "run_dir")
    }
    try:
        manifest, ticks, end = load_bridge(run_dir)
    except (OSError, ValueError) as exc:
        out.update(valid=False, why=[f"unreadable: {exc}"])
        return out
    measured_arm = arm_of(manifest)
    out["measured_arm"] = measured_arm
    out["bridge"] = bridge_metrics(ticks, end)
    why = []
    if measured_arm != rec.get("arm"):
        why.append(f"artifact hashes are arm {measured_arm}, planned {rec.get('arm')}")
    if not manifest.get("live") or manifest.get("code_identity_problems") != []:
        why.append("not live or code identity problems")
    ident = manifest.get("code_identity") or {}
    # Amendment 1: only the payload's verified sync counts as hardware. A localhost
    # rehearsal is also launched --live, but its identity source is the workstation git.
    if ident.get("source") != "payload_sync" or ident.get("manifest_verified") is not True:
        why.append(f"not a verified payload run (code identity source {ident.get('source')!r})")
    if rec.get("rehearsal"):
        why.append("runner recorded PHOENIX_REHEARSAL=1")
    if rec.get("campaign_commit") and ident.get("sha") != rec["campaign_commit"]:
        why.append(
            f"code {str(ident.get('sha'))[:12]} is not the campaign commit {rec['campaign_commit'][:12]}"
        )
    b = out["bridge"]
    if not b["one_policy_episode"]:
        why.append("not exactly one policy episode")
    if not b["lowstate_fresh_all_policy"]:
        why.append("LowState not fresh on every policy tick")
    if b["abnormal_faults"]:  # Amendment 2: the STOP drill is not exempt
        why.append(f"faults {b['abnormal_faults']}")
    if (
        rec.get("rope_loaded")
        or rec.get("touched")
        or (rec.get("aborted") and slot != "stop_drill")
    ):
        why.append("operator: rope loaded / touched / aborted")
    if (rec.get("physical_attempt") or 1) > 2:
        why.append(f"physical attempt {rec.get('physical_attempt')} of this slot (max 2)")
    if slot in PRESETS:
        summ = walk_summary(run_dir)
        if not summ or summ.get("outcome") != "complete":
            why.append(f"walk_cmd outcome {None if not summ else summ.get('outcome')}")
        pol = load_policy(run_dir)
        if pol is None:
            why.append("no policy.parquet")
        else:
            out["walk"] = walk_metrics(pol, slot, rec)
            if slot == "vx03_4s" and rec.get("tape_fwd_m") is None:
                why.append("no tape reading")
        out["tracking"] = joint_tracking(ticks)
    elif slot in ("G", "stop_drill"):
        out["stand"] = stand_metrics(ticks)
        if slot == "G" and not out["stand"].get("settle_window_ok"):
            why.append("E2 settle window refused")
    out["safety_event"] = bool(
        b["abnormal_faults"]
        or b["limit_clip_ticks_policy"]
        or (rec.get("aborted") and slot != "stop_drill")
    )
    out["valid"] = not why
    out["why"] = why
    return out


def number_physical_attempts(run_recs: list[dict]) -> None:
    """Amendment 2: the attempt index is the record's position among ALL records of its
    (block, slot) in campaign-file order, so a re-run block cannot reset it."""
    count: dict[tuple, int] = {}
    for r in run_recs:
        key = (r.get("block"), r.get("slot"))
        count[key] = count.get(key, 0) + 1
        r["physical_attempt"] = count[key]


def first_valid(runs: list[dict]) -> list[dict]:
    """First valid attempt per (block, slot); attempts beyond the second are not allowed."""
    chosen, seen = [], set()
    for r in sorted(
        runs,
        key=lambda r: (r["block"], r["slot"], r.get("physical_attempt") or r.get("attempt") or 1),
    ):
        key = (r["block"], r["slot"])
        n = r.get("physical_attempt") or r.get("attempt") or 1
        if key in seen or not r["valid"] or n > 2:
            continue
        seen.add(key)
        chosen.append(r)
    return chosen


def ladder_events(ladder_recs: list[dict], base: Path) -> list[dict]:
    """Amendment 2: abnormal bridge faults in live ladder stages count as safety events of
    the arm whose artifacts the bridge ran (from the manifest hashes)."""
    out = []
    for r in ladder_recs:
        d = Path(r.get("run_dir") or "")
        if not d.is_absolute():
            d = (base / d) if (base / d).exists() else REPO / d
        if not (d / "bridge.jsonl").is_file():
            continue
        try:
            manifest, ticks, end = load_bridge(d)
        except (OSError, ValueError):
            continue
        ident = manifest.get("code_identity") or {}
        if not manifest.get("live") or ident.get("source") != "payload_sync":
            continue
        b = bridge_metrics(ticks, end)
        if b["abnormal_faults"] or b["limit_clip_ticks_policy"]:
            out.append(
                {
                    "stage": r.get("stage"),
                    "arm": arm_of(manifest),
                    "faults": b["abnormal_faults"],
                    "limit_clip_ticks": b["limit_clip_ticks_policy"],
                    "run_dir": str(d),
                }
            )
    return out


def stop_conditions(runs: list[dict]) -> list[str]:
    """Campaign STOP conditions computable from the logs (reported; the operator stops live)."""
    found = []
    aborts: dict[tuple, int] = {}
    for r in runs:
        b = r.get("bridge") or {}
        tag = f"block {r.get('block')} {r.get('slot')} arm {r.get('measured_arm')}"
        if b.get("abnormal_faults"):
            found.append(f"{tag}: faults {b['abnormal_faults']}")
        if b.get("limit_clip_ticks_policy"):
            found.append(f"{tag}: limit clip under the policy")
        if r.get("slot") in PRESETS and any("aborted" in w for w in r.get("why") or []):
            key = (r.get("arm"), r.get("slot"))
            aborts[key] = aborts.get(key, 0) + 1
    found += [f"arm {a} {p}: {n} aborted runs" for (a, p), n in aborts.items() if n >= 2]
    return found


def decide(chosen: list[dict], all_runs: list[dict], ladder: list[dict] | None = None) -> dict:
    vals = {arm: {p: [] for p in PRESETS} for arm in ARMS}
    for r in chosen:
        if r["slot"] == "yaw05_3s" and r["walk"].get("Y") is not None:
            vals[r["arm"]]["yaw05_3s"].append(float(r["walk"]["Y"]))
        if r["slot"] == "vx03_4s" and r["walk"].get("F") is not None:
            vals[r["arm"]]["vx03_4s"].append(float(r["walk"]["F"]))
    out: dict[str, Any] = {"values": vals}
    if any(len(vals[a][p]) < MIN_VALID for a in ARMS for p in PRESETS):
        out["verdict"] = "INSUFFICIENT"
        return out
    per = {}
    for p in PRESETS:
        a, b = np.array(vals["A"][p]), np.array(vals["B"][p])
        per[p] = {
            "median_A": float(np.median(a)),
            "median_B": float(np.median(b)),
            "delta": float(np.median(b) - np.median(a)),
            "B_separated_above": bool(b.min() > a.max()),
            "B_separated_below": bool(b.max() < a.min()),
        }
    out["per_preset"] = per
    ev = {
        arm: any(r["safety_event"] for r in all_runs if r.get("measured_arm") == arm)
        or any(e["arm"] == arm for e in ladder or [])
        for arm in ARMS
    }
    out["safety_event"] = ev
    stands = {
        arm: [r["stand"] for r in chosen if r["slot"] == "G" and r["arm"] == arm] for arm in ARMS
    }
    ovs = [
        r["walk"].get("yaw_rate_overshoot")
        for r in chosen
        if r["arm"] == "B" and r["slot"] == "yaw05_3s"
    ]
    stand_ok = bool(stands["A"] and stands["B"]) and (
        max(s["max_tilt_deg"] for s in stands["B"])
        <= max(s["max_tilt_deg"] for s in stands["A"]) + TILT_MARGIN_DEG
        and max(s["dq_rms_max_calf"] for s in stands["B"])
        <= DQ_RATIO_MAX * max(s["dq_rms_max_calf"] for s in stands["A"])
    )
    better = (
        all(per[p]["B_separated_above"] and per[p]["delta"] >= SEPARATION_MIN for p in PRESETS)
        and not (ev["B"] and not ev["A"])
        and all(o is not None and o <= OVERSHOOT_MAX for o in ovs)
        and stand_ok
    )
    worse = any(
        per[p]["B_separated_below"] and per[p]["delta"] <= -SEPARATION_MIN for p in PRESETS
    ) or (ev["B"] and not ev["A"])
    out["stand_not_worse"] = stand_ok
    out["B_yaw_overshoots"] = ovs
    out["verdict"] = (
        "B OBVIOUSLY WORSE" if worse else "B OBVIOUSLY BETTER" if better else "INCONCLUSIVE"
    )
    return out


def model_id(chosen: list[dict]) -> dict:
    out: dict[str, Any] = {}
    for arm, spec in ARMS.items():
        rows = {}
        p = REPO / "results/friction_dr" / spec["sim_label"] / "replay" / "fric_results.json"
        if p.is_file():
            for r in json.loads(p.read_text()):
                if r.get("preset") == "yaw05_3s" and "yaw_deg" in r:
                    rows.setdefault(r["label"], {})["Y_sim"] = r["yaw_deg"] / 86.0
                if r.get("preset") == "vx03_4s" and "fwd_m" in r:
                    rows.setdefault(r["label"], {})["F_sim"] = r["fwd_m"] / 1.2
        sim_sag = None
        sp = REPO / spec["sim_sag"]
        if sp.is_file():
            ref = json.loads(sp.read_text())
            sim_sag = ac.group_mean_err(ref, "calf")
        hw_y = [
            r["walk"]["Y"]
            for r in chosen
            if r["arm"] == arm and r["slot"] == "yaw05_3s" and r["walk"].get("Y") is not None
        ]
        hw_f = [
            r["walk"]["F"]
            for r in chosen
            if r["arm"] == arm and r["slot"] == "vx03_4s" and r["walk"].get("F") is not None
        ]
        hw_sag = [
            r["stand"]["calf_sag_median"] for r in chosen if r["arm"] == arm and r["slot"] == "G"
        ]
        out[arm] = {
            "sim_replay": rows,
            "hw_Y_median": float(np.median(hw_y)) if hw_y else None,
            "hw_F_median": float(np.median(hw_f)) if hw_f else None,
            "sim_calf_sag": sim_sag,
            "hw_calf_sag_median": float(np.median(hw_sag)) if hw_sag else None,
            "hw_minus_sim_calf_sag": (
                (float(np.median(hw_sag)) - sim_sag) if hw_sag and sim_sag is not None else None
            ),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    recs = [json.loads(ln) for ln in args.campaign.read_text().splitlines() if ln.strip()]
    run_recs = [r for r in recs if r.get("kind") == "run"]
    header = next((r for r in recs if r.get("kind") == "header"), {})
    for r in run_recs:
        r.setdefault("campaign_commit", header.get("commit"))
    base = args.campaign.parent
    for r in run_recs:
        if not Path(r["run_dir"]).is_absolute():
            r["run_dir"] = (
                str((base / r["run_dir"]).resolve())
                if (base / r["run_dir"]).exists()
                else str(REPO / r["run_dir"])
            )
    number_physical_attempts(run_recs)
    runs = [analyze_run(r) for r in run_recs]
    chosen = first_valid(runs)
    ladder = ladder_events([r for r in recs if r.get("kind") == "ladder"], base)
    summary = {
        "prereg": "docs/preregistration/2026-10-07-gantry-ab-exploratory.md",
        "n_attempts": len(runs),
        "n_valid": sum(r["valid"] for r in runs),
        "invalid": [
            {k: r.get(k) for k in ("block", "arm", "slot", "attempt", "run_dir", "why")}
            for r in runs
            if not r["valid"]
        ],
        "decision": decide(chosen, runs, ladder),
        "ladder_safety_events": ladder,
        "stop_conditions_detected": stop_conditions(runs),
        "model_identification": model_id(chosen),
        "stop": [
            {k: r.get(k) for k in ("block", "arm", "slot")}
            | {
                "stop_latency_s": r["bridge"]["stop_latency_s"],
                "window_end_to_sit_s": r["bridge"]["window_end_to_sit_s"],
            }
            for r in runs
            if "bridge" in r
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "runs.json").write_text(json.dumps(runs, indent=2, default=str))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    lines = [
        f"# Gantry A/B analysis ({args.campaign})",
        "",
        f"Verdict (exploratory): **{summary['decision']['verdict']}**",
        "",
    ]
    lines += [
        "| block | arm | slot | att | valid | Y | F | overshoot | calf sag | max tilt | why |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        w, s = r.get("walk") or {}, r.get("stand") or {}
        f = lambda v: "" if v is None else f"{v:.3f}"  # noqa: E731
        lines.append(
            f"| {r.get('block')} | {r.get('measured_arm')} | {r.get('slot')} | {r.get('attempt')} | {r['valid']} | {f(w.get('Y'))} | {f(w.get('F'))} | "
            f"{f(w.get('yaw_rate_overshoot'))} | {f(s.get('calf_sag_median'))} | {f(s.get('max_tilt_deg'))} | {'; '.join(r.get('why') or [])} |"
        )
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(summary["decision"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
