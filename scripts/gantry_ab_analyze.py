"""Analyze the EXPLORATORY gantry A/B campaign (docs/preregistration/2026-10-07-gantry-ab-exploratory.md).

Reads only raw run logs (bridge.jsonl, policy.parquet, walk_cmd.jsonl) plus the operator
records the campaign runner wrote (campaign.jsonl). The arm of every run is taken from the
artifact hashes in its own bridge manifest, never from a label; a run whose hashes do not
match its planned arm is INVALID. Validity, metrics and the decision rule are the
preregistered ones; nothing is excluded by hand.

    python3 scripts/gantry_ab_analyze.py --campaign logs/gantry_ab/<id>/campaign.jsonl --out <dir>

Arms: the hard-coded A and B below unless the campaign header pins its own (``arms``, written by
scripts/hw_campaign.sh init from the campaign's arms file), so a 3-arm campaign needs no edit here.
The preregistered A-vs-B verdict is computed for arms A and B only; every arm gets the per-preset
tracking table (measured / commanded, mean and spread over the valid repeats) and the stop summary.

Writes <dir>/runs.json (every attempt, every metric), <dir>/summary.json and <dir>/report.md.
Exit 0 always when the analysis ran; the verdict is in the files.

Who typed the operator answers: run records carry ``answers_entered_by`` (operator |
agent_relaying_operator | agent_unconfirmed; hw_campaign.sh asks it after every run). Older
records do not; ``--corrections FILE`` sets it (and any other field) after the fact, keyed by
the record's harness_log, and every correction is listed in runs.json. A tape or yaw reading
entered as agent_unconfirmed is not a measurement and is dropped; touched / rope / aborted
answers entered that way are reported as unconfirmed, and such a walk counts only if the
operator's own remote judgement (A = yes) is on file.

Forward displacement is the tape. ``odom_leg_fwd_m`` is a separate ESTIMATE from kinematic
leg odometry (phoenix.sim2real.leg_odometry), reported beside the tape and never in its place.
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
from phoenix.sim2real import leg_odometry as lo  # noqa: E402
from phoenix.sim2real.walk_command import PRESETS as WALK_PRESETS  # noqa: E402

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
# Amendment 3: a deadman release latches policy_abort:external_estop in the policy node
# (ros2_policy_node.py _latch_abort("external_estop"), recorded by actuator_gate as
# policy_abort:<reason>). WHEN the release happened is judged from the ticks instead.
NORMAL_END_FAULTS = {
    "policy_abort:authority_window_complete",
    "policy_abort:external_estop",
    "estop_asserted",
    "bridge_shutdown",
}
# Amendment 4: the bridge never logs estop true on a policy tick; a release (and a planned
# end) is only observable on the FIRST tick after the policy episode, from these causes.
AUTH_END = "policy_abort:authority_window_complete"
RELEASE_CAUSES = {"policy_abort:external_estop", "estop_asserted"}
PHASE_TOL_S = 0.1  # real planned ends: policy episode within 17 ms of --authority-s
STAND_AUTHORITY_S = {"F": 2.0, "G": 5.0, "H": 10.0, "stop_drill": 5.0}  # harness_preflight.sh
WALK_AUTHORITY_MARGIN_S = 4.0  # harness_preflight.sh PHOENIX_WALK_AUTHORITY_MARGIN_S default
PRESETS = ("vx03_4s", "yaw05_3s")  # the preregistered A/B decision presets
WALK_SLOTS = tuple(WALK_PRESETS)  # every slot that walks (metrics, tape, abort handling)
DECISION_ARMS = ("A", "B")
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
        "q": np.stack(df["joint_pos"].to_numpy()).astype(float),
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
    if pol_i:
        k_sit = next((k for k in range(pol_i[-1], len(ticks)) if modes[k] == "sit"), None)
        out["window_end_to_sit_s"] = float(tt[k_sit] - tt[pol_i[-1]]) if k_sit is not None else None
    return out


def expected_authority_s(slot: str | None) -> float | None:
    """The --authority-s the harness passes for this slot or ladder stage (None if none)."""
    if slot in WALK_SLOTS:
        from phoenix.sim2real.walk_command import profile_duration

        return float(profile_duration(slot)) + WALK_AUTHORITY_MARGIN_S
    return STAND_AUTHORITY_S.get(slot or "")


def _zero_command_tick(ticks: list[dict], k0: int) -> int | None:
    """First tick k >= k0 of the hold that starts at k0 whose published command moves nothing:
    no policy target, the target is the posture measured at hold entry (or limit-clipped),
    and the next tick publishes the same target, also in hold. None if never shown."""
    if k0 >= len(ticks) or ticks[k0].get("mode") != "hold":
        return None
    q_entry = ticks[k0].get("q_unitree")
    clip = ticks[k0].get("limit_clip") or [False] * 12
    k = k0
    while k + 1 < len(ticks) and ticks[k].get("mode") == "hold":
        a, b = ticks[k], ticks[k + 1]
        ft = a.get("final_target_unitree")
        if (
            b.get("mode") == "hold"
            and a.get("publish") is True
            and b.get("publish") is True
            and a.get("requested_target_unitree") is None
            and b.get("requested_target_unitree") is None
            and ft is not None
            and q_entry is not None
            and len(ft) == len(q_entry) == len(clip)
            and ft == b.get("final_target_unitree")
            and all(f == q or c for f, q, c in zip(ft, q_entry, clip, strict=True))
        ):
            return k
        k += 1
    return None


def policy_exit(ticks: list[dict], authority_s: float | None) -> dict:
    """Amendment 4: classify how the single policy episode ended, from the transition between
    its last tick and the FIRST subsequent tick (the only place a release is observable).

    exit: planned_end | deadman_release | fault_exit | no_policy_episode |
    release_without_policy_episode | malformed. Any entry in ``problems`` means the timing or
    the class is not established from the log (fail closed: the run is invalid)."""
    probs: list[str] = []
    out: dict[str, Any] = {"exit": None, "problems": probs, "authority_s": authority_s}
    for k in ("k_policy_last", "t_policy_last_ns", "t_release_obs_ns", "t_release_rx_ns"):
        out[k] = None
    for k in ("t_zero_ns", "stop_latency_s", "release_window_s", "release_rx_to_zero_s"):
        out[k] = None
    ns = [t.get("t_mono_ns") for t in ticks]
    if not all(isinstance(v, int) for v in ns):
        probs.append("tick without an integer t_mono_ns")
        out["exit"] = "malformed"
        return out
    if any(b <= a for a, b in zip(ns, ns[1:], strict=False)):
        probs.append("t_mono_ns not strictly increasing (duplicate or reversed tick)")
    tk = [t.get("tick") for t in ticks]
    if not all(isinstance(v, int) for v in tk) or any(
        b != a + 1 for a, b in zip(tk, tk[1:], strict=False)
    ):
        probs.append("tick counter missing or not consecutive (a record is missing)")
    pol_i = [i for i, t in enumerate(ticks) if t.get("mode") == "policy"]
    if not pol_i:
        released = any(t.get("estop_value") is True for t in ticks)
        out["exit"] = "release_without_policy_episode" if released else "no_policy_episode"
        return out
    k0, k1 = pol_i[0], pol_i[-1]
    if k1 - k0 + 1 != len(pol_i):
        probs.append("more than one policy episode")
        out["exit"] = "malformed"
        return out
    out["k_policy_last"] = k1
    out["t_policy_last_ns"] = ns[k1]
    out["policy_episode_s"] = (ns[k1] - ns[k0]) / 1e9
    not_held = [k for k in pol_i if ticks[k].get("estop_value") is not False]
    if not_held:
        probs.append(f"deadman not shown held (estop not false) on policy tick(s) {not_held[:3]}")
    if any(ticks[k].get("estop_value") is True for k in range(k0)):
        probs.append("deadman released before the policy episode")
    if k1 + 1 >= len(ticks):
        probs.append("first post-policy tick missing (log ends under policy)")
        out["exit"] = "malformed"
        return out
    nxt = ticks[k1 + 1]
    est, cause = nxt.get("estop_value"), nxt.get("hold_cause")
    out["first_post_policy"] = {
        k: nxt.get(k) for k in ("tick", "mode", "estop_value", "hold_cause", "estop_age_s")
    }
    if est is None:
        probs.append("release signal missing: no estop_value on the first post-policy tick")
        out["exit"] = "malformed"
        return out
    if est is True and cause in RELEASE_CAUSES:
        out["exit"] = "deadman_release"
    elif est is False and cause == AUTH_END:
        out["exit"] = "planned_end"
    elif est is True or cause in RELEASE_CAUSES or cause == AUTH_END:
        probs.append(f"ambiguous transition: estop {est} with hold_cause {cause}")
        out["exit"] = "malformed"
        return out
    else:
        out["exit"] = "fault_exit"
        out["fault_cause"] = cause
    dur = out["policy_episode_s"]
    if out["exit"] == "planned_end":
        if authority_s is None:
            probs.append("planned end cannot be checked: no expected authority window")
        elif abs(dur - authority_s) > PHASE_TOL_S:
            probs.append(
                f"planned end after {dur:.3f} s of policy, authority window {authority_s:.3f} s"
            )
    if out["exit"] != "deadman_release":
        return out
    if authority_s is not None and dur >= authority_s + PHASE_TOL_S:
        probs.append(f"release after {dur:.3f} s, past the {authority_s:.3f} s authority window")
    t_obs = ns[k1 + 1]
    out["t_release_obs_ns"] = t_obs
    age = nxt.get("estop_age_s")
    if not isinstance(age, (int, float)) or not math.isfinite(age):
        probs.append("estop_age_s missing on the release tick")
    else:
        rx = t_obs - int(round(age * 1e9))
        out["t_release_rx_ns"] = rx
        if not ns[k1] < rx <= t_obs:
            probs.append(
                "release message receipt time is not between the last policy tick and t_release_obs"
            )
    if probs:
        return out  # nothing is estimated from an inconsistent transition
    out["release_window_s"] = (t_obs - ns[k1]) / 1e9
    kz = _zero_command_tick(ticks, k1 + 1)
    if kz is not None:
        out["t_zero_ns"] = ns[kz]
        out["stop_latency_s"] = (ns[kz] - t_obs) / 1e9
        out["release_rx_to_zero_s"] = (ns[kz] - out["t_release_rx_ns"]) / 1e9
    return out


def exit_is_clean(ex: dict, slot: str | None) -> bool:
    """The expected policy exit for the slot, fully established from the log."""
    if ex["problems"]:
        return False
    if slot == "stop_drill":
        return ex["exit"] == "deadman_release" and ex["t_zero_ns"] is not None
    return ex["exit"] == "planned_end"


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
    p = WALK_PRESETS.get(preset)
    has_yaw = (p.wz != 0) if p else preset.startswith("yaw")
    has_fwd = (p.vx != 0) if p else not has_yaw
    if has_yaw:
        out["Y"] = (yaw[ie] - yaw[i0]) / cmd_yaw if abs(cmd_yaw) > 1e-6 else None
        meas = tape.get("meas_yaw_deg")  # the operator's own reading, kept apart from the IMU Y
        out["Y_op"] = (
            math.radians(float(meas)) / cmd_yaw
            if meas is not None and abs(cmd_yaw) > 1e-6
            else None
        )
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
    if has_fwd:
        if not has_yaw:
            out["yaw_drift_deg"] = out["imu_yaw_deg"]
        tf = tape.get("tape_fwd_m")
        out["F"] = (float(tf) / cmd_fwd) if tf is not None and abs(cmd_fwd) > 1e-6 else None
        out["tape_lat_m"] = tape.get("tape_lat_m")
        if pol["odom_valid"][i0] and pol["odom_valid"][ie]:
            out["odom_fwd_m"] = float(
                np.linalg.norm(pol["base_pos"][ie, :2] - pol["base_pos"][i0, :2])
            )
        else:
            out["odom_fwd_m"] = None  # not valid on this run; never substituted for tape
        out.update(leg_odometry_metrics(pol, i0, i1, cmd_fwd))
    return out


def leg_odometry_metrics(pol: dict[str, np.ndarray], i0: int, i1: int, cmd_fwd: float) -> dict:
    """ESTIMATE (no-slip kinematic leg odometry), over the whole policy log like the tape
    (start mark -> end mark, rear feet), plus the forward-speed error while commanded."""
    if "q" not in pol:
        return {"odom_leg_fwd_m": None, "odom_leg_note": "no joint positions in the policy log"}
    ok = np.all(np.isfinite(pol["q"]), axis=1) & np.all(np.isfinite(pol["quat"]), axis=1)
    if ok.sum() < 10 or not ok[i0 : i1 + 1].all():
        return {"odom_leg_fwd_m": None, "odom_leg_note": "joint or IMU samples missing"}
    t, q, quat = pol["t"][ok], pol["q"][ok], pol["quat"][ok]
    tr = lo.integrate(t, q, quat)
    yaw0 = math.atan2(tr["R"][0, 1, 0], tr["R"][0, 0, 0])
    fwd = np.array([math.cos(yaw0), math.sin(yaw0), 0.0])
    left = np.array([-math.sin(yaw0), math.cos(yaw0), 0.0])
    rear = tr["feet"][:, [lo.LEGS.index("RL"), lo.LEGS.index("RR")], :].mean(axis=1)
    d = rear[-1] - rear[0]
    # forward speed along the initial heading, 0.2 s central difference, inside the command window
    j0, j1 = int(np.searchsorted(t, pol["t"][i0])), int(np.searchsorted(t, pol["t"][i1]))
    k = max(1, int(round(0.1 / max(float(np.median(np.diff(t))), 1e-3))))
    x = tr["base"] @ fwd
    v = (x[2 * k :] - x[: -2 * k]) / (t[2 * k :] - t[: -2 * k])
    vc = v[max(j0 - k, 0) : max(j1 - k, 0)]
    cc = pol["cmd"][ok][j0 + k : j1 + k, 0][: len(vc)]
    return {
        "odom_leg_kind": "ESTIMATE: kinematic leg odometry, no slip; not a measurement",
        "odom_leg_fwd_m": float(d @ fwd),
        "odom_leg_lat_m": float(d @ left),
        "F_odom": float(d @ fwd) / cmd_fwd if abs(cmd_fwd) > 1e-6 else None,
        "odom_leg_speed_rmse_mps": float(np.sqrt(np.mean((vc - cc) ** 2))) if len(vc) else None,
        "odom_leg_speed_mean_mps": float(np.mean(vc)) if len(vc) else None,
    }


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


def remote_judgement(run_dir: Path, slot: str | None) -> str | None:
    """The operator's own button answer on the physical remote (A yes / B no), if recorded."""
    for p in sorted(run_dir.glob("remote_*_judgement.json")):
        try:
            return json.loads(p.read_text()).get("result")
        except (OSError, ValueError):
            return None
    return None


def battery_record(rec: dict, run_dir: Path) -> dict:
    """State of charge the harness measured just before this run (its log), and the pack
    voltage the bridge logged on the policy ticks."""
    out: dict[str, Any] = {"soc_pre_pct": None, "power_v_median": None}
    log = Path(rec["harness_log"]) if rec.get("harness_log") else None
    if log is not None and log.is_file():
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("[battery]") and " battery " in line:
                try:
                    out["soc_pre_pct"] = float(line.split(" battery ", 1)[1].split("%")[0])
                except ValueError:
                    pass
    try:
        _, ticks, _ = load_bridge(run_dir)
    except (OSError, ValueError):
        return out
    v = [
        t["sensors"]["power_v"]
        for t in ticks
        if t.get("mode") == "policy"
        and isinstance(t.get("sensors"), dict)
        and t["sensors"].get("power_v")
    ]
    if v:
        out["power_v_median"] = float(np.median(v))
    return out


def apply_corrections(recs: list[dict], corrections: list[dict]) -> None:
    """Set fields on records after the fact, matched by harness_log; each change is kept."""
    for c in corrections:
        hit = [
            r
            for r in recs
            if r.get("kind") == c.get("kind", "run") and r.get("harness_log") == c["harness_log"]
        ]
        if len(hit) != 1:
            raise ValueError(f"correction {c['harness_log']!r} matches {len(hit)} records, not 1")
        r = hit[0]
        r.setdefault("corrections", []).append(
            {"set": c["set"], "was": {k: r.get(k) for k in c["set"]}, "reason": c["reason"]}
        )
        r.update(c["set"])


def analyze_run(rec: dict) -> dict:
    run_dir = Path(rec["run_dir"])
    slot = rec["slot"]
    out: dict[str, Any] = {
        k: rec.get(k) for k in ("block", "arm", "slot", "attempt", "physical_attempt", "run_dir")
    }
    by = rec.get("answers_entered_by") or "unrecorded"
    out["answers_entered_by"] = by
    out["corrections"] = rec.get("corrections", [])
    out["operator_remote_judgement"] = remote_judgement(run_dir, slot)
    out["battery"] = battery_record(rec, run_dir)
    if by == "agent_unconfirmed":
        dropped = [
            k for k in ("tape_fwd_m", "tape_lat_m", "meas_yaw_deg") if rec.get(k) is not None
        ]
        rec = {k: v for k, v in rec.items() if k not in dropped}
        out["unconfirmed_fields"] = ["touched", "aborted"] + (
            [] if rec.get("rope_loaded") is None else ["rope_loaded"]
        )
        out["dropped_unobserved_measurements"] = dropped
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
    if (
        by == "agent_unconfirmed"
        and slot in WALK_SLOTS
        and out["operator_remote_judgement"] != "yes"
    ):
        why.append(
            "answers typed by the agent without the operator, and no operator remote judgement 'yes'"
        )
    # Amendment 4: how the policy episode ended, from the transition to the first
    # non-policy tick (the operator's answer is metadata, never the only detector).
    ex = policy_exit(ticks, expected_authority_s(slot))
    out["policy_exit"] = ex
    if ex["problems"]:
        why.append("policy exit not established from the log: " + "; ".join(ex["problems"]))
    logs_abort = slot != "stop_drill" and ex["exit"] in (
        "deadman_release",
        "release_without_policy_episode",
    )
    stop_failed = False
    if slot == "stop_drill":
        if ex["exit"] != "deadman_release":
            why.append("STOP drill: the deadman was not released during policy authority")
        elif not ex["problems"] and ex["t_zero_ns"] is None:
            why.append("STOP drill FAILED: no zero-command hold shown after the release")
            stop_failed = True
    elif logs_abort:
        why.append(f"deadman released ({ex['exit']}): aborted run, from the logs")
    elif ex["exit"] == "fault_exit":
        why.append(f"policy episode ended by {ex.get('fault_cause')}, not the authority window")
    if (rec.get("physical_attempt") or 1) > 2:
        why.append(f"physical attempt {rec.get('physical_attempt')} of this slot (max 2)")
    if slot in WALK_SLOTS:
        summ = walk_summary(run_dir)
        if not summ or summ.get("outcome") != "complete":
            why.append(f"walk_cmd outcome {None if not summ else summ.get('outcome')}")
        pol = load_policy(run_dir)
        if pol is None:
            why.append("no policy.parquet")
        else:
            out["walk"] = walk_metrics(pol, slot, rec)
            if WALK_PRESETS[slot].vx != 0 and rec.get("tape_fwd_m") is None:
                why.append("no tape reading")
        out["tracking"] = joint_tracking(ticks)
    elif slot in ("G", "stop_drill"):
        out["stand"] = stand_metrics(ticks)
        if slot == "G" and not out["stand"].get("settle_window_ok"):
            why.append("E2 settle window refused")
    out["setup"] = rec.get("setup")  # gantry | floor as typed for W (None: older campaigns)
    out["operator_aborted"] = bool(rec.get("aborted"))
    out["aborted_from_logs"] = logs_abort
    # Logs win for detection; an operator-reported abort is never discarded.
    out["operator_log_discrepancy"] = (
        f"operator aborted={out['operator_aborted']}, logs show exit {ex['exit']}"
        if slot != "stop_drill" and out["operator_aborted"] != logs_abort
        else None
    )
    out["safety_event"] = bool(
        b["abnormal_faults"]
        or b["limit_clip_ticks_policy"]
        or (rec.get("aborted") and slot != "stop_drill")
        or logs_abort
        or ex["problems"]  # fail closed: an unestablished exit is treated as unsafe
        or ex["exit"] == "fault_exit"
        or stop_failed
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
        # Amendment 4: a live F/G policy episode that does not end as a planned end (a
        # deadman release under policy, a fault exit, or an unestablished exit) also counts.
        ex = policy_exit(ticks, expected_authority_s(r.get("stage")))
        had_policy = ex["exit"] not in ("no_policy_episode", "release_without_policy_episode")
        bad_exit = had_policy and not exit_is_clean(ex, r.get("stage"))
        if b["abnormal_faults"] or b["limit_clip_ticks_policy"] or bad_exit:
            out.append(
                {
                    "stage": r.get("stage"),
                    "arm": arm_of(manifest),
                    "faults": b["abnormal_faults"],
                    "limit_clip_ticks": b["limit_clip_ticks_policy"],
                    "policy_exit": ex["exit"] if had_policy else None,
                    "exit_problems": ex["problems"],
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
        if r.get("slot") in WALK_SLOTS and (
            r.get("operator_aborted") or r.get("aborted_from_logs")
        ):
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
    if not all(a in ARMS for a in DECISION_ARMS) or any(
        len(vals[a][p]) < MIN_VALID for a in DECISION_ARMS for p in PRESETS
    ):
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


def _spread(x: list[float]) -> dict:
    a = np.array(x, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
    }


TRACK_METRICS = ("F", "Y", "Y_op", "yaw_settling_s")


def tracking_table(chosen: list[dict]) -> dict:
    """arm -> slot -> metric -> mean/spread over the valid repeats. F is tape forward / commanded
    forward, Y is IMU yaw / commanded yaw, Y_op the operator's yaw / commanded (1.0 = tracked)."""
    out: dict[str, dict] = {}
    for r in chosen:
        if r["slot"] not in WALK_SLOTS or not r.get("walk"):
            continue
        for m in TRACK_METRICS:
            v = r["walk"].get(m)
            if v is not None:
                out.setdefault(r["arm"], {}).setdefault(r["slot"], {}).setdefault(m, []).append(
                    float(v)
                )
    return {
        arm: {slot: {m: _spread(v) for m, v in ms.items()} for slot, ms in slots.items()}
        for arm, slots in out.items()
    }


def stop_table(runs: list[dict]) -> dict:
    """arm -> STOP drill latencies (from the ticks) and window-end-to-sit times of every run."""
    out: dict[str, dict] = {}
    for r in runs:
        arm = r.get("measured_arm")
        if arm is None:
            continue
        d = out.setdefault(
            arm, {"stop_latency_s": [], "release_rx_to_zero_s": [], "window_end_to_sit_s": []}
        )
        ex = r.get("policy_exit") or {}
        if r.get("slot") == "stop_drill" and r.get("valid"):
            for k in ("stop_latency_s", "release_rx_to_zero_s"):
                if ex.get(k) is not None:
                    d[k].append(float(ex[k]))
        sit = (r.get("bridge") or {}).get("window_end_to_sit_s")
        if sit is not None:
            d["window_end_to_sit_s"].append(float(sit))
    return {a: {k: (_spread(v) if v else None) for k, v in d.items()} for a, d in out.items()}


def model_id(chosen: list[dict]) -> dict:
    out: dict[str, Any] = {}
    for arm, spec in ARMS.items():
        if "sim_label" not in spec:  # arms pinned by a campaign header carry no sim replay
            continue
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
    ap.add_argument("--campaign", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--classify-run",
        type=Path,
        metavar="RUN_DIR",
        help="print one run's policy-exit classification (Amendment 4); exit 0 only if it is "
        "the expected clean exit for --slot (used by the campaign runner after each run)",
    )
    ap.add_argument("--slot", help="with --classify-run: G, stop_drill or a walking preset")
    ap.add_argument(
        "--corrections",
        type=Path,
        help="JSON list of {harness_log, set: {field: value}, reason}: metadata corrections "
        "applied to the campaign's run records before analysis (kept in runs.json)",
    )
    ap.add_argument(
        "--remap",
        action="append",
        default=[],
        metavar="FROM=TO",
        help="rewrite a payload path prefix in run_dir / harness_log, e.g. "
        "logs/hw_sessions=logs/hw_sessions_orin for logs copied back from the payload",
    )
    args = ap.parse_args(argv)
    if args.classify_run is not None:
        if args.slot not in ("G", "stop_drill", *WALK_SLOTS):
            ap.error("--classify-run needs --slot G, stop_drill or a walking preset")
        try:
            _, ticks, _ = load_bridge(args.classify_run)
        except (OSError, ValueError) as exc:
            print(json.dumps({"clean": False, "error": str(exc)}))
            return 1
        ex = policy_exit(ticks, expected_authority_s(args.slot))
        clean = exit_is_clean(ex, args.slot)
        print(json.dumps({"slot": args.slot, "clean": clean} | ex, default=str))
        return 0 if clean else 1
    if args.campaign is None or args.out is None:
        ap.error("--campaign and --out are required (or use --classify-run)")
    recs = [json.loads(ln) for ln in args.campaign.read_text().splitlines() if ln.strip()]
    if args.corrections is not None:
        apply_corrections(recs, json.loads(args.corrections.read_text()))
    run_recs = [r for r in recs if r.get("kind") == "run"]
    header = next((r for r in recs if r.get("kind") == "header"), {})
    if header.get("arms"):  # the campaign pinned its own arms (hw_campaign.sh init)
        ARMS.clear()
        ARMS.update(
            {
                a: {k: v for k, v in spec.items() if k != "label"}
                for a, spec in header["arms"].items()
            }
        )
    for r in run_recs:
        r.setdefault("campaign_commit", header.get("commit"))
    base = args.campaign.parent
    remaps = [tuple(m.split("=", 1)) for m in args.remap]
    for r in recs:
        for key in ("run_dir", "harness_log"):
            for a, b in remaps:
                if isinstance(r.get(key), str) and r[key].startswith(a + "/"):
                    r[key] = b + r[key][len(a) :]
                    break
    for r in recs:
        log = r.get("harness_log")
        if isinstance(log, str) and not Path(log).is_absolute() and not Path(log).exists():
            if (base / Path(log).name).exists():
                r["harness_log"] = str(base / Path(log).name)
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
        "prereg": header.get("prereg")
        or "docs/preregistration/2026-10-07-gantry-ab-exploratory.md",
        "n_attempts": len(runs),
        "n_valid": sum(r["valid"] for r in runs),
        "invalid": [
            {k: r.get(k) for k in ("block", "arm", "slot", "attempt", "run_dir", "why")}
            for r in runs
            if not r["valid"]
        ],
        "decision": decide(chosen, runs, ladder),
        "tracking": tracking_table(chosen),
        "stop_summary": stop_table(runs),
        "ladder_safety_events": ladder,
        "stop_conditions_detected": stop_conditions(runs),
        "model_identification": model_id(chosen),
        "operator_log_discrepancies": [
            {k: r.get(k) for k in ("block", "arm", "slot", "run_dir", "operator_log_discrepancy")}
            for r in runs
            if r.get("operator_log_discrepancy")
        ],
        "stop": [
            {k: r.get(k) for k in ("block", "arm", "slot")}
            | {
                k: r["policy_exit"][k]
                for k in (
                    "exit",
                    "t_policy_last_ns",
                    "t_release_obs_ns",
                    "t_release_rx_ns",
                    "t_zero_ns",
                    "stop_latency_s",
                    "release_window_s",
                    "release_rx_to_zero_s",
                    "problems",
                )
            }
            | {"window_end_to_sit_s": r["bridge"]["window_end_to_sit_s"]}
            for r in runs
            if "policy_exit" in r
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
    lines += ["", "## Tracking (measured / commanded; mean +- std over valid repeats, n)", ""]
    lines += ["| arm | slot | metric | mean | std | n |", "|---|---|---|---|---|---|"]
    for arm, slots in sorted(summary["tracking"].items()):
        for slot, ms in slots.items():
            for m, sp in ms.items():
                lines.append(
                    f"| {arm} | {slot} | {m} | {sp['mean']:.3f} | {sp['std']:.3f} | {sp['n']} |"
                )
    lines += [
        "",
        "## Stop (seconds, from the ticks)",
        "",
        "| arm | measure | mean | max | n |",
        "|---|---|---|---|---|",
    ]
    for arm, d in sorted(summary["stop_summary"].items()):
        for k, sp in d.items():
            if sp:
                lines.append(f"| {arm} | {k} | {sp['mean']:.3f} | {sp['max']:.3f} | {sp['n']} |")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(summary["decision"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
