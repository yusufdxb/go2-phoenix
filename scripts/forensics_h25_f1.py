#!/usr/bin/env python3
"""Regenerate docs/forensics/h25_f1_metrics.json from the copied hardware evidence.

Reads the LowCmd bridge telemetry (``bridge.jsonl``) of the H25 stage-F attempt
F1 and of the other stages that ran the bridge, and recomputes every number in
``docs/forensics/h25_forensics_2026-09-22.md`` from the rows alone. The same
``phoenix.evaluation.metrics`` code that scores simulator episodes scores the
policy window here.

Definitions (also written into the JSON):

* policy tick: a bridge tick with ``mode == "policy"`` (the bridge forwarded a
  policy command to the motors, or in a dry run would have).
* raw request: ``policy.requested_target`` = default + 0.25 * raw action, before
  ANY clip, policy joint order.
* node-clipped target: ``policy.target``, after the policy node's slew clip.
* executed target: ``final_target_unitree`` (after the bridge's slew and limit
  clips), permuted to policy order.
* modified joint-tick: |executed - raw request| > 1e-5 rad.
* handoff: first policy tick; failure: first tick whose ``faults`` is non-empty
  (excluding teardown faults) after handoff.

Usage::

    PYTHONPATH=src python3 scripts/forensics_h25_f1.py \
        [--evidence logs/payload_evidence_20260922] \
        [--out docs/forensics/h25_f1_metrics.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from phoenix.evaluation.metrics import EpisodeTrace, compute_episode_metrics
from phoenix.evaluation.outcomes import evaluate_episode
from phoenix.sim2real.bridge_telemetry import read_telemetry, summarize
from phoenix.sim2real.go2_model import (
    JOINT_POSITION_LIMITS_RAD,
    LIMIT_ABORT_BAND_RAD,
    POLICY_JOINT_ORDER,
    TRAINING_DEFAULT_JOINT_POS,
    UNITREE_MOTOR_ORDER,
)
from phoenix.sim2real.safety import MAX_DELTA_PER_STEP_RAD

DEFAULT_EVIDENCE = Path("logs/payload_evidence_20260922")
SESSIONS = {
    "F1": "20260921_cc131039b662/F1_20260922T001048Z/bridge.jsonl",
    "E": "20260921_cc131039b662/E_20260922T000950Z/bridge.jsonl",
    "B_cc13103": "20260921_cc131039b662/B_20260921T235526Z/bridge.jsonl",
    "B_01355b2": "20260921_01355b22d9fe/B_20260921T235229Z/bridge.jsonl",
    "B_9df76d7_ramp": "20260922_9df76d7f3dcb/B_20260922T001841Z/bridge.jsonl",
}
TEARDOWN_FAULTS = {"bridge_shutdown", "lowstate_stale"}
EPS = 1e-5

# unitree index for each policy-order joint
U2P = np.asarray([UNITREE_MOTOR_ORDER.index(n) for n in POLICY_JOINT_ORDER])
LOWER = np.asarray([JOINT_POSITION_LIMITS_RAD[n][0] for n in POLICY_JOINT_ORDER])
UPPER = np.asarray([JOINT_POSITION_LIMITS_RAD[n][1] for n in POLICY_JOINT_ORDER])
DEFAULT_Q = np.asarray([TRAINING_DEFAULT_JOINT_POS[n] for n in POLICY_JOINT_ORDER])


def _vec(v) -> np.ndarray | None:
    if v is None or any(x is None for x in v):
        return None
    return np.asarray(v, dtype=np.float64)


def _stats(a: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    return {
        "mean_abs": float(np.mean(np.abs(a))),
        "rms": float(np.sqrt(np.mean(a**2))),
        "max_abs": float(np.max(np.abs(a))),
    }


def _per_joint(mask: np.ndarray) -> dict[str, float]:
    return {n: float(mask[:, j].mean()) for j, n in enumerate(POLICY_JOINT_ORDER)}


def analyse(path: Path, display_path: str) -> dict:
    manifest, ticks, end = read_telemetry(path)
    legacy_summary = summarize(manifest, ticks)
    out: dict = {
        "path": display_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "stage": manifest.get("stage"),
        "live": manifest.get("live"),
        "code_sha": (manifest.get("code_identity") or {}).get("sha"),
        "n_ticks": len(ticks),
        "mode_counts": legacy_summary["mode_counts"],
        "end_record": end,
        "existing_summary": {
            k: legacy_summary[k]
            for k in (
                "end_to_end_clip_pct",
                "bridge_slew_clip_pct",
                "policy_node_slew_clip_pct",
                "limit_clip_counts",
                "first_fault",
                "policy_window_s",
                "policy_ticks",
                "policy_new_command_ticks",
            )
        },
    }
    # limit clips split by mode: during hold the bridge clamps the HOLD target
    # (the measured folded pose is itself past the calf limit), which is not a
    # policy command.
    by_mode: dict[str, dict[str, int]] = {}
    for t in ticks:
        for j, hit in enumerate(t.get("limit_clip") or []):
            if hit:
                d = by_mode.setdefault(t.get("mode"), {})
                d[UNITREE_MOTOR_ORDER[j]] = d.get(UNITREE_MOTOR_ORDER[j], 0) + 1
    out["limit_clip_counts_by_mode"] = by_mode

    pol = [t for t in ticks if t.get("mode") == "policy"]
    out["policy_ticks"] = len(pol)
    if not pol:
        q0 = next((t["q_unitree"] for t in ticks if t.get("q_unitree")), None)
        if q0 is not None:
            qp = np.asarray(q0)[U2P]
            out["first_measured_q_minus_training_stance_rad"] = dict(
                zip(POLICY_JOINT_ORDER, (qp - DEFAULT_Q).round(4).tolist(), strict=True)
            )
        return out

    raw = np.stack([_vec(t["policy"]["raw_action"]) for t in pol])
    req = np.stack([_vec(t["policy"]["requested_target"]) for t in pol])
    node = np.stack([_vec(t["policy"]["target"]) for t in pol])
    bridge_in = np.stack([_vec(t["requested_target_unitree"])[U2P] for t in pol])
    exe = np.stack([_vec(t["final_target_unitree"])[U2P] for t in pol])
    q = np.stack([_vec(t["q_unitree"])[U2P] for t in pol])
    roll = np.asarray([t["policy"]["roll_rad"] for t in pol], dtype=np.float64)
    pitch = np.asarray([t["policy"]["pitch_rad"] for t in pol], dtype=np.float64)
    cmd = np.stack([_vec(t["policy"]["velocity_command_fed"]) for t in pol])
    blv = np.stack([_vec(t["policy"]["base_lin_vel_fed"]) for t in pol])
    t_ns = np.asarray([int(t["t_mono_ns"]) for t in pol], dtype=np.int64)

    if not np.allclose(node, bridge_in, atol=1e-6):
        raise SystemExit(f"{path}: policy.target != requested_target_unitree; wiring changed?")

    mod_total = np.abs(exe - req) > EPS
    mod_node = np.abs(node - req) > EPS
    mod_bridge = np.abs(exe - bridge_in) > EPS
    limit_mask = np.stack([np.asarray(t["limit_clip"], dtype=bool)[U2P] for t in pol])
    slew_mask = np.stack([np.asarray(t["slew_clip"], dtype=bool)[U2P] for t in pol])

    # first failure after handoff
    handoff_ns = int(t_ns[0])
    fail_tick = None
    for t in ticks:
        if int(t["t_mono_ns"]) < handoff_ns:
            continue
        real = [f for f in (t.get("faults") or []) if f not in TEARDOWN_FAULTS]
        if real:
            fail_tick = t
            break

    def beyond(targets: np.ndarray) -> np.ndarray:
        return np.maximum(LOWER - targets, targets - UPPER)

    def illegal_list(targets: np.ndarray, label: str) -> list[dict]:
        b = beyond(targets)
        rows = []
        for i, j in zip(*np.nonzero(b > LIMIT_ABORT_BAND_RAD), strict=True):
            lim = LOWER[j] if targets[i, j] < LOWER[j] else UPPER[j]
            rows.append(
                {
                    "source": label,
                    "policy_tick_index": int(i),
                    "bridge_tick": int(pol[i]["tick"]),
                    "t_since_handoff_s": (int(t_ns[i]) - handoff_ns) / 1e9,
                    "joint": POLICY_JOINT_ORDER[j],
                    "target_rad": float(targets[i, j]),
                    "hard_limit_rad": float(lim),
                    "abort_threshold_rad": float(
                        lim - LIMIT_ABORT_BAND_RAD
                        if lim == LOWER[j]
                        else lim + LIMIT_ABORT_BAND_RAD
                    ),
                    "beyond_limit_by_rad": float(b[i, j]),
                }
            )
        return rows

    # the tick whose request triggered the bridge abort (not a policy-mode tick)
    abort_request = None
    if fail_tick is not None and fail_tick.get("requested_target_unitree"):
        rt = np.asarray(fail_tick["requested_target_unitree"], dtype=np.float64)
        pt = rt[U2P]
        b = beyond(pt)
        j = int(np.argmax(b))
        lim = LOWER[j] if pt[j] < LOWER[j] else UPPER[j]
        abort_request = {
            "bridge_tick": int(fail_tick["tick"]),
            "joint": POLICY_JOINT_ORDER[j],
            "bridge_received_target_rad": float(pt[j]),
            "raw_policy_request_rad": float(_vec(fail_tick["policy"]["requested_target"])[j]),
            "hard_limit_rad": float(lim),
            "abort_band_rad": LIMIT_ABORT_BAND_RAD,
            "beyond_limit_by_rad": float(b[j]),
        }

    trace = EpisodeTrace(
        dt_s=0.02,
        command=cmd,
        lin_vel_b=None,
        yaw_rate=None,
        tilt_rad=np.arccos(np.clip(np.cos(roll) * np.cos(pitch), -1, 1)),
        roll_rad=roll,
        pitch_rad=pitch,
        raw_action=raw,
        requested_target=req,
        executed_target=exe,
        measured_q=q,
        joint_lower=LOWER,
        joint_upper=UPPER,
        planned_steps=None,
        joint_names=POLICY_JOINT_ORDER,
    )
    metrics = compute_episode_metrics(trace)
    safety_events = []
    if fail_tick is not None:
        safety_events = [f for f in fail_tick.get("faults") or [] if f not in TEARDOWN_FAULTS]
    live = bool(manifest.get("live"))
    evaluation = evaluate_episode(
        metrics,
        harness_completed=fail_tick is None,
        safety_events=safety_events,
        commanded_motion=False,
    )
    q0 = q[0]
    max_abs_raw_per_tick = np.max(np.abs(raw), axis=1)
    out.update(
        {
            "handoff_utc": pol[0]["utc"],
            "handoff_bridge_tick": int(pol[0]["tick"]),
            "policy_window_first_to_last_s": (int(t_ns[-1]) - handoff_ns) / 1e9,
            "handoff_to_failure_s": (
                (int(fail_tick["t_mono_ns"]) - handoff_ns) / 1e9 if fail_tick else None
            ),
            "failure_bridge_tick": int(fail_tick["tick"]) if fail_tick else None,
            "failure_faults": fail_tick.get("faults") if fail_tick else None,
            "abort_request": abort_request,
            "modification": {
                "joint_tick_rate_total": float(mod_total.mean()),
                "tick_rate_total_any_joint": float(mod_total.any(axis=1).mean()),
                "joint_tick_rate_policy_node": float(mod_node.mean()),
                "joint_tick_rate_bridge": float(mod_bridge.mean()),
                "per_joint_total": _per_joint(mod_total),
                "magnitude_exe_minus_raw_rad": _stats(exe - req),
                "n_joint_ticks": int(mod_total.size),
            },
            "bridge_flags": {
                "slew_clip_joint_tick_rate": float(slew_mask.mean()),
                "limit_clip_joint_tick_rate": float(limit_mask.mean()),
                "limit_clip_per_joint_policy_ticks": {
                    n: int(limit_mask[:, j].sum())
                    for j, n in enumerate(POLICY_JOINT_ORDER)
                    if limit_mask[:, j].any()
                },
            },
            "illegal_targets": {
                "raw_request_beyond_hard_limit_joint_ticks": int((beyond(req) > 0).sum()),
                "raw_request_beyond_abort_band_joint_ticks": int(
                    (beyond(req) > LIMIT_ABORT_BAND_RAD).sum()
                ),
                "bridge_received_beyond_abort_band_joint_ticks": int(
                    (beyond(bridge_in) > LIMIT_ABORT_BAND_RAD).sum()
                ),
                "first_raw_illegal": (illegal_list(req, "raw_request") or [None])[0],
                "raw_illegal_by_joint": {
                    n: int((beyond(req)[:, j] > LIMIT_ABORT_BAND_RAD).sum())
                    for j, n in enumerate(POLICY_JOINT_ORDER)
                    if (beyond(req)[:, j] > LIMIT_ABORT_BAND_RAD).any()
                },
            },
            "raw_action": {
                "max_abs": float(np.max(np.abs(raw))),
                "fraction_elements_abs_gt_1": float(np.mean(np.abs(raw) > 1.0)),
                "fraction_ticks_any_abs_gt_1": float(np.mean(max_abs_raw_per_tick > 1.0)),
                "max_abs_per_tick_first5": max_abs_raw_per_tick[:5].round(3).tolist(),
                "max_abs_per_tick_last5": max_abs_raw_per_tick[-5:].round(3).tolist(),
                "note": (
                    "training clamps actions to [-1, 1] (RslRlVecEnvWrapper clip_actions=1.0) "
                    "and feeds the CLAMPED action back as last_action; the deploy node clamps "
                    "neither"
                ),
            },
            "commanded_vs_executed": {
                "exe_target_t_minus_measured_q_t_plus_1_rad": (
                    _stats(exe[:-1] - q[1:]) if len(q) > 1 else None
                ),
                "raw_request_minus_measured_q_rad": _stats(req - q),
            },
            "stance_at_handoff": {
                "measured_q_minus_training_default_rad": dict(
                    zip(POLICY_JOINT_ORDER, (q0 - DEFAULT_Q).round(4).tolist(), strict=True)
                ),
                "max_abs_rad": float(np.max(np.abs(q0 - DEFAULT_Q))),
                "measured_q_beyond_hard_limit_joints": [
                    POLICY_JOINT_ORDER[j]
                    for j in range(12)
                    if q0[j] < LOWER[j] - 1e-9 or q0[j] > UPPER[j] + 1e-9
                ],
            },
            "observation_fed": {
                "base_lin_vel_fed_max_abs": float(np.max(np.abs(blv))),
                "velocity_command_fed_max_abs": float(np.max(np.abs(cmd))),
            },
            "attitude": {
                "max_abs_roll_rad": float(np.max(np.abs(roll))),
                "max_abs_pitch_rad": float(np.max(np.abs(pitch))),
            },
            "phoenix_evaluation": {
                "outcome": evaluation.outcome.value,
                "verdict": evaluation.verdict,
                "reasons": evaluation.reasons,
                "locomotion_success": evaluation.locomotion_success,
                "live": live,
            },
        }
    )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    p.add_argument("--out", type=Path, default=Path("docs/forensics/h25_f1_metrics.json"))
    args = p.parse_args(argv)
    result = {
        "schema": "phoenix-h25-forensics/v1",
        "generator": "scripts/forensics_h25_f1.py",
        "definitions": {
            "policy_tick": "bridge tick with mode == 'policy'",
            "raw_request": "policy.requested_target = default + 0.25 * raw_action, policy order",
            "executed": "final_target_unitree permuted to policy order (post node + bridge clip)",
            "modified_joint_tick": f"|executed - raw_request| > {EPS} rad",
            "handoff": "first policy tick",
            "failure": f"first tick at/after handoff with a fault other than {sorted(TEARDOWN_FAULTS)}",
            "illegal_target": (
                f"target beyond a hard URDF limit by more than {LIMIT_ABORT_BAND_RAD} rad "
                "(go2_model.LIMIT_ABORT_BAND_RAD, the bridge abort band)"
            ),
            "slew_cap_rad_per_step": MAX_DELTA_PER_STEP_RAD,
        },
        "sessions": {},
    }
    for name, rel in SESSIONS.items():
        path = args.evidence / rel
        if not path.exists():
            result["sessions"][name] = {"path": rel, "missing": True}
            continue
        result["sessions"][name] = analyse(path, rel)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def _clean(o):
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        return o

    args.out.write_text(json.dumps(_clean(result), indent=2, sort_keys=True) + "\n")
    f1 = result["sessions"].get("F1", {})
    m = f1.get("modification", {})
    print(
        f"F1: modified joint-ticks {m.get('joint_tick_rate_total')!r}, policy window "
        f"{f1.get('policy_window_first_to_last_s')!r} s, handoff->failure "
        f"{f1.get('handoff_to_failure_s')!r} s -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
