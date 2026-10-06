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

Fail-closed contract. The check refuses (raises :class:`Refusal`, never returns GO)
unless every input is provably comparable: the references and the telemetry carry the
same policy-artifact and deploy-config hashes; the boundary references are exactly the
calf=0.80 and all=0.75 plants; the hardware gains equal the reference gains; and each
telemetry file holds one still, healthy, live policy window of at least 2 s.

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
#: Hardware window starts this long after the END of the kp handover blend.
SETTLE_AFTER_BLEND_S = 1.0
#: Minimum settled-window length per telemetry file.
MIN_WINDOW_S = 2.0
#: Largest gap between consecutive window ticks (a telemetry dropout otherwise passes).
MAX_TICK_GAP_S = 0.1
TILT_MAX_DEG = 10.0
#: Per-joint dq RMS over the window must stay below this (robot standing still).
DQ_RMS_MAX_RAD_S = 0.05
#: Hardware gains must equal the reference gains within this.
GAIN_TOL = 1e-3
#: Commanded-velocity tick fields, looked up in ``tick`` and in ``tick["policy"]``.
VELOCITY_FIELDS = ("velocity_command_fed", "cmd_vel_received", "velocity_command", "cmd_vel")
VELOCITY_ZERO_TOL = 1e-6
G_TOL = 1e-9
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
#: Lock roles hashed into every reference and compared against bridge telemetry.
ARTIFACT_ROLES = ("policy.onnx", "policy.onnx.data", "policy.manifest")
IDENTITY_KEYS = (*ARTIFACT_ROLES, "config_semantic_sha256")


class Refusal(ValueError):  # noqa: N818
    """An input is missing, mismatched or of too low quality to compare: no verdict."""


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


def _gain_vec(value: Any) -> np.ndarray:
    """A scalar or 12-list gain -> a 12-vector (Unitree order for a list)."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(len(UNITREE_MOTOR_ORDER), float(arr))
    if arr.shape != (len(UNITREE_MOTOR_ORDER),):
        raise ValueError(f"gain has shape {arr.shape}, expected a scalar or 12 values")
    return arr


def load_telemetry(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Bridge telemetry JSONL files -> ``{"path", "manifest", "n_manifests", "ticks"}`` each.

    ``manifest`` is the file's FIRST record when it is a ``manifest`` record, else
    ``None`` (a file without one is refused by :func:`check_telemetry_manifest`).
    """
    out: list[dict[str, Any]] = []
    for p in paths:
        ticks: list[dict[str, Any]] = []
        manifest: dict[str, Any] | None = None
        n_manifests = 0
        first = True
        with Path(p).open() as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError as exc:
                    raise Refusal(f"{p}:{lineno}: unparsable telemetry line ({exc})") from exc
                kind = rec.get("record")
                if kind == "manifest":
                    n_manifests += 1
                    if first:
                        manifest = rec
                elif kind == "tick":
                    ticks.append(rec)
                first = False
        out.append(
            {"path": str(p), "manifest": manifest, "n_manifests": n_manifests, "ticks": ticks}
        )
    return out


def load_ticks(paths: Iterable[str | Path]) -> list[list[dict[str, Any]]]:
    """Bridge telemetry JSONL files -> one list of ``record == "tick"`` dicts per file."""
    return [f["ticks"] for f in load_telemetry(paths)]


# ---------------------------------------------------------------------------
# Artifact identity
# ---------------------------------------------------------------------------


def _is_sha256(v: Any) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def ref_identity(ref: Mapping[str, Any], label: str = "reference") -> dict[str, str]:
    """The identity block a reference must carry (see ``IDENTITY_KEYS``)."""
    ident = ref.get("identity")
    if not isinstance(ident, Mapping) or any(not _is_sha256(ident.get(k)) for k in IDENTITY_KEYS):
        raise Refusal(
            f"{label} has no complete artifact identity ({', '.join(IDENTITY_KEYS)}): "
            "regenerate it with sim-reference"
        )
    return {k: str(ident[k]) for k in IDENTITY_KEYS}


def manifest_identity(manifest: Mapping[str, Any], label: str = "telemetry") -> dict[str, str]:
    """Artifact and semantic-config hashes the bridge recorded in its manifest record."""
    try:
        arts = manifest["artifacts"]
        out = {role: arts[role]["sha256"] for role in ARTIFACT_ROLES}
        out["config_semantic_sha256"] = manifest["deploy_config"]["semantic_sha256"]
    except (KeyError, TypeError) as exc:
        raise Refusal(f"{label}: manifest lacks artifact/config hashes ({exc!r})") from exc
    if any(not _is_sha256(v) for v in out.values()):
        raise Refusal(f"{label}: manifest hashes are not sha256 hex digests")
    return {k: str(v) for k, v in out.items()}


def require_same_identity(
    label_a: str, a: Mapping[str, str], label_b: str, b: Mapping[str, str]
) -> None:
    for k in IDENTITY_KEYS:
        if a[k] != b[k]:
            raise Refusal(
                f"identity mismatch on {k}: {label_a} {a[k][:12]} != {label_b} {b[k][:12]}"
            )


def check_telemetry_manifest(
    file: Mapping[str, Any], ref_ident: Mapping[str, str], *, allow_rehearsal: bool = False
) -> dict[str, Any]:
    """Refuse unless the file's manifest names the reference's artifacts and is live."""
    label = str(file.get("path", "telemetry"))
    manifest = file.get("manifest")
    if manifest is None:
        raise Refusal(f"{label}: first record is not a manifest record")
    if int(file.get("n_manifests", 1)) != 1:
        raise Refusal(f"{label}: {file['n_manifests']} manifest records (concatenated sessions?)")
    require_same_identity(label, manifest_identity(manifest, label), "reference", ref_ident)
    live = manifest.get("live") is True
    if not live and not allow_rehearsal:
        raise Refusal(
            f"{label}: telemetry is not live (dry run, motors not commanded); "
            "--allow-rehearsal is tests-only"
        )
    code = manifest.get("code_identity") or {}
    return {
        "path": label,
        "stage": manifest.get("stage"),
        "live": live,
        "commit": code.get("sha"),
        "dirty": code.get("dirty"),
        "lock": (manifest.get("lock") or {}).get("name"),
    }


LEDGER_STAGES: tuple[str, ...] = ("G", "H")


def ledger_telemetry(
    session_dir: str | Path, ref_ident: Mapping[str, str], stages: Sequence[str] = LEDGER_STAGES
) -> tuple[list[str], list[dict[str, Any]]]:
    """Telemetry files that a counted hardware ledger record vouches for.

    The bridge manifest's ``live`` only says the motors were commanded; a localhost
    rehearsal is also ``live``. What separates hardware from rehearsal is the harness
    stage record: ``verdict`` GO, ``rehearsal`` exactly False, ``motors`` "live", the
    stand lock's config hash, and every evidence file still hashing to the record.
    Only files named by such records are returned.
    """
    from .deploy_contract import _evidence_file_rejections

    session = Path(session_dir)
    files: list[str] = []
    records: list[dict[str, Any]] = []
    for stage in stages:
        path = session / f"stage_{stage}.json"
        if not path.is_file():
            raise Refusal(f"{path}: no stage {stage} ledger record in the session")
        rec = json.loads(path.read_text())
        problems = []
        if rec.get("stage") != stage:
            problems.append(f"stage is {rec.get('stage')!r}")
        if rec.get("verdict") != "GO":
            problems.append(f"verdict is {rec.get('verdict')!r}, not GO")
        if rec.get("rehearsal") is not False:
            problems.append(f"rehearsal is {rec.get('rehearsal')!r}, not False")
        if rec.get("motors") != "live":
            problems.append(f"motors is {rec.get('motors')!r}, not live")
        if rec.get("lock_semantic_sha256") != ref_ident.get("config_semantic_sha256"):
            problems.append("record is for a different deploy config than the references")
        problems += _evidence_file_rejections(rec, session)
        if problems:
            raise Refusal(f"{path}: " + "; ".join(problems))
        named = [str(session / e["path"]) for e in rec["evidence_files"]]
        named = [f for f in named if f.endswith("bridge.jsonl")]
        if not named:
            raise Refusal(f"{path}: names no bridge telemetry")
        files += named
        records.append({"stage": stage, "record": str(path), "files": named})
    return files, records


# ---------------------------------------------------------------------------
# Hardware window
# ---------------------------------------------------------------------------


def _velocity_values(tick: Mapping[str, Any]) -> list[tuple[str, Any]]:
    found = []
    for src in (tick, tick.get("policy") or {}):
        if isinstance(src, Mapping):
            for name in VELOCITY_FIELDS:
                if src.get(name) is not None:
                    found.append((name, src[name]))
    return found


def settle_window(
    ticks: Sequence[Mapping[str, Any]],
    *,
    label: str = "telemetry",
    settle_s: float = SETTLE_AFTER_BLEND_S,
    min_window_s: float = MIN_WINDOW_S,
    strict_fields: bool = False,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """The settled policy window of one file and its quality report, or :class:`Refusal`.

    Gates: exactly one contiguous policy episode; window starts ``settle_s`` after the
    first policy tick whose gains equal the final gains (and ``handover_alpha`` is 1 when
    present); window >= ``min_window_s`` with no tick gap above ``MAX_TICK_GAP_S``; gains
    constant over it; every tick lowstate-fresh, fault-free, |tilt| < ``TILT_MAX_DEG``, with
    zero commanded velocity where that field exists; per-joint dq RMS < ``DQ_RMS_MAX_RAD_S``.
    """
    idx = [i for i, t in enumerate(ticks) if t.get("mode") == "policy"]
    if not idx:
        raise Refusal(f"{label}: no policy-mode ticks")
    if idx[-1] - idx[0] + 1 != len(idx):
        episodes = 1 + sum(1 for a, b in zip(idx, idx[1:], strict=False) if b != a + 1)
        raise Refusal(f"{label}: {episodes} separate policy episodes, exactly one required")
    pol = [ticks[i] for i in idx]
    try:
        ts = [int(t["t_mono_ns"]) for t in pol]
        kp = [_gain_vec(t["kp_eff"]) for t in pol]
        kd = [_gain_vec(t["kd_eff"]) for t in pol]
    except (KeyError, TypeError, ValueError) as exc:
        raise Refusal(
            f"{label}: policy tick lacks a usable t_mono_ns/kp_eff/kd_eff ({exc!r})"
        ) from exc
    if any(b <= a for a, b in zip(ts, ts[1:], strict=False)):
        raise Refusal(f"{label}: policy tick times are not strictly increasing")

    def at_final(i: int) -> bool:
        alpha = pol[i].get("handover_alpha")
        return (
            bool(np.allclose(kp[i], kp[-1], rtol=0.0, atol=1e-6))
            and bool(np.allclose(kd[i], kd[-1], rtol=0.0, atol=1e-6))
            and (float(alpha) >= 1.0 if alpha is not None else not strict_fields)
        )

    first = next((i for i in range(len(pol)) if at_final(i)), None)
    if first is None:
        raise Refusal(f"{label}: gains never reach their final value (handover incomplete)")
    t0 = ts[first] + int(round(settle_s * 1e9))
    win = [i for i in range(first, len(pol)) if ts[i] >= t0]
    if not win:
        raise Refusal(f"{label}: no policy ticks {settle_s} s after the end of the kp blend")
    span_s = (ts[win[-1]] - ts[win[0]]) / 1e9
    if span_s < min_window_s:
        raise Refusal(
            f"{label}: settled window is {span_s:.2f} s over {len(win)} tick(s), need >= "
            f"{min_window_s} s ({settle_s} s after the kp blend)"
        )
    max_gap_s = max((b - a) / 1e9 for a, b in zip(ts[win[0] :], ts[win[0] + 1 :], strict=False))
    if not max_gap_s <= MAX_TICK_GAP_S:
        raise Refusal(f"{label}: tick gap {max_gap_s:.3f} s > {MAX_TICK_GAP_S} s in the window")

    tilt_max = 0.0
    n_vel = 0
    for i in win:
        t, tk = pol[i], ts[i]
        where = f"{label}: tick t_mono_ns={tk}"
        if t.get("lowstate_fresh") is not True:
            raise Refusal(f"{where} has stale lowstate")
        if t.get("fault") is not None or t.get("faults"):
            raise Refusal(f"{where} carries a fault ({t.get('fault') or t.get('faults')})")
        if not np.allclose(kp[i], kp[-1], rtol=0.0, atol=1e-6) or not np.allclose(
            kd[i], kd[-1], rtol=0.0, atol=1e-6
        ):
            raise Refusal(f"{where}: gains change inside the settled window")
        if strict_fields and "tilt_deg" not in t:
            raise Refusal(f"{where}: hardware tick has no tilt_deg")
        if strict_fields and "velocity_command_fed" not in dict(_velocity_values(t)):
            raise Refusal(f"{where}: hardware tick has no velocity_command_fed")
        if "tilt_deg" in t:
            tilt = t["tilt_deg"]
            if not isinstance(tilt, (int, float)) or not abs(tilt) < TILT_MAX_DEG:
                raise Refusal(f"{where}: tilt_deg {tilt!r} not within +-{TILT_MAX_DEG} deg")
            tilt_max = max(tilt_max, abs(float(tilt)))
        for name, vec in _velocity_values(t):
            n_vel += 1
            # A stand-only run never receives /cmd_vel: the bridge logs [None, None, None].
            # What the policy was fed is velocity_command_fed, which must be zero.
            if name == "cmd_vel_received" and all(x is None for x in np.ravel(vec)):
                continue
            try:
                ok = all(abs(float(x)) <= VELOCITY_ZERO_TOL for x in np.ravel(vec))
            except (TypeError, ValueError):
                ok = False
            if not ok:
                raise Refusal(f"{where}: commanded velocity {name}={vec!r} is not zero")
    try:
        dq = np.asarray([pol[i]["dq_unitree"] for i in win], dtype=float)
        qq = np.asarray([pol[i]["q_unitree"] for i in win], dtype=float)
        tg = np.asarray([pol[i]["final_target_unitree"] for i in win], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise Refusal(f"{label}: window tick lacks q/dq/target ({exc!r})") from exc
    if dq.shape[1:] != (12,) or qq.shape[1:] != (12,) or tg.shape[1:] != (12,):
        raise Refusal(f"{label}: q/dq/target are not 12-vectors")
    if not (np.isfinite(dq).all() and np.isfinite(qq).all() and np.isfinite(tg).all()):
        raise Refusal(f"{label}: non-finite q/dq/target in the window")
    rms = np.sqrt(np.mean(np.square(dq), axis=0))
    worst = int(np.argmax(rms))
    if not float(rms[worst]) < DQ_RMS_MAX_RAD_S:
        raise Refusal(
            f"{label}: robot is moving, dq RMS {float(rms[worst]):.3f} rad/s on "
            f"{UNITREE_MOTOR_ORDER[worst]} >= {DQ_RMS_MAX_RAD_S}"
        )
    report = {
        "label": label,
        "passed": True,
        "n_policy_ticks": len(pol),
        "n_window": len(win),
        "window_s": span_s,
        "blend_end_to_window_s": settle_s,
        "max_tick_gap_s": max_gap_s,
        "tilt_max_deg": tilt_max,
        "dq_rms_max_rad_s": float(rms[worst]),
        "velocity_fields_checked": n_vel,
        "episodes": 1,
    }
    return [pol[i] for i in win], report


def select_settled_window(
    ticks: Sequence[Mapping[str, Any]], settle_s: float = SETTLE_AFTER_BLEND_S
) -> list[Mapping[str, Any]]:
    """Settled policy window of one file (:func:`settle_window` without the report)."""
    return settle_window(ticks, settle_s=settle_s)[0]


def summarize_hw(
    tick_sets: Sequence[Sequence[Mapping[str, Any]]],
    settle_s: float = SETTLE_AFTER_BLEND_S,
    labels: Sequence[str] | None = None,
    strict_fields: bool = False,
) -> dict[str, Any]:
    """Pool the settled windows of several bridge files into one per-joint summary
    (same shape as a simulator reference, minus ``g_true``). Every file must pass
    :func:`settle_window`; ``window_gates`` records each file's report."""
    if not tick_sets:
        raise Refusal("no telemetry files supplied")
    err: dict[str, list[float]] = {n: [] for n in POLICY_JOINT_ORDER}
    q: dict[str, list[float]] = {n: [] for n in POLICY_JOINT_ORDER}
    tgt: dict[str, list[float]] = {n: [] for n in POLICY_JOINT_ORDER}
    dq: dict[str, list[float]] = {n: [] for n in POLICY_JOINT_ORDER}
    kp: dict[str, list[float]] = {n: [] for n in POLICY_JOINT_ORDER}
    kd: dict[str, list[float]] = {n: [] for n in POLICY_JOINT_ORDER}
    n_ticks = 0
    gates = []
    for k, ticks in enumerate(tick_sets):
        window, report = settle_window(
            ticks,
            label=labels[k] if labels else f"file[{k}]",
            settle_s=settle_s,
            strict_fields=strict_fields,
        )
        gates.append(report)
        for t in window:
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
    med = {"err": err, "q": q, "target": tgt, "kp": kp, "kd": kd}
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "source": "hardware",
        "n": n_ticks,
        "skip_s": settle_s,
        "window_gates": gates,
    }
    for key, d in med.items():
        out[key] = {n: float(np.median(v)) for n, v in d.items()}
    out["dq_rms"] = {n: float(np.sqrt(np.mean(np.square(v)))) for n, v in dq.items()}
    out["joint_order"] = list(POLICY_JOINT_ORDER)
    return out


def summarize_telemetry(
    files: Sequence[Mapping[str, Any]],
    ref_ident: Mapping[str, str],
    *,
    allow_rehearsal: bool = False,
) -> dict[str, Any]:
    """Manifest identity and liveness checks, then :func:`summarize_hw` per-file gates."""
    if not files:
        raise Refusal("no telemetry files supplied")
    infos = [check_telemetry_manifest(f, ref_ident, allow_rehearsal=allow_rehearsal) for f in files]
    hw = summarize_hw(
        [f["ticks"] for f in files],
        labels=[str(f.get("path", "telemetry")) for f in files],
        strict_fields=not allow_rehearsal,  # real hardware must carry tilt and velocity fields
    )
    hw["identity"] = dict(ref_ident)
    hw["telemetry"] = infos
    hw["live"] = all(i["live"] for i in infos)
    hw["allow_rehearsal"] = bool(allow_rehearsal)
    return hw


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


def _g_true(ref: Mapping[str, Any], label: str) -> dict[str, float]:
    gt = ref.get("g_true")
    try:
        out = {n: float(gt[n]) for n in POLICY_JOINT_ORDER}  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as exc:
        raise Refusal(f"{label} has no complete g_true (per-joint plant factors)") from exc
    return out


def describe_factors(ref: Mapping[str, Any]) -> str:
    """Plant label from the reference's recorded ``g_true``: ``all=0.75``, ``calf=0.80``."""
    gt = _g_true(ref, "reference")
    vals = set(gt.values())
    if len(vals) == 1:
        return f"all={vals.pop():.2f}"
    parts = []
    for g in ("hip", "thigh", "calf"):
        gv = {v for n, v in gt.items() if joint_group(n) == g}
        parts.append(f"{g}={gv.pop():.2f}" if len(gv) == 1 else f"{g}=mixed")
    return ",".join(p for p in parts if not p.endswith("=1.00"))


def _require_factors(label: str, ref: Mapping[str, Any], want: Mapping[str, float]) -> None:
    """``want`` maps group -> factor; the reference's g_true must equal it per joint."""
    gt = _g_true(ref, label)
    for n in POLICY_JOINT_ORDER:
        w = want[joint_group(n)]
        if not abs(gt[n] - w) <= G_TOL:
            raise Refusal(
                f"{label} is not the required plant ({want}): {n} g_true {gt[n]:g} != {w:g}"
                f" (recorded plant: {describe_factors(ref)})"
            )


def _require_gains_equal(
    label_a: str, a: Mapping[str, Any], label_b: str, b: Mapping[str, Any]
) -> None:
    for key in ("kp", "kd"):
        for n in POLICY_JOINT_ORDER:
            va, vb = float(a[key][n]), float(b[key][n])
            if not abs(va - vb) <= GAIN_TOL:
                raise Refusal(f"{key} mismatch on {n}: {label_a} {va:g} != {label_b} {vb:g}")


def check_ref_consistent(label: str, ref: Mapping[str, Any], nominal: Mapping[str, Any]) -> None:
    """A simulator reference must share the nominal's schema, artifacts and gains, not have fallen."""
    if ref.get("schema") != SCHEMA or ref.get("source") != "sim":
        raise Refusal(f"{label}: not a {SCHEMA} simulator reference")
    if ref.get("fell") is not False:
        raise Refusal(f"{label}: the simulated robot fell (or fell is not recorded)")
    require_same_identity(label, ref_identity(ref, label), "nominal", ref_identity(nominal))
    _require_gains_equal(label, ref, "nominal", nominal)
    _g_true(ref, label)


def validate_boundaries(
    nominal: Mapping[str, Any],
    ref_calf_boundary: Mapping[str, Any],
    ref_all_boundary: Mapping[str, Any],
) -> dict[str, str]:
    """Refuse unless nominal is all x1.0, the calf boundary is calf x0.80 (others x1.0)
    and the global boundary is all x0.75; return the plant labels read from the files."""
    check_ref_consistent("nominal", nominal, nominal)
    _require_factors("nominal", nominal, {"hip": 1.0, "thigh": 1.0, "calf": 1.0})
    check_ref_consistent("boundary-calf", ref_calf_boundary, nominal)
    _require_factors(
        "boundary-calf",
        ref_calf_boundary,
        {"hip": 1.0, "thigh": 1.0, "calf": BOUNDARY_CALF_G},
    )
    check_ref_consistent("boundary-all", ref_all_boundary, nominal)
    _require_factors(
        "boundary-all", ref_all_boundary, dict.fromkeys(RULE_GROUPS + ("hip",), BOUNDARY_ALL_G)
    )
    return {
        "nominal": describe_factors(nominal),
        "calf_boundary": describe_factors(ref_calf_boundary),
        "global_boundary": describe_factors(ref_all_boundary),
    }


def check_hw_comparable(hw: Mapping[str, Any], nominal: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse unless ``hw`` is the same policy, the same gains, and a clean live window."""
    require_same_identity("hardware", _hw_identity(hw), "nominal", ref_identity(nominal))
    _require_gains_equal("hardware", hw, "nominal", nominal)
    source = hw.get("source")
    if source == "hardware":
        gates = hw.get("window_gates")
        if not gates or any(not g.get("passed") for g in gates):
            raise Refusal("hardware summary has no passing window-quality record")
        if hw.get("allow_rehearsal") is not True:
            if hw.get("live") is not True:
                raise Refusal("hardware summary is not from live telemetry")
            if not hw.get("ledger"):
                raise Refusal("hardware summary is not bound to G/H ledger records (use --session)")
        return {"window_gates": list(gates), "live": hw.get("live")}
    if source == "sim":
        _g_true(hw, "sim-as-hw")
        return {"window_gates": [], "live": None}
    raise Refusal(f"unknown summary source {source!r}")


def _hw_identity(hw: Mapping[str, Any]) -> dict[str, str]:
    return ref_identity(hw, "hardware summary")


def decide(
    hw: Mapping[str, Any],
    ref_calf_boundary: Mapping[str, Any],
    ref_all_boundary: Mapping[str, Any],
    ref_nominal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """GO iff ``calf_err_hw`` is below BOTH boundary calf errors (strict) AND every
    input gate passes; otherwise STOP, or :class:`Refusal` for unusable inputs.

    CALF test: boundary = calf err of the plant with calves x0.80. GLOBAL test:
    boundary = calf err of the plant with every joint x0.75. A weaker calf sags more,
    so a smaller error means a stronger calf. Thigh errors are reported only.
    Input gates: boundary plants are exactly those (from the recorded ``g_true``), all
    references share artifact hashes and gains and did not fall, ``hw`` has the same
    hashes, gains within ``GAIN_TOL`` and, for hardware, a passing window and live data.
    """
    if ref_nominal is None:
        raise Refusal("the nominal reference is required")
    labels = validate_boundaries(ref_nominal, ref_calf_boundary, ref_all_boundary)
    hw_gates = check_hw_comparable(hw, ref_nominal)
    calf_hw = group_mean_err(hw, "calf")
    b_calf = group_mean_err(ref_calf_boundary, "calf")
    b_all = group_mean_err(ref_all_boundary, "calf")
    nom = group_mean_err(ref_nominal, "calf")
    if not (nom > 0.0 and b_calf > nom and b_all > nom and math.isfinite(calf_hw)):
        raise Refusal(
            f"sign/monotonicity check failed: calf err nominal {nom:.4f}, "
            f"calf-0.80 {b_calf:.4f}, all-0.75 {b_all:.4f}, hardware {calf_hw:.4f} "
            "(weaker plants must sag more)"
        )
    calf_test = calf_hw < b_calf
    global_test = calf_hw < b_all
    th_hw = group_mean_err(hw, "thigh")
    th_calf = group_mean_err(ref_calf_boundary, "thigh")
    th_all = group_mean_err(ref_all_boundary, "thigh")
    return {
        "calf_err_hw": calf_hw,
        "calf_boundary_err": b_calf,
        "calf_boundary_label": labels["calf_boundary"],
        "calf_test_go": calf_test,
        "global_boundary_err": b_all,
        "global_boundary_label": labels["global_boundary"],
        "global_test_go": global_test,
        "verdict": VERDICT_GO if (calf_test and global_test) else VERDICT_STOP,
        "input_gates": {
            "identity": dict(ref_identity(ref_nominal)),
            "labels": labels,
            "gains_tol": GAIN_TOL,
            **hw_gates,
        },
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
