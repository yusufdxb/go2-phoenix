"""Post-run report for one GO2 hardware session, from recorded evidence only.

Offline. Reads what a session already wrote and computes nothing during the
control loop:

* ``<session>/stage_<X>.json`` -- the staged gate verdicts and their checks.
* ``<session>/<stage><k>/bridge.jsonl`` -- the per-tick record the final bridge
  wrote (``phoenix.sim2real.bridge_telemetry``). Line 1 is the manifest, every
  later line is one tick.

Every number below is derived from fields that are actually in those files. A
metric that the recorded evidence cannot support is reported as UNAVAILABLE
with the evidence that would be needed, never estimated and never silently
dropped. The repo has already been burned by a number that looked measured and
was not (the 33% "sim vs hardware" slew gap compared two different metrics).

The startup contamination this exists to prevent
------------------------------------------------

The GO2 starts folded. Its calf reads -2.77 to -2.82 rad, below the audited URDF
limit of -2.7227, so a HOLD clips to exactly the limit and the limit margin
reads zero. Slew clipping is ~100% while the motors are off and ``q`` never
moves. Reporting one clip rate over a whole window therefore says "66.7%" or
"100%" about a start pose, not about the policy.

So the clip rate is reported TWICE, over disjoint windows of the same policy
ticks: ``startup`` is everything before the settled window, ``settled`` is the
final :data:`DEFAULT_SETTLE_WINDOW_S` seconds. The settled number is the one
that describes the standing robot. Neither is a substitute for the other and
the report prints both with their tick counts, so a settled window with too few
samples is visible rather than quietly averaged away.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Imported rather than re-derived on purpose: two definitions of the same metric
# is precisely the failure this repo has already paid for.
from .bridge_telemetry import (
    HARDWARE_SLEW_METRIC,
    _end_to_end_clip,
    read_telemetry,
    summarize,
)
from .motor_crc import PHOENIX_FOR_MOTOR

#: Final seconds of the policy window treated as "settled". One second at the
#: 50 Hz control rate is 50 ticks, enough to be meaningful and short enough to
#: exclude the transient from a folded start.
DEFAULT_SETTLE_WINDOW_S = 1.0

#: A tick gap longer than this multiple of the OBSERVED median gap counts as a
#: missed tick. Calibrating against the observed median rather than an assumed
#: 50 Hz means the number stays honest if the loop ran at a different rate.
MISSED_TICK_GAP_FACTOR = 1.5

UNAVAILABLE = "UNAVAILABLE"


class Unavailable:
    """A metric the recorded evidence cannot support, and what would fix that."""

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Unavailable({self.reason!r})"


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50_s": float(np.percentile(values, 50)),
        "p95_s": float(np.percentile(values, 95)),
        "max_s": float(values.max()),
    }


def _fault_onsets(ticks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Transitions into a fault, not every tick that stayed faulted.

    A latched fault repeats on every later tick; counting rows would report one
    trip as hundreds.
    """
    onsets: list[dict[str, Any]] = []
    previous: Any = None
    for t in ticks:
        current = t.get("fault")
        if current is not None and current != previous:
            onsets.append({"tick": t.get("tick"), "t_mono_ns": t.get("t_mono_ns"), "fault": current})
        previous = current
    return onsets


def _deadman_transitions(ticks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Changes in the deadman's observed state, with the tick they happened on."""
    transitions: list[dict[str, Any]] = []
    previous: Any = "__unset__"
    for t in ticks:
        current = (t.get("estop_value"), bool(t.get("deadman_source_ok")))
        if previous != "__unset__" and current != previous:
            transitions.append(
                {
                    "tick": t.get("tick"),
                    "t_mono_ns": t.get("t_mono_ns"),
                    "from": {"estop_value": previous[0], "deadman_source_ok": previous[1]},
                    "to": {"estop_value": current[0], "deadman_source_ok": current[1]},
                }
            )
        previous = current
    return transitions


def _release_to_safe_output(ticks: Sequence[Mapping[str, Any]]) -> Any:
    """Seconds from the deadman going unsafe to the first non-policy output.

    "Unsafe" is the deadman reading True (pressed/released per the wire
    convention the gate applies) or its source becoming not-ok. "Safe output" is
    the first tick whose mode is no longer ``policy``. Both come from recorded
    fields; if the run never had the transition, this is UNAVAILABLE rather than
    a fabricated zero.
    """
    release_ns: int | None = None
    for t in ticks:
        estop_bad = bool(t.get("estop_value")) or not bool(t.get("deadman_source_ok"))
        if release_ns is None and estop_bad and t.get("mode") == "policy":
            release_ns = t.get("t_mono_ns")
            continue
        if release_ns is not None and t.get("mode") != "policy":
            return {
                "seconds": (int(t["t_mono_ns"]) - int(release_ns)) / 1e9,
                "to_mode": t.get("mode"),
            }
    if release_ns is None:
        return Unavailable(
            "no tick recorded the deadman going unsafe while the policy held authority; "
            "needs a run where the deadman is released during the policy window"
        )
    return Unavailable(
        "the deadman went unsafe but no later tick left policy mode in this file; "
        "needs the ticks following the release"
    )


def _window_clip(rows: Sequence[Mapping[str, Any]]) -> Any:
    """End-to-end clip percentage over these rows, or why it is unavailable."""
    usable = [r for r in rows if r.get("cmd_is_new")]
    if not usable:
        return Unavailable(
            "no tick in this window carried a NEW policy command "
            "(cmd_is_new), so there is nothing to compare against a request"
        )
    pct, _per_joint = _end_to_end_clip(list(usable))
    if pct is None:
        return Unavailable(
            "ticks in this window carry no requested_target_unitree / "
            "final_target_unitree pair to compare"
        )
    return {"pct": float(pct), "ticks": len(usable)}


def analyze_run(manifest: Mapping[str, Any], ticks: Sequence[Mapping[str, Any]],
                settle_window_s: float = DEFAULT_SETTLE_WINDOW_S) -> dict[str, Any]:
    """Everything measurable about one bridge run."""
    out: dict[str, Any] = {
        "schema": "phoenix-hardware-run-report/v1",
        "metric": HARDWARE_SLEW_METRIC,
        "settle_window_s": settle_window_s,
        "stage": manifest.get("stage"),
        "live": bool(manifest.get("live")),
        "commit": (manifest.get("code_identity") or {}).get("sha"),
        "branch": (manifest.get("code_identity") or {}).get("branch"),
        "deploy_config": manifest.get("deploy_config"),
        "lock": manifest.get("lock"),
        "start_utc": manifest.get("start_utc"),
        "n_ticks": len(ticks),
    }
    # The existing verdict summary, carried through rather than recomputed.
    out["bridge_summary"] = summarize(manifest, list(ticks))

    stamped = [t for t in ticks if t.get("t_mono_ns") is not None]
    if len(stamped) < 2:
        out["duration_s"] = Unavailable("fewer than two ticks carry t_mono_ns")
        out["bridge_rate_hz"] = Unavailable("fewer than two ticks carry t_mono_ns")
        out["tick_gap"] = Unavailable("fewer than two ticks carry t_mono_ns")
        out["missed_ticks"] = Unavailable("fewer than two ticks carry t_mono_ns")
        gaps = np.asarray([])
        duration = None
    else:
        t_ns = np.asarray([int(t["t_mono_ns"]) for t in stamped], dtype=np.int64)
        duration = float((t_ns[-1] - t_ns[0]) / 1e9)
        gaps = np.diff(t_ns) / 1e9
        out["duration_s"] = duration
        out["bridge_rate_hz"] = (len(stamped) - 1) / duration if duration > 0 else None
        out["tick_gap"] = _percentiles(gaps)
        median_gap = float(np.median(gaps))
        threshold = median_gap * MISSED_TICK_GAP_FACTOR
        long_gaps = gaps[gaps > threshold]
        # A gap of k periods means k-1 ticks did not happen.
        missed = int(sum(max(0, round(g / median_gap) - 1) for g in long_gaps))
        out["missed_ticks"] = {
            "count": missed,
            "long_gaps": int(long_gaps.size),
            "observed_period_s": median_gap,
            "threshold_s": threshold,
        }

    policy_rows = [t for t in ticks if t.get("mode") == "policy"]
    out["policy_ticks"] = len(policy_rows)
    new_cmd_rows = [t for t in policy_rows if t.get("cmd_is_new")]
    out["policy_new_command_ticks"] = len(new_cmd_rows)

    stamped_policy = [t for t in policy_rows if t.get("t_mono_ns") is not None]
    if len(stamped_policy) >= 2:
        p_ns = np.asarray([int(t["t_mono_ns"]) for t in stamped_policy], dtype=np.int64)
        window = float((p_ns[-1] - p_ns[0]) / 1e9)
        out["policy_window_s"] = window
        out["policy_rate_hz"] = (
            (len(new_cmd_rows) - 1) / window if window > 0 and len(new_cmd_rows) > 1 else None
        )
    else:
        out["policy_window_s"] = Unavailable("fewer than two policy ticks carry t_mono_ns")
        out["policy_rate_hz"] = Unavailable("fewer than two policy ticks carry t_mono_ns")

    # ---- command freshness -------------------------------------------------
    params = manifest.get("gate_params") or {}
    watchdog_s = params.get("watchdog_s")
    ages = [float(t["cmd_age_s"]) for t in policy_rows if t.get("cmd_age_s") is not None]
    out["max_cmd_age_s"] = max(ages) if ages else Unavailable(
        "no policy tick recorded cmd_age_s"
    )
    if watchdog_s is None:
        out["stale_commands"] = Unavailable(
            "the manifest carries no gate_params.watchdog_s to compare command age against"
        )
    elif not ages:
        out["stale_commands"] = Unavailable("no policy tick recorded cmd_age_s")
    else:
        out["stale_commands"] = {
            "count": int(sum(1 for a in ages if a > float(watchdog_s))),
            "watchdog_s": float(watchdog_s),
        }

    # ---- interventions -----------------------------------------------------
    onsets = _fault_onsets(ticks)
    out["watchdog_trips"] = {"count": len(onsets), "onsets": onsets}
    hold_causes: dict[str, int] = {}
    for t in ticks:
        cause = t.get("hold_cause")
        if cause:
            hold_causes[str(cause)] = hold_causes.get(str(cause), 0) + 1
    out["intervention_reasons"] = {
        "hold_causes": hold_causes,
        "faults": out["bridge_summary"].get("faults") or [],
        "policy_abort_reasons": out["bridge_summary"].get("policy_abort_reasons") or [],
    }
    out["safety_interventions"] = len(onsets) + len(hold_causes)

    transitions = _deadman_transitions(ticks)
    out["deadman_transitions"] = {"count": len(transitions), "transitions": transitions}
    out["deadman_release_to_safe_output"] = _release_to_safe_output(ticks)

    # ---- clip rate, startup separated from settled --------------------------
    if not stamped_policy:
        reason = "no policy tick carries t_mono_ns, so the window cannot be split in time"
        out["clip_startup"] = Unavailable(reason)
        out["clip_settled"] = Unavailable(reason)
    else:
        p_ns = np.asarray([int(t["t_mono_ns"]) for t in stamped_policy], dtype=np.int64)
        # Half-open on purpose: settled is (cutoff, end], startup is [start, cutoff].
        # With >= the tick sitting exactly on the boundary falls in BOTH senses
        # and in practice lands in settled, which is how a folded-start tick
        # leaked into the settled rate and reported 0.16% where the standing
        # robot clipped nothing at all. The boundary tick belongs to startup.
        cutoff = p_ns[-1] - int(settle_window_s * 1e9)
        settled = [t for t in stamped_policy if int(t["t_mono_ns"]) > cutoff]
        startup = [t for t in stamped_policy if int(t["t_mono_ns"]) <= cutoff]
        out["clip_startup"] = _window_clip(startup)
        out["clip_settled"] = _window_clip(settled)
        out["clip_window_ticks"] = {"startup": len(startup), "settled": len(settled)}

    # ---- command path, raw to final ----------------------------------------
    perm = np.asarray(PHOENIX_FOR_MOTOR, dtype=np.int64)
    diffs: list[float] = []
    for t in new_cmd_rows:
        # policy.requested_target is in POLICY joint order; final_target_unitree
        # is in UNITREE motor order. Permute before subtracting, exactly as the
        # clip metric does, or this number is a joint-order artifact.
        req = (t.get("policy") or {}).get("requested_target")
        fin = t.get("final_target_unitree")
        if not req or not fin or any(v is None for v in (*req, *fin)):
            continue
        diffs.append(
            float(np.max(np.abs(np.asarray(fin, dtype=float) - np.asarray(req, dtype=float)[perm])))
        )
    out["max_request_to_final_delta_rad"] = max(diffs) if diffs else Unavailable(
        "no new-command tick carries both policy.requested_target and final_target_unitree"
    )

    # ---- halts -------------------------------------------------------------
    out["unexpected_halts"] = {
        "count": len(onsets),
        "note": (
            "every fault onset is listed; the policy window's own end also latches a "
            "fault by design, so compare against policy_abort_reasons before calling "
            "one unexpected"
        ),
    }
    return out


def analyze_session(session: Path, settle_window_s: float = DEFAULT_SETTLE_WINDOW_S) -> dict:
    """Every bridge run in a session directory, plus the recorded stage verdicts."""
    session = Path(session)
    if not session.is_dir():
        raise NotADirectoryError(f"not a session directory: {session}")

    stages: dict[str, Any] = {}
    for path in sorted(session.glob("stage_*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            stages[path.stem] = {"error": f"unreadable: {exc}"}
            continue
        stages[path.stem] = {
            "verdict": data.get("verdict"),
            "stage": data.get("stage"),
            "commit": (data.get("code_identity") or {}).get("sha"),
            "rehearsal": bool(data.get("rehearsal")),
            "failed_checks": [
                c.get("name") for c in data.get("checks") or [] if not c.get("ok")
            ],
        }

    runs: list[dict[str, Any]] = []
    for telemetry in sorted(session.rglob("bridge.jsonl")):
        manifest, ticks, _end = read_telemetry(telemetry)
        run = analyze_run(manifest, ticks, settle_window_s=settle_window_s)
        run["run_dir"] = str(telemetry.parent.relative_to(session))
        run["telemetry"] = str(telemetry)
        runs.append(run)

    return {
        "schema": "phoenix-hardware-session-report/v1",
        "session": str(session),
        "stages": stages,
        "runs": runs,
        "settle_window_s": settle_window_s,
    }


# ------------------------------------------------------------------ rendering
def _fmt(value: Any, unit: str = "", digits: int = 3) -> str:
    if isinstance(value, Unavailable):
        return UNAVAILABLE
    if value is None:
        return UNAVAILABLE
    if isinstance(value, float):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def _verdict(report: Mapping[str, Any]) -> str:
    """PASS / FAIL / INSUFFICIENT DATA, from the recorded evidence only."""
    if not report["runs"]:
        return "INSUFFICIENT DATA"
    verdicts = [s.get("verdict") for s in report["stages"].values() if s.get("verdict")]
    if any(v == "NO-GO" for v in verdicts):
        return "FAIL"
    for run in report["runs"]:
        if isinstance(run.get("clip_settled"), Unavailable):
            return "INSUFFICIENT DATA"
        if isinstance(run.get("duration_s"), Unavailable):
            return "INSUFFICIENT DATA"
    if not verdicts:
        return "INSUFFICIENT DATA"
    return "PASS" if all(v == "GO" for v in verdicts) else "INSUFFICIENT DATA"


def render(report: Mapping[str, Any]) -> str:
    lines: list[str] = ["PHOENIX HARDWARE RUN", ""]
    stages = report["stages"]
    if stages:
        lines.append("staged gate verdicts")
        for name, s in stages.items():
            mark = s.get("verdict") or "?"
            extra = " (REHEARSAL, never counts)" if s.get("rehearsal") else ""
            lines.append(f"  {name:12s} {mark}{extra}")
            for failed in s.get("failed_checks") or []:
                lines.append(f"      FAILED: {failed}")
        lines.append("")
    else:
        lines += ["staged gate verdicts: none recorded in this session", ""]

    if not report["runs"]:
        lines += [
            "no bridge.jsonl found under this session, so no run can be reported.",
            "",
            "RESULT: INSUFFICIENT DATA",
        ]
        return "\n".join(lines)

    for run in report["runs"]:
        lines.append(f"run {run['run_dir']}  stage: {run.get('stage')}  live: {run.get('live')}")
        lines.append(f"commit: {run.get('commit')}")
        lines.append(f"duration: {_fmt(run.get('duration_s'), ' s')}")
        lines.append("")
        lines.append(f"  policy rate:         {_fmt(run.get('policy_rate_hz'), ' Hz')}")
        lines.append(f"  bridge rate:         {_fmt(run.get('bridge_rate_hz'), ' Hz')}")
        gap = run.get("tick_gap")
        if isinstance(gap, Unavailable):
            lines.append(f"  tick gap p50/p95/max:{UNAVAILABLE}")
        else:
            lines.append(
                "  tick gap p50/p95/max:"
                f" {gap['p50_s'] * 1e3:.2f} / {gap['p95_s'] * 1e3:.2f} /"
                f" {gap['max_s'] * 1e3:.2f} ms"
            )
        missed = run.get("missed_ticks")
        lines.append(
            f"  missed ticks:        {UNAVAILABLE}"
            if isinstance(missed, Unavailable)
            else f"  missed ticks:        {missed['count']}"
            f" (observed period {missed['observed_period_s'] * 1e3:.2f} ms)"
        )
        stale = run.get("stale_commands")
        lines.append(
            f"  stale commands:      {UNAVAILABLE}"
            if isinstance(stale, Unavailable)
            else f"  stale commands:      {stale['count']} (watchdog {stale['watchdog_s']} s)"
        )
        lines.append(f"  max command age:     {_fmt(run.get('max_cmd_age_s'), ' s')}")
        lines.append(f"  watchdog trips:      {run['watchdog_trips']['count']}")
        lines.append(f"  safety interventions:{run['safety_interventions']:>3}")
        lines.append(f"  deadman transitions: {run['deadman_transitions']['count']}")
        rel = run.get("deadman_release_to_safe_output")
        lines.append(
            f"  release to safe out: {UNAVAILABLE}"
            if isinstance(rel, Unavailable)
            else f"  release to safe out: {rel['seconds'] * 1e3:.1f} ms (to {rel['to_mode']})"
        )
        lines.append("")
        for label in ("startup", "settled"):
            clip = run.get(f"clip_{label}")
            if isinstance(clip, Unavailable):
                lines.append(f"  clip rate {label:8s}: {UNAVAILABLE}")
            else:
                lines.append(
                    f"  clip rate {label:8s}: {clip['pct']:.2f}%  ({clip['ticks']} new-command ticks)"
                )
        lines.append(
            f"  max request->final:  {_fmt(run.get('max_request_to_final_delta_rad'), ' rad', 4)}"
        )
        lines.append(f"  unexpected halts:    {run['unexpected_halts']['count']}")
        reasons = run["intervention_reasons"]
        if reasons["faults"]:
            lines.append(f"  faults:              {', '.join(reasons['faults'])}")
        if reasons["policy_abort_reasons"]:
            lines.append(f"  policy aborts:       {', '.join(reasons['policy_abort_reasons'])}")
        if reasons["hold_causes"]:
            causes = ", ".join(f"{k}x{v}" for k, v in reasons["hold_causes"].items())
            lines.append(f"  hold causes:         {causes}")
        lines.append("")

    unavailable: list[str] = []
    for run in report["runs"]:
        for key, value in run.items():
            if isinstance(value, Unavailable):
                unavailable.append(f"  {run['run_dir']}.{key}: {value.reason}")
    if unavailable:
        lines.append("UNAVAILABLE metrics and what would be needed:")
        lines += unavailable
        lines.append("")

    lines.append(f"RESULT: {_verdict(report)}")
    return "\n".join(lines)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Unavailable):
        return {"unavailable": True, "reason": obj.reason}
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("session", type=Path, help="a logs/hw_sessions/<session> directory")
    p.add_argument(
        "--settle-window-s",
        type=float,
        default=DEFAULT_SETTLE_WINDOW_S,
        help="final seconds of the policy window treated as settled (default 1.0)",
    )
    p.add_argument("--json-out", type=Path, default=None, help="also write the report as JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze_session(args.session, settle_window_s=args.settle_window_s)
    print(render(report))
    out = args.json_out or (args.session / "hardware_run_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(report), indent=2) + "\n")
    print(f"\njson: {out}")
    return 0 if _verdict(report) == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
