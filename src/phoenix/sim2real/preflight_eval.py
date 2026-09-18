"""GO / NO-GO verdicts for the staged GO2 hardware gates, from recorded evidence only.

Every function here is pure: it takes what a stage recorded (a topic probe report,
a deadman trace, a bridge telemetry file already parsed, the deploy config and
lock) and returns a list of :class:`Check`. Nothing here talks to ROS, starts a
process, or moves a motor, so the verdict logic is covered by the no-hardware test
suite and a stage cannot pass because a script forgot to look at something.

Stages
------
A  offline artifacts and tests                       motors OFF
B  on-robot dryrun, bridge publishing /lowcmd_dry     motors OFF
C  real physical deadman verified                     motors OFF
D  LowState / IMU / joint-state freshness verified    motors OFF
E  live bridge hold-current                           motors LIVE (hold only)
F  H25 cmd=0 stand, 2 s of policy authority           motors LIVE
G  H25 cmd=0 stand, 5 s                               motors LIVE
H  H25 cmd=0 stand, 10 s, three attempts              motors LIVE

A stage counts only if every earlier stage counts, for the SAME code commit and
the SAME lock file. Nothing transitions between stages automatically.

Where each numeric threshold comes from (none is new):

* ``RATE_FLOOR_HZ = 200``: the ``/joint_states`` floor the previous
  ``scripts/harness_preflight.sh`` P5 enforced. Field notes measured ``/lowstate``
  at 500 Hz; April's bridge republished ``/joint_states`` at 311 Hz.
* Maximum inter-message gap: the deploy config's ``safety.sensor_timeout_s`` for
  sensors and ``safety.estop_timeout_s`` for the deadman heartbeat. A gap that
  long would already trip the node's own fail-closed watchdog.
* Command / LowCmd gaps: the bridge's command watchdog (``watchdog_s``).
* Attitude: ``DEFAULT_ATTITUDE_INTERVENTION_RAD`` = 0.40 rad on BOTH pitch and
  roll, the policy node's abort threshold and the same number
  ``safety.attitude_intervention_rad`` overrides. It is deliberately below the
  run card's 25 degree operator-halt instruction so software intervenes first.
  This is NOT the simulator analysis bar (pitch 0.8 / roll 0.6, see
  ``phoenix.real_world.failure_detector.sim_analysis_thresholds``); the two were
  the same dataclass default until 2026-09-17 and are now split.
* Hold-test motion bound: one slew cap, ``MAX_DELTA_PER_STEP_RAD``.
* Stand authority tolerance: two control periods at the configured rate.

The hardware slew percentage is REPORTED by the stand stages and is never a pass
criterion: its legacy 5% threshold was defined on a different quantity
(``configs/sim2real/deploy_stand_h25.yaml`` header) and no corrected-metric
simulator baseline exists for this checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from phoenix.real_world.failure_detector import (
    DEFAULT_ATTITUDE_INTERVENTION_RAD,
    resolve_attitude_intervention_rad,
)

from .actuator_gate import REAL_DEADMAN_NODE_NAMES
from .go2_model import (
    JOINT_POSITION_LIMITS_RAD,
    LIMIT_ABORT_BAND_RAD,
    POLICY_JOINT_ORDER,
    UNITREE_MOTOR_ORDER,
    limits_in_order,
)
from .safety import MAX_DELTA_PER_STEP_RAD

#: Services and nodes that must not be running during a Phoenix session.
#:
#: ``come-here.service`` autostarts on Jetson boot. Measured on 2026-09-17 it
#: starves the payload (132 topics against 109, ``/lowstate`` at 349 Hz against
#: the field notes' 500) AND it can command the robot, so a Phoenix session
#: sharing the robot with it is both a degraded measurement and a second
#: uncommanded authority over the motors. The 2026-09-17 audit recorded that no
#: interlock refusing to run while it is active existed; this is that interlock.
COMPETING_SERVICES: tuple[str, ...] = ("come-here.service",)

#: Substrings that identify a competing node in the ROS graph. Matched
#: case-insensitively against node names, because a service can be started by
#: hand under a different unit name but still brings the same nodes up.
COMPETING_NODE_SUBSTRINGS: tuple[str, ...] = ("come_here", "come-here", "comehere")

#: ``/lowstate`` floor below which the payload is assumed to be starved. The
#: field notes measure 500 Hz; 349 Hz was recorded with come-here running.
LOWSTATE_STARVATION_FLOOR_HZ = 400.0

STAGE_SCHEMA = "phoenix-preflight-stage/v1"
STAGES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")
STAGE_TITLES: dict[str, str] = {
    "A": "offline artifacts, parity and tests (motors off)",
    "B": "on-robot dryrun to /lowcmd_dry (motors off)",
    "C": "real physical deadman verified (motors off)",
    "D": "LowState / IMU / joint-state freshness (motors off)",
    "E": "live bridge hold-current (motors LIVE, hold only)",
    "F": "H25 cmd=0 stand, 2 s (motors LIVE)",
    "G": "H25 cmd=0 stand, 5 s (motors LIVE)",
    "H": "H25 cmd=0 stand, 10 s x 3 attempts (motors LIVE)",
}
LIVE_STAGES = frozenset({"E", "F", "G", "H"})
STAND_AUTHORITY_S: dict[str, float] = {"F": 2.0, "G": 5.0, "H": 10.0}
H_ATTEMPTS = 3

RATE_FLOOR_HZ = 200.0
SENSOR_TOPICS = ("/lowstate", "/joint_states", "/imu/data")
COMMAND_TOPIC_DEFAULT = "/joint_group_position_controller/command"

#: Faults that may legitimately appear in a live stage's telemetry AFTER the
#: stage's own intended end. Anything else is a NO-GO.
_E_ALLOWED_FAULTS = frozenset({"estop_asserted", "bridge_shutdown"})
_STAND_END_FAULT = "policy_abort:authority_window_complete"
_STAND_ALLOWED_AFTER_END = frozenset({"estop_asserted", "bridge_shutdown"})


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    gating: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _age_within(age: float | None, limit: float) -> bool:
    """Fail closed: a missing age (no operating ticks recorded) is not fresh."""
    return age is not None and float(age) <= limit


def verdict(checks: Sequence[Check]) -> str:
    gating = [c for c in checks if c.gating]
    return "GO" if gating and all(c.ok for c in gating) else "NO-GO"


def _check(name: str, ok: bool, detail: str, gating: bool = True) -> Check:
    return Check(name=name, ok=bool(ok), detail=detail, gating=gating)


# ------------------------------------------------------------------ ledger
def ledger_status(
    records: Mapping[str, Mapping[str, Any] | None],
    *,
    current_sha: str | None,
    current_lock_sha256: str | None,
) -> dict[str, Any]:
    """Which stages count, and the next one permitted.

    A stage record counts only when its verdict is GO, it was produced by the
    current commit against the current lock file, every earlier stage counts, and
    it is not older than the stage before it.
    """
    rows: list[dict[str, Any]] = []
    all_prior_go = True
    prior_utc = ""
    next_stage: str | None = None
    for stage in STAGES:
        rec = records.get(stage)
        if rec is None:
            state, why = "MISSING", "no evidence recorded"
        elif rec.get("schema") != STAGE_SCHEMA:
            state, why = "INVALID", f"schema {rec.get('schema')!r}"
        elif (rec.get("code_identity") or {}).get("sha") != current_sha:
            state, why = "STALE", (
                f"recorded at commit {(rec.get('code_identity') or {}).get('sha')}, "
                f"current {current_sha}"
            )
        elif rec.get("lock_file_sha256") != current_lock_sha256:
            state, why = "STALE", "recorded against a different lock file"
        elif rec.get("verdict") != "GO":
            state, why = "NO-GO", "; ".join(
                c["name"]
                for c in rec.get("checks", [])
                if c.get("gating", True) and not c.get("ok")
            )
        elif rec.get("rehearsal"):
            state, why = "REHEARSAL", "rehearsal evidence (no robot) never counts"
        elif not all_prior_go:
            state, why = "BLOCKED", "an earlier stage does not count"
        elif str(rec.get("utc", "")) < prior_utc:
            state, why = "BLOCKED", "older than the stage before it"
        else:
            state, why = "GO", ""
        if state != "GO":
            all_prior_go = False
            if next_stage is None:
                next_stage = stage
        else:
            prior_utc = str((rec or {}).get("utc", ""))
        rows.append({"stage": stage, "title": STAGE_TITLES[stage], "state": state, "why": why})
    go = {r["stage"] for r in rows if r["state"] == "GO"}
    return {
        "stages": rows,
        "next_stage": next_stage,
        "ready_for_live_hold_E": {"A", "B", "C", "D"} <= go,
        "ready_for_stand_F": {"A", "B", "C", "D", "E"} <= go,
        "stand_gate_passed": set(STAGES) <= go,
    }


# ------------------------------------------------------------- topic probes
def _topic(probe: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return (probe.get("topics") or {}).get(name) or {}


def rate_checks(
    probe: Mapping[str, Any],
    topic: str,
    *,
    floor_hz: float | None,
    max_gap_s: float,
) -> list[Check]:
    t = _topic(probe, topic)
    count = int(t.get("count") or 0)
    rate = t.get("rate_hz")
    gap = t.get("max_gap_s")
    checks = [_check(f"{topic} flowing", count >= 2, f"{count} messages")]
    if floor_hz is not None:
        checks.append(
            _check(
                f"{topic} rate >= {floor_hz:g} Hz",
                rate is not None and rate >= floor_hz,
                f"measured {rate} Hz",
            )
        )
    checks.append(
        _check(
            f"{topic} max gap < {max_gap_s:g} s",
            gap is not None and gap < max_gap_s,
            f"measured max gap {gap} s",
        )
    )
    return checks


def publisher_check(
    probe: Mapping[str, Any], topic: str, expected_nodes: set[str], *, exact: bool = True
) -> Check:
    pubs = set(_topic(probe, topic).get("publishers") or [])
    ok = pubs == expected_nodes if exact else expected_nodes <= pubs
    return _check(
        f"{topic} publisher is {sorted(expected_nodes)}", ok, f"publishers {sorted(pubs)}"
    )


def sensor_content_checks(
    probe: Mapping[str, Any],
    *,
    attitude_intervention_rad: float | None = None,
) -> list[Check]:
    """Physically plausible sensor CONTENT, not just flow (stage D)."""
    attitude_limit = (
        DEFAULT_ATTITUDE_INTERVENTION_RAD
        if attitude_intervention_rad is None
        else float(attitude_intervention_rad)
    )
    checks: list[Check] = []
    low = _topic(probe, "/lowstate")
    q_min, q_max = low.get("q_min"), low.get("q_max")
    if q_min is None or q_max is None:
        checks.append(
            _check("/lowstate joint positions recorded", False, "probe has no q_min/q_max")
        )
    else:
        lo, hi = limits_in_order(UNITREE_MOTOR_ORDER)
        q_min_a, q_max_a = np.asarray(q_min, dtype=float), np.asarray(q_max, dtype=float)
        bad = [
            f"{n}[{a:.3f},{b:.3f}]"
            for n, a, b, low_lim, high_lim in zip(
                UNITREE_MOTOR_ORDER, q_min_a, q_max_a, lo, hi, strict=True
            )
            if a < low_lim - LIMIT_ABORT_BAND_RAD or b > high_lim + LIMIT_ABORT_BAND_RAD
        ]
        checks.append(
            _check(
                "/lowstate joint positions physically possible",
                not bad,
                "all 12 within hard limits +/- one slew cap" if not bad else f"outside: {bad}",
            )
        )
    checks.append(
        _check(
            "/lowstate has no non-finite joint values",
            int(low.get("non_finite") or 0) == 0 and low.get("non_finite") is not None,
            f"non_finite={low.get('non_finite')}",
        )
    )
    imu = _topic(probe, "/imu/data")
    roll = imu.get("max_abs_roll_rad")
    pitch = imu.get("max_abs_pitch_rad")
    checks.append(
        _check(
            "/imu/data attitude inside the abort thresholds",
            roll is not None
            and pitch is not None
            and roll < attitude_limit
            and pitch < attitude_limit,
            f"max |roll| {roll} (< {attitude_limit}), max |pitch| {pitch} (< {attitude_limit})",
        )
    )
    checks.append(
        _check(
            "/imu/data has no non-finite values",
            imu.get("non_finite") == 0,
            f"non_finite={imu.get('non_finite')}",
        )
    )
    names = set(_topic(probe, "/joint_states").get("names") or [])
    missing = sorted(set(POLICY_JOINT_ORDER) - names)
    checks.append(
        _check("/joint_states names all 12 policy joints", not missing, f"missing {missing}")
    )
    return checks


# ------------------------------------------------------------ bridge records
def contention_checks(probe: Mapping[str, Any]) -> list[Check]:
    """Refuse to run while another system shares the robot.

    Takes what a contention probe recorded: the active systemd units it found,
    the ROS node names it saw, and optionally the measured ``/lowstate`` rate.
    Every check is GATING. Absent evidence is a FAIL, not a pass: a stage that
    counted because nobody looked is exactly the failure mode the staged gates
    exist to prevent.
    """

    checks: list[Check] = []
    units = probe.get("active_units")
    if units is None:
        checks.append(
            _check(
                "competing services were checked",
                False,
                "probe recorded no active_units; run the contention probe first",
            )
        )
    else:
        active = sorted({str(u) for u in units})
        offenders = [u for u in active if u in COMPETING_SERVICES]
        checks.append(
            _check(
                f"none of {list(COMPETING_SERVICES)} is active",
                not offenders,
                (
                    (
                        f"active competing units: {offenders}; stop them "
                        "(sudo systemctl stop <unit>) before this session"
                    )
                    if offenders
                    else f"{len(active)} active unit(s) recorded, none competing"
                ),
            )
        )

    nodes = probe.get("ros_nodes")
    if nodes is None:
        checks.append(
            _check(
                "ROS graph was checked for competing nodes",
                False,
                "probe recorded no ros_nodes; run the contention probe first",
            )
        )
    else:
        names = [str(n) for n in nodes]
        offenders = sorted({n for n in names for s in COMPETING_NODE_SUBSTRINGS if s in n.lower()})
        checks.append(
            _check(
                "no competing node in the ROS graph",
                not offenders,
                (
                    f"competing nodes: {offenders}"
                    if offenders
                    else f"{len(names)} node(s), none competing"
                ),
            )
        )

    rate = probe.get("lowstate_rate_hz")
    if rate is not None:
        checks.append(
            _check(
                f"/lowstate at or above {LOWSTATE_STARVATION_FLOOR_HZ:g} Hz",
                float(rate) >= LOWSTATE_STARVATION_FLOOR_HZ,
                (
                    f"measured {float(rate):.1f} Hz; below the floor means the payload is "
                    "starved even if no competing unit was named"
                ),
            )
        )
    return checks


def manifest_checks(
    manifest: Mapping[str, Any],
    *,
    live: bool,
    stage: str,
    lock: Mapping[str, Any],
    expected_sha: str | None,
) -> list[Check]:
    checks: list[Check] = []
    checks.append(
        _check(
            "bridge mode",
            bool(manifest.get("live")) == live,
            f"live={manifest.get('live')}, expected {live}",
        )
    )
    checks.append(
        _check(
            "bridge stage label", manifest.get("stage") == stage, f"stage={manifest.get('stage')!r}"
        )
    )
    problems = manifest.get("startup_problems")
    checks.append(_check("bridge started with no startup problems", problems == [], f"{problems}"))
    ident = manifest.get("code_identity") or {}
    sha = ident.get("sha")
    checks.append(
        _check(
            "bridge ran the expected commit",
            bool(sha) and (expected_sha is None or str(sha).startswith(str(expected_sha))),
            f"bridge commit {sha}, expected {expected_sha}",
        )
    )
    checks.append(
        _check(
            "bridge code identity clean",
            manifest.get("code_identity_problems") == [],
            f"{manifest.get('code_identity_problems')}",
        )
    )
    semantic = (manifest.get("deploy_config") or {}).get("semantic_sha256")
    want_semantic = (lock.get("deploy_config") or {}).get("semantic_sha256")
    checks.append(
        _check(
            "bridge deploy config matches lock",
            semantic == want_semantic,
            f"{semantic} vs lock {want_semantic}",
        )
    )
    artifacts = manifest.get("artifacts") or {}
    for role in ("policy.onnx", "policy.onnx.data"):
        got = (artifacts.get(role) or {}).get("sha256")
        want = ((lock.get("artifacts") or {}).get(role) or {}).get("sha256")
        checks.append(
            _check(
                f"bridge saw locked {role}",
                got is not None and got == want,
                f"{got} vs lock {want}",
            )
        )
    params = manifest.get("gate_params") or {}
    checks.append(
        _check(
            "bridge slew cap is the deploy constant",
            params.get("max_delta") == MAX_DELTA_PER_STEP_RAD,
            f"max_delta={params.get('max_delta')}",
        )
    )
    checks.append(
        _check(
            "bridge real-deadman requirement matches mode",
            bool(params.get("deadman_required")) == live,
            f"deadman_required={params.get('deadman_required')}",
        )
    )
    limits = manifest.get("joint_limits_rad") or {}
    checks.append(
        _check(
            "bridge joint limits are the audited table",
            {k: tuple(v) for k, v in limits.items()} == JOINT_POSITION_LIMITS_RAD,
            "table matches go2_model" if limits else "no limits recorded",
        )
    )
    return checks


def _policy_rows(ticks: Sequence[Mapping[str, Any]]) -> tuple[int | None, int | None]:
    idx = [i for i, t in enumerate(ticks) if t.get("mode") == "policy"]
    return (idx[0], idx[-1]) if idx else (None, None)


def observation_checks(ticks: Sequence[Mapping[str, Any]]) -> list[Check]:
    """Every policy command declared zeros + stand-only and fed a zero command."""
    bad: list[str] = []
    for t in ticks:
        if t.get("mode") != "policy":
            continue
        pol = t.get("policy") or {}
        if pol.get("obs_source_code") != 0.0 or pol.get("stand_only") != 1.0:
            bad.append(
                f"tick {t.get('tick')}: source={pol.get('obs_source_code')} stand_only={pol.get('stand_only')}"
            )
        fed = pol.get("velocity_command_fed") or []
        if any(v not in (0.0, None) for v in fed) or any(
            v not in (0.0, None) for v in (pol.get("cmd_vel_received") or [])
        ):
            bad.append(f"tick {t.get('tick')}: nonzero velocity command")
        if any(v not in (0.0, None) for v in (pol.get("base_lin_vel_fed") or [])):
            bad.append(f"tick {t.get('tick')}: nonzero base_lin_vel fed in zeros mode")
    return [
        _check(
            "policy fed zeros base_lin_vel and a zero command, stand-only, every tick",
            not bad,
            "ok" if not bad else "; ".join(bad[:5]),
        )
    ]


def dryrun_checks(
    *,
    processes: Mapping[str, Any],
    probe: Mapping[str, Any],
    manifest: Mapping[str, Any],
    ticks: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    cfg: Mapping[str, Any],
    lock: Mapping[str, Any],
    expected_sha: str | None,
    watchdog_s: float,
) -> list[Check]:
    """Stage B: the exact locked config, every process alive, rates, outputs, no fault."""
    safety = cfg.get("safety") or {}
    sensor_timeout = float(safety["sensor_timeout_s"])
    command_topic = (cfg.get("topics") or {}).get("joint_command", COMMAND_TOPIC_DEFAULT)
    checks: list[Check] = []
    for phase in ("after_startup", "before_teardown"):
        alive = processes.get(phase) or {}
        dead = sorted(name for name, ok in alive.items() if not ok)
        checks.append(
            _check(
                f"every launched process alive {phase}",
                bool(alive) and not dead,
                f"dead: {dead}" if dead else f"alive: {sorted(alive)}",
            )
        )
    for topic in SENSOR_TOPICS:
        checks.extend(rate_checks(probe, topic, floor_hz=RATE_FLOOR_HZ, max_gap_s=sensor_timeout))
    checks.extend(rate_checks(probe, command_topic, floor_hz=None, max_gap_s=watchdog_s))
    checks.append(publisher_check(probe, command_topic, {"phoenix_policy_node"}))
    checks.extend(rate_checks(probe, "/lowcmd_dry", floor_hz=None, max_gap_s=watchdog_s))
    checks.append(publisher_check(probe, "/lowcmd_dry", {"phoenix_lowcmd_bridge"}))
    live_pubs = set(_topic(probe, "/lowcmd").get("publishers") or [])
    checks.append(
        _check(
            "nothing of ours publishes /lowcmd in a dryrun",
            "phoenix_lowcmd_bridge" not in live_pubs,
            f"/lowcmd publishers {sorted(live_pubs)}",
        )
    )
    checks.extend(
        manifest_checks(manifest, live=False, stage="B", lock=lock, expected_sha=expected_sha)
    )

    first, last = _policy_rows(ticks)
    checks.append(
        _check(
            "bridge reached policy mode",
            first is not None,
            f"{summary.get('policy_ticks')} policy ticks",
        )
    )
    if first is not None:
        end = next(
            (i for i, t in enumerate(ticks) if "bridge_shutdown" in (t.get("faults") or [])),
            len(ticks),
        )
        interruptions = [
            f"tick {t.get('tick')} {t.get('mode')}/{t.get('hold_cause')}"
            for t in ticks[first:end]
            if t.get("mode") != "policy"
        ]
        checks.append(
            _check(
                "policy authority continuous from first policy tick to teardown",
                not interruptions,
                "continuous" if not interruptions else "; ".join(interruptions[:5]),
            )
        )
    faults = set(summary.get("faults") or [])
    checks.append(
        _check(
            "no latched fault other than teardown",
            faults <= {"bridge_shutdown"},
            f"faults {sorted(faults)}",
        )
    )
    checks.append(
        _check(
            "LowState fresh while the bridge had authority",
            _age_within(summary.get("max_lowstate_age_s"), sensor_timeout),
            f"max LowState age {summary.get('max_lowstate_age_s')} s",
        )
    )
    checks.extend(observation_checks(ticks))
    checks.append(
        _check(
            "hardware slew metric recorded (reported, not gating)",
            True,
            f"end-to-end {summary.get('end_to_end_clip_pct')}%, bridge layer "
            f"{summary.get('bridge_slew_clip_pct')}%, policy node "
            f"{summary.get('policy_node_slew_clip_pct')}%",
            gating=False,
        )
    )
    checks.append(
        _check(
            "estop source in a dryrun is the synthetic heartbeat, NOT a deadman",
            True,
            f"deadman_source_ok seen: {sorted({bool(t.get('deadman_source_ok')) for t in ticks})}",
            gating=False,
        )
    )
    return checks


def deadman_trace_checks(trace: Mapping[str, Any], *, estop_timeout_s: float) -> list[Check]:
    """Stage C: the operator's physical switch, and only it, drives /phoenix/estop."""
    messages = sorted(trace.get("messages") or [], key=lambda m: m["t"])
    polls = trace.get("publisher_polls") or []
    phases = {p["name"]: p for p in trace.get("phases") or []}
    checks: list[Check] = []

    seen = [tuple(sorted(p.get("publishers") or [])) for p in polls if p.get("publishers")]
    bad_polls = [s for s in seen if len(s) != 1 or s[0] not in REAL_DEADMAN_NODE_NAMES]
    checks.append(
        _check(
            "exactly one real deadman node publishes /phoenix/estop",
            bool(seen) and not bad_polls,
            f"publisher sets seen {sorted(set(seen))}",
        )
    )
    times = [m["t"] for m in messages]
    gap = float(np.max(np.diff(times))) if len(times) > 1 else None
    checks.append(
        _check(
            f"heartbeat never silent for {estop_timeout_s:g} s",
            gap is not None and gap < estop_timeout_s,
            f"max gap {gap} s over {len(times)} messages",
        )
    )

    def in_phase(name: str) -> list[Mapping[str, Any]]:
        p = phases.get(name)
        if p is None:
            return []
        return [m for m in messages if p["t_start"] <= m["t"] <= p["t_end"]]

    hold = in_phase("hold")
    streak_start = None
    for m in hold:
        if m["value"]:
            streak_start = None
        elif streak_start is None:
            streak_start = m["t"]
    hold_ok = bool(hold) and not hold[-1]["value"] and streak_start is not None
    streak = (phases["hold"]["t_end"] - streak_start) if hold_ok else 0.0
    checks.append(
        _check(
            "holding the deadman gives a sustained False",
            hold_ok and streak >= estop_timeout_s,
            f"final False streak {streak:.2f} s (needs >= {estop_timeout_s:g} s)",
        )
    )
    release = in_phase("release")
    first_true = next((i for i, m in enumerate(release) if m["value"]), None)
    release_ok = first_true is not None and all(m["value"] for m in release[first_true:])
    checks.append(
        _check(
            "releasing the deadman asserts True and it stays True",
            release_ok,
            f"{len(release)} messages, first True at index {first_true}",
        )
    )
    rehold = in_phase("rehold")
    checks.append(
        _check(
            "holding again returns False (switch is live, not stuck)",
            bool(rehold) and not rehold[-1]["value"],
            f"{len(rehold)} messages",
        )
    )
    return checks


def hold_test_checks(
    *,
    manifest: Mapping[str, Any],
    ticks: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    cfg: Mapping[str, Any],
    lock: Mapping[str, Any],
    expected_sha: str | None,
    duration_s: float,
    hold_kp: float,
    hold_kd: float,
    watchdog_s: float,
) -> list[Check]:
    """Stage E: live bridge, no policy, holds measured posture without moving the robot."""
    sensor_timeout = float((cfg.get("safety") or {})["sensor_timeout_s"])
    checks = manifest_checks(manifest, live=True, stage="E", lock=lock, expected_sha=expected_sha)
    publishing = [t for t in ticks if t.get("publish")]
    end = next((i for i, t in enumerate(publishing) if t.get("mode") == "damp"), len(publishing))
    held = publishing[:end]
    checks.append(
        _check("bridge published hold commands", len(held) > 0, f"{len(held)} hold ticks")
    )
    not_hold = [t.get("mode") for t in held if t.get("mode") != "hold"]
    checks.append(
        _check(
            "every command before teardown was a hold", not not_hold, f"other modes {set(not_hold)}"
        )
    )
    gains = {(t.get("kp"), t.get("kd")) for t in held}
    checks.append(
        _check("hold used the hold gains", gains <= {(hold_kp, hold_kd)}, f"gains {sorted(gains)}")
    )
    checks.append(
        _check(
            "no policy node commanded",
            summary.get("policy_ticks") == 0,
            f"{summary.get('policy_ticks')} policy ticks",
        )
    )
    faults = set(summary.get("faults") or [])
    checks.append(
        _check(
            "no fault other than deadman release or teardown",
            faults <= _E_ALLOWED_FAULTS,
            f"faults {sorted(faults)}",
        )
    )
    first_ok = next((i for i, t in enumerate(held) if t.get("deadman_source_ok")), None)
    lost = (
        [t.get("tick") for t in held[first_ok:] if not t.get("deadman_source_ok")]
        if first_ok is not None
        else []
    )
    checks.append(
        _check(
            "real deadman verified as the only estop source, and stayed so",
            first_ok is not None and not lost,
            (
                "never verified"
                if first_ok is None
                else (
                    f"lost at ticks {lost[:5]}"
                    if lost
                    else f"verified from tick {held[first_ok].get('tick')}"
                )
            ),
        )
    )
    stale = [t.get("tick") for t in held if not t.get("lowstate_fresh")]
    checks.append(
        _check("LowState fresh on every hold tick", not stale, f"stale ticks {stale[:5]}")
    )
    span = (int(held[-1]["t_mono_ns"]) - int(held[0]["t_mono_ns"])) / 1e9 if len(held) > 1 else 0.0
    checks.append(_check(f"held for >= {duration_s:g} s", span >= duration_s, f"held {span:.2f} s"))
    checks.append(
        _check(
            f"bridge tick gap < {watchdog_s:g} s",
            (summary.get("max_tick_gap_s") or 1e9) < watchdog_s,
            f"max tick gap {summary.get('max_tick_gap_s')} s",
        )
    )
    excursion = summary.get("q_max_excursion_rad") or {}
    moved = {k: round(v, 4) for k, v in excursion.items() if v > MAX_DELTA_PER_STEP_RAD}
    checks.append(
        _check(
            f"no joint moved more than one slew cap ({MAX_DELTA_PER_STEP_RAD} rad) under hold",
            bool(excursion) and not moved,
            (
                f"moved: {moved}"
                if moved
                else f"max excursion {max(excursion.values(), default=None)} rad"
            ),
        )
    )
    checks.append(
        _check(
            "max LowState age within sensor timeout",
            _age_within(summary.get("max_lowstate_age_s"), sensor_timeout),
            f"{summary.get('max_lowstate_age_s')} s",
        )
    )
    return checks


def stand_checks(
    *,
    stage: str,
    manifest: Mapping[str, Any],
    ticks: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    cfg: Mapping[str, Any],
    lock: Mapping[str, Any],
    expected_sha: str | None,
    authority_s: float,
    watchdog_s: float,
    operator_confirmed_stand: bool,
) -> list[Check]:
    """Stages F / G / H (one attempt): the policy held a cmd=0 stand for its whole window."""
    control = cfg.get("control") or {}
    rate_hz = float(control.get("rate_hz", 50))
    sensor_timeout = float((cfg.get("safety") or {})["sensor_timeout_s"])
    attitude_limit = resolve_attitude_intervention_rad(dict(cfg.get("safety") or {}))
    checks = manifest_checks(manifest, live=True, stage=stage, lock=lock, expected_sha=expected_sha)

    first, last = _policy_rows(ticks)
    checks.append(
        _check(
            "policy took authority",
            first is not None,
            f"{summary.get('policy_ticks')} policy ticks",
        )
    )
    window = summary.get("policy_window_s") or 0.0
    need = authority_s - 2.0 / rate_hz
    checks.append(
        _check(
            f"policy authority lasted the full {authority_s:g} s window",
            window >= need,
            f"policy window {window:.3f} s (needs >= {need:.3f} s)",
        )
    )
    if first is not None and last is not None:
        interruptions = [
            f"tick {t.get('tick')} {t.get('mode')}/{t.get('hold_cause')}"
            for t in ticks[first : last + 1]
            if t.get("mode") != "policy"
        ]
        checks.append(
            _check(
                "authority continuous (no hold or damp inside the window)",
                not interruptions,
                "; ".join(interruptions[:5]) or "continuous",
            )
        )
    reasons = summary.get("policy_abort_reasons") or []
    checks.append(
        _check(
            "the policy ended only because its window completed",
            reasons == ["authority_window_complete"],
            f"policy abort reasons {reasons}",
        )
    )
    faults = list(summary.get("faults") or [])
    ok_faults = (
        bool(faults)
        and faults[0] == _STAND_END_FAULT
        and set(faults[1:]) <= _STAND_ALLOWED_AFTER_END
    )
    checks.append(
        _check("no bridge fault before the window ended", ok_faults, f"faults in order {faults}")
    )
    checks.append(
        _check(
            "no requested target beyond the hard-limit abort band",
            not any(f.startswith("target_beyond_limit") for f in faults),
            f"limit clips {summary.get('limit_clip_counts')}",
        )
    )
    policy_ticks = [t for t in ticks if t.get("mode") == "policy"]
    stale = [t.get("tick") for t in policy_ticks if not t.get("lowstate_fresh")]
    checks.append(
        _check(
            "LowState fresh on every policy tick",
            bool(policy_ticks) and not stale,
            f"stale {stale[:5]}",
        )
    )
    ages = [float(t["cmd_age_s"]) for t in policy_ticks if t.get("cmd_age_s") is not None]
    checks.append(
        _check(
            f"policy command age < watchdog ({watchdog_s:g} s) on every policy tick",
            bool(ages) and max(ages) <= watchdog_s,
            f"max command age {max(ages, default=None)} s",
        )
    )
    deadman = [bool(t.get("deadman_source_ok")) for t in policy_ticks]
    checks.append(
        _check(
            "real deadman armed throughout authority",
            bool(deadman) and all(deadman),
            f"{len(deadman)} ticks",
        )
    )
    roll, pitch = summary.get("max_abs_roll_rad"), summary.get("max_abs_pitch_rad")
    checks.append(
        _check(
            "attitude stayed inside the abort thresholds",
            roll is not None
            and pitch is not None
            and roll < attitude_limit
            and pitch < attitude_limit,
            f"max |roll| {roll}, max |pitch| {pitch}",
        )
    )
    checks.extend(observation_checks(ticks))
    checks.append(
        _check(
            "max LowState age within sensor timeout",
            _age_within(summary.get("max_lowstate_age_s"), sensor_timeout),
            f"{summary.get('max_lowstate_age_s')} s",
        )
    )
    checks.append(
        _check(
            "operator observed a held stand (feet on ground, no collapse, oscillation or buzz)",
            operator_confirmed_stand,
            "operator confirmation recorded" if operator_confirmed_stand else "not confirmed",
        )
    )
    checks.append(
        _check(
            "hardware slew clip activation (REPORTED, not a pass criterion)",
            True,
            f"{summary.get('metric')}: end-to-end {summary.get('end_to_end_clip_pct')}% "
            f"(per joint {summary.get('end_to_end_clip_pct_per_joint')}); "
            f"bridge layer {summary.get('bridge_slew_clip_pct')}% "
            f"(all policy ticks {summary.get('bridge_slew_clip_pct_all_policy_ticks')}%), "
            f"policy node {summary.get('policy_node_slew_clip_pct')}%, per joint "
            f"{summary.get('bridge_slew_clip_pct_per_joint')}",
            gating=False,
        )
    )
    return checks


__all__ = [
    "H_ATTEMPTS",
    "LIVE_STAGES",
    "RATE_FLOOR_HZ",
    "SENSOR_TOPICS",
    "STAGES",
    "STAGE_SCHEMA",
    "STAGE_TITLES",
    "STAND_AUTHORITY_S",
    "Check",
    "deadman_trace_checks",
    "dryrun_checks",
    "hold_test_checks",
    "ledger_status",
    "manifest_checks",
    "observation_checks",
    "publisher_check",
    "rate_checks",
    "sensor_content_checks",
    "stand_checks",
    "verdict",
]
