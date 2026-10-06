"""walk_command: profile envelope and ramps, deadman gating state machine, W harness wiring.

No ROS and no robot: the pure core is exercised directly, the harness only for its
refusals and its static shape.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from phoenix.sim2real import walk_command as wc
from phoenix.sim2real.walk_command import (
    ENVELOPE,
    LEAD_IN_S,
    PRESETS,
    RAMP_LIMITS,
    TAIL_S,
    WalkPreset,
    WalkRunner,
    command_at,
    profile_duration,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "scripts" / "harness_preflight.sh"
S = 1_000_000_000
TICK_NS = S // 20  # 20 Hz publish loop
ESTOP_EVERY = 2  # estop at 10 Hz


# ------------------------------------------------------------------- profile


def test_expected_presets_exist() -> None:
    assert set(PRESETS) == {"vx03_2s", "vx03_4s", "vx05_3s", "yaw05_3s", "vxm03_2s"}
    assert (PRESETS["vx03_2s"].vx, PRESETS["vx03_2s"].motion_s) == (0.3, 2.0)
    assert (PRESETS["vx05_3s"].vx, PRESETS["vx05_3s"].motion_s) == (0.5, 3.0)
    assert (PRESETS["yaw05_3s"].wz, PRESETS["yaw05_3s"].motion_s) == (0.5, 3.0)
    assert (PRESETS["vxm03_2s"].vx, PRESETS["vxm03_2s"].motion_s) == (-0.3, 2.0)
    assert (LEAD_IN_S, TAIL_S) == (2.0, 3.0)
    assert ENVELOPE == (0.5, 0.3, 0.8)
    assert RAMP_LIMITS == (0.5, 0.5, 1.0)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_is_inside_envelope_and_ramp_limits_at_100hz(name: str) -> None:
    dur = profile_duration(name)
    n = int(round(dur * 100))
    prev = None
    peak = [0.0, 0.0, 0.0]
    for k in range(n + 301):  # sample 3 s past the end too
        t = k / 100.0
        vx, vy, wz, phase = command_at(name, t)
        v = (vx, vy, wz)
        for i, cap in enumerate(ENVELOPE):
            assert abs(v[i]) <= cap + 1e-12, (name, t, v)
            peak[i] = max(peak[i], abs(v[i]))
        if prev is not None:
            for i, lim in enumerate(RAMP_LIMITS):
                assert abs(v[i] - prev[i]) / 0.01 <= lim + 1e-9, (name, t, i, v, prev)
        prev = v
    p = PRESETS[name]
    for i, target in enumerate(p.target):
        assert peak[i] == pytest.approx(abs(target))  # the target is actually reached


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_profile_starts_and_ends_at_exact_zero(name: str) -> None:
    dur = profile_duration(name)
    assert dur == LEAD_IN_S + PRESETS[name].motion_s + TAIL_S
    for t in (0.0, dur - 1e-9, dur, dur + 5.0):
        vx, vy, wz, _ = command_at(name, t)
        assert (vx, vy, wz) == (0.0, 0.0, 0.0), (name, t)
        assert all(str(x) == "0.0" for x in (vx, vy, wz)), "no negative zero"
    # zero for the whole lead-in, and back at zero (ramp-down fits in the tail) before the end
    for k in range(int(LEAD_IN_S * 100) + 1):
        assert command_at(name, k / 100.0)[:3] == (0.0, 0.0, 0.0)
    ramp_end = LEAD_IN_S + PRESETS[name].motion_s + wc._ramp_s(PRESETS[name])
    assert ramp_end < dur
    assert command_at(name, ramp_end + 0.01)[:3] == (0.0, 0.0, 0.0)


def test_phase_labels_and_single_segment() -> None:
    assert command_at("vx03_2s", 1.0)[3] == "lead_in"
    assert command_at("vx03_2s", 3.0)[3] == "motion"
    assert command_at("vx03_2s", 5.5)[3] == "tail"
    assert command_at("vx03_2s", 7.0)[3] == "done"
    assert command_at("vx03_2s", 3.0)[:3] == (0.3, 0.0, 0.0)
    assert command_at("vxm03_2s", 3.0)[0] == -0.3
    assert command_at("yaw05_3s", 4.0)[2] == 0.5  # counter-clockwise (positive wz)


@pytest.mark.parametrize(
    "bad",
    [
        WalkPreset("fast", 0.6, 0.0, 0.0, 2.0),
        WalkPreset("slide", 0.0, 0.4, 0.0, 2.0),
        WalkPreset("spin", 0.0, 0.0, 0.9, 2.0),
        WalkPreset("negfast", -0.51, 0.0, 0.0, 2.0),
        WalkPreset("nomove", 0.0, 0.0, 0.0, 2.0),
        WalkPreset("nolen", 0.3, 0.0, 0.0, 0.0),
        WalkPreset("tooshort", 0.5, 0.0, 0.0, 0.5),  # 1.0 s ramp does not fit in 0.5 s
    ],
)
def test_presets_outside_the_envelope_are_a_value_error(bad: WalkPreset) -> None:
    with pytest.raises(ValueError):
        wc.validate_preset(bad)
    with pytest.raises(ValueError):
        WalkRunner(bad)


def test_import_validates_every_preset_and_needs_no_ros() -> None:
    code = (
        "import sys; import phoenix.sim2real.walk_command as w; "
        "assert w.PRESETS; "
        "bad = [m for m in sys.modules if m.split('.')[0] in ('rclpy', 'geometry_msgs', 'std_msgs')]; "
        "assert not bad, bad"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


# ------------------------------------------------------------- state machine


def _drive(runner: WalkRunner, t0: float, t1: float, estop):
    """Tick at 20 Hz over [t0, t1); estop(t) -> bool | None (None = no message), at 10 Hz."""
    out = []
    k = int(round(t0 * 20))
    while k * TICK_NS < t1 * S:
        now = k * TICK_NS
        if k % ESTOP_EVERY == 0:
            v = estop(now / S)
            if v is not None:
                runner.on_estop(v, now)
        out.append((now / S, runner.step(now)))
        k += 1
    return out


def _held(_t: float) -> bool:
    return False


def test_nothing_is_published_before_one_second_of_continuous_hold() -> None:
    r = WalkRunner("vx03_2s")
    ticks = _drive(r, 0.0, 0.95, _held)
    assert all(c is None for _, c in ticks)
    ticks = _drive(r, 0.95, 1.2, _held)
    first = next(t for t, c in ticks if c is not None)
    assert 0.95 <= first <= 1.1
    assert r.started_ns is not None and r.started_ns == int(round(first * S))


def test_estop_true_blip_restarts_the_hold() -> None:
    r = WalkRunner("vx03_2s")
    blip = lambda t: True if 0.8 <= t < 0.9 else False  # noqa: E731
    ticks = _drive(r, 0.0, 1.8, blip)
    assert all(c is None for _, c in ticks)
    ticks = _drive(r, 1.8, 2.1, blip)
    assert any(c is not None for _, c in ticks)


def test_no_estop_messages_means_nothing_published_and_a_gap_breaks_continuity() -> None:
    r = WalkRunner("vx03_2s")
    assert all(c is None for _, c in _drive(r, 0.0, 3.0, lambda t: None))
    r = WalkRunner("vx03_2s")
    gap = lambda t: None if 0.5 <= t < 1.2 else False  # noqa: E731
    ticks = _drive(r, 0.0, 3.5, gap)
    first = next(t for t, c in ticks if c is not None)
    assert first >= 1.2 + 1.0 - 0.1  # a fresh 1 s hold after the gap, not 1 s since t=0


def test_profile_is_followed_from_the_start_time_and_completes() -> None:
    r = WalkRunner("vx03_2s")
    ticks = _drive(r, 0.0, 20.0, _held)
    cmds = [(t, c) for t, c in ticks if c is not None]
    t_start = r.started_ns / S
    for t, c in cmds:
        if c.phase in ("lead_in", "motion", "tail"):
            assert (c.vx, c.vy, c.wz) == command_at("vx03_2s", t - t_start)[:3]
            assert c.estop_value is False and c.estop_age_s is not None and c.estop_age_s <= 0.1
    phases = [c.phase for _, c in cmds]
    assert phases[0] == "lead_in" and "motion" in phases and "tail" in phases
    final = [c for _, c in cmds if c.phase == "final_zeros"]
    assert 19 <= len(final) <= 21  # 1.0 s at 20 Hz
    assert all((c.vx, c.vy, c.wz) == (0.0, 0.0, 0.0) for c in final)
    assert r.done and r.outcome == "complete" and r.exit_code == 0
    # nothing after done
    assert r.step(30 * S) is None


@pytest.mark.parametrize("release_value,outcome", [(True, "deadman_released")])
def test_release_during_motion_goes_to_zeros_immediately(release_value, outcome) -> None:
    r = WalkRunner("vx05_3s")
    release_t = 1.0 + LEAD_IN_S + 1.5  # mid-ramp/hold, nonzero command in flight
    est = lambda t: release_value if t >= release_t else False  # noqa: E731
    ticks = _drive(r, 0.0, 12.0, est)
    pre = [c for t, c in ticks if c is not None and t < release_t]
    assert any(c.vx > 0.4 for c in pre)  # it really was commanding
    post = [(t, c) for t, c in ticks if c is not None and t >= release_t]
    t0, c0 = post[0]
    assert t0 - release_t <= 0.1  # first tick after the release is already zeros
    assert all((c.vx, c.vy, c.wz) == (0.0, 0.0, 0.0) for _, c in post)
    assert all(c.phase == "abort_zeros" for _, c in post)
    span = post[-1][0] - post[0][0]
    assert 0.9 <= span <= 1.05  # zeros kept for 1.0 s
    assert r.done and r.outcome == outcome and r.exit_code == 2


def test_stale_estop_goes_to_zeros_and_records_estop_stale() -> None:
    r = WalkRunner("vx03_4s")
    stale_t = 1.0 + LEAD_IN_S + 1.0
    est = lambda t: False if t < stale_t else None  # messages simply stop  # noqa: E731
    ticks = _drive(r, 0.0, 12.0, est)
    post = [(t, c) for t, c in ticks if c is not None and t > stale_t + 0.5]
    assert post and all(c.phase == "abort_zeros" for _, c in post)
    first_abort = next(t for t, c in ticks if c is not None and c.phase == "abort_zeros")
    last_msg_t = 3.9  # last 10 Hz message before the stop at stale_t
    assert 0.5 < first_abort - last_msg_t <= 0.6  # last message older than 0.5 s
    assert all((c.vx, c.vy, c.wz) == (0.0, 0.0, 0.0) for _, c in post)
    assert r.done and r.outcome == "estop_stale" and r.exit_code == 2


def test_release_during_the_final_zeros_does_not_change_the_outcome() -> None:
    r = WalkRunner("vxm03_2s")
    end = 1.0 + profile_duration("vxm03_2s")
    est = lambda t: True if t >= end + 0.2 else False  # noqa: E731
    _drive(r, 0.0, end + 3.0, est)
    assert r.done and r.outcome == "complete" and r.exit_code == 0


def test_interrupt_while_running_holds_zeros_0p5s_then_exits_3() -> None:
    r = WalkRunner("vx03_2s")
    _drive(r, 0.0, 4.0, _held)
    now = int(4.0 * S)
    r.interrupt(now)
    c = r.step(now)
    assert c is not None and (c.vx, c.vy, c.wz) == (0.0, 0.0, 0.0) and c.phase == "interrupt_zeros"
    assert r.step(now + int(0.4 * S)) is not None
    assert r.step(now + int(0.5 * S)) is None
    assert r.done and r.outcome == "interrupted" and r.exit_code == 3


def test_interrupt_before_start_publishes_nothing() -> None:
    r = WalkRunner("vx03_2s")
    _drive(r, 0.0, 0.5, _held)
    r.interrupt(int(0.5 * S))
    assert r.step(int(0.55 * S)) is None
    assert r.done and r.outcome == "interrupted" and r.exit_code == 3


def test_wallclock_cap_is_duration_plus_five_seconds() -> None:
    r = WalkRunner("vx03_2s", start_deadline_s=30.0)
    ticks = _drive(r, 0.0, 30.0, lambda t: None)  # deadman never held, deadline out of the way
    assert all(c is None for _, c in ticks)
    assert r.done and r.outcome == "wallclock_cap" and r.exit_code == 4
    assert wc.WALLCLOCK_MARGIN_S == 5.0
    # cap while running: held from the start, cap = 7 + 5 = 12 s after node start
    r = WalkRunner("vx03_2s", wallclock_s=4.0)
    ticks = _drive(r, 0.0, 10.0, _held)
    assert r.done and r.outcome == "wallclock_cap"
    assert any(c is not None and c.phase == "abort_zeros" for _, c in ticks)


# ------------------------------------------------------------------- recorder


def test_recorder_writes_commands_and_a_summary_and_never_overwrites(tmp_path) -> None:
    path = tmp_path / "walk_cmd.jsonl"
    r = WalkRunner("vx03_2s")
    rec = wc.Recorder(path, "vx03_2s")
    for t, c in _drive(r, 0.0, 12.0, _held):
        if c is not None:
            rec.record(int(t * S), c)
    rec.summary(r.outcome, r.started_ns)
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    cmds, summary = lines[:-1], lines[-1]
    assert set(cmds[0]) == {
        "t_mono_ns",
        "phase",
        "vx",
        "vy",
        "wz",
        "estop_value",
        "estop_age_s",
    }
    assert summary["record"] == "summary" and summary["preset"] == "vx03_2s"
    assert summary["outcome"] == "complete" and summary["n_published"] == len(cmds)
    assert summary["started_t_mono_ns"] == r.started_ns
    assert summary["max_abs"] == [0.3, 0.0, 0.0]
    with pytest.raises(FileExistsError):
        wc.Recorder(path, "vx03_2s")


def test_cli_list_and_argument_errors(capsys) -> None:
    assert wc.main(["--list"]) == 0
    out = capsys.readouterr().out
    for name in PRESETS:
        assert name in out
    with pytest.raises(SystemExit):
        wc.parse_args(["--preset", "vx03_2s"])  # --out missing
    with pytest.raises(SystemExit):
        wc.parse_args(["--preset", "vx09_9s", "--out", "x"])  # unknown preset


# ---------------------------------------------------------------- W harness


def _function_body(text: str, name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}\n", text, re.S | re.M)
    assert m, name
    return m.group(1)


def _run(args, tmp_path, **env_over):
    env = os.environ.copy()
    env.update(PHOENIX_SESSION=str(tmp_path / "session"), PYTHONDONTWRITEBYTECODE="1")
    for k in ("PHOENIX_EXPECT_SHA", "PHOENIX_REHEARSAL"):
        env.pop(k, None)
    env.update(env_over)
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )


def test_w_refuses_without_a_preset_or_with_an_unknown_one(tmp_path) -> None:
    res = _run(["bash", str(HARNESS), "W"], tmp_path)
    assert res.returncode != 0 and "usage" in res.stderr and "launched" not in res.stdout
    res = _run(["bash", str(HARNESS), "W", "vx09_9s"], tmp_path)
    assert res.returncode != 0 and "unknown walk preset" in res.stderr
    assert "launched" not in res.stdout


def test_w_requires_the_expected_commit_and_launches_nothing(tmp_path) -> None:
    res = _run(["bash", str(HARNESS), "W", "vx03_2s"], tmp_path)
    assert res.returncode != 0
    assert "PHOENIX_EXPECT_SHA is required" in res.stderr
    assert "launched" not in res.stdout


def test_w_is_wired_like_stage_stand_and_is_never_a_ledger_stage() -> None:
    text = HARNESS.read_text()
    body = _function_body(text, "stage_walk")
    # the same safety chain as stage_stand
    for needle in (
        "need_sha",
        "ros_env",
        "assert_no_contention W",
        'live_confirm "W $preset ON GANTRY"',
        "assert_battery W",
        'new_run "W_${preset}"',
        "operator_gate arm",
        "operator_gate judgement",
        "operator_gate release",
        "--standup-s",
        "--authority-s",
        "--max-runtime-s",
        'wait_for_telemetry "$telemetry" \'t.get("standup_done")\'',
        "walk_command --preset",
        '--out "$RUN/walk_cmd.jsonl"',
        "safety.stand_only",
    ):
        assert needle in body, needle
    assert body.index("launch deadman") < body.index("launch lowcmd_bridge")
    assert body.index("launch lowcmd_bridge") < body.index("launch policy_node")
    assert body.index("launch policy_node") < body.index("launch walk_cmd")
    assert body.index("standup_done") < body.index("launch policy_node")
    # not a ledger stage: no evaluator, no ledger require
    assert "pf evaluate" not in body and "pf require" not in body
    assert not re.search(r"^\s*F\|G\|H\|W|F\|G\|H\|W\)", text, re.M)
    assert "    W) stage_walk" in text
    assert "launch walk_cmd" in body and "walk_cmd" in _function_body(text, "stop_all")
    # the help text lists W
    helped = subprocess.run(
        ["bash", str(HARNESS), "--help"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert re.search(r"harness_preflight\.sh W <preset>", helped.stdout)
    assert "PHOENIX_HW_SESSIONS_TEST_ONLY=1" in helped.stdout


def test_stage_labels_other_than_w_are_untouched() -> None:
    text = HARNESS.read_text()
    assert '    F|G|H) stage_stand "$cmd" ;;' in text
    body = _function_body(text, "stage_stand")
    assert "walk" not in body.lower()


# -------------------------------------------------------------- start deadline


def test_held_at_half_a_second_starts_well_before_the_deadline() -> None:
    r = WalkRunner("vx03_2s")
    ticks = _drive(r, 0.0, 12.0, lambda t: None if t < 0.5 else False)
    assert r.started_ns is not None and 1.4 <= r.started_ns / S <= 1.7
    assert r.outcome == "complete"
    assert any(c is not None and c.phase == "motion" for _, c in ticks)


def test_first_held_at_2p5s_is_late_start_and_never_publishes() -> None:
    r = WalkRunner("vx03_2s")
    ticks = _drive(r, 0.0, 10.0, lambda t: None if t < 2.5 else False)
    assert all(c is None for _, c in ticks)  # not even zeros
    assert r.started_ns is None
    assert r.done and r.outcome == "late_start" and r.exit_code == 5
    assert wc.EXIT_CODES["late_start"] == 5


def test_deadline_boundary_satisfied_at_exactly_3s_starts_after_3s_does_not() -> None:
    on_time = WalkRunner("vx03_2s")
    _drive(on_time, 0.0, 12.0, lambda t: None if t < 2.0 else False)  # held since 2.0 -> 3.0
    assert on_time.started_ns == 3 * S and on_time.outcome == "complete"
    late = WalkRunner("vx03_2s")
    ticks = _drive(late, 0.0, 12.0, lambda t: None if t < 2.1 else False)  # satisfied 3.1
    assert all(c is None for _, c in ticks) and late.outcome == "late_start"
    assert late.started_ns is None


def test_deadline_is_measured_from_the_first_tick_not_construction() -> None:
    r = WalkRunner("vx03_2s")
    base = 5_000_000 * S  # a large monotonic clock, first tick long after construction
    for k in range(0, 400):
        now = base + k * TICK_NS
        if k % ESTOP_EVERY == 0:
            r.on_estop(False, now)
        assert r.step(now) is None or r.started_ns is not None
    assert r.started_ns is not None and r.started_ns - base == S


def test_a_deadline_shorter_than_the_hold_is_rejected(capsys) -> None:
    with pytest.raises(ValueError):
        WalkRunner("vx03_2s", start_deadline_s=0.5)
    with pytest.raises(SystemExit):
        wc.parse_args(["--preset", "vx03_2s", "--out", "x", "--start-deadline-s", "0.5"])
    args = wc.parse_args(["--preset", "vx03_2s", "--out", "x"])
    assert args.start_deadline_s == 3.0


def test_motion_never_outlives_the_policy_window_for_any_preset() -> None:
    text = HARNESS.read_text()
    body = _function_body(text, "stage_walk")
    margin = float(re.search(r"PHOENIX_WALK_AUTHORITY_MARGIN_S:-([0-9.]+)", body).group(1))
    assert "--start-deadline-s 3.0" in body
    lag = 1.0  # walk_cmd launched right after the policy node
    for name, p in PRESETS.items():
        window_end = profile_duration(name) + margin
        latest_start = lag + wc.START_DEADLINE_S
        motion_end = latest_start + LEAD_IN_S + p.motion_s + wc._ramp_s(p)
        assert window_end - motion_end >= 2.0 - 1e-9, (name, window_end, motion_end)
