"""Tests for the fail-closed safety predicates shared by the deploy
nodes (policy node + lowcmd bridge + deadman/wireless adapters)."""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real.safety import (
    MAX_DELTA_PER_STEP_RAD,
    deadman_should_estop,
    estop_is_active,
    is_ready_to_command_motion,
    per_step_clip,
    per_step_clip_array,
    sensor_is_stale,
)

# 1 ms in nanoseconds. Tests use a synthetic clock so behaviour is exact.
MS = 1_000_000


# ---------------- estop_is_active ------------------------------------------


def test_estop_is_active_when_no_message_seen() -> None:
    # Fail-closed: never received any heartbeat ⇒ treat as estopped.
    assert estop_is_active(
        last_msg_received_ns=None,
        latest_value=None,
        now_ns=1000,
        timeout_s=0.5,
    ) is True


def test_estop_inactive_when_fresh_false() -> None:
    now = 100 * MS
    assert estop_is_active(
        last_msg_received_ns=now - 50 * MS,
        latest_value=False,
        now_ns=now,
        timeout_s=0.5,
    ) is False


def test_estop_active_when_fresh_true() -> None:
    now = 100 * MS
    assert estop_is_active(
        last_msg_received_ns=now - 50 * MS,
        latest_value=True,
        now_ns=now,
        timeout_s=0.5,
    ) is True


def test_estop_active_when_stale_even_if_last_value_was_false() -> None:
    # Publisher died after broadcasting False; we MUST treat as estopped.
    now = 1_000 * MS
    assert estop_is_active(
        last_msg_received_ns=now - 600 * MS,
        latest_value=False,
        now_ns=now,
        timeout_s=0.5,
    ) is True


def test_estop_just_inside_timeout_is_inactive() -> None:
    # 0.5 s timeout, message 0.49 s old ⇒ still trusted.
    now = 1_000 * MS
    assert estop_is_active(
        last_msg_received_ns=now - 490 * MS,
        latest_value=False,
        now_ns=now,
        timeout_s=0.5,
    ) is False


# ---------------- sensor_is_stale ------------------------------------------


def test_sensor_stale_when_never_seen() -> None:
    assert sensor_is_stale(last_msg_received_ns=None, now_ns=10 * MS, timeout_s=0.2) is True


def test_sensor_stale_when_too_old() -> None:
    now = 500 * MS
    assert sensor_is_stale(
        last_msg_received_ns=now - 250 * MS, now_ns=now, timeout_s=0.2
    ) is True


def test_sensor_fresh_inside_window() -> None:
    now = 500 * MS
    assert sensor_is_stale(
        last_msg_received_ns=now - 100 * MS, now_ns=now, timeout_s=0.2
    ) is False


# ---------------- deadman_should_estop -------------------------------------


def test_deadman_estopped_when_input_never_seen() -> None:
    assert deadman_should_estop(
        last_input_ns=None, button_held=True, now_ns=100 * MS, timeout_s=0.5
    ) is True


def test_deadman_estopped_when_button_released() -> None:
    now = 200 * MS
    assert deadman_should_estop(
        last_input_ns=now - 10 * MS, button_held=False, now_ns=now, timeout_s=0.5
    ) is True


def test_deadman_estopped_when_input_stale_even_with_button_held() -> None:
    # Most realistic failure mode: gamepad disconnect — last reported state
    # was "held" but no fresh inputs are arriving.
    now = 1_000 * MS
    assert deadman_should_estop(
        last_input_ns=now - 700 * MS, button_held=True, now_ns=now, timeout_s=0.5
    ) is True


def test_deadman_clear_when_held_and_fresh() -> None:
    now = 100 * MS
    assert deadman_should_estop(
        last_input_ns=now - 10 * MS, button_held=True, now_ns=now, timeout_s=0.5
    ) is False


# ---------------- per_step_clip --------------------------------------------


@pytest.mark.parametrize(
    "target,current,max_delta,expected",
    [
        (0.5, 0.0, 0.1, 0.1),    # clipped up
        (-0.5, 0.0, 0.1, -0.1),  # clipped down
        (0.05, 0.0, 0.1, 0.05),  # untouched
        (0.0, 0.5, 0.1, 0.4),    # clipped up toward target=0
    ],
)
def test_per_step_clip_bounds(target, current, max_delta, expected) -> None:
    assert per_step_clip(target, current, max_delta) == pytest.approx(expected)


# ---------------- per_step_clip_array (shared by policy + bridge) ----------


def test_per_step_clip_array_default_max_delta_matches_constant() -> None:
    # The default arg is the shared MAX_DELTA_PER_STEP_RAD; both call sites
    # rely on this being 0.175 rad/step.
    assert MAX_DELTA_PER_STEP_RAD == pytest.approx(0.175)
    target = np.array([1.0, -1.0, 0.05, -0.05])
    current = np.zeros(4)
    out = per_step_clip_array(target, current)
    np.testing.assert_allclose(out, [0.175, -0.175, 0.05, -0.05])


def test_per_step_clip_array_clips_relative_to_current() -> None:
    target = np.array([0.5, 0.5, 0.5, 0.5])
    current = np.array([0.0, 0.4, 0.45, 1.0])
    out = per_step_clip_array(target, current, max_delta=0.1)
    np.testing.assert_allclose(out, [0.1, 0.5, 0.5, 0.9])


def test_per_step_clip_array_returns_ndarray_for_lists() -> None:
    out = per_step_clip_array([0.5, -0.5], [0.0, 0.0], max_delta=0.1)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [0.1, -0.1])


# ---------------- is_ready_to_command_motion (startup fail-closed) ---------

# Common kwargs to keep tests compact.
_FRESH = {"estop_timeout_s": 0.5, "sensor_timeout_s": 0.2}


def _kw(now_ns: int, **overrides):
    base = {
        "now_ns": now_ns,
        "estop_last_ns": now_ns - 50 * MS,
        "estop_value": False,
        "imu_last_ns": now_ns - 30 * MS,
        "joint_state_last_ns": now_ns - 30 * MS,
        **_FRESH,
    }
    base.update(overrides)
    return base


def test_ready_when_everything_fresh() -> None:
    ok, reason = is_ready_to_command_motion(**_kw(now_ns=1_000 * MS))
    assert (ok, reason) == (True, None)


def test_not_ready_when_estop_never_received() -> None:
    ok, reason = is_ready_to_command_motion(
        **_kw(now_ns=1_000 * MS, estop_last_ns=None, estop_value=None)
    )
    assert ok is False
    assert reason == "estop_publisher_missing"


def test_not_ready_when_estop_received_but_value_none() -> None:
    # Defensive: if subscriber callback hasn't populated value yet.
    ok, reason = is_ready_to_command_motion(
        **_kw(now_ns=1_000 * MS, estop_value=None)
    )
    assert ok is False
    assert reason == "estop_publisher_missing"


def test_not_ready_when_estop_asserted_true() -> None:
    ok, reason = is_ready_to_command_motion(
        **_kw(now_ns=1_000 * MS, estop_value=True)
    )
    assert ok is False
    assert reason == "external_estop"


def test_not_ready_when_estop_heartbeat_stale_even_if_last_value_false() -> None:
    # The dangerous case: publisher died after broadcasting safe-to-run.
    now = 2_000 * MS
    ok, reason = is_ready_to_command_motion(
        **_kw(now_ns=now, estop_last_ns=now - 700 * MS)
    )
    assert ok is False
    assert reason == "estop_heartbeat_stale"


def test_not_ready_when_imu_never_seen() -> None:
    ok, reason = is_ready_to_command_motion(
        **_kw(now_ns=1_000 * MS, imu_last_ns=None)
    )
    assert ok is False
    assert reason == "sensor_missing"


def test_not_ready_when_joint_state_never_seen() -> None:
    ok, reason = is_ready_to_command_motion(
        **_kw(now_ns=1_000 * MS, joint_state_last_ns=None)
    )
    assert ok is False
    assert reason == "sensor_missing"


def test_not_ready_when_imu_stale() -> None:
    now = 1_000 * MS
    ok, reason = is_ready_to_command_motion(
        **_kw(now_ns=now, imu_last_ns=now - 300 * MS)
    )
    assert ok is False
    assert reason == "sensor_stale"


def test_not_ready_when_joint_state_stale() -> None:
    now = 1_000 * MS
    ok, reason = is_ready_to_command_motion(
        **_kw(now_ns=now, joint_state_last_ns=now - 300 * MS)
    )
    assert ok is False
    assert reason == "sensor_stale"


def test_estop_check_is_evaluated_before_sensor_check() -> None:
    # If both estop is missing AND sensors are stale, the estop reason
    # must surface (so operators see the most actionable cause first).
    now = 1_000 * MS
    ok, reason = is_ready_to_command_motion(
        **_kw(
            now_ns=now,
            estop_last_ns=None,
            estop_value=None,
            imu_last_ns=now - 500 * MS,
        )
    )
    assert ok is False
    assert reason == "estop_publisher_missing"


def test_startup_predicate_blocks_until_estop_arrives() -> None:
    # Reproduces the bug fixed in this commit: previously the policy
    # node treated the startup grace window as "free pass to publish
    # motion if IMU+joints arrived." This predicate now refuses motion
    # commands no matter what the timestamp is, until estop is heard.
    now = 100 * MS
    for elapsed_ms in (10, 100, 1_000, 10_000):
        ok, reason = is_ready_to_command_motion(
            **_kw(
                now_ns=now + elapsed_ms * MS,
                estop_last_ns=None,
                estop_value=None,
            )
        )
        assert ok is False
        assert reason == "estop_publisher_missing"


# ---------------- startup_state (first-message gate) -----------------------


from phoenix.sim2real.safety import startup_state


def _startup_kw(
    *,
    now_ns: int,
    node_started_ns: int = 0,
    seen_estop: bool = False,
    seen_imu: bool = False,
    seen_joint_state: bool = False,
    first_message_timeout_s: float = 15.0,
) -> dict:
    return {
        "seen_estop": seen_estop,
        "seen_imu": seen_imu,
        "seen_joint_state": seen_joint_state,
        "node_started_ns": node_started_ns,
        "now_ns": now_ns,
        "first_message_timeout_s": first_message_timeout_s,
    }


def test_startup_waiting_when_none_seen_within_timeout() -> None:
    state, reason = startup_state(**_startup_kw(now_ns=1_000 * MS))
    assert (state, reason) == ("waiting", None)


def test_startup_waiting_when_two_of_three_seen() -> None:
    state, reason = startup_state(
        **_startup_kw(now_ns=1_000 * MS, seen_estop=True, seen_imu=True)
    )
    assert (state, reason) == ("waiting", None)


def test_startup_ready_when_all_three_seen() -> None:
    state, reason = startup_state(
        **_startup_kw(
            now_ns=1_000 * MS,
            seen_estop=True,
            seen_imu=True,
            seen_joint_state=True,
        )
    )
    assert (state, reason) == ("ready", None)


def test_startup_aborts_after_timeout_with_missing_topic() -> None:
    # 20s elapsed > 15s timeout, estop + imu seen, joint_state missing.
    state, reason = startup_state(
        **_startup_kw(
            now_ns=20_000 * MS,
            seen_estop=True,
            seen_imu=True,
            seen_joint_state=False,
        )
    )
    assert state == "abort"
    assert reason == "first_message_timeout_joint_state"


def test_startup_abort_names_all_missing_topics() -> None:
    state, reason = startup_state(
        **_startup_kw(
            now_ns=20_000 * MS,
            seen_estop=False,
            seen_imu=False,
            seen_joint_state=False,
        )
    )
    assert state == "abort"
    # The reason must identify which topics missed. Exact wording: the
    # names are comma-joined in a stable order (estop, imu, joint_state).
    assert reason == "first_message_timeout_estop,imu,joint_state"


def test_startup_ready_before_timeout_even_if_slow() -> None:
    # 14s elapsed < 15s timeout, all seen — ready, not abort.
    state, reason = startup_state(
        **_startup_kw(
            now_ns=14_000 * MS,
            seen_estop=True,
            seen_imu=True,
            seen_joint_state=True,
        )
    )
    assert (state, reason) == ("ready", None)
