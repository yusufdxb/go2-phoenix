"""Walking configs feed a zero command when /cmd_vel is missing or stale."""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real.ros2_policy_node import (
    DEFAULT_CMD_VEL_TIMEOUT_S,
    effective_velocity_command,
    parse_cmd_vel_timeout_s,
)

STORED = np.array([0.4, 0.1, -0.2], dtype=np.float32)
NOW = 10_000_000_000


def test_walking_fresh_command_is_fed() -> None:
    cmd, stale = effective_velocity_command(STORED, NOW - 100_000_000, NOW, 0.5, False)
    assert not stale
    np.testing.assert_array_equal(cmd, STORED)


def test_walking_stale_command_is_zero() -> None:
    cmd, stale = effective_velocity_command(STORED, NOW - 600_000_000, NOW, 0.5, False)
    assert stale
    np.testing.assert_array_equal(cmd, np.zeros(3, dtype=np.float32))


def test_walking_never_received_is_zero() -> None:
    cmd, stale = effective_velocity_command(STORED, None, NOW, 0.5, False)
    assert stale
    np.testing.assert_array_equal(cmd, np.zeros(3, dtype=np.float32))


def test_stand_only_unchanged() -> None:
    for cmd_ns in (None, NOW, NOW - 10_000_000_000):
        cmd, stale = effective_velocity_command(STORED, cmd_ns, NOW, 0.5, True)
        assert not stale
        np.testing.assert_array_equal(cmd, np.zeros(3, dtype=np.float32))


def test_timeout_default_and_override() -> None:
    assert parse_cmd_vel_timeout_s({}) == DEFAULT_CMD_VEL_TIMEOUT_S == 0.5
    assert parse_cmd_vel_timeout_s({"safety": {"cmd_vel_timeout_s": 0.25}}) == 0.25


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf")])
def test_invalid_timeout_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="cmd_vel_timeout_s"):
        parse_cmd_vel_timeout_s({"safety": {"cmd_vel_timeout_s": bad}})
