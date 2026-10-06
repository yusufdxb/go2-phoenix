"""Battery floor for live stages: above 20 % (operator rule 2026-10-06), fails closed."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from phoenix.sim2real.hw_probe import DEFAULT_MIN_BATTERY_SOC, battery_verdict

HARNESS = Path(__file__).resolve().parents[1] / "scripts" / "harness_preflight.sh"


def test_default_floor_is_twenty_percent() -> None:
    assert DEFAULT_MIN_BATTERY_SOC == 20.0


@pytest.mark.parametrize(
    ("samples", "ok"),
    [
        ([21], True),
        ([80, 79, 81], True),
        ([20], False),  # must be ABOVE the floor
        ([19], False),
        ([80, 18, 80], False),  # one low sample fails
        ([0], False),  # missing BMS data reads 0
        ([], False),
        ([math.nan], False),
    ],
)
def test_battery_verdict(samples: list[float], ok: bool) -> None:
    assert battery_verdict(samples, 20.0)[0] is ok


def test_every_motors_live_stage_checks_the_battery() -> None:
    text = HARNESS.read_text()
    for fn, tag in (("stage_E", "E"), ("stage_stand", '"$stage"'), ("stage_walk", "W")):
        body = re.search(rf"^{fn}\(\) {{\n(.*?)^}}", text, re.S | re.M)
        assert body is not None, fn
        assert f"assert_battery {tag}" in body.group(1), fn
