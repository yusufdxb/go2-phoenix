"""Tests for scripts/go2_release_sport_mode.py: no robot, no unitree_sdk2py
network calls -- everything here uses fake MotionSwitcher / SportClient
doubles that implement the same (CheckMode/SelectMode/ReleaseMode/StandDown)
surface verified against unitree_sdk2py 1.0.1 on this workstation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "go2_release_sport_mode.py"

spec = importlib.util.spec_from_file_location("go2_release_sport_mode", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["go2_release_sport_mode"] = mod
spec.loader.exec_module(mod)  # type: ignore[union-attr]


class _FakeSwitcher:
    """CheckMode returns each name in ``sequence`` in turn (last one repeats)."""

    def __init__(self, sequence: list[str]) -> None:
        self.sequence = list(sequence)
        self.checks = 0
        self.select_mode_calls: list[str] = []
        self.release_mode_calls = 0

    def CheckMode(self):
        idx = min(self.checks, len(self.sequence) - 1)
        self.checks += 1
        return 0, {"name": self.sequence[idx]}

    def SelectMode(self, name: str):
        self.select_mode_calls.append(name)
        return 0, None

    def ReleaseMode(self):
        self.release_mode_calls += 1
        return 0, None


class _FakeSport:
    def __init__(self) -> None:
        self.stand_down_calls = 0

    def StandDown(self) -> int:
        self.stand_down_calls += 1
        return 0


# --------------------------------------------------------- select_mode gate
@pytest.mark.parametrize("bad_name", ["normal", "Normal", "NORMAL", "", "walk", "idle", "damp"])
def test_select_mode_hard_rejects_every_name_but_mcf_and_ai(bad_name: str) -> None:
    switcher = _FakeSwitcher(["mcf"])
    with pytest.raises(mod.ForbiddenModeNameError):
        mod.select_mode(switcher, bad_name, log=[])
    # Never even reached the client.
    assert switcher.select_mode_calls == []


@pytest.mark.parametrize("good_name", ["mcf", "ai"])
def test_select_mode_allows_only_mcf_and_ai(good_name: str) -> None:
    switcher = _FakeSwitcher(["mcf"])
    mod.select_mode(switcher, good_name, log=[])
    assert switcher.select_mode_calls == [good_name]


def test_allowed_select_mode_names_is_exactly_mcf_and_ai() -> None:
    assert mod.ALLOWED_SELECT_MODE_NAMES == frozenset({"mcf", "ai"})


# ------------------------------------------------------------- release loop
def test_release_sequence_stops_when_mode_becomes_empty() -> None:
    switcher = _FakeSwitcher(["mcf", "mcf", ""])
    sport = _FakeSport()
    result = mod.release_sequence(switcher, sport, sleep_fn=lambda s: None)
    assert result.released is True
    assert result.final_mode_name is None
    assert result.attempts == 2
    assert sport.stand_down_calls == 2
    assert switcher.release_mode_calls == 2
    # Never calls SelectMode during a release.
    assert switcher.select_mode_calls == []


def test_release_sequence_gives_up_after_max_attempts() -> None:
    switcher = _FakeSwitcher(["mcf"])  # never becomes empty
    sport = _FakeSport()
    result = mod.release_sequence(
        switcher, sport, max_attempts=3, sleep_fn=lambda s: None
    )
    assert result.released is False
    assert result.final_mode_name == "mcf"
    assert result.attempts == 3


def test_release_sequence_already_empty_makes_no_stand_down_calls() -> None:
    switcher = _FakeSwitcher([""])
    sport = _FakeSport()
    result = mod.release_sequence(switcher, sport, sleep_fn=lambda s: None)
    assert result.released is True
    assert result.attempts == 0
    assert sport.stand_down_calls == 0


# ------------------------------------------------------------------ restore
def test_restore_sequence_calls_select_mode_mcf_only() -> None:
    switcher = _FakeSwitcher(["mcf"])
    log = mod.restore_sequence(switcher)
    assert switcher.select_mode_calls == ["mcf"]
    assert any("SelectMode" in line for line in log)
    # Restore never touches ReleaseMode or CheckMode.
    assert switcher.release_mode_calls == 0
    assert switcher.checks == 0


def test_restore_sequence_rejects_bad_mode_even_if_caller_bypasses_argparse() -> None:
    switcher = _FakeSwitcher(["mcf"])
    with pytest.raises(mod.ForbiddenModeNameError):
        mod.restore_sequence(switcher, mode="normal")


# --------------------------------------------------------------------- CLI
def test_main_defaults_to_dry_run_and_exits_zero(capsys) -> None:
    code = mod.main([])
    assert code == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out


def test_main_live_without_i_am_at_the_robot_is_refused() -> None:
    assert mod.main(["--live"]) == 2


def test_main_i_am_at_the_robot_without_live_is_refused() -> None:
    assert mod.main(["--i-am-at-the-robot"]) == 2


def test_main_restore_dry_run_prints_select_mode_mcf(capsys) -> None:
    code = mod.main(["--restore"])
    assert code == 0
    out = capsys.readouterr().out
    assert "SelectMode('mcf')" in out


# ---------------------------------------- posture gate (Damp-free policy, item 8)
import math as _math  # noqa: E402

_STAND_Q = [0.0, 0.67, -1.3] * 4
_FOLDED_Q = [float(v) for v in mod.SIT_POSE_UNITREE]
_LEVEL = [1.0, 0.0, 0.0, 0.0]
_ON_SIDE = [_math.cos(_math.radians(40)), _math.sin(_math.radians(40)), 0.0, 0.0]  # 80 deg tilt


def _no_confirm(_prompt: str) -> str:
    raise AssertionError("must not ask for the gantry phrase when the robot is lying")


def _live(monkeypatch, posture, confirm_fn):
    switcher, sport = _FakeSwitcher(["mcf", ""]), _FakeSport()
    built = []
    monkeypatch.setattr(mod, "_build_real_clients", lambda: built.append(1) or (switcher, sport))
    rc = mod.main(
        ["--live", "--i-am-at-the-robot", "--poll-interval-s", "0"],
        read_posture=lambda: posture,
        confirm_fn=confirm_fn,
    )
    return rc, switcher, built


def test_release_refuses_when_standing(monkeypatch) -> None:
    rc, switcher, built = _live(monkeypatch, (_STAND_Q, _LEVEL), lambda _p: "")
    assert rc == 3 and built == [] and switcher.release_mode_calls == 0


def test_release_refuses_standing_with_a_wrong_gantry_phrase(monkeypatch) -> None:
    rc, switcher, built = _live(monkeypatch, (_STAND_Q, _LEVEL), lambda _p: "yes")
    assert rc == 3 and built == [] and switcher.release_mode_calls == 0


def test_release_refuses_when_posture_unreadable_and_unconfirmed(monkeypatch) -> None:
    rc, switcher, built = _live(monkeypatch, None, lambda _p: "")
    assert rc == 3 and built == [] and switcher.release_mode_calls == 0


def test_release_allowed_when_lying_folded(monkeypatch) -> None:
    rc, switcher, _ = _live(monkeypatch, (_FOLDED_Q, _LEVEL), _no_confirm)
    assert rc == 0 and switcher.release_mode_calls == 1


def test_release_allowed_when_on_its_side(monkeypatch) -> None:
    rc, switcher, _ = _live(monkeypatch, (_STAND_Q, _ON_SIDE), _no_confirm)
    assert rc == 0 and switcher.release_mode_calls == 1


def test_release_allowed_standing_only_with_the_exact_gantry_phrase(monkeypatch) -> None:
    rc, switcher, _ = _live(monkeypatch, (_STAND_Q, _LEVEL), lambda _p: mod.GANTRY_CONFIRM_PHRASE)
    assert rc == 0 and switcher.release_mode_calls == 1


def test_restore_needs_no_posture_gate(monkeypatch) -> None:
    switcher = _FakeSwitcher(["mcf"])
    monkeypatch.setattr(mod, "_build_real_clients", lambda: (switcher, _FakeSport()))
    rc = mod.main(["--live", "--i-am-at-the-robot", "--restore"], read_posture=lambda: None)
    assert rc == 0 and switcher.select_mode_calls == ["mcf"]
