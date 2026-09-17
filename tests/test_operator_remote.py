"""Operator remote gestures: the safety contract, not the ROS plumbing.

``wait_for_remote`` is the only part that touches ROS and it is a thin spin
loop; everything that can turn a button press into a GO decision lives in
:class:`RemoteGesture`, which is pure. These tests pin the properties a
single-operator motors-live gate depends on:

* a HELD button can never confirm (rising edge only),
* releasing the deadman is HALT in every mode that can produce a judgement,
* an unanswered prompt is ``no_answer`` with a nonzero exit code,
* pressing both judgement buttons is ``ambiguous``, never a silent yes.
"""

from __future__ import annotations

import json

import pytest

from phoenix.sim2real.operator_remote import (
    A_MASK,
    B_MASK,
    DEADMAN_MASK,
    MODES,
    START_MASK,
    RemoteGesture,
    build_parser,
    main,
)


def drive(mode: str, frames: list[int]) -> str | None:
    """Feed a key sequence through one recognizer, first verdict wins."""
    gesture = RemoteGesture(mode)
    for keys in frames:
        verdict = gesture.observe(keys)
        if verdict is not None:
            return verdict
    return None


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="unknown remote gesture mode"):
        RemoteGesture("go")


def test_every_mode_constructs():
    for mode in MODES:
        assert RemoteGesture(mode).mode == mode


# ---------------------------------------------------------------- arm mode
def test_arm_requires_a_rising_edge_on_start():
    # Deadman and Start already held when the recognizer starts listening.
    # A held button must not arm the stage; only a fresh press counts.
    held = DEADMAN_MASK | START_MASK
    assert drive("arm", [held, held, held]) is None


def test_arm_confirms_on_fresh_press_after_neutral():
    frames = [DEADMAN_MASK, DEADMAN_MASK, DEADMAN_MASK | START_MASK]
    assert drive("arm", frames) == "armed"


def test_arm_without_deadman_never_arms():
    assert drive("arm", [START_MASK, START_MASK, 0]) is None


def test_arm_neutral_resets_when_deadman_drops():
    # Neutral seen while holding, then the deadman is released: the operator
    # let go, so the next Start press starts from an unprimed recognizer.
    frames = [DEADMAN_MASK, 0, DEADMAN_MASK | START_MASK]
    assert drive("arm", frames) is None


def test_arm_ignores_judgement_buttons():
    frames = [DEADMAN_MASK, DEADMAN_MASK | A_MASK, DEADMAN_MASK | B_MASK]
    assert drive("arm", frames) is None


# ---------------------------------------------------- judgement mode: yes/no
def test_judgement_yes_needs_a_rising_edge():
    held = DEADMAN_MASK | A_MASK
    assert drive("judgement", [held, held]) is None


def test_judgement_yes_after_neutral():
    assert drive("judgement", [DEADMAN_MASK, DEADMAN_MASK | A_MASK]) == "yes"


def test_judgement_no_after_neutral():
    assert drive("judgement", [DEADMAN_MASK, DEADMAN_MASK | B_MASK]) == "no"


def test_judgement_both_buttons_is_ambiguous_not_yes():
    frames = [DEADMAN_MASK, DEADMAN_MASK | A_MASK | B_MASK]
    assert drive("judgement", frames) == "ambiguous"


def test_judgement_deadman_release_is_halt():
    assert drive("judgement", [DEADMAN_MASK, 0]) == "halt"


def test_judgement_halt_wins_over_a_simultaneous_yes():
    # Deadman gone in the same frame as A: the release is the operator's
    # instruction and must not be read as a confirmation.
    assert drive("judgement", [DEADMAN_MASK, A_MASK]) == "halt"


def test_judgement_halts_immediately_without_deadman():
    assert drive("judgement", [0]) == "halt"


# ------------------------------------------------------------- release mode
def test_release_reports_only_once_the_deadman_is_gone():
    assert drive("release", [DEADMAN_MASK, DEADMAN_MASK, 0]) == "released"


def test_release_while_held_is_silent():
    assert drive("release", [DEADMAN_MASK | A_MASK, DEADMAN_MASK]) is None


# ------------------------------------------------------------ input hygiene
def test_observe_accepts_an_integer_like_keys_field():
    # unitree_go/msg/WirelessController.keys arrives as a numpy-backed
    # integer; observe must not depend on it being a python int.
    gesture = RemoteGesture("judgement")
    gesture.observe(float(DEADMAN_MASK))
    assert gesture.observe(float(DEADMAN_MASK | A_MASK)) == "yes"


def test_unrelated_bits_do_not_confirm():
    noise = 0x4000
    frames = [DEADMAN_MASK, DEADMAN_MASK | noise]
    assert drive("judgement", frames) is None


# -------------------------------------------------------------- CLI contract
def test_parser_rejects_an_unknown_mode():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["sprint", "--out", "x.json"])


def test_parser_requires_an_output_path():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["judgement"])


def _stub_wait(monkeypatch, record):
    monkeypatch.setattr(
        "phoenix.sim2real.operator_remote.wait_for_remote",
        lambda mode, timeout_s: dict(record, mode=mode),
    )


@pytest.mark.parametrize(
    "result,expected_code",
    [("yes", 0), ("no", 0), ("halt", 0), ("no_answer", 2), ("ambiguous", 2)],
)
def test_main_exit_code_fails_closed_on_a_missing_answer(
    monkeypatch, tmp_path, result, expected_code
):
    _stub_wait(monkeypatch, {"result": result, "messages": 3, "elapsed_s": 1.0})
    out = tmp_path / "nested" / "remote.json"
    code = main(["judgement", "--out", str(out)])
    assert code == expected_code
    written = json.loads(out.read_text())
    assert written["result"] == result
    assert written["mode"] == "judgement"


def test_main_writes_evidence_even_when_the_operator_never_answered(monkeypatch, tmp_path):
    # A silent timeout must still leave a record; a stage cannot count on
    # evidence that was never written.
    _stub_wait(monkeypatch, {"result": "no_answer", "messages": 0, "elapsed_s": 120.0})
    out = tmp_path / "remote.json"
    assert main(["release", "--out", str(out)]) == 2
    written = json.loads(out.read_text())
    assert written["messages"] == 0
    assert written["result"] == "no_answer"
