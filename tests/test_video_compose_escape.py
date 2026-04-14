"""Sanity checks for the drawtext escape helper."""

from __future__ import annotations

from phoenix.demo.video_compose import _escape


def test_escape_colon() -> None:
    assert _escape("SIM: baseline") == r"SIM\: baseline"


def test_escape_apostrophe() -> None:
    assert _escape("don't") == r"don\'t"


def test_escape_backslash_first() -> None:
    # Backslash must be escaped before others, otherwise we double-escape.
    assert _escape("a\\b") == r"a\\b"


def test_escape_idempotent_on_plain_text() -> None:
    assert _escape("SIM") == "SIM"
    assert _escape("SIM + Phoenix") == "SIM + Phoenix"
