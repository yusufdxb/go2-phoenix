#!/usr/bin/env python3
"""Release the GO2's stock sport-mode controller so a custom low-level policy can
command the motors, and restore it afterward.

Official release sequence (unitree_sdk2py 1.0.1,
``comm/motion_switcher/motion_switcher_api.py``, verified 2026-09-24 on this
workstation: CHECK_MODE=1001, SELECT_MODE=1002, RELEASE_MODE=1003)::

    CheckMode()
    while CheckMode().name is not empty:
        SportClient.StandDown()
        MotionSwitcherClient.ReleaseMode()
        sleep(1..5 s)
        CheckMode()

The robot goes LIMP the moment ReleaseMode succeeds: it must already be hung on
the gantry or lying down before this script is run with ``--i-am-at-the-robot``.
Nothing here checks that a gantry is in use; the operator is the safety layer.

Hard, non-negotiable rule (memory: feedback_go2_motion_switcher.md, 2026-04-17
lab incident): SelectMode may NEVER be called with any name other than "mcf" or
"ai" on this GO2 EDU. "normal" is not a registered mode name; a request with it
silently returns status 0 while leaving the switcher uninitialized, which wedges
ReleaseMode/SelectMode/Sport-API for every subsequent call and requires a full
power cycle to clear. This script hard-rejects any other name in code (see
:func:`select_mode`), not just in a comment, and a test enforces it.

Modes:

* (default) ``--dry-run``: prints every call it WOULD make, makes none. Safe
  with no robot connected; used for CI and to review the sequence before a
  live run.
* ``--i-am-at-the-robot``: required, in addition to no ``--dry-run``, to
  execute for real. Two separate flags on purpose: no single typo arms it.
* ``--restore``: calls SelectMode("mcf") only, to hand the stock controller
  back to normal operation after a session. Mutually exclusive with a release
  run (an operator runs one or the other, never both in one invocation).

Usage::

    # Review what a release would do (no robot contact):
    python3 scripts/go2_release_sport_mode.py

    # Real release, robot already hung on the gantry or lying down:
    python3 scripts/go2_release_sport_mode.py --i-am-at-the-robot

    # Hand the stock controller back:
    python3 scripts/go2_release_sport_mode.py --restore --i-am-at-the-robot
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Any, Protocol

#: The ONLY two mode names this GO2 EDU may ever be SelectMode'd into.
#: See feedback_go2_motion_switcher.md: "normal" wedges the switcher and
#: requires a full power cycle. Hard-rejected in select_mode(), not just
#: documented, and tests/test_go2_release_sport_mode.py enforces it stays that way.
ALLOWED_SELECT_MODE_NAMES: frozenset[str] = frozenset({"mcf", "ai"})

DEFAULT_RESTORE_MODE = "mcf"
DEFAULT_POLL_INTERVAL_S = 2.0
DEFAULT_MAX_ATTEMPTS = 5


class ForbiddenModeNameError(ValueError):
    """Raised (never caught) when code asks to SelectMode a disallowed name."""


class MotionSwitcherLike(Protocol):
    def CheckMode(self) -> tuple[int, dict[str, Any] | None]: ...  # noqa: N802
    def SelectMode(self, name: str) -> tuple[int, Any]: ...  # noqa: N802
    def ReleaseMode(self) -> tuple[int, Any]: ...  # noqa: N802


class SportClientLike(Protocol):
    def StandDown(self) -> int: ...  # noqa: N802


@dataclass
class ReleaseResult:
    attempts: int
    final_mode_name: str | None
    released: bool
    log: list[str]


def select_mode(client: MotionSwitcherLike, name: str, *, log: list[str]) -> None:
    """The ONLY code path that may call ``client.SelectMode``.

    Hard-rejects any name outside :data:`ALLOWED_SELECT_MODE_NAMES` before the
    client is ever touched. This function exists specifically so that rule can
    be tested and cannot be bypassed by a call site that reaches into the
    client directly.
    """
    if name not in ALLOWED_SELECT_MODE_NAMES:
        raise ForbiddenModeNameError(
            f"refusing SelectMode({name!r}): only {sorted(ALLOWED_SELECT_MODE_NAMES)} are "
            "allowed on this GO2 EDU. 'normal' and any other name silently wedges the "
            "motion switcher (status 0, but ReleaseMode/SelectMode/Sport API then fail "
            "until a full power cycle). See feedback_go2_motion_switcher.md."
        )
    log.append(f"SelectMode({name!r})")
    code, _ = client.SelectMode(name)
    if code != 0:
        raise RuntimeError(f"SelectMode({name!r}) returned code={code}")


def check_mode(client: MotionSwitcherLike, *, log: list[str]) -> str | None:
    log.append("CheckMode()")
    code, data = client.CheckMode()
    if code != 0:
        raise RuntimeError(f"CheckMode() returned code={code}")
    name = (data or {}).get("name")
    return name if name else None


def release_sequence(
    switcher: MotionSwitcherLike,
    sport: SportClientLike,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    sleep_fn=time.sleep,
) -> ReleaseResult:
    """CheckMode -> while name non-empty: StandDown, ReleaseMode, sleep, CheckMode.

    Never calls SelectMode. Robot goes limp the tick ReleaseMode succeeds;
    :func:`main` is the only caller and it refuses to reach here without
    ``--i-am-at-the-robot``.
    """
    log: list[str] = []
    name = check_mode(switcher, log=log)
    attempts = 0
    while name and attempts < max_attempts:
        attempts += 1
        log.append(f"attempt {attempts}: mode={name!r}")
        log.append("SportClient.StandDown()")
        sport.StandDown()
        log.append("MotionSwitcherClient.ReleaseMode()")
        code, _ = switcher.ReleaseMode()
        if code != 0:
            log.append(f"ReleaseMode() returned code={code}, continuing")
        sleep_fn(poll_interval_s)
        name = check_mode(switcher, log=log)
    return ReleaseResult(attempts=attempts, final_mode_name=name, released=(name is None), log=log)


def restore_sequence(
    switcher: MotionSwitcherLike, *, mode: str = DEFAULT_RESTORE_MODE
) -> list[str]:
    log: list[str] = []
    select_mode(switcher, mode, log=log)
    return log


def _build_real_clients() -> tuple[MotionSwitcherLike, SportClientLike]:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient,
    )
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    switcher = MotionSwitcherClient()
    switcher.Init()
    sport = SportClient()
    sport.Init()
    return switcher, sport


class _DryRunSwitcher:
    """Prints what it would do; makes no network call. Reports a plausible
    'mcf' steady state so a dry-run's printed sequence terminates like a real
    release would."""

    def __init__(self) -> None:
        self._checks = 0

    def CheckMode(self) -> tuple[int, dict[str, Any] | None]:  # noqa: N802
        self._checks += 1
        name = "mcf" if self._checks == 1 else ""
        print(f"[DRY RUN] CheckMode() -> name={name!r}")
        return 0, {"name": name}

    def SelectMode(self, name: str) -> tuple[int, Any]:  # noqa: N802
        print(f"[DRY RUN] SelectMode({name!r})")
        return 0, None

    def ReleaseMode(self) -> tuple[int, Any]:  # noqa: N802
        print("[DRY RUN] ReleaseMode()")
        return 0, None


class _DryRunSport:
    def StandDown(self) -> int:  # noqa: N802
        print("[DRY RUN] SportClient.StandDown()")
        return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="print the sequence, make no robot contact (default).",
    )
    p.add_argument(
        "--i-am-at-the-robot",
        action="store_true",
        help="required, together with omitting --dry-run's default, to execute for real. "
        "The robot goes LIMP on release: it must already be on the gantry or lying down.",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="disable the dry-run default. Requires --i-am-at-the-robot as well; "
        "neither flag alone arms a live run.",
    )
    p.add_argument(
        "--restore",
        action="store_true",
        help="hand the stock controller back with SelectMode('mcf') only, instead of "
        "releasing it. Mutually exclusive with a release run.",
    )
    p.add_argument(
        "--restore-mode",
        default=DEFAULT_RESTORE_MODE,
        choices=sorted(ALLOWED_SELECT_MODE_NAMES),
        help=f"mode to restore to (default {DEFAULT_RESTORE_MODE!r}); only mcf/ai are ever valid.",
    )
    p.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    p.add_argument("--poll-interval-s", type=float, default=DEFAULT_POLL_INTERVAL_S)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    live = args.live and args.i_am_at_the_robot
    if args.live and not args.i_am_at_the_robot:
        print(
            "REFUSING: --live requires --i-am-at-the-robot as well. Neither flag alone "
            "arms a live run.",
            file=sys.stderr,
        )
        return 2
    if args.i_am_at_the_robot and not args.live:
        print(
            "REFUSING: --i-am-at-the-robot requires --live as well. Neither flag alone "
            "arms a live run.",
            file=sys.stderr,
        )
        return 2

    if live:
        print(
            "LIVE run: the robot will go LIMP on release. Confirm it is already hung on "
            "the gantry or lying down."
        )
        switcher, sport = _build_real_clients()
    else:
        print("DRY RUN (default): printing the sequence, no robot contact.")
        switcher, sport = _DryRunSwitcher(), _DryRunSport()

    if args.restore:
        log = restore_sequence(switcher, mode=args.restore_mode)
        for line in log:
            print(line)
        print(f"restore complete: mode={args.restore_mode!r}")
        return 0

    result = release_sequence(
        switcher,
        sport,
        max_attempts=args.max_attempts,
        poll_interval_s=args.poll_interval_s,
    )
    for line in result.log:
        print(line)
    if result.released:
        print(f"RELEASED after {result.attempts} attempt(s): mode is empty.")
        return 0
    print(
        f"NOT RELEASED after {result.attempts} attempt(s): mode still {result.final_mode_name!r}.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
