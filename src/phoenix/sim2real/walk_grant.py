"""Bounded walking authority: one finite, run-bound grant per walking launch (stdlib only).

Every walking launch (``scripts/harness_preflight.sh W`` for the gantry, ``FREE_FLOOR drill|walk``)
issues ONE ``walk_grant.json`` in its run directory. The LowCmd bridge and the policy node both
refuse a walking config without a grant that is valid for exactly this process, so a bridge and a
policy node started by hand can no longer open an unbounded walking window: the old behaviour was
``safety.max_runtime_s`` (120 s) at any ``/cmd_vel`` up to ``safety.max_command``.

A grant names: the mode (``GANTRY`` or ``FREE_FLOOR``), the approved preset, the run directory
(= run id), the code sha, the config and policy hashes, a wall-clock expiry, the authority window
and the per-axis command envelope. Authority window, runtime and envelope are RE-DERIVED from the
preset by the verifier, never trusted from the file, so a hand-edited grant cannot widen them.

This is accident-proofing, not tamper-proofing: the file is plain JSON and nothing here is a
secret. It closes the ways to walk by launching the wrong thing or an old thing; it cannot stop
someone who forges a grant on purpose. It says nothing about the rope: ``GANTRY`` still rests on
the operator's typed confirmation.

STOP never depends on this file. Expiry or a bad grant removes walking capability (the node
latches, the bridge gate latches into its existing hold-then-sit); the deadman and STOP paths do
not read the grant at all.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phoenix.sim2real.walk_command import PRESETS, profile_duration

SCHEMA = "phoenix-walk-grant/v1"
GRANT_FILE = "walk_grant.json"
RELEASED_FILE = "walk_grant.released"
LOCK_FILE = ".walk_issue.lock"
#: How long an issuer waits for another issuer's check-and-write before refusing.
LOCK_WAIT_S = 5.0

MODE_GANTRY = "GANTRY"
MODE_FREE_FLOOR = "FREE_FLOOR"
MODES = (MODE_GANTRY, MODE_FREE_FLOOR)

#: FREE_FLOOR drill: the walking config at a ZERO command (the STOP/deadman drill).
STAND_ZERO = "stand_zero"
#: The gantry presets the A/B campaign and the first gantry walks use. vx05_3s and vxm03_2s stay in
#: walk_command.PRESETS (listing, analyzer) but no launcher may authorize them.
GANTRY_PRESETS = ("vx03_2s", "vx03_4s", "yaw05_3s")
FREE_FLOOR_WALK_PRESET = "vx03_2s"
ALLOWED_PRESETS: dict[str, tuple[str, ...]] = {
    MODE_GANTRY: GANTRY_PRESETS,
    MODE_FREE_FLOOR: (STAND_ZERO, FREE_FLOOR_WALK_PRESET),
}
#: FREE_FLOOR is the incumbent model_2750 only; this is its ONNX sha256 (equal to the gate's
#: INCUMBENT["policy.onnx"], asserted by a test). Seed 50 can never obtain a FREE_FLOOR grant.
FREE_FLOOR_ONNX_SHA256 = "6d36796f857a490b3d446492581aa1952fa9afdb689bef88b09410db10512c32"

#: Policy-window margin after the profile ends (the harness's long-standing default, see W).
AUTHORITY_MARGIN_S = 4.0
#: The drill holds a zero command this long (the gate's DRILL_AUTHORITY_S).
DRILL_AUTHORITY_S = 5.0
#: Operational margin on top of the sequential startup waits (bridge armed, stand-up, policy mode).
LAUNCH_SLACK_S = 5.0
#: No grant may be valid for longer than this, whatever the arithmetic says.
MAX_WINDOW_S = 120.0
#: Command tolerance on the envelope (float noise only).
ENVELOPE_TOL = 1e-6


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(stamp: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if stamp is None else stamp, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def preset_envelope(mode: str, preset: str) -> tuple[float, float, float]:
    """Per-axis |command| ceiling for an approved (mode, preset); raises ``ValueError`` if not."""
    if mode not in ALLOWED_PRESETS or preset not in ALLOWED_PRESETS[mode]:
        raise ValueError(f"preset {preset!r} is not approved for mode {mode!r}")
    if preset == STAND_ZERO:
        return (0.0, 0.0, 0.0)
    vx, vy, wz = PRESETS[preset].target
    return (abs(vx), abs(vy), abs(wz))


def preset_authority_s(mode: str, preset: str) -> float:
    """Policy authority window for an approved (mode, preset): profile + margin (drill: fixed)."""
    preset_envelope(mode, preset)  # validates
    if preset == STAND_ZERO:
        return DRILL_AUTHORITY_S
    return float(profile_duration(PRESETS[preset])) + AUTHORITY_MARGIN_S


def preset_max_runtime_s(first_msg_s: float, authority_s: float) -> float:
    """The policy node's runtime cap, as the harness has always derived it."""
    return float(first_msg_s) + float(authority_s)


def preset_window_s(first_msg_s: float, standup_s: float, authority_s: float) -> float:
    """How long a grant is valid after issue: the sequential waits (bridge armed, stand-up
    complete, policy in control) at their worst, the authority window, and a small slack."""
    return 3.0 * float(first_msg_s) + float(standup_s) + float(authority_s) + LAUNCH_SLACK_S


def _body_sha256(body: Mapping[str, Any]) -> str:
    payload = {k: v for k, v in body.items() if k != "body_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _num(value: Any) -> float | None:
    """``value`` as a float if it is a finite real number, else None (never raises)."""
    return float(value) if _finite(value) else None


def read_grant(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """(grant, []) or (None, why). Never raises; a symlink or a malformed file is refused."""
    path = Path(path)
    if path.is_symlink():
        return None, [f"{path.name} is a symlink"]
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return None, [f"walk grant unreadable: {exc}"]
    if not isinstance(data, dict):
        return None, ["walk grant is not a JSON object"]
    return data, []


def grant_problems(
    grant: Mapping[str, Any] | None,
    *,
    run_id: str | None,
    code_sha: str | None,
    config_sha256: str | None = None,
    onnx_sha256: str | None = None,
    now: float | None = None,
    mode: str | None = None,
    rehearsal: bool | None = None,
) -> list[str]:
    """Every reason ``grant`` does not authorize THIS process now. Empty means it does.

    ``run_id`` and ``code_sha`` are always required (``None`` fails: a verifier that cannot say
    which run or which code it is must not accept anything). ``config_sha256`` and ``onnx_sha256``
    are checked when the verifier can compute them (the policy node does both, the bridge the
    config)."""
    if not isinstance(grant, Mapping):
        return ["no walk grant"]
    problems: list[str] = []
    if grant.get("schema") != SCHEMA:
        problems.append(f"grant has the wrong schema {grant.get('schema')!r}")
    if grant.get("body_sha256") != _body_sha256(grant):
        problems.append("grant body hash does not match (edited or truncated)")
    g_mode, g_preset = grant.get("mode"), grant.get("preset")
    if g_mode not in MODES:
        problems.append(f"grant mode {g_mode!r} is not one of {MODES}")
    elif g_preset not in ALLOWED_PRESETS[g_mode]:
        problems.append(
            f"grant preset {g_preset!r} is not approved for {g_mode} (approved: "
            f"{', '.join(ALLOWED_PRESETS[g_mode])})"
        )
    if mode is not None and g_mode != mode:
        problems.append(f"grant is for mode {g_mode!r}, this launch is {mode!r}")
    if g_mode == MODE_FREE_FLOOR and grant.get("onnx_sha256") != FREE_FLOOR_ONNX_SHA256:
        problems.append("a FREE_FLOOR grant is for the incumbent model_2750 ONNX only")
    if run_id is None or grant.get("run_id") != run_id:
        problems.append(f"grant is for run {grant.get('run_id')!r}, this process is run {run_id!r}")
    if not code_sha or grant.get("code_sha") != code_sha:
        problems.append("grant names different code than the code running")
    if config_sha256 is not None and grant.get("config_sha256") != config_sha256:
        problems.append("grant names a different deploy config than the one loaded")
    if onnx_sha256 is not None and grant.get("onnx_sha256") != onnx_sha256:
        problems.append("grant names a different policy ONNX than the one loaded")
    if rehearsal is not None and bool(grant.get("rehearsal")) != rehearsal:
        problems.append(
            "a rehearsal grant cannot authorize a live process (and the reverse)"
            if grant.get("rehearsal")
            else "a live grant is not valid in a rehearsal"
        )

    issued, expires = _num(grant.get("issued_unix")), _num(grant.get("expires_unix"))
    if issued is None or expires is None:
        problems.append("grant has no finite issue/expiry time")
    else:
        t = time.time() if now is None else now
        if expires - issued > MAX_WINDOW_S:
            problems.append(
                f"grant window {expires - issued:.0f} s exceeds the {MAX_WINDOW_S:.0f} s cap"
            )
        if t >= expires:
            problems.append(f"grant expired {t - expires:.1f} s ago")
        if t < issued - 5.0:
            problems.append("grant is dated in the future")

    if g_mode in MODES and g_preset in ALLOWED_PRESETS.get(g_mode, ()):
        want_env = preset_envelope(g_mode, g_preset)
        want_auth = preset_authority_s(g_mode, g_preset)
        env = grant.get("envelope")
        if not (
            isinstance(env, Sequence)
            and not isinstance(env, str)
            and len(env) == 3
            and all(_finite(v) for v in env)
            and all(abs(float(a) - b) <= ENVELOPE_TOL for a, b in zip(env, want_env, strict=True))
        ):
            problems.append(f"grant envelope {env!r} is not the preset's {list(want_env)}")
        auth = _num(grant.get("authority_s"))
        if auth is None or abs(auth - want_auth) > 1e-6:
            problems.append(
                f"grant authority_s {grant.get('authority_s')!r} is not the preset's {want_auth}"
            )
        first_msg = _num(grant.get("first_message_timeout_s"))
        standup = _num(grant.get("standup_s"))
        if first_msg is None or standup is None or first_msg <= 0 or standup < 0:
            problems.append("grant has no finite startup timing")
        elif issued is not None and expires is not None:
            want_window = preset_window_s(first_msg, standup, want_auth)
            if abs((expires - issued) - want_window) > 1e-3:
                problems.append(
                    f"grant window {expires - issued:.1f} s is not the {want_window:.1f} s the "
                    "preset allows"
                )
            runtime = _num(grant.get("max_runtime_s"))
            if runtime is None or abs(runtime - preset_max_runtime_s(first_msg, want_auth)) > 1e-6:
                problems.append("grant max_runtime_s is not the preset's")
    return problems


def window_remaining_s(grant: Mapping[str, Any], now: float | None = None) -> float:
    """Seconds of validity left (<= 0 means expired)."""
    return float(grant["expires_unix"]) - (time.time() if now is None else now)


def command_exceeds(
    command: Sequence[float], envelope: Sequence[float], tol: float = ENVELOPE_TOL
) -> bool:
    """True if any axis of ``command`` is non-finite or above the grant's envelope."""
    return any(
        (not math.isfinite(float(c))) or abs(float(c)) > float(e) + tol
        for c, e in zip(command, envelope, strict=True)
    )


# ------------------------------------------------------------------ issuing
def other_live_grants(root: Path, run_dir: Path, now: float) -> list[str]:
    """Grants under ``root/<session>/<run>/`` other than ``run_dir``'s that are still valid and
    not released. One walking authority at a time, whichever mode."""
    out: list[str] = []
    if not root.is_dir():
        return out
    for grant_path in sorted(root.glob(f"*/*/{GRANT_FILE}")):
        if grant_path.parent.resolve() == run_dir.resolve():
            continue
        if (grant_path.parent / RELEASED_FILE).exists():
            continue
        g, err = read_grant(grant_path)
        if err or g is None:
            # an unreadable grant (e.g. an issuer died mid-write) blocks conservatively, but only
            # as long as a grant could have lived: MAX_WINDOW_S after the file was last written.
            try:
                written = grant_path.lstat().st_mtime
            except OSError:
                written = now
            if now < written + MAX_WINDOW_S:
                out.append(
                    f"{grant_path.parent.name}: unreadable grant (treated as live for "
                    f"{written + MAX_WINDOW_S - now:.0f} s): {err}"
                )
        elif _finite(g.get("expires_unix")) and now < float(g["expires_unix"]):
            out.append(
                f"{grant_path.parent.name}: a {g.get('mode')} {g.get('preset')} grant is live "
                f"for another {float(g['expires_unix']) - now:.0f} s"
            )
    return out


@contextlib.contextmanager
def _issue_lock(root: Path) -> Iterator[str | None]:
    """Serialize the check-and-write of issuing across processes (``flock`` on a file in the
    session root). Yields None when held, else why it could not be taken. The kernel drops the
    lock when the holder exits or dies, so a crash cannot leave a stale lock or any authority."""
    fd = -1
    try:
        root.mkdir(parents=True, exist_ok=True)
        fd = os.open(root / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        yield f"cannot open the issue lock in {root}: {exc}"
        return
    try:
        deadline = time.monotonic() + LOCK_WAIT_S
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    yield f"another walking grant is being issued (lock busy {LOCK_WAIT_S:.0f} s)"
                    return
                time.sleep(0.02)
        yield None
    finally:
        os.close(fd)  # closing releases the flock


def issue_grant(
    *,
    mode: str,
    preset: str,
    run_dir: Path,
    code_sha: str,
    config_path: Path,
    onnx_path: Path,
    first_message_timeout_s: float,
    standup_s: float,
    session_root: Path | None = None,
    unreviewed: Sequence[str] = (),
    rehearsal: bool = False,
    now: float | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Write ``run_dir/walk_grant.json`` or return why not. One grant per run directory.

    ``unreviewed`` is the list of unreviewed free-floor walks (from ``free_floor_state``); a
    GANTRY grant is refused while any exists. ``session_root`` enables the one-live-grant check."""
    run_dir = Path(run_dir)
    problems: list[str] = []
    if mode not in MODES:
        return None, [f"mode {mode!r} is not one of {MODES}"]
    if preset not in ALLOWED_PRESETS[mode]:
        return None, [
            f"preset {preset!r} is not approved for {mode} "
            f"(approved: {', '.join(ALLOWED_PRESETS[mode])})"
        ]
    if not code_sha:
        problems.append("no code sha to bind the grant to")
    if not (_finite(first_message_timeout_s) and first_message_timeout_s > 0):
        problems.append("first_message_timeout_s must be a finite positive number")
    if not (_finite(standup_s) and standup_s >= 0):
        problems.append("standup_s must be finite and not negative")
    if mode == MODE_GANTRY and unreviewed:
        problems.extend(f"unreviewed free-floor walk blocks GANTRY: {u}" for u in unreviewed)
    t = time.time() if now is None else now
    try:
        onnx_sha = file_sha256(onnx_path)
        config_sha = file_sha256(config_path)
    except OSError as exc:
        return None, [*problems, f"cannot hash the policy/config: {exc}"]
    if mode == MODE_FREE_FLOOR and onnx_sha != FREE_FLOOR_ONNX_SHA256:
        problems.append(
            "FREE_FLOOR is the incumbent model_2750 only; this ONNX is a different policy"
        )
    if problems:
        return None, problems

    authority = preset_authority_s(mode, preset)
    grant: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": mode,
        "preset": preset,
        "run_id": run_dir.name,
        "code_sha": code_sha,
        "config_sha256": config_sha,
        "onnx_sha256": onnx_sha,
        "envelope": list(preset_envelope(mode, preset)),
        "authority_s": authority,
        "max_runtime_s": preset_max_runtime_s(first_message_timeout_s, authority),
        "first_message_timeout_s": float(first_message_timeout_s),
        "standup_s": float(standup_s),
        "issued_unix": t,
        "expires_unix": t + preset_window_s(first_message_timeout_s, standup_s, authority),
        "issued_utc": _utc(t),
        "rehearsal": bool(rehearsal),
    }
    grant["body_sha256"] = _body_sha256(grant)
    target = run_dir / GRANT_FILE
    with contextlib.ExitStack() as stack:
        if session_root is not None:
            # one walking authority at a time: the check for another live grant and this write
            # happen under the same lock, so two simultaneous issuers cannot both pass the check.
            busy = stack.enter_context(_issue_lock(Path(session_root)))
            if busy is not None:
                return None, [busy]
            others = other_live_grants(Path(session_root), run_dir, t)
            if others:
                return None, [f"another walking grant: {p}" for p in others]
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return None, [f"{target} already exists: one grant per run, never reused"]
        except OSError as exc:
            return None, [f"cannot write {target}: {exc}"]
        with os.fdopen(fd, "w") as handle:
            json.dump(grant, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return grant, []


def release(run_dir: Path) -> None:
    """Mark the run's grant finished so the next launch is not held back until it expires."""
    marker = Path(run_dir) / RELEASED_FILE
    if (Path(run_dir) / GRANT_FILE).exists() and not marker.exists():
        marker.write_text(_utc() + "\n")


# ---------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    import yaml  # local: the verifiers above stay stdlib-only

    from phoenix.sim2real import deploy_contract as dc
    from phoenix.sim2real.free_floor_state import unreviewed_runs
    from phoenix.sim2real.provenance import resolve_code_identity

    ap = argparse.ArgumentParser(description="Issue or release a bounded walking grant.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    iss = sub.add_parser("issue")
    iss.add_argument("--mode", required=True, choices=MODES)
    iss.add_argument("--preset", required=True)
    iss.add_argument("--run-dir", type=Path, required=True)
    iss.add_argument("--config", type=Path, required=True)
    iss.add_argument("--expect-sha", required=True)
    iss.add_argument("--standup-s", type=float, required=True)
    rel = sub.add_parser("release")
    rel.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.cmd == "release":
        release(args.run_dir)
        return 0

    repo = Path(__file__).resolve().parents[3]
    ident = resolve_code_identity(repo)
    if ident.sha != args.expect_sha:
        print(
            f"BLOCKED: running code {ident.sha} is not --expect-sha {args.expect_sha}",
            file=sys.stderr,
        )
        return 2
    cfg = yaml.safe_load(args.config.read_text())
    onnx = Path((cfg.get("policy") or {}).get("onnx_path") or "")
    first_msg = float((cfg.get("safety") or {}).get("first_message_timeout_s", 0))
    root = dc.session_root()
    grant, problems = issue_grant(
        mode=args.mode,
        preset=args.preset,
        run_dir=args.run_dir,
        code_sha=ident.sha or "",
        config_path=args.config,
        onnx_path=onnx,
        first_message_timeout_s=first_msg,
        standup_s=args.standup_s,
        session_root=root,
        unreviewed=unreviewed_runs(dc.default_session_dirs()),
        rehearsal=os.environ.get("PHOENIX_REHEARSAL") == "1",
    )
    if grant is None:
        for p in problems:
            print(f"BLOCKED: walk grant: {p}", file=sys.stderr)
        return 2
    print(
        f"[grant] {grant['mode']} {grant['preset']} run {grant['run_id']}: authority "
        f"{grant['authority_s']:.1f} s, valid until {_utc(grant['expires_unix'])}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
