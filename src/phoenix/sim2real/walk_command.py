"""Deadman-gated /cmd_vel walking-run command for the first GO2 walking test.

Publishes ONE short, preset ``/cmd_vel`` profile, and only while the operator's
deadman is held. It is the only thing in the walking harness that originates a
velocity command, so it is deliberately small, preset-only and fail-closed.

Profile of every attempt: zero lead-in (2.0 s), ONE motion segment, zero tail
(3.0 s). Each axis follows a rate-limited trapezoid (linear 0.5 m/s^2, yaw
1.0 rad/s^2) on every change, including the ramp back to zero, which must fit
inside the tail. A preset outside the envelope (|vx| <= 0.5, |vy| <= 0.3,
|wz| <= 0.8, the policy node's default ``safety.max_command``) is a ``ValueError``
at import time.

Deadman gating (``/phoenix/estop``, std_msgs/Bool, False = deadman held, published
at 10 Hz by ``wireless_estop_node``):

* nothing is published until estop has been False for 1.0 s continuously;
* then ``/cmd_vel`` is published at 20 Hz following the profile from that start;
* estop True, or its last message older than 0.5 s, at any point switches to
  zeros immediately, zeros are kept for 1.0 s, outcome ``deadman_released`` (or
  ``estop_stale``), exit code 2;
* normal completion: zeros for 1.0 s more, outcome ``complete``, exit 0;
* SIGINT/SIGTERM: zeros for 0.5 s, outcome ``interrupted``, exit 3;
* wall-clock cap = profile duration + 5 s from node start: zeros for 1.0 s,
  outcome ``wallclock_cap``, exit 4;
* start deadline (default 3.0 s from the first runner tick): if the 1.0 s hold is not
  satisfied by then, nothing is ever published, outcome ``late_start``, exit 5. A late
  deadman must never start motion that could still be running when the policy window
  ends (the window end makes the bridge sit the robot).

The state machine (:class:`WalkRunner`) and the profile (:func:`command_at`) are
pure and import without ROS; ROS is imported only inside :func:`run_node`.

Recording (``--out``, JSONL, never overwritten): one record per published command
``{t_mono_ns, phase, vx, vy, wz, estop_value, estop_age_s}`` and a final
``{record: "summary", preset, outcome, started_t_mono_ns, n_published, max_abs}``.

    python3 -m phoenix.sim2real.walk_command --list
    python3 -m phoenix.sim2real.walk_command --preset vx03_2s --out walk_cmd.jsonl
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------- profile

#: Per-axis envelope (vx, vy, wz): the policy node's default ``safety.max_command``.
ENVELOPE = (0.5, 0.3, 0.8)
#: Ramp limits (linear m/s^2 for vx and vy, rad/s^2 for wz), applied to every change.
RAMP_LIMITS = (0.5, 0.5, 1.0)
LEAD_IN_S = 2.0
TAIL_S = 3.0

PUBLISH_HZ = 20.0
HOLD_S = 1.0  # estop must be False this long before anything is published
STALE_S = 0.5  # last estop message older than this counts as a lost deadman
ZEROS_AFTER_ABORT_S = 1.0
ZEROS_AFTER_COMPLETE_S = 1.0
ZEROS_AFTER_INTERRUPT_S = 0.5
WALLCLOCK_MARGIN_S = 5.0
START_DEADLINE_S = 3.0  # from the first runner tick; the hold must be satisfied by then

_NS = 1_000_000_000

EXIT_CODES = {
    "complete": 0,
    "deadman_released": 2,
    "estop_stale": 2,
    "interrupted": 3,
    "wallclock_cap": 4,
    "late_start": 5,
}


@dataclass(frozen=True)
class WalkPreset:
    """One motion segment held at (vx, vy, wz) for ``motion_s`` after the lead-in."""

    name: str
    vx: float
    vy: float
    wz: float
    motion_s: float

    @property
    def target(self) -> tuple[float, float, float]:
        return (self.vx, self.vy, self.wz)


def _ramp_s(preset: WalkPreset) -> float:
    """Seconds the slowest axis needs to ramp between zero and its target."""
    return max(abs(v) / a for v, a in zip(preset.target, RAMP_LIMITS, strict=True))


def validate_preset(preset: WalkPreset) -> None:
    """Raise ``ValueError`` unless the preset is inside the envelope and ramp limits."""
    for axis, v, cap in zip(("vx", "vy", "wz"), preset.target, ENVELOPE, strict=True):
        if abs(v) > cap + 1e-12:
            raise ValueError(f"preset {preset.name}: |{axis}|={abs(v)} exceeds envelope {cap}")
    if not (preset.motion_s > 0):
        raise ValueError(f"preset {preset.name}: motion_s must be > 0, got {preset.motion_s}")
    if all(v == 0 for v in preset.target):
        raise ValueError(f"preset {preset.name}: no motion requested")
    ramp = _ramp_s(preset)
    if ramp > preset.motion_s:
        raise ValueError(
            f"preset {preset.name}: motion {preset.motion_s} s is shorter than the "
            f"{ramp:.3f} s ramp to the target"
        )
    if ramp > TAIL_S:
        raise ValueError(
            f"preset {preset.name}: ramp-down {ramp:.3f} s does not fit in the {TAIL_S} s tail"
        )


def _build_presets() -> dict[str, WalkPreset]:
    presets = {
        p.name: p
        for p in (
            WalkPreset("vx03_2s", 0.3, 0.0, 0.0, 2.0),
            WalkPreset("vx03_4s", 0.3, 0.0, 0.0, 4.0),
            WalkPreset("vx05_3s", 0.5, 0.0, 0.0, 3.0),
            WalkPreset("yaw05_3s", 0.0, 0.0, 0.5, 3.0),  # counter-clockwise
            WalkPreset("yawm05_3s", 0.0, 0.0, -0.5, 3.0),  # clockwise
            WalkPreset("yaw08_3s", 0.0, 0.0, 0.8, 3.0),  # envelope edge
            WalkPreset("vx02_yaw03_3s", 0.2, 0.0, 0.3, 3.0),  # combined forward + yaw
            WalkPreset("vxm03_2s", -0.3, 0.0, 0.0, 2.0),
        )
    }
    for p in presets.values():
        validate_preset(p)
    return presets


PRESETS: dict[str, WalkPreset] = _build_presets()


def profile_duration(preset: WalkPreset | str) -> float:
    """Total profile length in seconds: lead-in + motion + tail."""
    p = PRESETS[preset] if isinstance(preset, str) else preset
    return LEAD_IN_S + p.motion_s + TAIL_S


def _axis_at(target: float, accel: float, t: float, motion_s: float) -> float:
    if t <= LEAD_IN_S:
        return 0.0
    mag = abs(target)
    seg_t = t - LEAD_IN_S
    if seg_t < motion_s:
        mag = min(mag, accel * seg_t)
    else:
        mag = max(0.0, mag - accel * (seg_t - motion_s))
    v = mag if target > 0 else -mag
    return v + 0.0  # normalise -0.0 to 0.0


def command_at(preset: WalkPreset | str, t: float) -> tuple[float, float, float, str]:
    """``(vx, vy, wz, phase)`` at ``t`` seconds after the profile start.

    Phases: ``lead_in``, ``motion`` (ramp up and hold), ``tail`` (ramp down and
    zeros), ``done`` (after the profile; zeros).
    """
    p = PRESETS[preset] if isinstance(preset, str) else preset
    if t < 0:
        raise ValueError(f"t must be >= 0, got {t}")
    if t >= profile_duration(p):
        return (0.0, 0.0, 0.0, "done")
    vx, vy, wz = (_axis_at(v, a, t, p.motion_s) for v, a in zip(p.target, RAMP_LIMITS, strict=True))
    if t < LEAD_IN_S:
        phase = "lead_in"
    elif t < LEAD_IN_S + p.motion_s:
        phase = "motion"
    else:
        phase = "tail"
    return (vx, vy, wz, phase)


# ----------------------------------------------------------- pure state machine


@dataclass(frozen=True)
class Command:
    """One command the node must publish this tick."""

    vx: float
    vy: float
    wz: float
    phase: str
    estop_value: bool | None
    estop_age_s: float | None


_WAITING, _RUNNING, _FINISHING, _DONE = "waiting", "running", "finishing", "done"


class WalkRunner:
    """Deadman gating and profile sequencing. No ROS, no clock: callers pass ``now_ns``."""

    def __init__(
        self,
        preset: WalkPreset | str,
        *,
        start_ns: int | None = None,
        hold_s: float = HOLD_S,
        stale_s: float = STALE_S,
        wallclock_s: float | None = None,
        start_deadline_s: float = START_DEADLINE_S,
    ) -> None:
        self.preset = PRESETS[preset] if isinstance(preset, str) else preset
        validate_preset(self.preset)
        self._hold_ns = int(hold_s * _NS)
        self._stale_ns = int(stale_s * _NS)
        if start_deadline_s < hold_s:
            raise ValueError(
                f"start_deadline_s {start_deadline_s} is shorter than the {hold_s} s hold: "
                "the run could never start"
            )
        self._deadline_ns = int(start_deadline_s * _NS)
        #: ``None`` until the first :meth:`step`, which is node start for the deadline and cap.
        self._node_start_ns = start_ns
        cap_s = profile_duration(self.preset) + WALLCLOCK_MARGIN_S
        self._cap_ns = int((cap_s if wallclock_s is None else wallclock_s) * _NS)
        self._state = _WAITING
        self._estop: bool | None = None
        self._estop_ns: int | None = None
        self._held_since_ns: int | None = None
        self.started_ns: int | None = None
        self._finish_ns = 0
        self._finish_phase = ""
        self.outcome: str | None = None

    # -- inputs
    def on_estop(self, value: bool, now_ns: int) -> None:
        if self._estop_ns is not None and now_ns - self._estop_ns > self._stale_ns:
            self._held_since_ns = None  # a gap longer than the stale limit breaks "continuous"
        self._estop, self._estop_ns = bool(value), now_ns
        if self._estop:
            self._held_since_ns = None
        elif self._held_since_ns is None:
            self._held_since_ns = now_ns

    def interrupt(self, now_ns: int) -> None:
        """SIGINT/SIGTERM. Never started: finish silently. Otherwise zeros for 0.5 s."""
        if self._state == _DONE:
            return
        if self._state == _WAITING:
            self.outcome = "interrupted"
            self._state = _DONE
        elif self._state == _RUNNING:
            self._begin_finish("interrupted", now_ns, ZEROS_AFTER_INTERRUPT_S, "interrupt_zeros")
        else:  # already in a zero-hold: keep its outcome, shorten the hold to 0.5 s
            self._finish_ns = min(self._finish_ns, now_ns + int(ZEROS_AFTER_INTERRUPT_S * _NS))

    # -- clock tick
    @property
    def done(self) -> bool:
        return self._state == _DONE

    @property
    def exit_code(self) -> int:
        if self.outcome is None:
            raise RuntimeError("runner has no outcome yet")
        return EXIT_CODES[self.outcome]

    def estop_age_s(self, now_ns: int) -> float | None:
        return None if self._estop_ns is None else (now_ns - self._estop_ns) / _NS

    def _begin_finish(self, outcome: str, now_ns: int, zeros_s: float, phase: str) -> None:
        self.outcome = outcome
        self._finish_ns = now_ns + int(zeros_s * _NS)
        self._finish_phase = phase
        self._state = _FINISHING

    def _zero(self, phase: str, now_ns: int) -> Command:
        return Command(0.0, 0.0, 0.0, phase, self._estop, self.estop_age_s(now_ns))

    def step(self, now_ns: int) -> Command | None:
        """The command to publish now, or ``None`` (publish nothing)."""
        if self._state == _DONE:
            return None
        if self._node_start_ns is None:
            self._node_start_ns = now_ns
        if self._state == _FINISHING:
            if now_ns >= self._finish_ns:
                self._state = _DONE
                return None
            return self._zero(self._finish_phase, now_ns)
        capped = now_ns - self._node_start_ns >= self._cap_ns
        age = None if self._estop_ns is None else now_ns - self._estop_ns
        stale = age is None or age > self._stale_ns
        if self._state == _WAITING:
            if capped:
                self.outcome = "wallclock_cap"
                self._state = _DONE
                return None
            if stale:
                self._held_since_ns = None
            satisfied = (
                self._held_since_ns is not None and now_ns - self._held_since_ns >= self._hold_ns
            )
            elapsed_ns = now_ns - self._node_start_ns
            # Satisfied exactly AT the deadline still starts; anything later never does.
            if elapsed_ns > self._deadline_ns or (
                not satisfied and elapsed_ns == self._deadline_ns
            ):
                self.outcome = "late_start"
                self._state = _DONE
                return None
            if not satisfied:
                return None
            self._state = _RUNNING
            self.started_ns = now_ns
        # RUNNING
        assert self.started_ns is not None
        if self._estop is True:
            self._begin_finish("deadman_released", now_ns, ZEROS_AFTER_ABORT_S, "abort_zeros")
        elif stale:
            self._begin_finish("estop_stale", now_ns, ZEROS_AFTER_ABORT_S, "abort_zeros")
        elif capped:
            self._begin_finish("wallclock_cap", now_ns, ZEROS_AFTER_ABORT_S, "abort_zeros")
        else:
            t = (now_ns - self.started_ns) / _NS
            if t >= profile_duration(self.preset):
                self._begin_finish("complete", now_ns, ZEROS_AFTER_COMPLETE_S, "final_zeros")
            else:
                vx, vy, wz, phase = command_at(self.preset, t)
                return Command(vx, vy, wz, phase, self._estop, self.estop_age_s(now_ns))
        return self._zero(self._finish_phase, now_ns)


# ------------------------------------------------------------------- recording


class Recorder:
    """JSONL writer that refuses to overwrite and tracks the summary figures."""

    def __init__(self, path: Path, preset: str) -> None:
        self.preset = preset
        self.n_published = 0
        self.max_abs = [0.0, 0.0, 0.0]
        self._fh = open(path, "x", encoding="utf-8")

    def record(self, t_mono_ns: int, cmd: Command) -> None:
        self.n_published += 1
        for i, v in enumerate((cmd.vx, cmd.vy, cmd.wz)):
            self.max_abs[i] = max(self.max_abs[i], abs(v))
        rec = {
            "t_mono_ns": t_mono_ns,
            "phase": cmd.phase,
            "vx": cmd.vx,
            "vy": cmd.vy,
            "wz": cmd.wz,
            "estop_value": cmd.estop_value,
            "estop_age_s": cmd.estop_age_s,
        }
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    def summary(self, outcome: str, started_ns: int | None) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "record": "summary",
            "preset": self.preset,
            "outcome": outcome,
            "started_t_mono_ns": started_ns,
            "n_published": self.n_published,
            "max_abs": list(self.max_abs),
        }
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()
        self._fh.close()
        return rec


# -------------------------------------------------------------------- ROS node


def run_node(preset_name: str, out: Path, start_deadline_s: float = START_DEADLINE_S) -> int:
    """Run one attempt on ROS 2. Returns the process exit code."""
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Bool

    rec = Recorder(out, preset_name)
    runner = WalkRunner(preset_name, start_deadline_s=start_deadline_s)
    stop_flag: list[int] = []
    try:
        from rclpy.signals import SignalHandlerOptions

        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except ImportError:  # pragma: no cover - older rclpy
        rclpy.init()
    signal.signal(signal.SIGINT, lambda *_: stop_flag.append(1))
    signal.signal(signal.SIGTERM, lambda *_: stop_flag.append(1))
    node = rclpy.create_node("phoenix_walk_command")
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    pub = node.create_publisher(Twist, "/cmd_vel", qos)
    node.create_subscription(
        Bool,
        "/phoenix/estop",
        lambda m: runner.on_estop(bool(m.data), time.monotonic_ns()),
        QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT),
    )

    def tick() -> None:
        now = time.monotonic_ns()
        cmd = runner.step(now)
        if cmd is None:
            return
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = cmd.vx, cmd.vy, cmd.wz
        pub.publish(msg)
        rec.record(now, cmd)

    node.create_timer(1.0 / PUBLISH_HZ, tick)
    node.get_logger().info(
        f"walk_command up: preset={preset_name} duration={profile_duration(preset_name):.1f}s; "
        f"nothing is published until /phoenix/estop has been False for {HOLD_S}s "
        f"(start deadline {start_deadline_s}s)"
    )
    seen_stop = False
    try:
        while not runner.done:
            rclpy.spin_once(node, timeout_sec=0.01)
            if stop_flag and not seen_stop:
                seen_stop = True
                runner.interrupt(time.monotonic_ns())
        outcome = runner.outcome or "interrupted"
        summary = rec.summary(outcome, runner.started_ns)
        node.get_logger().info(f"walk_command {outcome}: {json.dumps(summary)}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return runner.exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    p.add_argument("--preset", choices=sorted(PRESETS), help="walking preset to run")
    p.add_argument("--out", type=Path, help="JSONL record path (must not exist)")
    p.add_argument("--list", action="store_true", help="list presets and exit")
    p.add_argument(
        "--start-deadline-s",
        type=float,
        default=START_DEADLINE_S,
        help="the 1 s deadman hold must be satisfied within this long of node start, else "
        f"outcome late_start and nothing is published (default {START_DEADLINE_S})",
    )
    args = p.parse_args(argv)
    if args.start_deadline_s < HOLD_S:
        p.error(f"--start-deadline-s must be >= {HOLD_S} (the deadman hold)")
    if not args.list and (args.preset is None or args.out is None):
        p.error("--preset and --out are required unless --list")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        for name, p in PRESETS.items():
            print(
                f"{name} vx={p.vx} vy={p.vy} wz={p.wz} motion={p.motion_s}s "
                f"total={profile_duration(p):.1f}s"
            )
        return 0
    return run_node(args.preset, args.out, args.start_deadline_s)


if __name__ == "__main__":
    sys.exit(main())
