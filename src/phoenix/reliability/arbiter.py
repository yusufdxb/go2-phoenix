"""Simplex safety arbiter: hand /cmd_vel from the learned policy to a fallback.

This is the actuation half of the reliability layer. The OOD monitor
(:mod:`phoenix.reliability.ood_monitor`) says *how* out-of-distribution the
policy's internal state is; the arbiter decides *whether and how* to hand
control to a safe fallback controller, and produces a blend weight the deploy
layer applies to the two controllers' outputs.

It is a Simplex architecture: an unverified high-performance controller (the
learned policy) supervised by a verified safe controller (a stand / classical
fallback), with a decision module that switches between them. The design
follows the review points codex raised (2026-07-16):

* **Hysteresis** — separate ``trip`` (engage) and ``clear`` (release)
  thresholds so a score dithering around one value can't chatter the shield.
* **Dwell / persistence** — a run of consecutive ticks above ``trip`` is
  required before engaging, and below ``clear`` before releasing, so a single
  spike neither trips nor clears it.
* **Bounded ramp** — the handoff blends over ``handoff_ticks`` rather than
  snapping, so the robot is never yanked from high-speed locomotion into an
  instant stand. Recovery ramps back over ``recover_ticks``.
* **Fail toward safe** — a non-finite score counts as maximal evidence of
  trouble (drives engagement, never release), matching the monitor's
  NaN -> +inf convention.
* **Re-trip beats recovery** — if the score climbs back above ``trip`` while
  recovering, the arbiter snaps back to full fallback immediately.

Like :mod:`phoenix.sim2real.mode_switch`, this module has no rclpy /
onnxruntime dependency: every branch is unit-tested without Isaac or the robot.

Blend convention: ``blend`` is the weight on the FALLBACK controller in
``[0, 1]``. The deploy layer computes
``cmd = (1 - blend) * learned + blend * fallback``. ``blend == 0`` is pure
learned policy; ``blend == 1`` is pure fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class ShieldState(str, Enum):
    NOMINAL = "nominal"  # learned policy in control
    HANDOFF = "handoff"  # ramping learned -> fallback
    FALLBACK = "fallback"  # safe controller in control
    RECOVERING = "recovering"  # ramping fallback -> learned


@dataclass(frozen=True)
class SimplexArbiterCfg:
    trip_threshold: float  # engage when score exceeds this
    clear_threshold: float  # eligible to release when score drops below this
    trip_persistence: int = 3  # consecutive over-trip ticks required to engage
    clear_persistence: int = 10  # consecutive under-clear ticks required to release
    handoff_ticks: int = 10  # ramp length, learned -> fallback
    recover_ticks: int = 25  # ramp length, fallback -> learned
    min_fallback_ticks: int = 20  # dwell in FALLBACK before release is even considered
    latch: bool = False  # if True, never auto-recover once engaged

    def __post_init__(self) -> None:
        if not self.clear_threshold < self.trip_threshold:
            raise ValueError("clear_threshold must be < trip_threshold (hysteresis)")
        for name in ("trip_persistence", "clear_persistence", "handoff_ticks", "recover_ticks"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.min_fallback_ticks < 0:
            raise ValueError("min_fallback_ticks must be >= 0")


@dataclass(frozen=True)
class ArbiterOutput:
    state: ShieldState
    blend: float  # weight on the fallback controller, [0, 1]

    @property
    def engaged(self) -> bool:
        """True whenever the shield is doing anything but pure learned control."""
        return self.state is not ShieldState.NOMINAL


class SimplexArbiter:
    """Stateful Simplex decision module; feed one OOD score per control tick."""

    def __init__(self, cfg: SimplexArbiterCfg) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._state = ShieldState.NOMINAL
        self._blend = 0.0
        self._ramp = 0  # ticks into the current HANDOFF / RECOVERING ramp
        self._fallback_ticks = 0  # ticks held in FALLBACK
        self._over_trip = 0  # consecutive ticks with score above trip
        self._under_clear = 0  # consecutive ticks with score below clear

    @property
    def state(self) -> ShieldState:
        return self._state

    @property
    def blend(self) -> float:
        return self._blend

    def update(self, score: float) -> ArbiterOutput:
        """Advance the arbiter one tick and return the current decision.

        A non-finite ``score`` is treated as maximal evidence of trouble: it
        counts toward tripping and never toward clearing.
        """
        cfg = self.cfg
        above_trip = (not math.isfinite(score)) or score > cfg.trip_threshold
        below_clear = math.isfinite(score) and score < cfg.clear_threshold

        # Persistence counters (reset the moment the streak breaks).
        self._over_trip = self._over_trip + 1 if above_trip else 0
        self._under_clear = self._under_clear + 1 if below_clear else 0

        if self._state is ShieldState.NOMINAL:
            self._blend = 0.0
            if self._over_trip >= cfg.trip_persistence:
                self._enter_handoff()

        elif self._state is ShieldState.HANDOFF:
            # Commit to the ramp; monitor score is ignored mid-handoff so the
            # transition can't chatter. Ramp is bounded by handoff_ticks.
            self._ramp += 1
            self._blend = min(1.0, self._ramp / cfg.handoff_ticks)
            if self._ramp >= cfg.handoff_ticks:
                self._state = ShieldState.FALLBACK
                self._blend = 1.0
                self._fallback_ticks = 0

        elif self._state is ShieldState.FALLBACK:
            self._blend = 1.0
            self._fallback_ticks += 1
            eligible = (not cfg.latch) and self._fallback_ticks >= cfg.min_fallback_ticks
            if eligible and self._under_clear >= cfg.clear_persistence:
                self._state = ShieldState.RECOVERING
                self._ramp = 0

        elif self._state is ShieldState.RECOVERING:
            # Safety dominates: a fresh trip aborts recovery back to full fallback.
            if self._over_trip >= cfg.trip_persistence:
                self._state = ShieldState.FALLBACK
                self._blend = 1.0
                self._fallback_ticks = 0
            else:
                self._ramp += 1
                self._blend = max(0.0, 1.0 - self._ramp / cfg.recover_ticks)
                if self._ramp >= cfg.recover_ticks:
                    self._state = ShieldState.NOMINAL
                    self._blend = 0.0

        return ArbiterOutput(state=self._state, blend=self._blend)

    def _enter_handoff(self) -> None:
        self._state = ShieldState.HANDOFF
        self._ramp = 0
        self._blend = 0.0
