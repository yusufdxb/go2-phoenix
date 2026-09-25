"""A crude simulated GO2 leg plant for FIX_STAND / FSM tests. NOT a robot model.

Why it exists: the 2026-09-21 ramp tests assumed the measured posture equals the
commanded one every tick (ideal tracking), which is exactly the assumption the F
run broke. This plant has what an ideal-tracking test hides:

* PD actuation ``tau = kp (q_cmd - q) - kd dq`` with torque saturation,
* a constant per-joint load torque (gravity-like), so every joint settles with a
  steady-state error ``load / kp``,
* inertia, integrated at 200 Hz under a 50 Hz command (decimation 4),
* one control tick of measurement latency,
* optional frozen joints (a blocked or unpowered leg).

Numbers are illustrative, chosen to produce lag and steady-state error of the
order the design must tolerate; they are not identified from the GO2. Tests
built on it are SEQUENCING tests of the controller logic, not hardware
validation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from phoenix.sim2real.go2_model import POLICY_JOINT_ORDER, UNITREE_MOTOR_ORDER

ORDER = POLICY_JOINT_ORDER
N = len(ORDER)

#: Measured at the start of stage F on 2026-09-21, Unitree motor order (calves past
#: the -2.72 hard limit), converted to the policy order below.
FOLDED_UNITREE = np.asarray(
    [-0.07, 1.25, -2.76, 0.05, 1.25, -2.8, -0.36, 1.28, -2.8, 0.37, 1.26, -2.78]
)
FOLDED = np.asarray([FOLDED_UNITREE[UNITREE_MOTOR_ORDER.index(n)] for n in ORDER])

#: Load torque per joint type, N m (illustrative, positive = pulls q down).
LOAD_BY_TYPE = {"hip": 0.5, "thigh": 3.0, "calf": -4.0}


def load_vector(scale: float = 1.0) -> np.ndarray:
    out = np.zeros(N)
    for i, name in enumerate(ORDER):
        kind = name.split("_")[1]
        out[i] = LOAD_BY_TYPE[kind] * scale
    return out


class Plant:
    def __init__(
        self,
        q0: Sequence[float],
        *,
        load_scale: float = 1.0,
        inertia: float = 0.05,
        tau_max: float = 23.7,
        frozen: Sequence[int] = (),
        latency_ticks: int = 1,
        physics_hz: float = 200.0,
        control_hz: float = 50.0,
    ) -> None:
        self.q = np.asarray(q0, dtype=np.float64).copy()
        self.dq = np.zeros(N)
        self.load = load_vector(load_scale)
        self.inertia = inertia
        self.tau_max = tau_max
        self.frozen = np.zeros(N, dtype=bool)
        self.frozen[list(frozen)] = True
        self.dt = 1.0 / physics_hz
        self.decimation = int(round(physics_hz / control_hz))
        self._delay: list[tuple[np.ndarray, np.ndarray]] = [
            (self.q.copy(), self.dq.copy()) for _ in range(latency_ticks)
        ]

    def measured(self) -> tuple[np.ndarray, np.ndarray]:
        """Measurement as of ``latency_ticks`` control ticks ago."""
        return self._delay[0] if self._delay else (self.q.copy(), self.dq.copy())

    def step(self, target: Sequence[float] | None, kp: float, kd: float) -> None:
        tgt = self.q if target is None else np.asarray(target, dtype=np.float64)
        for _ in range(self.decimation):
            tau = kp * (tgt - self.q) - kd * self.dq
            tau = np.clip(tau, -self.tau_max, self.tau_max) - self.load
            ddq = tau / self.inertia
            self.dq = np.where(self.frozen, 0.0, self.dq + ddq * self.dt)
            self.q = self.q + self.dq * self.dt
        if self._delay:
            self._delay.pop(0)
            self._delay.append((self.q.copy(), self.dq.copy()))


def test_plant_settles_with_steady_state_error() -> None:
    target = np.asarray([0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5])
    plant = Plant(target)
    for _ in range(200):
        plant.step(target, 60.0, 5.0)
    err = target - plant.q
    expected = load_vector() / 60.0  # kp (target - q) = load at rest
    assert np.allclose(err, expected, atol=1e-3)
    assert np.abs(err).max() > 0.05  # not ideal tracking


def test_frozen_joint_does_not_move() -> None:
    plant = Plant(FOLDED, frozen=[8])
    for _ in range(50):
        plant.step(FOLDED + 0.3, 60.0, 5.0)
    assert plant.q[8] == FOLDED[8]
