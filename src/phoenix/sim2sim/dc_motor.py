"""Isaac Lab ``DCMotor`` explicit actuator, replicated in numpy for MuJoCo.

Source transcribed (Isaac Lab checkout ``~/Sim/IsaacLab`` at commit
``a4a7602f29e755e2673fe0022ea35566df6dd7d5``):

* ``source/isaaclab/isaaclab/actuators/actuator_pd.py``
  ``IdealPDActuator.compute``::

      error_pos = q_des - q ; error_vel = qd_des - qd
      computed = stiffness * error_pos + damping * error_vel + tau_ff

* ``DCMotor.__init__``::

      vel_at_effort_lim = velocity_limit * (1 + effort_limit / saturation_effort)

* ``DCMotor._clip_effort``::

      qd_c   = clip(qd, -vel_at_effort_lim, vel_at_effort_lim)
      top    = saturation_effort * ( 1 - qd_c / velocity_limit)
      bottom = saturation_effort * (-1 - qd_c / velocity_limit)
      max_effort = clip(top,    max= effort_limit)
      min_effort = clip(bottom, min=-effort_limit)
      applied = clip(computed, min_effort, max_effort)

GO2 parameters are ``UNITREE_GO2_CFG.actuators["base_legs"]`` in
``source/isaaclab_assets/isaaclab_assets/robots/unitree.py``: stiffness 25.0,
damping 0.5, effort_limit 23.5, saturation_effort 23.5, velocity_limit 30.0,
friction 0.0. The position controller in Isaac Lab velocity tasks sends
``q_des`` only, so ``qd_des = 0`` and ``tau_ff = 0``.

This is computed once per PHYSICS step from the current joint state, as Isaac
Lab does (``Articulation.write_data_to_sim`` runs the actuator model before every
``sim.step``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DCMotorParams:
    stiffness: float
    damping: float
    effort_limit: float
    saturation_effort: float
    velocity_limit: float

    @property
    def vel_at_effort_lim(self) -> float:
        return self.velocity_limit * (1.0 + self.effort_limit / self.saturation_effort)


#: Isaac Lab ``UNITREE_GO2_CFG`` DCMotor parameters (all 12 leg joints).
GO2_DC_MOTOR = DCMotorParams(
    stiffness=25.0,
    damping=0.5,
    effort_limit=23.5,
    saturation_effort=23.5,
    velocity_limit=30.0,
)


def dc_motor_torque(
    q_des: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    params: DCMotorParams = GO2_DC_MOTOR,
    *,
    qd_des: np.ndarray | None = None,
    tau_ff: np.ndarray | None = None,
    stiffness_scale: np.ndarray | float = 1.0,
    damping_scale: np.ndarray | float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(computed_effort, applied_effort)`` exactly as Isaac Lab ``DCMotor``."""
    q_des = np.asarray(q_des, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    qd_des = np.zeros_like(qd) if qd_des is None else np.asarray(qd_des, dtype=np.float64)
    tau_ff = np.zeros_like(qd) if tau_ff is None else np.asarray(tau_ff, dtype=np.float64)
    kp = params.stiffness * np.asarray(stiffness_scale, dtype=np.float64)
    kd = params.damping * np.asarray(damping_scale, dtype=np.float64)
    computed = kp * (q_des - q) + kd * (qd_des - qd) + tau_ff
    applied = clip_dc_motor_effort(computed, qd, params)
    return computed, applied


def clip_dc_motor_effort(
    effort: np.ndarray, qd: np.ndarray, params: DCMotorParams = GO2_DC_MOTOR
) -> np.ndarray:
    """``DCMotor._clip_effort``: four-quadrant torque-speed saturation."""
    qd_c = np.clip(qd, -params.vel_at_effort_lim, params.vel_at_effort_lim)
    top = params.saturation_effort * (1.0 - qd_c / params.velocity_limit)
    bottom = params.saturation_effort * (-1.0 - qd_c / params.velocity_limit)
    max_effort = np.minimum(top, params.effort_limit)
    min_effort = np.maximum(bottom, -params.effort_limit)
    return np.minimum(np.maximum(effort, min_effort), max_effort)


class TargetDelay:
    """Delay joint targets by a fixed number of physics steps.

    Mirrors Isaac Lab ``DelayBuffer`` (``isaaclab/utils/buffers/delay_buffer.py``
    over ``circular_buffer.py``) as used by
    :class:`phoenix.sim_env.delayed_dc_motor.DelayedDCMotor`: each physics step
    the newest target is appended and the one ``lag`` steps old is returned. On
    the first append after reset every slot is filled with that first value
    (``CircularBuffer.append`` ``is_first_push``), so the first ``lag`` steps
    return the first target, not the reset pose.
    """

    def __init__(self, lag: int) -> None:
        if lag < 0:
            raise ValueError("lag must be >= 0")
        self.lag = int(lag)
        self._buf: list[np.ndarray] = []

    def reset(self) -> None:
        self._buf = []

    def __call__(self, target: np.ndarray) -> np.ndarray:
        value = np.asarray(target, dtype=np.float64).copy()
        if not self._buf:
            self._buf = [value.copy() for _ in range(self.lag + 1)]
        else:
            self._buf.append(value)
            self._buf.pop(0)
        return self._buf[0]


__all__ = [
    "DCMotorParams",
    "GO2_DC_MOTOR",
    "TargetDelay",
    "clip_dc_motor_effort",
    "dc_motor_torque",
]
