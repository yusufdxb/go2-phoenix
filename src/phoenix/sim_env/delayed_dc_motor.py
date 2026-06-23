"""DC-motor actuator with delayed command application (sim-to-real latency).

Isaac Lab ships ``DelayedPDActuator`` (extends ``IdealPDActuator``), but the Go2
uses an explicit ``DCMotor`` which has no delayed variant. This mirrors the
``DelayedPDActuator`` ``DelayBuffer`` pattern on the ``DCMotor`` base so the
policy trains against the real robot's command latency. Until now the
``domain_randomization.actuator_latency_steps`` `[1,5]` range was stored but
never consumed (``go2_env_cfg.py`` no-op note), so the trained policy assumed
zero actuation delay — a known sim-to-real gap.

It delays ONLY the setpoints (positions/velocities/efforts) by a per-env random
number of physics steps drawn from ``[min_delay, max_delay]`` at each reset, then
defers to ``DCMotor.compute``. Stiffness and damping are untouched, so this
composes with the ``motor_strength_scale`` gain DR and does NOT re-introduce the
gain-zeroing failure mode that the explicit-actuator work hit before.

Sim-gated: imports Isaac Lab + torch at module load, so it is lazy-imported from
``go2_env_cfg.build_env_cfg`` and never pulled into the no-sim CI surface.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.actuators import DCMotor
from isaaclab.actuators.actuator_pd_cfg import DCMotorCfg
from isaaclab.utils import DelayBuffer, configclass

__all__ = ["DelayedDCMotor", "DelayedDCMotorCfg"]


class DelayedDCMotor(DCMotor):
    """``DCMotor`` whose command setpoints lag by a random number of physics steps."""

    cfg: DelayedDCMotorCfg

    def __init__(self, cfg: DelayedDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # Draw a fresh per-env lag in [min_delay, max_delay] physics steps.
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        for buf in (
            self.positions_delay_buffer,
            self.velocities_delay_buffer,
            self.efforts_delay_buffer,
        ):
            buf.set_time_lag(time_lags, env_ids)
            buf.reset(env_ids)

    def compute(self, control_action, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
        # Delay each setpoint that is actually present (position-controlled Go2
        # populates joint_positions; velocities/efforts are typically None).
        if control_action.joint_positions is not None:
            control_action.joint_positions = self.positions_delay_buffer.compute(
                control_action.joint_positions
            )
        if control_action.joint_velocities is not None:
            control_action.joint_velocities = self.velocities_delay_buffer.compute(
                control_action.joint_velocities
            )
        if control_action.joint_efforts is not None:
            control_action.joint_efforts = self.efforts_delay_buffer.compute(
                control_action.joint_efforts
            )
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedDCMotorCfg(DCMotorCfg):
    """Cfg for :class:`DelayedDCMotor`. ``min_delay``/``max_delay`` are physics steps."""

    class_type: type = DelayedDCMotor
    min_delay: int = 0
    max_delay: int = 0
