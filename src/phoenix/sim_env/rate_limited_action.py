"""Rate-limited joint-position ActionTerm for the GO2 stand/walk policies.

This module imports Isaac Lab at load time, so it is imported LAZILY from
``go2_env_cfg.build_env_cfg`` (inside a sim context) and is never pulled into
the no-sim CI surface. The pure clamp it relies on lives in
:mod:`phoenix.sim_env.rate_limit` (CI-safe, unit-tested).

Why this exists: the deploy stack clips every joint target to
``measured_q ± MAX_DELTA_PER_STEP_RAD`` per control tick. Training previously
applied the raw target with no such cap, so the policy never experienced the
limiter until hardware. :class:`RateLimitedJointPositionAction` re-applies the
exact deploy limiter inside the env, once per policy step (in ``process_actions``,
against the joint position at step start), so the clipped target is then held
across all decimation substeps, mirroring how the Jetson bridge holds one
clipped command per 50 Hz tick.
"""

from __future__ import annotations

import torch
import warp as wp
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from phoenix.sim_env.rate_limit import MAX_DELTA_PER_STEP_RAD, rate_limit_targets


class RateLimitedJointPositionAction(JointPositionAction):
    """``JointPositionAction`` that caps the per-step joint-target slew.

    ``process_actions`` first computes the standard target
    (``raw_action * scale + default_offset``) via the parent, then clips it to
    ``current_joint_pos ± cfg.max_delta_per_step``. Because the clip happens once
    per policy step and the result is stored in ``self._processed_actions``, the
    inherited ``apply_actions`` sends the SAME clipped target on every decimation
    substep, matching the deploy bridge's hold-one-command-per-tick behaviour.
    """

    cfg: RateLimitedJointPositionActionCfg

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        # prev_command mode needs a per-env memory of the last applied target.
        # Reset it to the default pose (use_default_offset) for the reset envs;
        # the robot also resets near the default pose, so the first post-reset
        # command starts on-distribution.
        if self.cfg.clip_mode != "prev_command":
            return
        if getattr(self, "_prev_target", None) is None:
            self._prev_target = self._offset.detach().clone()
        elif env_ids is None:
            self._prev_target[:] = self._offset
        else:
            self._prev_target[env_ids] = self._offset[env_ids]

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        if self.cfg.clip_mode == "prev_command":
            # True slew-rate limiter: clip the delta vs the PREVIOUS COMMAND, not
            # the measured state. Decouples the command from sensor noise and lets
            # the command stay anchored to the intended target under disturbance,
            # at the cost of allowing the command to drift from measured q.
            if getattr(self, "_prev_target", None) is None:
                self._prev_target = self._offset.detach().clone()
            self._processed_actions = rate_limit_targets(
                self._processed_actions, self._prev_target, self.cfg.max_delta_per_step
            )
            self._prev_target = self._processed_actions.detach().clone()
            return
        # Default measured_q mode: clip vs the joint position (matches the deploy
        # bridge). Optional clip_ref_noise models the deploy noisy-encoder clip.
        current_q = wp.to_torch(self._asset.data.joint_pos)[:, self._joint_ids]
        if self.cfg.clip_ref_noise > 0.0:
            current_q = current_q + (
                (torch.rand_like(current_q) * 2.0 - 1.0) * self.cfg.clip_ref_noise
            )
        self._processed_actions = rate_limit_targets(
            self._processed_actions, current_q, self.cfg.max_delta_per_step
        )


@configclass
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Cfg for :class:`RateLimitedJointPositionAction`.

    ``max_delta_per_step`` defaults to the canonical deploy cap so sim and the
    Jetson bridge share one value by construction.

    ``clip_mode``:
      * ``"measured_q"`` (default) clips the target to ``measured_q ± max_delta``,
        matching the current Jetson bridge. ``clip_ref_noise`` (half-width, rad)
        adds uniform noise to that reference to model the noisy encoder.
      * ``"prev_command"`` clips the per-step delta vs the previous applied target
        (a true slew-rate limiter). ``clip_ref_noise`` is ignored in this mode.
    """

    class_type: type = RateLimitedJointPositionAction
    max_delta_per_step: float = MAX_DELTA_PER_STEP_RAD
    clip_ref_noise: float = 0.0
    clip_mode: str = "measured_q"
