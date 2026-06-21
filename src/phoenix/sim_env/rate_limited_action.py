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

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        current_q = wp.to_torch(self._asset.data.joint_pos)[:, self._joint_ids]
        self._processed_actions = rate_limit_targets(
            self._processed_actions, current_q, self.cfg.max_delta_per_step
        )


@configclass
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Cfg for :class:`RateLimitedJointPositionAction`.

    ``max_delta_per_step`` defaults to the canonical deploy cap so sim and the
    Jetson bridge share one value by construction.
    """

    class_type: type = RateLimitedJointPositionAction
    max_delta_per_step: float = MAX_DELTA_PER_STEP_RAD
