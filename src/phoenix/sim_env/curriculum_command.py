"""Isaac Lab command term for the recipe W3 symmetric curriculum (imports Isaac Lab).

Imported lazily by ``go2_env_cfg._apply_commands`` only when ``command.curriculum`` is
enabled. Samples exactly as ``UniformVelocityCommand`` (uniform over the configured
ranges), then multiplies the new command by the current stage factor, so every stage is
symmetric by construction. Per-step it accumulates each env's settled commanded and
achieved body-frame vx; at every resample (including episode reset) the finished segment
goes to :class:`phoenix.sim_env.command_curriculum.CurriculumState`, which decides
advancement with forward and backward tracking measured separately.
"""

from __future__ import annotations

import json
from dataclasses import fields

import torch
import warp as wp
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

from phoenix.sim_env.command_curriculum import CurriculumSpec, CurriculumState


class SymmetricCurriculumVelocityCommand(UniformVelocityCommand):
    cfg: SymmetricCurriculumVelocityCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        spec_kw = {f.name: getattr(cfg, "curriculum_" + f.name) for f in fields(CurriculumSpec)
                   if hasattr(cfg, "curriculum_" + f.name)}
        self.state = CurriculumState(CurriculumSpec(**spec_kw))
        n = self.num_envs
        self._steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self._sum_cmd = torch.zeros(n, device=self.device)
        self._sum_vx = torch.zeros(n, device=self.device)
        self._settled = torch.zeros(n, dtype=torch.long, device=self.device)
        for k in ("curriculum_stage", "curriculum_factor", "curriculum_fwd_ratio", "curriculum_bwd_ratio"):
            self.metrics[k] = torch.zeros(n, device=self.device)

    def _update_metrics(self):
        super()._update_metrics()
        self._steps += 1
        settled = self._steps > self.state.spec.settle_steps
        vx = wp.to_torch(self.robot.data.root_lin_vel_b)[:, 0]
        self._sum_cmd += torch.where(settled, self.vel_command_b[:, 0], 0.0)
        self._sum_vx += torch.where(settled, vx, 0.0)
        self._settled += settled.long()
        f, b = self.state.ratios()
        self.metrics["curriculum_stage"][:] = float(self.state.stage)
        self.metrics["curriculum_factor"][:] = self.state.factor
        self.metrics["curriculum_fwd_ratio"][:] = -1.0 if f is None else f
        self.metrics["curriculum_bwd_ratio"][:] = -1.0 if b is None else b

    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long) if not isinstance(env_ids, slice) \
            else torch.arange(self.num_envs, device=self.device)
        n_set = self._settled[ids]
        done = n_set > 0
        if bool(done.any()):
            cmd = (self._sum_cmd[ids][done] / n_set[done]).tolist()
            vx = (self._sum_vx[ids][done] / n_set[done]).tolist()
            ns = n_set[done].tolist()
            for c, v, k in zip(cmd, vx, ns, strict=True):
                if c != 0.0:
                    self.state.record_segment(c, v, k)
        for buf in (self._steps, self._sum_cmd, self._sum_vx, self._settled):
            buf[ids] = 0
        if self.state.maybe_advance(int(self._env.common_step_counter)):
            print("[phoenix curriculum] " + json.dumps(self.state.history[-1]), flush=True)
        super()._resample_command(env_ids)
        self.vel_command_b[ids] *= self.state.factor


@configclass
class SymmetricCurriculumVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = SymmetricCurriculumVelocityCommand
    curriculum_factors: tuple = (0.2, 0.4, 0.6, 1.0)
    curriculum_full_max_vx: float = 1.0
    curriculum_min_cmd_frac: float = 0.5
    curriculum_ratio_threshold: float = 0.7
    curriculum_window: int = 2000
    curriculum_min_segments_per_direction: int = 500
    curriculum_check_every_steps: int = 1200
    curriculum_min_stage_steps: int = 2400
    curriculum_max_stage_steps: int = 24000
    curriculum_settle_steps: int = 50
    curriculum_consecutive_checks: int = 2
