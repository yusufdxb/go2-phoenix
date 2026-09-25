"""Isaac Lab manager terms that need Isaac base classes (Isaac-only module).

Imports Isaac Lab at module load, so it is only imported from
:mod:`phoenix.velocity.env_cfg` (itself Isaac-only) and never from CI code.
All decisions and math live in pure modules (:mod:`.curriculum`, :mod:`.mdp`,
:mod:`.spec`); this file only moves data between the env and those modules.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from isaaclab.managers import ManagerTermBase

from .contract import ACTION_SCALE, ACTOR_OBS_DIM
from .curriculum import CommandCurriculum
from .mdp import as_torch, episode_mean_kernel_math, slew_clip_fraction_math
from .spec import CRITIC_OBS_DIM, joint_order_problems, spec_from_dict

#: Attribute on the env instance where the live curriculum object is exposed to
#: the training script (manifest writing at every checkpoint save).
ENV_CURRICULUM_ATTR = "phoenix_velocity_curriculum"


def _env_ids_tensor(env: Any, env_ids: Any) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(env.device)
    return torch.as_tensor(list(env_ids), device=env.device)


def assert_joint_contract(
    env: Any, env_ids: Any, asset_cfg: Any, action_name: str, critic_group: str = "critic"
) -> None:
    """Startup event: fail the env build unless it matches the 45-D contract.

    Checks the articulation joint order and default pose against
    ``contract.JOINT_ORDER`` / ``DEFAULT_JOINT_POS`` (the joint_pos_rel /
    joint_vel_rel observation terms and the action term all index joints in the
    articulation order), the action term's own joint list and scale, and the
    actor / critic observation dimensions.
    """
    asset = env.scene[asset_cfg.name]
    names = list(asset.joint_names)
    default_q = as_torch(asset.data.default_joint_pos)[0].tolist()
    problems = joint_order_problems(names, default_q)

    term = env.action_manager.get_term(action_name)
    term_names = list(getattr(term, "_joint_names", []))
    if term_names != names:
        problems.append(f"action term joints {term_names} != articulation order {names}")
    scale = term.cfg.scale
    if not isinstance(scale, (int, float)) or abs(float(scale) - ACTION_SCALE) > 1e-12:
        problems.append(f"action term scale {scale!r} != contract {ACTION_SCALE}")
    if not getattr(term.cfg, "use_default_offset", False):
        problems.append("action term does not offset by the default pose")

    dims = env.observation_manager.group_obs_dim
    if tuple(dims.get("policy", ())) != (ACTOR_OBS_DIM,):
        problems.append(f"policy obs dim {dims.get('policy')} != ({ACTOR_OBS_DIM},)")
    if tuple(dims.get(critic_group, ())) != (CRITIC_OBS_DIM,):
        problems.append(f"critic obs dim {dims.get(critic_group)} != ({CRITIC_OBS_DIM},)")
    if problems:
        raise RuntimeError("PhoenixVelocity env violates the contract:\n  " + "\n  ".join(problems))
    print(
        f"[phoenix.velocity] contract check PASSED: joint order {names}, "
        f"policy obs {dims['policy']}, critic obs {dims[critic_group]}, action scale {scale}",
        flush=True,
    )


class VelocityCommandCurriculum(ManagerTermBase):
    """Curriculum term: feeds finished episodes to :class:`CommandCurriculum`.

    Called by Isaac's CurriculumManager at every reset, BEFORE the reward
    manager clears its episode sums, so ``_episode_sums`` still holds the
    finished episodes' weighted tracking sums and ``episode_length_buf`` their
    lengths.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        spec = spec_from_dict(cfg.params["spec"])
        self.command_name: str = cfg.params["command_name"]
        self.curriculum = CommandCurriculum.from_spec(
            spec.commands,
            spec.curriculum,
            tracking_std=float(spec.reward("track_lin_vel_xy_exp").params["std"]),
            num_envs=env.num_envs,
        )
        self._apply_ranges(env)
        setattr(env, ENV_CURRICULUM_ATTR, self.curriculum)
        self._last = {"lin_score": 0.0, "yaw_score": 0.0, "termination_rate": 0.0}

    def _apply_ranges(self, env: Any) -> None:
        ranges = self.curriculum.ranges
        cfg_ranges = env.command_manager.get_term(self.command_name).cfg.ranges
        cfg_ranges.lin_vel_x = tuple(ranges.lin_vel_x)
        cfg_ranges.lin_vel_y = tuple(ranges.lin_vel_y)
        cfg_ranges.ang_vel_z = tuple(ranges.ang_vel_z)

    def __call__(
        self,
        env: Any,
        env_ids: Sequence[int] | torch.Tensor | slice,
        command_name: str,
        spec: dict,
        lin_term: str = "track_lin_vel_xy_exp",
        yaw_term: str = "track_ang_vel_z_exp",
    ) -> dict[str, float]:
        ids = _env_ids_tensor(env, env_ids)
        steps = env.episode_length_buf[ids]
        done = steps > 0  # the initial reset of never-stepped envs is not an episode
        if bool(done.any()):
            ids, steps = ids[done], steps[done]
            rm = env.reward_manager
            lin = episode_mean_kernel_math(
                rm._episode_sums[lin_term][ids], rm.get_term_cfg(lin_term).weight, env.step_dt, steps
            )
            yaw = episode_mean_kernel_math(
                rm._episode_sums[yaw_term][ids], rm.get_term_cfg(yaw_term).weight, env.step_dt, steps
            )
            terminated = env.termination_manager.terminated[ids]
            decision = self.curriculum.observe(
                int(ids.numel()), int(terminated.sum().item()), float(lin.sum()), float(yaw.sum())
            )
            if decision.evaluated:
                self._last = {
                    "lin_score": decision.lin_score,
                    "yaw_score": decision.yaw_score,
                    "termination_rate": decision.termination_rate,
                }
            if decision.expanded:
                self._apply_ranges(env)
                print(
                    f"[phoenix.velocity] curriculum level {decision.level}/"
                    f"{self.curriculum.max_level}: {self.curriculum.ranges.to_dict()}",
                    flush=True,
                )
        r = self.curriculum.ranges
        return {
            "level": float(self.curriculum.level),
            "lin_vel_x_min": r.lin_vel_x[0],
            "lin_vel_x_max": r.lin_vel_x[1],
            "lin_vel_y_max": r.lin_vel_y[1],
            "ang_vel_z_max": r.ang_vel_z[1],
            **{k: float(v) for k, v in self._last.items()},
        }


class VelocityTelemetry(ManagerTermBase):
    """Logging-only curriculum term (changes nothing): per-step health of the policy.

    Reported under ``Curriculum/telemetry/*`` in tensorboard:
      slew_clip_frac      fraction of motor-steps whose joint-target step exceeds
                          the deploy slew clip (0.175 rad)
      target_outside_soft fraction of joint targets outside the soft joint limits
      action_abs_mean / action_abs_p99   raw action magnitude
      lin_err / yaw_err   instantaneous |cmd - measured| (m/s, rad/s), all envs
    """

    def __call__(
        self, env: Any, env_ids: Any, action_name: str, asset_cfg: Any, slew_clip_rad: float
    ) -> dict[str, float]:
        am = env.action_manager
        term = am.get_term(action_name)
        scale = float(term.cfg.scale)
        a, prev = am.action, am.prev_action
        asset = env.scene[asset_cfg.name]
        q0 = as_torch(asset.data.default_joint_pos)
        limits = as_torch(asset.data.soft_joint_pos_limits)
        target = q0 + scale * a
        outside = (target < limits[..., 0]) | (target > limits[..., 1])
        cmd = env.command_manager.get_command("base_velocity")
        v = as_torch(asset.data.root_lin_vel_b)
        w = as_torch(asset.data.root_ang_vel_b)
        absa = a.abs().flatten()
        k = max(1, int(math.ceil(0.99 * absa.numel())))
        return {
            "slew_clip_frac": float(slew_clip_fraction_math(a, prev, scale, slew_clip_rad)),
            "target_outside_soft": float(outside.float().mean()),
            "action_abs_mean": float(absa.mean()),
            "action_abs_p99": float(absa.kthvalue(k).values),
            "lin_err": float(torch.linalg.norm(cmd[:, :2] - v[:, :2], dim=1).mean()),
            "yaw_err": float((cmd[:, 2] - w[:, 2]).abs().mean()),
        }


__all__ = [
    "ENV_CURRICULUM_ATTR",
    "VelocityCommandCurriculum",
    "VelocityTelemetry",
    "assert_joint_contract",
]
