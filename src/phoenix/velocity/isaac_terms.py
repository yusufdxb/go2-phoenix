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
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import ManagerTermBase
from isaaclab.utils import configclass

from .contract import ACTION_SCALE, ACTOR_OBS_DIM
from .curriculum import CommandCurriculum
from .mdp import (
    as_torch,
    episode_mean_kernel_math,
    episode_survival_kernel_math,
    slew_clip_fraction_math,
)
from .spec import CRITIC_OBS_DIM, joint_order_problems, spec_from_dict

#: Attribute on the env instance where the live curriculum object is exposed to
#: the training script (manifest writing at every checkpoint save).
ENV_CURRICULUM_ATTR = "phoenix_velocity_curriculum"
#: Bool tensor on the env: envs the cold_start reset event just put in the
#: deployment start state; consumed by ColdStartVelocityCommand.reset.
ENV_COLD_START_ATTR = "phoenix_cold_start_pending"
#: Startup telemetry families (cold-start command direction) and window: a cold
#: start "started" if its mean velocity on the commanded axis over
#: STARTUP_WINDOW_S after the command switch is >= 50% of the command.
STARTUP_FAMILIES = ("fwd", "back", "lat", "yaw")
STARTUP_WINDOW_S = (1.0, 2.0)
STARTUP_RING = 512


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
            tracking_std=float(
                spec.curriculum.gate_kernel_std or spec.reward("track_lin_vel_xy_exp").params["std"]
            ),
            num_envs=env.num_envs,
        )
        self._apply_ranges(env)
        setattr(env, ENV_CURRICULUM_ATTR, self.curriculum)
        self._gate_std = spec.curriculum.gate_kernel_std
        self._last = {
            "lin_score": 0.0,
            "yaw_score": 0.0,
            "lin_kernel": 0.0,
            "termination_rate": 0.0,
        }

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
            gate_acc = getattr(
                env.command_manager.get_term(command_name), "gate_lin_kernel_sum", None
            )
            if self._gate_std and gate_acc is not None:
                # kernel*dt summed at the gate std (weight 1): same math as the reward path
                lin_src, lin_w = gate_acc[ids], 1.0
            else:
                lin_src, lin_w = rm._episode_sums[lin_term][ids], rm.get_term_cfg(lin_term).weight
            lin = episode_mean_kernel_math(lin_src, lin_w, env.step_dt, steps)
            yaw = episode_mean_kernel_math(
                rm._episode_sums[yaw_term][ids],
                rm.get_term_cfg(yaw_term).weight,
                env.step_dt,
                steps,
            )
            lin_surv = episode_survival_kernel_math(lin_src, lin_w, env.max_episode_length_s)
            terminated = env.termination_manager.terminated[ids]
            decision = self.curriculum.observe(
                int(ids.numel()),
                int(terminated.sum().item()),
                float(lin.sum()),
                float(yaw.sum()),
                float(lin_surv.sum()),
            )
            if decision.evaluated:
                self._last = {
                    "lin_score": decision.lin_score,
                    "yaw_score": decision.yaw_score,
                    "lin_kernel": decision.lin_survival_kernel,
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
            "lin_vel_y_min": r.lin_vel_y[0],
            "lin_vel_y_max": r.lin_vel_y[1],
            "ang_vel_z_min": r.ang_vel_z[0],
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
        cmd_term_obj = env.command_manager.get_term("base_velocity")
        cold_ep = getattr(cmd_term_obj, "cold_episode", None)
        cold_pend = getattr(cmd_term_obj, "cold_pending", None)
        absa = a.abs().flatten()
        k = max(1, int(math.ceil(0.99 * absa.numel())))
        return {
            "slew_clip_frac": float(slew_clip_fraction_math(a, prev, scale, slew_clip_rad)),
            "target_outside_soft": float(outside.float().mean()),
            "action_abs_mean": float(absa.mean()),
            "action_abs_p99": float(absa.kthvalue(k).values),
            "lin_err": float(torch.linalg.norm(cmd[:, :2] - v[:, :2], dim=1).mean()),
            "yaw_err": float((cmd[:, 2] - w[:, 2]).abs().mean()),
            # fraction of envs whose current episode began cold / still settling
            "cold_start_episode_frac": (
                float(cold_ep.float().mean()) if cold_ep is not None else 0.0
            ),
            "cold_start_settling_frac": (
                float(cold_pend.float().mean()) if cold_pend is not None else 0.0
            ),
            **(cmd_term_obj.startup_rates() if hasattr(cmd_term_obj, "startup_rates") else {}),
        }


__all__ = [
    "ENV_COLD_START_ATTR",
    "ENV_CURRICULUM_ATTR",
    "ColdStartVelocityCommand",
    "ColdStartVelocityCommandCfg",
    "VelocityCommandCurriculum",
    "VelocityTelemetry",
    "assert_joint_contract",
    "cold_start_reset",
    "restore_curriculum_state",
    "sample_cold_start_commands",
    "write_cold_state",
]


# ----------------------------------------------------------------- cold start


def restore_curriculum_state(
    env: Any, manifest_state: dict[str, Any], command_name: str = "base_velocity"
) -> None:
    """Put a resumed run's command curriculum back where it stopped.

    ``manifest_state`` is the ``curriculum_final_state.json`` the run wrote
    (CommandCurriculum.manifest_dict). Restores level, counters and history,
    applies the level's ranges to the command term and resamples every env's
    command under them (the env was built at level 0).
    """
    cur = getattr(env, ENV_CURRICULUM_ATTR)
    cur.load_state_dict(
        {
            "level": manifest_state["level"],
            "evaluations": manifest_state.get("evaluations", 0),
            "total_episodes": manifest_state.get("total_episodes", 0),
            "history": manifest_state.get("level_history", []),
        }
    )
    term = env.command_manager.get_term(command_name)
    r = cur.ranges
    term.cfg.ranges.lin_vel_x = tuple(r.lin_vel_x)
    term.cfg.ranges.lin_vel_y = tuple(r.lin_vel_y)
    term.cfg.ranges.ang_vel_z = tuple(r.ang_vel_z)
    term._resample(torch.arange(env.num_envs, device=env.device))


def write_cold_state(env: Any, env_ids: torch.Tensor, asset_name: str = "robot") -> None:
    """Deployment start state: default joint pose exactly, joint and root velocity zero.

    Root pose (position, yaw) is left as the preceding reset term set it; roll and
    pitch are zero there. Shared by the ``cold_start`` reset event and the
    evaluation, so both mean the same state.
    """
    asset = env.scene[asset_name]
    q0 = as_torch(asset.data.default_joint_pos)[env_ids].clone()
    asset.write_joint_position_to_sim_index(position=q0, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(len(env_ids), 6, device=q0.device), env_ids=env_ids
    )


def cold_start_reset(env: Any, env_ids: Any, fraction: float, asset_name: str = "robot") -> None:
    """Reset event (after reset_base / reset_robot_joints): a ``fraction`` of the
    resetting envs start cold (:func:`write_cold_state`) and are flagged for the
    command term's stand -> low-command schedule."""
    ids = _env_ids_tensor(env, env_ids)
    mask = getattr(env, ENV_COLD_START_ATTR, None)
    if mask is None:
        mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        setattr(env, ENV_COLD_START_ATTR, mask)
    mask[ids] = False
    cold = ids[torch.rand(ids.numel(), device=env.device) < float(fraction)]
    if cold.numel():
        write_cold_state(env, cold, asset_name)
        mask[cold] = True


def sample_cold_start_commands(
    families: torch.Tensor, n: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """(n, 3) commands: pick a family by weight, draw its axis value uniformly in [lo, hi].

    ``families`` is (F, 4) rows ``(axis, lo, hi, weight)`` (``ColdStartSpec.families``).
    """
    dev = families.device
    fam = torch.multinomial(families[:, 3], n, replacement=True, generator=generator)
    axis = families[fam, 0].long()
    lo, hi = families[fam, 1], families[fam, 2]
    u = torch.rand(n, device=dev, generator=generator)
    cmd = torch.zeros(n, 3, device=dev)
    cmd[torch.arange(n, device=dev), axis] = lo + (hi - lo) * u
    return cmd


class ColdStartVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand plus the cold-start schedule.

    For envs the ``cold_start`` event flagged at reset: zero command for a
    ``settle_s`` draw, then one low command from the families, held for a
    normal resampling interval; then ordinary sampling. Other envs behave
    exactly like UniformVelocityCommand.
    """

    cfg: ColdStartVelocityCommandCfg

    def __init__(self, cfg: ColdStartVelocityCommandCfg, env: Any):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.cold_episode = torch.zeros(n, dtype=torch.bool, device=dev)
        self.cold_pending = torch.zeros(n, dtype=torch.bool, device=dev)
        self.cold_countdown = torch.zeros(n, device=dev)
        self.cold_command = torch.zeros(n, 3, device=dev)
        self._families = torch.tensor(cfg.cold_start_families, dtype=torch.float, device=dev)
        # curriculum-gate kernel accumulator (sum of exp(-err^2/std^2) * dt this episode)
        self.gate_lin_kernel_sum = torch.zeros(n, device=dev)
        # startup telemetry: did a cold start actually start moving?
        self.startup_active = torch.zeros(n, dtype=torch.bool, device=dev)
        self.startup_t = torch.zeros(n, device=dev)
        self.startup_ratio_sum = torch.zeros(n, device=dev)
        self.startup_count = torch.zeros(n, device=dev)
        self.startup_family = torch.zeros(n, dtype=torch.long, device=dev)
        self._ring = torch.full((len(STARTUP_FAMILIES), STARTUP_RING), float("nan"), device=dev)
        self._ring_ptr = [0] * len(STARTUP_FAMILIES)
        self.startup_dropped = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)  # resamples: clears any pending schedule
        ids = _env_ids_tensor(self._env, env_ids)
        self.cold_episode[ids] = False
        self.gate_lin_kernel_sum[ids] = 0.0
        self.startup_dropped += int(self.startup_active[ids].sum())
        self.startup_active[ids] = False
        mask = getattr(self._env, ENV_COLD_START_ATTR, None)
        if mask is not None:
            cold = ids[mask[ids]]
            mask[ids] = False
            if cold.numel():
                lo, hi = self.cfg.cold_start_settle_s
                settle = torch.empty(cold.numel(), device=self.device).uniform_(lo, hi)
                self.cold_episode[cold] = True
                self.cold_pending[cold] = True
                self.cold_countdown[cold] = settle
                self.cold_command[cold] = sample_cold_start_commands(self._families, cold.numel())
                self.vel_command_b[cold] = 0.0
                self.is_standing_env[cold] = True
                # hold the low command for a full normal interval after the switch
                r_lo, r_hi = self.cfg.resampling_time_range
                self.time_left[cold] = settle + torch.empty_like(settle).uniform_(r_lo, r_hi)
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        self.cold_pending[env_ids] = False

    def _update_command(self):
        self.cold_countdown -= self._env.step_dt * self.cold_pending.float()
        switch = self.cold_pending & (self.cold_countdown <= 0.0)
        if bool(switch.any()):
            self.vel_command_b[switch] = self.cold_command[switch]
            self.is_standing_env[switch] = False
            self.cold_pending[switch] = False
            c = self.cold_command[switch]
            axis = c.abs().argmax(dim=1)
            fam = torch.where(axis == 0, torch.where(c[:, 0] > 0, 0, 1), axis + 1)
            self.startup_family[switch] = fam
            self.startup_active[switch] = True
            self.startup_t[switch] = 0.0
            self.startup_ratio_sum[switch] = 0.0
            self.startup_count[switch] = 0.0
        super()._update_command()

    def _update_metrics(self):
        super()._update_metrics()
        dt = self._env.step_dt
        vb = as_torch(self.robot.data.root_lin_vel_b)
        live = self._env.episode_length_buf > 0  # match the reward manager's episode sums
        if self.cfg.gate_kernel_std > 0.0:
            err2 = ((self.vel_command_b[:, :2] - vb[:, :2]) ** 2).sum(dim=1)
            self.gate_lin_kernel_sum += torch.exp(-err2 / self.cfg.gate_kernel_std**2) * dt * live
        act = self.startup_active
        if bool(act.any()):
            wz = as_torch(self.robot.data.root_ang_vel_b)[:, 2]
            vel = torch.stack([vb[:, 0], vb[:, 1], wz], dim=1)
            axis = torch.clamp(self.startup_family - 1, min=0)  # fwd/back -> 0, lat -> 1, yaw -> 2
            cmd_ax = self.vel_command_b.gather(1, axis[:, None]).squeeze(1)
            v_ax = vel.gather(1, axis[:, None]).squeeze(1)
            self.startup_t += dt * act
            in_win = (
                act
                & (self.startup_t >= STARTUP_WINDOW_S[0])
                & (self.startup_t < STARTUP_WINDOW_S[1])
            )
            safe = torch.where(cmd_ax.abs() > 1e-6, cmd_ax, torch.ones_like(cmd_ax))
            self.startup_ratio_sum += torch.where(in_win, v_ax / safe, torch.zeros_like(v_ax))
            self.startup_count += in_win.float()
            done = act & (self.startup_t >= STARTUP_WINDOW_S[1])
            if bool(done.any()):
                started = (
                    self.startup_ratio_sum[done] / self.startup_count[done].clamp(min=1.0)
                ) >= 0.5
                fams = self.startup_family[done]
                for f in range(len(STARTUP_FAMILIES)):
                    self._push(f, started[fams == f].float())
                self.startup_active[done] = False

    def _push(self, family: int, outcomes: torch.Tensor) -> None:
        outcomes = outcomes[-STARTUP_RING:]
        k = outcomes.numel()
        if k == 0:
            return
        idx = (self._ring_ptr[family] + torch.arange(k, device=self.device)) % STARTUP_RING
        self._ring[family, idx] = outcomes
        self._ring_ptr[family] = (self._ring_ptr[family] + k) % STARTUP_RING

    def startup_rates(self) -> dict[str, float]:
        """Share of the last STARTUP_RING classified cold starts per family that started."""
        out = {}
        for f, name in enumerate(STARTUP_FAMILIES):
            row = self._ring[f]
            valid = ~torch.isnan(row)
            out[f"startup_rate_{name}"] = (
                float(row[valid].mean()) if bool(valid.any()) else float("nan")
            )
        return out


@configclass
class ColdStartVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = ColdStartVelocityCommand
    cold_start_families: tuple = ()
    cold_start_settle_s: tuple[float, float] = (0.5, 1.5)
    #: > 0: accumulate the curriculum-gate lin kernel at this std (CurriculumSpec.gate_kernel_std).
    gate_kernel_std: float = 0.0
