"""Bridge a :class:`FailureCurriculum` into the env's reset lifecycle.

Isaac Lab's ``ManagerBasedRLEnv._reset_idx(env_ids)`` is called whenever
the env decides to reset a subset of parallel envs. We wrap it so that,
after the normal reset has run (terrain pose, joint positions, command),
a curriculum-selected subset of envs is *re-overridden* with the
snapshot captured in a real-world failure parquet.

Keeping the bridge as a wrapper instead of an ``EventTermCfg`` avoids
poking Isaac Lab's configclass system and means it can be removed
simply by not calling :func:`install`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from phoenix.replay.trajectory_reader import InitialState, load_initial_state

from .curriculum import FailureCurriculum

logger = logging.getLogger("phoenix.adaptation.reset_bridge")


class _InitialStateCache:
    """Load failure-parquet initial states once, reuse across resets."""

    def __init__(self, paths: list[Path]) -> None:
        self._paths = paths
        self._cache: dict[int, InitialState] = {}

    def get(self, pool_idx: int) -> InitialState:
        if pool_idx not in self._cache:
            self._cache[pool_idx] = load_initial_state(self._paths[pool_idx], row=0)
        return self._cache[pool_idx]


def install(env: Any, curriculum: FailureCurriculum) -> None:
    """Monkey-patch ``env._reset_idx`` so curriculum assignments take effect."""
    if curriculum.pool.empty() or curriculum.failure_fraction <= 0.0:
        logger.info("Curriculum is empty or inactive; skipping reset bridge.")
        return

    import torch

    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    original_reset_idx = unwrapped._reset_idx
    cache = _InitialStateCache(list(curriculum.pool.paths))
    device = unwrapped.device

    def _patched_reset_idx(env_ids):
        original_reset_idx(env_ids)
        if env_ids is None or len(env_ids) == 0:
            return
        assignment = curriculum.assign(
            int(env_ids.shape[0]) if hasattr(env_ids, "shape") else len(env_ids)
        )
        if (assignment < 0).all():
            return

        robot = unwrapped.scene["robot"]
        env_origins = (
            unwrapped.scene.env_origins[env_ids]
            if hasattr(unwrapped.scene, "env_origins")
            else None
        )

        for local_idx, pool_idx in enumerate(assignment):
            if pool_idx < 0:
                continue
            state = cache.get(int(pool_idx))
            global_env_id = int(env_ids[local_idx])

            pos = torch.as_tensor(state.base_pos, device=device, dtype=torch.float32)
            if env_origins is not None:
                pos = pos + env_origins[local_idx]
            # Parquet stores quat as (x,y,z,w) to match ROS conventions; Isaac
            # Lab's write_root_pose_to_sim expects (w,x,y,z), so roll by 1.
            quat_xyzw = torch.as_tensor(state.base_quat, device=device, dtype=torch.float32)
            quat = torch.roll(quat_xyzw, shifts=1, dims=-1)
            jpos = torch.as_tensor(state.joint_pos, device=device, dtype=torch.float32)
            jvel = torch.as_tensor(state.joint_vel, device=device, dtype=torch.float32)

            robot.write_root_pose_to_sim(
                torch.cat([pos, quat], dim=-1).unsqueeze(0),
                env_ids=torch.as_tensor([global_env_id], device=device, dtype=torch.int64),
            )
            robot.write_joint_state_to_sim(
                jpos.unsqueeze(0),
                jvel.unsqueeze(0),
                env_ids=torch.as_tensor([global_env_id], device=device, dtype=torch.int64),
            )

        # Announce the fraction that came from the curriculum so it shows up in logs.
        n_failure = int((assignment >= 0).sum())
        logger.debug(
            "Curriculum reseeded %d/%d envs from failure parquets", n_failure, len(assignment)
        )

    unwrapped._reset_idx = _patched_reset_idx
    unwrapped.phoenix_curriculum = curriculum
    logger.info(
        "Reset bridge installed: %d failure trajectories, %.1f%% failure fraction.",
        len(curriculum.pool),
        100.0 * curriculum.failure_fraction,
    )


# Keep public API minimal.
__all__ = ["install"]


# Re-export InitialState so `from phoenix.adaptation.reset_bridge import InitialState`
# works without pulling in the replay module name.
_ = np  # silence unused-import warning in type-stub consumers
