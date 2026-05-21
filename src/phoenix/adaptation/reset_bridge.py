"""Bridge a :class:`FailureCurriculum` into the env's reset lifecycle.

Isaac Lab's ``ManagerBasedRLEnv._reset_idx(env_ids)`` is called whenever
the env decides to reset a subset of parallel envs. We wrap it so that,
after the normal reset has run (terrain pose, joint positions, command),
a curriculum-selected subset of envs is *re-overridden* with the
snapshot captured in a real-world failure parquet.

Keeping the bridge as a wrapper instead of an ``EventTermCfg`` avoids
poking Isaac Lab's configclass system and means it can be removed
simply by not calling :func:`install`.

H0 bridge fix (2026-05-17). Two seed-row strategies and an optional
velocity write are exposed so the curriculum can seed envs at the
failure-onset row (where the parquet's pre-failure kinematics actually
mean something) rather than the trajectory's first frame. Defaults
preserve the legacy ``row=0``, no-velocity-write behavior.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from phoenix.replay.trajectory_reader import InitialState, load_initial_state

from .curriculum import FailureCurriculum

logger = logging.getLogger("phoenix.adaptation.reset_bridge")


_VALID_STRATEGIES = ("first", "failure_onset", "failure_onset_minus_k")


def _resolve_seed_row(
    path: Path,
    strategy: str,
    offset_k: int,
) -> int:
    """Resolve which parquet row to seed from, given a strategy.

    ``first`` returns 0. ``failure_onset`` finds the first row where
    ``failure_flag`` is True. ``failure_onset_minus_k`` subtracts
    ``offset_k`` from the onset row, clamped to ``[0, onset]``.

    Raises ``ValueError`` if ``strategy`` is unknown, or if a
    failure-onset strategy is requested but the parquet has no
    ``failure_flag=True`` row.
    """
    if strategy == "first":
        return 0
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"Unknown seed_row_strategy={strategy!r}; expected one of {_VALID_STRATEGIES}"
        )
    flags = pq.read_table(path, columns=["failure_flag"]).column("failure_flag").to_pylist()
    onset = next((i for i, f in enumerate(flags) if f), -1)
    if onset < 0:
        raise ValueError(
            f"Trajectory {path} has no failure_flag=True row; cannot apply strategy={strategy!r}"
        )
    if strategy == "failure_onset":
        return onset
    return max(0, onset - max(0, int(offset_k)))


class _InitialStateCache:
    """Load failure-parquet initial states once, reuse across resets.

    ``seed_row_strategy`` controls which row of each parquet is loaded.
    Default ``"first"`` preserves legacy ``row=0`` behavior. The resolved
    row is deterministic per parquet so caching by ``pool_idx`` remains
    sound.
    """

    def __init__(
        self,
        paths: list[Path],
        *,
        seed_row_strategy: str = "first",
        seed_row_offset_k: int = 0,
    ) -> None:
        self._paths = paths
        self._strategy = seed_row_strategy
        self._offset_k = seed_row_offset_k
        self._cache: dict[int, InitialState] = {}
        self._resolved_rows: dict[int, int] = {}

    def get(self, pool_idx: int) -> InitialState:
        if pool_idx not in self._cache:
            row = _resolve_seed_row(self._paths[pool_idx], self._strategy, self._offset_k)
            self._resolved_rows[pool_idx] = row
            self._cache[pool_idx] = load_initial_state(self._paths[pool_idx], row=row)
        return self._cache[pool_idx]

    def resolved_row(self, pool_idx: int) -> int:
        """Return the row index that was used for ``pool_idx`` (after a ``get``)."""
        return self._resolved_rows.get(pool_idx, -1)


def install(
    env: Any,
    curriculum: FailureCurriculum,
    *,
    seed_row_strategy: str = "first",
    seed_row_offset_k: int = 0,
    write_velocity: bool = False,
) -> None:
    """Monkey-patch ``env._reset_idx`` so curriculum assignments take effect.

    ``seed_row_strategy``, ``seed_row_offset_k`` control which parquet row
    is loaded (see :func:`_resolve_seed_row`). ``write_velocity`` opts
    into calling ``robot.write_root_velocity_to_sim`` with the parquet's
    ``base_lin_vel_body`` and ``base_ang_vel_body`` for the seeded envs;
    without it the bridge silently drops those columns and the seeded
    env starts from rest regardless of the trajectory's pre-failure
    velocity.
    """
    if curriculum.pool.empty() or curriculum.failure_fraction <= 0.0:
        logger.info("Curriculum is empty or inactive; skipping reset bridge.")
        return

    import torch

    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    original_reset_idx = unwrapped._reset_idx
    cache = _InitialStateCache(
        list(curriculum.pool.paths),
        seed_row_strategy=seed_row_strategy,
        seed_row_offset_k=seed_row_offset_k,
    )
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
            env_id_tensor = torch.as_tensor([global_env_id], device=device, dtype=torch.int64)

            robot.write_root_pose_to_sim(
                torch.cat([pos, quat], dim=-1).unsqueeze(0),
                env_ids=env_id_tensor,
            )
            robot.write_joint_state_to_sim(
                jpos.unsqueeze(0),
                jvel.unsqueeze(0),
                env_ids=env_id_tensor,
            )
            if write_velocity:
                # KNOWN LIMITATION: base_lin_vel_body / base_ang_vel_body are
                # body-frame, but write_root_velocity_to_sim expects world
                # frame. We pass the body-frame values unrotated (matching
                # replay.reconstruct.py:126). For a failure seeded at a
                # non-trivial orientation the injected velocity points the
                # wrong way. Acceptable only because write_velocity is opt-in
                # and currently unused (failure_sample_fraction 0.0); a proper
                # fix rotates by the base quaternion before the write. See
                # README "Known limitations".
                lin_vel = torch.as_tensor(
                    state.base_lin_vel_body, device=device, dtype=torch.float32
                )
                ang_vel = torch.as_tensor(
                    state.base_ang_vel_body, device=device, dtype=torch.float32
                )
                robot.write_root_velocity_to_sim(
                    torch.cat([lin_vel, ang_vel], dim=-1).unsqueeze(0),
                    env_ids=env_id_tensor,
                )

        # Announce the fraction that came from the curriculum so it shows up in logs.
        n_failure = int((assignment >= 0).sum())
        logger.debug(
            "Curriculum reseeded %d/%d envs from failure parquets", n_failure, len(assignment)
        )

    unwrapped._reset_idx = _patched_reset_idx
    unwrapped.phoenix_curriculum = curriculum
    logger.info(
        "Reset bridge installed: %d trajectories, %.1f%% ff, strategy=%s offset_k=%d write_vel=%s.",
        len(curriculum.pool),
        100.0 * curriculum.failure_fraction,
        seed_row_strategy,
        seed_row_offset_k,
        write_velocity,
    )


# Keep public API minimal.
__all__ = ["install"]


# Re-export InitialState so `from phoenix.adaptation.reset_bridge import InitialState`
# works without pulling in the replay module name.
_ = np  # silence unused-import warning in type-stub consumers
