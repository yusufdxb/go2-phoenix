"""Tests for ``phoenix.adaptation.reset_bridge``.

Gating branches (empty pool, zero-fraction) are pure-numpy and run in
CI. The monkey-patch behaviour needs torch tensors; those tests skip
if torch isn't importable.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from phoenix.adaptation.curriculum import FailureCurriculum, TrajectoryPool
from phoenix.adaptation.reset_bridge import (
    _InitialStateCache,
    _resolve_seed_row,
    install,
)
from phoenix.real_world.trajectory_logger import TrajectoryLogger, TrajectoryStep


def _write_failure_parquet(path: Path) -> None:
    with TrajectoryLogger(path) as log:
        log.append(
            TrajectoryStep(
                step=0,
                timestamp_s=0.0,
                base_pos=np.asarray([1.0, 2.0, 0.4], dtype=np.float32),
                base_quat=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                base_lin_vel_body=np.zeros(3, dtype=np.float32),
                base_ang_vel_body=np.zeros(3, dtype=np.float32),
                joint_pos=np.arange(12, dtype=np.float32) * 0.1,
                joint_vel=np.zeros(12, dtype=np.float32),
                command_vel=np.asarray([0.5, 0.0, 0.0], dtype=np.float32),
                action=np.zeros(12, dtype=np.float32),
                contact_forces=np.ones(4, dtype=np.float32),
                failure_flag=True,
                failure_mode="attitude",
            )
        )


class _FakeRobot:
    def __init__(self) -> None:
        self.root_pose_calls: list[tuple] = []
        self.joint_state_calls: list[tuple] = []
        self.root_velocity_calls: list[tuple] = []

    def write_root_pose_to_sim(self, tensor, env_ids):  # noqa: ANN001
        self.root_pose_calls.append((tensor.clone(), env_ids.clone()))

    def write_joint_state_to_sim(self, jp, jv, env_ids):  # noqa: ANN001
        self.joint_state_calls.append((jp.clone(), jv.clone(), env_ids.clone()))

    def write_root_velocity_to_sim(self, tensor, env_ids):  # noqa: ANN001
        self.root_velocity_calls.append((tensor.clone(), env_ids.clone()))


class _FakeScene(dict):
    """Mapping-like scene with an env_origins tensor."""

    def __init__(self, robot: _FakeRobot, env_origins) -> None:  # noqa: ANN001
        super().__init__(robot=robot)
        self.env_origins = env_origins


def _fake_env(robot, env_origins, device):  # noqa: ANN001
    inner_calls: list[object] = []

    def _reset_idx(env_ids):
        inner_calls.append(env_ids)

    unwrapped = SimpleNamespace(
        scene=_FakeScene(robot, env_origins),
        device=device,
        _reset_idx=_reset_idx,
    )
    return SimpleNamespace(unwrapped=unwrapped), inner_calls, unwrapped


# -------------------- gating (no torch) ------------------------------------


def test_install_skips_when_pool_empty() -> None:
    curriculum = FailureCurriculum(TrajectoryPool(paths=[]), failure_fraction=0.3)
    env = SimpleNamespace(unwrapped=SimpleNamespace(_reset_idx=lambda ids: None))
    original = env.unwrapped._reset_idx
    install(env, curriculum)
    assert env.unwrapped._reset_idx is original
    assert not hasattr(env.unwrapped, "phoenix_curriculum")


def test_install_skips_when_failure_fraction_zero(tmp_path: Path) -> None:
    p = tmp_path / "f.parquet"
    _write_failure_parquet(p)
    curriculum = FailureCurriculum(TrajectoryPool(paths=[p]), failure_fraction=0.0)
    env = SimpleNamespace(unwrapped=SimpleNamespace(_reset_idx=lambda ids: None))
    original = env.unwrapped._reset_idx
    install(env, curriculum)
    assert env.unwrapped._reset_idx is original


# -------------------- behaviour (requires torch) ---------------------------


def test_patched_reset_calls_original_and_rewrites_pose(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    p = tmp_path / "f.parquet"
    _write_failure_parquet(p)
    curriculum = FailureCurriculum(TrajectoryPool(paths=[p]), failure_fraction=1.0, seed=0)
    robot = _FakeRobot()
    device = "cpu"
    env_origins = torch.zeros(4, 3)  # 4 envs, zeroed origins
    env, inner_calls, unwrapped = _fake_env(robot, env_origins, device)

    install(env, curriculum)
    assert unwrapped.phoenix_curriculum is curriculum

    env_ids = torch.tensor([0, 2], dtype=torch.int64)
    unwrapped._reset_idx(env_ids)

    # Original must run first, exactly once, with the same env_ids.
    assert len(inner_calls) == 1
    assert torch.equal(inner_calls[0], env_ids)
    # With failure_fraction=1.0, both envs get overridden.
    assert len(robot.root_pose_calls) == 2
    assert len(robot.joint_state_calls) == 2

    # Quaternion must be rolled xyzw->wxyz. Parquet has (0,0,0,1) => wxyz (1,0,0,0).
    first_pose, first_ids = robot.root_pose_calls[0]
    assert first_pose.shape == (1, 7)
    quat_out = first_pose[0, 3:].tolist()
    assert quat_out == pytest.approx([1.0, 0.0, 0.0, 0.0])
    # Base_pos preserved (env_origins are zero) and env_ids is a 1-elem int64 tensor.
    assert first_pose[0, :3].tolist() == pytest.approx([1.0, 2.0, 0.4])
    assert first_ids.dtype == torch.int64 and first_ids.numel() == 1


# -------------------- H0 bridge-fix: seed-row + velocity write -------------


def _write_multi_row_parquet(path: Path, n_stable: int, n_failure: int, vx_stable: float) -> None:
    """Write a parquet with ``n_stable`` rows of normal trot then ``n_failure``
    rows of failure-flagged kinematics. Velocity is encoded per-row so the
    seeded row can be uniquely identified by its ``base_lin_vel_body``.
    """
    with TrajectoryLogger(path) as log:
        for i in range(n_stable + n_failure):
            failing = i >= n_stable
            log.append(
                TrajectoryStep(
                    step=i,
                    timestamp_s=i * 0.02,
                    base_pos=np.asarray([0.0, 0.0, 0.3], dtype=np.float32),
                    base_quat=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                    base_lin_vel_body=np.asarray(
                        [vx_stable + 0.01 * i, 0.0, 0.0], dtype=np.float32
                    ),
                    base_ang_vel_body=np.asarray([0.0, 0.0, 0.001 * i], dtype=np.float32),
                    joint_pos=np.full(12, 0.1 * i, dtype=np.float32),
                    joint_vel=np.zeros(12, dtype=np.float32),
                    command_vel=np.asarray([0.5, 0.0, 0.0], dtype=np.float32),
                    action=np.zeros(12, dtype=np.float32),
                    contact_forces=np.ones(4, dtype=np.float32),
                    failure_flag=failing,
                    failure_mode="slip" if failing else None,
                )
            )


def test_resolve_seed_row_strategies(tmp_path: Path) -> None:
    p = tmp_path / "slip.parquet"
    _write_multi_row_parquet(p, n_stable=70, n_failure=20, vx_stable=0.5)

    assert _resolve_seed_row(p, "first", 0) == 0
    assert _resolve_seed_row(p, "failure_onset", 0) == 70
    assert _resolve_seed_row(p, "failure_onset_minus_k", 5) == 65
    # Clamp at 0 when k > onset.
    assert _resolve_seed_row(p, "failure_onset_minus_k", 999) == 0
    # Negative k is treated as 0.
    assert _resolve_seed_row(p, "failure_onset_minus_k", -3) == 70


def test_resolve_seed_row_rejects_unknown_strategy(tmp_path: Path) -> None:
    p = tmp_path / "slip.parquet"
    _write_multi_row_parquet(p, n_stable=5, n_failure=2, vx_stable=0.5)
    with pytest.raises(ValueError, match="Unknown seed_row_strategy"):
        _resolve_seed_row(p, "totally_made_up", 0)


def test_resolve_seed_row_rejects_parquet_with_no_failure(tmp_path: Path) -> None:
    p = tmp_path / "all_stable.parquet"
    _write_multi_row_parquet(p, n_stable=10, n_failure=0, vx_stable=0.5)
    with pytest.raises(ValueError, match="no failure_flag=True row"):
        _resolve_seed_row(p, "failure_onset", 0)


def test_cache_loads_failure_onset_row(tmp_path: Path) -> None:
    p = tmp_path / "slip.parquet"
    _write_multi_row_parquet(p, n_stable=70, n_failure=20, vx_stable=0.5)
    cache = _InitialStateCache([p], seed_row_strategy="failure_onset_minus_k", seed_row_offset_k=5)
    state = cache.get(0)
    assert cache.resolved_row(0) == 65
    # Row 65: vx_stable + 0.01 * 65 = 0.5 + 0.65 = 1.15
    assert state.base_lin_vel_body[0] == pytest.approx(1.15, rel=1e-3)
    # joint_pos at row 65 = 0.1 * 65 = 6.5
    assert state.joint_pos[0] == pytest.approx(6.5, rel=1e-3)


def test_install_writes_velocity_when_opted_in(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    p = tmp_path / "slip.parquet"
    _write_multi_row_parquet(p, n_stable=70, n_failure=20, vx_stable=0.5)
    curriculum = FailureCurriculum(TrajectoryPool(paths=[p]), failure_fraction=1.0, seed=0)
    robot = _FakeRobot()
    env, _, unwrapped = _fake_env(robot, torch.zeros(2, 3), "cpu")

    install(
        env,
        curriculum,
        seed_row_strategy="failure_onset_minus_k",
        seed_row_offset_k=5,
        write_velocity=True,
    )
    unwrapped._reset_idx(torch.tensor([0, 1], dtype=torch.int64))

    # Both envs got the failure seed (ff=1.0).
    assert len(robot.root_pose_calls) == 2
    assert len(robot.joint_state_calls) == 2
    assert len(robot.root_velocity_calls) == 2

    # Velocity tensor is (lin_vel, ang_vel) concatenated => shape (1, 6).
    vel_tensor, vel_ids = robot.root_velocity_calls[0]
    assert vel_tensor.shape == (1, 6)
    # Row 65 lin_vel x = 0.5 + 0.01 * 65 = 1.15; ang_vel z = 0.001 * 65 = 0.065
    assert vel_tensor[0, 0].item() == pytest.approx(1.15, rel=1e-3)
    assert vel_tensor[0, 5].item() == pytest.approx(0.065, rel=1e-3)
    assert vel_ids.dtype == torch.int64 and vel_ids.numel() == 1


def test_install_skips_velocity_by_default(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    p = tmp_path / "slip.parquet"
    _write_multi_row_parquet(p, n_stable=70, n_failure=20, vx_stable=0.5)
    curriculum = FailureCurriculum(TrajectoryPool(paths=[p]), failure_fraction=1.0, seed=0)
    robot = _FakeRobot()
    env, _, unwrapped = _fake_env(robot, torch.zeros(2, 3), "cpu")

    install(env, curriculum)  # defaults: first / 0 / write_velocity=False
    unwrapped._reset_idx(torch.tensor([0, 1], dtype=torch.int64))

    assert len(robot.root_pose_calls) == 2
    assert len(robot.joint_state_calls) == 2
    # Default path must NOT call the velocity writer (preserves legacy behavior).
    assert robot.root_velocity_calls == []


def test_patched_reset_is_noop_when_no_envs(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    p = tmp_path / "f.parquet"
    _write_failure_parquet(p)
    curriculum = FailureCurriculum(TrajectoryPool(paths=[p]), failure_fraction=1.0, seed=0)
    robot = _FakeRobot()
    env, inner_calls, unwrapped = _fake_env(robot, torch.zeros(2, 3), "cpu")

    install(env, curriculum)
    unwrapped._reset_idx(torch.tensor([], dtype=torch.int64))
    # Original still gets called; no overrides applied.
    assert len(inner_calls) == 1
    assert robot.root_pose_calls == []
    assert robot.joint_state_calls == []
