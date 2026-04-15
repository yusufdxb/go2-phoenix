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
from phoenix.adaptation.reset_bridge import install
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

    def write_root_pose_to_sim(self, tensor, env_ids):  # noqa: ANN001
        self.root_pose_calls.append((tensor.clone(), env_ids.clone()))

    def write_joint_state_to_sim(self, jp, jv, env_ids):  # noqa: ANN001
        self.joint_state_calls.append((jp.clone(), jv.clone(), env_ids.clone()))


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
    curriculum = FailureCurriculum(
        TrajectoryPool(paths=[p]), failure_fraction=1.0, seed=0
    )
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


def test_patched_reset_is_noop_when_no_envs(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    p = tmp_path / "f.parquet"
    _write_failure_parquet(p)
    curriculum = FailureCurriculum(
        TrajectoryPool(paths=[p]), failure_fraction=1.0, seed=0
    )
    robot = _FakeRobot()
    env, inner_calls, unwrapped = _fake_env(robot, torch.zeros(2, 3), "cpu")

    install(env, curriculum)
    unwrapped._reset_idx(torch.tensor([], dtype=torch.int64))
    # Original still gets called; no overrides applied.
    assert len(inner_calls) == 1
    assert robot.root_pose_calls == []
    assert robot.joint_state_calls == []
