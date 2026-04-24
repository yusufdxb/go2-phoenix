"""Tests for ``phoenix.sim2real.verify_deploy``.

The parity-check logic is pure numpy so it's tested by injecting fake
inference callables. The parquet-to-obs builder is tested against a
tiny parquet written by the real TrajectoryLogger so the schema stays
in lock-step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phoenix.real_world.trajectory_logger import TrajectoryLogger, TrajectoryStep
from phoenix.sim2real.observation import JointOrder, ObservationBuilder
from phoenix.sim2real.verify_deploy import (
    ParityReport,
    build_obs_from_parquet,
    verify_parity,
)

# ------------ verify_parity -------------------------------------------------


def _fake_infer(scale: float = 1.0, bias: float = 0.0):
    def _fn(obs: np.ndarray) -> np.ndarray:
        return scale * obs[:12] + bias

    return _fn


def test_verify_parity_identical_inference_passes() -> None:
    obs = [np.random.default_rng(0).standard_normal(48).astype(np.float32) for _ in range(5)]
    fn = _fake_infer()
    report = verify_parity(obs, fn, fn, tol=1e-6)
    assert isinstance(report, ParityReport)
    assert report.passed is True
    assert report.steps_checked == 5
    assert report.max_abs_diff == pytest.approx(0.0, abs=1e-12)


def test_verify_parity_drift_over_tol_fails() -> None:
    obs = [np.ones(48, dtype=np.float32) for _ in range(3)]
    report = verify_parity(obs, _fake_infer(), _fake_infer(bias=1e-2), tol=1e-4)
    assert report.passed is False
    assert report.max_abs_diff == pytest.approx(1e-2, abs=1e-8)
    assert report.steps_checked == 3
    assert len(report.per_step_max) == 3


def test_verify_parity_drift_under_tol_passes() -> None:
    obs = [np.ones(48, dtype=np.float32) for _ in range(3)]
    report = verify_parity(obs, _fake_infer(), _fake_infer(bias=1e-6), tol=1e-4)
    assert report.passed is True
    assert report.max_abs_diff < 1e-4


def test_verify_parity_empty_obs_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        verify_parity([], _fake_infer(), _fake_infer(), tol=1e-4)


# ------------ build_obs_from_parquet ---------------------------------------


_JOINTS = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)
_DEFAULT_Q = {
    **{n: 0.0 for n in _JOINTS if "hip" in n},
    **{n: 0.8 for n in _JOINTS if "thigh" in n and n.startswith("F")},
    **{n: 1.0 for n in _JOINTS if "thigh" in n and n.startswith("R")},
    **{n: -1.5 for n in _JOINTS if "calf" in n},
}


def _write_tiny_parquet(path: Path, n: int = 4) -> None:
    with TrajectoryLogger(path) as log:
        for i in range(n):
            log.append(
                TrajectoryStep(
                    step=i,
                    timestamp_s=0.02 * i,
                    base_pos=np.asarray([0.0, 0.0, 0.4], dtype=np.float32),
                    base_quat=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                    base_lin_vel_body=np.asarray([0.5, 0.0, 0.0], dtype=np.float32),
                    base_ang_vel_body=np.zeros(3, dtype=np.float32),
                    joint_pos=np.asarray([_DEFAULT_Q[n] for n in _JOINTS], dtype=np.float32),
                    joint_vel=np.zeros(12, dtype=np.float32),
                    command_vel=np.asarray([0.5, 0.0, 0.0], dtype=np.float32),
                    action=np.zeros(12, dtype=np.float32),
                    contact_forces=np.ones(4, dtype=np.float32),
                    failure_flag=False,
                    failure_mode=None,
                )
            )


def _builder() -> ObservationBuilder:
    return ObservationBuilder(JointOrder(_JOINTS), _DEFAULT_Q)


def test_build_obs_from_parquet_yields_proprio_obs(tmp_path: Path) -> None:
    p = tmp_path / "t.parquet"
    _write_tiny_parquet(p, n=4)
    obs = list(build_obs_from_parquet(p, _builder(), pad_zeros=0))
    assert len(obs) == 4
    for o in obs:
        assert o.shape == (48,)
        assert o.dtype == np.float32


def test_build_obs_from_parquet_applies_pad_zeros(tmp_path: Path) -> None:
    p = tmp_path / "t.parquet"
    _write_tiny_parquet(p, n=2)
    obs = list(build_obs_from_parquet(p, _builder(), pad_zeros=187))
    assert len(obs) == 2
    assert obs[0].shape == (235,)
    # Padding region is all zero.
    assert np.all(obs[0][48:] == 0.0)


def test_build_obs_from_parquet_respects_max_steps(tmp_path: Path) -> None:
    p = tmp_path / "t.parquet"
    _write_tiny_parquet(p, n=10)
    obs = list(build_obs_from_parquet(p, _builder(), max_steps=3))
    assert len(obs) == 3
