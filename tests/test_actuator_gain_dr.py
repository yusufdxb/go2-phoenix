"""Motor-strength DR must scale NOMINAL gains, not compound across resets (2026-09-26 bug)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from phoenix.sim_env.go2_env_cfg import scale_explicit_actuator_gains  # noqa: E402


def _env(n: int = 512, kp: float = 25.0, kd: float = 0.5):
    act = SimpleNamespace(stiffness=torch.full((n, 12), kp), damping=torch.full((n, 12), kd))
    asset = SimpleNamespace(actuators={"go2": act}, num_instances=n, device="cpu")
    return SimpleNamespace(scene={"robot": asset}), SimpleNamespace(name="robot"), act


def test_repeated_resets_stay_within_the_configured_range() -> None:
    torch.manual_seed(0)
    env, cfg, act = _env()
    for _ in range(200):
        scale_explicit_actuator_gains(env, None, cfg, (0.85, 1.15), (0.85, 1.15))
    r = act.stiffness / 25.0
    assert float(r.min()) >= 0.85 - 1e-6 and float(r.max()) <= 1.15 + 1e-6
    d = act.damping / 0.5
    assert float(d.min()) >= 0.85 - 1e-6 and float(d.max()) <= 1.15 + 1e-6


def test_partial_reset_only_touches_those_envs_and_scales_nominal() -> None:
    torch.manual_seed(1)
    env, cfg, act = _env(n=8)
    scale_explicit_actuator_gains(env, None, cfg, (0.5, 0.5), None)
    assert torch.allclose(act.stiffness, torch.full((8, 12), 12.5))
    scale_explicit_actuator_gains(env, torch.tensor([2, 5]), cfg, (2.0, 2.0), None)
    assert torch.allclose(act.stiffness[[2, 5]], torch.full((2, 12), 50.0))  # 25 x 2, not 12.5 x 2
    assert torch.allclose(act.stiffness[[0, 1, 3, 4, 6, 7]], torch.full((6, 12), 12.5))


def test_single_startup_call_is_unchanged() -> None:
    torch.manual_seed(2)
    env, cfg, act = _env(n=4)
    scale_explicit_actuator_gains(env, None, cfg, (0.9, 1.1), (0.9, 1.1))
    r = act.stiffness / 25.0
    assert float(r.min()) >= 0.9 - 1e-6 and float(r.max()) <= 1.1 + 1e-6
