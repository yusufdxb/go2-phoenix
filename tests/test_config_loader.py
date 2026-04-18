"""Tests for the layered YAML config loader.

These tests run in the CI (non-sim) job — they exercise only OmegaConf
and the Phoenix loader, with no Isaac Lab dependency.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phoenix.sim_env.config_loader import load_layered_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = REPO_ROOT / "configs"


def test_base_env_loads() -> None:
    cfg = load_layered_config(CONFIGS / "env" / "base.yaml").to_container()
    assert cfg["env"]["task_name"] == "Isaac-Velocity-Rough-Unitree-Go2-v0"
    assert cfg["env"]["num_envs"] == 4096
    assert cfg["domain_randomization"]["enabled"] is True
    # perturbation disabled in base
    assert cfg["perturbation"]["enabled"] is False


def test_rough_inherits_base() -> None:
    cfg = load_layered_config(CONFIGS / "env" / "rough.yaml").to_container()
    # rough overlays terrain; base fields remain present
    assert cfg["env"]["num_envs"] == 4096
    assert cfg["robot"]["name"] == "unitree_go2"


def test_perturbation_overlay_flips_flag() -> None:
    cfg = load_layered_config(CONFIGS / "env" / "perturbation.yaml").to_container()
    assert cfg["perturbation"]["enabled"] is True
    assert cfg["perturbation"]["push_velocity_xy"] == pytest.approx(1.5)
    # DR ranges from base still present (overlay did not touch them)
    assert cfg["domain_randomization"]["friction_range"] == [0.3, 1.5]


def test_slippery_overlay_narrows_friction() -> None:
    cfg = load_layered_config(CONFIGS / "env" / "slippery.yaml").to_container()
    fr_lo, fr_hi = cfg["domain_randomization"]["friction_range"]
    assert fr_lo < 0.3 and fr_hi < 1.0, "slippery.yaml must narrow friction to a low-grip regime"


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_layered_config(CONFIGS / "env" / "does_not_exist.yaml")


def test_circular_defaults_raise(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("defaults:\n  - b\nfoo: 1\n")
    b.write_text("defaults:\n  - a\nbar: 2\n")
    with pytest.raises(ValueError, match="Circular"):
        load_layered_config(a)


# --- unwired-sections audit ------------------------------------------------
# These tests guard against the silent no-op class of bugs. `build_env_cfg`
# only plumbs a subset of YAML into the Isaac Lab env_cfg; the rest is
# aspirational documentation. `_unwired_sections_present` is called from
# `build_env_cfg` and emits a log warning on every unwired section, so a
# future silent-drop regression would surface during any eval or retrain.

from phoenix.sim_env.go2_env_cfg import _unwired_sections_present  # noqa: E402


def test_unwired_detects_reward_section() -> None:
    data = {"env": {}, "reward": {"track_lin_vel_xy": 1.0}}
    assert "reward" in _unwired_sections_present(data)


def test_unwired_detects_observation_noise() -> None:
    data = {"env": {}, "observation": {"noise": {"joint_pos": 0.01}}}
    assert "observation.noise" in _unwired_sections_present(data)


def test_unwired_detects_termination_and_robot_subsections() -> None:
    data = {
        "env": {},
        "termination": {"pitch_threshold_rad": 0.8},
        "robot": {"init_state": {}, "actuator": {"stiffness": 25.0}},
    }
    result = _unwired_sections_present(data)
    assert "termination" in result
    assert "robot.init_state" in result
    assert "robot.actuator" in result


def test_unwired_empty_when_only_wired_sections_present() -> None:
    # env / command / domain_randomization / perturbation / seed are all
    # actually plumbed, so a config containing only those should produce
    # no unwired-section warnings.
    data = {
        "env": {},
        "command": {"lin_vel_x": [-1.0, 1.0]},
        "domain_randomization": {"enabled": True},
        "perturbation": {"enabled": False},
        "seed": 42,
    }
    assert _unwired_sections_present(data) == []


def test_unwired_flags_base_yaml_current_state() -> None:
    """Regression guard: base.yaml today contains reward, observation.noise,
    termination, robot.init_state, and robot.actuator — all unwired. If any
    of those graduate to wired, this test should be updated at the same time
    as removing the YAML key from the _UNWIRED_TOP_LEVEL tuple."""
    cfg = load_layered_config(CONFIGS / "env" / "base.yaml").to_container()
    unwired = set(_unwired_sections_present(cfg))
    assert {
        "reward",
        "observation.noise",
        "termination",
        "robot.init_state",
        "robot.actuator",
    }.issubset(unwired)
