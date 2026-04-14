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
