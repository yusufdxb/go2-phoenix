"""Every key in every shipped env config is either wired or flagged. No silent claims.

This closes the CATEGORY rather than enumerating known offenders. The repo has
now been bitten four separate times by a YAML key that reads as if it configures
the robot and is in fact discarded:

* ``robot.init_state`` claimed hips of 0.0 while training used +/-0.1, which is
  where the deploy hip bug came from.
* ``terrain`` was dropped entirely, so every rough-versus-slippery comparison was
  a friction contrast on identical ground.
* three of five ``perturbation`` keys were ignored, and the term it does drive is
  reset-mode, so the "push every 5 s" the config implies never happens.
* ``domain_randomization`` keys were dropped inside a wired section.

Enumerating those four does not stop a fifth. A config author must not be able to
add a plausible-looking knob that silently does nothing, so this test asserts the
closure property: every top-level key of every shipped config is either consumed
by ``build_env_cfg`` or named by the unwired-flagging machinery.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from phoenix.sim_env.go2_env_cfg import (  # noqa: E402
    _UNWIRED_TOP_LEVEL,
    _unwired_sections_present,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs/env"

#: Keys ``build_env_cfg`` actually consumes, taken from its ``data.get(...)``
#: call sites. ``defaults`` is consumed by the layered loader before the builder
#: ever sees the document.
WIRED_TOP_LEVEL = {
    "action",
    "command",
    "domain_randomization",
    "env",
    "fixture",
    "observation",
    "perturbation",
    "reward",
    "robot",
    "seed",
}
LOADER_KEYS = {"defaults"}


def config_files() -> list[Path]:
    return sorted(CONFIG_DIR.glob("*.yaml"))


def test_the_config_directory_is_not_empty() -> None:
    """Guard: an empty glob would make every parametrized test vacuously pass."""
    assert len(config_files()) >= 5


@pytest.mark.parametrize("path", config_files(), ids=lambda p: p.name)
def test_every_top_level_key_is_wired_or_flagged(path: Path) -> None:
    data = yaml.safe_load(path.read_text()) or {}
    accounted = WIRED_TOP_LEVEL | LOADER_KEYS | set(_UNWIRED_TOP_LEVEL)
    unaccounted = set(data) - accounted
    assert not unaccounted, (
        f"{path.name} declares {sorted(unaccounted)}, which build_env_cfg neither "
        f"consumes nor flags as unwired. Either wire it, or add it to "
        f"_UNWIRED_TOP_LEVEL so it warns instead of silently doing nothing."
    )


@pytest.mark.parametrize("path", config_files(), ids=lambda p: p.name)
def test_declared_unwired_keys_actually_warn(path: Path) -> None:
    """A key listed as unwired must be REPORTED when a config declares it."""
    data = yaml.safe_load(path.read_text()) or {}
    declared_unwired = set(data) & set(_UNWIRED_TOP_LEVEL)
    if not declared_unwired:
        pytest.skip(f"{path.name} declares no unwired top-level key")
    reported = set(_unwired_sections_present(data))
    missing = declared_unwired - reported
    assert not missing, f"{path.name} declares {sorted(missing)} but nothing flags it"


def test_a_plausible_new_knob_is_caught_rather_than_ignored() -> None:
    """The closure property itself: an invented sensor knob must not pass silently."""
    invented = {"env": {"task_name": "x"}, "lidar": {"enabled": True, "range_m": 12.0}}
    accounted = WIRED_TOP_LEVEL | LOADER_KEYS | set(_UNWIRED_TOP_LEVEL)
    assert "lidar" not in accounted, (
        "If a future change adds 'lidar' to the wired set, this test must be "
        "updated deliberately rather than by accident."
    )
    unaccounted = set(invented) - accounted
    assert unaccounted == {"lidar"}


def test_terrain_is_flagged_not_silently_dropped() -> None:
    """Regression: terrain was fully silent and invalidated a multi-seed study."""
    assert "terrain" in _UNWIRED_TOP_LEVEL
    assert "terrain" in _unwired_sections_present({"terrain": {"type": "generator"}})
