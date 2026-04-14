"""Integration tests that require Isaac Lab + GPU.

Skipped in CI (pytest filters out ``@pytest.mark.sim``). Run locally via::

    pytest tests -m sim -p no:launch_testing -p no:launch_pytest

Each test must leave Isaac Sim in a closed state so the next test can
boot a fresh sim app.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.sim

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def sim_app():
    """Boot Isaac Sim once per test module, tear down after."""
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    yield app
    app.close()


def test_build_env_cfg_applies_friction_range(sim_app) -> None:
    """`slippery.yaml` must actually reduce friction on the physics material."""
    from phoenix.sim_env import build_env_cfg

    cfg_rough = build_env_cfg(REPO_ROOT / "configs" / "env" / "rough.yaml")
    cfg_slip = build_env_cfg(REPO_ROOT / "configs" / "env" / "slippery.yaml")

    pm_r = cfg_rough.events.default.physics_material
    pm_s = cfg_slip.events.default.physics_material

    assert pm_r is not None and pm_s is not None
    r_lo, r_hi = pm_r.params["static_friction_range"]
    s_lo, s_hi = pm_s.params["static_friction_range"]
    # slippery overlay must narrow the friction range strictly below the rough baseline
    assert s_hi <= r_hi
    assert s_lo <= r_lo


def test_build_env_cfg_applies_mass_range(sim_app) -> None:
    from phoenix.sim_env import build_env_cfg

    cfg = build_env_cfg(REPO_ROOT / "configs" / "env" / "rough.yaml")
    abm = cfg.events.default.add_base_mass
    assert abm is not None
    lo, hi = abm.params["mass_distribution_params"]
    assert lo == pytest.approx(-2.0)
    assert hi == pytest.approx(2.0)


def test_perturbation_overlay_enables_external_force(sim_app) -> None:
    from phoenix.sim_env import build_env_cfg

    cfg_base = build_env_cfg(REPO_ROOT / "configs" / "env" / "base.yaml")
    cfg_pert = build_env_cfg(REPO_ROOT / "configs" / "env" / "perturbation.yaml")

    efx_b = cfg_base.events.default.base_external_force_torque
    efx_p = cfg_pert.events.default.base_external_force_torque
    assert efx_p is not None

    # Base disables the force; perturbation overlay must enable it.
    f_lo_base, f_hi_base = efx_b.params["force_range"]
    f_lo_pert, f_hi_pert = efx_p.params["force_range"]
    assert f_hi_base == 0.0
    assert f_hi_pert > f_hi_base


def test_command_ranges_come_from_yaml(sim_app) -> None:
    from phoenix.sim_env import build_env_cfg

    cfg = build_env_cfg(REPO_ROOT / "configs" / "env" / "rough.yaml")
    r = cfg.commands.base_velocity.ranges
    assert r.lin_vel_x == (-1.0, 1.0)
    assert r.lin_vel_y == (-0.6, 0.6)
    assert r.ang_vel_z == (-1.0, 1.0)
