"""Tests for the sensor-to-policy observation parity gate.

The gate itself is the regression test for the zeroed-``base_lin_vel`` bug.
These tests prove the gate passes on the real deploy configuration AND that
it actually fails on each way the assembly can go wrong, because a gate that
cannot fail is not a gate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from phoenix.sim2real.obs_parity import (
    DEFAULT_DEPLOY_CONFIG,
    OBS_TERM_ORDER,
    REPO_ROOT,
    TRAINING_ENV_CONFIG,
    check_observation_parity,
    check_zeros_fallback_is_explicit,
    load_training_term_order,
)
from phoenix.sim2real.observation import JointOrder, ObservationBuilder

DEPLOY_CONFIGS = sorted((REPO_ROOT / "configs" / "sim2real").glob("*.yaml"))


def test_deploy_configs_were_actually_found() -> None:
    """An empty glob would silently parametrize the gate below out of existence."""
    assert DEPLOY_CONFIGS, f"no deploy configs under {REPO_ROOT / 'configs' / 'sim2real'}"
    assert DEFAULT_DEPLOY_CONFIG in DEPLOY_CONFIGS


def _builder_from(cfg_path: Path) -> ObservationBuilder:
    cfg = yaml.safe_load(cfg_path.read_text())
    return ObservationBuilder(
        JointOrder(tuple(cfg["joint_order"])), cfg["control"]["default_joint_pos"]
    )


@pytest.mark.parametrize("cfg_path", DEPLOY_CONFIGS, ids=lambda p: p.name)
def test_every_deploy_config_passes_the_parity_gate(cfg_path: Path) -> None:
    report = check_observation_parity(_builder_from(cfg_path))
    assert report.passed, report.summary()
    assert [t.name for t in report.terms] == list(OBS_TERM_ORDER)


def test_training_include_list_matches_the_deploy_term_order() -> None:
    order = load_training_term_order(TRAINING_ENV_CONFIG)
    assert order is not None, f"{TRAINING_ENV_CONFIG} must exist in a checkout"
    assert order == OBS_TERM_ORDER


def test_missing_training_config_is_reported_not_ignored(tmp_path) -> None:
    report = check_observation_parity(
        _builder_from(DEFAULT_DEPLOY_CONFIG),
        config_path=tmp_path / "does_not_exist.yaml",
    )
    assert not report.passed
    assert any("cannot cross-check term order" in f for f in report.failures)


def test_zeros_fallback_contract_holds() -> None:
    assert check_zeros_fallback_is_explicit() == ()


# ---------------------------------------------------------------------------
# Negative controls: the gate must FAIL on each defect class.
# ---------------------------------------------------------------------------


class _ZeroingBuilder(ObservationBuilder):
    """Reproduces the shipped bug: one trained term hardcoded to zero."""

    def __init__(self, *args, term_slice=slice(0, 3), **kwargs):
        super().__init__(*args, **kwargs)
        self._term_slice = term_slice

    def build(self, **kwargs):
        obs = super().build(**kwargs)
        obs[self._term_slice] = 0.0
        return obs


class _SwappingBuilder(ObservationBuilder):
    """Swaps base_lin_vel and base_ang_vel: right dims, wrong order."""

    def build(self, **kwargs):
        obs = super().build(**kwargs)
        obs[0:3], obs[3:6] = obs[3:6].copy(), obs[0:3].copy()
        return obs


class _RescalingBuilder(ObservationBuilder):
    """Applies a scale factor training never applied."""

    def build(self, **kwargs):
        obs = super().build(**kwargs)
        obs[24:36] *= 0.05  # joint_vel scaled, a common upstream convention
        return obs


def _args() -> tuple:
    cfg = yaml.safe_load(DEFAULT_DEPLOY_CONFIG.read_text())
    return (JointOrder(tuple(cfg["joint_order"])), cfg["control"]["default_joint_pos"])


def test_gate_fails_when_base_lin_vel_is_hardcoded_to_zero() -> None:
    report = check_observation_parity(_ZeroingBuilder(*_args(), term_slice=slice(0, 3)))
    assert not report.passed
    assert any("base_lin_vel" in f and "constant zero" in f for f in report.failures)


def test_gate_fails_when_any_other_term_is_zeroed() -> None:
    # joint_vel dims 24..35: the same defect elsewhere in the vector.
    report = check_observation_parity(_ZeroingBuilder(*_args(), term_slice=slice(24, 36)))
    assert not report.passed
    assert any("joint_vel" in f for f in report.failures)


def test_gate_fails_on_term_order_swap() -> None:
    report = check_observation_parity(_SwappingBuilder(*_args()))
    assert not report.passed
    names = {t.name for t in report.terms if not t.passed}
    assert {"base_lin_vel", "base_ang_vel"} <= names


def test_gate_fails_on_an_unexpected_scale_factor() -> None:
    report = check_observation_parity(_RescalingBuilder(*_args()))
    assert not report.passed
    assert any("joint_vel" in f for f in report.failures)


def test_gate_summary_is_readable() -> None:
    report = check_observation_parity(_ZeroingBuilder(*_args()))
    summary = report.summary()
    assert "OBSERVATION PARITY: FAIL" in summary
    for name in OBS_TERM_ORDER:
        assert name in summary


def test_probe_values_are_all_distinct_and_nonzero() -> None:
    """Without this the gate could pass on a term aliased onto another."""
    from phoenix.sim2real.obs_parity import _probe_inputs

    probe = _probe_inputs(12)
    for name, arr in probe.items():
        assert not np.allclose(arr, 0.0), name
    flat = [tuple(np.asarray(v, dtype=np.float64).ravel()[:3]) for v in probe.values()]
    assert len(set(flat)) == len(flat)
