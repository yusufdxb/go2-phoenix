"""Tests for ``phoenix.training.evaluate._resolve_slew_reference``.

The deploy-equivalent slew metric needs the affine map the Isaac Lab
``joint_pos`` action term applies (``target = offset + scale * action``).
Reading it off the live action term is what keeps the sim metric and the
Jetson limiter from drifting apart, so a missing term or a missing attribute
must fail loudly rather than fall back to the old raw-action-delta number.

Importing this module also exercises the lazy-torch convention: torch is not
installed in the CI venv, so ``phoenix.training.evaluate`` must import
without it.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from phoenix.training.evaluate import _resolve_slew_reference, _to_numpy


class _FakeTerm:
    def __init__(self, offset, scale, joint_ids=slice(None), action_dim=4) -> None:
        self._offset = offset
        self._scale = scale
        self._joint_ids = joint_ids
        self._asset = types.SimpleNamespace(data=types.SimpleNamespace(joint_pos=None))
        self.action_dim = action_dim


class _FakeManager:
    active_terms = ["joint_pos"]

    def __init__(self, term, name: str = "joint_pos") -> None:
        self._term = term
        self._name = name

    def get_term(self, name: str):
        if name != self._name:
            raise KeyError(name)
        return self._term


def _env(manager):
    return types.SimpleNamespace(unwrapped=types.SimpleNamespace(action_manager=manager))


def test_array_offset_and_float_scale() -> None:
    offset = np.tile(np.array([0.1, -0.3, 0.8, 0.0], dtype=np.float32), (2, 1))
    term = _FakeTerm(offset=offset, scale=0.25)
    ref = _resolve_slew_reference(_env(_FakeManager(term)), _to_numpy)
    assert np.allclose(ref.default_q, offset)
    assert ref.action_scale == 0.25
    assert ref.joint_ids == slice(None)
    assert ref.asset is term._asset


def test_scalar_offset_is_broadcast_to_action_dim() -> None:
    term = _FakeTerm(offset=0.0, scale=0.25, action_dim=12)
    ref = _resolve_slew_reference(_env(_FakeManager(term)), _to_numpy)
    assert ref.default_q.shape == (12,)
    assert np.all(ref.default_q == 0.0)


def test_per_joint_scale_array_is_kept() -> None:
    scale = np.full((2, 4), 0.25, dtype=np.float32)
    term = _FakeTerm(offset=np.zeros((2, 4), dtype=np.float32), scale=scale)
    ref = _resolve_slew_reference(_env(_FakeManager(term)), _to_numpy)
    assert np.allclose(ref.action_scale, scale)


def test_missing_action_manager_raises() -> None:
    env = types.SimpleNamespace(unwrapped=types.SimpleNamespace())
    with pytest.raises(RuntimeError, match="action_manager"):
        _resolve_slew_reference(env, _to_numpy)


def test_missing_joint_pos_term_raises() -> None:
    term = _FakeTerm(offset=0.0, scale=0.25)
    manager = _FakeManager(term, name="something_else")
    with pytest.raises(RuntimeError, match="joint_pos"):
        _resolve_slew_reference(_env(manager), _to_numpy)


def test_missing_scale_attribute_raises() -> None:
    term = _FakeTerm(offset=0.0, scale=0.25)
    del term._scale
    with pytest.raises(RuntimeError, match="_scale"):
        _resolve_slew_reference(_env(_FakeManager(term)), _to_numpy)


def test_scalar_offset_without_action_dim_raises() -> None:
    term = _FakeTerm(offset=0.0, scale=0.25, action_dim=0)
    with pytest.raises(RuntimeError, match="action_dim"):
        _resolve_slew_reference(_env(_FakeManager(term)), _to_numpy)
