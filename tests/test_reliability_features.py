"""Tests for policy-internal feature extraction (Phase 3 scaffold).

The numpy reference is fully exercised here; the torch on-robot extractor is
parity-checked against it wherever torch is installed (skipped in the
pure-python CI lane).
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.reliability.features import (
    elu,
    forward_hidden,
    policy_features,
)
from phoenix.reliability.ood_monitor import MahalanobisScorer


def _random_actor(obs_dim: int, hidden: list[int], action_dim: int, seed: int = 0):
    """Return a list of (W, b) Linear layers: obs -> hidden... -> action."""
    rng = np.random.default_rng(seed)
    dims = [obs_dim, *hidden, action_dim]
    layers = []
    for i in range(len(dims) - 1):
        w = rng.standard_normal((dims[i + 1], dims[i])) * 0.3
        b = rng.standard_normal(dims[i + 1]) * 0.1
        layers.append((w, b))
    return layers


# --- ELU --------------------------------------------------------------------


def test_elu_matches_definition():
    assert elu(np.array([0.0]))[0] == pytest.approx(0.0)
    assert elu(np.array([2.5]))[0] == pytest.approx(2.5)
    assert elu(np.array([-1.0]))[0] == pytest.approx(np.expm1(-1.0))
    # Saturates toward -alpha for very negative input.
    assert elu(np.array([-50.0]))[0] == pytest.approx(-1.0, abs=1e-6)


# --- forward_hidden ---------------------------------------------------------


def test_forward_hidden_layer_count_and_shapes():
    layers = _random_actor(obs_dim=12, hidden=[64, 48, 32], action_dim=12)
    obs = np.zeros((5, 12))
    hidden = forward_hidden(obs, layers)
    # One activation per Linear except the action head.
    assert len(hidden) == 3
    assert [h.shape for h in hidden] == [(5, 64), (5, 48), (5, 32)]


def test_forward_hidden_applies_normalizer():
    layers = _random_actor(obs_dim=6, hidden=[8], action_dim=6)
    obs = np.full((3, 6), 5.0)
    mean = np.full(6, 5.0)
    std = np.ones(6)
    # With mean == obs, normalized input is 0 -> first linear is just bias.
    hidden = forward_hidden(obs, layers, mean=mean, std=std)
    expected_pre = layers[0][1]  # bias only
    np.testing.assert_allclose(hidden[0][0], elu(expected_pre), rtol=1e-9)


# --- policy_features --------------------------------------------------------


def test_policy_features_latent_and_obs_shapes():
    layers = _random_actor(obs_dim=10, hidden=[32, 24, 16], action_dim=10)
    obs = np.zeros((7, 10))
    feats = policy_features(obs, layers, taps=("mid", "penultimate"))
    # mid of 3 hidden = index 1 (dim 24), penultimate = index 2 (dim 16).
    assert feats["latent"].shape == (7, 24 + 16)
    assert feats["obs"].shape == (7, 10)


def test_policy_features_single_row_promoted():
    layers = _random_actor(obs_dim=8, hidden=[16, 16], action_dim=8)
    feats = policy_features(np.zeros(8), layers)
    assert feats["latent"].shape[0] == 1
    assert feats["obs"].shape == (1, 8)


# --- end-to-end with the Phase 1 monitor (no torch, no Isaac) ---------------


def test_latent_features_feed_monitor_and_flag_ood():
    layers = _random_actor(obs_dim=16, hidden=[64, 48, 32], action_dim=16, seed=3)
    rng = np.random.default_rng(5)

    nominal_obs = rng.standard_normal((3000, 16))
    nominal_latent = policy_features(nominal_obs, layers)["latent"]
    scorer = MahalanobisScorer.fit(nominal_latent)

    held_in = policy_features(rng.standard_normal((400, 16)), layers)["latent"]
    ood_obs = rng.standard_normal((400, 16)) + 6.0  # shifted observation regime
    held_ood = policy_features(ood_obs, layers)["latent"]

    assert scorer.score(held_ood).mean() > scorer.score(held_in).mean()


# --- torch parity (runs only where torch is installed) ----------------------


def test_torch_extractor_matches_numpy_reference():
    torch = pytest.importorskip("torch")
    from phoenix.reliability.features import TorchActivationExtractor

    obs_dim, hidden, action_dim = 12, [64, 48, 32], 12
    layers = _random_actor(obs_dim, hidden, action_dim, seed=7)

    # Build the matching nn.Sequential: Linear, ELU, Linear, ELU, ..., Linear.
    mods: list = []
    dims = [obs_dim, *hidden, action_dim]
    for i in range(len(dims) - 1):
        lin = torch.nn.Linear(dims[i], dims[i + 1])
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(layers[i][0], dtype=torch.float32))
            lin.bias.copy_(torch.tensor(layers[i][1], dtype=torch.float32))
        mods.append(lin)
        if i < len(dims) - 2:
            mods.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*mods).eval()

    extractor = TorchActivationExtractor(actor, taps=("mid", "penultimate"))
    obs = np.random.default_rng(1).standard_normal((8, obs_dim))
    torch_feats = extractor(torch.tensor(obs, dtype=torch.float32))
    np_feats = policy_features(obs, layers, taps=("mid", "penultimate"))
    extractor.close()

    np.testing.assert_allclose(torch_feats["latent"], np_feats["latent"], atol=1e-4)
