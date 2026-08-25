"""Regression tests for the ONNX-export observation normalizer.

These lock the fix for an audit finding (2026-05-21): the previous
``_load_normalizer`` searched for top-level checkpoint keys
(``obs_norm_state_dict`` etc.) that rsl_rl 3.x never writes. rsl_rl keeps
the ``EmpiricalNormalization`` buffers *inside* ``actor_state_dict`` under
``obs_normalizer._mean`` / ``obs_normalizer._var``. The old code therefore
always returned ``None`` and exported a policy with no observation
normalization at all, even though every training config sets
``empirical_normalization: true``, a silent train/deploy mismatch the
ONNX-vs-TorchScript parity gate cannot catch.

The statistics-extraction logic is pure dict traversal (no torch), so it
runs in the no-sim / no-ros CI job. The torch ``_Normalizer`` numeric
behaviour is covered separately under the ``sim`` marker.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real.export import (
    _RSL_RL_NORM_EPS,
    _extract_normalizer_stats,
    _find_buffer,
)


def test_eps_matches_rsl_rl_default() -> None:
    # rsl_rl.modules.EmpiricalNormalization defaults to eps=1e-2 and
    # normalizes as (x - mean) / (std + eps). The exporter must use the
    # same constant or low-variance obs dims are scaled wrong.
    assert _RSL_RL_NORM_EPS == 1e-2


def test_extract_from_rsl_rl_3x_layout() -> None:
    # rsl_rl 3.x: normalizer buffers live inside actor_state_dict.
    actor_sd = {
        "mlp.0.weight": np.zeros((512, 48)),
        "mlp.0.bias": np.zeros(512),
        "obs_normalizer._mean": np.full((1, 48), 0.3),
        "obs_normalizer._var": np.full((1, 48), 4.0),
        "obs_normalizer._std": np.full((1, 48), 2.0),
        "obs_normalizer.count": np.asarray(1000),
    }
    ckpt = {"actor_state_dict": actor_sd}
    stats = _extract_normalizer_stats(ckpt, actor_sd)
    assert stats is not None
    mean, var = stats
    np.testing.assert_allclose(np.asarray(mean).reshape(-1), 0.3)
    np.testing.assert_allclose(np.asarray(var).reshape(-1), 4.0)


def test_extract_returns_none_when_no_normalizer() -> None:
    # Policy trained with empirical_normalization: false, no buffers.
    actor_sd = {
        "mlp.0.weight": np.zeros((512, 48)),
        "mlp.0.bias": np.zeros(512),
    }
    assert _extract_normalizer_stats({"actor_state_dict": actor_sd}, actor_sd) is None


def test_extract_from_legacy_top_level_layout() -> None:
    # Older / hand-rolled checkpoints: a separate top-level dict.
    actor_sd = {"mlp.0.weight": np.zeros((512, 48))}
    ckpt = {
        "actor_state_dict": actor_sd,
        "obs_norm_state_dict": {"mean": np.full(48, 1.5), "var": np.full(48, 9.0)},
    }
    stats = _extract_normalizer_stats(ckpt, actor_sd)
    assert stats is not None
    mean, var = stats
    np.testing.assert_allclose(np.asarray(mean).reshape(-1), 1.5)
    np.testing.assert_allclose(np.asarray(var).reshape(-1), 9.0)


def test_find_buffer_matches_by_suffix() -> None:
    state = {"a.b.obs_normalizer._mean": 7, "other": 1}
    assert _find_buffer(state, ("obs_normalizer._mean",)) == 7
    assert _find_buffer(state, ("does.not.exist",)) is None


@pytest.mark.parametrize(
    "var,expected_std",
    [
        (4.0, 2.0 + _RSL_RL_NORM_EPS),  # unit-ish variance
        (0.0, 0.0 + _RSL_RL_NORM_EPS),  # near-constant obs dim: eps dominates
    ],
)
def test_normalizer_std_formula_documents_eps_placement(var: float, expected_std: float) -> None:
    # Documents the exact transform without needing torch: the exporter's
    # std is sqrt(var) + eps, eps added OUTSIDE the sqrt, matching rsl_rl.
    std = float(np.sqrt(var)) + _RSL_RL_NORM_EPS
    assert std == pytest.approx(expected_std)


# --- shield latent taps ------------------------------------------------------


def test_default_tap_indices_matches_phase3_study():
    """The 3-hidden-layer actor used in the Isaac study taps Linear 2 and 3.

    Those inputs are the post-ELU outputs of hidden layers 2 and 3, giving the
    384-dim latent (256 + 128) the deployed monitor is fit on. If this drifts,
    every fitted artifact silently becomes invalid.
    """
    from phoenix.sim2real.export import default_tap_indices

    assert default_tap_indices(3) == [2, 3]


def test_default_tap_indices_shapes():
    from phoenix.sim2real.export import default_tap_indices

    assert default_tap_indices(1) == [1]
    assert default_tap_indices(2) == [1, 2]
    assert default_tap_indices(4) == [2, 4]
    for n in range(1, 8):
        taps = default_tap_indices(n)
        assert taps == sorted(set(taps))
        assert all(1 <= t <= n for t in taps)


def test_default_tap_indices_rejects_no_hidden_layers():
    import pytest

    from phoenix.sim2real.export import default_tap_indices

    with pytest.raises(ValueError, match="no hidden layers"):
        default_tap_indices(0)


# --- checkpoint_has_obs_normalizer ------------------------------------------
#
# These pin the defect that made deploy/flat_v4_latent.onnx fail the deploy
# parity gate. scripts/reliability_rollout.py hardcoded
# ``empirical_normalization: True``. flat-v4's checkpoint carries no normalizer
# buffers, so rsl_rl built a fresh EmpiricalNormalization (mean=0, std=1) whose
# forward is still ``(x - 0) / (1 + 1e-2)``: a silent 1% shrink of every
# observation. The exporter correctly found no statistics and exported a
# raw-observation policy, so the two paths differed by ~1% on every frame
# (worst recorded latent gap 6.44 against a latent of scale 673, 100% of
# frames). The stand-v3 checkpoint DOES carry the buffers, which is exactly
# why it passed the same gate. Both sides must now derive the answer here.


def test_checkpoint_without_normalizer_reports_false() -> None:
    from phoenix.sim2real.export import checkpoint_has_obs_normalizer

    ckpt = {
        "actor_state_dict": {
            "mlp.0.weight": np.zeros((512, 48)),
            "mlp.0.bias": np.zeros(512),
            "distribution.std_param": np.zeros(12),
        }
    }
    assert checkpoint_has_obs_normalizer(ckpt) is False


def test_checkpoint_with_normalizer_reports_true() -> None:
    from phoenix.sim2real.export import checkpoint_has_obs_normalizer

    ckpt = {
        "actor_state_dict": {
            "mlp.0.weight": np.zeros((512, 48)),
            "obs_normalizer._mean": np.zeros((1, 48)),
            "obs_normalizer._var": np.ones((1, 48)),
        }
    }
    assert checkpoint_has_obs_normalizer(ckpt) is True


def test_checkpoint_normalizer_agrees_with_exporter() -> None:
    # The whole point of the helper is that it cannot disagree with the
    # exporter's own decision. Same inputs, same answer, always.
    from phoenix.sim2real.export import checkpoint_has_obs_normalizer

    for actor_sd in (
        {"mlp.0.weight": np.zeros((4, 4))},
        {"mlp.0.weight": np.zeros((4, 4)), "obs_normalizer._mean": np.zeros(4),
         "obs_normalizer._var": np.ones(4)},
    ):
        ckpt = {"actor_state_dict": actor_sd}
        exporter_found = _extract_normalizer_stats(ckpt, actor_sd) is not None
        assert checkpoint_has_obs_normalizer(ckpt) is exporter_found


def test_unreadable_checkpoint_path_does_not_raise() -> None:
    from phoenix.sim2real.export import checkpoint_has_obs_normalizer

    assert checkpoint_has_obs_normalizer("/nonexistent/does-not-exist.pt") is False
