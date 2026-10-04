"""G6: velocity ``slew_sat_hinge_l2`` scales the action delta by the LIVE action term's scale.

Guards the slew-hinge unit bug (policy-action units vs the 0.15 rad threshold,
ad81c70): with action scale 0.25 an unscaled hinge fires at 0.0375 rad of target
motion instead of 0.15 rad. Walk checkpoints (e.g. model_2750) train on this path.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from phoenix.velocity import mdp  # noqa: E402


def _env(action, prev, scale, name: str = mdp.ACTION_NAME):
    term = SimpleNamespace(cfg=SimpleNamespace(scale=scale))
    am = SimpleNamespace(
        action=torch.tensor([action], dtype=torch.float32),
        prev_action=torch.tensor([prev], dtype=torch.float32),
        get_term=lambda n: term if n == name else (_ for _ in ()).throw(KeyError(n)),
    )
    return SimpleNamespace(action_manager=am)


def test_hinge_threshold_is_in_joint_target_radians() -> None:
    # |da| = 0.5 policy units * 0.25 = 0.125 rad < 0.15 rad: must NOT fire.
    # (Unscaled, 0.5 > 0.15 would fire: the pre-fix bug.)
    out = mdp.slew_sat_hinge_l2(_env([0.5], [0.0], 0.25), threshold=0.15)
    assert out.item() == 0.0


def test_hinge_value_uses_scaled_delta() -> None:
    # |da| = 1.0 * 0.25 = 0.25 rad; excess over 0.15 = 0.10; squared = 0.01.
    out = mdp.slew_sat_hinge_l2(_env([1.0], [0.0], 0.25), threshold=0.15)
    assert out.item() == pytest.approx(0.01, abs=1e-7)


def test_hinge_follows_the_live_scale_not_a_constant() -> None:
    # Same action delta, different live scale: result must track the term's cfg.scale.
    a, p = [1.0], [0.0]
    s_half = mdp.slew_sat_hinge_l2(_env(a, p, 0.5), threshold=0.15).item()
    s_quarter = mdp.slew_sat_hinge_l2(_env(a, p, 0.25), threshold=0.15).item()
    assert s_half == pytest.approx((0.5 - 0.15) ** 2, abs=1e-7)
    assert s_quarter == pytest.approx((0.25 - 0.15) ** 2, abs=1e-7)


def test_non_scalar_scale_is_refused() -> None:
    with pytest.raises(TypeError, match="scalar action scale"):
        mdp.slew_sat_hinge_l2(_env([1.0], [0.0], [0.25, 0.25]), threshold=0.15)
