"""LEGACY (Phoenix v2): the failure-seeded curriculum belongs conceptually to Ashfall.
Phoenix v2 fine-tunes through ``training.ppo_runner --resume`` with a targeted env
overlay instead. See docs/legacy/README.md.

Failure-curriculum fine-tuning.

The adaptation loop warm-starts a policy from a baseline checkpoint and
continues PPO training, but with a fraction of rollouts seeded from real
failure trajectories (replayed in sim with perturbed physics).

* :mod:`phoenix.adaptation.fine_tune` — CLI entry point; wraps
  :class:`rsl_rl.runners.OnPolicyRunner` with the failure curriculum.
* :mod:`phoenix.adaptation.curriculum` — pure-python curriculum scheduler,
  testable in CI.
"""

from .curriculum import FailureCurriculum, TrajectoryPool

__all__ = ["FailureCurriculum", "TrajectoryPool"]


def install_reset_bridge(env, curriculum):  # pragma: no cover - requires Isaac Lab
    """Install the curriculum reset bridge on ``env`` (requires torch + Isaac Lab)."""
    from .reset_bridge import install

    install(env, curriculum)
