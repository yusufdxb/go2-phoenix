"""Failure-curriculum fine-tuning.

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
