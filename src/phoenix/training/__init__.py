"""Training entry points.

* :mod:`phoenix.training.ppo_runner` — baseline and adaptation PPO loops
  over the Phoenix GO2 env, using rsl_rl's :class:`OnPolicyRunner`.
* :mod:`phoenix.training.evaluate` — headless rollout + video capture for
  checkpoint evaluation and the side-by-side demo video.
* :mod:`phoenix.training.agent_cfg` — builds an
  :class:`RslRlOnPolicyRunnerCfg` from a YAML training config.
"""

from .agent_cfg import build_runner_cfg

__all__ = ["build_runner_cfg"]
