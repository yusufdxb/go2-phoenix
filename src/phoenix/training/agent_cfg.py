"""Translate a Phoenix training YAML into an rsl_rl runner cfg.

We start from Isaac Lab's upstream ``UnitreeGo2RoughPPORunnerCfg``
instance (so ``obs_groups`` / ``actor`` / ``critic`` / other defaults are
inherited correctly), then patch only the fields the Phoenix YAML
overrides. This is brittle to upstream renames by design — if
Isaac Lab changes the cfg shape, we want to fail loudly rather than
silently diverge from what the reference training script does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def build_runner_cfg(data: dict[str, Any], task_name: str) -> RslRlOnPolicyRunnerCfg:
    """Build an rsl_rl runner cfg for ``task_name``, patched by Phoenix YAML."""
    import gymnasium as gym

    # Resolve the upstream rsl_rl cfg entry point to a cfg-class and instantiate.
    entry = gym.spec(task_name).kwargs["rsl_rl_cfg_entry_point"]
    if not isinstance(entry, str):
        raise TypeError(f"Unexpected rsl_rl_cfg_entry_point type: {type(entry)}")
    module_name, class_name = entry.rsplit(":", 1) if ":" in entry else entry.rsplit(".", 1)
    import importlib

    cfg_cls = getattr(importlib.import_module(module_name), class_name)
    cfg = cfg_cls()

    run = data["run"]
    algo = data["algorithm"]
    pol = data["policy"]
    runner = data["runner"]

    cfg.num_steps_per_env = int(runner["num_steps_per_env"])
    cfg.max_iterations = int(run["max_iterations"])
    cfg.save_interval = int(run["save_interval"])
    cfg.experiment_name = run["name"]
    cfg.empirical_normalization = bool(runner.get("empirical_normalization", True))
    cfg.seed = int(run.get("seed", 42))
    cfg.device = run.get("device", "cuda:0")

    cfg.policy.init_noise_std = float(pol["init_noise_std"])
    cfg.policy.actor_hidden_dims = list(pol["actor_hidden_dims"])
    cfg.policy.critic_hidden_dims = list(pol["critic_hidden_dims"])
    cfg.policy.activation = pol["activation"]

    cfg.algorithm.value_loss_coef = float(algo["value_loss_coef"])
    cfg.algorithm.use_clipped_value_loss = bool(algo["use_clipped_value_loss"])
    cfg.algorithm.clip_param = float(algo["clip_param"])
    cfg.algorithm.entropy_coef = float(algo["entropy_coef"])
    cfg.algorithm.num_learning_epochs = int(algo["num_learning_epochs"])
    cfg.algorithm.num_mini_batches = int(algo["num_mini_batches"])
    cfg.algorithm.learning_rate = float(algo["learning_rate"])
    cfg.algorithm.schedule = algo["schedule"]
    cfg.algorithm.gamma = float(algo["gamma"])
    cfg.algorithm.lam = float(algo["lam"])
    cfg.algorithm.desired_kl = float(algo["desired_kl"])
    cfg.algorithm.max_grad_norm = float(algo["max_grad_norm"])

    return cfg
