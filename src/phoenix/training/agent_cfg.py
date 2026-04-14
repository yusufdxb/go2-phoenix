"""Translate a Phoenix training YAML into an rsl_rl runner cfg."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def build_runner_cfg(data: dict[str, Any]) -> RslRlOnPolicyRunnerCfg:
    """Construct an rsl_rl :class:`RslRlOnPolicyRunnerCfg` from a dict."""
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
    )

    run = data["run"]
    algo = data["algorithm"]
    pol = data["policy"]
    runner = data["runner"]

    policy_cfg = RslRlPpoActorCriticCfg(
        init_noise_std=float(pol["init_noise_std"]),
        actor_hidden_dims=list(pol["actor_hidden_dims"]),
        critic_hidden_dims=list(pol["critic_hidden_dims"]),
        activation=pol["activation"],
    )
    algo_cfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=float(algo["value_loss_coef"]),
        use_clipped_value_loss=bool(algo["use_clipped_value_loss"]),
        clip_param=float(algo["clip_param"]),
        entropy_coef=float(algo["entropy_coef"]),
        num_learning_epochs=int(algo["num_learning_epochs"]),
        num_mini_batches=int(algo["num_mini_batches"]),
        learning_rate=float(algo["learning_rate"]),
        schedule=algo["schedule"],
        gamma=float(algo["gamma"]),
        lam=float(algo["lam"]),
        desired_kl=float(algo["desired_kl"]),
        max_grad_norm=float(algo["max_grad_norm"]),
    )

    cfg = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=int(runner["num_steps_per_env"]),
        max_iterations=int(run["max_iterations"]),
        save_interval=int(run["save_interval"]),
        experiment_name=run["name"],
        empirical_normalization=bool(runner.get("empirical_normalization", True)),
        policy=policy_cfg,
        algorithm=algo_cfg,
        seed=int(run.get("seed", 42)),
        device=run.get("device", "cuda:0"),
    )
    return cfg
