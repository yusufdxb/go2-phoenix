"""Checkpoint loading helpers shared by baseline / fine-tune / evaluate.

rsl_rl 3.0's :meth:`OnPolicyRunner.load` silently accepts ``strict=False``
and returns without telling the caller what was actually restored. When
warm-starting a fine-tune from a baseline it's important to verify that
the learned actor weights, the Gaussian ``std_param``, and the empirical
observation normalizer (if any) all round-tripped. This helper reports
exactly that so a bad load is visible in the training log instead of
surfacing as a mysteriously bad first iteration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from rsl_rl.runners import OnPolicyRunner

logger = logging.getLogger("phoenix.training.checkpoint")

# Keys the rsl_rl PPO checkpoint uses for the actor/critic modules.
_ACTOR_KEY = "actor_state_dict"
_CRITIC_KEY = "critic_state_dict"

# Well-known parameter / buffer names inside a rsl_rl 3.x MLPModel actor.
_STD_PARAM_KEYS = ("distribution.std_param", "distribution.log_std_param")
_OBS_NORM_KEYS = ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer.count")


def load_runner_checkpoint(
    runner: OnPolicyRunner,
    path: str | Path,
    *,
    load_actor: bool = True,
    load_critic: bool = True,
    load_optimizer: bool = False,
    load_iteration: bool = False,
) -> dict[str, Any]:
    """Load a rsl_rl 3.x checkpoint into ``runner`` and log what was restored.

    Returns a summary dict so callers can assert what round-tripped and
    surface it in training logs. All verification is done with
    ``torch.allclose`` against the live actor/critic state dicts after the
    load, so a silent partial load shows up as ``actor_match=False``.
    """
    import torch

    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    load_cfg = {
        "actor": bool(load_actor),
        "critic": bool(load_critic),
        "optimizer": bool(load_optimizer),
        "iteration": bool(load_iteration),
    }
    runner.load(str(ckpt_path), load_cfg=load_cfg, strict=False)

    # Re-read the checkpoint so we can verify what actually landed in the
    # live modules. ``weights_only=False`` matches rsl_rl's own load call.
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")

    summary: dict[str, Any] = {
        "path": str(ckpt_path),
        "iter": ckpt.get("iter"),
        "load_cfg": load_cfg,
    }

    if load_actor and _ACTOR_KEY in ckpt:
        summary.update(_verify_module(runner.alg.actor, ckpt[_ACTOR_KEY], prefix="actor"))
    if load_critic and _CRITIC_KEY in ckpt:
        summary.update(_verify_module(runner.alg.critic, ckpt[_CRITIC_KEY], prefix="critic"))

    # Surface the Gaussian std we are *actually* starting fine-tuning from.
    std_key = (
        next((k for k in _STD_PARAM_KEYS if k in ckpt[_ACTOR_KEY]), None) if load_actor else None
    )
    if std_key is not None:
        std_tensor = ckpt[_ACTOR_KEY][std_key].detach().float()
        summary["actor_std_mean"] = float(std_tensor.mean().item())
        summary["actor_std_min"] = float(std_tensor.min().item())
        summary["actor_std_max"] = float(std_tensor.max().item())

    # Empirical obs normalizer buffers, if the actor used them during
    # baseline training. Upstream GO2 cfg disables this by default, so
    # it's expected to be absent from the checkpoint — the helper just
    # notes whether it was there so a config mismatch is obvious.
    summary["actor_obs_normalizer_in_ckpt"] = bool(
        load_actor and any(k in ckpt[_ACTOR_KEY] for k in _OBS_NORM_KEYS)
    )

    logger.info("Checkpoint summary: %s", summary)
    return summary


def _verify_module(
    live_module: Any,
    ckpt_state: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    """Compare a live nn.Module's state_dict to the saved state dict."""
    import torch

    live_state = live_module.state_dict()
    common_keys = [k for k in ckpt_state if k in live_state]
    missing_in_live = [k for k in ckpt_state if k not in live_state]
    missing_in_ckpt = [k for k in live_state if k not in ckpt_state]

    mismatches: list[str] = []
    for k in common_keys:
        ck = ckpt_state[k].to(live_state[k].device, dtype=live_state[k].dtype)
        if not torch.allclose(ck, live_state[k]):
            mismatches.append(k)

    return {
        f"{prefix}_match": not mismatches and not missing_in_live,
        f"{prefix}_mismatched_keys": mismatches,
        f"{prefix}_ckpt_only_keys": missing_in_live,
        f"{prefix}_live_only_keys": missing_in_ckpt,
    }
