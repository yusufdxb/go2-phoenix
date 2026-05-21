"""Export a trained rsl_rl checkpoint to ONNX for on-robot inference.

rsl_rl 3.0.x stores the actor as ``actor_state_dict`` with keys
``mlp.<2k>.weight/bias``. We rebuild a plain ``torch.nn.Sequential`` that
mirrors those shapes (no dependency on rsl_rl's internal classes for
inference), trace it to ONNX, and verify numerical parity against
onnxruntime on 16 random inputs.

The verify step is what makes the deploy script trustworthy — a silent
normalization mismatch is the most common cause of sim-to-real failure.

Observation normalization
-------------------------
When trained with ``empirical_normalization: true`` (the default for
every config in ``configs/train/``), rsl_rl wraps the MLP in an
:class:`rsl_rl.modules.EmpiricalNormalization` layer. In rsl_rl 3.x that
layer lives *inside* the actor module, so its running statistics are
serialized inside ``actor_state_dict`` under the buffer keys
``obs_normalizer._mean`` / ``obs_normalizer._var``. The exported policy
MUST reapply that same normalization or the deployed policy sees raw,
un-normalized observations — a silent train/deploy mismatch that the
ONNX-vs-TorchScript parity gate cannot catch, because both sides export
the same (wrong) wrapper. :func:`_extract_normalizer_stats` locates
those buffers; :class:`_Normalizer` reproduces rsl_rl's exact
``(x - mean) / (sqrt(var) + eps)`` with ``eps = 1e-2``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("phoenix.sim2real.export")

# Must match rsl_rl.modules.EmpiricalNormalization's default ``eps``.
# rsl_rl normalizes as ``(x - mean) / (std + eps)`` with eps=1e-2; using a
# different epsilon at export time silently shifts every observation the
# deployed policy sees, badly so for near-constant (low-variance) dims.
_RSL_RL_NORM_EPS = 1e-2

# Buffer name fragments rsl_rl 3.x uses for the in-actor EmpiricalNormalization.
_NORM_MEAN_SUFFIXES = ("obs_normalizer._mean", "obs_normalizer.mean")
_NORM_VAR_SUFFIXES = ("obs_normalizer._var", "obs_normalizer.var")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export rsl_rl checkpoint to ONNX.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="ONNX output path")
    p.add_argument(
        "--obs-dim",
        type=int,
        default=None,
        help="Override inferred policy observation dimension",
    )
    p.add_argument(
        "--action-dim",
        type=int,
        default=None,
        help="Override inferred action dimension",
    )
    p.add_argument("--verify", action="store_true", help="Verify torch vs onnxruntime parity")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np
    import torch

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    actor_sd = _extract_actor_state_dict(ckpt)

    layer_keys = sorted(
        (k for k in actor_sd if k.endswith(".weight") and ("mlp" in k or "actor" in k)),
        key=_layer_index,
    )
    if not layer_keys:
        raise RuntimeError(f"No MLP weights found in actor state_dict. Keys: {list(actor_sd)}")

    obs_dim = args.obs_dim if args.obs_dim is not None else int(actor_sd[layer_keys[0]].shape[1])
    action_dim = (
        args.action_dim if args.action_dim is not None else int(actor_sd[layer_keys[-1]].shape[0])
    )
    hidden_dims = [int(actor_sd[k].shape[0]) for k in layer_keys[:-1]]
    logger.info("Inferred obs_dim=%d action_dim=%d hidden=%s", obs_dim, action_dim, hidden_dims)

    actor = _build_actor_mlp(obs_dim, action_dim, hidden_dims)
    _load_actor_weights(actor, actor_sd, layer_keys)
    actor.to(args.device).eval()

    policy = _ExportablePolicy(actor, ckpt, actor_sd).to(args.device).eval()

    dummy = torch.zeros(1, obs_dim, dtype=torch.float32, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        policy,
        dummy,
        str(args.output),
        input_names=["obs"],
        output_names=["action"],
        opset_version=args.opset,
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )
    logger.info("Exported ONNX -> %s", args.output)

    # TorchScript fallback for environments without onnxruntime.
    ts_path = args.output.with_suffix(".pt")
    scripted = torch.jit.trace(policy, dummy)
    scripted.save(str(ts_path))
    logger.info("Wrote TorchScript fallback -> %s", ts_path)

    if args.verify:
        import onnxruntime as ort

        session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        max_diff = 0.0
        with torch.inference_mode():
            for _ in range(16):
                x = rng.standard_normal((4, obs_dim)).astype(np.float32)
                y_torch = policy(torch.from_numpy(x).to(args.device)).cpu().numpy()
                y_onnx = session.run(["action"], {"obs": x})[0]
                max_diff = max(max_diff, float(np.max(np.abs(y_torch - y_onnx))))
        logger.info("Max torch<->onnx abs diff: %.3e", max_diff)
        if max_diff > 1e-4:
            logger.error("Parity check FAILED (>1e-4). ONNX export is not trustworthy.")
            return 2
        logger.info("Parity check passed.")

    return 0


def _extract_actor_state_dict(ckpt: dict) -> dict:
    """Return the actor weights regardless of rsl_rl version.

    rsl_rl >=3.0 stores ``actor_state_dict``. Older versions use
    ``model_state_dict`` with an ``actor.*`` prefix.
    """
    if "actor_state_dict" in ckpt:
        return ckpt["actor_state_dict"]
    if "model_state_dict" in ckpt:
        prefix = "actor."
        return {
            k[len(prefix) :]: v for k, v in ckpt["model_state_dict"].items() if k.startswith(prefix)
        }
    raise KeyError(f"checkpoint has no actor weights; keys={list(ckpt)}")


def _layer_index(key: str) -> int:
    """Extract the numeric layer index from keys like ``mlp.4.weight``."""
    for piece in key.split("."):
        if piece.isdigit():
            return int(piece)
    return -1


def _build_actor_mlp(obs_dim: int, action_dim: int, hidden_dims: list[int]):
    import torch.nn as nn

    layers: list[nn.Module] = []
    prev = obs_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ELU())
        prev = h
    layers.append(nn.Linear(prev, action_dim))
    return nn.Sequential(*layers)


def _load_actor_weights(module, actor_sd: dict, layer_keys: list[str]) -> None:
    """Copy weights into the Sequential built by :func:`_build_actor_mlp`.

    The Sequential's layer indices are 0, 2, 4, ... (Linear, ELU, Linear, ELU, ...);
    rsl_rl keys use the same even indices under ``mlp.``.
    """
    import torch

    seq = module
    for key in layer_keys:
        idx = _layer_index(key)
        w = actor_sd[key]
        b = actor_sd[key.replace(".weight", ".bias")]
        with torch.no_grad():
            seq[idx].weight.copy_(w)
            seq[idx].bias.copy_(b)


def _find_buffer(state: dict[str, Any], suffixes: tuple[str, ...]) -> Any | None:
    """Return the first value in ``state`` whose key ends with any suffix.

    rsl_rl 3.x stores the EmpiricalNormalization buffers inside the actor
    state dict under ``obs_normalizer._mean`` / ``obs_normalizer._var``.
    Older / variant layouts may key them slightly differently or nest
    them under a sub-module prefix, so we match by suffix.
    """
    for key, value in state.items():
        if any(key.endswith(s) for s in suffixes):
            return value
    return None


def _extract_normalizer_stats(ckpt: dict, actor_sd: dict) -> tuple[Any, Any] | None:
    """Locate the observation-normalizer (mean, var) statistics, if any.

    Pure dict traversal (no torch) so the lookup logic is unit-testable
    in CI. Searches, in order:

    1. ``actor_sd`` itself — rsl_rl 3.x keeps the EmpiricalNormalization
       buffers inside ``actor_state_dict``.
    2. Legacy top-level checkpoint dicts (``obs_norm_state_dict`` etc.)
       used by older rsl_rl / hand-rolled exporters.

    Returns ``(mean, var)`` (whatever array-like type the checkpoint
    stored) or ``None`` when the policy was trained without obs
    normalization.
    """
    mean = _find_buffer(actor_sd, _NORM_MEAN_SUFFIXES)
    var = _find_buffer(actor_sd, _NORM_VAR_SUFFIXES)
    if mean is not None and var is not None:
        return mean, var

    for legacy_key in (
        "obs_norm_state_dict",
        "empirical_normalization",
        "obs_normalizer_state_dict",
    ):
        state = ckpt.get(legacy_key)
        if not isinstance(state, dict):
            continue
        l_mean = _find_buffer(state, _NORM_MEAN_SUFFIXES) or state.get("mean")
        l_var = _find_buffer(state, _NORM_VAR_SUFFIXES) or state.get("var")
        if l_mean is not None and l_var is not None:
            return l_mean, l_var

    return None


# --- nn.Module wrappers -----------------------------------------------------

try:  # pragma: no cover - only meaningful with torch available
    import torch
    import torch.nn as nn

    class _ExportablePolicy(nn.Module):
        """Wraps the actor Sequential + optional input normalizer."""

        def __init__(self, actor: nn.Module, ckpt: dict, actor_sd: dict) -> None:
            super().__init__()
            self.actor = actor
            self.normalizer = _load_normalizer(ckpt, actor_sd)
            if self.normalizer is None:
                logger.info("No obs normalizer found in checkpoint; exporting raw-obs policy.")
            else:
                logger.info(
                    "Reconstructed obs normalizer (eps=%.0e) from checkpoint.",
                    _RSL_RL_NORM_EPS,
                )

        def forward(self, obs):  # noqa: D401
            if self.normalizer is not None:
                obs = self.normalizer(obs)
            return self.actor(obs)

    class _Normalizer(nn.Module):
        """Applies saved empirical normalization to the observation tensor.

        Reproduces ``rsl_rl.modules.EmpiricalNormalization.forward`` exactly:
        ``(x - mean) / (sqrt(var) + eps)`` with ``eps = 1e-2``. The
        epsilon is part of the trained transform — it is added to the
        standard deviation, not buried under it — so it must match the
        training-time value or low-variance observation dims are scaled
        wrong by orders of magnitude.
        """

        def __init__(self, mean: torch.Tensor, var: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("mean", mean.to(torch.float32))
            std = torch.sqrt(var.to(torch.float32)) + _RSL_RL_NORM_EPS
            self.register_buffer("std", std)

        def forward(self, x):
            return (x - self.mean) / self.std

    def _load_normalizer(ckpt: dict, actor_sd: dict):
        """Rebuild the EmpiricalNormalization layer from the checkpoint, if present.

        rsl_rl 3.x serializes the normalizer buffers inside
        ``actor_state_dict``; :func:`_extract_normalizer_stats` finds them
        wherever they live. Returns ``None`` when the policy was trained
        without obs normalization.
        """
        stats = _extract_normalizer_stats(ckpt, actor_sd)
        if stats is None:
            return None
        mean, var = stats
        return _Normalizer(
            torch.as_tensor(mean, dtype=torch.float32).reshape(-1),
            torch.as_tensor(var, dtype=torch.float32).reshape(-1),
        )

except ImportError:  # pragma: no cover - CI context

    class _ExportablePolicy:  # type: ignore[no-redef]
        """Placeholder for CI where torch is missing."""

        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("torch required for _ExportablePolicy")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    sys.exit(main())
