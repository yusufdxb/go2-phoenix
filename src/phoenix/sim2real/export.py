"""Export a trained rsl_rl checkpoint to ONNX for on-robot inference.

rsl_rl 3.0.x stores the actor as ``actor_state_dict`` with keys
``mlp.<2k>.weight/bias``. We rebuild a plain ``torch.nn.Sequential`` that
mirrors those shapes (no dependency on rsl_rl's internal classes for
inference), trace it to ONNX, and verify numerical parity against
onnxruntime on 16 random inputs.

The verify step is what makes the deploy script trustworthy — a silent
normalization mismatch is the most common cause of sim-to-real failure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("phoenix.sim2real.export")


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

    policy = _ExportablePolicy(actor, ckpt).to(args.device).eval()

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


# --- nn.Module wrappers -----------------------------------------------------

try:  # pragma: no cover - only meaningful with torch available
    import torch
    import torch.nn as nn

    class _ExportablePolicy(nn.Module):
        """Wraps the actor Sequential + optional input normalizer."""

        def __init__(self, actor: nn.Module, ckpt: dict) -> None:
            super().__init__()
            self.actor = actor
            self.normalizer = _load_normalizer(ckpt)

        def forward(self, obs):  # noqa: D401
            if self.normalizer is not None:
                obs = self.normalizer(obs)
            return self.actor(obs)

    class _Normalizer(nn.Module):
        """Applies saved empirical normalization to the observation tensor."""

        def __init__(self, mean: torch.Tensor, var: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("mean", mean.to(torch.float32))
            self.register_buffer("std", torch.sqrt(var.to(torch.float32)) + 1e-8)

        def forward(self, x):
            return (x - self.mean) / self.std

    def _load_normalizer(ckpt: dict):
        """Rebuild the EmpiricalNormalization layer from the checkpoint, if present."""
        state = (
            ckpt.get("obs_norm_state_dict")
            or ckpt.get("empirical_normalization")
            or ckpt.get("obs_normalizer_state_dict")
            or {}
        )
        mean = state.get("mean") if isinstance(state, dict) else None
        var = state.get("var") if isinstance(state, dict) else None
        if mean is None or var is None:
            return None
        return _Normalizer(
            torch.as_tensor(mean, dtype=torch.float32),
            torch.as_tensor(var, dtype=torch.float32),
        )

except ImportError:  # pragma: no cover - CI context

    class _ExportablePolicy:  # type: ignore[no-redef]
        """Placeholder for CI where torch is missing."""

        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("torch required for _ExportablePolicy")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    sys.exit(main())
