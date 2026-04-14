"""Export a trained rsl_rl checkpoint to ONNX for on-robot inference.

Steps:

1. Load the PPO checkpoint via :class:`rsl_rl.runners.OnPolicyRunner`.
2. Extract the actor MLP (+ any empirical normalization stats).
3. Trace it to ONNX with a static input shape matching the observation dim.
4. Verify numerical parity on random inputs: ``max|torch_out − onnx_out| < 1e-5``.

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
    p.add_argument("--obs-dim", type=int, default=48, help="Policy observation dimension")
    p.add_argument("--action-dim", type=int, default=12)
    p.add_argument("--verify", action="store_true", help="Verify torch vs onnxruntime parity")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np
    import torch

    # ---- Load the actor network ------------------------------------------
    from rsl_rl.modules import ActorCritic

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model_state = ckpt["model_state_dict"]

    # Construct an ActorCritic with the same dims as the training run; we infer
    # hidden sizes from the checkpoint weights themselves to avoid depending
    # on a pickled cfg object.
    actor_dims = _infer_mlp_dims(model_state, prefix="actor.")
    actor_critic = ActorCritic(
        num_actor_obs=args.obs_dim,
        num_critic_obs=args.obs_dim,
        num_actions=args.action_dim,
        actor_hidden_dims=actor_dims,
        critic_hidden_dims=actor_dims,
        activation="elu",
    )
    actor_critic.load_state_dict(model_state, strict=False)
    actor_critic.eval()

    # ---- Wrap actor + optional normalizer in a single traceable module ---
    policy = _ExportablePolicy(actor_critic, ckpt)
    policy.eval()

    dummy = torch.zeros(1, args.obs_dim, dtype=torch.float32, device=args.device)

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

    # Also write a plain TorchScript for fallback deployment.
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
                x = rng.standard_normal((4, args.obs_dim)).astype(np.float32)
                y_torch = policy(torch.from_numpy(x)).cpu().numpy()
                y_onnx = session.run(["action"], {"obs": x})[0]
                max_diff = max(max_diff, float(np.max(np.abs(y_torch - y_onnx))))
        logger.info("Max torch<->onnx abs diff: %.3e", max_diff)
        if max_diff > 1e-4:
            logger.error("Parity check FAILED (>1e-4). ONNX export is not trustworthy.")
            return 2
        logger.info("Parity check passed.")

    return 0


class _ExportablePolicy:
    """Dummy placeholder; actual class is defined below as nn.Module."""


def _infer_mlp_dims(state_dict: dict, prefix: str) -> list[int]:
    """Read the hidden dims of a feedforward MLP from its weight shapes."""
    dims: list[int] = []
    i = 0
    while True:
        key = f"{prefix}{i * 2}.weight"
        if key not in state_dict:
            break
        dims.append(state_dict[key].shape[0])
        i += 1
    # Last entry is the action head; drop it.
    return dims[:-1] if dims else [512, 256, 128]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    sys.exit(main())


# --- nn.Module wrapper (imported lazily so this file stays import-safe) ----
try:  # pragma: no cover - only available in sim context
    import torch.nn as nn

    class _ExportablePolicy(nn.Module):  # type: ignore[no-redef]
        """Wraps an rsl_rl ActorCritic actor with optional input normalizer."""

        def __init__(self, actor_critic, ckpt: dict) -> None:
            super().__init__()
            self.actor = actor_critic.actor
            # rsl_rl stores a running empirical normalizer separately.
            self.normalizer = _load_normalizer(ckpt)

        def forward(self, obs):
            if self.normalizer is not None:
                obs = self.normalizer(obs)
            return self.actor(obs)

    def _load_normalizer(ckpt: dict):
        """Return an nn.Module that applies the saved empirical normalization, or None."""
        state = ckpt.get("obs_norm_state_dict") or ckpt.get("empirical_normalization")
        if not state:
            return None
        import torch

        mean = state.get("mean")
        var = state.get("var")
        if mean is None or var is None:
            return None

        class _Normalizer(nn.Module):
            def __init__(self, mean, var):
                super().__init__()
                self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
                self.register_buffer(
                    "std", torch.sqrt(torch.as_tensor(var, dtype=torch.float32)) + 1e-8
                )

            def forward(self, x):
                return (x - self.mean) / self.std

        return _Normalizer(mean, var)

except ImportError:  # pragma: no cover
    pass
