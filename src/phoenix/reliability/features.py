"""Policy-internal feature extraction for the OOD monitor.

The monitor scores two feature sources and the eval harness compares them
("latents vs obs"):

* **latent** — the post-activation outputs of the policy MLP's hidden
  layers (mid + penultimate by default). This is the "reads the model's
  own internal state" signal.
* **obs** — the (normalized) observation vector the policy consumes.

Phoenix's exported actor is a plain ``nn.Sequential`` of
``Linear, ELU, Linear, ELU, ..., Linear`` (see
:func:`phoenix.sim2real.export._build_actor_mlp`), with an optional
:class:`EmpiricalNormalization` applied to the observation first. That
structure is simple enough to reproduce exactly in numpy, which buys two
things:

1. A **deployment-agnostic feature contract** the on-robot torch
   extractor must match bit-for-bit (parity, mirroring the ONNX parity
   gate philosophy already in this repo).
2. A monitor pipeline that is **unit-testable in CI without torch or
   Isaac** — the numpy reference stands in for the policy.

:class:`TorchActivationExtractor` is the on-robot path: it hooks the ELU
outputs of the real Sequential. It is lazy about torch (no module-level
import) so this file stays in the pure-python CI lane, exactly like the
rest of ``phoenix.reliability``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


LinearLayer = tuple[np.ndarray, np.ndarray]  # (weight (out,in), bias (out,))


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ELU activation, matching ``torch.nn.ELU`` (default ``alpha=1``).

    ``expm1`` keeps the negative branch numerically clean for large
    magnitudes.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 0, x, alpha * np.expm1(x))


def _tap_index(name: str, n_hidden: int) -> int:
    """Resolve a symbolic tap name to a hidden-layer index."""
    if n_hidden < 1:
        raise ValueError("policy has no hidden layers to tap")
    table = {
        "first": 0,
        "mid": (n_hidden - 1) // 2,
        "penultimate": n_hidden - 1,
        "last": n_hidden - 1,
    }
    if name not in table:
        raise KeyError(f"unknown tap {name!r}; choose from {sorted(table)}")
    return table[name]


def forward_hidden(
    obs: np.ndarray,
    linears: Sequence[LinearLayer],
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Numpy reference forward returning every post-ELU hidden activation.

    ``obs`` is ``(n, obs_dim)`` (1-D promoted to a single row). ``linears``
    is the ordered list of ``(weight, bias)`` for each ``Linear`` in the
    actor Sequential. When ``mean`` / ``std`` are given, the observation is
    normalized as ``(obs - mean) / std`` first — the exact transform the
    exported policy applies (rsl_rl EmpiricalNormalization; ``std`` here is
    already ``sqrt(var) + eps``).

    Returns one array per hidden layer (all Linears except the final one,
    which is the action head and has no ELU).
    """
    x = np.asarray(obs, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if mean is not None and std is not None:
        x = (x - np.asarray(mean, np.float64)) / np.asarray(std, np.float64)

    hidden: list[np.ndarray] = []
    n = len(linears)
    for i, (w, b) in enumerate(linears):
        x = x @ np.asarray(w, np.float64).T + np.asarray(b, np.float64)
        if i < n - 1:  # ELU after every Linear except the action head
            x = elu(x)
            hidden.append(x)
    return hidden


def policy_features(
    obs: np.ndarray,
    linears: Sequence[LinearLayer],
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    taps: Sequence[str] = ("mid", "penultimate"),
) -> dict[str, np.ndarray]:
    """Extract the monitor's two feature sources from a batch of observations.

    Returns ``{"latent": ..., "obs": ...}`` where ``latent`` is the
    concatenation of the tapped hidden activations and ``obs`` is the
    (normalized, if stats given) observation vector. Both are ``(n, d)``.
    The eval harness fits an independent scorer on each and compares them.
    """
    x = np.asarray(obs, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    normed = x
    if mean is not None and std is not None:
        normed = (x - np.asarray(mean, np.float64)) / np.asarray(std, np.float64)

    hidden = forward_hidden(x, linears, mean=mean, std=std)
    idx = [_tap_index(t, len(hidden)) for t in taps]
    latent = np.concatenate([hidden[i] for i in idx], axis=1)
    return {"latent": latent, "obs": normed}


class TorchActivationExtractor:
    """On-robot feature tap: hooks the ELU outputs of the real policy.

    Wraps the exported actor ``nn.Sequential`` and registers forward hooks
    on its ``ELU`` modules so a single forward pass yields both the action
    and the tapped hidden activations, with no change to the policy graph.
    Produces the SAME ``{"latent", "obs"}`` contract as
    :func:`policy_features`; a parity test asserts they match the numpy
    reference within tolerance (mirroring this repo's ONNX parity gate).

    Torch is imported lazily so importing this module never pulls torch
    into the CI lane. NOT exercised in the pure-python CI (torch absent);
    the parity test runs wherever torch is installed (the Isaac / deploy
    environment).
    """

    def __init__(
        self,
        actor: "torch.nn.Module",
        *,
        taps: Sequence[str] = ("mid", "penultimate"),
        normalizer: "torch.nn.Module | None" = None,
    ) -> None:
        import torch  # noqa: F401  (lazy)

        self.actor = actor
        self.normalizer = normalizer
        self.taps = tuple(taps)
        self._elu_indices = [
            i for i, m in enumerate(actor) if m.__class__.__name__ == "ELU"
        ]
        if not self._elu_indices:
            raise ValueError("actor has no ELU layers to tap")
        self._captured: dict[int, np.ndarray] = {}
        self._handles = []
        for i in self._elu_indices:
            self._handles.append(actor[i].register_forward_hook(self._make_hook(i)))

    def _make_hook(self, layer_i: int):
        def hook(_module, _inp, out):
            self._captured[layer_i] = out.detach().cpu().numpy().astype(np.float64)

        return hook

    def __call__(self, obs: "torch.Tensor") -> dict[str, np.ndarray]:
        import torch

        with torch.inference_mode():
            x = obs
            if self.normalizer is not None:
                x = self.normalizer(x)
            normed = x.detach().cpu().numpy().astype(np.float64)
            self._captured.clear()
            self.actor(x)

        hidden = [self._captured[i] for i in self._elu_indices]
        idx = [_tap_index(t, len(hidden)) for t in self.taps]
        latent = np.concatenate([hidden[i] for i in idx], axis=1)
        return {"latent": latent, "obs": normed}

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
