"""Quasi-random sampling of physics variations for replay episodes.

A Halton sequence gives better coverage of the parameter cube than
independent uniform samples for small N (the regime we care about: 16
variations per failure).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

_FIRST_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _halton(index: int, base: int) -> float:
    """Radical-inverse Halton value for ``index`` in the given base."""
    result = 0.0
    f = 1.0
    i = index
    while i > 0:
        f /= base
        result += f * (i % base)
        i //= base
    return result


def halton_sequence(n: int, dim: int, *, skip: int = 1) -> np.ndarray:
    """Return an ``(n, dim)`` array of Halton values in [0, 1)."""
    if dim > len(_FIRST_PRIMES):
        raise ValueError(f"Halton supports up to {len(_FIRST_PRIMES)} dims, got {dim}")
    out = np.empty((n, dim), dtype=np.float64)
    for i in range(n):
        for d in range(dim):
            out[i, d] = _halton(i + skip, _FIRST_PRIMES[d])
    return out


@dataclass(frozen=True)
class VariationSample:
    """One concrete perturbation relative to a logged trajectory's center."""

    friction_delta: float
    mass_delta_kg: float
    push_velocity_delta: float
    push_yaw_delta: float


class VariationSampler:
    """Generates :class:`VariationSample` instances from a bounds dict.

    The bounds are (lo, hi) intervals applied as *deltas* to the center
    physics config recorded in the source trajectory.
    """

    _FIELDS: Sequence[str] = (
        "friction_delta",
        "mass_delta_kg",
        "push_velocity_delta",
        "push_yaw_delta",
    )

    def __init__(self, bounds: Mapping[str, Sequence[float]], *, seed: int = 17) -> None:
        self.bounds = {k: tuple(v) for k, v in bounds.items()}
        self.seed = seed
        missing = [f for f in self._FIELDS if f not in self.bounds]
        if missing:
            raise KeyError(f"variation bounds missing keys: {missing}")

    def sample(self, n: int) -> list[VariationSample]:
        """Draw ``n`` quasi-random samples."""
        raw = halton_sequence(n, len(self._FIELDS), skip=max(self.seed, 1))
        samples: list[VariationSample] = []
        for row in raw:
            values = {}
            for i, field in enumerate(self._FIELDS):
                lo, hi = self.bounds[field]
                values[field] = float(lo + row[i] * (hi - lo))
            samples.append(VariationSample(**values))
        return samples
