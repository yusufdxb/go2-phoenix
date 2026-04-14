"""Layered YAML config loader for Phoenix.

Supports a lightweight ``defaults:`` key that lists other YAML files
(relative to the current file) to merge *before* the current document.
This mirrors the subset of Hydra we actually use, without requiring
Hydra to run inside Isaac Lab's Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass
class PhoenixConfig:
    """Thin wrapper around an OmegaConf tree with source tracking."""

    cfg: DictConfig
    source: Path

    def to_container(self) -> dict[str, Any]:
        return OmegaConf.to_container(self.cfg, resolve=True)  # type: ignore[return-value]


def _merge_defaults(path: Path, seen: set[Path]) -> DictConfig:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Circular config default detected at {resolved}")
    seen.add(resolved)

    raw = OmegaConf.load(path)
    assert isinstance(raw, DictConfig)

    defaults = raw.pop("defaults", None)
    merged: DictConfig = OmegaConf.create({})

    if defaults is not None:
        for entry in defaults:
            sibling = path.parent / f"{entry}.yaml"
            if not sibling.exists():
                raise FileNotFoundError(
                    f"Default '{entry}' referenced by {path} not found at {sibling}"
                )
            merged = OmegaConf.merge(merged, _merge_defaults(sibling, seen))  # type: ignore[assignment]

    merged = OmegaConf.merge(merged, raw)  # type: ignore[assignment]
    return merged  # type: ignore[return-value]


def load_layered_config(path: str | Path) -> PhoenixConfig:
    """Load a YAML file, resolving ``defaults:`` chains and interpolations."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    merged = _merge_defaults(p, seen=set())
    OmegaConf.resolve(merged)
    return PhoenixConfig(cfg=merged, source=p.resolve())
