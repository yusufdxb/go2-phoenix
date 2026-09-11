"""Reconstruct real-world failure episodes in Isaac Sim.

Given a Parquet trajectory written by :class:`phoenix.real_world.TrajectoryLogger`,
this module:

1. Reads the initial state (pose + joint positions + velocity command).
2. Samples N variation configurations via Halton quasi-random sequences.
3. Rolls each variation forward in Isaac Sim, producing training seeds for
   the adaptation loop.

The pure-python pieces (variation sampling, trajectory IO) are testable
in CI; ``reconstruct.py`` itself needs Isaac Lab.
"""

from .controller_history import (
    REPLAY_APPROXIMATE,
    REPLAY_EXACT,
    REPLAY_STATE_ONLY,
    ControllerHistory,
)
from .failure_capsule import CAPSULE_SCHEMA_VERSION, write_failure_capsule
from .trajectory_reader import TrajectoryReader, load_initial_state
from .variations import VariationSample, VariationSampler

__all__ = [
    "CAPSULE_SCHEMA_VERSION",
    "ControllerHistory",
    "REPLAY_APPROXIMATE",
    "REPLAY_EXACT",
    "REPLAY_STATE_ONLY",
    "TrajectoryReader",
    "load_initial_state",
    "VariationSample",
    "VariationSampler",
    "write_failure_capsule",
]
