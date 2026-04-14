"""Phoenix GO2 simulation environment.

Wraps Isaac Lab's ``Isaac-Velocity-Rough-Unitree-Go2-v0`` task with
failure-oriented domain randomization, terrain overlays, and perturbation
scheduling defined in YAML. The :func:`build_env_cfg` factory is the single
entry point used by the training, adaptation, and replay modules.
"""

from .config_loader import PhoenixConfig, load_layered_config
from .go2_env_cfg import build_env_cfg, make_gym_env

__all__ = [
    "PhoenixConfig",
    "load_layered_config",
    "build_env_cfg",
    "make_gym_env",
]
