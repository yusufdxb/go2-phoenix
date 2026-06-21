"""Pure math for the stand-fixture scenario (CI-safe, no torch/isaaclab).

Kept separate from :mod:`phoenix.sim_env.fixture_hold` (which touches the live
Isaac Lab env) so the geometry can be unit-tested without a sim app.
"""

from __future__ import annotations

import math

__all__ = ["roll_quat_wxyz"]


def roll_quat_wxyz(roll_rad: float) -> tuple[float, float, float, float]:
    """Unit quaternion (w, x, y, z) for a rotation of ``roll_rad`` about +x.

    Isaac Lab's ``write_root_pose_to_sim`` expects (w, x, y, z) order. A trunk
    roll tilts the left and right legs oppositely, modelling the stand fixture's
    measured ~0.32 rad L/R asymmetry as a single fixed roll.
    """
    half = 0.5 * roll_rad
    return (math.cos(half), math.sin(half), 0.0, 0.0)
