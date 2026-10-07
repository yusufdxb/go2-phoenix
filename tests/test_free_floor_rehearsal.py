"""The FREE_FLOOR choreography against the fake GO2 (scripts/free_floor_rehearsal.py).

Needs ROS 2 Humble, the unitree_go messages and the archived model_2750 bundle, and a clean tracked
tree (the bridge refuses a dirty one); it skips cleanly anywhere else. Two scenarios run by default
(the happy path and one lifecycle failure); run the whole battery with
``python3 scripts/free_floor_rehearsal.py``.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import free_floor_rehearsal as rh  # noqa: E402

pytestmark = [pytest.mark.ros, pytest.mark.slow]


@pytest.mark.skipif(rh.available() is not None, reason="needs ROS 2, unitree messages, model_2750")
@pytest.mark.parametrize("scenario", ["happy_reject", "walk_bridge_dies_during_cmd"])
def test_rehearsal_scenario(scenario):
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        pytest.skip("tracked tree is dirty; the bridge refuses to go live on it")
    assert rh.main([scenario]) == 0
