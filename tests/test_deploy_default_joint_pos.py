"""Every deploy config must carry the training-time nominal joint pose.

``control.default_joint_pos`` is not a cosmetic default. The locomotion action
term is built with ``use_default_offset=True``, so the same vector is used
twice on the robot:

* as the joint-position observation reference, ``joint_pos - default_q``
  (``phoenix/sim2real/observation.py``), and
* as the action offset, ``target = default_q + action_scale * action``
  (``phoenix/sim2real/ros2_policy_node.py``).

If it disagrees with the pose the policy was trained around, the policy is fed
off-distribution observations and its output is applied about the wrong origin,
on every affected joint, on every tick. Five of six configs once had all four
hips at 0.0 instead of L=+0.1 / R=-0.1; this test exists so a sixth config
cannot reintroduce that.

Pure YAML and pure python on purpose: it must run in the CI job that has no
Isaac Lab, no torch, and no ROS 2.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = REPO_ROOT / "configs" / "sim2real"

# The authority is IsaacLab's ``UNITREE_GO2_CFG.init_state.joint_pos`` in
# ``source/isaaclab_assets/isaaclab_assets/robots/unitree.py``, which is
# expressed as regex patterns over joint names:
#
#     ".*L_hip_joint":      0.1
#     ".*R_hip_joint":     -0.1
#     "F[L,R]_thigh_joint": 0.8
#     "R[L,R]_thigh_joint": 1.0
#     ".*_calf_joint":     -1.5
#
# expanded here to the twelve GO2 joints. Change this table only when the
# training asset changes, never to make a deploy config pass.
TRAINING_DEFAULT_JOINT_POS: dict[str, float] = {
    "FL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RL_hip_joint": 0.1,
    "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RL_calf_joint": -1.5,
    "RR_calf_joint": -1.5,
}

# Discovered, not hardcoded: a new configs/sim2real/deploy*.yaml is covered the
# moment it lands, without anyone remembering to update this file.
DEPLOY_CONFIGS = sorted(DEPLOY_DIR.glob("deploy*.yaml"))


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_deploy_configs_are_discovered() -> None:
    """Guard the glob itself: an empty parametrization would pass silently."""
    assert DEPLOY_DIR.is_dir(), f"missing deploy config directory: {DEPLOY_DIR}"
    assert len(DEPLOY_CONFIGS) >= 6, (
        f"expected at least the six known deploy configs in {DEPLOY_DIR}, "
        f"found {[p.name for p in DEPLOY_CONFIGS]}"
    )


@pytest.mark.parametrize("config_path", DEPLOY_CONFIGS, ids=lambda p: p.name)
def test_default_joint_pos_matches_training(config_path: Path) -> None:
    cfg = _load(config_path)
    control = cfg.get("control")
    assert isinstance(control, dict), f"{config_path.name}: missing a `control` block"

    default_joint_pos = control.get("default_joint_pos")
    assert isinstance(
        default_joint_pos, dict
    ), f"{config_path.name}: missing `control.default_joint_pos`"

    assert set(default_joint_pos) == set(TRAINING_DEFAULT_JOINT_POS), (
        f"{config_path.name}: default_joint_pos joint names differ from the training asset. "
        f"missing={sorted(set(TRAINING_DEFAULT_JOINT_POS) - set(default_joint_pos))} "
        f"unexpected={sorted(set(default_joint_pos) - set(TRAINING_DEFAULT_JOINT_POS))}"
    )

    wrong = {
        name: (value, TRAINING_DEFAULT_JOINT_POS[name])
        for name, value in default_joint_pos.items()
        if value != pytest.approx(TRAINING_DEFAULT_JOINT_POS[name], abs=1e-9)
    }
    assert not wrong, (
        f"{config_path.name}: default_joint_pos disagrees with the training pose from "
        f"IsaacLab UNITREE_GO2_CFG.init_state (joint: got, expected) -> {wrong}. "
        "use_default_offset=True makes this both the action offset and the "
        "joint-position observation reference, so a mismatch is a deployment bug."
    )


@pytest.mark.parametrize("config_path", DEPLOY_CONFIGS, ids=lambda p: p.name)
def test_joint_order_covers_default_joint_pos(config_path: Path) -> None:
    """``default_q`` is built by indexing default_joint_pos with joint_order.

    A name present in one and absent from the other is a KeyError at node
    construction, or a joint silently commanded about the wrong origin.
    """
    cfg = _load(config_path)
    joint_order = cfg.get("joint_order")
    assert isinstance(joint_order, list), f"{config_path.name}: missing `joint_order`"
    assert len(joint_order) == len(set(joint_order)), f"{config_path.name}: duplicate joint_order"
    assert set(joint_order) == set(
        TRAINING_DEFAULT_JOINT_POS
    ), f"{config_path.name}: joint_order does not name exactly the twelve GO2 joints"
