"""The joint model the final actuator gate relies on: orders, pose, hard limits.

Every permutation and every joint is covered, because the failure these guard
against (a leg swap, a joint with no limit) passes every shape check.
"""

from __future__ import annotations

import ast
import itertools
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from phoenix.sim2real import motor_crc
from phoenix.sim2real.go2_model import (
    JOINT_LIMITS_PROVENANCE,
    JOINT_POSITION_LIMITS_RAD,
    LIMIT_ABORT_BAND_RAD,
    POLICY_JOINT_ORDER,
    TRAINING_DEFAULT_JOINT_POS,
    UNITREE_EXAMPLE_FOLDED_POSE,
    UNITREE_EXAMPLE_STAND_POSE,
    UNITREE_MOTOR_ORDER,
    limits_in_order,
    verify_default_pose,
    verify_joint_model,
)
from phoenix.sim2real.motor_crc import PHOENIX_FOR_MOTOR, phoenix_to_unitree, unitree_to_phoenix
from phoenix.sim2real.safety import MAX_DELTA_PER_STEP_RAD

REPO_ROOT = Path(__file__).resolve().parent.parent

# Verbatim from unitreerobotics/unitree_ros robots/go2_description/urdf/go2_description.urdf
# (<limit lower= upper=>), fetched 2026-09-12. A second, independent literal copy
# so an edit to the module table cannot also silently edit the test.
URDF_LIMITS = {
    "hip": (-1.0472, 1.0472),
    "F_thigh": (-1.5708, 3.4907),
    "R_thigh": (-0.5236, 4.5379),
    "calf": (-2.7227, -0.83776),
}


def _urdf_limit(name: str) -> tuple[float, float]:
    if name.endswith("hip_joint"):
        return URDF_LIMITS["hip"]
    if name.endswith("thigh_joint"):
        return URDF_LIMITS["F_thigh"] if name.startswith("F") else URDF_LIMITS["R_thigh"]
    return URDF_LIMITS["calf"]


@pytest.mark.parametrize("name", POLICY_JOINT_ORDER)
def test_every_joint_limit_is_the_urdf_value(name: str) -> None:
    assert JOINT_POSITION_LIMITS_RAD[name] == _urdf_limit(name)
    lo, hi = JOINT_POSITION_LIMITS_RAD[name]
    assert lo < hi


def test_limit_table_covers_exactly_the_twelve_leg_joints() -> None:
    assert set(JOINT_POSITION_LIMITS_RAD) == set(POLICY_JOINT_ORDER) == set(UNITREE_MOTOR_ORDER)
    assert JOINT_LIMITS_PROVENANCE["kind"].startswith("hard URDF limits")


def test_abort_band_is_the_slew_cap_not_a_new_number() -> None:
    assert LIMIT_ABORT_BAND_RAD == MAX_DELTA_PER_STEP_RAD


@pytest.mark.parametrize("name", POLICY_JOINT_ORDER)
def test_training_pose_is_inside_limits(name: str) -> None:
    lo, hi = JOINT_POSITION_LIMITS_RAD[name]
    assert lo < TRAINING_DEFAULT_JOINT_POS[name] < hi


@pytest.mark.parametrize(
    "pose", [UNITREE_EXAMPLE_FOLDED_POSE, UNITREE_EXAMPLE_STAND_POSE], ids=["folded", "stand"]
)
def test_unitree_example_poses_are_inside_hard_limits(pose) -> None:
    lo, hi = limits_in_order(UNITREE_MOTOR_ORDER)
    q = np.asarray(pose)
    assert np.all(q >= lo) and np.all(q <= hi)


def test_soft_limits_would_reject_unitrees_own_folded_calf() -> None:
    """Why the envelope is the hard range: Isaac's 0.9 soft range excludes -2.65."""
    lo, hi = JOINT_POSITION_LIMITS_RAD["FR_calf_joint"]
    mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    soft_lo = mid - 0.9 * half
    assert UNITREE_EXAMPLE_FOLDED_POSE[2] < soft_lo
    assert UNITREE_EXAMPLE_FOLDED_POSE[2] > lo


def test_limits_in_order_rejects_unknown_joint() -> None:
    with pytest.raises(KeyError):
        limits_in_order(["FL_hip_joint", "FL_knee_joint"])


def test_real_permutation_is_a_correct_name_mapping() -> None:
    assert verify_joint_model(POLICY_JOINT_ORDER, PHOENIX_FOR_MOTOR) == []


@pytest.mark.parametrize("i,j", list(itertools.combinations(range(12), 2)))
def test_every_transposition_of_the_permutation_is_caught(i: int, j: int) -> None:
    perm = list(PHOENIX_FOR_MOTOR)
    perm[i], perm[j] = perm[j], perm[i]
    assert verify_joint_model(POLICY_JOINT_ORDER, perm), f"swap of motors {i},{j} not caught"


@pytest.mark.parametrize("i,j", list(itertools.combinations(range(12), 2)))
def test_every_joint_order_transposition_is_caught(i: int, j: int) -> None:
    order = list(POLICY_JOINT_ORDER)
    order[i], order[j] = order[j], order[i]
    assert verify_joint_model(order, PHOENIX_FOR_MOTOR)


def test_non_permutation_is_caught() -> None:
    assert verify_joint_model(POLICY_JOINT_ORDER, [0] * 12)
    assert verify_joint_model(POLICY_JOINT_ORDER, list(range(11)))


@pytest.mark.parametrize("k", range(12))
def test_each_motor_receives_its_own_joint_through_the_permutation(k: int) -> None:
    phoenix = [0.0] * 12
    phoenix[POLICY_JOINT_ORDER.index(UNITREE_MOTOR_ORDER[k])] = 1.0
    unitree = phoenix_to_unitree(phoenix)
    assert unitree.index(1.0) == k
    assert unitree_to_phoenix(unitree) == phoenix


def test_motor_crc_index_constants_match_unitree_order() -> None:
    for leg_index, leg in enumerate(("FR", "FL", "RR", "RL")):
        for part_index, part in enumerate(("hip", "thigh", "calf")):
            k = getattr(motor_crc, f"{leg}_{part_index}")
            assert k == 3 * leg_index + part_index
            assert UNITREE_MOTOR_ORDER[k] == f"{leg}_{part}_joint"


def test_lowstate_bridge_motor_names_match_unitree_order() -> None:
    """Parsed from source: the node imports rclpy at module scope."""
    tree = ast.parse((REPO_ROOT / "src/phoenix/sim2real/lowstate_bridge_node.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "MOTOR_NAMES":
            assert tuple(ast.literal_eval(node.value)) == UNITREE_MOTOR_ORDER
            return
    pytest.fail("MOTOR_NAMES not found in lowstate_bridge_node.py")


def test_native_runtime_joint_map_matches() -> None:
    src = (REPO_ROOT / "runtime/phoenix_core/src/joint_map.cpp").read_text()
    names = re.search(r"kPolicyJointNames = \{(.*?)\};", src, re.S).group(1)
    assert tuple(re.findall(r'"([A-Za-z_]+)"', names)) == POLICY_JOINT_ORDER
    perm = re.search(r"kPhoenixForMotor = \{(.*?)\};", src, re.S).group(1)
    perm = re.sub(r"//[^\n]*", "", perm)
    assert tuple(int(v) for v in re.findall(r"\d+", perm)) == PHOENIX_FOR_MOTOR


@pytest.mark.parametrize(
    "config_path",
    sorted((REPO_ROOT / "configs/sim2real").glob("deploy*.yaml")),
    ids=lambda p: p.name,
)
def test_every_deploy_config_has_training_pose_and_order(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    assert verify_default_pose(cfg["control"]["default_joint_pos"]) == []
    assert verify_joint_model(cfg["joint_order"], PHOENIX_FOR_MOTOR) == []


def test_verify_default_pose_catches_the_historical_zero_hip() -> None:
    bad = dict(TRAINING_DEFAULT_JOINT_POS, FL_hip_joint=0.0)
    problems = verify_default_pose(bad)
    assert len(problems) == 1 and "FL_hip_joint" in problems[0]
