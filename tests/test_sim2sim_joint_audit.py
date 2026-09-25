"""Joint-order and sign audit: Isaac order, MuJoCo model order, Unitree SDK order."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phoenix.sim2sim.deploy_spec import load_deploy_spec, spec_from_phoenix_manifest
from phoenix.sim2sim.joint_audit import (
    ISAAC_ORDER,
    MJLAB_JOINT_IDS_MAP,
    PER_LEG_ORDER,
    SDK_ORDER,
    index_map,
    order_maps,
)
from phoenix.velocity.contract import CommandRanges, build_manifest

REPO = Path(__file__).resolve().parents[1]
PC = REPO / "configs" / "sim2sim" / "positive_control"
H25 = REPO / "configs" / "sim2real" / "manifests" / "h25_stand_only.json"


def phoenix_spec():
    m = build_manifest(
        checkpoint_sha256="0" * 64,
        commands=CommandRanges((-1.0, 1.0), (-0.5, 0.5), (-1.0, 1.0), 0.1),
        git_sha="a" * 40,
        git_dirty=False,
        seed=1,
        task="t",
        simulator="isaaclab",
        reward_scales={},
        domain_randomization={},
        curriculum={},
    )
    return spec_from_phoenix_manifest(
        m,
        name="synthetic",
        required_envelope={"lin_vel_x": 0.5, "lin_vel_y": 0.3, "ang_vel_z": 0.6},
    )


# ------------------------------------------------------------------ order (pure)


def test_three_orders_are_the_documented_ones():
    assert ISAAC_ORDER[:4] == ("FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint")
    assert ISAAC_ORDER[4] == "FL_thigh_joint" and ISAAC_ORDER[8] == "FL_calf_joint"
    assert PER_LEG_ORDER[:3] == ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint")
    assert PER_LEG_ORDER[3] == "FR_hip_joint" and PER_LEG_ORDER[6] == "RL_hip_joint"
    assert SDK_ORDER[:3] == ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint")
    assert (
        SDK_ORDER[3] == "FL_hip_joint"
        and SDK_ORDER[6] == "RR_hip_joint"
        and SDK_ORDER[9] == "RL_hip_joint"
    )


def test_per_leg_to_sdk_equals_unitree_mjlab_deploy_map():
    # The map Unitree ships in unitree_rl_mjlab's GO2 deploy.yaml is exactly the
    # name-derived per-leg (FL,FR,RL,RR) -> SDK (FR,FL,RR,RL) map.
    assert order_maps()["per_leg_to_sdk"] == MJLAB_JOINT_IDS_MAP


def test_isaac_to_sdk_map_values():
    assert order_maps()["isaac_to_sdk"] == (3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8)


@pytest.mark.parametrize(
    "src,dst", [(ISAAC_ORDER, SDK_ORDER), (PER_LEG_ORDER, SDK_ORDER), (ISAAC_ORDER, PER_LEG_ORDER)]
)
def test_maps_roundtrip_by_name(src, dst):
    fwd = index_map(src, dst)
    back = index_map(dst, src)
    for i, n in enumerate(src):
        assert dst[fwd[i]] == n
    v_src = np.arange(12.0) + 100.0
    v_dst = np.empty(12)
    v_dst[list(fwd)] = v_src  # src[i]'s value lands at dst index fwd[i]
    v_back = np.empty(12)
    v_back[list(back)] = v_dst
    assert np.array_equal(v_back, v_src)


def test_index_map_rejects_non_permutations():
    with pytest.raises(ValueError):
        index_map(ISAAC_ORDER, ISAAC_ORDER[:-1] + ("FL_hip_joint",))


def test_spec_sdk_maps_match_what_each_stack_ships():
    rl = load_deploy_spec(PC / "rl_sar_go2_robot_lab.json")
    hi = load_deploy_spec(PC / "rl_sar_go2_himloco.json")
    h25 = load_deploy_spec(H25)
    assert rl.sdk_joint_ids_map == tuple(range(12))  # rl_sar joint_mapping identity
    assert hi.sdk_joint_ids_map == (
        3,
        4,
        5,
        0,
        1,
        2,
        9,
        10,
        11,
        6,
        7,
        8,
    )  # rl_sar himloco joint_mapping
    assert h25.sdk_joint_ids_map == order_maps()["isaac_to_sdk"]
    assert phoenix_spec().sdk_joint_ids_map == order_maps()["isaac_to_sdk"]
    for s in (rl, hi, h25):
        assert not [p for p in s.manifest_problems if "sdk_joint_ids_map" in p]


def test_stated_wrong_sdk_map_is_a_manifest_problem(tmp_path):
    data = json.loads((PC / "rl_sar_go2_robot_lab.json").read_text())
    data["sdk_joint_ids_map"] = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(data))
    spec = load_deploy_spec(p)
    assert any("sdk_joint_ids_map" in x for x in spec.manifest_problems)


# ------------------------------------------------------------------ MuJoCo


def test_mujoco_model_joint_order_is_per_leg():
    mujoco = pytest.importorskip("mujoco")
    from phoenix.sim2sim.model import load_go2_model

    model, idx, _ = load_go2_model("real_go2")
    hinge = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    by_adr = sorted(hinge, key=lambda j: model.jnt_qposadr[j])
    names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in by_adr)
    assert names == PER_LEG_ORDER
    # The runner reads the sim by NAME, so its JOINT_ORDER qpos addresses are the
    # per-leg addresses permuted by isaac_to_per_leg.
    base = int(model.jnt_qposadr[by_adr[0]])
    assert tuple(int(a) - base for a in idx.qpos_adr) == order_maps()["isaac_to_per_leg"]


@pytest.mark.parametrize("which", ["phoenix", "robot_lab", "himloco", "h25"])
def test_sign_audit_through_each_deploy_path(which):
    pytest.importorskip("mujoco")
    from phoenix.sim2sim.joint_audit import sign_audit

    spec = {
        "phoenix": phoenix_spec,
        "robot_lab": lambda: load_deploy_spec(PC / "rl_sar_go2_robot_lab.json"),
        "himloco": lambda: load_deploy_spec(PC / "rl_sar_go2_himloco.json"),
        "h25": lambda: load_deploy_spec(H25),
    }[which]()
    res = sign_audit(spec)
    bad = [j for j in res["joints"] if not j["pass"]]
    assert not bad, bad
    assert [j["joint"] for j in res["joints"]] == list(spec.joint_order)


@pytest.mark.parametrize("joint", [0, 4, 8])
def test_sign_audit_catches_a_sign_inversion_in_the_deploy_path(joint):
    # The audit trusts joint NAMES (a consistently renamed spec is self-consistent);
    # what it catches in the deploy path is a sign error on a joint.
    pytest.importorskip("mujoco")
    from dataclasses import replace

    from phoenix.sim2sim.joint_audit import sign_audit

    spec = load_deploy_spec(H25)
    scale = spec.action_scale.copy()
    scale[joint] = -scale[joint]
    # obs_builder "terms" routes postprocess through the per-joint scale path.
    flipped = replace(spec, obs_builder="terms", action_scale=scale)
    res = sign_audit(flipped)
    assert not res["pass"]
    assert [j["joint"] for j in res["joints"] if not j["pass"]] == [spec.joint_order[joint]]


def test_unitree_hardware_stand_pose_stands_in_this_sign_convention():
    pytest.importorskip("mujoco")
    from phoenix.sim2sim.joint_audit import unitree_stand_pose_check

    r = unitree_stand_pose_check()
    assert r["feet_below_base"]
    assert 0.25 < r["base_height_m"] < 0.40
