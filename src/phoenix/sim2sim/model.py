"""Load the vendored MuJoCo Menagerie GO2 and map it onto the contract JOINT_ORDER.

MuJoCo is imported lazily inside functions so this module can be imported (and
its pure helpers tested) without MuJoCo installed.

Two model profiles
------------------
``isaac_matched`` (default)
    The Menagerie model with every parameter that has an Isaac Lab training
    counterpart set to that counterpart (see ``docs/sim2sim/model_audit.md`` and
    :mod:`phoenix.sim2sim.isaac_reference`):

    * physics timestep 0.005 s (Isaac ``sim.dt``);
    * joint passive damping 0, frictionloss 0, armature 0 (Isaac explicit
      DCMotor: sim drive gains zeroed, jointFriction 0, armature unauthored);
    * calf inertials replaced by the Isaac calf + fixed foot composite
      (0.194 kg per leg instead of Menagerie's 0.241352 kg);
    * foot contacts condim 3 (sliding friction only; PhysX has no torsional or
      rolling friction on these shapes) with a single friction coefficient;
    * the other collision geoms condim 3 instead of Menagerie's frictionless
      condim 1, since PhysX contacts on the trunk and legs are frictional.

    Contact solver parameters (soft contacts, ``impratio``, elliptic cone) are
    MuJoCo's and are deliberately NOT tuned toward PhysX: exposing that
    difference is the point of the second simulator.

``menagerie_raw``
    The vendored MJCF as published, except the physics timestep, which must be
    0.005 s for the 4x decimation to give the 50 Hz policy rate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from phoenix.velocity.contract import JOINT_ORDER, NUM_JOINTS, PHYSICS_HZ
from phoenix.velocity.observation import rotation_matrix_wxyz

from .isaac_reference import (
    ISAAC_BODY_INERTIALS,
    ISAAC_FOOT_INERTIA,
    ISAAC_FOOT_MASS,
    ISAAC_FOOT_POS_IN_CALF,
    ISAAC_JOINT_ARMATURE,
    ISAAC_JOINT_FRICTION,
    ISAAC_JOINT_PASSIVE_DAMPING,
)

PROFILE_ISAAC = "isaac_matched"
PROFILE_MENAGERIE = "menagerie_raw"
PROFILES = (PROFILE_ISAAC, PROFILE_MENAGERIE)

PHYSICS_DT = 1.0 / PHYSICS_HZ
#: Nominal foot/ground friction for the ``isaac_matched`` profile. Isaac's
#: terrain is 1.0 with "multiply" combine and the robot material is randomized
#: in [0.3, 1.5] (Phoenix ``friction_range``); Isaac Lab's upstream velocity task
#: uses static 0.8 / dynamic 0.6. MuJoCo has one coefficient; 0.8 is the
#: Menagerie foot value and sits inside the trained range.
NOMINAL_FOOT_FRICTION = 0.8

LEGS = ("FL", "FR", "RL", "RR")

#: Repository root, from src/phoenix/sim2sim/model.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE_XML = _REPO_ROOT / "assets" / "mujoco" / "unitree_go2" / "scene.xml"

#: Provenance of the vendored model (also in assets/mujoco/unitree_go2/PROVENANCE.md).
MENAGERIE_PROVENANCE = {
    "repository": "https://github.com/google-deepmind/mujoco_menagerie",
    "path": "unitree_go2",
    "repo_head_commit": "822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7",
    "latest_commit_touching_path": "71f066ad0be9cd271f7ed58c030243ef157af9f4",
    "fetched": "2026-09-21",
    "go2_xml_sha256": "50adb09a4365293e2acdaf2010ae35a82b0f09fea18ae51806fef91e310ed04a",
    "scene_xml_sha256": "b56123ea2bf09070bf4054f0be9aa418e49041ddddce44dbbf7ab35e8b732641",
    "license": "BSD-3-Clause (Unitree Robotics), assets/mujoco/unitree_go2/LICENSE",
}


class ModelMappingError(RuntimeError):
    """The MuJoCo model does not map onto the contract joint order by name."""


@dataclass(frozen=True)
class JointIndexMap:
    """MuJoCo indices for the contract :data:`JOINT_ORDER`, resolved by NAME.

    ``qpos_adr[i]`` / ``dof_adr[i]`` / ``actuator_id[i]`` all refer to joint
    ``JOINT_ORDER[i]``.
    """

    joint_names: tuple[str, ...]
    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    actuator_id: np.ndarray
    base_body_id: int
    base_qpos_adr: int
    base_dof_adr: int
    foot_geom_ids: tuple[int, ...]
    trunk_geom_ids: tuple[int, ...]
    floor_geom_id: int
    joint_range: np.ndarray = field(repr=False)  # (12, 2), JOINT_ORDER


def _name(model: Any, obj: Any, idx: int) -> str:
    import mujoco

    return mujoco.mj_id2name(model, obj, idx) or ""


def build_joint_index_map(model: Any) -> JointIndexMap:
    """Resolve every contract joint in ``model`` by name and assert the wiring."""
    import mujoco

    problems: list[str] = []
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1:
        raise ModelMappingError(f"expected exactly one free joint, found {len(free)}")
    free_j = free[0]
    base_body = int(model.jnt_bodyid[free_j])
    if _name(model, mujoco.mjtObj.mjOBJ_BODY, base_body) != "base":
        problems.append("free joint is not on body 'base'")

    qpos_adr, dof_adr, act_id = [], [], []
    for name in JOINT_ORDER:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            problems.append(f"joint {name} missing from the MuJoCo model")
            continue
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            problems.append(f"joint {name} is not a hinge")
        qpos_adr.append(int(model.jnt_qposadr[j]))
        dof_adr.append(int(model.jnt_dofadr[j]))
        acts = [
            a
            for a in range(model.nu)
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
            and int(model.actuator_trnid[a, 0]) == j
        ]
        if len(acts) != 1:
            problems.append(f"joint {name} has {len(acts)} actuators, expected 1")
            act_id.append(-1)
            continue
        a = acts[0]
        if abs(float(model.actuator_gear[a, 0]) - 1.0) > 1e-12:
            problems.append(f"actuator on {name} has gear {model.actuator_gear[a, 0]}, expected 1")
        if int(model.actuator_biastype[a]) != int(mujoco.mjtBias.mjBIAS_NONE):
            problems.append(f"actuator on {name} is not a pure motor (has a bias/PD term)")
        act_id.append(a)
    if problems:
        raise ModelMappingError("; ".join(problems))

    def geoms_of(body_name: str) -> list[int]:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == b and model.geom_contype[g] | model.geom_conaffinity[g]
        ]

    foot = []
    for leg in LEGS:
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg)
        if g < 0:
            raise ModelMappingError(f"foot geom {leg} missing")
        foot.append(g)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor < 0:
        raise ModelMappingError("floor geom missing (load scene.xml, not go2.xml)")

    qa = np.asarray(qpos_adr, dtype=np.int64)
    jr = np.asarray(
        [
            model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in JOINT_ORDER
        ],
        dtype=np.float64,
    )
    index_map = JointIndexMap(
        joint_names=tuple(JOINT_ORDER),
        qpos_adr=qa,
        dof_adr=np.asarray(dof_adr, dtype=np.int64),
        actuator_id=np.asarray(act_id, dtype=np.int64),
        base_body_id=base_body,
        base_qpos_adr=int(model.jnt_qposadr[free_j]),
        base_dof_adr=int(model.jnt_dofadr[free_j]),
        foot_geom_ids=tuple(foot),
        trunk_geom_ids=tuple(geoms_of("base")),
        floor_geom_id=int(floor),
        joint_range=jr,
    )
    assert_index_map(model, index_map)
    return index_map


def assert_index_map(model: Any, index_map: JointIndexMap) -> None:
    """Re-derive every index from its name and fail on any disagreement."""
    import mujoco

    if len(index_map.joint_names) != NUM_JOINTS or tuple(index_map.joint_names) != JOINT_ORDER:
        raise ModelMappingError("index map joint names are not the contract JOINT_ORDER")
    if len(set(index_map.qpos_adr.tolist())) != NUM_JOINTS:
        raise ModelMappingError("qpos addresses are not unique")
    for i, name in enumerate(JOINT_ORDER):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if int(model.jnt_qposadr[j]) != int(index_map.qpos_adr[i]):
            raise ModelMappingError(f"qpos address of {name} disagrees with its name")
        if int(model.jnt_dofadr[j]) != int(index_map.dof_adr[i]):
            raise ModelMappingError(f"dof address of {name} disagrees with its name")
        a = int(index_map.actuator_id[i])
        if int(model.actuator_trnid[a, 0]) != j:
            raise ModelMappingError(f"actuator {a} does not drive {name}")
    if index_map.base_qpos_adr != 0 or index_map.base_dof_adr != 0:
        raise ModelMappingError("free joint is expected at qpos[0:7] / qvel[0:6]")


def _quat_from_matrix(r: np.ndarray) -> np.ndarray:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(r, dtype=np.float64).reshape(9))
    return q


def isaac_calf_composite(leg: str) -> tuple[float, np.ndarray, np.ndarray]:
    """Isaac calf + fixed foot as one rigid body: ``(mass, com, inertia_tensor_at_com)``.

    Everything in the calf frame (identical in Isaac and MuJoCo at q = 0).
    Parallel-axis theorem on the two Isaac bodies. Pure numpy.
    """
    m1, c1, d1, q1 = ISAAC_BODY_INERTIALS[f"{leg}_calf"]
    r1 = rotation_matrix_wxyz(q1)
    i1 = r1 @ np.diag(d1) @ r1.T
    m2 = ISAAC_FOOT_MASS
    c2 = np.asarray(ISAAC_FOOT_POS_IN_CALF, dtype=np.float64)
    i2 = np.diag(ISAAC_FOOT_INERTIA)
    c1 = np.asarray(c1, dtype=np.float64)
    mass = m1 + m2
    com = (m1 * c1 + m2 * c2) / mass

    def shift(m: float, d: np.ndarray) -> np.ndarray:
        return m * (float(d @ d) * np.eye(3) - np.outer(d, d))

    inertia = i1 + shift(m1, c1 - com) + i2 + shift(m2, c2 - com)
    return mass, com, inertia


def body_inertia_tensor(model: Any, body_id: int) -> tuple[float, np.ndarray, np.ndarray]:
    """``(mass, com, inertia tensor at com)`` of a MuJoCo body, in its body frame."""
    r = rotation_matrix_wxyz(model.body_iquat[body_id])
    tensor = r @ np.diag(model.body_inertia[body_id]) @ r.T
    return float(model.body_mass[body_id]), np.asarray(model.body_ipos[body_id]).copy(), tensor


def _set_body_inertia(model: Any, body_id: int, mass: float, com: np.ndarray, tensor: np.ndarray):
    evals, evecs = np.linalg.eigh(tensor)
    if np.linalg.det(evecs) < 0:
        evecs[:, 0] = -evecs[:, 0]
    model.body_mass[body_id] = mass
    model.body_ipos[body_id] = com
    model.body_inertia[body_id] = evals
    model.body_iquat[body_id] = _quat_from_matrix(evecs)


def load_go2_model(
    profile: str = PROFILE_ISAAC,
    *,
    scene_xml: str | os.PathLike[str] | None = None,
    foot_friction: float | None = None,
) -> tuple[Any, JointIndexMap, dict[str, Any]]:
    """Load the GO2 scene, apply ``profile``, return ``(model, index_map, applied)``.

    ``applied`` records every override so the report can carry it.
    """
    import mujoco

    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")
    path = Path(scene_xml or os.environ.get("PHOENIX_GO2_MJCF", DEFAULT_SCENE_XML))
    model = mujoco.MjModel.from_xml_path(str(path))
    index_map = build_joint_index_map(model)
    applied: dict[str, Any] = {"profile": profile, "scene_xml": str(path)}

    model.opt.timestep = PHYSICS_DT
    applied["timestep"] = PHYSICS_DT

    if profile == PROFILE_ISAAC:
        dofs = index_map.dof_adr
        model.dof_damping[dofs] = ISAAC_JOINT_PASSIVE_DAMPING
        model.dof_frictionloss[dofs] = ISAAC_JOINT_FRICTION
        model.dof_armature[dofs] = ISAAC_JOINT_ARMATURE
        applied["joint_damping"] = ISAAC_JOINT_PASSIVE_DAMPING
        applied["joint_frictionloss"] = ISAAC_JOINT_FRICTION
        applied["joint_armature"] = ISAAC_JOINT_ARMATURE
        calf_mass = {}
        for leg in LEGS:
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf")
            mass, com, tensor = isaac_calf_composite(leg)
            _set_body_inertia(model, b, mass, com, tensor)
            calf_mass[leg] = mass
        applied["calf_inertial"] = "isaac calf + foot composite"
        applied["calf_mass_kg"] = calf_mass
        for g in range(model.ngeom):
            if model.geom_contype[g] | model.geom_conaffinity[g]:
                model.geom_condim[g] = 3
        applied["collision_condim"] = 3
        friction = NOMINAL_FOOT_FRICTION if foot_friction is None else float(foot_friction)
    else:
        friction = None if foot_friction is None else float(foot_friction)

    if friction is not None:
        for g in index_map.foot_geom_ids:
            model.geom_friction[g, 0] = friction
    applied["foot_friction"] = (
        float(model.geom_friction[index_map.foot_geom_ids[0], 0]) if friction is None else friction
    )
    # Recompute derived constants (invweight0 etc.) after changing inertials.
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    applied["total_mass_kg"] = float(np.sum(model.body_mass))
    applied["integrator"] = int(model.opt.integrator)
    applied["cone"] = int(model.opt.cone)
    applied["impratio"] = float(model.opt.impratio)
    return model, index_map, applied


__all__ = [
    "DEFAULT_SCENE_XML",
    "JointIndexMap",
    "LEGS",
    "MENAGERIE_PROVENANCE",
    "ModelMappingError",
    "NOMINAL_FOOT_FRICTION",
    "PHYSICS_DT",
    "PROFILES",
    "PROFILE_ISAAC",
    "PROFILE_MENAGERIE",
    "assert_index_map",
    "body_inertia_tensor",
    "build_joint_index_map",
    "isaac_calf_composite",
    "load_go2_model",
]
