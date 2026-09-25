"""Joint-order and sign audit: the #1 community sim2real failure, checked two ways.

1. ORDER (pure): the three GO2 joint orders in circulation and the maps between
   them are derived by NAME and compared with the maps other stacks hard-code:

   * Isaac Lab / PhysX breadth-first (Phoenix ``JOINT_ORDER``):
     FL,FR,RL,RR hips, then thighs, then calves;
   * per-leg FL,FR,RL,RR (MuJoCo MJCF body order, Newton, unitree_rl_mjlab,
     legged_gym / HIMLoco);
   * Unitree SDK ``motor_cmd`` per-leg FR,FL,RR,RL.

2. SIGN (MuJoCo): through a spec's own deploy path (raw action at policy index
   ``i`` -> ``spec.postprocess`` -> name permutation -> PD), command +0.1 rad on
   one joint with the base pinned in the air, and check that the joint named
   ``spec.joint_order[i]`` moved by about +0.1 rad, nothing else moved, and the
   foot of THAT leg moved the way the joint's axis says it must:

   * hip (axis +x): foot moves to +y in the base frame (outward for a left leg,
     inward for a right leg);
   * thigh (axis +y): foot moves to -x (backward);
   * calf (axis +y, range [-2.72, -0.84]): the knee opens, the hip-to-foot
     distance grows.

   Unitree's URDF, the MuJoCo Menagerie MJCF and Isaac's USD share these axes
   (``isaac_reference`` docstring; ``go2_model.JOINT_POSITION_LIMITS_RAD``), and
   Unitree's own hardware stand example pose (calf -1.3, thigh 0.67) is a valid
   standing pose in this convention; :func:`unitree_stand_pose_check` checks it.
   ASSUMPTION: the SDK's motor q is the URDF joint angle with the same sign (the
   Unitree stand example standing correctly on hardware is the evidence).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from phoenix.sim2real.go2_model import UNITREE_EXAMPLE_STAND_POSE, UNITREE_MOTOR_ORDER
from phoenix.velocity.contract import JOINT_ORDER

from .deploy_spec import DeploySpec, joint_group

LEGS_PER_LEG_ORDER = ("FL", "FR", "RL", "RR")
PER_LEG_ORDER: tuple[str, ...] = tuple(
    f"{leg}_{j}_joint" for leg in LEGS_PER_LEG_ORDER for j in ("hip", "thigh", "calf")
)
ISAAC_ORDER: tuple[str, ...] = tuple(JOINT_ORDER)
SDK_ORDER: tuple[str, ...] = tuple(UNITREE_MOTOR_ORDER)

#: unitree_rl_mjlab deploy/robots/go2/config/policy/velocity/v0/params/deploy.yaml
MJLAB_JOINT_IDS_MAP: tuple[int, ...] = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)

AUDIT_DELTA_RAD = 0.1
#: A commanded joint must reach this fraction of the delta; others must stay
#: within the still tolerance. PD with the spec's own gains, pinned base.
MOVED_FRACTION = 0.5
STILL_TOL_RAD = 0.01
FOOT_MIN_MOVE_M = 0.005


def index_map(src: Sequence[str], dst: Sequence[str]) -> tuple[int, ...]:
    """``m[i]`` = index in ``dst`` of ``src[i]`` (src i -> dst index)."""
    if sorted(src) != sorted(dst) or len(set(src)) != len(src):
        raise ValueError("orders must be permutations of the same names")
    return tuple(list(dst).index(n) for n in src)


def order_maps() -> dict[str, tuple[int, ...]]:
    return {
        "isaac_to_sdk": index_map(ISAAC_ORDER, SDK_ORDER),
        "per_leg_to_sdk": index_map(PER_LEG_ORDER, SDK_ORDER),
        "isaac_to_per_leg": index_map(ISAAC_ORDER, PER_LEG_ORDER),
        "sdk_to_isaac": index_map(SDK_ORDER, ISAAC_ORDER),
    }


def unitree_stand_pose_check() -> dict[str, Any]:
    """FK of Unitree's hardware stand example pose (SDK order) in the MuJoCo model."""
    import mujoco

    from .model import load_go2_model

    model, idx, _ = load_go2_model("real_go2")
    data = mujoco.MjData(model)
    q_sim = np.asarray([UNITREE_EXAMPLE_STAND_POSE[SDK_ORDER.index(n)] for n in JOINT_ORDER])
    data.qpos[0:3] = (0.0, 0.0, 1.0)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[idx.qpos_adr] = q_sim
    mujoco.mj_forward(model, data)
    feet = [data.geom_xpos[g].copy() for g in idx.foot_geom_ids]
    lowest = min(float(p[2] - model.geom_size[g][0]) for p, g in zip(feet, idx.foot_geom_ids, strict=True))
    return {
        "base_height_m": 1.0 - lowest,
        "feet_below_base": all(p[2] < 1.0 - 0.15 for p in feet),
        "feet_xy": [p[:2].tolist() for p in feet],
    }


def sign_audit(spec: DeploySpec, *, settle_s: float = 0.4, physics_hz: int = 1000) -> dict[str, Any]:
    """Command +0.1 rad on each policy joint through the spec's deploy path. MuJoCo."""
    import mujoco

    from .gate_runner import spec_permutation
    from .model import load_go2_model

    model, idx, _ = load_go2_model("real_go2", timestep=1.0 / physics_hz)
    model.opt.gravity[:] = 0.0  # pinned base in the air: gravity only adds sag
    data = mujoco.MjData(model)
    perm = spec_permutation(spec)
    kp = np.empty(12)
    kd = np.empty(12)
    kp[perm], kd[perm] = spec.kp, spec.kd
    base_q = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    foot_of_leg = {leg: g for leg, g in zip(("FL", "FR", "RL", "RR"), idx.foot_geom_ids, strict=True)}
    hip_body = {leg: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_thigh") for leg in foot_of_leg}

    def run(target_sim: np.ndarray, q0_sim: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
        mujoco.mj_resetData(model, data)
        data.qpos[0:7] = base_q
        data.qpos[idx.qpos_adr] = q0_sim
        mujoco.mj_forward(model, data)
        for _ in range(int(settle_s * physics_hz)):
            q = data.qpos[idx.qpos_adr]
            qd = data.qvel[idx.dof_adr]
            data.ctrl[idx.actuator_id] = kp * (target_sim - q) - kd * qd
            mujoco.mj_step(model, data)
            data.qpos[0:7] = base_q
            data.qvel[0:6] = 0.0
        mujoco.mj_forward(model, data)
        feet = {leg: data.geom_xpos[g].copy() for leg, g in foot_of_leg.items()}
        reach = {leg: float(np.linalg.norm(feet[leg] - data.xpos[hip_body[leg]])) for leg in foot_of_leg}
        return data.qpos[idx.qpos_adr].copy(), feet, reach

    zero = np.zeros(12)
    _, t0 = spec.postprocess(zero)[:2]
    t0_sim = np.empty(12)
    t0_sim[perm] = t0
    q_ref, feet_ref, reach_ref = run(t0_sim, t0_sim)

    results = []
    for i, name in enumerate(spec.joint_order):
        raw = np.zeros(12)
        # POSITIVE policy action must move the joint in its POSITIVE URDF direction
        # (Isaac JointPositionAction: target = default + |scale| * a). Dividing by
        # the signed scale would hide a sign inversion.
        raw[i] = AUDIT_DELTA_RAD / abs(float(spec.action_scale[i]))
        action, targets, _sat = spec.postprocess(raw)
        tgt_sim = np.empty(12)
        tgt_sim[perm] = targets
        q, feet, reach = run(tgt_sim, t0_sim)
        dq = q - q_ref
        j_sim = JOINT_ORDER.index(name)
        moved_idx = int(np.argmax(np.abs(dq)))
        others = np.delete(np.abs(dq), j_sim)
        leg = name[:2]
        grp = joint_group(name)
        dfoot = feet[leg] - feet_ref[leg]
        other_feet = max(float(np.linalg.norm(feet[lg] - feet_ref[lg])) for lg in feet if lg != leg)
        if grp == "hip":
            expect, ok_dir = "foot +y", dfoot[1] > FOOT_MIN_MOVE_M
        elif grp == "thigh":
            expect, ok_dir = "foot -x", dfoot[0] < -FOOT_MIN_MOVE_M
        else:
            expect, ok_dir = "leg extends", reach[leg] - reach_ref[leg] > FOOT_MIN_MOVE_M / 2
        ok = (
            JOINT_ORDER[moved_idx] == name
            and dq[j_sim] > MOVED_FRACTION * AUDIT_DELTA_RAD
            and float(np.max(others)) < STILL_TOL_RAD
            and other_feet < FOOT_MIN_MOVE_M
            and bool(ok_dir)
        )
        results.append({
            "policy_index": i,
            "joint": name,
            "sdk_index": int(spec.sdk_joint_ids_map[i]),
            "joint_moved_rad": float(dq[j_sim]),
            "largest_other_rad": float(np.max(others)),
            "moved_joint": JOINT_ORDER[moved_idx],
            "foot_delta_m": dfoot.tolist(),
            "reach_delta_m": reach[leg] - reach_ref[leg],
            "largest_other_foot_move_m": other_feet,
            "expected": expect,
            "pass": bool(ok),
        })
    return {"pass": all(r["pass"] for r in results), "joints": results}


__all__ = [
    "AUDIT_DELTA_RAD",
    "ISAAC_ORDER",
    "MJLAB_JOINT_IDS_MAP",
    "PER_LEG_ORDER",
    "SDK_ORDER",
    "index_map",
    "order_maps",
    "sign_audit",
    "unitree_stand_pose_check",
]
