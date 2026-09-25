"""Isaac Lab GO2 physical parameters, transcribed from the training USD. Pure Python.

Source: the asset ``UNITREE_GO2_CFG`` spawns,
``{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd``. Fetched 2026-09-21 from
``https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd``
(the 5.0 and 5.1 production buckets serve byte-identical files):

* ``go2.usd`` sha256 ``ba171c972b987d8c8fb7157ccad2ba9c0c1fed105755d2d8af46bef96cc11c6d``
* ``Props/instanceable_meshes.usd`` (collision primitives) sha256
  ``2902646d0f4c13c9ecae3ac9046e32c7d18902eef9679de9829f642a15c93fb5``

Values were read with ``pxr.Usd`` (``physics:mass``, ``physics:centerOfMass``,
``physics:diagonalInertia``, ``physics:principalAxes`` on each rigid body;
``physics:lowerLimit/upperLimit`` (degrees), ``physics:axis``,
``physics:localRot0``, ``physxJoint:jointFriction`` on each revolute joint).
``physxJoint:armature`` is NOT authored on any joint, and ``UNITREE_GO2_CFG``
leaves the actuator ``armature`` at None (use the USD value), so Isaac trains
with armature 0 (ASSUMPTION: PhysX's default for an unauthored armature is 0; a
runtime read of ``Articulation.data.joint_armature`` would confirm it, which
needs the GPU this session does not own).

Frames: every GO2 revolute joint in the USD has ``localRot0 == localRot1``, so
each child body frame coincides with the parent frame at q = 0, the same
convention as the MuJoCo Menagerie MJCF (both derive from the Unitree URDF).
Hip joints are ``axis X`` with identity ``localRot`` -> +x in the parent frame.
Thigh and calf joints are ``axis X`` with ``localRot0 = (0.7071, 0, 0, 0.7071)``
(+90 deg about z) -> +y in the parent frame.
"""

from __future__ import annotations

ISAAC_GO2_USD_SHA256 = "ba171c972b987d8c8fb7157ccad2ba9c0c1fed105755d2d8af46bef96cc11c6d"
ISAAC_GO2_MESHES_USD_SHA256 = "2902646d0f4c13c9ecae3ac9046e32c7d18902eef9679de9829f642a15c93fb5"

#: body -> (mass kg, com xyz in body frame, principal inertia diag, principal axes quat wxyz)
ISAAC_BODY_INERTIALS: dict[
    str, tuple[float, tuple[float, float, float], tuple[float, float, float], tuple[float, ...]]
] = {
    "base": (
        6.921,
        (0.021111997, 0.0, -0.005366),
        (0.024453087, 0.09807712, 0.10702679),
        (0.999958, 0.0016130266, 0.008992232, -0.0008444425),
    ),
    "FL_hip": (
        0.678,
        (-0.0054, 0.00194, -0.000105),
        (0.0004799672, 0.00088402943, 0.0005960034),
        (0.9999788, -0.0024969478, 0.0047282153, 0.0037199105),
    ),
    "FR_hip": (
        0.678,
        (-0.0054, -0.00194, -0.000105),
        (0.0004799672, 0.00088402943, 0.0005960034),
        (0.9999788, 0.0024969478, 0.0047282153, -0.0037199105),
    ),
    "RL_hip": (
        0.678,
        (0.0054, 0.00194, -0.000105),
        (0.0004799672, 0.00088402943, 0.0005960034),
        (0.9999788, -0.0024969478, -0.0047282153, -0.0037199105),
    ),
    "RR_hip": (
        0.678,
        (0.0054, -0.00194, -0.000105),
        (0.0004799672, 0.00088402943, 0.0005960034),
        (0.9999788, 0.0024969478, -0.0047282153, 0.0037199105),
    ),
    "FL_thigh": (
        1.152,
        (-0.00374, -0.0223, -0.0327),
        (0.005841487, 0.005949727, 0.00087878696),
        (0.97662413, 0.074123584, 0.045750134, -0.19651426),
    ),
    "FR_thigh": (
        1.152,
        (-0.00374, 0.0223, -0.0327),
        (0.005841487, 0.005949727, 0.00087878696),
        (0.97662413, -0.074123584, 0.045750134, 0.19651426),
    ),
    "RL_thigh": (
        1.152,
        (-0.00374, -0.0223, -0.0327),
        (0.005841487, 0.005949727, 0.00087878696),
        (0.97662413, 0.074123584, 0.045750134, -0.19651426),
    ),
    "RR_thigh": (
        1.152,
        (-0.00374, 0.0223, -0.0327),
        (0.005841487, 0.005949727, 0.00087878696),
        (0.97662413, -0.074123584, 0.045750134, 0.19651426),
    ),
    "FL_calf": (
        0.154,
        (0.00548, -0.000975, -0.115),
        (0.001080271, 0.0011000754, 0.000032553424),
        (0.999887, 0.003973435, -0.008161128, -0.011986799),
    ),
    "FR_calf": (
        0.154,
        (0.00548, 0.000975, -0.115),
        (0.001080271, 0.0011000754, 0.000032553424),
        (0.999887, -0.003973435, -0.008161128, 0.011986799),
    ),
    "RL_calf": (
        0.154,
        (0.00548, -0.000975, -0.115),
        (0.001080271, 0.0011000754, 0.000032553424),
        (0.999887, 0.003973435, -0.008161128, -0.011986799),
    ),
    "RR_calf": (
        0.154,
        (0.00548, 0.000975, -0.115),
        (0.001080271, 0.0011000754, 0.000032553424),
        (0.999887, -0.003973435, -0.008161128, 0.011986799),
    ),
}

#: Isaac has a separate foot body per leg, fixed to the calf at (0, 0, -0.213) in
#: the calf frame, mass 0.04 kg, COM at its origin, isotropic inertia 9.6e-6.
ISAAC_FOOT_MASS = 0.04
ISAAC_FOOT_POS_IN_CALF = (0.0, 0.0, -0.213)
ISAAC_FOOT_INERTIA = (0.0000096, 0.0000096, 0.0000096)
#: Two 0.001 kg head bodies fixed to the base (Head_upper, Head_lower).
ISAAC_HEAD_MASS_TOTAL = 0.002
#: Sum of every rigid body's ``physics:mass`` in the USD.
ISAAC_TOTAL_MASS = 15.019

#: Foot collision: sphere radius 0.022 m centred at (-0.002, 0, -0.213) in the calf frame.
ISAAC_FOOT_SPHERE_RADIUS = 0.022
ISAAC_FOOT_SPHERE_POS_IN_CALF = (-0.002, 0.0, -0.213)

#: Revolute joint limits from the USD, degrees converted to radians (5 dp).
ISAAC_JOINT_LIMITS_RAD: dict[str, tuple[float, float]] = {
    **{f"{leg}_hip_joint": (-1.0472, 1.0472) for leg in ("FL", "FR", "RL", "RR")},
    **{f"{leg}_thigh_joint": (-1.5708, 3.4907) for leg in ("FL", "FR")},
    **{f"{leg}_thigh_joint": (-0.5236, 4.5379) for leg in ("RL", "RR")},
    **{f"{leg}_calf_joint": (-2.7227, -0.83776) for leg in ("FL", "FR", "RL", "RR")},
}

#: Joint axis in the parent frame (see module docstring).
ISAAC_JOINT_AXES: dict[str, tuple[float, float, float]] = {
    **{f"{leg}_hip_joint": (1.0, 0.0, 0.0) for leg in ("FL", "FR", "RL", "RR")},
    **{f"{leg}_thigh_joint": (0.0, 1.0, 0.0) for leg in ("FL", "FR", "RL", "RR")},
    **{f"{leg}_calf_joint": (0.0, 1.0, 0.0) for leg in ("FL", "FR", "RL", "RR")},
}

ISAAC_JOINT_ARMATURE = 0.0  # unauthored in the USD, see module docstring
ISAAC_JOINT_FRICTION = 0.0  # physxJoint:jointFriction = 0 and DCMotorCfg friction=0.0
ISAAC_JOINT_PASSIVE_DAMPING = 0.0  # explicit actuator: sim drive gains are zeroed
#: USD drive maxForce; Isaac Lab overwrites it for explicit actuators, the DCMotor
#: clip (23.5 N m) is what limits torque during training.
ISAAC_USD_DRIVE_MAX_FORCE = {"hip": 23.7, "thigh": 23.7, "calf": 45.43}

#: PhysX ``physxJoint:maxJointVelocity`` (deg/s in the USD: 1724.603 hip/thigh,
#: 899.5437 calf), converted to rad/s. ``DCMotorCfg.velocity_limit_sim`` is None
#: in ``UNITREE_GO2_CFG``, so Isaac Lab keeps these USD values as the SOLVER
#: velocity clamp (the DCMotor ``velocity_limit`` 30 only shapes the torque-speed
#: curve). MuJoCo has no equivalent clamp; the runner counts exceedances instead.
ISAAC_JOINT_MAX_VEL_RAD_S: dict[str, float] = {
    **{f"{leg}_hip_joint": 30.1 for leg in ("FL", "FR", "RL", "RR")},
    **{f"{leg}_thigh_joint": 30.1 for leg in ("FL", "FR", "RL", "RR")},
    **{f"{leg}_calf_joint": 15.7 for leg in ("FL", "FR", "RL", "RR")},
}

#: Isaac Lab velocity-task terrain material: static=dynamic friction 1.0, combine
#: mode "multiply" (``velocity_env_cfg.py``); the robot material is then
#: randomized per env by ``physics_material`` (Phoenix ``friction_range [0.3, 1.5]``).
ISAAC_TERRAIN_FRICTION = 1.0

__all__ = [
    "ISAAC_BODY_INERTIALS",
    "ISAAC_FOOT_INERTIA",
    "ISAAC_FOOT_MASS",
    "ISAAC_FOOT_POS_IN_CALF",
    "ISAAC_FOOT_SPHERE_POS_IN_CALF",
    "ISAAC_FOOT_SPHERE_RADIUS",
    "ISAAC_GO2_MESHES_USD_SHA256",
    "ISAAC_GO2_USD_SHA256",
    "ISAAC_HEAD_MASS_TOTAL",
    "ISAAC_JOINT_ARMATURE",
    "ISAAC_JOINT_AXES",
    "ISAAC_JOINT_FRICTION",
    "ISAAC_JOINT_LIMITS_RAD",
    "ISAAC_JOINT_MAX_VEL_RAD_S",
    "ISAAC_JOINT_PASSIVE_DAMPING",
    "ISAAC_TERRAIN_FRICTION",
    "ISAAC_TOTAL_MASS",
    "ISAAC_USD_DRIVE_MAX_FORCE",
]
