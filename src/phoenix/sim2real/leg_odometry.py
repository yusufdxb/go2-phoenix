"""Kinematic leg odometry for one GO2 walking run (offline, proprioception only).

There is no on-robot ground truth for body displacement while the sport service is
released: ``policy.parquet`` has ``base_pos`` zeros and ``odom_valid`` False. This module
estimates displacement from what IS recorded, the measured joint angles and the IMU
orientation, under a no-slip assumption:

* foot positions in the body frame come from the GO2 leg kinematics (MJCF
  ``assets/mujoco/unitree_go2/go2.xml``: hip offset, thigh 0.213 m, calf 0.213 m);
* rotated into a gravity-aligned frame by the IMU quaternion;
* the stance pair is the lower diagonal pair of the trot (FL+RR or FR+RL, mean world z);
  its feet are taken as fixed on the floor, so the base moves by minus their mean
  displacement per step. A per-foot height band does not work on this robot: swing
  clearance is 1-2 cm, the same size as the front/rear kinematic height offset, which the
  diagonal pairing cancels (one front and one rear foot per pair). The bridge's
  ``foot_force`` is no help either: it stays constant through a walk.

The result is an ESTIMATE. Foot slip, the stance heuristic and joint encoder offsets all
bias it, and slip is exactly what a friction study changes, so it is never a substitute
for an external measurement (tape, video). ``rear_feet_forward_m`` is reported in the
same convention as the operator's tape (mean of the two rear feet, start mark to end mark,
along the initial heading), so the two can be compared run by run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .go2_model import POLICY_JOINT_ORDER

LEGS = ("FL", "FR", "RL", "RR")
#: Body-frame hip joint origins (MJCF base -> {leg}_hip).
HIP_OFFSET = {
    "FL": (0.1934, 0.0465, 0.0),
    "FR": (0.1934, -0.0465, 0.0),
    "RL": (-0.1934, 0.0465, 0.0),
    "RR": (-0.1934, -0.0465, 0.0),
}
THIGH_Y = 0.0955  # hip -> thigh, +y for left legs
L_THIGH = 0.213
CALF_FOOT = (-0.002, 0.0, -0.213)  # calf joint -> foot sphere centre
SCHEMA = "phoenix-leg-odometry/v1"


def foot_positions_body(q_policy: np.ndarray) -> np.ndarray:
    """(T, 12) joint angles in policy order -> (T, 4, 3) foot centres in the body frame, LEGS order."""
    q = np.asarray(q_policy, dtype=float)
    idx = {n: i for i, n in enumerate(POLICY_JOINT_ORDER)}
    out = np.zeros((q.shape[0], 4, 3))
    for k, leg in enumerate(LEGS):
        qh = q[:, idx[f"{leg}_hip_joint"]]
        qt = q[:, idx[f"{leg}_thigh_joint"]]
        qc = q[:, idx[f"{leg}_calf_joint"]]
        # calf frame -> thigh frame (rotation about y by qc), then thigh -> hip (about y by qt)
        cx, cz = CALF_FOOT[0], CALF_FOOT[2]
        fx = np.cos(qc) * cx + np.sin(qc) * cz
        fz = -np.sin(qc) * cx + np.cos(qc) * cz
        fz = fz - L_THIGH
        x = np.cos(qt) * fx + np.sin(qt) * fz
        z = -np.sin(qt) * fx + np.cos(qt) * fz
        y = np.full_like(x, THIGH_Y if leg in ("FL", "RL") else -THIGH_Y)
        # hip roll about x by qh
        y2 = np.cos(qh) * y - np.sin(qh) * z
        z2 = np.sin(qh) * y + np.cos(qh) * z
        hx, hy, hz = HIP_OFFSET[leg]
        out[:, k] = np.stack([x + hx, y2 + hy, z2 + hz], axis=1)
    return out


def _rotmat_xyzw(qxyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = (qxyzw[:, i] for i in range(4))
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=1,
    )


def integrate(t: np.ndarray, q_policy: np.ndarray, quat_xyzw: np.ndarray) -> dict[str, np.ndarray]:
    """Base and foot trajectories in the IMU world frame (yaw included), origin at the first sample."""
    rot = _rotmat_xyzw(np.asarray(quat_xyzw, dtype=float))
    p_rel = np.einsum("tij,tkj->tki", rot, foot_positions_body(q_policy))  # (T, 4, 3) foot - base, world
    z = p_rel[:, :, 2]
    a_lower = (z[:, 0] + z[:, 3]) < (z[:, 1] + z[:, 2])  # FL+RR below FR+RL
    stance = np.where(a_lower[:, None], [True, False, False, True], [False, True, True, False])
    base = np.zeros((len(t), 3))
    for i in range(1, len(t)):
        both = stance[i] & stance[i - 1]
        if not both.any():
            both = stance[i]
        base[i] = base[i - 1] - (p_rel[i, both] - p_rel[i - 1, both]).mean(axis=0)
    feet = base[:, None, :] + p_rel
    return {"t": np.asarray(t, float), "base": base, "feet": feet, "stance": stance, "R": rot}


def _heading_frame(rot0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yaw0 = float(np.arctan2(rot0[1, 0], rot0[0, 0]))
    fwd = np.array([np.cos(yaw0), np.sin(yaw0), 0.0])
    left = np.array([-np.sin(yaw0), np.cos(yaw0), 0.0])
    return fwd, left


def estimate_run(run_dir: Path, window: tuple[float, float] | None = None) -> dict[str, Any]:
    """Leg-odometry displacement for one run directory (needs policy.parquet).

    ``window`` is (start, end) in policy-log seconds; default is the whole policy log.
    """
    import pandas as pd

    df = pd.read_parquet(Path(run_dir) / "policy.parquet")
    t = df["timestamp_s"].to_numpy(dtype=float)
    q = np.stack(df["joint_pos"].to_numpy()).astype(float)
    quat = np.stack(df["base_quat"].to_numpy()).astype(float)
    keep = np.all(np.isfinite(q), axis=1) & np.all(np.isfinite(quat), axis=1)
    if window is not None:
        keep &= (t >= window[0]) & (t <= window[1])
    t, q, quat = t[keep], q[keep], quat[keep]
    if len(t) < 10:
        return {"schema": SCHEMA, "available": False, "reason": "fewer than 10 policy samples"}
    tr = integrate(t, q, quat)
    fwd, left = _heading_frame(tr["R"][0])
    d_base = tr["base"][-1] - tr["base"][0]
    rear = tr["feet"][:, [LEGS.index("RL"), LEGS.index("RR")], :].mean(axis=1)
    d_rear = rear[-1] - rear[0]
    yaw = np.unwrap(np.arctan2(tr["R"][:, 1, 0], tr["R"][:, 0, 0]))
    return {
        "schema": SCHEMA,
        "available": True,
        "kind": "ESTIMATE: kinematic leg odometry, no-slip assumption; not a measurement",
        "window_s": [float(t[0]), float(t[-1])],
        "n_samples": int(len(t)),
        "base_forward_m": float(d_base @ fwd),
        "base_lateral_m": float(d_base @ left),
        "rear_feet_forward_m": float(d_rear @ fwd),
        "rear_feet_lateral_m": float(d_rear @ left),
        "yaw_deg": float(np.degrees(yaw[-1] - yaw[0])),
        "stance_feet_mean": float(tr["stance"].sum(axis=1).mean()),
    }
