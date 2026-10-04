"""Plant test: inject each fault while standing, the base sinks monotonically to the sit.

Uses the crude PD leg plant of tests/test_fix_stand_plant.py (NOT a robot model). Base
height is a planar forward-kinematics estimate, mean over the four legs of
``L cos(hip) (cos(thigh) + cos(thigh + calf))`` with L = 0.213 m. LIMIT OF THE MODEL: the
plant has no trunk mass, ground contact or free fall, so it measures how fast the
COMMANDED sit lowers the body through the joints; it cannot show a collapse under
lost torque. That is why the sink-speed bound is a check on the sit trajectory plus the
plant's PD tracking, not a hardware prediction. FALL_SOFT is excluded: the robot is
already down in that case, not standing.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real.actuator_gate import ActuatorGate
from phoenix.sim2real.command_wire import KIND_ABORT, encode
from phoenix.sim2real.go2_model import limits_in_order
from phoenix.sim2real.motor_crc import PHOENIX_FOR_MOTOR
from phoenix.sim2real.sit_policy import SIT_POSE_UNITREE
from tests.test_actuator_gate import DEFAULT_P, ORDER, ns, params, policy_cmd
from tests.test_fix_stand_plant import Plant, load_vector

PERM = np.asarray(PHOENIX_FOR_MOTOR)
L_LEG = 0.213
DT = 0.02
MAX_SINK_M_S = 0.5
MAX_RISE_M_PER_TICK = 5e-3  # one-tick PD ripple / open-loop start offset
MAX_REBOUND_M = 0.02  # largest height gain over the running minimum (open-loop start offset)


class LimitedPlant(Plant):
    """The shared plant plus hard joint stops (it has none, so a loaded joint would
    run through the folded pose; the real calf stops at its limit)."""

    def step(self, target, kp, kd):
        super().step(target, kp, kd)
        lo, hi = limits_in_order(ORDER)
        hit = (self.q < lo) | (self.q > hi)
        self.q = np.clip(self.q, lo, hi)
        self.dq = np.where(hit, 0.0, self.dq)


def to_u(p):
    return np.asarray(p)[PERM]


def to_p(u):
    out = np.empty(12)
    out[PERM] = u
    return out


def base_height(q_p) -> float:
    q = dict(zip(ORDER, np.asarray(q_p), strict=True))
    hs = []
    for leg in ("FL", "FR", "RL", "RR"):
        hip, th, ca = q[f"{leg}_hip_joint"], q[f"{leg}_thigh_joint"], q[f"{leg}_calf_joint"]
        hs.append(L_LEG * np.cos(hip) * (np.cos(th) + np.cos(th + ca)))
    return float(np.mean(hs))


def run_plant(fault: str, t_fault=1.0, t_end=7.0):
    kp = params().kp
    # The shared plant's load is "illustrative" and, as written, pushes the legs toward
    # EXTENSION (a spring, it raises the body when stiffness drops). Negate it so gravity
    # flexes the legs, as on the robot; start at rest under the policy gains.
    plant = LimitedPlant(DEFAULT_P + load_vector() / kp)
    plant.load = -load_vector()
    gate = ActuatorGate(params(), ns(0.0))
    flags = dict(lowstate=True, cmd=True, estop=False, nan_q=False)
    heights, recs, seq = [], [], 0
    for k in range(int(round(t_end / DT))):
        t = k * DT
        if abs(t - t_fault) < DT / 2:
            if fault == "stale_lowstate":
                flags["lowstate"] = False
            elif fault == "non_finite_lowstate":
                flags["nan_q"] = True
            elif fault == "estop":
                flags["estop"] = True
            elif fault == "stale_command":
                flags["cmd"] = False
            elif fault == "policy_abort":
                flags["cmd"] = False
                label, data = encode(ORDER, seq=999, kind=KIND_ABORT, target=DEFAULT_P, abort_reason="t")
                gate.on_command(ns(t), label, data)
            elif fault == "shutdown":
                gate.request_shutdown(ns(t))
        q_p, dq_p = plant.measured()
        if flags["lowstate"]:
            q_u = np.full(12, np.nan) if flags["nan_q"] else to_u(q_p)
            gate.on_lowstate(ns(t), q_u, to_u(dq_p))
        gate.on_estop(ns(t), flags["estop"])
        if flags["cmd"]:
            seq += 1
            label, data = policy_cmd(DEFAULT_P, seq=seq)
            gate.on_command(ns(t), label, data)
        rec = gate.tick(ns(t))
        if rec["publish"]:
            plant.step(to_p(rec["final_target_unitree"]), rec["kp"], rec["kd"])
        rec["t"] = t
        recs.append(rec)
        heights.append(base_height(plant.q))
    return np.asarray(heights), recs


@pytest.mark.parametrize(
    "fault",
    ["stale_lowstate", "non_finite_lowstate", "estop", "stale_command", "policy_abort", "shutdown"],
)
def test_base_height_sinks_monotonically_to_the_sit_slowly(fault: str) -> None:
    t_fault = 1.0
    h, recs = run_plant(fault, t_fault=t_fault)
    k0 = int(round(t_fault / DT))
    assert recs[k0 - 5]["mode"] == "policy"
    seg = h[k0:]
    dh = np.diff(seg)
    sink_speed = -dh / DT
    stand_h = base_height(DEFAULT_P)
    sit_h = base_height(to_p(SIT_POSE_UNITREE))
    rebound = float(np.max(seg - np.minimum.accumulate(seg)))
    msg = (
        f"{fault}: rebound {rebound * 1e3:.1f} mm, stand {stand_h:.3f} m, sit {sit_h:.3f} m, final {seg[-1]:.3f} m, "
        f"max rise {dh.max() * 1e3:.3f} mm/tick, peak sink {sink_speed.max():.3f} m/s"
    )
    # Monotone to within PD-transition ripple of the crude plant (measured, see msg).
    assert dh.max() <= MAX_RISE_M_PER_TICK, msg
    assert rebound <= MAX_REBOUND_M, msg
    assert sink_speed.max() < MAX_SINK_M_S, msg
    # Folded low-kp hold (shutdown) may sag onto the hard stops, below the sit pose.
    assert -0.03 < seg[-1] - sit_h < 0.03 or (fault == "shutdown" and seg[-1] < sit_h + 0.03), msg
    assert seg[-1] < stand_h - 0.15, msg
    assert all(r["kp"] > 0.0 for r in recs if r["publish"])
