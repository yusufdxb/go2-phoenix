"""Stand-up -> policy gain handover and fault-during-handover, on the crude PD leg plant.

The plant is NOT a robot model (see tests/test_fix_stand_plant.py): these are sequencing
and bound checks on the gain schedule the bridge commands, not hardware predictions.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real.actuator_gate import ActuatorGate
from phoenix.sim2real.command_wire import KIND_ABORT, encode
from tests.test_actuator_gate import DEFAULT_P, ORDER, ns, params, policy_cmd
from tests.test_fault_sit_plant import DT, LimitedPlant, base_height, to_p, to_u
from tests.test_fix_stand_plant import FOLDED, load_vector

STANDUP_S = 1.0
STANDUP_KP, STANDUP_KD = 60.0, 5.0
POLICY_KP, POLICY_KD = 25.0, 0.5

#: Stated bounds (measured values are asserted in the failure message).
#: Per-tick gain change: (60-25) / (handover_s / dt) = 0.7 and (5-0.5) / 50 = 0.09.
MAX_DKP_PER_TICK = 0.7 + 1e-9
MAX_DKD_PER_TICK = 0.09 + 1e-9
#: Largest tick-to-tick change of the commanded joint torque kp (q_cmd - q) - kd dq, N m.
MAX_TORQUE_STEP_NM = 2.0
#: Largest base-height dip below the height at handover start, metres.
MAX_HANDOVER_DIP_M = 0.01


def run(handover_s=1.0, t_end=4.0, fault_at=None, abort=True):
    plant = LimitedPlant(np.clip(FOLDED, -3.0, 3.0))
    plant.load = -load_vector()
    gate = ActuatorGate(params(standup_s=STANDUP_S, handover_s=handover_s), ns(0.0))
    recs, heights, taus, seq = [], [], [], 0
    fault_sent = False
    for k in range(int(round(t_end / DT))):
        t = k * DT
        q_p, dq_p = plant.measured()
        gate.on_lowstate(ns(t), to_u(q_p), to_u(dq_p))
        gate.on_estop(ns(t), False)
        if fault_at is not None and t >= fault_at - 1e-9 and not fault_sent:
            fault_sent = True
            label, data = encode(
                ORDER, seq=9999, kind=KIND_ABORT, target=DEFAULT_P, abort_reason="t"
            )
            gate.on_command(ns(t), label, data)
        elif not fault_sent:
            seq += 1
            label, data = policy_cmd(DEFAULT_P, seq=seq)
            gate.on_command(ns(t), label, data)
        rec = gate.tick(ns(t))
        rec["t"] = t
        if rec["publish"]:
            tgt = np.asarray(to_p(rec["final_target_unitree"]))
            taus.append(rec["kp"] * (tgt - q_p) - rec["kd"] * np.asarray(dq_p))
            plant.step(tgt, rec["kp"], rec["kd"])
        else:
            taus.append(np.zeros(12))
        recs.append(rec)
        heights.append(base_height(plant.q))
    return recs, np.asarray(heights), np.asarray(taus)


def _policy_idx(recs):
    return [i for i, r in enumerate(recs) if r["mode"] == "policy"]


def test_gains_blend_linearly_with_bounded_per_tick_change() -> None:
    recs, _, _ = run()
    idx = _policy_idx(recs)
    assert idx, "never reached POLICY"
    kps = [r["kp"] for r in recs if r["publish"]]
    kds = [r["kd"] for r in recs if r["publish"]]
    assert all(k > 0.0 for k in kps)  # kp never <= 0 anywhere
    first, last = recs[idx[0]], recs[idx[-1]]
    assert first["kp"] == pytest.approx(
        STANDUP_KP, abs=MAX_DKP_PER_TICK
    )  # starts at stand-up gains
    assert last["kp"] == pytest.approx(POLICY_KP) and last["kd"] == pytest.approx(POLICY_KD)
    # across the whole run, including stand-up -> first policy tick and the end of the blend
    dkp = np.abs(np.diff(kps))
    dkd = np.abs(np.diff(kds))
    msg = f"max dkp {dkp.max():.4f}, max dkd {dkd.max():.4f}"
    assert dkp.max() <= MAX_DKP_PER_TICK, msg
    assert dkd.max() <= MAX_DKD_PER_TICK, msg
    # the blend is monotone and takes about handover_s
    blend = [recs[i]["kp"] for i in idx if recs[i]["handover_alpha"] < 1.0]
    assert all(a >= b for a, b in zip(blend, blend[1:], strict=False))
    assert len(blend) == pytest.approx(STANDUP_S / DT, abs=2)


def test_handover_torque_step_and_height_dip_are_bounded_and_beat_the_step() -> None:
    recs, heights, taus = run()
    idx = _policy_idx(recs)
    i0 = idx[0]
    win = slice(i0 - 2, i0 + int(1.0 / DT) + 3)
    step = np.abs(np.diff(taus[win], axis=0)).max()
    dip = float(heights[i0] - heights[win].min())
    msg = f"max torque step {step:.3f} N m, base height dip {dip * 1e3:.2f} mm"
    assert step <= MAX_TORQUE_STEP_NM, msg
    assert dip <= MAX_HANDOVER_DIP_M, msg
    # the same plant with the old one-tick gain step: strictly larger torque step
    recs0, _, taus0 = run(handover_s=0.0)
    j0 = _policy_idx(recs0)[0]
    step0 = np.abs(np.diff(taus0[j0 - 2 : j0 + 3], axis=0)).max()
    assert step0 > step, f"step baseline {step0:.3f} N m not above blended {step:.3f}"


@pytest.mark.parametrize("phase", ["mid_standup", "mid_handover"])
def test_fault_holds_at_entry_gains_then_sits(phase: str) -> None:
    probe, _, _ = run()
    idx = _policy_idx(probe)
    t_fault = 0.5 if phase == "mid_standup" else probe[idx[0]]["t"] + 0.5
    recs, _, _ = run(fault_at=t_fault)
    k0 = int(round(t_fault / DT))
    before = recs[k0 - 1]
    assert before["mode"] == ("standup" if phase == "mid_standup" else "policy")
    entry = (before["kp"], before["kd"])
    p = params()
    hold = [r for r in recs[k0:] if r["mode"] == "hold"]
    assert hold, "no hold after the fault"
    sit_start = next(i for i, r in enumerate(recs) if i >= k0 and r["mode"] == "sit")
    held = recs[k0:sit_start]
    gains = {(r["kp"], r["kd"]) for r in held}
    assert len(gains) == 1, gains  # constant, not stepping down tick to tick
    kp_h, kd_h = gains.pop()
    assert kp_h >= p.hold_kp and kd_h >= p.hold_kd
    assert kp_h == pytest.approx(max(entry[0], p.hold_kp), abs=MAX_DKP_PER_TICK)
    assert kd_h == pytest.approx(max(entry[1], p.hold_kd), abs=MAX_DKD_PER_TICK)
    assert recs[sit_start]["kp"] == p.sit.sit_kp
    assert (recs[sit_start]["t"] - recs[k0]["t"]) == pytest.approx(p.sit.fault_hold_s, abs=2 * DT)
    assert all(r["kp"] > 0.0 for r in recs if r["publish"])


def test_fault_after_the_handover_still_holds_at_hold_gains() -> None:
    recs, _, _ = run(t_end=5.0, fault_at=3.5)
    k0 = int(round(3.5 / DT))
    assert recs[k0 - 1]["kp"] == pytest.approx(POLICY_KP)
    assert recs[k0]["mode"] == "hold"
    assert (recs[k0]["kp"], recs[k0]["kd"]) == (params().hold_kp, params().hold_kd)
