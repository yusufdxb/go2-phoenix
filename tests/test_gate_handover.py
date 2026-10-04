"""Stand-up -> policy gain handover and fault-during-handover, on the crude PD leg plant.

The plant is NOT a robot model (see tests/test_fix_stand_plant.py): these are sequencing
and bound checks on the gain schedule the bridge commands, not hardware predictions.
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real.actuator_gate import SIT_GAIN_RAMP_KP_PER_TICK, ActuatorGate
from phoenix.sim2real.command_wire import KIND_ABORT, encode
from tests.test_actuator_gate import DEFAULT_P, ORDER, ns, params, policy_cmd
from tests.test_fault_sit_plant import DT, LimitedPlant, base_height, to_p, to_u
from tests.test_fix_stand_plant import FOLDED, load_vector

STANDUP_S = 1.0
STANDUP_KP, STANDUP_KD = 60.0, 5.0
POLICY_KP, POLICY_KD = 25.0, 0.5

#: Stated bounds (measured values are asserted in the failure message).
#: Per-tick kp change: (60-25) / (handover_s / dt) = 0.7. kd is not blended at the
#: handover (2026-10-04 desk gate): it steps 5 -> 0.5 once, on the first POLICY tick.
#: Every OTHER kd change (sit ramp) stays within 0.09 per tick.
MAX_DKP_PER_TICK = 0.7 + 1e-9
MAX_DKD_PER_TICK = 0.09 + 1e-9
KD_HANDOVER_STEP = STANDUP_KD - POLICY_KD
#: Largest tick-to-tick change of the commanded joint torque kp (q_cmd - q) - kd dq, N m,
#: in the production order (the harness launches the policy node only after the bridge
#: reports standup_done, so the first POLICY tick finds the stance held at 60/5).
MAX_TORQUE_STEP_NM = 2.0
#: Worst case the harness cannot produce: policy commands already flowing, so POLICY
#: starts on the tick the ramp ends, with the plant still moving at ramp speed and the kd
#: step multiplying it. Measured 3.78 N m (calf); stated bound 4.0 N m, under a tenth of
#: the smallest joint torque limit (23.5 N m).
MAX_TORQUE_STEP_IMMEDIATE_NM = 4.0
#: Time the production order leaves between standup_done and the first policy command,
#: lower bound used here (node start, ONNX load and first-message gating take longer).
PRODUCTION_POLICY_DELAY_S = 0.5
#: Largest base-height dip below the height at handover start, metres.
MAX_HANDOVER_DIP_M = 0.01


def run(handover_s=1.0, t_end=4.0, fault_at=None, abort=True, policy_delay_s=None):
    """``policy_delay_s=None``: policy commands from t=0 (POLICY starts as the ramp ends).
    A number: commands start that long after the gate first reports standup_done."""
    plant = LimitedPlant(np.clip(FOLDED, -3.0, 3.0))
    plant.load = -load_vector()
    gate = ActuatorGate(params(standup_s=STANDUP_S, handover_s=handover_s), ns(0.0))
    recs, heights, taus, seq = [], [], [], 0
    fault_sent = False
    done_t = None
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
        elif not fault_sent and (
            policy_delay_s is None or (done_t is not None and t >= done_t + policy_delay_s)
        ):
            seq += 1
            label, data = policy_cmd(DEFAULT_P, seq=seq)
            gate.on_command(ns(t), label, data)
        rec = gate.tick(ns(t))
        rec["t"] = t
        if done_t is None and rec.get("standup_done"):
            done_t = t
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
    # kd steps to the policy value on the first POLICY tick and never changes afterwards
    assert first["kd"] == pytest.approx(POLICY_KD)
    assert recs[idx[0] - 1]["kd"] == pytest.approx(STANDUP_KD)
    assert {r["kd"] for r in recs[idx[0] : idx[-1] + 1]} == {POLICY_KD}
    # across the whole run, including stand-up -> first policy tick and the end of the blend
    dkp = np.abs(np.diff(kps))
    dkd = np.abs(np.diff(kds))
    msg = f"max dkp {dkp.max():.4f}, max dkd {dkd.max():.4f}"
    assert dkp.max() <= MAX_DKP_PER_TICK, msg
    big_kd = [d for d in dkd if d > MAX_DKD_PER_TICK]
    assert big_kd == [pytest.approx(KD_HANDOVER_STEP)], msg  # the single handover kd step
    # the blend is monotone and takes about handover_s
    blend = [recs[i]["kp"] for i in idx if recs[i]["handover_alpha"] < 1.0]
    assert all(a >= b for a, b in zip(blend, blend[1:], strict=False))
    assert len(blend) == pytest.approx(STANDUP_S / DT, abs=2)


def _handover_step(recs, heights, taus):
    i0 = _policy_idx(recs)[0]
    win = slice(i0 - 2, i0 + int(1.0 / DT) + 3)
    return np.abs(np.diff(taus[win], axis=0)).max(), float(heights[i0] - heights[win].min())


def test_handover_torque_step_is_bounded_and_beats_the_step_in_the_production_order() -> None:
    step, _dip = _handover_step(*run(t_end=5.0, policy_delay_s=PRODUCTION_POLICY_DELAY_S))
    assert step <= MAX_TORQUE_STEP_NM, f"max torque step {step:.3f} N m"
    # the same plant and order with the old one-tick gain step: strictly larger torque step
    step0, _ = _handover_step(
        *run(handover_s=0.0, t_end=5.0, policy_delay_s=PRODUCTION_POLICY_DELAY_S)
    )
    assert step0 > step, f"step baseline {step0:.3f} N m not above blended {step:.3f}"


def test_immediate_takeover_worst_case_is_bounded() -> None:
    """Policy already commanding when the ramp ends (not producible by the harness)."""
    step, dip = _handover_step(*run())
    msg = f"max torque step {step:.3f} N m, base height dip {dip * 1e3:.2f} mm"
    assert step <= MAX_TORQUE_STEP_IMMEDIATE_NM, msg
    assert dip <= MAX_HANDOVER_DIP_M, msg


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
    # the sit starts at the hold gains and blends down (no 60/5 -> 30/1.5 step in one tick)
    assert p.sit.sit_kp <= recs[sit_start]["kp"] <= max(kp_h, p.sit.sit_kp)
    if kp_h > p.sit.sit_kp:
        assert recs[sit_start]["kp"] == pytest.approx(kp_h - SIT_GAIN_RAMP_KP_PER_TICK)
    assert (recs[sit_start]["t"] - recs[k0]["t"]) == pytest.approx(p.sit.fault_hold_s, abs=2 * DT)
    assert all(r["kp"] > 0.0 for r in recs if r["publish"])


def test_fault_after_the_handover_still_holds_at_hold_gains() -> None:
    recs, _, _ = run(t_end=5.0, fault_at=3.5)
    k0 = int(round(3.5 / DT))
    assert recs[k0 - 1]["kp"] == pytest.approx(POLICY_KP)
    assert recs[k0]["mode"] == "hold"
    assert (recs[k0]["kp"], recs[k0]["kd"]) == (params().hold_kp, params().hold_kd)


@pytest.mark.parametrize("fault_at", [0.5, 1.3])
def test_sit_after_a_stood_up_gain_hold_blends_down_within_the_handover_bound(fault_at) -> None:
    """Reviewer follow-up: a fault hold at 60/5 (stand-up or handover gains) then the
    controlled sit (30/1.5) must not step the gains in one tick."""
    recs, _, _ = run(t_end=6.0, fault_at=fault_at)
    p = params()
    pub = [r for r in recs if r["publish"] and r["t"] >= fault_at - 2 * DT]
    dkp = np.abs(np.diff([r["kp"] for r in pub]))
    dkd = np.diff([r["kd"] for r in pub])
    assert dkp.max() <= MAX_DKP_PER_TICK, f"max per-tick kp step {dkp.max():.3f}"
    # kd never DROPS faster than the sit ramp; the one rise allowed is a fault after the
    # handover kd step lifting 0.5 to the hold kd (more damping, never less)
    assert -dkd.min() <= MAX_DKD_PER_TICK, f"max per-tick kd drop {-dkd.min():.3f}"
    assert dkd.max() <= max(MAX_DKD_PER_TICK, p.hold_kd - POLICY_KD) + 1e-9
    sit = [r for r in recs if r["mode"] == "sit"]
    assert sit and sit[0]["kp"] > p.sit.sit_kp, "sit did not start from the hold gains"
    assert sit[-1]["kp"] == p.sit.sit_kp and sit[-1]["kd"] == p.sit.sit_kd
    assert all(r["kp"] > 0.0 and r["kd"] > 0.0 for r in pub)


def test_every_published_tick_records_handover_gains_and_torque_margin() -> None:
    recs, _, _ = run(t_end=3.0)
    pub = [r for r in recs if r["publish"]]
    assert pub
    for r in pub:
        assert r["kp_eff"] == r["kp"] and r["kd_eff"] == r["kd"]
        assert len(r["torque_margin_nm"]) == 12 and all(
            m is not None for m in r["torque_margin_nm"]
        )
    blend = [r for r in pub if r["mode"] == "policy" and r["handover_alpha"] is not None]
    assert blend and 0.0 <= blend[0]["handover_alpha"] < 1.0
    assert blend[-1]["handover_alpha"] == pytest.approx(1.0, abs=0.03)
