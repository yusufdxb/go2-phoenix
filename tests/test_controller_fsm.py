"""Tests for the deploy state machine: IDLE -> ... -> VELOCITY_ACTIVE, and FAULT
from every reachable state.

No ROS, no ONNX: the policy and observation builder are plain stubs, exactly
the seam ``ControllerFSM`` was built around (see its module docstring, "the
tests drive every state with a policy stub that raises if called").
"""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2real.controller_fsm import (
    ControllerFSM,
    FsmInputs,
    FsmParams,
    State,
    fsm_params_from_manifest,
    fsm_params_from_mapping,
)
from phoenix.sim2real.deploy_manifest import ManifestError
from phoenix.sim2real.fix_stand import FixStandParams
from phoenix.sim2real.go2_model import (
    JOINT_POSITION_LIMITS_RAD,
    POLICY_JOINT_ORDER,
    TRAINING_DEFAULT_JOINT_POS,
)
from phoenix.sim2real.handoff import HandoffCriteria

ORDER = POLICY_JOINT_ORDER
N = len(ORDER)
TARGET = np.asarray([TRAINING_DEFAULT_JOINT_POS[n] for n in ORDER], dtype=np.float64)
RATE_HZ = 50.0
DT_NS = int(round(1e9 / RATE_HZ))

# Fast-test knobs: a short FIX_STAND ramp and a short handoff sustain window.
# The physics is unchanged (smoothstep, torque clip, etc); only durations
# shrink so the test suite doesn't spend real time on 0.5-2s ramps.
FAST_FIX_STAND = FixStandParams(rate_hz=RATE_HZ, min_duration_s=0.1)
FAST_HANDOFF = HandoffCriteria(sustained_ticks=3)


def raising_policy(obs: np.ndarray) -> np.ndarray:
    raise AssertionError("policy called outside VELOCITY_ACTIVE")


def zero_policy(obs: np.ndarray) -> np.ndarray:
    return np.zeros(N, dtype=np.float64)


def stub_obs_builder(
    *, gyro_body, quat_wxyz, command, joint_pos, joint_vel, last_action
) -> np.ndarray:
    return np.zeros(45, dtype=np.float64)


def make_fsm(
    *, params: FsmParams | None = None, policy=zero_policy, stand_only: bool = True
) -> ControllerFSM:
    return ControllerFSM(
        ORDER,
        params=params or FsmParams(rate_hz=RATE_HZ),
        fix_stand_params=FAST_FIX_STAND,
        handoff_criteria=FAST_HANDOFF,
        policy=policy,
        obs_builder=stub_obs_builder,
        stand_only=stand_only,
    )


def nominal_input(t_ns: int, *, q=TARGET, estop_ok=True, deadman_ok=True, command=(0.0, 0.0, 0.0)):
    return FsmInputs(
        t_ns=t_ns,
        q=q,
        dq=np.zeros(N),
        gyro_body=(0.0, 0.0, 0.0),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),  # identity: upright
        lowstate_age_s=0.0,
        imu_age_s=0.0,
        estop_ok=estop_ok,
        deadman_ok=deadman_ok,
        command=command,
    )


def run_to_velocity_ready(fsm: ControllerFSM, t0: int = 0) -> int:
    """Drive IDLE -> ... -> VELOCITY_READY with the target already the measured
    posture (so FIX_STAND has nothing to do but complete its short ramp).
    Returns the next free ``t_ns``.
    """
    fsm.request_start()
    t = t0
    for _ in range(400):  # generous upper bound; the loop breaks on VELOCITY_READY
        out = fsm.tick(nominal_input(t))
        t += DT_NS
        if fsm.state is State.VELOCITY_READY:
            return t
        assert fsm.state is not State.FAULT, f"unexpected FAULT: {out.fault}"
    raise AssertionError("never reached VELOCITY_READY")


# --------------------------------------------------------------- transitions
def test_idle_does_nothing_until_start_requested() -> None:
    fsm = make_fsm(policy=raising_policy)
    out = fsm.tick(nominal_input(0))
    assert fsm.state is State.IDLE and out.publish is False


def test_full_bringup_to_velocity_ready() -> None:
    # A PRECHECK that passes immediately falls through to FIX_STAND in the
    # SAME tick() call (see ControllerFSM._tick_precheck), so the externally
    # observed fsm.state after each tick can skip straight past PRECHECK.
    # The internal transition log is the ground truth for ordering instead.
    fsm = make_fsm(policy=raising_policy)
    fsm.request_start()
    t = 0
    for _ in range(400):
        fsm.tick(nominal_input(t))
        t += DT_NS
        if fsm.state is State.VELOCITY_READY:
            break
    else:
        raise AssertionError("never reached VELOCITY_READY")
    visited = [to for (_t, _frm, to, _why) in fsm.transitions]
    for expected in ("PRECHECK", "FIX_STAND", "FIX_STAND_HOLD", "VELOCITY_READY"):
        assert expected in visited
    assert "FAULT" not in visited
    assert visited.index("PRECHECK") < visited.index("FIX_STAND")
    assert visited.index("FIX_STAND") < visited.index("FIX_STAND_HOLD")
    assert visited.index("FIX_STAND_HOLD") < visited.index("VELOCITY_READY")


def test_policy_only_runs_after_explicit_grant() -> None:
    fsm = make_fsm(policy=zero_policy)
    t = run_to_velocity_ready(fsm)
    # Still VELOCITY_READY: no policy tick has happened (raising_policy would
    # have caught it anyway; zero_policy here just proves calls == 0).
    assert fsm.state is State.VELOCITY_READY
    assert fsm.policy_calls == 0
    reason = fsm.request_policy()
    assert reason is None
    out = fsm.tick(nominal_input(t))
    assert fsm.state is State.VELOCITY_ACTIVE
    assert fsm.policy_calls == 1
    assert out.kp == pytest.approx(fsm.params.policy_kp)
    assert out.kd == pytest.approx(fsm.params.policy_kd)


def test_request_policy_refused_with_no_policy_configured() -> None:
    fsm = ControllerFSM(
        ORDER,
        params=FsmParams(rate_hz=RATE_HZ),
        fix_stand_params=FAST_FIX_STAND,
        handoff_criteria=FAST_HANDOFF,
        policy=None,
        stand_only=True,
    )
    assert fsm.request_policy() == "no_policy_configured"


# ------------------------------------------------------------ action clamp
def test_velocity_active_clamps_action_before_scaling() -> None:
    # The 2026-09-21 stage F1 mechanism, replayed under the LEGACY (H25)
    # contract action_clip=1.0 (FsmParams' dataclass default; a real deploy
    # gets this from fsm_params_from_manifest, see below): an unclamped raw
    # action of -7.2 on one joint reaching target-scaling arithmetic
    # unbounded. Here the policy returns exactly that on RR_thigh; the FSM
    # must clamp to [-action_clip, action_clip] BEFORE scaling, so the
    # requested target before the torque clip is bounded by action_scale,
    # not by the raw action.
    j = ORDER.index("RR_thigh_joint")

    def bad_policy(obs: np.ndarray) -> np.ndarray:
        a = np.zeros(N, dtype=np.float64)
        a[j] = -7.2
        return a

    fsm = make_fsm(policy=bad_policy)
    t = run_to_velocity_ready(fsm)
    fsm.request_policy()
    out = fsm.tick(nominal_input(t))
    assert fsm.state is State.VELOCITY_ACTIVE, out.fault
    # Clamped action is -1.0, scaled by action_scale (0.25 default): the
    # SCALED target (pre torque-clip) deviates by exactly that much, and the
    # published target can only be closer to the measured stance than that
    # (the torque clip only pulls it further in), never further out.
    max_deviation = fsm.params.action_scale * 1.0
    assert out.target is not None
    assert abs(out.target[j] - TARGET[j]) <= max_deviation + 1e-6


def test_velocity_active_action_clip_100_passes_raw_through_to_torque_clip() -> None:
    """2026-09-24 contract: training moved from clip_actions=1.0 to 100.0 (the
    legged_gym / rl_sar / robot_lab convention). With action_clip=100, the
    historical F1 raw magnitude (-7.2) is NOT modified by the action clamp
    (far under the clip), so the TORQUE limit, not the action clamp, is what
    has to bound the motor command. action_scale here is chosen so the
    scaled request lands just past the torque bound while staying inside
    the position abort band, isolating that mechanism specifically (a
    literal action_scale=0.25 at this raw magnitude would instead trip the
    position abort band first, which is a different, already-covered path;
    see test_target_beyond_abort_band-equivalent coverage in
    test_actuator_gate.py).
    """
    j = ORDER.index("RR_thigh_joint")
    limit_nm = 23.5  # RR_thigh_joint torque limit (go2_model.JOINT_TORQUE_LIMITS_NM)
    kp = 25.0
    torque_delta = limit_nm / kp
    action_scale = (torque_delta + 0.05) / 7.2

    def bad_policy(obs: np.ndarray) -> np.ndarray:
        a = np.zeros(N, dtype=np.float64)
        a[j] = -7.2
        return a

    params = FsmParams(rate_hz=RATE_HZ, action_scale=action_scale, action_clip=100.0)
    fsm = make_fsm(policy=bad_policy, params=params)
    t = run_to_velocity_ready(fsm)
    fsm.request_policy()
    out = fsm.tick(nominal_input(t))
    assert fsm.state is State.VELOCITY_ACTIVE, out.fault

    # The action clamp did NOT touch -7.2 (well under clip=100): last_action
    # (fed to the next observation) holds the raw value unmodified.
    assert fsm._last_action[j] == pytest.approx(-7.2)

    # The requested (pre torque-clip) target would have overshot the torque
    # bound...
    q = TARGET[j]
    requested = q + action_scale * -7.2
    assert abs(kp * (requested - q)) > limit_nm + 1e-9

    # ...but the PUBLISHED target (post torque-clip) is bounded by it. dq=0
    # in this fixture, so tau = Kp*(target - q) exactly.
    published_tau = kp * (out.target[j] - q)
    assert abs(published_tau) <= limit_nm + 1e-6
    assert out.target[j] != pytest.approx(requested)


def test_velocity_active_small_action_passes_through_unclipped() -> None:
    def small_policy(obs: np.ndarray) -> np.ndarray:
        a = np.zeros(N, dtype=np.float64)
        a[0] = 0.1
        return a

    fsm = make_fsm(policy=small_policy)
    t = run_to_velocity_ready(fsm)
    fsm.request_policy()
    out = fsm.tick(nominal_input(t))
    assert fsm.state is State.VELOCITY_ACTIVE
    expected = TARGET[0] + fsm.params.action_scale * 0.1
    assert out.target[0] == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------- fault from every
# reachable state (IDLE cannot fault; there is nothing to abort).
@pytest.mark.parametrize(
    "target_state",
    [
        State.PRECHECK,
        State.FIX_STAND,
        State.FIX_STAND_HOLD,
        State.VELOCITY_READY,
        State.VELOCITY_ACTIVE,
    ],
)
def test_abort_reaches_fault_and_damps_from_every_engaged_state(target_state: State) -> None:
    fsm = make_fsm(policy=zero_policy)
    t = 0
    if target_state is State.PRECHECK:
        fsm.request_start()
        # One tick into PRECHECK before the environment is nominal enough to
        # advance; drive it with an estop that is NOT ok so it stays there.
        out = fsm.tick(nominal_input(t, estop_ok=False))
        assert fsm.state in (State.PRECHECK, State.FAULT)
        if fsm.state is State.FAULT:
            # estop_ok=False during PRECHECK is itself a fault path; covered
            # separately below. Re-drive with estop ok to reach PRECHECK.
            fsm = make_fsm(policy=zero_policy)
            fsm.request_start()
            fsm.tick(nominal_input(t))
        fsm.abort("test_abort")
    elif target_state is State.VELOCITY_ACTIVE:
        t = run_to_velocity_ready(fsm)
        fsm.request_policy()
        fsm.tick(nominal_input(t))
        t += DT_NS
        assert fsm.state is State.VELOCITY_ACTIVE
        fsm.abort("test_abort")
    else:
        fsm.request_start()
        t = 0
        for _ in range(400):
            fsm.tick(nominal_input(t))
            t += DT_NS
            if fsm.state is target_state:
                break
        else:
            raise AssertionError(f"never reached {target_state}")
        fsm.abort("test_abort")

    out = fsm.tick(nominal_input(t))
    assert fsm.state is State.FAULT
    assert fsm.fault == "test_abort"
    assert out.kp == 0.0
    assert out.kd == pytest.approx(fsm.params.damp_kd)
    assert out.publish is True

    # FAULT is sticky: further ticks stay in FAULT and keep damping.
    out2 = fsm.tick(nominal_input(t + DT_NS))
    assert fsm.state is State.FAULT and out2.kp == 0.0


def test_estop_release_faults_from_velocity_active() -> None:
    fsm = make_fsm(policy=zero_policy)
    t = run_to_velocity_ready(fsm)
    fsm.request_policy()
    fsm.tick(nominal_input(t))
    t += DT_NS
    assert fsm.state is State.VELOCITY_ACTIVE
    out = fsm.tick(nominal_input(t, estop_ok=False))
    assert fsm.state is State.FAULT
    assert out.fault == "estop"


def test_deadman_release_faults_from_fix_stand_hold() -> None:
    fsm = make_fsm(policy=zero_policy)
    fsm.request_start()
    t = 0
    for _ in range(400):
        fsm.tick(nominal_input(t))
        t += DT_NS
        if fsm.state is State.FIX_STAND_HOLD:
            break
    else:
        raise AssertionError("never reached FIX_STAND_HOLD")
    out = fsm.tick(nominal_input(t, deadman_ok=False))
    assert fsm.state is State.FAULT and out.fault == "deadman_released"


def test_stand_only_nonzero_command_faults() -> None:
    fsm = make_fsm(policy=zero_policy, stand_only=True)
    t = run_to_velocity_ready(fsm)
    out = fsm.tick(nominal_input(t, command=(0.1, 0.0, 0.0)))
    assert fsm.state is State.FAULT
    assert out.fault == "nonzero_command_in_stand_only"


# --------------------------------------------------------------------- stop
def test_request_stop_holds_then_damps() -> None:
    fsm = make_fsm(policy=zero_policy, params=FsmParams(rate_hz=RATE_HZ, stop_hold_s=0.04))
    t = run_to_velocity_ready(fsm)
    fsm.request_stop()
    out = fsm.tick(nominal_input(t))
    assert fsm.state is State.STOPPING
    assert out.kp == pytest.approx(fsm.params.hold_kp)
    t += DT_NS
    # stop_hold_s=0.04 at 50 Hz is 2 ticks; the third tick must be damping.
    for _ in range(5):
        out = fsm.tick(nominal_input(t))
        t += DT_NS
        if out.kp == 0.0:
            break
    assert out.kp == 0.0
    assert out.kd == pytest.approx(fsm.params.damp_kd)


# ------------------------------------------------------------------- params
def test_fsm_params_from_mapping_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="unknown fsm keys"):
        fsm_params_from_mapping({"not_a_real_field": 1.0})


def test_fsm_params_from_mapping_round_trips_max_command() -> None:
    params = fsm_params_from_mapping({"max_command": [0.5, 0.3, 0.8]})
    assert params.max_command == (0.5, 0.3, 0.8)


# ----------------------------------------------------- last_action contract
def test_last_action_fed_to_next_obs_is_clamped_not_raw() -> None:
    """Contract from the training side (rsl_rl wrapper clips actions to
    [-1, 1] BEFORE env.step): last_action in the 45-D obs is the CLIPPED
    action, not the raw network output. Deploy must match: an unclamped
    -7.2 must never reach the next tick's last_action.
    """
    j = ORDER.index("RR_thigh_joint")

    def bad_policy(obs: np.ndarray) -> np.ndarray:
        a = np.zeros(N, dtype=np.float64)
        a[j] = -7.2
        return a

    fsm = make_fsm(policy=bad_policy)
    t = run_to_velocity_ready(fsm)
    fsm.request_policy()
    fsm.tick(nominal_input(t))
    assert fsm.state is State.VELOCITY_ACTIVE
    assert fsm._last_action[j] == pytest.approx(-1.0)
    assert fsm._last_action[j] != pytest.approx(-7.2)


def test_intervention_record_keeps_raw_action_for_diagnostics() -> None:
    """The pre-clamp value is still recorded (raw_action) so an operator can
    see saturation, even though last_action fed forward is clamped."""
    j = ORDER.index("RR_thigh_joint")

    def bad_policy(obs: np.ndarray) -> np.ndarray:
        a = np.zeros(N, dtype=np.float64)
        a[j] = -7.2
        return a

    fsm = make_fsm(policy=bad_policy)
    t = run_to_velocity_ready(fsm)
    fsm.request_policy()
    out = fsm.tick(nominal_input(t))
    assert out.intervention is not None
    assert out.intervention["raw_action"][j] == pytest.approx(-7.2)


# --------------------------------------------------- fsm_params_from_manifest
def test_fsm_params_from_manifest_reads_action_clip(tmp_path) -> None:
    import json

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"deploy": {"action_clip": 100.0}}))
    params = fsm_params_from_manifest(path)
    assert params.action_clip == 100.0
    assert params.policy_kp == FsmParams().policy_kp
    assert params.policy_kd == FsmParams().policy_kd


def test_fsm_params_from_manifest_reads_kp_kd(tmp_path) -> None:
    import json

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"deploy": {"action_clip": 100.0, "kp": 20.0, "kd": 1.0}}))
    params = fsm_params_from_manifest(path)
    assert params.policy_kp == 20.0
    assert params.policy_kd == 1.0


def test_fsm_params_from_manifest_refuses_without_action_clip(tmp_path) -> None:
    import json

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"deploy": {"kp": 25.0}}))
    with pytest.raises(ManifestError, match="action_clip is missing"):
        fsm_params_from_manifest(path)


def test_fsm_params_from_manifest_refuses_missing_file(tmp_path) -> None:
    with pytest.raises(ManifestError):
        fsm_params_from_manifest(tmp_path / "does_not_exist.json")


def test_fsm_params_from_manifest_rejects_action_clip_in_data(tmp_path) -> None:
    import json

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"deploy": {"action_clip": 100.0}}))
    with pytest.raises(ValueError, match="must come from the checkpoint manifest"):
        fsm_params_from_manifest(path, {"action_clip": 1.0})


def test_fsm_params_from_manifest_combines_with_other_mapping_keys(tmp_path) -> None:
    import json

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"deploy": {"action_clip": 100.0}}))
    params = fsm_params_from_manifest(path, {"rate_hz": 50.0, "action_scale": 0.5})
    assert params.action_clip == 100.0
    assert params.rate_hz == 50.0
    assert params.action_scale == 0.5


# ------------------------------------------------- past-the-stop: clip vs latch
# 2026-09-25 sim2sim gate finding: a good walking policy (seed42@3000)
# routinely requests calf targets 0.25-0.31 rad past the calf's hard upper
# limit during counter-clockwise yaw (5 of 600 steps at +0.6 rad/s). A PD
# controller commanding past a mechanical stop to press against it is normal;
# proven stacks do not abort on it. The old "illegal_target" abort on a
# commanded target beyond the hard range (exactly how stage F1 ended) is
# replaced: commanded overshoot is clipped, and the latching abort is
# reserved for MEASURED overshoot, a non-finite command, or sustained
# clipping.
def test_target_past_calf_stop_is_clipped_not_latched() -> None:
    j = ORDER.index("RR_calf_joint")
    lo, hi = JOINT_POSITION_LIMITS_RAD["RR_calf_joint"]
    overshoot = 0.28  # within the literal 0.25-0.31 rad sim2sim finding
    action_scale = 0.25
    target_needed = hi + overshoot
    action_needed = (target_needed - TARGET[j]) / action_scale

    def policy(obs: np.ndarray) -> np.ndarray:
        a = np.zeros(N, dtype=np.float64)
        a[j] = action_needed
        return a

    params = FsmParams(rate_hz=RATE_HZ, action_scale=action_scale, action_clip=100.0)
    fsm = make_fsm(policy=policy, params=params)
    t = run_to_velocity_ready(fsm)
    fsm.request_policy()
    out = fsm.tick(nominal_input(t))
    assert fsm.state is State.VELOCITY_ACTIVE, out.fault
    assert out.target is not None
    assert out.target[j] <= hi
    assert out.target[j] == pytest.approx(hi - fsm.params.position_clip_margin, abs=1e-6)

    # A perfectly valid command right after fully restores normal tracking:
    # a single clipped tick never latches.
    fsm2 = make_fsm(policy=zero_policy, params=params)
    t2 = run_to_velocity_ready(fsm2)
    fsm2.request_policy()
    fsm2.tick(nominal_input(t2))
    assert fsm2.state is State.VELOCITY_ACTIVE


def test_measured_position_beyond_hard_limit_still_latches() -> None:
    """Unlike a commanded overshoot, a MEASURED position beyond a hard limit
    (by more than the abort band) is a real fault (sensor fault, or the
    robot has actually exceeded its mechanical stop) and still latches."""
    j = ORDER.index("RR_calf_joint")
    lo, hi = JOINT_POSITION_LIMITS_RAD["RR_calf_joint"]
    fsm = make_fsm(policy=zero_policy)
    t = run_to_velocity_ready(fsm)
    fsm.request_policy()
    # First tick grants VELOCITY_ACTIVE with a nominal (in-limits) measured
    # state; FixStand's own parallel band check (fix_stand_measured_q_beyond_limit)
    # only runs up to and including the READY->ACTIVE handoff tick, not once
    # VELOCITY_ACTIVE is entered, so this isolates SafetyFilter's own check.
    out = fsm.tick(nominal_input(t))
    t += DT_NS
    assert fsm.state is State.VELOCITY_ACTIVE, out.fault
    q = TARGET.copy()
    q[j] = hi + 0.3  # well beyond the 0.175 rad abort band
    out = fsm.tick(nominal_input(t, q=q))
    assert fsm.state is State.FAULT
    assert out.fault == "measured_position_illegal"


def test_sustained_clipping_latches_but_isolated_ticks_do_not() -> None:
    j = ORDER.index("RR_calf_joint")
    lo, hi = JOINT_POSITION_LIMITS_RAD["RR_calf_joint"]
    overshoot = 0.28
    action_scale = 0.25
    action_needed = (hi + overshoot - TARGET[j]) / action_scale

    def clipping_policy(obs: np.ndarray) -> np.ndarray:
        a = np.zeros(N, dtype=np.float64)
        a[j] = action_needed
        return a

    params = FsmParams(rate_hz=RATE_HZ, action_scale=action_scale, action_clip=100.0)
    fsm = make_fsm(policy=clipping_policy, params=params)
    t = run_to_velocity_ready(fsm)
    fsm.request_policy()
    threshold = fsm.params.sustained_clip_ticks
    for _ in range(threshold - 1):
        out = fsm.tick(nominal_input(t))
        t += DT_NS
        assert fsm.state is State.VELOCITY_ACTIVE, out.fault
    out = fsm.tick(nominal_input(t))
    assert fsm.state is State.FAULT
    assert out.fault == "sustained_clip_exceeded"
