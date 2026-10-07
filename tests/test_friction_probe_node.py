"""friction_probe_node: pure abort logic, the probe timeline, the wire it publishes, the FP harness.

No ROS is imported here (the node imports it lazily), so this runs in the plain test venv.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from phoenix.sim2real import friction_probe as fp
from phoenix.sim2real.command_wire import KIND_ABORT, KIND_POLICY, decode, wire_label
from phoenix.sim2real.friction_probe_node import (
    AbortLimits,
    ProbeSession,
    SingleJointSequencer,
    Snapshot,
    check_abort,
    encode_command,
)
from phoenix.sim2real.go2_model import POLICY_JOINT_ORDER

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = REPO_ROOT / "scripts" / "harness_preflight.sh"
ORDER = fp.PROBE_JOINT_ORDER
LIM = AbortLimits(wall_cap_s=1000.0)
Q_STAND = np.asarray([0.0, 0.8, -1.5] * 4)


def snap(q=None, estop=False, ls_age=0.01, estop_age=0.05, dq=None) -> Snapshot:
    q = Q_STAND.copy() if q is None else q
    return Snapshot(
        q=q,
        dq=np.zeros(12) if dq is None else dq,
        tau=np.zeros(12),
        lowstate_age_s=ls_age,
        estop=estop,
        estop_age_s=estop_age,
    )


def test_no_abort_when_quiet() -> None:
    assert check_abort(snap(), Q_STAND, {0}, LIM, 1.0) is None


def test_deadman_released_or_missing_aborts() -> None:
    assert check_abort(snap(estop=True), Q_STAND, {0}, LIM, 1.0).startswith("external_estop")
    assert check_abort(snap(estop=None), Q_STAND, {0}, LIM, 1.0) == "estop_publisher_missing"
    assert check_abort(snap(estop_age=0.6), Q_STAND, {0}, LIM, 1.0) == "estop_heartbeat_stale"


def test_stale_lowstate_and_nan_abort() -> None:
    assert check_abort(snap(ls_age=0.3), Q_STAND, {0}, LIM, 1.0).startswith("sensor_stale")
    q = Q_STAND.copy()
    q[4] = np.nan
    assert check_abort(snap(q=q), Q_STAND, {0}, LIM, 1.0) == "nan_in_joint_state"


def test_other_joint_motion_aborts_but_active_and_previous_are_exempt() -> None:
    q = Q_STAND.copy()
    q[5] += 0.06
    assert "other_joint_moved:" in check_abort(snap(q=q), Q_STAND, {0}, LIM, 1.0)
    assert check_abort(snap(q=q), Q_STAND, {0, 5}, LIM, 1.0) is None  # previous joint stiction
    q2 = Q_STAND.copy()
    q2[5] += 0.04
    assert check_abort(snap(q=q2), Q_STAND, {0}, LIM, 1.0) is None  # under 0.05


def test_excursion_cap_applies_even_to_exempt_joints() -> None:
    q = Q_STAND.copy()
    q[0] += fp.MAX_OFFSET_RAD + 0.06
    reason = check_abort(snap(q=q), Q_STAND, {0}, LIM, 1.0)
    assert reason is not None and "excursion" in reason
    q[0] = Q_STAND[0] + fp.MAX_OFFSET_RAD + 0.04
    assert check_abort(snap(q=q), Q_STAND, {0}, LIM, 1.0) is None


def test_wall_clock_cap() -> None:
    lim = AbortLimits(wall_cap_s=5.0)
    assert check_abort(snap(), Q_STAND, {0}, lim, 5.1).startswith("max_runtime")


def run_session(joints, quiet_q=Q_STAND, abort_at=None, max_t=600.0):
    s = ProbeSession(joints, LIM, settle_s=2.0)
    t, results = 0.0, []
    while t < max_t:
        sn = snap(q=quiet_q.copy(), estop=(abort_at is not None and t >= abort_at))
        r = s.step(t, sn, int(t * 1e9))
        results.append((t, r))
        if r.done or r.phase == "abort":
            break
        t += 1.0 / fp.TICK_HZ
    return s, results


def test_no_command_during_the_bridge_hold_then_targets_start_at_q0() -> None:
    s, res = run_session(["FL_hip_joint"], max_t=2.5)
    assert all(r.target is None for t, r in res if t < 2.0)
    first = next(r for t, r in res if r.target is not None)
    assert np.allclose(first.target, Q_STAND)  # measured q0 under the sequencer's own settle
    assert np.allclose(s.q0, Q_STAND)


def test_single_joint_ramp_lands_on_the_requested_joint_and_completes() -> None:
    joint = "FL_thigh_joint"
    i = ORDER.index(joint)
    s, res = run_session([joint])
    assert res[-1][1].done and res[-1][1].phase == "done"
    offs = np.asarray([r.target - Q_STAND for _, r in res if r.target is not None])
    others = np.delete(offs, i, axis=1)
    assert np.abs(others).max() < 1e-12  # no other joint is ever commanded off q0
    assert offs[:, i].max() > 0.2 and offs[:, i].min() < -0.2  # both directions
    assert np.abs(offs[:, i]).max() <= fp.MAX_OFFSET_RAD + 1e-9
    assert {r.row["active"] for _, r in res if r.row} == {joint}


def test_two_joint_subset_runs_in_the_given_order() -> None:
    joints = ["FR_hip_joint", "FL_hip_joint"]
    s, res = run_session(joints)
    seen = []
    for _, r in res:
        if r.row and r.row["active"] not in seen:
            seen.append(r.row["active"])
    assert seen == joints and res[-1][1].done


def test_single_joint_sequencer_default_joint_matches_full_sequencer_for_joint0() -> None:
    a, b = SingleJointSequencer(Q_STAND, 0), fp.ProbeSequencer(Q_STAND)
    for k in range(600):
        t = k / fp.TICK_HZ
        assert np.allclose(a.step(t, Q_STAND), b.step(t, Q_STAND))


def test_abort_mid_probe_returns_abort_with_no_target() -> None:
    s, res = run_session(["FL_hip_joint"], abort_at=5.0)
    last = res[-1][1]
    assert last.phase == "abort" and last.target is None
    assert last.reason.startswith("external_estop")


def test_wire_command_is_accepted_by_the_bridge_decoder_in_policy_order() -> None:
    target = np.arange(12, dtype=float) * 0.01
    label, data = encode_command(
        POLICY_JOINT_ORDER,
        ORDER,
        seq=7,
        kind=KIND_POLICY,
        target=target,
        snap=snap(),
        elapsed_s=3.0,
    )
    assert label == wire_label(POLICY_JOINT_ORDER)
    cmd = decode(label, data, wire_label(POLICY_JOINT_ORDER))
    assert cmd.seq == 7 and cmd.kind == KIND_POLICY
    f = cmd.fields
    assert f["stand_only"][0] == 1.0 and f["obs_source_code"][0] == 0.0
    assert not np.any(f["velocity_command_fed"]) and not np.any(f["cmd_vel_received"])
    for name in ORDER:  # probe (motor) order value lands on the same joint name in wire order
        assert cmd.target[POLICY_JOINT_ORDER.index(name)] == pytest.approx(
            target[ORDER.index(name)]
        )
    label, data = encode_command(
        POLICY_JOINT_ORDER,
        ORDER,
        seq=8,
        kind=KIND_ABORT,
        target=target,
        snap=None,
        elapsed_s=4.0,
        abort_reason="authority_window_complete",
    )
    assert decode(label, data, label).abort_reason == "authority_window_complete"


# ------------------------------------------------------------------ harness FP stage
def _harness() -> str:
    return HARNESS.read_text()


def test_fp_stage_is_wired_with_every_interlock_and_no_standup() -> None:
    src = _harness()
    fp_body = src.split("stage_fp() {", 1)[1].split("\n}\n", 1)[0]
    assert "    FP) stage_fp" in src
    for need in (
        "need_sha; ros_env",
        "assert_no_contention FP",
        "assert_battery FP",
        '"FP HUNG FEET OFF GROUND"',
        "new_run FP",
        "operator_gate arm FP",
        "operator_gate release FP",
        "lowcmd_bridge_node --live",
        "friction_probe_node",
        "stop_all",
    ):
        assert need in fp_body, need
    assert "--standup-s" not in fp_body.replace("# No --standup-s", "")
    assert "pf evaluate" not in fp_body and "pf require" not in fp_body  # not a ledger stage
    assert "friction_probe" in src.split("stop_all() {", 1)[1].split("}", 1)[0]


def test_fp_refuses_without_expected_sha(tmp_path) -> None:
    env = os.environ.copy()
    env.update(PHOENIX_SESSION=str(tmp_path / "s"), PYTHONDONTWRITEBYTECODE="1")
    env.pop("PHOENIX_EXPECT_SHA", None)
    env.pop("PHOENIX_REHEARSAL", None)
    p = subprocess.run(
        ["bash", str(HARNESS), "FP", "FL_hip_joint"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )
    assert p.returncode != 0 and "PHOENIX_EXPECT_SHA is required" in p.stderr


def test_the_existing_live_banner_is_unchanged_for_ground_stages() -> None:
    src = _harness()
    assert "FEET ON THE GROUND on the fall mat" in src  # default banner kept for E..H, W
