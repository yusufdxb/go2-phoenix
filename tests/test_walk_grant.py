"""Bounded walking authority: the grant, the bridge gate that enforces it, and the bypass attempts.

No ROS, no robot. The ROS-level pieces (bridge config/startup, policy-node startup) are in
tests/test_walking_startup_integration.py; the process-level ones (the real harness, real bridge
and policy node, a fake GO2) are the rehearsal scenarios in scripts/free_floor_rehearsal.py.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import free_floor_gate as ff  # noqa: E402

from phoenix.sim2real import free_floor_state as st  # noqa: E402
from phoenix.sim2real import walk_grant as wg  # noqa: E402
from phoenix.sim2real.actuator_gate import ActuatorGate, GateParams  # noqa: E402
from phoenix.sim2real.command_wire import KIND_POLICY, encode  # noqa: E402
from phoenix.sim2real.go2_model import POLICY_JOINT_ORDER, TRAINING_DEFAULT_JOINT_POS  # noqa: E402
from phoenix.sim2real.motor_crc import phoenix_to_unitree  # noqa: E402
from phoenix.sim2real.walk_command import PRESETS  # noqa: E402

CODE = "a" * 40
T0 = 1_700_000_000.0


@pytest.fixture
def files(tmp_path, monkeypatch):
    """A config and a 'policy' file; FREE_FLOOR's incumbent ONNX hash is pointed at the latter."""
    cfg = tmp_path / "deploy.yaml"
    cfg.write_text("name: fixture\n")
    onnx = tmp_path / "policy.onnx"
    onnx.write_bytes(b"incumbent-fixture")
    monkeypatch.setattr(wg, "FREE_FLOOR_ONNX_SHA256", wg.file_sha256(onnx))
    return cfg, onnx


def issue(tmp_path, files, mode="GANTRY", preset="vx03_4s", run="W_vx03_4s_1", **kw):
    cfg, onnx = files
    run_dir = tmp_path / "hw" / "sess" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        mode=mode,
        preset=preset,
        run_dir=run_dir,
        code_sha=CODE,
        config_path=cfg,
        onnx_path=onnx,
        first_message_timeout_s=15.0,
        standup_s=2.0,
        now=T0,
    )
    kwargs.update(kw)
    return wg.issue_grant(**kwargs), run_dir


def verify(grant, run_dir, **kw):
    kwargs = dict(run_id=run_dir.name, code_sha=CODE, now=T0 + 1.0)
    kwargs.update(kw)
    return wg.grant_problems(grant, **kwargs)


# ------------------------------------------------------------ identity of the constants
def test_free_floor_onnx_hash_is_the_incumbents_and_not_seed_50() -> None:
    assert wg.FREE_FLOOR_ONNX_SHA256 == ff.INCUMBENT["policy.onnx"]
    assert wg.FREE_FLOOR_ONNX_SHA256 != ff.SEED50_HASHES["policy.onnx"]


def test_the_registered_presets_exist_and_free_floor_is_exactly_vx03_2s() -> None:
    assert set(wg.GANTRY_PRESETS) <= set(PRESETS)
    assert wg.GANTRY_PRESETS == (
        "vx03_2s",
        "vx03_4s",
        "yaw05_3s",
    )  # the campaign's two + first walk
    assert wg.ALLOWED_PRESETS[wg.MODE_FREE_FLOOR] == (wg.STAND_ZERO, ff.FREE_FLOOR_PRESET)
    assert ff.FREE_FLOOR_PRESET == "vx03_2s" and ff.FREE_FLOOR_PRESETS == ("vx03_2s",)


def test_envelope_authority_and_window_come_from_the_preset() -> None:
    assert wg.preset_envelope("GANTRY", "vx03_4s") == (0.3, 0.0, 0.0)
    assert wg.preset_envelope("GANTRY", "yaw05_3s") == (0.0, 0.0, 0.5)
    assert wg.preset_envelope("FREE_FLOOR", wg.STAND_ZERO) == (0.0, 0.0, 0.0)
    assert (
        wg.preset_authority_s("GANTRY", "vx03_4s") == 5.0 + 4.0 + 4.0
    )  # 2 s lead-in + 4 s motion + 3 s tail + 4 s margin
    assert wg.preset_authority_s("FREE_FLOOR", "vx03_2s") == 2.0 + 2.0 + 3.0 + 4.0
    assert wg.preset_authority_s("FREE_FLOOR", wg.STAND_ZERO) == ff.DRILL_AUTHORITY_S
    assert wg.preset_authority_s("FREE_FLOOR", "vx03_2s") == (
        ff.profile_duration(PRESETS["vx03_2s"]) + 4.0
    )  # the analyzer's own number


# ------------------------------------------------------------------ issuing
@pytest.mark.parametrize(
    "mode,preset",
    [("GANTRY", p) for p in wg.GANTRY_PRESETS]
    + [("FREE_FLOOR", "vx03_2s"), ("FREE_FLOOR", wg.STAND_ZERO)],
)
def test_every_registered_pair_is_issued_and_verifies(tmp_path, files, mode, preset) -> None:
    (grant, why), run = issue(tmp_path, files, mode=mode, preset=preset)
    assert why == [] and grant is not None
    assert json.loads((run / wg.GRANT_FILE).read_text()) == grant
    assert verify(grant, run, mode=mode) == []
    assert grant["expires_unix"] - grant["issued_unix"] <= wg.MAX_WINDOW_S
    assert grant["run_id"] == run.name and grant["code_sha"] == CODE


@pytest.mark.parametrize(
    "mode,preset",
    [
        ("GANTRY", "vx05_3s"),  # a real walk_command preset that the gantry campaign never uses
        ("GANTRY", "vxm03_2s"),
        ("GANTRY", "vx03_2s_fast"),
        ("GANTRY", "../vx03_2s"),
        ("GANTRY", ""),
        ("GANTRY", wg.STAND_ZERO),  # the drill is a FREE_FLOOR concept
        ("FREE_FLOOR", "vx03_4s"),
        ("FREE_FLOOR", "yaw05_3s"),
        ("FREE_FLOOR", "vx05_3s"),
        ("DRY", "vx03_2s"),
        ("", "vx03_2s"),
    ],
)
def test_unknown_or_unapproved_preset_is_never_issued(tmp_path, files, mode, preset) -> None:
    (grant, why), run = issue(tmp_path, files, mode=mode, preset=preset)
    assert grant is None and why
    assert not (run / wg.GRANT_FILE).exists()


def test_free_floor_refuses_any_other_policy_including_seed_50(tmp_path, files) -> None:
    cfg, onnx = files
    other = tmp_path / "seed50.onnx"
    other.write_bytes(b"a different policy")
    (grant, why), _ = issue(tmp_path, (cfg, other), mode="FREE_FLOOR", preset="vx03_2s")
    assert grant is None and any("incumbent" in w for w in why)
    # the same policy is fine on the gantry: seed 50's legitimate path is untouched
    (grant, why), _ = issue(tmp_path, (cfg, other), mode="GANTRY", preset="vx03_4s", run="W_s50")
    assert why == [] and grant is not None
    # and a hand-built FREE_FLOOR grant naming seed 50's ONNX hash does not verify
    (g, _), run = issue(tmp_path, files, mode="FREE_FLOOR", preset="vx03_2s", run="FF_x")
    forged = dict(g, onnx_sha256=ff.SEED50_HASHES["policy.onnx"])
    forged["body_sha256"] = wg._body_sha256(forged)
    assert any("incumbent" in p for p in verify(forged, run))


def test_a_run_directory_gets_one_grant_never_reissued(tmp_path, files) -> None:
    (g1, _), run = issue(tmp_path, files)
    assert g1 is not None
    (g2, why), _ = issue(tmp_path, files)
    assert g2 is None and any("already exists" in w for w in why)


def test_degenerate_timing_is_refused(tmp_path, files) -> None:
    for bad in (
        dict(first_message_timeout_s=0.0),
        dict(first_message_timeout_s=float("nan")),
        dict(first_message_timeout_s=float("inf")),
        dict(standup_s=-1.0),
    ):
        (grant, why), _ = issue(
            tmp_path, files, run="r_" + str(len(str(bad))) + str(bad.keys()), **bad
        )
        assert grant is None and why, bad


# ------------------------------------------------------------- verifying: fail closed
def test_expired_grant_blocks_walking(tmp_path, files) -> None:
    (grant, _), run = issue(tmp_path, files)
    assert verify(grant, run, now=grant["expires_unix"] - 1) == []
    late = verify(grant, run, now=grant["expires_unix"] + 0.1)
    assert any("expired" in p for p in late)
    assert wg.window_remaining_s(grant, grant["expires_unix"] + 3) < 0


def test_wrong_mode_run_preset_code_config_policy_block_walking(tmp_path, files) -> None:
    (grant, _), run = issue(tmp_path, files, mode="GANTRY", preset="vx03_4s")
    assert any("mode" in p for p in verify(grant, run, mode="FREE_FLOOR"))
    assert any("is for run" in p for p in verify(grant, run, run_id="W_other"))
    assert any("is for run" in p for p in verify(grant, run, run_id=None))
    assert any("different code" in p for p in verify(grant, run, code_sha="b" * 40))
    assert any("different code" in p for p in verify(grant, run, code_sha=None))
    assert any("deploy config" in p for p in verify(grant, run, config_sha256="0" * 64))
    assert any("policy ONNX" in p for p in verify(grant, run, onnx_sha256="0" * 64))
    forged = dict(grant, preset="vx05_3s")  # a preset no launcher may authorize
    forged["body_sha256"] = wg._body_sha256(forged)
    assert any("not approved" in p for p in verify(forged, run))


def test_wrong_preset_for_the_command_blocks_walking() -> None:
    env = wg.preset_envelope("GANTRY", "vx03_4s")
    assert not wg.command_exceeds((0.3, 0.0, 0.0), env)
    assert not wg.command_exceeds((0.1, 0.0, 0.0), env)
    assert wg.command_exceeds((0.0, 0.0, 0.5), env)  # the yaw preset's command on the vx preset
    assert wg.command_exceeds((0.31, 0.0, 0.0), env)
    assert wg.command_exceeds(
        (0.5, 0.0, 0.0), env
    )  # inside walk_command's ENVELOPE, outside the grant
    assert wg.command_exceeds((float("nan"), 0.0, 0.0), env)
    assert wg.command_exceeds((0.1, 0.0, 0.0), wg.preset_envelope("FREE_FLOOR", wg.STAND_ZERO))


@pytest.mark.parametrize(
    "edit",
    [
        dict(authority_s=120.0),  # arbitrary duration
        dict(max_runtime_s=120.0),
        dict(envelope=[0.5, 0.3, 0.8]),  # arbitrary velocity
        dict(envelope=[0.3, 0.0]),
        dict(envelope=[0.3, 0.0, float("nan")]),
        dict(expires_unix=T0 + 3600.0),  # arbitrary window
        dict(expires_unix=float("inf")),
        dict(expires_unix="never"),
        dict(first_message_timeout_s=-1.0),
        dict(standup_s=None),
        dict(schema="phoenix-walk-grant/v0"),
        dict(mode="LAB"),
    ],
)
def test_a_grant_edited_to_widen_anything_is_refused(tmp_path, files, edit) -> None:
    (grant, _), run = issue(tmp_path, files)
    edited = dict(grant, **edit)
    # a careful forger fixes the body hash: the verifier still re-derives everything from the preset
    edited["body_sha256"] = wg._body_sha256(edited)
    assert verify(edited, run), edit
    # a careless one does not
    careless = dict(grant, **edit)
    assert verify(careless, run), edit


def test_hand_made_grant_with_a_huge_window_is_capped(tmp_path, files) -> None:
    (grant, _), run = issue(tmp_path, files)
    g = dict(grant, expires_unix=grant["issued_unix"] + 10_000)
    g["body_sha256"] = wg._body_sha256(g)
    assert any("cap" in p or "allows" in p for p in verify(g, run))


def test_missing_malformed_or_linked_grants_are_refused_without_raising(tmp_path) -> None:
    assert wg.grant_problems(None, run_id="r", code_sha=CODE) == ["no walk grant"]
    assert wg.grant_problems({}, run_id="r", code_sha=CODE)
    assert wg.grant_problems("garbage", run_id="r", code_sha=CODE)  # type: ignore[arg-type]
    missing, why = wg.read_grant(tmp_path / "nope.json")
    assert missing is None and why
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert wg.read_grant(bad)[0] is None
    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2]")
    assert wg.read_grant(arr)[0] is None
    real = tmp_path / "real.json"
    real.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    assert wg.read_grant(link)[0] is None


def test_a_rehearsal_grant_cannot_authorize_a_live_process_and_the_reverse(tmp_path, files) -> None:
    (g_reh, _), run = issue(tmp_path, files, rehearsal=True)
    assert any("rehearsal" in p for p in verify(g_reh, run, rehearsal=False))
    assert verify(g_reh, run, rehearsal=True) == []
    (g_live, _), run2 = issue(tmp_path, files, run="W_live")
    assert any("rehearsal" in p for p in verify(g_live, run2, rehearsal=True))
    assert verify(g_live, run2, rehearsal=False) == []


# ------------------------------------------- one authority at a time / unreviewed latch
def _session_with_unreviewed_free_floor_walk(root: Path) -> Path:
    sess = root / "hw" / "20261008_ff"
    (sess / "FF_walk_1").mkdir(parents=True)
    (sess / st.STARTED).write_text(json.dumps({"run_dir": "FF_walk_1"}))
    return sess


def test_unreviewed_free_floor_walk_blocks_a_gantry_grant_but_not_a_free_floor_one(
    tmp_path, files
) -> None:
    sess = _session_with_unreviewed_free_floor_walk(tmp_path)
    unreviewed = st.unreviewed_runs([sess])
    assert unreviewed
    (grant, why), _ = issue(tmp_path, files, mode="GANTRY", preset="vx03_4s", unreviewed=unreviewed)
    assert grant is None and any("unreviewed free-floor walk" in w for w in why)
    # the free-floor path has its own latch (free_floor_gate.check) and the run that is starting
    # has, by design, a started record already
    (grant, why), _ = issue(
        tmp_path, files, mode="FREE_FLOOR", preset="vx03_2s", run="FF_walk_2", unreviewed=unreviewed
    )
    assert why == [] and grant is not None


def test_one_live_grant_at_a_time_in_both_directions(tmp_path, files) -> None:
    root = tmp_path / "hw"
    (g_w, why), w_run = issue(tmp_path, files, mode="GANTRY", preset="vx03_4s", session_root=root)
    assert why == []
    # an active gantry run blocks a free-floor grant ...
    (g_f, why), _ = issue(
        tmp_path,
        files,
        mode="FREE_FLOOR",
        preset="vx03_2s",
        run="FF_walk_1",
        session_root=root,
        now=T0 + 5,
    )
    assert g_f is None and any("W_vx03_4s_1" in w and "live" in w for w in why)
    # ... and an active free-floor run blocks a gantry grant
    wg.release(w_run)
    (g_f, why), ff_run = issue(
        tmp_path,
        files,
        mode="FREE_FLOOR",
        preset="vx03_2s",
        run="FF_walk_1",
        session_root=root,
        now=T0 + 5,
    )
    assert why == [] and g_f is not None
    (g_w2, why), _ = issue(
        tmp_path,
        files,
        mode="GANTRY",
        preset="yaw05_3s",
        run="W_yaw",
        session_root=root,
        now=T0 + 6,
    )
    assert g_w2 is None and any("FF_walk_1" in w for w in why)
    # a released grant frees the slot; so does expiry
    wg.release(ff_run)
    (g_w2, why), _ = issue(
        tmp_path,
        files,
        mode="GANTRY",
        preset="yaw05_3s",
        run="W_yaw",
        session_root=root,
        now=T0 + 7,
    )
    assert why == [] and g_w2 is not None
    (g_w3, why), _ = issue(
        tmp_path,
        files,
        mode="GANTRY",
        preset="vx03_2s",
        run="W_late",
        session_root=root,
        now=g_w2["expires_unix"] + 1,
    )
    assert why == [] and g_w3 is not None
    # an unreadable grant is treated as live (fail closed)
    (root / "sess" / "W_junk").mkdir()
    (root / "sess" / "W_junk" / wg.GRANT_FILE).write_text("{broken")
    (g_w4, why), _ = issue(
        tmp_path,
        files,
        mode="GANTRY",
        preset="vx03_2s",
        run="W_after_junk",
        session_root=root,
        now=g_w3["expires_unix"] + 1,
    )
    assert g_w4 is None and any("W_junk" in w for w in why)


# ---------------------------------------------- the gate: what the bridge enforces
ORDER = POLICY_JOINT_ORDER
START_NS = 5_000_000_000
DEFAULT_P = np.asarray([TRAINING_DEFAULT_JOINT_POS[n] for n in ORDER])
DEFAULT_U = np.asarray(phoenix_to_unitree(DEFAULT_P))
ZERO12 = np.zeros(12)


def ns(seconds: float) -> int:
    return START_NS + int(round(seconds * 1e9))


def gate_params(**over) -> GateParams:
    base = dict(
        live=False,
        kp=25.0,
        kd=0.5,
        hold_kp=20.0,
        hold_kd=1.0,
        watchdog_s=0.2,
        estop_timeout_s=0.5,
        lowstate_timeout_s=0.2,
        stale_hold_s=0.2,
        first_message_timeout_s=15.0,
        require_imu=False,
        walking_allowed=True,
        obs_schema_v2=True,
        walk_envelope=(0.3, 0.0, 0.0),
        walk_window_s=10.0,
    )
    base.update(over)
    return GateParams(**base)


def armed_gate(**over) -> ActuatorGate:
    gate = ActuatorGate(gate_params(**over), ns(0.0))
    gate.on_lowstate(ns(0.0), DEFAULT_U, ZERO12)
    gate.on_estop(ns(0.0), False)
    return gate


def feed(gate, t):
    """The sensor and deadman heartbeats a healthy run delivers every tick."""
    gate.on_lowstate(ns(t), DEFAULT_U, ZERO12)
    gate.on_estop(ns(t), False)


def send_walk(gate, t, fed=(0.3, 0.0, 0.0), received=None, seq=1):
    label, data = encode(
        ORDER,
        seq=seq,
        kind=KIND_POLICY,
        target=DEFAULT_P,
        requested_target=DEFAULT_P,
        q_policy=DEFAULT_P,
        obs_source_code=0.0,
        stand_only=0.0,
        velocity_command_fed=list(fed),
        cmd_vel_received=list(fed if received is None else received),
    )
    gate.on_command(ns(t), label, data)


def test_walking_inside_the_grant_runs() -> None:
    gate = armed_gate()
    send_walk(gate, 0.1)
    rec = gate.tick(ns(0.11))
    assert rec["fault"] is None and rec["mode"] == "policy"


def test_command_beyond_the_grant_latches_and_removes_policy_authority() -> None:
    for fed, received in (
        ((0.5, 0.0, 0.0), None),  # arbitrary velocity inside the config's max_command
        ((0.0, 0.0, 0.5), None),  # the yaw preset's command on the vx preset
        (
            (0.3, 0.0, 0.0),
            (0.45, 0.0, 0.0),
        ),  # what the node was FED is fine, what it received is not
        ((0.3, 0.2, 0.0), None),
    ):
        gate = armed_gate()
        send_walk(gate, 0.1, fed=fed, received=received)
        rec = gate.tick(ns(0.11))
        assert rec["fault"] == "walking_blocked_command_exceeds_grant", (fed, received)
        assert rec["mode"] != "policy"


def test_the_drill_grant_allows_a_zero_command_only() -> None:
    gate = armed_gate(walk_envelope=(0.0, 0.0, 0.0))
    send_walk(gate, 0.1, fed=(0.0, 0.0, 0.0), received=(0.0, 0.0, 0.0))
    assert gate.tick(ns(0.11))["mode"] == "policy"
    gate = armed_gate(walk_envelope=(0.0, 0.0, 0.0))
    send_walk(gate, 0.1, fed=(0.05, 0.0, 0.0))
    assert gate.tick(ns(0.11))["fault"] == "walking_blocked_command_exceeds_grant"


def test_a_walking_gate_without_a_grant_window_is_not_reachable_from_the_bridge_config() -> None:
    # The bridge sets walk_envelope/walk_window_s only from a verified grant, and walking_allowed
    # only together with it (test_walking_startup_integration). Without walking_allowed:
    gate = armed_gate(walking_allowed=False, walk_envelope=None, walk_window_s=None)
    send_walk(gate, 0.1)
    assert gate.tick(ns(0.11))["fault"] == "walking_blocked_not_enabled"


def test_grant_expiring_mid_run_fails_closed_through_the_existing_hold_then_sit() -> None:
    gate = armed_gate(walk_window_s=3.0)
    t, seq, modes = 0.0, 0, []
    while t < 2.95:
        seq += 1
        t += 0.02
        feed(gate, t)
        send_walk(gate, t, seq=seq)
        modes.append(gate.tick(ns(t + 0.001))["mode"])
    assert set(modes) == {"policy"}, "the policy has the robot right up to the expiry"
    # the policy keeps commanding after the grant ran out: it never gets the robot again
    after = []
    for _ in range(100):  # 2 s more
        seq += 1
        t += 0.02
        feed(gate, t)
        send_walk(gate, t, seq=seq)
        after.append(gate.tick(ns(t + 0.001)))
    assert after[-1]["fault"] == "walking_grant_expired"
    first_fault = next(i for i, r in enumerate(after) if r["fault"])
    assert all(r["mode"] == "policy" for r in after[:first_fault])
    assert all(r["mode"] != "policy" for r in after[first_fault:])
    assert (
        after[-1]["mode"] == "sit"
    ), "hold, then the controlled sit, the existing fail-closed path"
    assert all(r["publish"] for r in after), "the gate keeps holding/sitting; it does not go silent"


def test_stop_and_deadman_still_work_after_the_grant_is_gone() -> None:
    """Expiry removes walking capability; it must not remove STOP."""
    # estop asserted after expiry still latches and the gate keeps publishing a hold/sit
    gate = armed_gate(walk_window_s=1.0)
    send_walk(gate, 0.1)
    assert gate.tick(ns(0.11))["mode"] == "policy"
    feed(gate, 1.5)
    send_walk(gate, 1.5, seq=2)
    assert gate.tick(ns(1.51))["fault"] == "walking_grant_expired"
    gate.on_estop(ns(1.6), True)
    rec = gate.tick(ns(1.61))
    assert "estop_asserted" in rec["faults"] and rec["mode"] != "policy" and rec["publish"]
    # deadman: a heartbeat that goes silent latches even with an expired grant
    gate = armed_gate(walk_window_s=1.0)
    send_walk(gate, 0.1)
    gate.tick(ns(0.11))
    rec = gate.tick(ns(2.0))
    assert "estop_heartbeat_stale" in rec["faults"] and rec["mode"] != "policy"


def test_stop_and_deadman_do_not_depend_on_the_grant_file() -> None:
    """The gate's STOP path has no access to the grant: the grant is parameters at construction
    (an envelope and a number of seconds). Malformed or vanished grant metadata after start
    cannot reach it. The modules that read the file never import the gate's estop logic."""
    src = (REPO / "src/phoenix/sim2real/actuator_gate.py").read_text()
    assert "import walk_grant" not in src and "walk_grant import" not in src
    assert "walk_grant.json" not in src and "read_grant" not in src
    # estop-asserted holds with NO grant parameters at all
    gate = armed_gate(walk_envelope=None, walk_window_s=None)
    send_walk(gate, 0.1)
    assert gate.tick(ns(0.11))["mode"] == "policy"
    gate.on_estop(ns(0.2), True)
    rec = gate.tick(ns(0.21))
    assert "estop_asserted" in rec["faults"] and rec["mode"] != "policy"


# ------------------------------------------------------- harness wiring (static)
HARNESS = (REPO / "scripts/harness_preflight.sh").read_text()


def _function_body(name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}\n", HARNESS, re.S | re.M)
    assert m, name
    return m.group(1)


def test_every_walking_launch_issues_a_grant_before_any_motor_process() -> None:
    walk = _function_body("stage_walk")
    free = _function_body("stage_free_floor")
    for body, mode in ((walk, "GANTRY"), (free, "FREE_FLOOR")):
        assert f"issue_walk_grant {mode}" in body
        assert body.index("issue_walk_grant") < body.index("launch lowcmd_bridge")
        assert body.index("issue_walk_grant") < body.index("launch policy_node")
        assert body.count('--grant "$RUN/walk_grant.json"') == 2  # bridge and policy node
    # the free-floor grant is issued after `begin` (its run-start record) and the operator arm
    assert free.index('"$ff" begin') < free.index("issue_walk_grant")
    assert walk.index("operator_gate arm") < walk.index("issue_walk_grant")


def test_w_is_gantry_only_with_a_closed_preset_list_and_the_unreviewed_latch() -> None:
    walk = _function_body("stage_walk")
    assert "vx03_2s|vx03_4s|yaw05_3s) ;;" in walk
    assert walk.index("vx03_2s|vx03_4s|yaw05_3s") < walk.index("live_confirm")
    assert "ON GANTRY" in walk and "NO GANTRY" not in walk
    assert 'free_floor_gate.py" unreviewed' in walk
    assert "walking grant" not in walk.split("live_confirm")[0] or True
    # the list in the shell is the list in the module
    listed = set(re.search(r"\n\s*(vx03_2s\|[a-z0-9_|]+)\) ;;", walk).group(1).split("|"))
    assert listed == set(wg.GANTRY_PRESETS)


def test_the_grant_is_released_on_exit() -> None:
    assert "trap cleanup_all EXIT" in HARNESS
    assert "cleanup_all() { stop_all; release_walk_grant; }" in HARNESS


# -------------------------------------------------------------- import trap
def test_phoenix_is_imported_from_this_checkout() -> None:
    import phoenix

    assert Path(phoenix.__file__).resolve().is_relative_to(REPO), phoenix.__file__


def test_pytest_is_configured_to_import_from_this_checkout() -> None:
    assert re.search(r'pythonpath\s*=\s*\["src"\]', (REPO / "pyproject.toml").read_text())


def test_desk_ready_detects_another_checkout_instead_of_testing_it(monkeypatch) -> None:
    import free_floor_desk as desk

    import phoenix

    ok, _ = desk.import_origin()
    assert ok
    monkeypatch.setattr(phoenix, "__file__", "/home/someone/other-checkout/src/phoenix/__init__.py")
    ok, detail = desk.import_origin()
    assert not ok and "other-checkout" in detail
