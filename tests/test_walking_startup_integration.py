"""Walking startup, end to end, for the real model_2750 artifacts (no robot).

The deploy contract exempts a ``phoenix-velocity-obs/v2`` (45-D, no base_lin_vel
term) policy from the "zeros requires stand_only" rule, but the runtime that
actually starts on the robot repeated the old rule in two more places: the policy
node's constructor and the LowCmd bridge's per-command walking check. The stand
configs (stages F/G/H) never reach either, so only a test that starts the WALKING
config through the production path can show that the first walk is not refused.

What is real here: the walking deploy config and its stand-phase twin, the archived
model_2750 ONNX, manifest and checkpoint, the lock generator, the stage-record
writer (``preflight.write_stage``), the evidence check
(``deploy_contract.walking_evidence_problems``), ``resolve_startup`` with the argv the
harness uses, ``_PhoenixPolicyNode.__init__`` and one ``_control_step``, and the
bridge's own config/startup path and ``ActuatorGate``.

What is synthetic: the F and G stage records and their telemetry files (fixtures,
written under ``tmp_path`` and read only through the documented test-only
``PHOENIX_HW_SESSIONS`` override), the sensor messages, and the publisher, which is
replaced by a recorder after construction so no command leaves the process. ROS runs
localhost-only on a private domain. No robot, Jetson or actuator is involved.

Needs ROS 2 (rclpy, message packages), onnxruntime and the archived artifacts;
skipped otherwise (CI).
"""

from __future__ import annotations

import os
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.ros

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("onnxruntime")
pytest.importorskip("sensor_msgs.msg")

from phoenix.sim2real import preflight  # noqa: E402
from phoenix.sim2real import preflight_eval as pe  # noqa: E402
from phoenix.sim2real.actuator_gate import ActuatorGate, Mode  # noqa: E402
from phoenix.sim2real.command_wire import KIND_POLICY, decode, wire_label  # noqa: E402
from phoenix.sim2real.deploy_contract import (  # noqa: E402
    HW_SESSIONS_ENV,
    HW_SESSIONS_TEST_FLAG,
    artifact_paths,
    build_lock,
    load_lock,
    semantic_config_sha256,
    stand_twin,
    validate_deploy_contract,
    walking_evidence_problems,
)
from phoenix.sim2real.go2_model import (  # noqa: E402
    POLICY_JOINT_ORDER,
    TRAINING_DEFAULT_JOINT_POS,
    UNITREE_MOTOR_ORDER,
)
from phoenix.sim2real.motor_crc import phoenix_to_unitree  # noqa: E402
from phoenix.sim2real.provenance import resolve_code_identity  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
WALK_CFG = REPO / "configs/sim2real/proposed/deploy_walk_iter2750.yaml"
STAND_CFG = REPO / "configs/sim2real/deploy_walk_iter2750_stand.yaml"
STAND_LOCK = REPO / "configs/sim2real/locks/deploy_walk_iter2750_stand.lock.yaml"
#: model_2750 ONNX sha256 prefix recorded in the stand lock (e03013d1 is the .pt).
CHECKPOINT_SHA_PREFIX = "e03013d1"

_walk = yaml.safe_load(WALK_CFG.read_text())
if not all(p.is_file() for p in artifact_paths(_walk).values()):
    pytest.skip("archived model_2750 artifacts are not on this machine", allow_module_level=True)


def _walk_cfg() -> dict:
    return yaml.safe_load(WALK_CFG.read_text())


# --------------------------------------------------------------- F/G fixtures
def _write_stage_fixture(session: Path, stage: str) -> Path:
    """One synthetic GO record written by the production writer under the REAL stand lock."""
    run_dir = session / f"run_{stage}1"
    run_dir.mkdir(parents=True)
    (run_dir / "bridge.jsonl").write_text(f'{{"fixture": "synthetic stage {stage} telemetry"}}\n')
    lock = load_lock(STAND_LOCK)
    ctx = types.SimpleNamespace(
        identity=resolve_code_identity(REPO),
        id_problems=[],  # fixture: the record claims a clean code identity
        cfg_path=STAND_CFG,
        lock_path=STAND_LOCK,
        lock=lock,
        lock_sha=preflight.file_sha256(STAND_LOCK),
        session=session,
        label=lock.get("name"),
    )
    checks = [pe.Check(name="synthetic fixture", ok=True, detail="test only, not hardware")]
    preflight.write_stage(ctx, stage, checks, {"run_dir": str(run_dir)})
    return session / f"stage_{stage}.json"


@pytest.fixture
def sessions(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "hw_sessions"
    root.mkdir()
    monkeypatch.setenv(HW_SESSIONS_ENV, str(root))
    monkeypatch.setenv(HW_SESSIONS_TEST_FLAG, "1")
    monkeypatch.delenv("PHOENIX_REHEARSAL", raising=False)
    return root


@pytest.fixture
def fg_evidence(sessions) -> Path:
    session = sessions / "2026-10-04_fixture"
    for stage in ("F", "G"):
        _write_stage_fixture(session, stage)
    return session


@pytest.fixture
def walk_lock(tmp_path) -> Path:
    cfg = _walk_cfg()
    lock = build_lock(
        cfg,
        WALK_CFG,
        name="walk-iter2750-fixture",
        repo_config_path=str(WALK_CFG.relative_to(REPO)),
        checkpoint=cfg["policy"]["checkpoint_path"],
    )
    path = tmp_path / "walk.lock.yaml"
    path.write_text(yaml.safe_dump(lock, sort_keys=False))
    return path


# ------------------------------------------------- the artifacts are the gated ones
def test_walking_config_is_the_stand_lock_twin_with_the_same_artifacts(walk_lock) -> None:
    stand_lock = load_lock(STAND_LOCK)
    assert semantic_config_sha256(stand_twin(_walk_cfg())) == (
        stand_lock["deploy_config"]["semantic_sha256"]
    )
    walk = load_lock(walk_lock)
    for role in ("policy.onnx", "policy.onnx.data", "policy.manifest", "checkpoint"):
        assert walk["artifacts"][role]["sha256"] == stand_lock["artifacts"][role]["sha256"], role
    from phoenix.sim2real.activation import file_sha256

    ckpt = artifact_paths(_walk_cfg())["checkpoint"]
    assert file_sha256(ckpt).startswith(CHECKPOINT_SHA_PREFIX)


# --------------------------------------------- evidence gate (negative controls)
def test_walking_is_refused_without_fg_evidence(sessions) -> None:
    problems = validate_deploy_contract(_walk_cfg())
    assert any("stage F has no GO hardware record" in p for p in problems)


def test_walking_is_refused_with_only_stage_f(sessions) -> None:
    _write_stage_fixture(sessions / "only_f", "F")
    problems = walking_evidence_problems(_walk_cfg())
    assert len(problems) == 1 and "stage G" in problems[0]


def test_walking_is_refused_when_evidence_telemetry_changed(fg_evidence) -> None:
    (fg_evidence / "run_G1" / "bridge.jsonl").write_text("edited after the record\n")
    problems = walking_evidence_problems(_walk_cfg())
    assert any("stage G" in p and "hash differs" in p for p in problems)


def test_walking_is_refused_for_a_rehearsal_record(sessions, monkeypatch) -> None:
    monkeypatch.setenv("PHOENIX_REHEARSAL", "1")
    for stage in ("F", "G"):
        _write_stage_fixture(sessions / "rehearsal", stage)
    assert walking_evidence_problems(_walk_cfg())


def test_evidence_override_without_the_test_flag_never_counts(fg_evidence, monkeypatch) -> None:
    monkeypatch.delenv(HW_SESSIONS_TEST_FLAG)
    assert any("test-only" in p for p in walking_evidence_problems(_walk_cfg()))


# ----------------------------------------------------- the production startup path
def test_contract_passes_once_fg_evidence_exists(fg_evidence) -> None:
    assert validate_deploy_contract(_walk_cfg()) == []


def test_resolve_startup_with_the_harness_argv_passes(fg_evidence, walk_lock) -> None:
    from phoenix.sim2real.ros2_policy_node import parse_args, resolve_startup

    args = parse_args(
        [
            "--config",
            str(WALK_CFG),
            "--lock",
            str(walk_lock),
            "--authority-s",
            "10",
            "--max-runtime-s",
            "25",
        ]
    )
    cfg = yaml.safe_load(args.config.read_text())
    onnx, problems = resolve_startup(cfg, args)
    assert problems == []
    assert onnx == Path(cfg["policy"]["onnx_path"])
    assert cfg["safety"]["stand_only"] is False


class _Recorder:
    def __init__(self) -> None:
        self.msgs: list = []

    def publish(self, msg) -> None:
        self.msgs.append(msg)


@pytest.fixture
def isolated_ros(monkeypatch):
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setenv("ROS_DOMAIN_ID", "199")
    monkeypatch.delenv("CYCLONEDDS_URI", raising=False)
    rclpy.init()
    try:
        yield
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def _sensor_msgs(node, *, cmd_vx: float):
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Imu, JointState
    from std_msgs.msg import Bool

    imu = Imu()
    imu.orientation.w = 1.0  # upright: projected gravity (0, 0, -1)
    js = JointState()
    names = list(reversed(POLICY_JOINT_ORDER))  # by-name remap must be exercised
    js.name = names
    js.position = [float(TRAINING_DEFAULT_JOINT_POS[n]) for n in names]
    js.velocity = [0.0] * 12
    estop = Bool()
    estop.data = False
    twist = Twist()
    twist.linear.x = cmd_vx
    node._on_estop(estop)
    node._on_imu(imu)
    node._on_joint_state(js)
    node._on_cmd_vel(twist)


def test_policy_node_constructs_and_reaches_the_first_walking_command(
    fg_evidence, isolated_ros
) -> None:
    from phoenix.sim2real.ros2_policy_node import _PhoenixPolicyNode

    cfg = _walk_cfg()
    node = _PhoenixPolicyNode(cfg, Path(cfg["policy"]["onnx_path"]), authority_s=10.0)
    try:
        assert node.stand_only is False
        assert node.base_lin_vel_source == "zeros"
        assert node.actor_obs_builder is not None and node.actor_obs_builder.dim == 45
        assert node.session.get_inputs()[0].shape[-1] == 45
        recorder = _Recorder()
        node.cmd_pub = recorder  # transport only: nothing is put on DDS
        _sensor_msgs(node, cmd_vx=0.3)
        node._control_step()
        assert not node._estopped, node._abort_reason
        assert len(recorder.msgs) == 1
        msg = recorder.msgs[0]
        cmd = decode(msg.layout.dim[0].label, msg.data, wire_label(POLICY_JOINT_ORDER))
        assert cmd.kind == KIND_POLICY
        assert cmd.fields["stand_only"][0] == 0.0
        assert np.allclose(cmd.fields["velocity_command_fed"], [0.3, 0.0, 0.0])
        assert np.all(np.isfinite(cmd.target))
    finally:
        node.node.destroy_node()


def test_bridge_accepts_the_policy_nodes_first_walking_command(
    fg_evidence, walk_lock, isolated_ros, tmp_path
) -> None:
    """The bridge's own production config path, fed the node's real first wire message."""
    pytest.importorskip("unitree_go.msg")
    from phoenix.sim2real.lowcmd_bridge_node import _build_config, _parse_args, startup_problems
    from phoenix.sim2real.ros2_policy_node import _PhoenixPolicyNode

    cfg = _walk_cfg()
    node = _PhoenixPolicyNode(cfg, Path(cfg["policy"]["onnx_path"]), authority_s=10.0)
    try:
        recorder = _Recorder()
        node.cmd_pub = recorder
        _sensor_msgs(node, cmd_vx=0.3)
        node._control_step()
        msg = recorder.msgs[0]
    finally:
        node.node.destroy_node()

    args = _parse_args(
        ["--config", str(WALK_CFG), "--lock", str(walk_lock), "--telemetry", str(tmp_path / "t")]
    )
    bridge_cfg = _build_config(args)
    problems, _manifest = startup_problems(bridge_cfg)
    assert problems == []
    params = bridge_cfg.gate_params()
    assert params.walking_allowed is True

    t0 = 1_000_000_000
    gate = ActuatorGate(params, t0)
    default_p = [TRAINING_DEFAULT_JOINT_POS[n] for n in POLICY_JOINT_ORDER]
    q_u = phoenix_to_unitree(default_p)
    assert len(q_u) == len(UNITREE_MOTOR_ORDER)
    gate.on_lowstate(t0, q_u, [0.0] * 12)
    gate.on_imu(t0, (1.0, 0.0, 0.0, 0.0))
    gate.on_estop(t0, False)
    gate.on_command(t0 + 1_000_000, msg.layout.dim[0].label, list(msg.data))
    rec = gate.tick(t0 + 2_000_000)
    assert rec["fault"] is None, rec["fault"]
    assert rec["mode"] == Mode.POLICY.value


def test_bridge_still_refuses_walking_for_a_stand_only_config(
    fg_evidence, isolated_ros, tmp_path
) -> None:
    pytest.importorskip("unitree_go.msg")
    from phoenix.sim2real.lowcmd_bridge_node import _build_config, _parse_args

    args = _parse_args(
        ["--config", str(STAND_CFG), "--lock", str(STAND_LOCK), "--telemetry", str(tmp_path / "t")]
    )
    assert _build_config(args).gate_params().walking_allowed is False


def test_bridge_walking_flag_requires_evidence(sessions, walk_lock, tmp_path) -> None:
    pytest.importorskip("unitree_go.msg")
    from phoenix.sim2real.lowcmd_bridge_node import _build_config, _parse_args, startup_problems

    args = _parse_args(
        ["--config", str(WALK_CFG), "--lock", str(walk_lock), "--telemetry", str(tmp_path / "t")]
    )
    bridge_cfg = _build_config(args)
    problems, _ = startup_problems(bridge_cfg)
    assert any("stage F has no GO hardware record" in p for p in problems)
    assert bridge_cfg.gate_params().walking_allowed is False


class _RecordingSession:
    """The real onnxruntime session, with every observation it is fed recorded."""

    def __init__(self, session) -> None:
        self._session = session
        self.obs: list[np.ndarray] = []

    def run(self, outputs, feeds):
        self.obs.append(np.array(feeds["obs"], copy=True))
        return self._session.run(outputs, feeds)

    def __getattr__(self, name):
        return getattr(self._session, name)


def _roll_quat_xyzw(roll: float) -> tuple[float, float, float, float]:
    return (float(np.sin(roll / 2)), 0.0, 0.0, float(np.cos(roll / 2)))


def test_first_policy_tick_uses_the_current_imu_sample_not_a_stale_one(
    fg_evidence, isolated_ros
) -> None:
    """The sim's reset-tick gravity mismatch has no deploy analogue.

    The 2026-10-04 rollout parity found Isaac's first observation after an env reset
    carries projected gravity up to 0.082 away from the post-reset pose (closer to the
    PRE-reset orientation). On the robot there is no reset and the v2 builder holds no
    state: every tick's gravity is computed from the IMU message the gate ladder just
    judged fresh. Pinned here on the real node: the very first policy tick, and the tick
    after an orientation change, each carry the gravity of the CURRENT sample, and the
    first tick's last_action is zero (the training reset value).
    """
    from sensor_msgs.msg import Imu

    from phoenix.sim2real.observation import projected_gravity_from_quat
    from phoenix.sim2real.observation_v2 import obs_slices
    from phoenix.sim2real.ros2_policy_node import _PhoenixPolicyNode

    cfg = _walk_cfg()
    node = _PhoenixPolicyNode(cfg, Path(cfg["policy"]["onnx_path"]), authority_s=10.0)
    try:
        node.cmd_pub = _Recorder()
        rec = _RecordingSession(node.session)
        node.session = rec
        _sensor_msgs(node, cmd_vx=0.0)
        g = obs_slices()["projected_gravity"]
        la = obs_slices()["last_action"]
        for roll in (0.25, -0.15, 0.0):
            imu = Imu()
            q = _roll_quat_xyzw(roll)
            imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w = q
            node._on_imu(imu)
            node._control_step()
            assert not node._estopped, node._abort_reason
            assert np.allclose(rec.obs[-1][0, g], projected_gravity_from_quat(*q), atol=1e-6)
        assert len(rec.obs) == 3
        assert np.all(rec.obs[0][0, la] == 0.0)
        assert np.any(rec.obs[1][0, la] != 0.0)  # fed back from tick 1, as in training
    finally:
        node.node.destroy_node()


def test_policy_never_runs_before_a_fresh_imu_sample(fg_evidence, isolated_ros) -> None:
    """No IMU sample yet, or a stale one: no observation is built and no policy runs."""
    from phoenix.sim2real.ros2_policy_node import _PhoenixPolicyNode

    cfg = _walk_cfg()
    node = _PhoenixPolicyNode(cfg, Path(cfg["policy"]["onnx_path"]), authority_s=10.0)
    try:
        node.cmd_pub = _Recorder()
        rec = _RecordingSession(node.session)
        node.session = rec
        _sensor_msgs(node, cmd_vx=0.0)
        node._latest_imu, node._latest_imu_ns, node._seen_imu = None, None, False
        node._control_step()  # startup: IMU never seen -> startup default, no policy
        assert rec.obs == [] and not node._estopped
        _sensor_msgs(node, cmd_vx=0.0)
        node._latest_imu_ns -= int(1e9)  # 1 s old, well past sensor_timeout_s 0.2
        node._control_step()
        assert rec.obs == []
        assert node._estopped and node._abort_reason == "sensor_stale"
    finally:
        node.node.destroy_node()


def test_isolated_ros_env_is_restored() -> None:
    # Guard against the fixtures above leaking a domain/localhost override into other tests.
    assert os.environ.get("ROS_DOMAIN_ID") != "199"
