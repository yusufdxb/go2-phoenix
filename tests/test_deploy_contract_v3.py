"""Phoenix v2 deploy contract, frozen as contract v3 (docs/research/DEPLOY_CONTRACT.md).

The training side is Isaac Lab, which CI cannot import. ``_isaac_reference_step`` below
re-states the three source lines that define it, verified against the installed Isaac
Lab on 2026-09-22 (``isaaclab_rl/rsl_rl/vecenv_wrapper.py:171``,
``isaaclab/managers/action_manager.py:388-389``,
``isaaclab/envs/mdp/actions/joint_actions.py:172-174``,
``isaaclab/envs/mdp/observations.py:671``). The same comparison against the LIVE env,
not this re-statement, is ``scripts/phoenix_v2_contract_check.py``; its result is
under ``results/phoenix_v2/contract/``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from phoenix.sim2real.action_map import node_soft_limit, policy_action_map
from phoenix.sim2real.go2_model import POLICY_JOINT_ORDER, TRAINING_DEFAULT_JOINT_POS
from phoenix.sim2real.observation import (
    OBS_TERM_ORDER,
    JointOrder,
    ObservationBuilder,
    assemble_policy_observation,
    term_slices,
)
from phoenix.sim2real.safety import TRAINED_ACTION_CLIP, rate_limit_array

PATHOLOGICAL = np.array([-10.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 10.0], dtype=np.float32)
DEFAULT = np.asarray([TRAINING_DEFAULT_JOINT_POS[n] for n in POLICY_JOINT_ORDER], np.float32)
SCALE = 0.25


def _isaac_reference_step(raw: np.ndarray, clip: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Training plant, one policy step: ``(ActionManager.action, processed target)``.

    wrapper:        actions = clamp(actions, -clip, clip)
    ActionManager:  self._action[:] = action          (what last_action observes)
    JointPosition:  processed = raw_actions * scale + offset   (offset = default pose)
    """
    a = np.clip(raw.astype(np.float32), -clip, clip)
    return a, a * np.float32(SCALE) + DEFAULT


def _rows() -> np.ndarray:
    """Every pathological value on every joint, plus mixed-sign rows."""
    rows = [np.full(12, v, np.float32) for v in PATHOLOGICAL]
    rng = np.random.default_rng(7)
    rows += [rng.choice(PATHOLOGICAL, 12).astype(np.float32) for _ in range(32)]
    return np.stack(rows)


@pytest.mark.parametrize("value", PATHOLOGICAL.tolist())
def test_clamp_parity_on_each_pathological_value(value: float) -> None:
    raw = np.full(12, value, np.float32)
    fed, target = policy_action_map(raw, DEFAULT, SCALE, TRAINED_ACTION_CLIP)
    ref_a, ref_t = _isaac_reference_step(raw)
    np.testing.assert_allclose(fed, ref_a, atol=0, rtol=0)
    np.testing.assert_allclose(target, ref_t, atol=1e-7)
    assert np.all(np.abs(fed) <= 1.0)
    assert np.all(np.abs(target - DEFAULT) <= SCALE + 1e-7)


def test_action_scale_and_joint_order_parity() -> None:
    for raw in _rows():
        _, target = policy_action_map(raw, DEFAULT, SCALE, TRAINED_ACTION_CLIP)
        _, ref_t = _isaac_reference_step(raw)
        np.testing.assert_allclose(target, ref_t, atol=1e-7)
    # One joint at a time: the index that moves is the index that was commanded.
    for j, name in enumerate(POLICY_JOINT_ORDER):
        raw = np.zeros(12, np.float32)
        raw[j] = 3.0
        _, target = policy_action_map(raw, DEFAULT, SCALE, TRAINED_ACTION_CLIP)
        moved = np.flatnonzero(np.abs(target - DEFAULT) > 0)
        assert moved.tolist() == [j], name
        assert target[j] == pytest.approx(TRAINING_DEFAULT_JOINT_POS[name] + SCALE)


def test_last_action_parity_over_a_sequence() -> None:
    """The fed-back action at step k is the training plant's ActionManager.action at k."""
    rng = np.random.default_rng(11)
    seq = rng.choice(PATHOLOGICAL, (200, 12)).astype(np.float32)
    builder = ObservationBuilder(JointOrder(POLICY_JOINT_ORDER), TRAINING_DEFAULT_JOINT_POS)
    last_deploy = np.zeros(12, np.float32)  # ActionManager.reset zeroes _action
    last_train = np.zeros(12, np.float32)
    sl = term_slices(12)["last_action"]
    for raw in seq:
        obs = builder.build(
            base_lin_vel=np.zeros(3, np.float32),
            base_ang_vel=np.zeros(3, np.float32),
            projected_gravity=np.array([0, 0, -1], np.float32),
            velocity_command=np.zeros(3, np.float32),
            joint_pos=DEFAULT.copy(),
            joint_vel=np.zeros(12, np.float32),
            last_action=last_deploy,
        )
        np.testing.assert_array_equal(obs[sl], last_train)
        last_deploy, _ = policy_action_map(raw, DEFAULT, SCALE, TRAINED_ACTION_CLIP)
        last_train, _ = _isaac_reference_step(raw)
    np.testing.assert_array_equal(last_deploy, last_train)


def test_legacy_map_is_the_defect_the_contract_forbids() -> None:
    """action_clip=None (pre-v2) feeds 10.0 back; the trained plant never sees above 1."""
    raw = np.full(12, 10.0, np.float32)
    fed, target = policy_action_map(raw, DEFAULT, SCALE, None)
    assert fed.max() == 10.0
    assert np.abs(target - DEFAULT).max() == pytest.approx(2.5)


def test_observation_order_and_units_match_the_training_terms() -> None:
    assert OBS_TERM_ORDER == (
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "velocity_command",
        "joint_pos",
        "joint_vel",
        "last_action",
    )
    builder = ObservationBuilder(JointOrder(POLICY_JOINT_ORDER), TRAINING_DEFAULT_JOINT_POS)
    q = DEFAULT + np.arange(12, dtype=np.float32) * 0.01
    obs = assemble_policy_observation(
        builder,
        base_lin_vel=np.array([0.1, 0.2, 0.3], np.float32),
        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        base_ang_vel=np.array([0.4, 0.5, 0.6], np.float32),
        velocity_command=np.array([0.7, 0.0, -0.2], np.float32),
        joint_pos=q,
        joint_vel=np.full(12, 2.0, np.float32),
        last_action=np.full(12, 0.5, np.float32),
    )
    sl = term_slices(12)
    assert obs.shape == (48,)
    np.testing.assert_allclose(obs[sl["projected_gravity"]], [0, 0, -1])
    # joint_pos_rel: Isaac's mdp.joint_pos_rel is q - default_joint_pos, no scale.
    np.testing.assert_allclose(obs[sl["joint_pos"]], q - DEFAULT, atol=1e-7)
    np.testing.assert_allclose(obs[sl["velocity_command"]], [0.7, 0.0, -0.2])


def test_rate_limiter_runs_after_the_clamp_and_only_on_sent_targets() -> None:
    """Deploy chain: clamp -> scale -> rate limit (bridge) -> hard envelope.

    A request inside the bound passes bit-exact; a step request is spread over ticks and
    converges to exactly the clamped target, never to the unclamped one.
    """
    dq = 0.075
    prev = DEFAULT.copy()
    raw = np.full(12, 10.0, np.float32)
    _, req = policy_action_map(raw, DEFAULT, SCALE, TRAINED_ACTION_CLIP)
    for _ in range(10):
        sent = np.asarray(rate_limit_array(req, prev, dq), np.float32)
        assert np.all(np.abs(sent - prev) <= dq + 1e-6)
        prev = sent
    np.testing.assert_allclose(prev, DEFAULT + SCALE, atol=1e-6)
    small = DEFAULT + 0.01
    np.testing.assert_array_equal(np.asarray(rate_limit_array(small, DEFAULT, dq), np.float32), small)
    # The node itself never rate-limits in prev_command mode (single limiter, in the bridge).
    np.testing.assert_array_equal(node_soft_limit(req, DEFAULT, "prev_command", dq), req)


def test_mode_switch_path_uses_the_same_action_map() -> None:
    """The dual-policy blend used to skip the clamp on target AND last_action."""
    node_mod = pytest.importorskip("phoenix.sim2real.ros2_policy_node")
    from phoenix.sim2real.mode_switch import ModeSwitchCfg, initial_state

    class _Sess:
        def __init__(self, value: float) -> None:
            self.value = value

        def run(self, _names, _feed):
            return [np.full((1, 12), self.value, np.float32)]

    state, ticks = initial_state(ModeSwitchCfg())
    fake = SimpleNamespace(
        mode_cfg=ModeSwitchCfg(),
        stand_session=_Sess(9.0),
        walk_session=_Sess(-7.0),
        obs_builder=ObservationBuilder(JointOrder(POLICY_JOINT_ORDER), TRAINING_DEFAULT_JOINT_POS),
        default_q=DEFAULT,
        action_scale=SCALE,
        action_clip=TRAINED_ACTION_CLIP,
        _velocity_command=np.zeros(3, np.float32),
        _last_action_stand=np.zeros(12, np.float32),
        _last_action_walk=np.zeros(12, np.float32),
        _last_action=np.zeros(12, np.float32),
        obs_pad_zeros=0,
        mode_state=state,
        mode_ticks=ticks,
    )
    target, active = node_mod._PhoenixPolicyNode._compute_mode_switch_target(
        fake,
        q=DEFAULT,
        qd=np.zeros(12, np.float32),
        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        base_ang_vel=np.zeros(3, np.float32),
        base_lin_vel=np.zeros(3, np.float32),
    )
    np.testing.assert_allclose(target, DEFAULT + SCALE, atol=1e-6)  # stand active, clamped
    np.testing.assert_array_equal(fake._last_action_stand, np.ones(12, np.float32))
    assert float(active.max()) == 9.0  # the logged raw action stays raw (layer 1)


def test_heading_command_is_wired_and_off_means_direct_yaw_rate() -> None:
    from phoenix.sim_env.go2_env_cfg import _apply_commands

    vel = SimpleNamespace(
        ranges=SimpleNamespace(),
        resampling_time_range=None,
        rel_standing_envs=0.0,
        heading_command=True,
        rel_heading_envs=1.0,
        heading_control_stiffness=0.5,
    )
    env_cfg = SimpleNamespace(commands=SimpleNamespace(base_velocity=vel))
    cmd = {
        "lin_vel_x": [-0.5, 1.0],
        "lin_vel_y": [-0.3, 0.3],
        "ang_vel_z": [-0.5, 0.5],
        "resample_time_s": 10.0,
        "rel_standing_envs": 0.05,
        "heading_command": False,
        "heading_stiffness": 0.5,
    }
    _apply_commands(env_cfg, cmd)
    assert vel.heading_command is False and vel.rel_heading_envs == 0.0
    assert vel.ranges.ang_vel_z == (-0.5, 0.5)
    assert vel.resampling_time_range == (10.0, 10.0)


def _walk_fixture(n_t=250, n_e=3, drift=False):
    from phoenix.monitor.stand_metrics import score_stand_rollout

    T, N = n_t, n_e  # noqa: N806
    dt = 0.02
    cmd = np.zeros((T, N, 3), np.float32)
    cmd[:, :, 0] = 0.5
    if drift:  # heading-style yaw command that changes a little every step
        cmd[:, :, 2] = 0.3 + 0.001 * np.arange(T)[:, None]
    linv = cmd.copy()
    angv = cmd.copy()
    linv[:, 2, 0] = 0.1  # env 2 barely moves forward
    req = np.tile(DEFAULT, (T, N, 1))
    z = np.zeros((T, N, 12), np.float32)
    grav = np.tile(np.array([0, 0, -1], np.float32), (T, N, 1))
    height = np.full((T, N), 0.30, np.float32)
    alive = np.ones((T, N), bool)
    lo = DEFAULT - 1.0
    hi = DEFAULT + 1.0
    metrics, eps = score_stand_rollout(
        raw=z, req=req, sent=req.copy(), q0=req.copy(), q1=req.copy(), tau_c=z, tau_a=z,
        grav=grav, height=height, linv=linv, angv=angv, cmd=cmd, alive=alive,
        contact_term=np.zeros(N, bool), ended=np.zeros(N, bool), default=DEFAULT, lo=lo, hi=hi,
        dt=dt, abort_band=0.175, joint_names=list(POLICY_JOINT_ORDER),
    )
    return eps, dict(linv=linv, angv=angv, cmd=cmd, valid=alive, height=height, qd=z, req=req,
                     tau_c=z, tau_a=z, dt=dt)


def test_walk_v2_keeps_a_settled_window_under_a_drifting_yaw_command() -> None:
    """Regression for the Gate L scorer defect: amendment 3's scorer empties the window."""
    from phoenix.monitor.stand_metrics import score_walk_episodes, score_walk_v2

    eps, kw = _walk_fixture(drift=True)
    score_walk_episodes(eps, linv=kw["linv"], angv=kw["angv"], cmd=kw["cmd"], valid=kw["valid"], dt=kw["dt"])
    assert all(ep["mean_lin_vel_error_m_s"] == float("inf") for ep in eps)  # the defect
    out = score_walk_v2(eps, **kw)
    assert out["walk2_settled_fraction"] == pytest.approx(0.8)  # 250 steps minus the 1 s settle
    assert [ep["walk2_success"] for ep in eps] == [True, True, False]
    assert eps[2]["walk2_checks"]["progress"] is False and eps[2]["walk2_checks"]["lin_err"] is False
    assert eps[2]["walk2_progress_ratio"] == pytest.approx(0.2, abs=1e-6)


def test_walk_v2_flags_collapse_and_joint_speed() -> None:
    from phoenix.monitor.stand_metrics import score_walk_v2

    eps, kw = _walk_fixture()
    kw["height"] = kw["height"].copy()
    kw["height"][100, 0] = 0.12
    kw["qd"] = kw["qd"].copy()
    kw["qd"][50, 1, 3] = 40.0
    score_walk_v2(eps, **kw)
    assert eps[0]["walk2_checks"]["height"] is False
    assert eps[1]["walk2_checks"]["joint_speed"] is False


def test_h25_v3_deploy_config_carries_the_amendment_7_values() -> None:
    from pathlib import Path

    import yaml

    from phoenix.sim2real.deploy_contract import (
        load_lock,
        semantic_config_sha256,
        validate_deploy_contract,
    )

    path = Path("configs/sim2real/deploy_stand_h25_v3.yaml")
    cfg = yaml.safe_load(path.read_text())
    assert validate_deploy_contract(cfg) == []
    assert cfg["limiter"]["mode"] == "prev_command"
    assert cfg["limiter"]["max_delta_per_step"] == 0.035
    assert cfg["limiter"]["tracking_abort_rad"] == 1.4
    assert cfg["control"]["action_clip"] == 1.0
    lock = load_lock(Path("configs/sim2real/locks/deploy_stand_h25_v3.lock.yaml"))
    assert lock["deploy_config"]["semantic_sha256"] == semantic_config_sha256(cfg)
