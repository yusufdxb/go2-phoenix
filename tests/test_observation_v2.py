"""Deploy v2 (45-D) observation: parity with the training contract, manifest refusal, ONNX step."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from phoenix.sim2real.observation import JointOrder
from phoenix.sim2real.observation_v2 import (
    V2_SCHEMA_SHA256,
    ManifestObservationBuilder,
    ManifestObservationError,
    canonical_stand_observation,
    manifest_actor_schema,
    parity_failures,
    reference_observations,
)
from phoenix.velocity import contract as c
from phoenix.velocity.observation import build_actor_observation, quat_xyzw_to_wxyz

#: Training's per-term scales as the Isaac env config applies them (env_cfg._scale), written
#: out as literals so a change to the contract table cannot silently re-bless the deploy side.
TRAINING_V2_SCALES = {
    "base_ang_vel": 0.2,
    "projected_gravity": 1.0,
    "velocity_command": 1.0,
    "joint_pos_rel": 1.0,
    "joint_vel": 0.05,
    "last_action": 1.0,
}
TRAINING_V2_TERM_ORDER = [
    "base_ang_vel",
    "projected_gravity",
    "velocity_command",
    "joint_pos_rel",
    "joint_vel",
    "last_action",
]


def _manifest() -> dict:
    return {
        "observation": {
            "schema_version": c.OBS_SCHEMA_V2,
            "schema_sha256": c.obs_schema_sha256(c.OBS_SCHEMA_V2),
            "actor_dim": 45,
            "actor_terms": list(TRAINING_V2_TERM_ORDER),
            "actor_scales": dict(TRAINING_V2_SCALES),
        }
    }


def _builder() -> ManifestObservationBuilder:
    return ManifestObservationBuilder(c.OBS_SCHEMA_V2, c.JOINT_ORDER, c.DEFAULT_JOINT_POS)


def _rot(axis, angle):
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    kx = np.asarray([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * (kx @ kx)


def test_v2_hash_is_pinned_and_matches_the_contract() -> None:
    assert V2_SCHEMA_SHA256 == c.obs_schema_sha256(c.OBS_SCHEMA_V2)
    assert V2_SCHEMA_SHA256 == "53a41f30142f6afdec92428314ae5092886d81e5ac9a59936f12c634fad7805a"
    assert [t[0] for t in c.ACTOR_OBS_TERMS] == TRAINING_V2_TERM_ORDER
    assert c.actor_obs_scales(c.OBS_SCHEMA_V2) == TRAINING_V2_SCALES
    assert c.ACTOR_OBS_DIM == 45


def test_probe_parity_has_no_failures() -> None:
    assert parity_failures(_builder()) == []


@pytest.mark.parametrize("seed", range(5))
def test_deploy_builder_equals_training_terms_with_joint_permutation(seed) -> None:
    """Every term, exactly (<= 1e-6), including a shuffled /joint_states order."""
    rng = np.random.default_rng(seed)
    ros_names = list(c.JOINT_ORDER)
    rng.shuffle(ros_names)
    q_policy = np.asarray([c.DEFAULT_JOINT_POS[j] for j in c.JOINT_ORDER]) + rng.uniform(
        -0.4, 0.4, 12
    )
    qd_policy = rng.normal(0.0, 3.0, 12)
    # What arrives from ROS is in ros_names order; the node's remap puts it in policy order.
    q_ros = np.asarray([q_policy[c.JOINT_ORDER.index(n)] for n in ros_names])
    qd_ros = np.asarray([qd_policy[c.JOINT_ORDER.index(n)] for n in ros_names])
    idx = JointOrder(tuple(c.JOINT_ORDER)).remap(ros_names)
    gyro, cmd, last = rng.normal(0, 1, 3), rng.uniform(-0.5, 0.5, 3), rng.normal(0, 1, 12)
    axis, angle = rng.normal(size=3), float(rng.uniform(0, 0.7))
    n = np.asarray(axis) / np.linalg.norm(axis)
    quat_xyzw = (*(n * np.sin(angle / 2)), float(np.cos(angle / 2)))

    got = _builder().build(
        base_ang_vel=gyro,
        quat_xyzw=quat_xyzw,
        velocity_command=cmd,
        joint_pos=q_ros[idx],
        joint_vel=qd_ros[idx],
        last_action=last,
    )
    # Independent training-side definition: SI terms in contract order, then the scales.
    default = np.asarray([c.DEFAULT_JOINT_POS[j] for j in c.JOINT_ORDER])
    terms = {
        "base_ang_vel": gyro,
        "projected_gravity": _rot(axis, angle).T @ np.asarray([0.0, 0.0, -1.0]),
        "velocity_command": cmd,
        "joint_pos_rel": q_policy - default,
        "joint_vel": qd_policy,
        "last_action": last,
    }
    want = np.concatenate([terms[t] * TRAINING_V2_SCALES[t] for t in TRAINING_V2_TERM_ORDER])
    assert got.shape == (45,) and got.dtype == np.float32
    for name, sl in c.obs_slices().items():
        assert np.max(np.abs(got[sl] - want[sl])) <= 1e-6, name
    # And it is the shared sim-to-sim function, bit for bit.
    shared = build_actor_observation(
        gyro_body=gyro,
        quat_wxyz=quat_xyzw_to_wxyz(quat_xyzw),
        command=cmd,
        joint_pos=q_policy,
        joint_vel=qd_policy,
        last_action=last,
        obs_schema_version=c.OBS_SCHEMA_V2,
    )
    assert np.array_equal(got, shared)


def test_no_base_lin_vel_dims() -> None:
    assert "base_lin_vel" not in c.obs_slices()
    assert canonical_stand_observation(_builder()).shape == (45,)


def test_manifest_accepted() -> None:
    assert manifest_actor_schema(_manifest()) == c.OBS_SCHEMA_V2


def test_manifest_without_velocity_schema_keeps_the_pre_v2_path() -> None:
    assert manifest_actor_schema({"deploy": {"action_clip": 100.0}}) is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda o: o.update(schema_sha256="0" * 64),
        lambda o: o.update(schema_version="phoenix-velocity-obs/v9"),
        lambda o: o.update(actor_dim=48),
        lambda o: o.update(actor_terms=["base_lin_vel", *TRAINING_V2_TERM_ORDER]),
        lambda o: o.update(actor_terms=list(reversed(TRAINING_V2_TERM_ORDER))),
        lambda o: o["actor_scales"].update(joint_vel=1.0),
    ],
)
def test_manifest_disagreement_refuses(mutate) -> None:
    m = copy.deepcopy(_manifest())
    mutate(m["observation"])
    with pytest.raises(ManifestObservationError):
        manifest_actor_schema(m)


def test_builder_refuses_wrong_joint_order_or_pose() -> None:
    with pytest.raises(ManifestObservationError):
        ManifestObservationBuilder(
            c.OBS_SCHEMA_V2, list(reversed(c.JOINT_ORDER)), c.DEFAULT_JOINT_POS
        )
    pose = dict(c.DEFAULT_JOINT_POS, FL_hip_joint=0.2)
    with pytest.raises(ManifestObservationError):
        ManifestObservationBuilder(c.OBS_SCHEMA_V2, c.JOINT_ORDER, pose)


def test_reference_observations_are_deterministic() -> None:
    a = reference_observations(_builder(), n=20, seed=3)
    assert a.shape == (20, 45) and np.array_equal(
        a, reference_observations(_builder(), n=20, seed=3)
    )


# ------------------------------------------------------------------ iter2750 artifacts
ITER2750 = os.environ.get("PHOENIX_ITER2750_DIR")


@pytest.mark.skipif(not ITER2750, reason="PHOENIX_ITER2750_DIR not set (iter2750 artifacts absent)")
def test_node_policy_step_on_iter2750_onnx() -> None:
    """The node's per-tick path (builder -> ONNX 'obs' -> clip -> target), offline, closed loop."""
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
    root = Path(ITER2750)
    manifest = json.loads((root / "phoenix_manifest.json").read_text())
    assert manifest_actor_schema(manifest) == c.OBS_SCHEMA_V2
    builder = ManifestObservationBuilder(
        c.OBS_SCHEMA_V2, manifest["joint_order"], manifest["default_joint_pos"]
    )
    session = ort.InferenceSession(str(root / "policy.onnx"), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "obs"  # the node feeds {"obs": obs}
    assert session.get_inputs()[0].shape[-1] == builder.dim == 45
    clip = float(manifest["deploy"]["action_clip"])
    default = np.asarray([manifest["default_joint_pos"][j] for j in manifest["joint_order"]])
    rng = np.random.default_rng(0)
    last = np.zeros(12, dtype=np.float32)
    for _ in range(25):
        obs = builder.build(
            base_ang_vel=rng.normal(0, 0.1, 3),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
            velocity_command=np.zeros(3),
            joint_pos=default + rng.normal(0, 0.02, 12),
            joint_vel=rng.normal(0, 0.2, 12),
            last_action=last,
        ).reshape(1, -1)
        raw = session.run(["action"], {"obs": obs})[0][0]
        assert raw.shape == (12,) and np.all(np.isfinite(raw))
        action = np.clip(raw, -clip, clip).astype(np.float32)
        target = default + float(manifest["action"]["scale"]) * action
        assert target.shape == (12,) and np.all(np.isfinite(target))
        last = action
