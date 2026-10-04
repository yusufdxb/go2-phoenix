"""G2: the deploy code path's contract equals training, on the REAL iter2750 manifest.

Set ``PHOENIX_ITER2750_MANIFEST`` to the candidate's ``phoenix_manifest.json``
(e.g. ~/Archive/phoenix-deploy-candidates-2026-10-03/eval/iter2750_gate/).
Skipped when unset or absent so CI stays green without the checkpoint archive.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from phoenix.sim2real.controller_fsm import fsm_params_from_manifest
from phoenix.sim2real.deploy_manifest import cfg_manifest_mismatches, load_deploy_manifest
from phoenix.sim2real.observation import JointOrder, ObservationBuilder
from phoenix.velocity import contract as c

_ENV = "PHOENIX_ITER2750_MANIFEST"
_PATH = os.environ.get(_ENV, "")
pytestmark = pytest.mark.skipif(
    not (_PATH and Path(_PATH).is_file()),
    reason=f"{_ENV} unset or not a file (real iter2750 phoenix_manifest.json required)",
)

_DEPLOY_YAML = Path(__file__).resolve().parents[1] / "configs" / "sim2real" / "deploy.yaml"
_EXPECTED_SHA = "53a41f30142f6afdec92428314ae5092886d81e5ac9a59936f12c634fad7805a"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(Path(_PATH).read_text())


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(_DEPLOY_YAML.read_text())


def test_observation_contract_matches_training_code(manifest) -> None:
    obs = manifest["observation"]
    assert obs["schema_version"] == c.OBS_SCHEMA_V2
    assert obs["schema_sha256"] == _EXPECTED_SHA == c.obs_schema_sha256(c.OBS_SCHEMA_V2)
    assert c.manifest_obs_schema(manifest) == c.OBS_SCHEMA_V2
    assert obs["actor_dim"] == c.ACTOR_OBS_DIM == 45
    assert "base_lin_vel" not in obs["actor_terms"]
    assert obs["actor_terms"] == [t[0] for t in c.ACTOR_OBS_TERMS]
    assert obs["actor_scales"] == c.actor_obs_scales(c.OBS_SCHEMA_V2)
    assert load_deploy_manifest(_PATH).obs_schema_version == c.OBS_SCHEMA_V2


def test_action_and_rates_match_training_constants(manifest) -> None:
    assert manifest["action"]["scale"] == c.ACTION_SCALE == 0.25
    assert manifest["action"]["dim"] == c.ACTION_DIM == 12
    assert manifest["control_hz"] == c.CONTROL_HZ == 50
    assert manifest["physics_hz"] == c.PHYSICS_HZ == 200


def test_deploy_block_drives_the_fsm_params(manifest) -> None:
    d = load_deploy_manifest(_PATH)
    assert (d.action_clip, d.kp, d.kd) == (100.0, 25.0, 0.5)
    p = fsm_params_from_manifest(_PATH)
    assert (p.action_clip, p.policy_kp, p.policy_kd) == (100.0, 25.0, 0.5)


def test_joint_order_and_default_pose_match_training(manifest, cfg) -> None:
    assert tuple(manifest["joint_order"]) == c.JOINT_ORDER
    assert manifest["default_joint_pos"] == c.DEFAULT_JOINT_POS
    # Permutation: a shuffled /joint_states must remap into the manifest order.
    names = list(manifest["joint_order"])
    shuffled = names[::-1]
    idx = JointOrder(tuple(names)).remap(shuffled)
    assert [shuffled[i] for i in idx] == names
    assert idx.tolist() == list(range(11, -1, -1))


def test_deploy_yaml_agrees_with_manifest(manifest, cfg) -> None:
    # The node takes joint order, default pose, action scale and rate from the YAML,
    # not the manifest: the cross-check is what ties the two together.
    assert cfg_manifest_mismatches(cfg, _PATH) == []
    ob = ObservationBuilder(
        JointOrder(tuple(cfg["joint_order"])), cfg["control"]["default_joint_pos"]
    )
    q = np.array([manifest["default_joint_pos"][n] for n in manifest["joint_order"]], np.float32)
    assert np.allclose(q, [cfg["control"]["default_joint_pos"][n] for n in ob.joint_order.names])


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda c_: c_["control"].__setitem__("action_scale", 0.5), "action_scale"),
        (lambda c_: c_["control"].__setitem__("rate_hz", 100), "rate_hz"),
        (
            lambda c_: c_["control"]["default_joint_pos"].__setitem__("FL_hip_joint", 0.0),
            "FL_hip_joint",
        ),
        (
            lambda c_: c_.__setitem__("joint_order", list(reversed(c_["joint_order"]))),
            "joint_order",
        ),
    ],
)
def test_cross_check_detects_drift(cfg, mutate, needle) -> None:
    import copy

    bad = copy.deepcopy(cfg)
    mutate(bad)
    problems = cfg_manifest_mismatches(bad, _PATH)
    assert problems and any(needle in p for p in problems)
