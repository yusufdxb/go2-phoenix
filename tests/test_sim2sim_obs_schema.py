"""The sim2sim deploy spec builds observations with the manifest's schema."""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.sim2sim.deploy_spec import DeploySpecError, spec_from_phoenix_manifest
from phoenix.velocity import contract as c
from phoenix.velocity.observation import build_actor_observation

ENVELOPE = {"lin_vel_x": 0.5, "lin_vel_y": 0.3, "ang_vel_z": 0.8}
DEFAULT_Q = np.asarray([c.DEFAULT_JOINT_POS[j] for j in c.JOINT_ORDER])
STATE = dict(
    gyro_body=np.array([1.0, -2.0, 3.0]),
    quat_wxyz=np.array([1.0, 0, 0, 0]),
    command=[0.5, -0.3, 0.8],
    joint_pos=DEFAULT_Q + 0.01,
    joint_vel=np.arange(12) * -2.0,
    last_action=np.linspace(-1, 1, 12),
)


def _manifest(version: str) -> dict:
    return c.build_manifest(
        checkpoint_sha256="a" * 64,
        commands=c.CommandRanges((-0.3, 0.6), (-0.3, 0.3), (-1.0, 1.0), 0.1),
        git_sha="b" * 40,
        git_dirty=False,
        seed=1,
        task="t",
        simulator="isaaclab",
        reward_scales={},
        domain_randomization={},
        curriculum={},
        obs_schema_version=version,
        deploy={"kp": 25.0, "kd": 0.5, "action_clip": 100.0},
    )


@pytest.mark.parametrize("version", [c.OBS_SCHEMA_V1, c.OBS_SCHEMA_V2])
def test_deploy_spec_uses_the_manifest_schema(version: str) -> None:
    spec = spec_from_phoenix_manifest(_manifest(version), name="x", required_envelope=ENVELOPE)
    assert spec.manifest_problems == ()
    assert spec.obs_schema == version and spec.summary()["obs_schema"] == version
    got = spec.build_obs(**STATE)
    assert np.array_equal(got, build_actor_observation(**STATE, obs_schema_version=version))


def test_v2_differs_from_v1_only_by_the_two_scaled_terms() -> None:
    v1 = spec_from_phoenix_manifest(_manifest(c.OBS_SCHEMA_V1), name="a", required_envelope=ENVELOPE)
    v2 = spec_from_phoenix_manifest(_manifest(c.OBS_SCHEMA_V2), name="b", required_envelope=ENVELOPE)
    a, b = v1.build_obs(**STATE), v2.build_obs(**STATE)
    sl = c.obs_slices()
    assert np.allclose(b[sl["base_ang_vel"]], 0.2 * a[sl["base_ang_vel"]])
    assert np.allclose(b[sl["joint_vel"]], 0.05 * a[sl["joint_vel"]])


def test_unknown_schema_is_reported_and_refused() -> None:
    m = _manifest(c.OBS_SCHEMA_V2)
    m["observation"]["schema_version"] = "phoenix-velocity-obs/v9"
    spec = spec_from_phoenix_manifest(m, name="x", required_envelope=ENVELOPE)
    assert spec.manifest_problems and spec.obs_schema is None
    with pytest.raises(DeploySpecError):
        spec.build_obs(**STATE)
