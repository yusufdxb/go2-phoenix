"""Tests for the deploy-manifest reader: deploy.action_clip must equal the
training clip, and is refused (not defaulted) when the manifest omits it."""

from __future__ import annotations

import json

import pytest

from phoenix.sim2real.deploy_manifest import ManifestError, load_deploy_manifest


def _write(tmp_path, data, name="manifest.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_loads_action_clip_kp_kd_action_scale(tmp_path) -> None:
    path = _write(
        tmp_path,
        {"deploy": {"action_clip": 100.0, "kp": 25.0, "kd": 0.5, "action_scale": 0.25}},
    )
    m = load_deploy_manifest(path)
    assert m.action_clip == 100.0
    assert m.kp == 25.0
    assert m.kd == 0.5
    assert m.action_scale == 0.25
    assert m.path == path


def test_action_clip_only_kp_kd_scale_optional(tmp_path) -> None:
    path = _write(tmp_path, {"deploy": {"action_clip": 100.0}})
    m = load_deploy_manifest(path)
    assert m.action_clip == 100.0
    assert m.kp is None and m.kd is None and m.action_scale is None


def test_per_joint_action_scale_list(tmp_path) -> None:
    path = _write(tmp_path, {"deploy": {"action_clip": 100.0, "action_scale": [0.25] * 12}})
    m = load_deploy_manifest(path)
    assert m.action_scale == tuple([0.25] * 12)


def test_missing_deploy_block_refused(tmp_path) -> None:
    path = _write(tmp_path, {"action": {"training_clip": 1.0}})
    with pytest.raises(ManifestError, match="no 'deploy' object"):
        load_deploy_manifest(path)


def test_missing_action_clip_refused(tmp_path) -> None:
    path = _write(tmp_path, {"deploy": {"kp": 25.0}})
    with pytest.raises(ManifestError, match="action_clip is missing"):
        load_deploy_manifest(path)


def test_null_action_clip_refused(tmp_path) -> None:
    path = _write(tmp_path, {"deploy": {"action_clip": None}})
    with pytest.raises(ManifestError, match="action_clip is missing"):
        load_deploy_manifest(path)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_non_positive_or_non_finite_action_clip_refused(tmp_path, bad) -> None:
    path = _write(tmp_path, {"deploy": {"action_clip": bad}})
    with pytest.raises(ManifestError, match="positive and finite"):
        load_deploy_manifest(path)


def test_non_numeric_action_clip_refused(tmp_path) -> None:
    path = _write(tmp_path, {"deploy": {"action_clip": "one hundred"}})
    with pytest.raises(ManifestError, match="not a number"):
        load_deploy_manifest(path)


def test_missing_file_refused(tmp_path) -> None:
    with pytest.raises(ManifestError, match="could not be read"):
        load_deploy_manifest(tmp_path / "does_not_exist.json")


def test_not_json_refused(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("not json {")
    with pytest.raises(ManifestError, match="not valid JSON"):
        load_deploy_manifest(path)
