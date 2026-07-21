"""Tests for the executable controller bundle manifest."""

from __future__ import annotations

import json

import pytest

from phoenix.reliability.bundle import (
    BundleManifest,
    build_manifest,
    file_sha256,
    value_sha256,
    verify_bundle,
)

CONTROLLER = {
    "control_dt_s": 0.02,
    "action_scale": 0.25,
    "max_delta_per_step_rad": 0.175,
    "default_joint_pos": {"FL_hip_joint": 0.0, "FL_thigh_joint": 0.8},
    "joint_order": ["FL_hip_joint", "FL_thigh_joint"],
}


@pytest.fixture
def files(tmp_path):
    policy = tmp_path / "policy.onnx"
    artifact = tmp_path / "shield.npz"
    policy.write_bytes(b"policy-bytes")
    artifact.write_bytes(b"artifact-bytes")
    return {"policy_onnx": policy, "shield_artifact": artifact}


def test_file_sha256_is_content_addressed(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert file_sha256(a) == file_sha256(b)
    b.write_bytes(b"different")
    assert file_sha256(a) != file_sha256(b)


def test_value_sha256_is_key_order_independent():
    assert value_sha256({"a": 1, "b": 2}) == value_sha256({"b": 2, "a": 1})
    assert value_sha256({"a": 1}) != value_sha256({"a": 2})


def test_build_manifest_requires_every_controller_key(files):
    incomplete = dict(CONTROLLER)
    del incomplete["action_scale"]
    with pytest.raises(ValueError, match="missing required keys"):
        build_manifest(files=files, controller=incomplete)


def test_build_manifest_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="policy_onnx"):
        build_manifest(files={"policy_onnx": tmp_path / "nope"}, controller=CONTROLLER)


def test_verify_passes_on_unchanged_bundle(files):
    manifest = build_manifest(files=files, controller=CONTROLLER)
    verify_bundle(manifest, files=files, controller=CONTROLLER, artifact_control_dt_s=0.02)


def test_verify_detects_changed_policy(files):
    manifest = build_manifest(files=files, controller=CONTROLLER)
    files["policy_onnx"].write_bytes(b"a-different-policy")
    with pytest.raises(ValueError, match="policy_onnx"):
        verify_bundle(manifest, files=files, controller=CONTROLLER)


@pytest.mark.parametrize(
    "key,value",
    [
        ("action_scale", 0.5),
        ("max_delta_per_step_rad", 0.3),
        ("default_joint_pos", {"FL_hip_joint": 0.1, "FL_thigh_joint": 0.8}),
        ("joint_order", ["FL_thigh_joint", "FL_hip_joint"]),
    ],
)
def test_verify_detects_changed_controller_value(files, key, value):
    """Each of these silently changes what the robot does against the same artifact."""
    manifest = build_manifest(files=files, controller=CONTROLLER)
    changed = dict(CONTROLLER)
    changed[key] = value
    with pytest.raises(ValueError, match=key):
        verify_bundle(manifest, files=files, controller=changed)


def test_verify_rejects_wrong_control_rate(files):
    """Arming/persistence/ramps are counted in ticks, so the rate must match exactly."""
    manifest = build_manifest(files=files, controller={**CONTROLLER, "control_dt_s": 0.01})
    with pytest.raises(ValueError, match="counted in ticks"):
        verify_bundle(
            manifest,
            files=files,
            controller={**CONTROLLER, "control_dt_s": 0.01},
            artifact_control_dt_s=0.02,
        )


def test_verify_reports_every_problem_at_once(files):
    manifest = build_manifest(files=files, controller=CONTROLLER)
    files["shield_artifact"].write_bytes(b"changed")
    with pytest.raises(ValueError) as excinfo:
        verify_bundle(
            manifest,
            files=files,
            controller={**CONTROLLER, "action_scale": 0.9},
            artifact_control_dt_s=0.01,
        )
    message = str(excinfo.value)
    assert "shield_artifact" in message
    assert "action_scale" in message
    assert "counted in ticks" in message


def test_verify_can_reject_a_dirty_tree(files):
    manifest = BundleManifest(
        files={},
        controller=CONTROLLER,
        code_commit="abc123",
        code_dirty=True,
    )
    with pytest.raises(ValueError, match="dirty working tree"):
        verify_bundle(manifest, files={}, controller=CONTROLLER, allow_dirty=False)


def test_manifest_roundtrip_and_bundle_id(tmp_path, files):
    manifest = build_manifest(files=files, controller=CONTROLLER, note="pilot")
    path = manifest.write(tmp_path / "bundle.json")
    loaded = BundleManifest.read(path)
    assert loaded.files == manifest.files
    assert loaded.controller == manifest.controller
    assert loaded.bundle_id == manifest.bundle_id
    assert loaded.note == "pilot"


def test_manifest_rejects_unknown_version(tmp_path, files):
    manifest = build_manifest(files=files, controller=CONTROLLER)
    path = manifest.write(tmp_path / "bundle.json")
    payload = json.loads(path.read_text())
    payload["manifest_version"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported manifest version"):
        BundleManifest.read(path)


def test_bundle_id_changes_with_the_controller(files):
    a = build_manifest(files=files, controller=CONTROLLER)
    b = build_manifest(files=files, controller={**CONTROLLER, "action_scale": 0.5})
    assert a.bundle_id != b.bundle_id
