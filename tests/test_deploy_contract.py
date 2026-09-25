"""Deploy contract (stand-only guard, walking blocked) and the artifact lock."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from phoenix.sim2real import deploy_contract as dc
from phoenix.sim2real.activation import file_sha256, pin_config_text

REPO_ROOT = Path(__file__).resolve().parent.parent
H25 = REPO_ROOT / "configs/sim2real/deploy_stand_h25.yaml"
H25_LOCK = REPO_ROOT / "configs/sim2real/locks/deploy_stand_h25.lock.yaml"
DEPLOY_CONFIGS = sorted((REPO_ROOT / "configs/sim2real").glob("deploy*.yaml"))


def _h25() -> dict:
    return yaml.safe_load(H25.read_text())


# ------------------------------------------------------------------ contract
def test_h25_is_a_valid_stand_only_config() -> None:
    cfg = _h25()
    assert dc.validate_deploy_contract(cfg) == []
    assert dc.is_stand_only(cfg)
    assert cfg["observation"]["base_lin_vel_source"] == "zeros"


@pytest.mark.parametrize("path", DEPLOY_CONFIGS, ids=lambda p: p.name)
def test_every_shipped_deploy_config_passes_the_contract(path: Path) -> None:
    assert dc.validate_deploy_contract(yaml.safe_load(path.read_text())) == []


def test_zeros_source_without_stand_only_is_refused() -> None:
    cfg = _h25()
    cfg["safety"]["stand_only"] = False
    problems = dc.validate_deploy_contract(cfg)
    assert any(
        "walking is blocked while observation.base_lin_vel_source=zeros" in p for p in problems
    )


def test_stand_only_must_be_explicit() -> None:
    cfg = _h25()
    del cfg["safety"]["stand_only"]
    assert any(
        "safety.stand_only must be set explicitly" in p for p in dc.validate_deploy_contract(cfg)
    )
    cfg["safety"]["stand_only"] = "yes"
    assert any(
        "safety.stand_only must be set explicitly" in p for p in dc.validate_deploy_contract(cfg)
    )


def test_walking_with_odom_is_still_refused_while_walking_is_disabled() -> None:
    assert dc.WALKING_ENABLED is False
    cfg = _h25()
    cfg["safety"]["stand_only"] = False
    cfg["observation"]["base_lin_vel_source"] = "odom"
    problems = dc.validate_deploy_contract(cfg)
    assert any("walking deploy configs are blocked" in p for p in problems)


def test_mode_switch_cannot_run_in_a_stand_only_config() -> None:
    cfg = _h25()
    cfg["policy"]["mode_switch"] = {"enabled": True}
    assert any("mode_switch" in p for p in dc.validate_deploy_contract(cfg))


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (
            lambda c: c["control"]["default_joint_pos"].__setitem__("RR_hip_joint", 0.0),
            "RR_hip_joint",
        ),
        (lambda c: c["joint_order"].reverse(), "joint_order"),
        (lambda c: c["control"].__setitem__("rate_hz", 100), "rate_hz"),
        (lambda c: c["safety"].pop("sensor_timeout_s"), "sensor_timeout_s"),
        (lambda c: c["safety"].__setitem__("estop_timeout_s", 0), "estop_timeout_s"),
        (lambda c: c["safety"].__setitem__("attitude_intervention_rad", 0.5), "25 degree"),
        (lambda c: c["observation"].pop("base_lin_vel_source"), "base_lin_vel_source"),
    ],
)
def test_contract_catches_each_unsafe_edit(mutate, needle) -> None:
    cfg = _h25()
    mutate(cfg)
    assert any(needle in p for p in dc.validate_deploy_contract(cfg))


# ------------------------------------------------------------ semantic hash
def test_semantic_hash_survives_activation_pinning_and_comments() -> None:
    text = H25.read_text()
    pinned_text, pinned = pin_config_text(
        text, "/robot/phoenix/stand-h25", {"policy.onnx", "policy.pt"}
    )
    assert set(pinned) == {"onnx_path", "torchscript_path"}
    stripped = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    base = dc.semantic_config_sha256(yaml.safe_load(text))
    assert dc.semantic_config_sha256(yaml.safe_load(pinned_text)) == base
    assert dc.semantic_config_sha256(yaml.safe_load(stripped)) == base


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["safety"].__setitem__("sensor_timeout_s", 0.3),
        lambda c: c["control"]["default_joint_pos"].__setitem__("FL_calf_joint", -1.4),
        lambda c: c["policy"].__setitem__("onnx_path", "checkpoints/other/policy_v2.onnx"),
        lambda c: c["safety"].__setitem__("stand_only", False),
    ],
)
def test_semantic_hash_changes_on_meaningful_edits(mutate) -> None:
    cfg = _h25()
    edited = copy.deepcopy(cfg)
    mutate(edited)
    assert dc.semantic_config_sha256(edited) != dc.semantic_config_sha256(cfg)


# ---------------------------------------------------------------------- lock
def test_h25_lock_matches_the_current_config() -> None:
    """Editing the H25 config without updating its lock fails here."""
    lock = dc.load_lock(H25_LOCK)
    assert lock["deploy_config"]["semantic_sha256"] == dc.semantic_config_sha256(_h25())
    assert lock["stand_only"] is True
    assert set(lock["artifacts"]) == {"policy.onnx", "policy.onnx.data", "policy.pt", "checkpoint"}
    assert lock["canonical_bench"]["legacy_threshold_is_gate"] is False


def test_h25_lock_matches_real_artifacts_when_present(monkeypatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    cfg = _h25()
    if not Path(cfg["policy"]["onnx_path"]).is_file():
        pytest.skip("H25 checkpoint artifacts are gitignored and absent here")
    lock = dc.load_lock(H25_LOCK)
    problems, observed = dc.verify_lock(
        lock,
        cfg,
        H25.relative_to(REPO_ROOT),
        required_roles=("policy.onnx", "policy.onnx.data", "policy.pt", "checkpoint"),
    )
    assert problems == []
    assert (
        observed["artifacts"]["policy.onnx.data"]["sha256"]
        == lock["artifacts"]["policy.onnx.data"]["sha256"]
    )


def _fake_bundle(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    for name, payload in [
        ("policy.onnx", b"graph"),
        ("policy.onnx.data", b"weights"),
        ("policy.pt", b"torchscript"),
        ("latest.pt", b"checkpoint"),
    ]:
        (ckpt / name).write_bytes(payload)
    cfg = _h25()
    cfg["policy"]["onnx_path"] = str(ckpt / "policy.onnx")
    cfg["policy"]["torchscript_path"] = str(ckpt / "policy.pt")
    cfg_path = tmp_path / "deploy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    observed = dc.observed_artifact_hashes(cfg, cfg_path)
    lock = {
        "schema": dc.LOCK_SCHEMA,
        "deploy_config": {"semantic_sha256": observed["deploy_config"]["semantic_sha256"]},
        "artifacts": {role: {"sha256": e["sha256"]} for role, e in observed["artifacts"].items()},
    }
    return cfg, cfg_path, lock, ckpt


def test_verify_lock_passes_on_matching_files(tmp_path) -> None:
    cfg, cfg_path, lock, _ = _fake_bundle(tmp_path)
    problems, _ = dc.verify_lock(lock, cfg, cfg_path)
    assert problems == []


def test_verify_lock_catches_a_swapped_external_data_sidecar(tmp_path) -> None:
    cfg, cfg_path, lock, ckpt = _fake_bundle(tmp_path)
    (ckpt / "policy.onnx.data").write_bytes(b"different weights")
    problems, _ = dc.verify_lock(lock, cfg, cfg_path)
    assert any("policy.onnx.data" in p for p in problems)


def test_verify_lock_catches_a_missing_required_artifact(tmp_path) -> None:
    cfg, cfg_path, lock, ckpt = _fake_bundle(tmp_path)
    (ckpt / "policy.onnx.data").unlink()
    problems, _ = dc.verify_lock(lock, cfg, cfg_path)
    assert any("required artifact policy.onnx.data is missing" in p for p in problems)


def test_verify_lock_catches_a_present_but_wrong_optional_artifact(tmp_path) -> None:
    cfg, cfg_path, lock, ckpt = _fake_bundle(tmp_path)
    (ckpt / "latest.pt").write_bytes(b"another checkpoint")
    problems, _ = dc.verify_lock(lock, cfg, cfg_path)
    assert any("checkpoint" in p for p in problems)


def test_verify_lock_catches_a_config_edit(tmp_path) -> None:
    cfg, cfg_path, lock, _ = _fake_bundle(tmp_path)
    cfg["safety"]["max_runtime_s"] = 999
    problems, _ = dc.verify_lock(lock, cfg, cfg_path)
    assert any("semantic sha256" in p for p in problems)


def test_load_lock_rejects_malformed(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema: phoenix-deploy-lock/v1\nartifacts: {policy.onnx: {sha256: nope}}\n")
    with pytest.raises(ValueError):
        dc.load_lock(bad)
    bad.write_text("schema: something-else\n")
    with pytest.raises(ValueError):
        dc.load_lock(bad)


def test_file_sha256_matches_hashlib(tmp_path) -> None:
    import hashlib

    p = tmp_path / "x.bin"
    p.write_bytes(b"abc" * 1000)
    assert file_sha256(p) == hashlib.sha256(b"abc" * 1000).hexdigest()
