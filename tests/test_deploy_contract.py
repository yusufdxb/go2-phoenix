"""Deploy contract (stand-only guard, walking blocked) and the artifact lock."""

from __future__ import annotations

import copy
import json
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


def _v2_manifest() -> dict:
    return {
        "observation": {
            "schema_version": dc.OBS_SCHEMA_V2,
            "actor_dim": 45,
            "actor_terms": ["base_ang_vel", "projected_gravity", "velocity_command"],
        }
    }


@pytest.mark.parametrize("path", DEPLOY_CONFIGS, ids=lambda p: p.name)
def test_every_shipped_deploy_config_passes_the_contract(path: Path) -> None:
    cfg = yaml.safe_load(path.read_text())
    manifest = None
    mp = (cfg.get("policy") or {}).get("manifest_path")
    if mp and not Path(mp).is_file():
        # The read-only archive is absent here (CI): same contract, with the manifest's
        # v2 declaration stood in for. On the workstation the real file is read.
        manifest = _v2_manifest()
    assert dc.validate_deploy_contract(cfg, manifest=manifest, evidence_sessions=[]) == []


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


def test_pre_v2_walking_with_odom_is_refused_without_hardware_evidence(tmp_path) -> None:
    cfg = _h25()
    cfg["safety"]["stand_only"] = False
    cfg["observation"]["base_lin_vel_source"] = "odom"
    problems = dc.validate_deploy_contract(cfg, evidence_sessions=[tmp_path])
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
        text, "/home/unitree/phoenix/stand-h25", {"policy.onnx", "policy.pt"}
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


# ------------------------------------------- v2 observation + evidence-gated walking
WALK = REPO_ROOT / "configs/sim2real/proposed/deploy_walk_iter2750.yaml"
WALK_STAND = REPO_ROOT / "configs/sim2real/deploy_walk_iter2750_stand.yaml"
WALK_STAND_LOCK = REPO_ROOT / "configs/sim2real/locks/deploy_walk_iter2750_stand.lock.yaml"


def _walk_fixture(tmp_path, *, onnx_bytes=b"graph"):
    """A walking config whose artifacts are small files in tmp_path, plus its manifest."""
    art = tmp_path / "art"
    art.mkdir(parents=True, exist_ok=True)
    (art / "policy.onnx").write_bytes(onnx_bytes)
    (art / "policy.onnx.data").write_bytes(b"weights")
    manifest = _v2_manifest()
    (art / "m.json").write_text(json.dumps(manifest))
    cfg = yaml.safe_load(WALK.read_text())
    cfg["policy"]["onnx_path"] = str(art / "policy.onnx")
    cfg["policy"]["manifest_path"] = str(art / "m.json")
    cfg["policy"]["torchscript_path"] = str(art / "policy.pt")
    return cfg, manifest


def _record(cfg, stage, **over):
    twin = dc.stand_twin(cfg)
    observed = dc.observed_artifact_hashes(cfg, "x")["artifacts"]
    rec = {
        "schema": "phoenix-preflight-stage/v1",
        "stage": stage,
        "verdict": "GO",
        "rehearsal": False,
        "motors": "live",
        "code_identity_problems": [],
        "lock_semantic_sha256": dc.semantic_config_sha256(twin),
        "lock_artifact_sha256": {
            r: observed[r]["sha256"] for r in ("policy.onnx", "policy.onnx.data", "policy.manifest")
        },
    }
    rec.update(over)
    return rec


def _session(tmp_path, name, records):
    d = tmp_path / "sessions" / name
    d.mkdir(parents=True, exist_ok=True)
    for stage, rec in records.items():
        (d / f"stage_{stage}.json").write_text(json.dumps(rec))
    return d


def test_shipped_walk_config_is_refused_today_and_names_the_missing_evidence(tmp_path) -> None:
    problems = dc.validate_deploy_contract(
        yaml.safe_load(WALK.read_text()), manifest=_v2_manifest(), evidence_sessions=[tmp_path]
    )
    text = " ".join(problems)
    assert "walking deploy configs are blocked" in text
    # the draft's artifacts live in a read-only archive that CI lacks: either the missing
    # files or the missing F/G records are named, never silence
    assert "stage F" in text or "cannot hash" in text


def test_walking_refused_without_evidence(tmp_path) -> None:
    cfg, manifest = _walk_fixture(tmp_path)
    problems = dc.validate_deploy_contract(cfg, manifest=manifest, evidence_sessions=[])
    text = " ".join(problems)
    assert (
        "stage F has no GO hardware record" in text and "stage G has no GO hardware record" in text
    )


def test_walking_refused_with_evidence_for_a_different_artifact_hash(tmp_path) -> None:
    other, _ = _walk_fixture(tmp_path / "other", onnx_bytes=b"another graph")
    cfg, manifest = _walk_fixture(tmp_path)
    s = _session(tmp_path, "s", {"F": _record(other, "F"), "G": _record(other, "G")})
    problems = dc.validate_deploy_contract(cfg, manifest=manifest, evidence_sessions=[s])
    assert any("policy.onnx hash differs" in p for p in problems)


def test_walking_refused_with_evidence_of_only_stage_f(tmp_path) -> None:
    cfg, manifest = _walk_fixture(tmp_path)
    s = _session(tmp_path, "s", {"F": _record(cfg, "F")})
    problems = dc.validate_deploy_contract(cfg, manifest=manifest, evidence_sessions=[s])
    text = " ".join(problems)
    assert "stage G has no GO hardware record" in text and "stage F has no GO" not in text


@pytest.mark.parametrize(
    "over,needle",
    [
        ({"verdict": "NO-GO"}, "verdict"),
        ({"rehearsal": True}, "rehearsal"),
        ({"motors": "off"}, "motors"),
        ({"lock_semantic_sha256": "0" * 64}, "stand-phase twin"),
        ({"lock_artifact_sha256": {}}, "hash differs"),
    ],
)
def test_walking_evidence_must_be_live_go_and_for_the_twin(tmp_path, over, needle) -> None:
    cfg, manifest = _walk_fixture(tmp_path)
    s = _session(tmp_path, "s", {"F": _record(cfg, "F", **over), "G": _record(cfg, "G")})
    problems = dc.validate_deploy_contract(cfg, manifest=manifest, evidence_sessions=[s])
    assert any(needle in p for p in problems)


def test_walking_allowed_with_valid_f_and_g_evidence(tmp_path) -> None:
    cfg, manifest = _walk_fixture(tmp_path)
    s = _session(tmp_path, "s", {"F": _record(cfg, "F"), "G": _record(cfg, "G")})
    assert dc.validate_deploy_contract(cfg, manifest=manifest, evidence_sessions=[s]) == []
    # F and G may come from different sessions
    f_only = _session(tmp_path, "sf", {"F": _record(cfg, "F")})
    g_only = _session(tmp_path, "sg", {"G": _record(cfg, "G")})
    assert dc.walking_evidence_problems(cfg, sessions=[f_only, g_only]) == []


def test_walking_evidence_is_per_artifact_not_per_config_edit(tmp_path) -> None:
    cfg, manifest = _walk_fixture(tmp_path)
    s = _session(tmp_path, "s", {"F": _record(cfg, "F"), "G": _record(cfg, "G")})
    edited = json.loads(json.dumps(cfg))
    edited["safety"]["max_runtime_s"] = 999  # not the config that was stood
    assert dc.walking_evidence_problems(edited, sessions=[s])


def test_v2_manifest_replaces_the_base_lin_vel_zeros_rule(tmp_path) -> None:
    cfg, manifest = _walk_fixture(tmp_path)
    stand = dc.stand_twin(cfg)
    assert dc.validate_deploy_contract(stand, manifest=manifest) == []
    # walking + zeros is NOT refused by the pre-v2 rule for a v2 policy (evidence gates it)
    problems = dc.validate_deploy_contract(cfg, manifest=manifest, evidence_sessions=[])
    assert not any("base_lin_vel_source=zeros" in p for p in problems)
    # v2 requires the config to say v2, the manifest layout to be v2, no odom source
    bad = json.loads(json.dumps(stand))
    del bad["observation"]["schema_version"]
    assert any("must say v2 too" in p for p in dc.validate_deploy_contract(bad, manifest=manifest))
    bad = json.loads(json.dumps(stand))
    bad["observation"]["base_lin_vel_source"] = "odom"
    assert any("no base_lin_vel" in p for p in dc.validate_deploy_contract(bad, manifest=manifest))
    wrong_dim = json.loads(json.dumps(manifest))
    wrong_dim["observation"]["actor_dim"] = 48
    assert any(
        "not the v2 layout" in p for p in dc.validate_deploy_contract(stand, manifest=wrong_dim)
    )
    # a v2 declaration with an unreadable manifest fails closed
    stand["policy"]["manifest_path"] = str(tmp_path / "missing.json")
    assert any("cannot be read" in p for p in dc.validate_deploy_contract(stand))


def test_walk_stand_config_is_the_walk_config_with_stand_only() -> None:
    walk, stand = yaml.safe_load(WALK.read_text()), yaml.safe_load(WALK_STAND.read_text())
    assert dc.is_stand_only(stand) and not dc.is_stand_only(walk)
    assert dc.semantic_config_sha256(dc.stand_twin(walk)) == dc.semantic_config_sha256(stand)
    lock = dc.load_lock(WALK_STAND_LOCK)
    assert lock["deploy_config"]["semantic_sha256"] == dc.semantic_config_sha256(stand)
    assert set(lock["artifacts"]) >= {"policy.onnx", "policy.onnx.data", "policy.manifest"}


def test_walk_stand_lock_matches_real_artifacts_when_present() -> None:
    stand = yaml.safe_load(WALK_STAND.read_text())
    if not Path(stand["policy"]["onnx_path"]).is_file():
        pytest.skip("iter2750 archive artifacts are absent here")
    problems, _ = dc.verify_lock(
        dc.load_lock(WALK_STAND_LOCK),
        stand,
        WALK_STAND,
        required_roles=("policy.onnx", "policy.onnx.data", "policy.manifest", "policy.pt"),
    )
    assert problems == []


def test_lock_generator_round_trips(tmp_path) -> None:
    cfg, _ = _walk_fixture(tmp_path)
    stand = dc.stand_twin(cfg)
    path = tmp_path / "stand.yaml"
    path.write_text(yaml.safe_dump(stand))
    out = tmp_path / "stand.lock.yaml"
    assert dc.main(["lock", "--config", str(path), "--out", str(out), "--name", "t"]) == 0
    assert dc.verify_lock(dc.load_lock(out), stand, path)[0] == []
