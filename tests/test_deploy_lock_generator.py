"""Lock generator: checkpoint role, parity reference inputs, canonical bench; PATH_KEYS; stage titles."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from phoenix.sim2real import activation
from phoenix.sim2real import preflight_eval as pe
from phoenix.sim2real.deploy_contract import (
    artifact_paths,
    build_lock,
    canonical_bench_record,
    parity_input_record,
    verify_lock,
)

REPO = Path(__file__).resolve().parents[1]
WALK_CFG = REPO / "configs" / "sim2real" / "deploy_walk_iter2750_stand.yaml"


def _cfg(tmp_path, with_ckpt=True):
    cfg = yaml.safe_load(WALK_CFG.read_text())
    art = tmp_path / "art"
    art.mkdir()
    for name in ("policy.onnx", "policy.onnx.data", "policy.pt", "m.json", "model_1.pt"):
        (art / name).write_bytes(name.encode())
    cfg["policy"]["onnx_path"] = str(art / "policy.onnx")
    cfg["policy"]["torchscript_path"] = str(art / "policy.pt")
    cfg["policy"]["manifest_path"] = str(art / "m.json")
    if with_ckpt:
        cfg["policy"]["checkpoint_path"] = str(art / "model_1.pt")
    else:
        cfg["policy"].pop("checkpoint_path", None)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return cfg, path, art


def test_path_keys_pin_the_manifest_and_checkpoint() -> None:
    assert "manifest_path" in activation.PATH_KEYS
    assert "checkpoint_path" in activation.PATH_KEYS
    cfg = {"policy": {"onnx_path": "a.onnx", "manifest_path": "m.json", "checkpoint_path": "c.pt"}}
    req = activation.required_paths(cfg)
    assert req["policy.manifest_path"] == "m.json"
    text = 'policy:\n  onnx_path: "x/a.onnx"\n  manifest_path: "x/m.json"\n'
    out, pinned = activation.pin_config_text(text, "/b", present={"a.onnx", "m.json"})
    assert pinned == {"onnx_path": "/b/a.onnx", "manifest_path": "/b/m.json"}
    assert '"/b/m.json"' in out


def test_checkpoint_role_comes_from_checkpoint_path_else_latest(tmp_path) -> None:
    cfg, _, art = _cfg(tmp_path)
    assert artifact_paths(cfg)["checkpoint"] == art / "model_1.pt"
    cfg["policy"].pop("checkpoint_path")
    assert artifact_paths(cfg)["checkpoint"] == art / "latest.pt"


def test_lock_names_the_checkpoint_and_verifies(tmp_path) -> None:
    cfg, path, art = _cfg(tmp_path)
    lock = build_lock(cfg, path, name="t", repo_config_path="x.yaml", checkpoint=art / "model_1.pt")
    assert "checkpoint" in lock["artifacts"] and lock["checkpoint_source"].endswith(
        "art/model_1.pt"
    )
    roles = ("policy.onnx", "policy.onnx.data", "policy.pt", "checkpoint")
    problems, _ = verify_lock(lock, cfg, path, required_roles=roles)
    assert problems == []


def test_lock_refuses_a_checkpoint_the_config_does_not_reach(tmp_path) -> None:
    cfg, path, art = _cfg(tmp_path)
    other = art / "other.pt"
    other.write_bytes(b"x")
    with pytest.raises(ValueError, match="not the checkpoint the config reaches"):
        build_lock(cfg, path, name="t", repo_config_path="x.yaml", checkpoint=other)


def test_parity_input_record_pins_the_sha_and_validates_the_spec() -> None:
    rel = "configs/sim2real/parity/walk_iter2750_v2_synthetic_obs.npz"
    if not (REPO / rel).is_file():
        pytest.skip("synthetic reference batch not generated yet")
    rec = parity_input_record(f"{rel}:npz:obs")
    assert rec["kind"] == "npz" and rec["key"] == "obs" and len(rec["sha256"]) == 64
    with np.load(REPO / rel) as data:
        assert data["obs"].shape[-1] == 45
    for bad in (f"{rel}:npz", f"{rel}:parquet:k", f"{rel}:csv", "missing.npz:npz:obs"):
        with pytest.raises((ValueError, FileNotFoundError)):
            parity_input_record(bad)


ITER2750 = os.environ.get("PHOENIX_ITER2750_DIR")


@pytest.mark.skipif(not ITER2750, reason="PHOENIX_ITER2750_DIR not set")
def test_canonical_bench_on_iter2750_is_finite() -> None:
    pytest.importorskip("onnxruntime")
    cfg = yaml.safe_load(WALK_CFG.read_text())
    rec = canonical_bench_record(cfg)
    assert np.isfinite(rec["abs_action_inf"]) and rec["legacy_threshold_is_gate"] is False


def test_stage_titles_name_the_active_config_and_default_is_unchanged() -> None:
    assert pe.stage_title("F") == pe.STAGE_TITLES["F"] == "H25 cmd=0 stand, 2 s (motors LIVE)"
    assert pe.stage_title("H", "walk-iter2750-stand") == (
        "walk-iter2750-stand cmd=0 stand, 10 s x 3 attempts (motors LIVE)"
    )
    assert pe.stage_title("A", "x") == pe.STAGE_TITLES["A"]
    assert pe.STAND_AUTHORITY_S == {"F": 2.0, "G": 5.0, "H": 10.0} and pe.H_ATTEMPTS == 3
