"""Tests for ``phoenix.sim2real.activation``.

The property under test is not "the bytes arrived" (``sha256sum -c`` already
proves that) but "the config the node reads names the bytes that arrived".
Every test here is a way that can be false while a transfer check reports
success, which is the exact failure this module exists to end.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from phoenix.sim2real.activation import (
    SUMS_NAME,
    find_config,
    main,
    parse_sha256sums,
    pin_config_text,
    required_paths,
    unpinned_roles,
    verify_activation,
)

STAGING_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stage_payload_bundle.sh"

CONFIG_TEMPLATE = """\
# phoenix-stand-h25-lat-noise, parity max_abs_diff 3.1e-07 (supersedes stand-v3)
policy:
  onnx_path: "checkpoints/phoenix-stand-h25-lat-noise/policy.onnx"  # external data
  torchscript_path: "checkpoints/phoenix-stand-h25-lat-noise/policy.pt"
  obs_pad_zeros: 0
reliability:
  enabled: false
"""


def _write_bundle(tmp_path: Path, config_text: str = CONFIG_TEMPLATE) -> Path:
    """A staged bundle: the two policy files, a config, and a real manifest."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "policy.onnx").write_bytes(b"onnx-graph-bytes")
    (bundle / "policy.pt").write_bytes(b"torchscript-bytes")
    (bundle / "deploy_stand_h25.yaml").write_text(config_text)
    return bundle


def _write_sums(bundle: Path) -> dict[str, str]:
    lines = []
    for path in sorted(bundle.iterdir()):
        if path.is_file() and path.name != SUMS_NAME:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  ./{path.name}")
    (bundle / SUMS_NAME).write_text("\n".join(lines) + "\n")
    return parse_sha256sums((bundle / SUMS_NAME).read_text())


def _pin(bundle: Path) -> int:
    return main(["pin", "--bundle", str(bundle)])


def _verify(bundle: Path) -> int:
    return main(["verify", "--bundle", str(bundle)])


# ---------------- (a) a relative path surviving pinning ----------------------


def test_pin_refuses_when_a_path_key_is_not_rewritten(tmp_path, capsys) -> None:
    """A flow mapping is valid YAML the line-based rewriter cannot touch.

    Without the post-condition, pin exits 0, the path stays relative, and the
    node resolves it against the payload's CWD onto an older checkpoint.
    """
    flow = (
        "policy: {onnx_path: policy.onnx, torchscript_path: policy.pt}\n"
        "reliability:\n  enabled: false\n"
    )
    bundle = _write_bundle(tmp_path, flow)

    assert _pin(bundle) == 1
    err = capsys.readouterr().err
    assert "REFUSING TO PIN" in err
    assert "policy.onnx_path" in err
    # Fail closed: the config on disk is untouched, nothing was half-activated.
    assert (bundle / "deploy_stand_h25.yaml").read_text() == flow


def test_unpinned_roles_flags_relative_and_out_of_bundle_paths(tmp_path) -> None:
    target = tmp_path / "bundle"
    cfg = {
        "policy": {
            "onnx_path": "checkpoints/old/policy.onnx",
            "torchscript_path": str(tmp_path / "elsewhere" / "policy.pt"),
        }
    }
    problems = unpinned_roles(cfg, target)
    assert len(problems) == 2
    assert any("onnx_path" in p for p in problems)
    assert any("torchscript_path" in p for p in problems)


def test_verify_rejects_a_relative_path(tmp_path, capsys) -> None:
    """The staged-but-never-pinned bundle: bytes fine, wrong policy runs."""
    bundle = _write_bundle(tmp_path)
    _write_sums(bundle)

    assert _verify(bundle) == 1
    err = capsys.readouterr().err
    assert "ACTIVATION UNVERIFIED" in err
    assert "relative path" in err


def test_pin_then_verify_round_trips(tmp_path, capsys) -> None:
    bundle = _write_bundle(tmp_path)
    assert _pin(bundle) == 0
    _write_sums(bundle)  # manifest is taken AFTER pinning, as the script does
    assert _verify(bundle) == 0

    out = capsys.readouterr().out
    assert "ACTIVATION VERIFIED" in out
    text = (bundle / "deploy_stand_h25.yaml").read_text()
    assert str(bundle.resolve() / "policy.onnx") in text
    # Provenance comments survive the rewrite.
    assert "parity max_abs_diff 3.1e-07" in text
    assert "# external data" in text


# ---------------- (b) a hash mismatch ---------------------------------------


def test_verify_rejects_a_hash_mismatch(tmp_path, capsys) -> None:
    """The bundle's own manifest disagrees with the file the config names."""
    bundle = _write_bundle(tmp_path)
    assert _pin(bundle) == 0
    _write_sums(bundle)
    (bundle / "policy.onnx").write_bytes(b"a-different-policy-entirely")

    assert _verify(bundle) == 1
    err = capsys.readouterr().err
    assert "ACTIVATION UNVERIFIED" in err
    assert "policy.onnx" in err
    assert SUMS_NAME in err


def test_verify_rejects_a_file_absent_from_the_manifest(tmp_path) -> None:
    bundle = _write_bundle(tmp_path)
    assert _pin(bundle) == 0
    sums = _write_sums(bundle)
    sums.pop("policy.onnx")

    cfg = {"policy": {"onnx_path": str(bundle / "policy.onnx")}}
    problems = verify_activation(bundle, cfg, sums)
    assert len(problems) == 1
    assert "unverifiable" in problems[0]


def test_verify_rejects_a_missing_external_data_sidecar(tmp_path) -> None:
    """An external-data ONNX is two files and the sidecar is never in the config."""
    bundle = _write_bundle(tmp_path)
    (bundle / "policy.onnx.data").write_bytes(b"weights")
    sums = _write_sums(bundle)
    (bundle / "policy.onnx.data").unlink()

    cfg = {"policy": {"onnx_path": str(bundle / "policy.onnx")}}
    problems = verify_activation(bundle, cfg, sums)
    assert any("sidecar" in p for p in problems)


# ---------------- (c) a path escaping the bundle ----------------------------


def test_verify_rejects_a_path_outside_the_bundle(tmp_path) -> None:
    outside = tmp_path / "stale"
    outside.mkdir()
    (outside / "policy.onnx").write_bytes(b"stand-v3-era-export")

    bundle = _write_bundle(tmp_path)
    sums = _write_sums(bundle)

    cfg = {"policy": {"onnx_path": str(outside / "policy.onnx")}}
    problems = verify_activation(bundle, cfg, sums)
    assert len(problems) == 1
    assert "points outside the bundle" in problems[0]


def test_verify_rejects_a_symlink_escaping_the_bundle(tmp_path) -> None:
    """Absolute and inside the bundle by name, but resolving elsewhere."""
    outside = tmp_path / "stale"
    outside.mkdir()
    (outside / "real.onnx").write_bytes(b"stand-v3-era-export")

    bundle = _write_bundle(tmp_path)
    sums = _write_sums(bundle)
    link = bundle / "linked.onnx"
    link.symlink_to(outside / "real.onnx")

    cfg = {"policy": {"onnx_path": str(link)}}
    problems = verify_activation(bundle, cfg, sums)
    assert len(problems) == 1
    assert "points outside the bundle" in problems[0]


def test_verify_reports_a_missing_file(tmp_path) -> None:
    bundle = _write_bundle(tmp_path)
    sums = _write_sums(bundle)
    cfg = {"policy": {"onnx_path": str(bundle / "never_staged.onnx")}}
    problems = verify_activation(bundle, cfg, sums)
    assert any("missing file" in p for p in problems)


# ---------------- (d) activation silently skipped ---------------------------


def test_staging_script_pins_and_verifies(tmp_path) -> None:
    """An unwired module fixes nothing: the deploy path must actually call it."""
    text = STAGING_SCRIPT.read_text()
    pin = re.search(r"activation\.py[\"']?\s+pin\b", text)
    verify = re.search(r"activation\.py[\"']?\s+verify\b", text)
    assert pin, "staging never pins the bundle's config"
    assert verify, "staging never verifies activation"

    # Pinning rewrites the config, so a manifest taken first would be stale.
    assert pin.start() < text.index("sha256sum ./*"), "pin must run before SHA256SUMS is generated"
    # The module has to travel, or the payload cannot re-verify.
    assert re.search(
        r"cp\s+-f\s+.*activation\.py.*\$DEST/activation\.py", text
    ), "activation.py must be copied into the bundle"


def test_verify_fails_closed_without_a_manifest(tmp_path, capsys) -> None:
    bundle = _write_bundle(tmp_path)
    assert _verify(bundle) == 1
    assert "ACTIVATION UNVERIFIED" in capsys.readouterr().err


def test_verify_refuses_an_ambiguous_bundle(tmp_path, capsys) -> None:
    """Two configs means the operator picks, which is the thing being removed."""
    bundle = _write_bundle(tmp_path)
    _write_sums(bundle)
    (bundle / "deploy_other.yaml").write_text(CONFIG_TEMPLATE)

    assert _verify(bundle) == 1
    err = capsys.readouterr().err
    assert "ACTIVATION UNVERIFIED" in err
    assert "exactly one" in err

    with pytest.raises(ValueError, match="exactly one"):
        find_config(bundle)


def test_pin_refuses_when_an_enabled_artifact_was_never_staged(tmp_path, capsys) -> None:
    cfg_text = CONFIG_TEMPLATE.replace(
        "reliability:\n  enabled: false\n",
        'reliability:\n  enabled: true\n  artifact: "checkpoints/run/shield.npz"\n',
    )
    bundle = _write_bundle(tmp_path, cfg_text)

    assert _pin(bundle) == 1
    err = capsys.readouterr().err
    assert "REFUSING TO PIN" in err
    assert "reliability.artifact" in err


# ---------------- supporting behaviour --------------------------------------


def test_required_paths_is_gated_on_the_flags_the_node_checks(tmp_path) -> None:
    cfg = {
        "policy": {
            "onnx_path": "a.onnx",
            "mode_switch": {
                "enabled": False,
                "stand_onnx_path": "s.onnx",
                "walk_onnx_path": "w.onnx",
            },
        },
        "reliability": {"enabled": False, "artifact": "shield.npz"},
    }
    assert set(required_paths(cfg)) == {"policy.onnx_path"}

    cfg["policy"]["mode_switch"]["enabled"] = True
    cfg["reliability"]["enabled"] = True
    assert set(required_paths(cfg)) == {
        "policy.onnx_path",
        "policy.mode_switch.stand_onnx_path",
        "policy.mode_switch.walk_onnx_path",
        "reliability.artifact",
    }


def test_pin_config_text_leaves_unstaged_names_alone(tmp_path) -> None:
    text = 'policy:\n  onnx_path: "checkpoints/run/policy.onnx"\n'
    out, pinned = pin_config_text(text, tmp_path, present=set())
    assert out == text
    assert pinned == {}


def test_parse_sha256sums_strips_dot_slash_and_binary_marker() -> None:
    sums = parse_sha256sums("aa  ./policy.onnx\nbb  *policy.pt\n\ncc\n")
    assert sums == {"policy.onnx": "aa", "policy.pt": "bb"}
