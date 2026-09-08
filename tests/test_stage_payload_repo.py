"""Tests for ``scripts/stage_payload_repo.sh`` in its local-destination mode.

The payload cannot ``git fetch`` at the lab, so this script is the only way
current code reaches it. The properties that matter: exactly the tracked
execution set travels, the manifest describes what travelled, and the record
of *which commit* travelled cannot claim a clean tree when it was not.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "stage_payload_repo.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("rsync") is None or shutil.which("git") is None,
    reason="needs rsync + git on PATH",
)


def _run(dest: Path, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.run(
        ["bash", str(SCRIPT), str(dest)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )


def _tracked_sync_set() -> list[str]:
    out = subprocess.run(
        [
            "git",
            "ls-files",
            "--",
            "src",
            "configs",
            "scripts",
            "docs",
            "pyproject.toml",
            "README.md",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def test_local_stage_sends_exactly_the_tracked_execution_set(tmp_path) -> None:
    dest = tmp_path / "payload-repo"
    # The developer's tree may be dirty while this test runs; that is not what
    # this test is about (see the dirty tests below).
    proc = _run(dest, {"ALLOW_DIRTY": "1"})
    assert proc.returncode == 0, proc.stdout + proc.stderr

    sent = {
        str(p.relative_to(dest))
        for p in dest.rglob("*")
        if p.is_file() and p.name not in ("PAYLOAD_SHA256SUMS", "PAYLOAD_SYNC.txt")
    }
    assert sent == set(_tracked_sync_set())
    # nothing outside the execution set, and no caches
    assert not (dest / "reliability_eval").exists()
    assert not (dest / "checkpoints").exists()
    assert not list(dest.rglob("__pycache__"))

    # the manifest verifies on the destination, the way it will on the payload
    check = subprocess.run(
        ["sha256sum", "-c", "--quiet", "PAYLOAD_SHA256SUMS"],
        cwd=dest,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stdout + check.stderr

    record = (dest / "PAYLOAD_SYNC.txt").read_text()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert f"commit:  {head}" in record
    assert "verify:  sha256sum -c PAYLOAD_SHA256SUMS" in record


def test_manifest_lists_every_sent_file(tmp_path) -> None:
    dest = tmp_path / "payload-repo"
    assert _run(dest, {"ALLOW_DIRTY": "1"}).returncode == 0
    listed = {
        line.split(maxsplit=1)[1].strip()
        for line in (dest / "PAYLOAD_SHA256SUMS").read_text().splitlines()
        if line.strip()
    }
    assert listed == set(_tracked_sync_set())


def _tree_is_dirty() -> bool:
    r = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    return bool(r.stdout.strip())


@pytest.mark.skipif(_tree_is_dirty(), reason="needs a clean working tree to prove the refusal")
def test_refuses_a_dirty_tree_and_records_dirty_when_forced(tmp_path) -> None:
    """A modified tracked file, and an untracked file inside the sync set,
    must both count as dirty: untracked files do not travel, so a clean
    ``dirty: false`` record would misdescribe the payload."""
    probe = REPO_ROOT / "scripts" / "_untracked_probe_for_test.sh"
    try:
        probe.write_text("#!/usr/bin/env bash\n")
        refused = _run(tmp_path / "a")
        assert refused.returncode == 1
        assert "REFUSING TO STAGE" in refused.stderr

        forced = _run(tmp_path / "b", {"ALLOW_DIRTY": "1"})
        assert forced.returncode == 0, forced.stdout + forced.stderr
        assert "dirty:   true" in (tmp_path / "b" / "PAYLOAD_SYNC.txt").read_text()
        # and the untracked probe did NOT travel
        assert not (tmp_path / "b" / "scripts" / probe.name).exists()
    finally:
        probe.unlink(missing_ok=True)


def test_colon_in_a_local_path_is_not_a_host(tmp_path) -> None:
    dest = tmp_path / "a:b"
    proc = _run(dest, {"ALLOW_DIRTY": "1"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (dest / "PAYLOAD_SYNC.txt").exists()
