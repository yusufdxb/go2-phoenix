"""Code identity: git on the workstation, the synced manifest on the payload."""

from __future__ import annotations

import hashlib
import subprocess

import pytest

from phoenix.sim2real.provenance import (
    EXPECTED_BRANCH,
    identity_problems,
    resolve_code_identity,
)


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", EXPECTED_BRANCH)
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (root / "a.py").write_text("x = 1\n")
    _git(root, "add", "a.py")
    _git(root, "commit", "-q", "-m", "init")
    return root


def test_clean_git_checkout_on_expected_branch(repo) -> None:
    ident = resolve_code_identity(repo)
    assert ident.source == "git" and not ident.dirty and len(ident.sha) == 40
    assert identity_problems(ident) == []
    assert identity_problems(ident, expected_sha=ident.sha[:7]) == []


def test_dirty_tracked_file_is_a_problem(repo) -> None:
    (repo / "a.py").write_text("x = 2\n")
    ident = resolve_code_identity(repo)
    assert ident.dirty
    assert any("dirty" in p for p in identity_problems(ident))


def test_wrong_branch_and_wrong_sha_are_problems(repo) -> None:
    _git(repo, "checkout", "-q", "-b", "main")
    ident = resolve_code_identity(repo)
    problems = identity_problems(ident, expected_sha="deadbeef")
    assert any("branch" in p for p in problems)
    assert any("does not match expected" in p for p in problems)


def test_short_expected_sha_is_refused(repo) -> None:
    ident = resolve_code_identity(repo)
    assert identity_problems(ident, expected_sha=ident.sha[:4])


def _write_payload(root, commit="abc1234" + "0" * 33, branch=EXPECTED_BRANCH, dirty="false"):
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "m.py").write_text("print('hi')\n")
    digest = hashlib.sha256((root / "src" / "m.py").read_bytes()).hexdigest()
    (root / "PAYLOAD_SHA256SUMS").write_text(f"{digest}  src/m.py\n")
    (root / "PAYLOAD_SYNC.txt").write_text(
        f"commit:  {commit}\nbranch:  {branch}\ndirty:   {dirty}\n"
    )


def test_payload_sync_wins_over_a_stale_git_checkout(repo) -> None:
    _write_payload(repo)
    ident = resolve_code_identity(repo)
    assert ident.source == "payload_sync"
    assert ident.sha.startswith("abc1234") and ident.manifest_verified is True
    assert identity_problems(ident, expected_sha="abc1234") == []


def test_payload_file_modified_after_sync_is_a_problem(tmp_path) -> None:
    _write_payload(tmp_path)
    (tmp_path / "src" / "m.py").write_text("print('edited on the payload')\n")
    ident = resolve_code_identity(tmp_path)
    assert ident.manifest_verified is False
    assert identity_problems(ident)


def test_payload_synced_from_dirty_tree_is_a_problem(tmp_path) -> None:
    _write_payload(tmp_path, dirty="true")
    assert any("dirty" in p for p in identity_problems(resolve_code_identity(tmp_path)))


def test_neither_git_nor_payload_is_unknown(tmp_path) -> None:
    ident = resolve_code_identity(tmp_path)
    assert ident.source == "unknown" and ident.sha is None
    assert identity_problems(ident)
