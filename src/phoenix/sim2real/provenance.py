"""Which code is running: the exact commit, read without mutating anything.

Two layouts exist and they disagree about where the truth is:

* **A git checkout** (the workstation). ``git rev-parse HEAD`` is the identity and
  tracked modifications make it dirty.
* **The payload.** ``scripts/stage_payload_repo.sh`` cannot ``git fetch`` (no
  egress), so it rsyncs the tracked files ON TOP of whatever old checkout the
  payload already has and writes ``PAYLOAD_SYNC.txt`` plus
  ``PAYLOAD_SHA256SUMS``. There ``git rev-parse HEAD`` still answers, and it
  answers with the April commit: exactly the wrong identity. So when
  ``PAYLOAD_SYNC.txt`` is present it wins, and it counts only if every file in
  the manifest still hashes to what was sent.

Nothing here runs ``git fetch``, ``merge``, ``checkout`` or ``reset``. A
preflight must observe the code, never move it.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: The corrected source of truth for hardware work. Recorded here so the
#: preflight can refuse any other branch instead of defaulting to ``main``.
EXPECTED_BRANCH = "feat/causal-viability-replication"

PAYLOAD_SYNC_NAME = "PAYLOAD_SYNC.txt"
PAYLOAD_SUMS_NAME = "PAYLOAD_SHA256SUMS"


@dataclass
class CodeIdentity:
    sha: str | None
    source: str  # "payload_sync" | "git" | "unknown"
    branch: str | None
    dirty: bool
    manifest_verified: bool | None = None
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_sync(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def resolve_code_identity(repo_root: str | Path) -> CodeIdentity:
    """Resolve the running code's commit. Read-only. Never raises for a bad tree."""
    root = Path(repo_root)
    sync_path = root / PAYLOAD_SYNC_NAME
    if sync_path.is_file():
        info = _parse_sync(sync_path.read_text())
        ident = CodeIdentity(
            sha=info.get("commit") or None,
            source="payload_sync",
            branch=info.get("branch") or None,
            dirty=info.get("dirty", "true") != "false",
        )
        sums_path = root / PAYLOAD_SUMS_NAME
        if not sums_path.is_file():
            ident.manifest_verified = False
            ident.problems.append(f"{PAYLOAD_SYNC_NAME} present but {PAYLOAD_SUMS_NAME} missing")
            return ident
        mismatched: list[str] = []
        checked = 0
        for line in sums_path.read_text().splitlines():
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            digest, name = parts[0], parts[1].strip().lstrip("*")
            target = root / name
            checked += 1
            if not target.is_file() or _sha256(target) != digest:
                mismatched.append(name)
        ident.manifest_verified = checked > 0 and not mismatched
        if checked == 0:
            ident.problems.append(f"{PAYLOAD_SUMS_NAME} lists no files")
        if mismatched:
            ident.problems.append(
                f"{len(mismatched)} synced file(s) no longer match {PAYLOAD_SUMS_NAME}, "
                f"first: {mismatched[:3]}"
            )
        if not ident.sha:
            ident.problems.append(f"{PAYLOAD_SYNC_NAME} has no commit line")
        return ident

    sha = _git(root, "rev-parse", "HEAD")
    if sha is None:
        return CodeIdentity(
            sha=None,
            source="unknown",
            branch=None,
            dirty=True,
            problems=[f"{root} is neither a git checkout nor a staged payload repo"],
        )
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    ident = CodeIdentity(sha=sha, source="git", branch=branch, dirty=bool(status))
    if status is None:
        ident.problems.append("git status failed")
        ident.dirty = True
    return ident


def identity_problems(
    ident: CodeIdentity,
    *,
    expected_sha: str | None = None,
    expected_branch: str | None = EXPECTED_BRANCH,
    allow_dirty: bool = False,
) -> list[str]:
    """Every reason this identity must not be trusted for a hardware stage."""
    problems = list(ident.problems)
    if not ident.sha:
        problems.append("code identity unresolved")
        return problems
    if expected_branch is not None and ident.branch != expected_branch:
        problems.append(f"branch is {ident.branch!r}, expected {expected_branch!r}")
    if expected_sha is not None:
        want = expected_sha.strip().lower()
        if len(want) < 7 or not ident.sha.lower().startswith(want):
            problems.append(f"commit {ident.sha} does not match expected {expected_sha}")
    if ident.dirty and not allow_dirty:
        problems.append("working tree is dirty: the recorded commit would not describe the code")
    if ident.source == "payload_sync" and ident.manifest_verified is not True:
        problems.append("payload manifest not verified")
    return problems


__all__ = [
    "EXPECTED_BRANCH",
    "CodeIdentity",
    "identity_problems",
    "resolve_code_identity",
]
