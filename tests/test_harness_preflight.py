"""Smoke test for scripts/harness_preflight.sh --mock.

Runs the script's mock path end to end. The mock path skips T7 rsync, git
ff-only merge, verify_deploy, ROS2 bridge bringup, and physical confirmation
prompts (auto-confirms). It still executes pytest on the harness self-tests,
so this test would be self-referential if it ran via pytest. To avoid that
recursion, we monkey-patch the script invocation to point at a dummy pytest
target that always passes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "harness_preflight.sh"


def test_script_exists_and_executable():
    assert SCRIPT.exists()
    st = os.stat(SCRIPT)
    assert st.st_mode & 0o111


def test_mock_runs_to_completion(tmp_path):
    """Build a tiny tmp 'repo' with just the script + a trivial passing test
    and run preflight --mock there. Avoids the recursion of running pytest on
    these very tests."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "tests").mkdir()
    shutil.copy2(SCRIPT, repo / "scripts" / SCRIPT.name)
    (repo / "scripts" / SCRIPT.name).chmod(0o755)
    # Create the two pytest targets the script names in --mock mode
    (repo / "tests" / "test_harness_diversity.py").write_text(
        "def test_trivial():\n    assert True\n"
    )
    (repo / "tests" / "test_harness_record.py").write_text(
        "def test_trivial():\n    assert True\n"
    )
    # Provide a pyproject so pytest doesn't traverse upward into the real repo
    (repo / "pyproject.toml").write_text("[tool.pytest.ini_options]\ntestpaths=[\"tests\"]\n")

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    res = subprocess.run(
        ["bash", "scripts/harness_preflight.sh", "--mock"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "PREFLIGHT COMPLETE" in res.stdout
