"""Tests for scripts/harness_record.sh (exercises --mock path only).

The real recorder needs ROS2 + ffmpeg + a /dev/video device, none of which
are guaranteed in CI. The mock path is the surface we care about for the
script's argument plumbing and manifest layout.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "harness_record.sh"


@pytest.fixture
def tmp_repo(tmp_path):
    """Copy the script into a tmp repo skeleton so it doesn't write into the
    real data/ tree during tests."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scripts").mkdir()
    shutil.copy2(SCRIPT, repo / "scripts" / SCRIPT.name)
    (repo / "scripts" / SCRIPT.name).chmod(0o755)
    return repo


def test_script_exists_and_is_executable():
    assert SCRIPT.exists(), SCRIPT
    st = os.stat(SCRIPT)
    assert st.st_mode & 0o111, "harness_record.sh must be executable"


def test_mock_run_writes_manifest(tmp_repo):
    env = os.environ.copy()
    res = subprocess.run(
        ["bash", "scripts/harness_record.sh", "--mock", "gate7", "selftest_block", "3"],
        cwd=tmp_repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    # locate the lab dir
    lab = tmp_repo / "data" / "lab"
    assert lab.is_dir()
    days = list(lab.iterdir())
    assert len(days) == 1
    runs = list(days[0].iterdir())
    assert len(runs) == 1
    run = runs[0]
    assert run.name.startswith("gate7_selftest_block_")
    manifest = run / "manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert data["block"] == "gate7"
    assert data["scenario"] == "selftest_block"
    assert data["mock"] is True
    assert "/lowstate" in data["topics"]
    assert "/joint_states" in data["topics"]
    assert "/phoenix/estop" in data["topics"]


def test_missing_args_fail(tmp_repo):
    res = subprocess.run(
        ["bash", "scripts/harness_record.sh", "--mock"],
        cwd=tmp_repo,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert res.returncode != 0
    assert "block label required" in res.stderr or "block label required" in res.stdout
