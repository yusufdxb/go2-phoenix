"""Tests for scripts/harness_eod.sh (--mock).

Exercises the rsync + checksum path against a tmp source tree, writing into
a tmp "T7" directory (--mock points T7_MOUNT at /tmp/t7_mock by default;
we override via env to keep the test hermetic).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "harness_eod.sh"


@pytest.fixture
def staged_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, repo / "scripts" / SCRIPT.name)
    (repo / "scripts" / SCRIPT.name).chmod(0o755)
    # seed some parquets + a lab run
    (repo / "data" / "failures").mkdir(parents=True)
    (repo / "data" / "failures" / "pqa_slip_0001.parquet").write_bytes(b"PAR1stub_slip")
    (repo / "data" / "failures" / "pqa_push_lat_0001.parquet").write_bytes(b"PAR1stub_push")
    # a non-parquet file that should NOT be transferred
    (repo / "data" / "failures" / "scratch.txt").write_text("ignore me")

    day = repo / "data" / "lab" / "2026-05-17"
    run = day / "gate7_floor_hold_120000"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text('{"block":"gate7"}')
    (run / "camera.mp4").write_bytes(b"\x00MP4stub")
    return repo


def test_script_exists():
    assert SCRIPT.exists()


def test_mock_rsync_runs_clean(staged_repo, tmp_path):
    t7 = tmp_path / "t7"
    env = os.environ.copy()
    env["T7_MOUNT"] = str(t7)
    env["LAB_DATE"] = "2026-05-17"
    # --mock honors T7_MOUNT env override (we set it above) and auto-mkdir
    res = subprocess.run(
        ["bash", "scripts/harness_eod.sh", "--mock"],
        cwd=staged_repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    # parquets transferred
    assert (t7 / "data" / "failures" / "pqa_slip_0001.parquet").exists()
    assert (t7 / "data" / "failures" / "pqa_push_lat_0001.parquet").exists()
    # non-parquet skipped
    assert not (t7 / "data" / "failures" / "scratch.txt").exists()
    # lab tree transferred
    assert (
        t7 / "data" / "lab" / "2026-05-17" / "gate7_floor_hold_120000" / "manifest.json"
    ).exists()
    assert (t7 / "data" / "lab" / "2026-05-17" / "gate7_floor_hold_120000" / "camera.mp4").exists()
    assert "OK" in res.stdout


def test_eod_fails_when_t7_missing(staged_repo, tmp_path):
    """Non-mock invocation should fail if the T7 mount doesn't exist."""
    env = os.environ.copy()
    env["T7_MOUNT"] = str(tmp_path / "nope_not_mounted")
    env["LAB_DATE"] = "2026-05-17"
    res = subprocess.run(
        ["bash", "scripts/harness_eod.sh"],
        cwd=staged_repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert res.returncode != 0
    assert "T7 not mounted" in res.stderr
