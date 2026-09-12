"""scripts/harness_preflight.sh and scripts/dryrun_pipeline.sh fail loud.

These run the scripts' refusal paths for real (no ROS process is ever launched by
them) and check the source for the specific false-green patterns this pass removed.
The full stage A run is not invoked here: it runs this very test suite.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = REPO_ROOT / "scripts" / "harness_preflight.sh"
DRYRUN = REPO_ROOT / "scripts" / "dryrun_pipeline.sh"
ESTOP = REPO_ROOT / "scripts" / "estop_publisher.sh"


def _run(args, tmp_path, **env_over):
    env = os.environ.copy()
    env.update(
        PHOENIX_SESSION=str(tmp_path / "session"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.pop("PHOENIX_EXPECT_SHA", None)
    env.pop("PHOENIX_REHEARSAL", None)
    env.update(env_over)
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )


def test_scripts_exist_and_are_executable() -> None:
    for script in (HARNESS, DRYRUN, ESTOP):
        assert script.exists() and os.stat(script).st_mode & 0o111, script


def test_harness_never_moves_code_and_has_no_t7_or_main_default() -> None:
    src = HARNESS.read_text()
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    for banned in (
        "git fetch",
        "git merge",
        "git pull",
        "git checkout",
        "git reset",
        "rsync",
        "T7",
        "PHOENIX_BRANCH",
        ":-main",
    ):
        assert banned not in code, banned


def test_no_false_green_patterns() -> None:
    for script in (HARNESS, DRYRUN):
        code = [
            line for line in script.read_text().splitlines() if not line.lstrip().startswith("#")
        ]
        for line in code:
            if "|| true" in line:
                # Only allowed on signal delivery to an already-exiting process and on
                # reaping, both of which follow an explicit liveness check.
                assert re.search(
                    r"kill -INT|wait \"\$pid\"", line
                ), f"{script.name}: {line.strip()}"
        joined = "\n".join(code)
        assert "/tmp/" not in joined, f"{script.name} writes to /tmp"


def test_dryrun_takes_the_selected_config_and_never_hardcodes_one() -> None:
    src = "\n".join(
        line for line in DRYRUN.read_text().splitlines() if not line.lstrip().startswith("#")
    )
    assert "configs/sim2real/deploy.yaml" not in src
    assert "--onnx" not in src
    harness = HARNESS.read_text()
    assert '--config "$DEPLOY_CFG" --lock "$DEPLOY_LOCK"' in harness


def test_no_suspended_or_unloaded_feet_in_the_h25_path() -> None:
    src = HARNESS.read_text().lower()
    assert "feet unloaded" not in src and "suspended" not in src
    assert "feet on the ground" in src


def test_status_on_an_empty_session_is_no_go(tmp_path) -> None:
    res = _run(["bash", str(HARNESS), "status"], tmp_path)
    assert res.returncode != 0
    assert "READY FOR STAND-ONLY LIVE GO2 TEST (stage F):  NO" in res.stdout


def test_hardware_stages_require_the_expected_commit(tmp_path) -> None:
    for stage in ("B", "C", "D", "E", "F", "G", "H"):
        res = _run(["bash", str(HARNESS), stage], tmp_path)
        assert res.returncode != 0, stage
        assert "PHOENIX_EXPECT_SHA is required" in res.stderr, (stage, res.stderr)
        assert "launched" not in res.stdout


def test_live_stage_without_prior_evidence_launches_nothing(tmp_path) -> None:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    res = _run(["bash", str(HARNESS), "E"], tmp_path, PHOENIX_EXPECT_SHA=sha)
    assert res.returncode != 0
    assert "launched" not in res.stdout


def test_payload_stage_a_requires_bundle_and_sha(tmp_path) -> None:
    res = _run(["bash", str(HARNESS), "A", "--payload"], tmp_path)
    assert res.returncode != 0 and "PHOENIX_EXPECT_SHA is required" in res.stderr


def test_dryrun_refuses_without_config_lock_and_sha(tmp_path) -> None:
    res = _run(["bash", str(DRYRUN)], tmp_path)
    assert res.returncode != 0 and "--config must name an existing deploy config" in res.stderr


def test_estop_publisher_refuses_outside_a_dryrun(tmp_path) -> None:
    res = _run(["bash", str(ESTOP)], tmp_path)
    assert res.returncode != 0
    assert "NOT A DEADMAN" in res.stderr
