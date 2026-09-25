"""scripts/harness_preflight.sh and scripts/dryrun_pipeline.sh fail loud.

These run the scripts' refusal paths for real (no ROS process is ever launched by
them) and check the source for the specific false-green patterns this pass removed.
The full stage A run is not invoked here: it runs this very test suite.
"""

from __future__ import annotations

import json
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


def _harness_source() -> str:
    return HARNESS.read_text()


def test_every_on_robot_stage_runs_the_contention_interlock() -> None:
    """come-here can command the robot; no stage may start while it is up."""
    src = _harness_source()
    assert "assert_no_contention() {" in src
    # B..E plus the shared stand function that serves F, G and H.
    for call in (
        "assert_no_contention B",
        "assert_no_contention C",
        "assert_no_contention D",
        "assert_no_contention E",
        'assert_no_contention "$stage"',
    ):
        assert call in src, f"missing {call}"


def test_contention_evidence_is_per_stage_not_overwritten() -> None:
    # A single contention.json would let a later stage inherit an earlier
    # stage's evidence, which is the same class of bug as a stale ledger.
    src = _harness_source()
    assert "contention_${stage}.json" in src
    assert '"$SESSION/contention.json"' not in src


def test_the_operator_gate_defaults_to_the_terminal() -> None:
    """Wiring the remote must not silently change how a live stage is confirmed."""
    src = _harness_source()
    assert 'if [[ "${PHOENIX_OPERATOR_REMOTE:-0}" != "1" ]]; then' in src


def test_the_operator_gate_fails_closed_on_an_unusable_answer() -> None:
    src = _harness_source()
    gate = src.split("operator_gate() {", 1)[1].split("\n}", 1)[0]
    # A timeout or an ambiguous double press must halt, never become a pass.
    assert 'halt "operator remote gave no usable judgement' in gate
    assert 'halt "operator remote returned' in gate
    # Only an explicit A is a yes.
    assert "yes) printf 'y'" in gate
    assert "no|halt) printf 'n'" in gate


def test_the_live_stand_prompts_go_through_the_operator_gate() -> None:
    src = _harness_source()
    stand = src.split("stage_stand() {", 1)[1]
    assert "operator_gate arm " in stand
    assert "operator_gate judgement " in stand
    assert "operator_gate release " in stand
    # The raw reads they replaced must be gone from the stand path.
    assert 'Press Enter to start: " _' not in stand


PREP = REPO_ROOT / "scripts" / "payload_prep.sh"


def test_payload_prep_exists_and_is_executable() -> None:
    assert PREP.exists()
    assert os.access(PREP, os.X_OK)


def test_payload_prep_moves_no_motors() -> None:
    """It runs before any gate; it must not touch the command path."""
    src = PREP.read_text()
    for forbidden in ("lowcmd_bridge_node", "ros2_policy_node", "--live", "/lowcmd"):
        assert forbidden not in src, f"payload_prep references {forbidden}"


def test_payload_prep_fails_closed_on_this_workstation(tmp_path) -> None:
    # A workstation has no enP8p1s0, so the NIC check must fail and the script
    # must exit non-zero. A prep script that exits 0 on the wrong host is worse
    # than none, because the operator would trust it.
    res = subprocess.run([str(PREP)], cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
    assert res.returncode != 0
    assert "NOT READY" in res.stdout + res.stderr


def test_payload_prep_checks_every_known_session_killer() -> None:
    src = PREP.read_text()
    for needle in (
        "come-here.service",
        "NetworkInterface",
        "getusersitepackages",
        "contention",
        "no RTC" if "no RTC" in src else "RTC",
    ):
        assert needle in src, f"payload_prep does not check {needle}"


def test_payload_prep_never_suggests_pip_editable_install() -> None:
    # The payload's setuptools 59.6.0 predates PEP 660, so that command fails.
    src = PREP.read_text()
    assert "do NOT use pip install -e ." in src
    assert "python3 -m pip install --no-deps --no-build-isolation -e ." not in src


def test_shipped_dds_config_binds_the_payload_nic() -> None:
    import xml.etree.ElementTree as ET

    xml = REPO_ROOT / "configs" / "payload" / "cyclonedds.xml"
    assert xml.exists(), "the DDS config must ship so it is not hand-built at the lab"
    root = ET.parse(xml).getroot()
    ns = {"c": "https://cdds.io/config"}
    names = [n.get("name") for n in root.findall(".//c:NetworkInterface", ns)]
    assert names == ["enP8p1s0"], f"expected the payload NIC, got {names}"


BUNDLE = REPO_ROOT / "scripts" / "stage_payload_bundle.sh"


def test_bundle_staging_binds_stage_a_evidence_to_head_and_the_lock() -> None:
    """Stage A evidence is commit-bound; staging must not accept a dead SHA.

    Amending a commit after running stage A produces exactly that situation, and
    without this check it is only caught by `A --payload` on the robot, which is
    the most expensive place to find it.
    """
    src = BUNDLE.read_text()
    assert "rev-parse HEAD" in src
    assert "but HEAD is" in src
    assert "lock_file_sha256" in src
    # A rehearsal must never be stageable as real evidence.
    assert 'r.get("rehearsal")' in src
    assert "code_identity" in src


def test_bundle_staging_refuses_evidence_from_another_commit(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "parity_gate.json").write_text('{"passed": true}')
    session = tmp_path / "session"
    session.mkdir()
    (session / "stage_A.json").write_text(
        json.dumps(
            {
                "stage": "A",
                "verdict": "GO",
                "mode": "workstation",
                "rehearsal": False,
                # A commit that is not HEAD.
                "code_identity": {"sha": "0" * 40, "dirty": False},
            }
        )
    )
    (session / "parity_golden.npz").write_bytes(b"not a real npz")
    res = subprocess.run(
        [
            str(BUNDLE),
            str(checkpoint),
            "configs/sim2real/deploy_stand_h25.yaml",
            str(tmp_path / "dest"),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "STAGE_A_SESSION": str(session)},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert res.returncode != 0
    combined = res.stdout + res.stderr
    assert "but HEAD is" in combined
    assert "REFUSING TO STAGE" in combined
    assert not (tmp_path / "dest").exists(), "a refused stage must leave no bundle"


def test_launchers_pin_single_threaded_blas() -> None:
    for script in (HARNESS, DRYRUN):
        text = script.read_text()
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            assert f'{var}="${{{var}:-1}}"' in text, (script.name, var)
