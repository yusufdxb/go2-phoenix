"""Exploratory gantry A/B package: decision rule, arm identity, and runner safety properties."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/gantry_ab_campaign.sh"
PREREG = REPO / "docs/preregistration/2026-10-07-gantry-ab-exploratory.md"


def _analyze():
    spec = importlib.util.spec_from_file_location(
        "gantry_ab_analyze", REPO / "scripts/gantry_ab_analyze.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gantry_ab_analyze"] = mod
    spec.loader.exec_module(mod)
    return mod


ga = _analyze()


def _chosen(y_a, f_a, y_b, f_b, tilt=(2.0, 2.0), dq=(0.03, 0.03)):
    runs = []
    for arm, ys, fs, ti, d in (("A", y_a, f_a, tilt[0], dq[0]), ("B", y_b, f_b, tilt[1], dq[1])):
        for k, (y, f) in enumerate(zip(ys, fs, strict=True)):
            runs.append(
                {
                    "arm": arm,
                    "measured_arm": arm,
                    "block": k,
                    "slot": "yaw05_3s",
                    "walk": {"Y": y, "yaw_rate_overshoot": 1.0},
                    "safety_event": False,
                }
            )
            runs.append(
                {
                    "arm": arm,
                    "measured_arm": arm,
                    "block": k,
                    "slot": "vx03_4s",
                    "walk": {"F": f},
                    "safety_event": False,
                }
            )
            runs.append(
                {
                    "arm": arm,
                    "measured_arm": arm,
                    "block": k,
                    "slot": "G",
                    "stand": {"max_tilt_deg": ti, "dq_rms_max_calf": d, "calf_sag_median": 0.13},
                    "safety_event": False,
                }
            )
    return runs


def test_decision_better_worse_inconclusive_insufficient():
    r = _chosen([0.2, 0.22, 0.21], [0.6, 0.58, 0.61], [0.5, 0.55, 0.6], [0.8, 0.82, 0.79])
    assert ga.decide(r, r)["verdict"] == "B OBVIOUSLY BETTER"
    r = _chosen([0.5, 0.55, 0.6], [0.8, 0.82, 0.79], [0.2, 0.22, 0.21], [0.6, 0.58, 0.61])
    assert ga.decide(r, r)["verdict"] == "B OBVIOUSLY WORSE"
    r = _chosen([0.2, 0.5, 0.21], [0.6, 0.58, 0.61], [0.45, 0.55, 0.6], [0.8, 0.82, 0.79])
    assert ga.decide(r, r)["verdict"] == "INCONCLUSIVE"  # yaw not separated
    r = _chosen([0.2], [0.6], [0.5], [0.8])
    assert ga.decide(r, r)["verdict"] == "INSUFFICIENT"


def test_better_requires_no_new_safety_event_and_no_worse_stand():
    r = _chosen([0.2, 0.22, 0.21], [0.6, 0.58, 0.61], [0.5, 0.55, 0.6], [0.8, 0.82, 0.79])
    r[-1]["safety_event"] = True  # a B-only safety event
    assert ga.decide(r, r)["verdict"] == "B OBVIOUSLY WORSE"
    r = _chosen(
        [0.2, 0.22, 0.21], [0.6, 0.58, 0.61], [0.5, 0.55, 0.6], [0.8, 0.82, 0.79], tilt=(2.0, 5.0)
    )
    assert ga.decide(r, r)["verdict"] == "INCONCLUSIVE"


def test_first_valid_uses_first_valid_attempt_and_never_a_third():
    runs = [
        {"block": 1, "slot": "G", "attempt": 1, "valid": False},
        {"block": 1, "slot": "G", "attempt": 2, "valid": True},
        {"block": 2, "slot": "G", "attempt": 3, "valid": True},
    ]
    chosen = ga.first_valid(runs)
    assert [(c["block"], c["attempt"]) for c in chosen] == [(1, 2)]


def test_arm_identity_from_artifact_hashes_only():
    for arm, want in ga.ARMS.items():
        man = {
            "artifacts": {
                k: {"sha256": want[k]} for k in ("checkpoint", "policy.onnx", "policy.onnx.data")
            }
        }
        assert ga.arm_of(man) == arm
    mixed = {
        "artifacts": {
            "checkpoint": {"sha256": ga.ARMS["A"]["checkpoint"]},
            "policy.onnx": {"sha256": ga.ARMS["B"]["policy.onnx"]},
            "policy.onnx.data": {"sha256": ga.ARMS["B"]["policy.onnx.data"]},
        }
    }
    assert ga.arm_of(mixed) is None


def test_quat_yaw_xyzw():
    yaw = np.radians(30.0)
    q = np.array([[0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]])
    assert ga.quat_yaw(q)[0] == pytest.approx(yaw)


def test_runner_hashes_match_analysis_and_prereg():
    src, pre = RUNNER.read_text(), PREREG.read_text()
    for arm, want in ga.ARMS.items():
        for key, table in (
            ("policy.onnx", "ONNX_SHA"),
            ("policy.onnx.data", "DATA_SHA"),
            ("checkpoint", "CKPT_SHA"),
        ):
            block = src[src.index(f"declare -A {table}") :].split(")", 1)[0]
            assert f"[{arm}]={want[key]}" in block
            assert want[key] in pre


def test_runner_walks_only_the_two_preregistered_presets_inside_blocks():
    src = RUNNER.read_text()
    assert "ORDER=(A B B A B A)" in src
    assert re.search(r"for slot in G vx03_4s yaw05_3s; do", src)
    walk_calls = re.findall(r'run_harness "\$arm" walk [^\n]*', src)
    assert len(walk_calls) == 1 and 'W "$slot"' in walk_calls[0]
    assert "vx05" not in src and "vxm03" not in src
    assert '[[ "$stage" =~ ^[ABCDEFG]$ ]]' in src  # the ladder can never run W
    # never pre-answers a LIVE phrase and never switches rehearsal on (it only records it)
    assert "PHOENIX_CONFIRM" not in src
    assert not re.search(r"PHOENIX_REHEARSAL=|export PHOENIX_REHEARSAL", src)


def test_runner_refuses_without_a_campaign(tmp_path):
    res = subprocess.run(["bash", str(RUNNER), "status"], cwd=REPO, capture_output=True, text=True)
    if (REPO / "logs/gantry_ab/CURRENT").exists():
        pytest.skip("a campaign exists in this checkout")
    assert res.returncode != 0
    assert "no campaign" in res.stderr or "clean checkout" in res.stderr


def _bridge(tmp_path, source, verified, sha="c" * 40):
    arm = ga.ARMS["B"]
    man = {
        "record": "manifest",
        "live": True,
        "code_identity_problems": [],
        "code_identity": {"source": source, "manifest_verified": verified, "sha": sha},
        "artifacts": {
            k: {"sha256": arm[k]} for k in ("checkpoint", "policy.onnx", "policy.onnx.data")
        },
    }
    tick = {"record": "tick", "t_mono_ns": 0, "mode": "silent", "faults": []}
    d = tmp_path / source
    d.mkdir()
    (d / "bridge.jsonl").write_text("\n".join(json.dumps(r) for r in (man, tick)) + "\n")
    return d


def test_rehearsal_and_foreign_commit_runs_are_invalid(tmp_path):
    rehearsal = ga.analyze_run(
        {
            "run_dir": str(_bridge(tmp_path, "git", None)),
            "arm": "B",
            "slot": "stop_drill",
            "block": 1,
        }
    )
    assert not rehearsal["valid"]
    assert any("not a verified payload run" in w for w in rehearsal["why"])
    d = _bridge(tmp_path, "payload_sync", True)
    other = ga.analyze_run(
        {
            "run_dir": str(d),
            "arm": "B",
            "slot": "stop_drill",
            "block": 1,
            "campaign_commit": "d" * 40,
        }
    )
    assert any("not the campaign commit" in w for w in other["why"])
    flagged = ga.analyze_run(
        {"run_dir": str(d), "arm": "B", "slot": "stop_drill", "block": 1, "rehearsal": True}
    )
    assert any("PHOENIX_REHEARSAL" in w for w in flagged["why"])
