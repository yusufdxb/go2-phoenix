"""End-to-end offline loop on SYNTHETIC telemetry: calibrate -> assess -> condition.

Synthetic files only exercise the plumbing; they are not evidence about the GO2.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

from .test_monitor import records

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("phoenix_loop", ROOT / "scripts" / "phoenix_loop.py")
loop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loop)  # type: ignore[union-attr]


def _jsonl(path: Path, recs) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return path


def test_calibrate_assess_condition_round_trip(tmp_path):
    nominal = [_jsonl(tmp_path / f"nom{i}.jsonl", records(1500, seed=i)) for i in range(2)]
    base = tmp_path / "baseline.json"
    assert (
        loop.main(["calibrate", *map(str, nominal), "--regime", "stand", "--out", str(base)]) == 0
    )

    s = np.ones(12)
    s[8] = 0.6  # Unitree motor order index 8 = RR_calf
    deg = _jsonl(tmp_path / "deg.jsonl", records(800, s=s, seed=5))
    health = tmp_path / "health.json"
    assert (
        loop.main(
            ["assess", str(deg), "--baseline", str(base), "--regime", "stand", "--out", str(health)]
        )
        == 0
    )
    doc = json.loads(health.read_text())
    states = {h["joint"]: h["state"] for h in doc["health"]}
    assert states["RR_calf_joint"] == "DEGRADED"
    assert sum(v == "DEGRADED" for v in states.values()) == 1

    env_dir = tmp_path / "configs" / "env" / "conditioned"
    overlay = env_dir / "rr_calf.yaml"
    assert (
        loop.main(
            ["condition", str(health), "--parent-env", "../stand_v3_h25", "--out", str(overlay)]
        )
        == 0
    )
    ov = yaml.safe_load(overlay.read_text())
    lo, hi = ov["domain_randomization"]["targeted_actuator"]["joints"]["RR_calf_joint"]
    assert lo < 0.6 < hi <= 1.0
    assert ov["defaults"] == ["../stand_v3_h25"]


def test_assess_refuses_other_regime(tmp_path):
    nominal = [_jsonl(tmp_path / f"n{i}.jsonl", records(1500, seed=i)) for i in range(2)]
    base = tmp_path / "b.json"
    loop.main(["calibrate", *map(str, nominal), "--regime", "stand", "--out", str(base)])
    assert loop.main(["assess", str(nominal[0]), "--baseline", str(base), "--regime", "walk"]) == 2


def test_calibrate_refuses_low_fidelity_run(tmp_path):
    bad = _jsonl(tmp_path / "short.jsonl", records(100))  # < 10 s of authority
    assert (
        loop.main(["calibrate", str(bad), "--regime", "stand", "--out", str(tmp_path / "b")]) == 2
    )


def test_condition_refuses_when_nothing_degraded(tmp_path):
    nominal = [_jsonl(tmp_path / f"n{i}.jsonl", records(1500, seed=i)) for i in range(2)]
    base = tmp_path / "b.json"
    loop.main(["calibrate", *map(str, nominal), "--regime", "stand", "--out", str(base)])
    health = tmp_path / "h.json"
    loop.main(
        [
            "assess",
            str(nominal[1]),
            "--baseline",
            str(base),
            "--regime",
            "stand",
            "--out",
            str(health),
        ]
    )
    assert (
        loop.main(
            ["condition", str(health), "--parent-env", "x", "--out", str(tmp_path / "o.yaml")]
        )
        == 3
    )
