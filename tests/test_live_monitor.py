"""Live monitor: streaming, one record at a time, agrees with the offline path (synthetic data)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from phoenix.monitor.live import LiveMonitor
from phoenix.monitor.residual import calibrate

from .test_monitor import J, records, stats_of

ROOT = Path(__file__).resolve().parents[1]


def _baseline():
    return calibrate([stats_of(records(1500, seed=i)) for i in range(3)], regime="stand")


def test_streaming_flags_the_injected_joint_after_persistence():
    mon = LiveMonitor(_baseline())
    s = np.ones(12)
    s[J] = 0.6
    reports = [r for r in (mon.feed(rec) for rec in records(12 * 50 + 1, s=s, seed=4)) if r]
    assert len(reports) == 12  # one per 50-tick window; the manifest is ignored
    states = [r[J].state for r in reports]
    assert "DEGRADED" not in states[:7]  # persistence: not before 8 windows
    assert states[-1] == "DEGRADED"
    assert all(h.state != "DEGRADED" for i, h in enumerate(reports[-1]) if i != J)


def test_script_once_writes_one_report_per_window(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "live", ROOT / "scripts" / "phoenix_live_monitor.py"
    )
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)  # type: ignore[union-attr]
    base = tmp_path / "b.json"
    base.write_text(json.dumps(_baseline().to_dict()))
    tele = tmp_path / "bridge.jsonl"
    tele.write_text("\n".join(json.dumps(r) for r in records(5 * 50 + 1)) + "\n")
    out = tmp_path / "h.jsonl"
    assert (
        live.main([str(tele), "--baseline", str(base), "--out", str(out), "--once", "--quiet"]) == 0
    )
    lines = out.read_text().splitlines()
    assert len(lines) == 5 and json.loads(lines[-1])["window"] == 5
