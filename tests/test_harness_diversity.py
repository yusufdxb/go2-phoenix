"""Self-contained tests for scripts/harness_diversity.py.

Runs without ROS2 / Isaac / torch. Uses pyarrow (already a phoenix dep) to
build synthetic trajectory parquets that satisfy the schema enforced by
``phoenix.real_world.trajectory_logger``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "harness_diversity.py"


def _load_module():
    # The script is not a package; load it by path.
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("harness_diversity", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_parquet(
    path: Path,
    *,
    rows: int,
    tip_at: int | None,
) -> None:
    """Write a minimal parquet matching the recorder schema.

    If ``tip_at`` is set, base_quat rotates past the pitch threshold from that
    step onwards so the detector fires an ATTITUDE event.
    """
    base_quat = []
    base_pos = []
    for i in range(rows):
        if tip_at is not None and i >= tip_at:
            half = 1.0  # pitch ~57 deg, well past 0.8 rad threshold
            q = [0.0, float(np.sin(half / 2)), 0.0, float(np.cos(half / 2))]
        else:
            q = [0.0, 0.0, 0.0, 1.0]
        base_quat.append(q)
        base_pos.append([0.0, 0.0, 0.35])

    def vec(n):
        return [[0.0] * n for _ in range(rows)]

    table = pa.table(
        {
            "step": list(range(rows)),
            "timestamp_s": [i * 0.02 for i in range(rows)],
            "base_pos": base_pos,
            "base_quat": base_quat,
            "base_lin_vel_body": vec(3),
            "base_ang_vel_body": vec(3),
            "joint_pos": vec(12),
            "joint_vel": vec(12),
            "command_vel": vec(3),
            "action": vec(12),
            "contact_forces": vec(4),
            "failure_flag": [False] * rows,
            "failure_mode": [None] * rows,
        }
    )
    pq.write_table(table, path)


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def test_mode_from_name(mod):
    f = mod._mode_from_name
    assert f("pqa_slip_001") == "slip"
    assert f("pqa_push_lat_001") == "push_lat"
    assert f("pqa_ramp_dn_0042") == "ramp_dn"
    assert f("pqa_deadman_freeze_999") == "deadman_freeze"
    # Gate7 parquets should NOT be picked up under default prefix
    assert f("gate7_floor_001") is None
    # malformed
    assert f("noprefix") is None


def test_diversity_pass(tmp_path, mod):
    modes = ["slip", "push_lat", "ramp_dn", "yaw_shove"]
    for m in modes:
        for rep in range(2):
            _write_parquet(tmp_path / f"pqa_{m}_{rep:04d}.parquet", rows=400, tip_at=250)
    summary = mod.scan(tmp_path, include_prefixes=["pqa_"], min_rows=200)
    ok, fails = mod.evaluate(summary, min_modes=4, min_replicates=2)
    assert ok, f"expected PASS got fails={fails}"
    assert len(summary["modes"]) == 4
    for m in modes:
        assert len(summary["modes"][m]) == 2


def test_diversity_fails_on_too_few_modes(tmp_path, mod):
    # only 3 modes
    for m in ["slip", "push_lat", "ramp_dn"]:
        for rep in range(2):
            _write_parquet(tmp_path / f"pqa_{m}_{rep:04d}.parquet", rows=400, tip_at=250)
    summary = mod.scan(tmp_path, include_prefixes=["pqa_"], min_rows=200)
    ok, fails = mod.evaluate(summary, min_modes=4, min_replicates=2)
    assert not ok
    assert any("fewer than 4 distinct modes" in f for f in fails)


def test_diversity_fails_on_short_parquet(tmp_path, mod):
    # 4 modes but one replicate is too short
    modes = ["slip", "push_lat", "ramp_dn", "yaw_shove"]
    for m in modes:
        _write_parquet(tmp_path / f"pqa_{m}_0000.parquet", rows=400, tip_at=250)
        _write_parquet(tmp_path / f"pqa_{m}_0001.parquet", rows=400, tip_at=250)
    # corrupt one
    _write_parquet(tmp_path / "pqa_slip_0001.parquet", rows=100, tip_at=50)
    summary = mod.scan(tmp_path, include_prefixes=["pqa_"], min_rows=200)
    ok, fails = mod.evaluate(summary, min_modes=4, min_replicates=2)
    assert not ok
    assert any("rows 100" in f or "rows" in f for f in fails)


def test_diversity_fails_on_silent_parquet(tmp_path, mod):
    # Detector finds no events (tip_at=None) -> events_ok False
    modes = ["slip", "push_lat", "ramp_dn", "yaw_shove"]
    for m in modes:
        for rep in range(2):
            _write_parquet(tmp_path / f"pqa_{m}_{rep:04d}.parquet", rows=400, tip_at=None)
    summary = mod.scan(tmp_path, include_prefixes=["pqa_"], min_rows=200)
    ok, fails = mod.evaluate(summary, min_modes=4, min_replicates=2)
    assert not ok
    assert any("no detector events" in f for f in fails)


def test_self_test_passes(mod):
    rc = mod.self_test()
    assert rc == 0


def test_gate7_parquets_excluded(tmp_path, mod):
    # Gate7 parquets should be ignored under default prefix; only pqa_ counts
    _write_parquet(tmp_path / "gate7_floor_001.parquet", rows=400, tip_at=None)
    modes = ["slip", "push_lat", "ramp_dn", "yaw_shove"]
    for m in modes:
        for rep in range(2):
            _write_parquet(tmp_path / f"pqa_{m}_{rep:04d}.parquet", rows=400, tip_at=250)
    summary = mod.scan(tmp_path, include_prefixes=["pqa_"], min_rows=200)
    # gate7 should not appear in either files (check basename, not full path
    # since pytest's tmp_path name itself may contain "gate7") or modes
    assert all(not Path(r["path"]).name.startswith("gate7") for r in summary["files"])
    assert "floor" not in summary["modes"]
