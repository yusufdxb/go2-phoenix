"""Tests for scripts/sweep_run.py.

These run on the no-sim CI: no torch, no isaaclab. Phoenix rule per
``feedback_phoenix_lazy_torch.md`` — gate any heavyweight import with
``pytest.importorskip`` if you ever add one. None needed here yet.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_RUN_PATH = REPO_ROOT / "scripts" / "sweep_run.py"


def _load_sweep_run():
    """Import scripts/sweep_run.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("phoenix_sweep_run", SWEEP_RUN_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phoenix_sweep_run"] = mod
    spec.loader.exec_module(mod)
    return mod


sweep_run = _load_sweep_run()


# ----------------------------------------------------------------------------
# Spec parsing
# ----------------------------------------------------------------------------


def _write_minimal_spec(path: Path) -> Path:
    spec = {
        "name": "unit-test-spec",
        "base_train_config": "configs/train/ppo_stand_v3.yaml",
        "base_env_config": "configs/env/stand_v3.yaml",
        "axes": {
            "friction_range": {
                "field": "domain_randomization.friction_range",
                "values": [[0.3, 1.5], [0.05, 0.4]],
            },
            "push_velocity_xy": {
                "field": "perturbation.push_velocity_xy",
                "enable_field": "perturbation.enabled",
                "values": [0.0, 1.0],
            },
            "action_rate": {
                "field": "reward.action_rate",
                "values": [-2.0],
            },
        },
        "scales": {
            "smoke": {"num_envs": 256, "max_iterations": 50, "eval": False},
            "full": {"num_envs": 4096, "max_iterations": 300, "eval": True},
        },
    }
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def test_load_spec_parses_axes_and_scales(tmp_path: Path) -> None:
    spec_path = _write_minimal_spec(tmp_path / "spec.yaml")
    spec = sweep_run.load_spec(spec_path)
    assert spec.name == "unit-test-spec"
    assert len(spec.axes) == 3
    axis_by_name = {a.name: a for a in spec.axes}
    assert axis_by_name["friction_range"].field == "domain_randomization.friction_range"
    assert axis_by_name["push_velocity_xy"].enable_field == "perturbation.enabled"
    assert "smoke" in spec.scales and "full" in spec.scales
    assert spec.scales["smoke"].num_envs == 256
    assert spec.scales["full"].max_iterations == 300


def test_load_spec_rejects_missing_required(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: foo\naxes: {}\n")
    with pytest.raises(ValueError):
        sweep_run.load_spec(bad)


def test_load_spec_rejects_empty_axis_values(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    yaml.safe_dump(
        {
            "name": "x",
            "base_train_config": "a",
            "base_env_config": "b",
            "axes": {"f": {"field": "domain_randomization.friction_range", "values": []}},
            "scales": {"smoke": {"num_envs": 1, "max_iterations": 1}},
        },
        open(bad, "w"),
    )
    with pytest.raises(ValueError, match="non-empty"):
        sweep_run.load_spec(bad)


def test_load_spec_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sweep_run.load_spec(tmp_path / "nope.yaml")


# ----------------------------------------------------------------------------
# Cell expansion
# ----------------------------------------------------------------------------


def test_expand_cells_produces_cartesian_product(tmp_path: Path) -> None:
    spec_path = _write_minimal_spec(tmp_path / "spec.yaml")
    spec = sweep_run.load_spec(spec_path)
    cells = sweep_run.expand_cells(spec)
    # 2 (friction) x 2 (push) x 1 (action_rate) = 4 cells
    assert len(cells) == 4
    # Index is contiguous + unique
    assert [c.index for c in cells] == [0, 1, 2, 3]
    # Push axis enable_field flips automatically when value > 0
    push_zero_cells = [c for c in cells if c.fields["perturbation.push_velocity_xy"] == 0.0]
    push_nonzero_cells = [c for c in cells if c.fields["perturbation.push_velocity_xy"] == 1.0]
    assert all(c.fields["perturbation.enabled"] is False for c in push_zero_cells)
    assert all(c.fields["perturbation.enabled"] is True for c in push_nonzero_cells)


def test_default_grid_has_twelve_cells() -> None:
    spec = sweep_run.load_spec(REPO_ROOT / "configs" / "train" / "sweep_stand_v3_stress.yaml")
    cells = sweep_run.expand_cells(spec)
    assert len(cells) == 12, f"expected 12 cells in default grid, got {len(cells)}"


# ----------------------------------------------------------------------------
# Config materialization
# ----------------------------------------------------------------------------


def test_materialize_writes_valid_yaml(tmp_path: Path) -> None:
    spec_path = _write_minimal_spec(tmp_path / "spec.yaml")
    spec = sweep_run.load_spec(spec_path)
    cells = sweep_run.expand_cells(spec)
    scale = spec.scales["smoke"]
    out_dir = tmp_path / "_generated" / "sweep_test"

    env_path, train_path = sweep_run.materialize_cell_configs(
        cells[0], spec, scale, out_dir,
    )
    assert env_path.exists() and train_path.exists()

    # Both files must be parseable YAML and contain the expected keys
    env_doc = yaml.safe_load(env_path.read_text())
    train_doc = yaml.safe_load(train_path.read_text())

    assert "defaults" in env_doc
    assert "domain_randomization" in env_doc
    assert env_doc["domain_randomization"]["friction_range"] == [0.3, 1.5]
    assert env_doc["reward"]["action_rate"] == -2.0

    assert "defaults" in train_doc
    assert train_doc["run"]["max_iterations"] == scale.max_iterations
    assert train_doc["run"]["_cli_num_envs"] == scale.num_envs
    assert train_doc["env"]["config"].endswith("__env.yaml")


def test_materialize_all_cells_unique_paths(tmp_path: Path) -> None:
    spec_path = _write_minimal_spec(tmp_path / "spec.yaml")
    spec = sweep_run.load_spec(spec_path)
    cells = sweep_run.expand_cells(spec)
    scale = spec.scales["smoke"]
    out_dir = tmp_path / "out"
    paths = set()
    for c in cells:
        env_p, train_p = sweep_run.materialize_cell_configs(c, spec, scale, out_dir)
        paths.add(env_p)
        paths.add(train_p)
    assert len(paths) == 2 * len(cells), "every cell must get unique env + train files"


# ----------------------------------------------------------------------------
# Dry-run end-to-end (CLI surface)
# ----------------------------------------------------------------------------


def test_dry_run_creates_expected_cell_count(tmp_path: Path, capsys) -> None:
    spec_path = _write_minimal_spec(tmp_path / "spec.yaml")
    rc = sweep_run.main(
        [
            "--spec", str(spec_path),
            "--dry-run",
            "--config-root", str(tmp_path / "_generated"),
            "--out-root", str(tmp_path / "logs"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "cells       : 4" in out

    # Locate the generated sweep dir
    gen_dirs = list((tmp_path / "_generated").glob("sweep_*"))
    assert len(gen_dirs) == 1
    summary_path = gen_dirs[0] / "_dry_run_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["cell_count"] == 4
    assert len(summary["cells"]) == 4


def test_dry_run_with_limit(tmp_path: Path, capsys) -> None:
    spec_path = _write_minimal_spec(tmp_path / "spec.yaml")
    rc = sweep_run.main(
        [
            "--spec", str(spec_path),
            "--dry-run",
            "--limit", "2",
            "--config-root", str(tmp_path / "_generated"),
            "--out-root", str(tmp_path / "logs"),
        ]
    )
    assert rc == 0
    gen_dirs = list((tmp_path / "_generated").glob("sweep_*"))
    summary = json.loads((gen_dirs[0] / "_dry_run_summary.json").read_text())
    assert summary["cell_count"] == 2


# ----------------------------------------------------------------------------
# Leaderboard schema
# ----------------------------------------------------------------------------


def test_leaderboard_init_and_append(tmp_path: Path) -> None:
    path = tmp_path / "leaderboard.csv"
    sweep_run.init_leaderboard(path)
    # Header present
    rows = list(csv.reader(path.open()))
    assert rows[0] == sweep_run.LEADERBOARD_FIELDS

    sweep_run.append_leaderboard_row(
        path,
        {
            "cell_id": "cell_000",
            "cell_index": 0,
            "friction_lo": 0.3,
            "friction_hi": 1.5,
            "push_velocity_xy": 0.0,
            "action_rate": -2.0,
            "num_envs": 256,
            "max_iterations": 50,
            "scale": "smoke",
            "status": "ok",
            "survival_rate": 0.98,
            "mean_episode_len_s": 19.6,
            "slew_saturation_pct": 0.004,
            "mean_ang_vel_err": 0.011,
            "fall_count": 0,
            "wall_time_s": 132.5,
        },
    )
    rows = list(csv.reader(path.open()))
    assert len(rows) == 2  # header + 1 row
    body = rows[1]
    assert body[0] == "cell_000"
    assert body[sweep_run.LEADERBOARD_FIELDS.index("status")] == "ok"


def test_leaderboard_fills_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "leaderboard.csv"
    sweep_run.init_leaderboard(path)
    sweep_run.append_leaderboard_row(path, {"cell_id": "cell_001"})
    rows = list(csv.reader(path.open()))
    assert rows[1][0] == "cell_001"
    # All other fields blank, not crashed
    assert all(v == "" for v in rows[1][1:])


# ----------------------------------------------------------------------------
# Cell label / id sanity
# ----------------------------------------------------------------------------


def test_cell_label_is_filesystem_safe(tmp_path: Path) -> None:
    spec_path = _write_minimal_spec(tmp_path / "spec.yaml")
    spec = sweep_run.load_spec(spec_path)
    for cell in sweep_run.expand_cells(spec):
        label = cell.label()
        # No path separators, no whitespace, no shell meta
        assert "/" not in label
        assert " " not in label
        assert "$" not in label
