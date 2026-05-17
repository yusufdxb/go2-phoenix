#!/usr/bin/env python3
"""Sim-side stress-test sweep runner for Phoenix.

Reads a sweep spec YAML (e.g. configs/train/sweep_stand_v3_stress.yaml),
expands the axes into a Cartesian grid of cells, materializes one
per-cell train + env config under configs/_generated/sweep_<ts>/, and
optionally invokes the existing training entry point
(phoenix.training.ppo_runner via scripts/train.sh) for each cell.

Cell metrics, when an eval pass is configured, are appended to
logs/sweeps/<ts>/leaderboard.csv.

This module has no module-level torch / isaaclab import on purpose: it
must run on the no-sim CI for tests. The training subprocess is the
thing that needs Isaac Lab; this runner only reads/writes YAML + CSV
and exec's other shells.

Usage:
    python scripts/sweep_run.py --spec configs/train/sweep_stand_v3_stress.yaml --dry-run
    python scripts/sweep_run.py --spec configs/train/sweep_stand_v3_stress.yaml --smoke --limit 1
    python scripts/sweep_run.py --spec configs/train/sweep_stand_v3_stress.yaml --scale full
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("phoenix.sweep_run")

REPO_ROOT = Path(__file__).resolve().parent.parent

LEADERBOARD_FIELDS = [
    "cell_id",
    "cell_index",
    "friction_lo",
    "friction_hi",
    "push_velocity_xy",
    "action_rate",
    "num_envs",
    "max_iterations",
    "scale",
    "status",
    "survival_rate",
    "mean_episode_len_s",
    "slew_saturation_pct",
    "mean_ang_vel_err",
    "fall_count",
    "wall_time_s",
    "checkpoint_dir",
    "metrics_path",
    "notes",
]


# ----------------------------------------------------------------------------
# Spec parsing
# ----------------------------------------------------------------------------


@dataclass
class AxisSpec:
    name: str
    field: str
    values: list[Any]
    enable_field: str | None = None


@dataclass
class ScaleSpec:
    name: str
    num_envs: int
    max_iterations: int
    eval: bool


@dataclass
class SweepSpec:
    name: str
    base_train_config: Path
    base_env_config: Path
    resume: Path | None
    axes: list[AxisSpec]
    scales: dict[str, ScaleSpec]
    raw: dict[str, Any] = field(default_factory=dict)


def load_spec(path: Path) -> SweepSpec:
    """Parse a sweep spec YAML into a SweepSpec object.

    Raises ValueError on a malformed spec (missing required fields,
    empty axis, unknown types). The runner refuses to materialize cells
    from a half-valid spec to avoid silent partial sweeps.
    """
    if not path.exists():
        raise FileNotFoundError(f"sweep spec not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"sweep spec must be a mapping, got {type(raw).__name__}")

    for key in ("name", "base_train_config", "base_env_config", "axes", "scales"):
        if key not in raw:
            raise ValueError(f"sweep spec missing required key: {key}")

    axes_raw = raw["axes"]
    if not isinstance(axes_raw, dict) or not axes_raw:
        raise ValueError("sweep spec.axes must be a non-empty mapping")
    axes: list[AxisSpec] = []
    for axis_name, body in axes_raw.items():
        if not isinstance(body, dict):
            raise ValueError(f"axis {axis_name!r} must be a mapping")
        if "field" not in body or "values" not in body:
            raise ValueError(f"axis {axis_name!r} requires 'field' and 'values'")
        values = body["values"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"axis {axis_name!r}.values must be a non-empty list")
        axes.append(
            AxisSpec(
                name=axis_name,
                field=str(body["field"]),
                values=list(values),
                enable_field=body.get("enable_field"),
            )
        )

    scales_raw = raw["scales"]
    if not isinstance(scales_raw, dict) or not scales_raw:
        raise ValueError("sweep spec.scales must be a non-empty mapping")
    scales: dict[str, ScaleSpec] = {}
    for scale_name, body in scales_raw.items():
        if not isinstance(body, dict):
            raise ValueError(f"scale {scale_name!r} must be a mapping")
        for k in ("num_envs", "max_iterations"):
            if k not in body:
                raise ValueError(f"scale {scale_name!r} missing {k}")
        scales[scale_name] = ScaleSpec(
            name=scale_name,
            num_envs=int(body["num_envs"]),
            max_iterations=int(body["max_iterations"]),
            eval=bool(body.get("eval", False)),
        )

    return SweepSpec(
        name=str(raw["name"]),
        base_train_config=Path(str(raw["base_train_config"])),
        base_env_config=Path(str(raw["base_env_config"])),
        resume=Path(str(raw["resume"])) if raw.get("resume") else None,
        axes=axes,
        scales=scales,
        raw=raw,
    )


# ----------------------------------------------------------------------------
# Cell expansion
# ----------------------------------------------------------------------------


@dataclass
class Cell:
    index: int
    axis_values: dict[str, Any]  # axis_name -> value
    fields: dict[str, Any]  # dotted-field -> value (with enable_field flips applied)

    @property
    def id(self) -> str:
        return f"cell_{self.index:03d}"

    def label(self) -> str:
        """Human-friendly label used in filenames and logs."""
        parts = [self.id]
        for axis_name, value in self.axis_values.items():
            parts.append(f"{axis_name}={_compact(value)}")
        return "__".join(parts)


def _compact(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "-".join(_compact(v) for v in value)
    if isinstance(value, float):
        # Use a short form: -2.0 -> -2p0, 0.05 -> 0p05
        s = f"{value:g}"
        return s.replace(".", "p").replace("-", "neg")
    return str(value)


def expand_cells(spec: SweepSpec) -> list[Cell]:
    """Cartesian expansion of axes -> list of Cells in deterministic order."""
    axis_names = [a.name for a in spec.axes]
    value_lists = [a.values for a in spec.axes]
    field_lookup = {a.name: a for a in spec.axes}

    cells: list[Cell] = []
    for idx, combo in enumerate(itertools.product(*value_lists)):
        axis_values: dict[str, Any] = dict(zip(axis_names, combo, strict=True))
        fields: dict[str, Any] = {}
        for axis_name, value in axis_values.items():
            axis = field_lookup[axis_name]
            fields[axis.field] = value
            if axis.enable_field is not None:
                # Convention: a non-zero/non-empty value enables the feature.
                enable = bool(value) if not isinstance(value, (list, tuple)) else len(value) > 0
                # For numeric axes (push magnitude) treat > 0 as enable.
                if isinstance(value, (int, float)):
                    enable = value > 0
                fields[axis.enable_field] = enable
        cells.append(Cell(index=idx, axis_values=axis_values, fields=fields))
    return cells


# ----------------------------------------------------------------------------
# Config materialization
# ----------------------------------------------------------------------------


def _set_dotted(tree: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = tree
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def materialize_cell_configs(
    cell: Cell,
    spec: SweepSpec,
    scale: ScaleSpec,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Write env + train YAML for one cell into out_dir. Returns (env_path, train_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Env cell config: inherits from base_env_config via defaults: [<stem>].
    # The layered loader resolves defaults relative to the env file's parent;
    # since we write into configs/_generated/sweep_<ts>/, we need a relative
    # ../../env/<stem> reference for the loader to find the base.
    env_overrides: dict[str, Any] = {}
    train_overrides: dict[str, Any] = {}
    # Heuristic: env-side fields are anything other than the reward.* and
    # the run.*/algorithm.* train fields. We treat any dotted-field that
    # starts with reward. as env-side (reward lives in env yaml in Phoenix).
    # Train-side overrides come from the scale (num_envs, max_iterations).
    for dotted, value in cell.fields.items():
        _set_dotted(env_overrides, dotted, value)

    env_relative_default = _relative_default(out_dir, spec.base_env_config)
    env_yaml = {"defaults": [env_relative_default], **env_overrides}
    env_path = out_dir / f"{cell.label()}__env.yaml"
    env_path.write_text(yaml.safe_dump(env_yaml, sort_keys=False))

    # Train cell config: inherits from base_train_config, points env.config
    # at the freshly written env file, and applies scale overrides.
    train_relative_default = _relative_default(out_dir, spec.base_train_config)
    train_yaml: dict[str, Any] = {
        "defaults": [train_relative_default],
        "run": {
            "name": f"sweep_{spec.name}/{cell.label()}",
            "max_iterations": scale.max_iterations,
            "_cli_num_envs": scale.num_envs,
        },
        "env": {"config": _safe_relpath(env_path, REPO_ROOT)},
    }
    train_path = out_dir / f"{cell.label()}__train.yaml"
    train_path.write_text(yaml.safe_dump(train_yaml, sort_keys=False))

    return env_path, train_path


def _safe_relpath(target: Path, anchor: Path) -> str:
    """Like Path.relative_to(anchor) but falls back to absolute when target is outside anchor."""
    try:
        return str(target.resolve().relative_to(anchor.resolve()))
    except ValueError:
        return str(target.resolve())


def _relative_default(out_dir: Path, base_config: Path) -> str:
    """Return a defaults: stem that the layered loader can resolve.

    The loader joins (current_file.parent / f"{entry}.yaml"). To inherit
    from a YAML in a different directory, the stem must be a relative
    path like '../../env/stand_v3' (without the .yaml suffix).
    """
    # Resolve relative to out_dir, strip .yaml.
    base_abs = (REPO_ROOT / base_config).resolve()
    rel = os.path.relpath(base_abs.with_suffix(""), out_dir.resolve())
    return rel


# ----------------------------------------------------------------------------
# Leaderboard
# ----------------------------------------------------------------------------


def init_leaderboard(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS)
        w.writeheader()


def append_leaderboard_row(path: Path, row: dict[str, Any]) -> None:
    # Fill missing keys with empty string so writerow doesn't raise.
    filled = {k: row.get(k, "") for k in LEADERBOARD_FIELDS}
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS)
        w.writerow(filled)


def cell_to_row_base(cell: Cell, scale: ScaleSpec) -> dict[str, Any]:
    fr = cell.axis_values.get("friction_range")
    if isinstance(fr, (list, tuple)) and len(fr) == 2:
        fr_lo, fr_hi = fr
    else:
        fr_lo = fr_hi = ""
    return {
        "cell_id": cell.id,
        "cell_index": cell.index,
        "friction_lo": fr_lo,
        "friction_hi": fr_hi,
        "push_velocity_xy": cell.axis_values.get("push_velocity_xy", ""),
        "action_rate": cell.axis_values.get("action_rate", ""),
        "num_envs": scale.num_envs,
        "max_iterations": scale.max_iterations,
        "scale": scale.name,
    }


# ----------------------------------------------------------------------------
# Cell execution
# ----------------------------------------------------------------------------


def run_cell(
    cell: Cell,
    spec: SweepSpec,
    scale: ScaleSpec,
    train_path: Path,
    leaderboard_path: Path,
    timeout_s: int,
) -> dict[str, Any]:
    """Invoke scripts/train.sh for one cell. Returns the leaderboard row.

    The training subprocess writes its own log dir under
    checkpoints/<run_name>/<stamp>/. We capture wall time and let
    train.sh handle the actual training; any post-train eval is a
    separate step (not yet wired here — TODO when the FULL scale lands).
    """
    import time

    cmd: list[str] = [
        str(REPO_ROOT / "scripts" / "train.sh"),
        str(train_path.relative_to(REPO_ROOT)),
        "--headless",
        "--num-envs",
        str(scale.num_envs),
        "--max-iterations",
        str(scale.max_iterations),
    ]
    if spec.resume is not None:
        cmd += ["--resume", str(spec.resume)]

    logger.info("cell %s : %s", cell.id, " ".join(shlex.quote(c) for c in cmd))
    row = cell_to_row_base(cell, scale)
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=False,
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
        wall = time.time() - start
        row["wall_time_s"] = f"{wall:.1f}"
        if result.returncode == 0:
            row["status"] = "ok"
        else:
            row["status"] = f"fail_rc{result.returncode}"
            row["notes"] = (result.stderr or "")[-500:].replace("\n", " | ")
    except subprocess.TimeoutExpired:
        row["wall_time_s"] = f"{timeout_s}"
        row["status"] = "timeout"
        row["notes"] = f"exceeded {timeout_s}s"
    except FileNotFoundError as err:
        row["status"] = "no_train_sh"
        row["notes"] = str(err)

    append_leaderboard_row(leaderboard_path, row)
    return row


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phoenix sim sweep runner")
    p.add_argument("--spec", type=Path, required=True, help="sweep spec YAML")
    p.add_argument(
        "--scale", choices=("smoke", "full"), default="smoke",
        help="scale preset from the spec",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="alias for --scale smoke (kept for ergonomics)",
    )
    p.add_argument("--dry-run", action="store_true", help="print cells, do not execute")
    p.add_argument("--limit", type=int, default=None, help="run only the first N cells")
    p.add_argument(
        "--out-root", type=Path, default=REPO_ROOT / "logs" / "sweeps",
        help="leaderboard + logs output root",
    )
    p.add_argument(
        "--config-root", type=Path, default=REPO_ROOT / "configs" / "_generated",
        help="root for materialized per-cell configs",
    )
    p.add_argument(
        "--timeout-s", type=int, default=60 * 60,
        help="per-cell timeout (default 1 h)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    args = parse_args(argv)
    if args.smoke:
        args.scale = "smoke"

    spec = load_spec(args.spec)
    if args.scale not in spec.scales:
        raise SystemExit(f"scale {args.scale!r} not in spec.scales {list(spec.scales)}")
    scale = spec.scales[args.scale]

    cells = expand_cells(spec)
    if args.limit is not None:
        cells = cells[: args.limit]

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg_dir = args.config_root / f"sweep_{stamp}"
    log_dir = args.out_root / stamp
    leaderboard_path = log_dir / "leaderboard.csv"

    print(f"[sweep] spec        : {args.spec}")
    print(f"[sweep] name        : {spec.name}")
    print(f"[sweep] scale       : {scale.name} (num_envs={scale.num_envs}, iters={scale.max_iterations})")
    print(f"[sweep] cells       : {len(cells)}")
    print(f"[sweep] cfg_dir     : {cfg_dir}")
    print(f"[sweep] leaderboard : {leaderboard_path}")
    print(f"[sweep] dry-run     : {args.dry_run}")
    print("")

    # Always materialize configs (cheap, and the dry-run wants real files
    # to prove the materialization path works).
    cfg_dir.mkdir(parents=True, exist_ok=True)
    materialized: list[tuple[Cell, Path]] = []
    for cell in cells:
        env_path, train_path = materialize_cell_configs(cell, spec, scale, cfg_dir)
        materialized.append((cell, train_path))
        print(f"  [{cell.index:3d}] {cell.label()}")
        for f_dotted, v in cell.fields.items():
            print(f"        {f_dotted} = {v}")

    if args.dry_run:
        # Also dump a JSON summary the test suite can pick up.
        summary_path = cfg_dir / "_dry_run_summary.json"
        summary = {
            "spec": str(args.spec),
            "scale": scale.name,
            "cell_count": len(cells),
            "cells": [
                {
                    "index": c.index,
                    "id": c.id,
                    "label": c.label(),
                    "axis_values": _jsonable(c.axis_values),
                    "fields": _jsonable(c.fields),
                }
                for c in cells
            ],
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\n[sweep] dry-run complete; summary at {summary_path}")
        return 0

    # Real run: init leaderboard, execute each cell.
    init_leaderboard(leaderboard_path)
    for cell, train_path in materialized:
        run_cell(cell, spec, scale, train_path, leaderboard_path, args.timeout_s)

    print(f"\n[sweep] done; leaderboard at {leaderboard_path}")
    return 0


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


if __name__ == "__main__":
    sys.exit(main())
