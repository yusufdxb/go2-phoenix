"""Tests for scripts/loop_closure.sh.

The script orchestrates Isaac Lab runs, so the test stands a stub ``python3``
in front of it on PATH and drives the orchestration logic only. What is under
test is what the audit found wrong: three "independent seeds" that were never
passed to fine_tune, an augmentation step whose zero-file outcome read as
success, a held-out parquet that was named but never used, and no assertion
that the runs differed at all.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "loop_closure.sh"

STUB_PYTHON = r"""#!/usr/bin/env bash
# Stub python3 for loop_closure.sh tests. Real interpreter for everything the
# script needs to actually compute; canned behaviour for the Isaac entry points.
set -euo pipefail
REAL_PY="__REAL_PY__"
if [[ "${1:-}" == "-c" || "${1:-}" == "-" ]]; then
    exec "$REAL_PY" "$@"
fi
if [[ "${1:-}" != "-m" ]]; then
    exec "$REAL_PY" "$@"
fi
MODULE="$2"
shift 2
ARGS=("$@")

case "$MODULE" in
phoenix.adaptation.fine_tune)
    for a in "${ARGS[@]}"; do
        if [[ "$a" == "--help" ]]; then
            echo "usage: fine_tune [--config CONFIG] [--resume RESUME]"
            echo "  --trajectory-dir TRAJECTORY_DIR"
            for flag in ${STUB_FT_FLAGS-"--seed --curriculum-seed"}; do
                echo "  $flag VALUE"
            done
            exit 0
        fi
    done
    seed=""
    curr=""
    for i in "${!ARGS[@]}"; do
        [[ "${ARGS[$i]}" == "--seed" ]] && seed="${ARGS[$((i + 1))]}"
        [[ "${ARGS[$i]}" == "--curriculum-seed" ]] && curr="${ARGS[$((i + 1))]}"
    done
    n=0
    [[ -f .stub_counter ]] && n=$(cat .stub_counter)
    n=$((n + 1))
    echo "$n" > .stub_counter
    run_dir=$(printf "checkpoints/phoenix-adapt-loop-closure/2026-01-01_00-00-%02d" "$n")
    mkdir -p "$run_dir"
    echo "stub" > "$run_dir/model_499.pt"
    if [[ "${STUB_NO_SEEDS_JSON-0}" != "1" ]]; then
        printf '{"training_seed": %s, "curriculum_seed": %s, "config_seed": 42, "resolved_from": "cli"}\n' \
            "${STUB_SEED_OVERRIDE-$seed}" "$curr" > "$run_dir/seeds.json"
    fi
    echo "[stub] fine_tune seed=$seed curriculum_seed=$curr -> $run_dir"
    ;;
phoenix.replay.reconstruct)
    outdir=""
    for i in "${!ARGS[@]}"; do
        [[ "${ARGS[$i]}" == "--output-dir" ]] && outdir="${ARGS[$((i + 1))]}"
    done
    mkdir -p "$outdir"
    echo '{"controller": "zero_action_diagnostic"}' > "$outdir/replay_summary.json"
    for ((v = 0; v < ${STUB_VARIANTS-0}; v++)); do
        if [[ -n "${STUB_VARIANT_FROM-}" ]]; then
            cp "${STUB_VARIANT_FROM}" "$outdir/variant_${v}.parquet"
        else
            echo "variant$v" > "$outdir/variant_${v}.parquet"
        fi
    done
    echo "[stub] reconstruct wrote ${STUB_VARIANTS-0} variants"
    ;;
phoenix.training.evaluate)
    out=""
    for i in "${!ARGS[@]}"; do
        [[ "${ARGS[$i]}" == "--metrics-out" ]] && out="${ARGS[$((i + 1))]}"
    done
    printf '%s\n' "${ARGS[*]}" >> eval_calls.log
    mkdir -p "$(dirname "$out")"
    cat > "$out" <<'JSON'
{"num_episodes": 32, "mean_episode_return": 1.0, "mean_episode_length_s": 20.0,
 "success_rate": 1.0, "failure_rate": 0.0, "mean_lin_vel_error": 0.09,
 "mean_ang_vel_error": 0.08, "per_term_rewards": {},
 "slew_saturation_pct": 0.033, "slew_metric_definition": "deploy_clip_activation_v2",
 "legacy_raw_action_delta_pct": 0.0033}
JSON
    echo "[stub] evaluate -> $out"
    ;;
*)
    exec "$REAL_PY" -m "$MODULE" "${ARGS[@]}"
    ;;
esac
"""


@pytest.fixture
def staged(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, repo / "scripts" / SCRIPT.name)
    (repo / "scripts" / SCRIPT.name).chmod(0o755)
    (repo / "scripts" / "_activate.sh").write_text("# stubbed for tests\n")

    ckpt = repo / "checkpoints" / "phoenix-flat" / "2026-04-16_21-39-16"
    ckpt.mkdir(parents=True)
    (ckpt / "model_999.pt").write_text("stub")

    for cfg in (
        "configs/replay/variations.yaml",
        "configs/env/flat.yaml",
        "configs/train/adaptation_loop_closure.yaml",
    ):
        path = repo / cfg
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")

    failures = repo / "data" / "failures"
    failures.mkdir(parents=True)
    (failures / "train.parquet").write_bytes(b"PAR1train")
    (failures / "heldout.parquet").write_bytes(b"PAR1heldout")

    binroot = tmp_path / "bin"
    binroot.mkdir()
    stub = binroot / "python3"
    stub.write_text(STUB_PYTHON.replace("__REAL_PY__", sys.executable))
    stub.chmod(0o755)
    return repo, binroot


def _run(staged, args=(), **env_extra):
    repo, binroot = staged
    env = os.environ.copy()
    env["PATH"] = f"{binroot}:{env['PATH']}"
    env.pop("VIRTUAL_ENV", None)
    env.update({k: str(v) for k, v in env_extra.items()})
    return subprocess.run(
        [
            "bash",
            "scripts/loop_closure.sh",
            "data/failures/train.parquet",
            "data/failures/heldout.parquet",
            *args,
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_script_exists():
    assert SCRIPT.exists()


def test_happy_path_passes_distinct_seeds_and_records_them(staged):
    repo, _ = staged
    res = _run(staged, STUB_VARIANTS=2)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"

    out_dirs = list((repo / "docs").glob("loop_closure_*"))
    assert len(out_dirs) == 1
    report = (out_dirs[0] / "report.md").read_text()
    manifest = json.loads((out_dirs[0] / "seeds.json").read_text())

    # Three distinct training seeds, each with its own curriculum seed, and one
    # shared evaluation seed, all recorded separately.
    assert [r["training_seed"] for r in manifest["runs"]] == [42, 43, 44]
    assert [r["curriculum_seed"] for r in manifest["runs"]] == [1042, 1043, 1044]
    assert manifest["evaluation_seed"] == 20260911
    assert "| training seed | curriculum RNG seed | evaluation seed |" in report

    # fine_tune was actually given the seeds.
    for seed in (42, 43, 44):
        log = (out_dirs[0] / f"adapt_seed{seed}.log").read_text()
        assert f"seed={seed} curriculum_seed={seed + 1000}" in log

    # Every eval ran at the evaluation seed, not at a training seed.
    eval_calls = (repo / "eval_calls.log").read_text().splitlines()
    assert len(eval_calls) == 4
    assert all("--seed 20260911" in call for call in eval_calls)

    # Pool accounting is real: 1 original + 2 variants.
    assert "Training pool: 3 parquets (1 real + 2 Halton variants)" in report
    assert "Pool augmented: true" in report


def test_fine_tune_without_seed_support_is_a_hard_stop(staged):
    res = _run(staged, STUB_VARIANTS=2, STUB_FT_FLAGS="")
    assert res.returncode != 0
    assert "does not accept --seed --curriculum-seed" in res.stderr
    assert "--curriculum-seed INT" in res.stderr
    assert "seeds.json" in res.stderr


def test_zero_augmentation_fails_loudly(staged):
    res = _run(staged, STUB_VARIANTS=0)
    assert res.returncode != 0
    assert "produced 0 variant trajectories" in res.stderr
    # It must name the real reason rather than blaming the operator.
    assert "does NOT currently emit Parquet variant trajectories" in res.stderr


def test_zero_augmentation_can_be_opted_into_and_is_recorded(staged):
    repo, _ = staged
    res = _run(staged, ("--allow-unaugmented",), STUB_VARIANTS=0)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    report = next((repo / "docs").glob("loop_closure_*")).joinpath("report.md").read_text()
    assert "Pool augmented: false" in report
    assert "Training pool: 1 parquets (1 real + 0 Halton variants)" in report


def test_recorded_seed_mismatch_is_caught(staged):
    res = _run(staged, STUB_VARIANTS=1, STUB_SEED_OVERRIDE=42)
    assert res.returncode != 0
    assert "recorded" in res.stderr and "training_seed=42, asked for 43" in res.stderr


def test_missing_seeds_json_is_caught(staged):
    res = _run(staged, STUB_VARIANTS=1, STUB_NO_SEEDS_JSON=1)
    assert res.returncode != 0
    assert "cannot verify which seed this" in res.stderr


def test_duplicate_requested_seeds_are_rejected_before_any_gpu_time(staged):
    res = _run(staged, ("--seeds", "42 42 43"), STUB_VARIANTS=1)
    assert res.returncode != 0
    assert "is not distinct" in res.stderr


def test_identical_heldout_content_is_rejected(staged):
    repo, _ = staged
    (repo / "data" / "failures" / "heldout.parquet").write_bytes(b"PAR1train")
    res = _run(staged, STUB_VARIANTS=1)
    assert res.returncode != 0
    assert "identical contents" in res.stderr


def test_heldout_leaking_into_the_training_pool_is_caught(staged):
    repo, _ = staged
    res = _run(
        staged,
        STUB_VARIANTS=1,
        STUB_VARIANT_FROM=str(repo / "data" / "failures" / "heldout.parquet"),
    )
    assert res.returncode != 0
    assert "held-out trajectory content found in the training pool" in res.stderr


def test_real_directory_at_the_pool_symlink_path_is_refused(staged):
    repo, _ = staged
    (repo / "data" / "failures" / "loop_closure_train").mkdir()
    res = _run(staged, STUB_VARIANTS=1)
    assert res.returncode != 0
    assert "is not a symlink" in res.stderr


def test_report_does_not_claim_a_heldout_evaluation(staged):
    repo, _ = staged
    res = _run(staged, STUB_VARIANTS=1)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    report = next((repo / "docs").glob("loop_closure_*")).joinpath("report.md").read_text()
    assert "was NOT evaluated" in report
    assert "No held-out scenario evaluation" in report
    assert "held-out scenario" not in report.split("## Held-out parquet")[0].lower()
