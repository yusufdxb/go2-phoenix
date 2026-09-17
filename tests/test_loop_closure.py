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

#: The seed the staged variations config declares. The held-out arm must use a
#: different one or its perturbation points are the training pool's.
TRAIN_VARIATION_SEED = 1234

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
    policy=""
    varseed=""
    traj=""
    for i in "${!ARGS[@]}"; do
        [[ "${ARGS[$i]}" == "--output-dir" ]] && outdir="${ARGS[$((i + 1))]}"
        [[ "${ARGS[$i]}" == "--policy" ]] && policy="${ARGS[$((i + 1))]}"
        [[ "${ARGS[$i]}" == "--variation-seed" ]] && varseed="${ARGS[$((i + 1))]}"
        [[ "${ARGS[$i]}" == "--trajectory" ]] && traj="${ARGS[$((i + 1))]}"
    done
    # The real entry point now REQUIRES a policy (or an explicit opt-in to the
    # zero-action diagnostic). A stub that accepted a bare call would hide a
    # regression where loop_closure stops driving the replay.
    if [[ -z "$policy" ]]; then
        echo "[stub] reconstruct called without --policy" >&2
        exit 2
    fi
    printf '%s\n' "traj=$traj policy=$policy varseed=$varseed outdir=$outdir" >> reconstruct_calls.log
    mkdir -p "$outdir"
    if [[ -n "$varseed" ]]; then
        # Held-out arm.
        n=${STUB_HELDOUT_VARIANTS-4}
        failed=${STUB_HELDOUT_FAILED-1}
        printf '{"controller": "policy_driven_replay", "variation_seed": %s}\n' "$varseed" \
            > "$outdir/replay_summary.json"
        printf '{"variants_written": %s, "variants_with_failure": %s, "variants": []}\n' \
            "$n" "$failed" > "$outdir/variants_index.json"
        echo "[stub] heldout replay $n variants, $failed failed"
        exit 0
    fi
    echo '{"controller": "policy_driven_replay"}' > "$outdir/replay_summary.json"
    for ((v = 0; v < ${STUB_VARIANTS-0}; v++)); do
        if [[ -n "${STUB_VARIANT_FROM-}" ]]; then
            cp "${STUB_VARIANT_FROM}" "$outdir/variant_${v}.parquet"
        else
            echo "variant$v" > "$outdir/variant_${v}.parquet"
        fi
    done
    printf '{"variants_written": %s, "variants_with_failure": 0, "variants": []}\n' \
        "${STUB_VARIANTS-0}" > "$outdir/variants_index.json"
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

    # The locked H25 stand deliverable, which is what the hardware gates run.
    ckpt = repo / "checkpoints" / "phoenix-stand-h25-lat-noise" / "2026-06-22_21-08-20"
    ckpt.mkdir(parents=True)
    (ckpt / "model_799.pt").write_text("stub")

    for cfg in (
        "configs/env/stand_v3_h25.yaml",
        "configs/train/adaptation_loop_closure.yaml",
    ):
        path = repo / cfg
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")

    # The script reads the training variation seed out of this file so it can
    # refuse a held-out arm that would draw the same Halton points.
    variations = repo / "configs" / "replay" / "variations.yaml"
    variations.parent.mkdir(parents=True, exist_ok=True)
    variations.write_text(f"variations:\n  seed: {TRAIN_VARIATION_SEED}\n  per_trajectory: 4\n")

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


def _report(repo) -> str:
    return next((repo / "docs").glob("loop_closure_*")).joinpath("report.md").read_text()


# ------------------------------------------------ the replay is actually driven
def test_the_replay_stage_drives_the_baseline_policy(staged):
    """A zero-action replay reproduces nothing; the stage must pass --policy."""
    repo, _ = staged
    res = _run(staged, STUB_VARIANTS=2)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    calls = (repo / "reconstruct_calls.log").read_text().splitlines()
    # The first replay is the training one: driven by the baseline policy.
    assert "policy=checkpoints/phoenix-stand-h25-lat-noise" in calls[0]
    assert "traj=data/failures/train.parquet" in calls[0]


def test_the_locked_h25_stand_is_the_default_baseline(staged):
    """The script used to adapt walking v3b on flat.yaml, not what ships."""
    repo, _ = staged
    res = _run(staged, STUB_VARIANTS=1)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    report = _report(repo)
    assert "phoenix-stand-h25-lat-noise" in report
    assert "configs/env/stand_v3_h25.yaml" in report
    evals = (repo / "eval_calls.log").read_text()
    assert "configs/env/stand_v3_h25.yaml" in evals
    assert "flat.yaml" not in evals


# --------------------------------------------------------- the held-out arm
def test_the_heldout_arm_runs_and_is_reported(staged):
    repo, _ = staged
    res = _run(staged, STUB_VARIANTS=2, STUB_HELDOUT_VARIANTS=4, STUB_HELDOUT_FAILED=1)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    report = _report(repo)
    assert "## Held-out arm" in report
    assert "held-out failure rate" in report
    assert "0.2500" in report  # 1 of 4 variants failed
    # One held-out replay per policy: baseline plus one per training seed.
    calls = (repo / "reconstruct_calls.log").read_text().splitlines()
    heldout_calls = [c for c in calls if "traj=data/failures/heldout.parquet" in c]
    assert len(heldout_calls) == 4


def test_the_heldout_arm_uses_disjoint_halton_points(staged):
    repo, _ = staged
    res = _run(staged, STUB_VARIANTS=1)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    calls = (repo / "reconstruct_calls.log").read_text().splitlines()
    heldout = [c for c in calls if "traj=data/failures/heldout.parquet" in c]
    assert heldout, "no held-out replay ran"
    for call in heldout:
        assert "varseed=20260917" in call
        assert f"varseed={TRAIN_VARIATION_SEED}" not in call


def test_reusing_the_training_variation_seed_is_refused(staged):
    """Same Halton points as training means the arm is not held out."""
    res = _run(staged, ("--heldout-variation-seed", str(TRAIN_VARIATION_SEED)), STUB_VARIANTS=1)
    assert res.returncode != 0
    assert "the SAME Halton points training saw" in res.stderr


def test_a_failed_heldout_replay_is_a_hard_stop(staged):
    # STUB_HELDOUT_VARIANTS=0 is what a refused hardware-capture seed looks
    # like from here: nothing to measure, so the arm must not report a rate.
    res = _run(staged, STUB_VARIANTS=1, STUB_HELDOUT_VARIANTS=0)
    assert res.returncode != 0
    assert "produced 0 variants" in res.stderr


def test_skipping_the_heldout_arm_is_recorded_and_cannot_pass(staged):
    repo, _ = staged
    res = _run(staged, ("--skip-heldout",), STUB_VARIANTS=1)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    report = _report(repo)
    assert "held-out ARM was SKIPPED" in report
    assert "no held-out evidence and no generalization claim" in report


# ------------------------------------------------- the decision cannot cherry-pick
def test_the_decision_rule_is_not_any_seed_that_improved(staged):
    repo, _ = staged
    res = _run(staged, STUB_VARIANTS=1)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    report = _report(repo)
    decision = report.split("## Decision")[1]
    assert "MEAN adapted success_rate across ALL" in decision
    assert "pre-declared" in decision.lower()
    assert "cherry-picking" in decision
    # The old rule, and the old "pick best seed" next step, must be gone.
    assert "passes if ANY adapted seed" not in report
    assert "pick best seed" not in report
    assert "Do NOT pick the best" in report


def test_the_report_still_states_what_the_run_is_not(staged):
    repo, _ = staged
    res = _run(staged, STUB_VARIANTS=1)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    report = _report(repo)
    assert "Sim-only" in report
    assert "correlated copies of ONE captured state" in report or "copies of ONE" in report
    assert "No fresh post-training hardware trial" in report
