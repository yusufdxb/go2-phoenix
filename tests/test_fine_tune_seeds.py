"""Three seeds must stay three quantities, and the artifact must prove it.

``scripts/loop_closure.sh`` looped over seeds 42/43/44, named its output
directories ``_seed42`` / ``_seed43`` / ``_seed44``, and passed none of them to
``fine_tune``, which read ``cfg["run"]["seed"]``. All three "independent seeds"
were the same run. These tests pin the CLI contract the repaired orchestrator
probes for, and pin the seed record it reads back instead of trusting its own
intent.
"""

from __future__ import annotations

import json

import pytest

from phoenix.adaptation.fine_tune import (
    DEFAULT_CONFIG_SEED,
    SEEDS_ARTIFACT,
    SEEDS_FIELDS,
    parse_args,
    resolve_seeds,
    write_seeds,
)


def _config(seed=7):
    config = {"run": {"name": "adapt", "device": "cuda:0"}, "curriculum": {}}
    if seed is not None:
        config["run"]["seed"] = seed
    return config


# -------------------- CLI contract -----------------------------------------


def test_cli_accepts_both_seeds_and_keeps_them_distinct():
    args = parse_args(
        ["--config", "adapt.yaml", "--seed", "43", "--curriculum-seed", "1043"]
    )
    assert args.seed == 43
    assert args.curriculum_seed == 1043


def test_cli_defaults_are_none_not_zero():
    args = parse_args(["--config", "adapt.yaml"])
    assert args.seed is None
    assert args.curriculum_seed is None


def test_help_advertises_the_flags_the_orchestrator_probes(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--help"])
    help_text = capsys.readouterr().out
    assert "--seed" in help_text
    assert "--curriculum-seed" in help_text


# -------------------- resolution -------------------------------------------


def test_seed_overrides_the_config_value():
    seeds = resolve_seeds(_config(seed=7), seed=43)
    assert seeds["training_seed"] == 43
    assert seeds["config_seed"] == 7
    assert seeds["resolved_from"] == "cli"


def test_curriculum_seed_defaults_to_the_training_seed_never_zero():
    seeds = resolve_seeds(_config(seed=7), seed=43)
    assert seeds["curriculum_seed"] == 43
    # And when the config seeds the run, the curriculum follows the config seed
    # rather than FailureCurriculum's own seed=0 default.
    assert resolve_seeds(_config(seed=7))["curriculum_seed"] == 7
    assert resolve_seeds(_config(seed=0), seed=5)["curriculum_seed"] == 5


def test_curriculum_seed_is_independent_when_given():
    seeds = resolve_seeds(_config(seed=7), seed=43, curriculum_seed=1043)
    assert (seeds["training_seed"], seeds["curriculum_seed"]) == (43, 1043)
    assert seeds["resolved_from"] == "cli"


def test_omitting_both_records_the_config_provenance_honestly():
    seeds = resolve_seeds(_config(seed=7))
    assert seeds == {
        "training_seed": 7,
        "curriculum_seed": 7,
        "config_seed": 7,
        "resolved_from": "config",
    }


def test_a_config_without_a_seed_records_the_default_it_actually_used():
    seeds = resolve_seeds(_config(seed=None))
    assert seeds["config_seed"] == DEFAULT_CONFIG_SEED
    assert seeds["training_seed"] == DEFAULT_CONFIG_SEED
    assert seeds["resolved_from"] == "config"


@pytest.mark.parametrize(
    "kwargs",
    [dict(seed=-1), dict(curriculum_seed=-1), dict(seed=True), dict(seed=1.5)],
)
def test_unusable_seeds_are_refused(kwargs):
    with pytest.raises(ValueError):
        resolve_seeds(_config(), **kwargs)


def test_unusable_config_seed_is_refused():
    with pytest.raises(ValueError, match="run.seed"):
        resolve_seeds({"run": {"seed": "42"}})


# -------------------- the artifact -----------------------------------------


def test_seeds_json_has_exactly_the_documented_shape(tmp_path):
    seeds = resolve_seeds(_config(seed=7), seed=44, curriculum_seed=1044)
    path = write_seeds(tmp_path, seeds)
    assert path.name == SEEDS_ARTIFACT
    written = json.loads(path.read_text())
    assert sorted(written) == sorted(SEEDS_FIELDS)
    assert written == {
        "training_seed": 44,
        "curriculum_seed": 1044,
        "config_seed": 7,
        "resolved_from": "cli",
    }
    assert all(isinstance(written[k], int) for k in SEEDS_FIELDS[:3])


def test_distinct_runs_record_distinct_seeds(tmp_path):
    recorded = []
    for seed in (42, 43, 44):
        run_dir = tmp_path / f"run_seed{seed}"
        seeds = resolve_seeds(_config(seed=7), seed=seed, curriculum_seed=seed + 1000)
        recorded.append(json.loads(write_seeds(run_dir, seeds).read_text()))
    assert [r["training_seed"] for r in recorded] == [42, 43, 44]
    assert [r["curriculum_seed"] for r in recorded] == [1042, 1043, 1044]
    assert len({r["training_seed"] for r in recorded}) == 3
    assert len({r["curriculum_seed"] for r in recorded}) == 3


def test_seed_record_is_not_silently_rewritten(tmp_path):
    seeds = resolve_seeds(_config(), seed=42)
    write_seeds(tmp_path, seeds)
    write_seeds(tmp_path, seeds)  # byte-identical rewrite is a no-op
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_seeds(tmp_path, resolve_seeds(_config(), seed=43))


def test_malformed_seed_record_refused(tmp_path):
    with pytest.raises(ValueError, match="exactly"):
        write_seeds(tmp_path, {"training_seed": 1})
    with pytest.raises(ValueError, match="exactly"):
        write_seeds(tmp_path, {**resolve_seeds(_config(), seed=1), "evaluation_seed": 3})
