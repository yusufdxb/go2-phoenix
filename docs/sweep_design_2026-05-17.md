# Phoenix Sim Sweep Design (2026-05-17)

## Why a sweep, why now

The v0.2 deployment checkpoint at commit `93e4a94` (phoenix-stand-v3,
10240-env fine-tune) already cleared the existing one-off sim benchmark:
32/32 survival, slew 0.33%, mean ang_vel_err 0.011. That benchmark uses
the same DR ranges the policy trained against (friction 0.3 to 1.5, no
pushes, fixed reward weights), so passing it tells us the policy is
self-consistent, not that it is robust off-distribution.

The 2026-04-21 live Gate 7 run blew that result up (33.06% per-step slew
saturation) because the hardware exposed regimes the sim never sampled:
stand-on-stand posture offset, IMU noise floor, lab-floor friction
distribution. Before the next lab session we want a sim-side proxy that
deliberately stresses the policy along the axes most likely to matter.
A small Cartesian sweep, run sim-only, is cheap insurance against
burning another lab slot on a config that the sim could have flagged.

This is a **stress-test grid**, not a hyperparameter search. Cell count
is small (12 to 20) on purpose: each cell trains a fresh fine-tune from
the v3 checkpoint, then evaluates on the cell's perturbed env.

## Sweep axes

Each axis ties to one concrete YAML override field. All three are
already wired through `phoenix.sim_env.build_env_cfg` (verified against
`docs/sweep_design_2026-05-17.md`'s notes on the unwired audit, see
`tests/test_config_loader.py::test_unwired_empty_when_only_wired_sections_present`).

| Axis | Config field | Why it matters |
|---|---|---|
| friction | `domain_randomization.friction_range` (in `configs/env/base.yaml`, also overridden by `configs/env/slippery.yaml`) | Lab-floor coefficient is unknown; v3 trained on [0.3, 1.5] only. Slippery overlay narrows to [0.05, 0.4]. We sweep across both regimes. |
| push magnitude | `perturbation.push_velocity_xy` (off by default in base, enabled in `configs/env/perturbation.yaml` at 1.5 m/s) | The stand-v3 fine-tune did NOT enable perturbations. We probe whether the policy holds posture against a 0.5 / 1.0 / 1.5 m/s lateral impulse. |
| action smoothness penalty | `reward.action_rate` (stand_v3.yaml = -2.0, base = -0.05) | The v3 retrain bumped this to -2.0 to fight the OOD-action problem. We probe whether a higher penalty (-3.0, -5.0) gives more headroom on hardware without collapsing tracking. |

Two additional axes are *available* in the spec format but excluded from
the default grid to keep the cell count small:

* `command.lin_vel_x` range (stand collapses to [0, 0])
* `curriculum.failure_modes` subset (per `feedback_go2_mcf_gait` notes,
  combined vx + yaw degrades on real hardware; not a sim-stress concern)

## Default grid (12 cells, "stand-v3 stress")

3 friction regimes x 2 push magnitudes x 2 action-rate penalties = 12 cells.

```
friction_range  : [(0.3, 1.5), (0.15, 0.8), (0.05, 0.4)]
push_velocity_xy: [0.0, 1.0]
action_rate     : [-2.0, -3.0]
```

Cell IDs follow the format `cell_<idx>__f<fl>-<fh>__p<push>__a<arate>`,
e.g. `cell_03__f0.05-0.4__p1.0__a-2.0`.

## Two scales

| Scale | num_envs | max_iterations | per-cell wall (RTX 5070, est) | total grid wall |
|---|---|---|---|---|
| SMOKE | 256 | 50 | 2 to 5 min | 24 to 60 min (12 cells) |
| FULL | 4096 | 300 | 25 to 40 min | 5 to 8 h (12 cells) |

FULL recommendation comes from the existing stand-v3 retrain (500 iters
at 10240 envs took ~45 min on the lab-PC 5080 per commit message on
93e4a94), scaled to 4096 envs and 300 iters. Mewtwo's 5070 is roughly
70% of the 5080, so the upper bound is reasonable. The user can adjust
in the spec file before kicking it off.

## Eval metrics

All metrics produced per cell, collected into `logs/sweeps/<ts>/leaderboard.csv`:

| Metric | Source | Pass threshold (stand-v3 reference) |
|---|---|---|
| survival_rate | `phoenix.training.evaluate.RolloutMetrics.success_rate` | >= 0.95 |
| mean_episode_len_s | `RolloutMetrics.mean_episode_length_s` | >= 18.0 (of 20 s episodes) |
| slew_saturation_pct | `RolloutMetrics.slew_saturation_pct` (`phoenix.training.slew.slew_saturation_rate`) | < 0.05 (Gate 7 bar) |
| mean_ang_vel_err | `RolloutMetrics.mean_ang_vel_error` | < 0.05 rad/s |
| fall_count | num_episodes - successes (derived in runner) | <= 5% of num_episodes |

The runner does not implement the eval pass itself; it shells out to
`scripts/eval_stand_v3.sh` (or equivalent for non-stand cells), parses
the `metrics.json` written by `phoenix.training.evaluate.main`, and
appends one row per cell. For the SMOKE scale we skip eval entirely and
only verify the cell trains for one iteration without crashing (the
checkpoint dir's existence is the success signal).

## Spec file format

YAML, lives anywhere on disk. Documented schema:

```yaml
# configs/train/sweep_stand_v3_stress.yaml
name: stand-v3-stress-2026-05-17
base_train_config: configs/train/ppo_stand_v3.yaml   # the cell config defaults to inherit from
base_env_config: configs/env/stand_v3.yaml           # the cell env config defaults to inherit from
resume: checkpoints/phoenix-stand-v3/latest.pt       # all cells fine-tune from this checkpoint
axes:
  friction_range:
    field: domain_randomization.friction_range
    values:
      - [0.3, 1.5]
      - [0.15, 0.8]
      - [0.05, 0.4]
  push_velocity_xy:
    field: perturbation.push_velocity_xy
    values: [0.0, 1.0]
    enable_field: perturbation.enabled   # set to true automatically when value > 0
  action_rate:
    field: reward.action_rate
    values: [-2.0, -3.0]
scales:
  smoke:
    num_envs: 256
    max_iterations: 50
    eval: false
  full:
    num_envs: 4096
    max_iterations: 300
    eval: true
```

## What "ready to run" means

After this sweep, we will have one of three outcomes for each cell:
1. Cell trains and meets all five eval gates : policy is robust on that axis.
2. Cell trains but fails one or more gates : we know which off-distribution
   regime breaks v3, and can build a v4 config that explicitly trains on it.
3. Cell fails to train (NaN, instant fall) : the cell's config is itself
   broken (e.g. friction 0.05 with no posture stabilization). Useful
   negative result; we exclude that regime from hardware testing.

The leaderboard.csv is the artifact the user reads to decide what to
ship to the lab.

## Cross-references

* env wiring : `src/phoenix/sim_env/go2_env_cfg.py` (`_apply_rewards`, `_apply_domain_randomization`, `_apply_perturbation`)
* evaluate.py metrics : `src/phoenix/training/evaluate.py` (`RolloutMetrics`)
* slew metric : `src/phoenix/training/slew.py` (`slew_saturation_rate`)
* failure curriculum (out of scope here, axis available) : `src/phoenix/adaptation/curriculum.py`
