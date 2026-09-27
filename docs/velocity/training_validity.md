# PhoenixVelocity training validity: motor-gain randomization

## Every PhoenixVelocity run before commit 62e912a trained on the wrong plant

The motor-strength domain randomization (`phoenix.sim_env.go2_env_cfg.scale_explicit_actuator_gains`)
multiplied the **current** actuator stiffness and damping by a new U(0.85, 1.15) factor. PhoenixVelocity
schedules it at **every episode reset** (`mode="reset"`, since `12ab3d9`), so each env's gains performed a
multiplicative random walk instead of being redrawn around nominal.

Measured live in Isaac (1024 envs), stiffness / nominal, 10th / 50th / 90th percentile:

| resets per env | q10  | median | q90  |
|---------------:|-----:|-------:|-----:|
| 1              | 0.88 | 1.00   | 1.12 |
| 10             | 0.66 | 0.97   | 1.40 |
| 40             | 0.42 | 0.86   | 1.74 |

The training plant therefore changed during every run, getting harder the longer the process ran, and a
process restart silently reset it (the walk_vnext3_seed47 1500 -> 1750 "resume discontinuity": same
model_1500 weights, action |a| p99 1.99 fresh, 2.99 after 40 resets, 2.03 with the gains restored).

**Affected (trained under compounding gains):** walk_v1 seeds 42-44, walk_v2 seeds 45-46,
walk_vnext_seed47, walk_vnext2_seed47, walk_vnext3_seed47 (0-1500 and the resumed 1500-3000 segment,
whose plant was fresh again at the restart and then drifted).

**Also affected:** every Isaac evaluation (`scripts/eval_velocity.py`, `scripts/analyze_startup_reward.py`)
run before 62e912a. They reset the env per test case, so later cases and later checkpoints in a batch ran
on more-drifted gains (order contamination). MuJoCo sim2sim results (gate v3, `sim2sim_cold_start.py`) do
not use this function and are unaffected.

**Fix (62e912a):** nominal gains are captured on the first call; every call sets nominal x factor.
Startup-mode use (stand task, one call) is unchanged. `tests/test_actuator_gain_dr.py` fails on the old code.

## The first valid training result under the intended stationary DR

`walk_vnext3c_seed47` (2026-09-26): the exact vnext3 recipe (spec `vnext3_spec()`, seed 47, 8192 envs),
trained from iteration 0 with the fix. It is the first PhoenixVelocity run whose motor-gain distribution is
the intended stationary U(0.85, 1.15) x nominal for its whole duration. Results are recorded in the
project log; conclusions from earlier runs about long-horizon drift should be re-checked against it.
