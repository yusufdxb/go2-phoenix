# Deployment-mismatch ablation: harness works, first run is NOT a valid measurement

Run 2026-09-12, `phoenix-flat-v4/latest.pt` on `configs/env/flat_v4.yaml`,
64 envs, 500 steps, seed 0. Raw: `deploy_ablation.json`.

```
python scripts/deploy_ablation.py \
    --checkpoint checkpoints/phoenix-flat-v4/latest.pt \
    --env-config configs/env/flat_v4.yaml --num-envs 64 --steps 500
```

## Status

The harness executes all eight cells against one unmodified checkpoint and
writes machine-readable output. **The numbers it produced are not yet usable as
evidence, and must not be cited.** Two independent problems, both in the
measurement rather than the ablations themselves.

| cell | slew | terminations | episodes ended | fall rate |
|---|---:|---:|---:|---:|
| correct | 0.9992 | 3 | 3 | 1.000 |
| zero_base_lin_vel | 0.9989 | 19 | 63 | 0.302 |
| historical_hip_offset | 0.9993 | 2 | 5 | 0.400 |
| velocity_plus_hip | 0.9987 | 32 | 74 | 0.432 |
| deploy_limiter_only | 0.9993 | 3 | 3 | 1.000 |
| hip_plus_limiter | 0.9994 | 3 | 61 | 0.049 |
| velocity_plus_limiter | 0.9993 | 5 | 8 | 0.625 |
| all_three | 0.9996 | 4 | 58 | 0.069 |

## Problem 1: the slew metric is saturated

Every cell lands between 0.9987 and 0.9996. A metric that reads 99.9% under the
clean control and 99.9% under all three mismatches at once cannot separate them.

It is not a joint-order bug. Indexing `measured_q` by the action term's
`joint_ids` was added and changed the result by zero to sixteen significant
figures, which means `joint_ids` is identity for this env.

The plausible reading is that a walking policy at this action scale genuinely
saturates the per-step clip almost always, since `|target - measured_q|`
includes PD tracking error, not just the action delta. If that is right it is a
substantive finding about deploying `flat-v4` behind the limiter, but it is
saturated either way and cannot serve as the ablation's discriminator. A
discriminating variant needs either a margin distribution (how far past the cap,
not merely whether) or a stand policy where the clean rate is low.

## Problem 2: the cells are not comparable

`episodes_ended` ranges from 3 to 74 across cells given identical 500 steps and
64 envs. A fall rate computed over 3 episodes is not comparable to one over 74,
and the two columns disagree about direction: `correct` shows the WORST fall
rate (1.000) purely because only 3 episodes ended at all.

Terminations per fixed env-step budget is the better statistic and does show a
large spread: 3 for the clean control against 19 for zeroed `base_lin_vel` and
32 with the hip offset added. That is suggestive, and it points the same way as
the P0 fix, but the episode accounting has to be sound before it is worth
anything. It is recorded here as an observation, not a result.

## What to change before this is worth running again

1. Fix the episode accounting so every cell is compared over the same number of
   completed episodes, or report terminations per env-step with the episode
   count held fixed by construction.
2. Replace or supplement the saturated rate with a clip-margin distribution.
3. Run a stand policy as well, where the clean-control clip rate is low enough
   for the metric to have headroom, which is also the configuration the
   historical Gate 7 number came from.

Until those land, the only claim supported here is that the grid mechanism
works: eight cells, one checkpoint, no retraining, and the observation and
action transformations are unit-tested independently in
`tests/test_deploy_ablation.py`.
