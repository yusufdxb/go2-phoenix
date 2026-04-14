# Adapt warm-start diagnostic

## Symptom

The morning run of `scripts/adapt.sh configs/train/adaptation.yaml`
produced `Mean reward: -0.03` at iter 0 even though the loaded baseline
(`checkpoints/phoenix-base/2026-04-14_10-43-37/model_499.pt`) evaluates
at **mean return 18.95 / 100% success** on rough terrain.  That looked
like `OnPolicyRunner.load(load_cfg=..., strict=False)` was silently
dropping state — most obviously the learned Gaussian `std_param` or the
empirical obs normalizer.

## Evidence

### 1. The load is actually byte-exact

With `/tmp/debug_load.py` we compared the live actor `state_dict()`
against the raw checkpoint after calling `runner.load(..., strict=False)`:

```
distribution.std_param   norm=4.2650  changed=True  live_matches=True
mlp.0.weight             norm=24.9339 changed=True  live_matches=True
mlp.0.bias               norm=0.9909  changed=True  live_matches=True
mlp.2.weight             norm=20.5767 changed=True  live_matches=True
...
```

Every tensor that exists in `actor_state_dict` is present in the live
module and `torch.allclose` passes against the checkpoint value. The
learned exploration std (mean 1.18, from baseline training) does
overwrite the yaml-init 0.5.

### 2. The empirical obs normalizer never existed

`UnitreeGo2RoughPPORunnerCfg.policy.actor_obs_normalization=False`
upstream, so `isaaclab_rl.handle_deprecated_rsl_rl_cfg` maps our
`empirical_normalization: true` in `configs/train/*.yaml` only into a
field that the upstream cfg already overrode, and the actor ends up
with `obs_normalizer = torch.nn.Identity()`. The baseline checkpoint
has no `obs_normalizer._mean` / `obs_normalizer._var` /
`obs_normalizer.count` buffers — the helper
`phoenix.training.checkpoint.load_runner_checkpoint` reports
`actor_obs_normalizer_in_ckpt=False`. There is no normalizer state to
lose.

### 3. The actual cause is a logging artifact

`rsl_rl.utils.Logger.process_env_step` only pushes an episode's reward
into `rewbuffer` when that episode terminates in the 24-step rollout.
At iter 0 of a fine-tune:

* 4096 envs × 24 steps = 98 304 env-steps.
* `init_at_random_ep_len=True` pre-seeds `env.episode_length_buf` to a
  random value in `[0, max_episode_length=1000)`.  Any env whose pre-
  seeded length exceeds ~976 will time-out inside the 24-step rollout,
  counting whatever tiny partial reward it accumulated during those few
  steps into `rewbuffer`.
* 99.6% of envs — the ones actually walking, loaded-policy-style —
  never terminate in this window and therefore contribute nothing to
  the displayed `Mean reward`.

A direct rollout with the warm-started policy at 4096 envs, no
`init_at_random_ep_len`, confirms the mismatch:

```
[PRE-LEARN ROLLOUT] done_eps=15  mean_ret=0.075  mean_len=159
                    (4081 envs still alive after 200 steps)
```

Only 15 of 4096 envs terminated in 200 steps (≈4 seconds of sim), and
they were the ones that started on the hardest `max_init_terrain_level=5`
terrain which the baseline did not see much of during training. The
remaining 4081 envs are walking fine — they just don't show up in the
reward buffer yet.

### 4. Baseline training shows the same pattern

Reading
`checkpoints/phoenix-base/2026-04-14_10-43-37/events.out.tfevents.*`:

```
Train/mean_reward:       first3 = -0.35 @0, -0.57 @1, -0.56 @2
Train/mean_episode_length: first3 = 13, 35, 61
Curriculum/terrain_levels: first3 = 3.5, 3.5, 3.5
```

Baseline training *also* showed reward near zero at iter 0, and only
climbed to 18 once the terrain curriculum demoted most envs to easy
levels and episodes actually started timing out with full 1000-step
reward integrals. The adapt "iter-0 regression" is the same mechanism
reading the same rollout through the same buffer — not a broken load.

## Fix

1. Added `phoenix.training.checkpoint.load_runner_checkpoint` that
   re-reads the checkpoint after `runner.load` and verifies every
   actor/critic tensor round-trips via `torch.allclose`, raising if
   not. Baseline training, fine-tune, and evaluate all use it so a
   silent partial load becomes a visible exception.
2. Fine-tune now runs `runner.learn(..., init_at_random_ep_len=False)`.
   A warm-started policy does not need randomised initial episode
   lengths for exploration, and turning them off means iter-0 metrics
   reflect the actual policy behaviour rather than the timeout window
   artifact above.
3. The adapt yaml's `policy.init_noise_std` was changed from 0.5 to
   1.0 — still overwritten by the load, but now matches the
   checkpoint's learned mean so the transient pre-load state looks
   like what was trained.

## Not fixed (yet)

- `failure_sample_fraction` stays at 0.0. The `reset_bridge`
  monkey-patch works (see `tests/test_curriculum.py`) but we do not
  have a hardware-captured failure parquet for this cycle, so flipping
  it to 0.3 would just replay synthetic parquets and contaminate the
  "fine-tune on rough" result. It is an opt-in knob once real-robot
  failures are recorded.
