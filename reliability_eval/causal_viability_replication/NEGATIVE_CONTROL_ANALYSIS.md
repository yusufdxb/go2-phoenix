# Pre-onset negative control: leak or noise floor

Study: `phoenix_causal_viability_replication_v1`, run 2026-07-29. Three processes x {stand, walk} x
{motor degradation, observation corruption}, 576 independent blocks, 12 independent protocols.
Analysis date: 2026-08-24. Analysis only. No simulator run was executed for this document, and the
frozen artifacts were read, never regenerated.

---

## Verdict

**It is a leak, not a noise floor.**

The pre-onset window is bit-identical between arms in exactly one block per run, and the negative
control is exactly `0.000` pp there, in all 12 process-cells, with the per-environment fall vectors
element-for-element equal. In every subsequent block the arms enter the pre-onset window with
different observations, and the negative control becomes nonzero. A noise floor cannot produce that
pattern: a stochastic floor would be nonzero in the first block too.

The channel is identified mechanically. The observation returned by `env.reset()` at the head of
each block carries **stale root linear velocity, root angular velocity, and projected gravity from
the terminal state of the previous block**. The previous block's terminal state is exactly what the
treatment changes. So arm assignment in block `b-1` determines the first policy input of block `b`,
which is inside block `b`'s pre-onset window.

**The frozen gate `pre_onset_negative_controls_include_zero` FAILED and stays FAILED.** The criterion
is not re-specified here. This document explains why it failed, and reports one leak-free estimate
that the frozen study already contains.

Consequence for the paper, stated plainly per `paper/outline.md` Section 6 branch (b): **the effect
estimates in `result_table.md` are not clean, and the study cannot go to ICRA in its current form.**
The interaction survives on the leak-free subset (Section 4), so the finding is very likely real,
but "very likely real" is not what the frozen protocol promised. The fix is a re-run, and it is
cheap: the entire study is 1.53 h of wall clock.

---

## 1. Mechanical trace

### 1.1 The fallback itself is provably inert before onset

`scripts/reliability_closed_loop.py`:

- L745 unshielded arm: `applied_blend = np.zeros(args.envs)`, unconditionally, every tick.
- L753 oracle arm: `since = tick - (block.onset_tick + args.oracle_delay_ticks)`, then
  `b = min(1.0, (since + 1) / handoff) if since >= 0 else 0.0`. For `tick < onset_tick`, `since < 0`,
  so `b` is exactly `0.0`.
- L779 `blended = actions * torch.as_tensor(1.0 - applied_blend, ...)`. With `applied_blend == 0.0`
  this is multiplication by exactly `1.0` in both arms.
- Disturbance injection is gated at onset in every family: L714 (`onset_obs` capture),
  L724 (observation corruption, `tick >= onset_tick`), L731 (command). Motor injection is gated at
  `tick == block.onset_tick`.
- L796 `pre_onset_fall |= new_falls & (tick < block.onset_tick)`.

`shield.step(...)` is called identically in both arms on every tick, so there is no extra RNG draw,
no extra buffer write, and no differing action attributable to arm identity inside a block.
`src/phoenix/reliability/replication.py` L335-338 and L340-344 re-verify this on the recorded data: the
unshielded arm carries zero engagement, no switch tick, and zero blend dose everywhere.

**There is no within-block treatment leak.** The leak is across blocks.

### 1.2 The random number streams are fully synchronized between arms

This is the claim that has to be nailed down before anything can be attributed to noise, and the
frozen artifacts settle it.

Observation noise is enabled. `configs/env/stand_v3_h25.yaml` resolves (via `configs/env/base.yaml`
L67-72) to `joint_pos: 0.01, joint_vel: 1.5, base_lin_vel: 0.1, base_ang_vel: 0.2,
projected_gravity: 0.05`, applied by `_apply_observation_noise` in
`src/phoenix/sim_env/go2_env_cfg.py` L461-495 (`policy.enable_corruption = True` at L476).

The noise is visible in the recorded `initial_obs` and it is provably applied:

```
process_01/stand_motor, unshielded arm, initial_obs minus reset_state
  joint_pos dims 12:24, residual after removing the per-dim default-pose offset:
      min -0.0102  max +0.0101      (config half-width 0.01)
  joint_vel dims 24:36, residual:
      min -1.4999  max +1.4999      (config half-width 1.5)
```

Those same dimensions are **bit-identical between the two arms**, in every block, in all 12
process-cells. Per-dimension cross-arm max absolute difference of `initial_obs`,
`process_01/stand_motor`:

```
dims  0- 8 : 0.6886 0.2654 1.7103 3.6076 5.8297 0.7503 0.4874 1.0823 1.0718
dims  9-47 : 0.0 (all thirty-nine)
```

A shared RNG stream that produced a different draw for one arm would perturb the `joint_vel` noise
too. It does not. **The generators are synchronized; the noise realisation is not the difference
between the arms.**

### 1.3 What actually differs: a stale root-state buffer in the reset observation

Dimensions 0 to 8 of the 48-dim policy observation are `base_lin_vel` (0:3), `base_ang_vel` (3:6),
and `projected_gravity` (6:9). The observation layout is confirmed from the artifacts themselves:
dims 9:12 (velocity command) and dims 36:48 (last action) are exactly `0.0` at reset, dims 12:24
track `reset_state` joint positions to within the 0.01 noise half-width, and dims 24:36 track joint
velocities to within the 1.5 half-width.

Two facts, both read directly from the frozen `.npz` files:

**(a) The recorded physical reset state is bit-identical between arms and has exactly zero
velocity.** `reset_state` (root state 13 + joint pos 12 + joint vel 12 = 37, captured at
`reliability_closed_loop.py` L669-676, immediately after `env.reset()` at L667) satisfies, for all
48 blocks of all 24 runs:

```
max abs cross-arm difference of reset_state = 0.0   (all 12 process-cells)
max abs root lin_vel_w over every block, env, cell  = 0.000000
max abs root ang_vel_w over every block, env, cell  = 0.000000
```

**(b) The observation returned by that same `env.reset()` does not have zero velocity, and its
velocity content grows once a previous block exists.** With `reset_settle_ticks = 0`
(`protocol.json: params.reset_settle_ticks`, so the L683-689 settle loop never executes),
`initial_obs` at L690 is the observation of the reset state itself:

```
                       block 0            blocks 1..47
stand_motor  base_lin_vel  max 0.8924        max 1.6447
stand_motor  base_ang_vel  max 0.2831        max 4.7226
walk_motor   base_ang_vel  max 0.2985        max 5.3367
```

The true root velocity is exactly zero, the configured noise half-widths on these terms are 0.1 and
0.2, and the observed values reach 5.34. The content is neither the true state nor the noise. It is
the previous episode's terminal motion: a fallen or falling quadruped carries several rad/s of body
rate, and that is what the policy sees on the first tick of the next block.

ASSUMPTION: the precise Isaac-internal reason is that `env.reset()` writes the reset root state to
the physics view but the observation is computed from cached `root_lin_vel_b` / `root_ang_vel_b` /
`projected_gravity_b` buffers that the write does not invalidate. This was not confirmed by running
the simulator, which is out of scope here. **Nothing below depends on that attribution**; the
recorded facts (a) and (b) stand on their own.

### 1.4 The leak is treatment-caused, and block 0 is the internal control that proves it

Block 0 of every run is the first block executed, so it has no predecessor inside the run and its
stale buffer is whatever the environment held at construction, which is identical in both arms.
Every later block inherits a predecessor whose trajectory the treatment altered.

The prediction is sharp and it holds in all 12 process-cells:

```
cell          block 0: cross-arm initial_obs diff   onset_obs diff   pre_onset effect   fall vectors identical
stand_motor              0.0                            0.0            +0.000 pp             3/3
stand_obs                0.0                            0.0            +0.000 pp             3/3
walk_motor               0.0                            0.0            +0.000 pp             3/3
walk_obs                 0.0                            0.0            +0.000 pp             3/3

cell          blocks 1..31 (disturbed):                              pre_onset effect   fall vectors identical
stand_motor   initial_obs diverges in 47/47 blocks                     -0.672 pp            73/93
stand_obs     initial_obs diverges in 47/47 blocks                     +1.747 pp            68/93
walk_motor    initial_obs diverges in 47/47 blocks                     +0.403 pp            14/93
walk_obs      initial_obs diverges in 47/47 blocks                     -1.815 pp            25/93
```

`onset_obs` is captured at the registered onset tick (L714), 100 to 200 ticks after reset. In block 0
it is bit-identical between arms in all 12 process-cells, so the **entire pre-onset window** of
block 0 is bit-identical, not merely its first frame. The harness is therefore deterministic within
a block given its entry state, and a pre-onset effect of exactly zero is obtainable. It is obtained
in block 0 and lost in every block after it.

That is the whole leak-versus-noise question, answered. A noise floor is a property of the
estimator and would be present in block 0. It is not present in block 0.

### 1.5 The leak's magnitude tracks the treatment's effectiveness

The mechanism predicts that when the fallback helps, the oracle arm ends blocks upright, so its
stale buffer is quiet; when the fallback harms, the oracle arm ends blocks falling, so its stale
buffer is loud. Mean absolute stale root velocity, `|initial_obs[0:6]|`, over disturbed blocks:

| Cell | stale \|v\| unshielded | stale \|v\| oracle | difference | oracle effect direction |
|---|---:|---:|---:|---|
| Standing, motor | 0.1153 | 0.1578 | -0.0425 | oracle harms (-14.47 pp) |
| Standing, observation | 0.1673 | 0.0802 | +0.0871 | oracle helps (+6.21 pp) |
| Walking, motor | 0.2050 | 0.2195 | -0.0145 | oracle harms (-6.24 pp) |
| Walking, observation | 0.1582 | 0.0796 | +0.0786 | oracle helps (+11.57 pp) |

The sign of the stale-velocity imbalance matches the sign of the treatment effect in all four cells.
The leak is not an incidental desynchronisation; it is a feedback path from the outcome of block
`b-1` into the initial condition of block `b`, and its direction is set by the treatment.

This has a direction of bias worth stating: the feedback is **positive**. Where the fallback helps,
it also hands its own arm a cleaner start next block; where it harms, it hands its own arm a dirtier
one. Both inflate the magnitude of the estimated effect. ASSUMPTION: that the sign correlation above
implies inflation rather than some cancelling higher-order path; the block-0 comparison in Section 4
is the direct test, not this argument.

### 1.6 Alternatives ruled out while tracing

- **Non-reproducible RNG.** Ruled out by Section 1.2, and independently by the preflight replay. The
  v6 preflight runs an A-B-N-N-A subset in which row 4 replays block 0 in the same process and same
  arm. `reset_state` reproduces exactly (`max abs diff 0.0`) and `initial_obs` does not
  (`0.7968`), but the difference is confined to dims 0 to 8; **dims 9 to 47 are bit-identical**
  (`max 0.0`). Same signature as the cross-arm divergence, same cause. An RNG that failed to restore
  would have moved the `joint_vel` noise.
- **Observation normalisation drift.** `EmpiricalNormalization.forward` (rsl_rl 5.0.1,
  `modules/normalization.py`) is pure; statistics move only in `update`, which the rollout never
  calls. Installed 5.0.1 matches `bundle.json: versions.rsl_rl_lib`.
- **Physics nondeterminism at reset.** `reset_state` is bit-identical in all 48 blocks of all 24
  runs, and block 0's pre-onset window is bit-identical through onset.
- **Motor-gain carry-over.** Gains are restored to 1.0 at the head of every block with readback
  verification (`reliability_closed_loop.py` L556-586, called at L664).
- **Command carry-over.** The `command` family is not used by this study; all 12 cells are `motor`
  or `obs` per `registry.json`.

---

## 2. Is 0.65 to 1.76 pp consistent with pure Monte Carlo noise?

No, not for two of the four cells, and the block-0 result above makes the question moot. The
computation is reported anyway because the paper's Section 6 P4 currently commits to a stated noise
floor and that number must not be invented.

Unit of analysis is the disturbed block, matching the frozen estimand: 32 disturbed blocks x 3
processes = 96 per cell, 16 environments per block, 1536 environment pairs per cell.

| Cell | Pre-onset effect, pp | Block sd, pp | se, pp | 95% block-bootstrap CI, pp | Independent-binomial se, pp | Sign-flip perm p (200k) |
|---|---:|---:|---:|---|---:|---:|
| Standing, motor | -0.651 | 3.204 | 0.327 | [-1.302, -0.065] | 0.343 | 0.0755 |
| Standing, observation | +1.693 | 3.072 | 0.314 | [+1.107, +2.344] | 0.378 | < 0.00001 |
| Walking, motor | +0.391 | 8.969 | 0.915 | [-1.367, +2.214] | 0.957 | 0.7245 |
| Walking, observation | -1.758 | 7.181 | 0.733 | [-3.190, -0.326] | 0.850 | 0.0226 |
| Pooled, n = 384 blocks | -0.081 | | 0.320 | [-0.700, +0.553] | | |

These reproduce `result_table.md` to the reported precision, which confirms the committed artifacts
regenerate. The independent-binomial standard error, computed from the pooled pre-onset fall rate as
`sqrt(2 p (1-p) / 1536)`, agrees with the block-level standard error to within 5 to 20 percent, so
block clustering is mild for this outcome and the block bootstrap is not obviously anticonservative
here.

**Implied Monte Carlo floor, if the arms were merely two independent draws:**

> **+/-0.62 pp for the standing cells and +/-1.44 to +/-1.79 pp for the walking cells
> (1.96 x se, n = 96 blocks per cell, 1536 environment pairs).**

Measured against that floor:

- Walking / motor, +0.391 pp, is well inside. Consistent with noise.
- Standing / motor, -0.651 pp, is at 2.0 se, permutation p = 0.076. Marginal.
- Walking / observation, -1.758 pp, is at 2.4 se, p = 0.023, which does not survive a Bonferroni
  correction over four cells (0.023 x 4 = 0.091).
- Standing / observation, +1.693 pp, is at 5.4 se, p < 1e-5, and survives any correction. 30
  unshielded pre-onset falls against 4 oracle out of 1536 environment pairs per arm, with 25 of 96
  blocks carrying a differing pre-onset fall vector.

Under a pure-noise null with four tests, roughly 0.2 cells would be expected below p = 0.05. Two are,
one of them by five orders of magnitude. **The pre-onset values are not fully explainable as Monte
Carlo noise**, and Section 1 says why: the pre-onset windows are not an A/A comparison after block 0.

Absolute pre-onset fall counts, disturbed blocks, 1536 environment pairs per arm per cell:

| Cell | Unshielded | Oracle | Difference | Blocks with differing pre-onset fall vector |
|---|---:|---:|---:|---:|
| Standing, motor | 9 | 19 | -10 | 20 / 96 |
| Standing, observation | 30 | 4 | +26 | 25 / 96 |
| Walking, motor | 120 | 114 | +6 | 79 / 96 |
| Walking, observation | 77 | 104 | -27 | 68 / 96 |

---

## 3. What this does to the study as frozen

The pre-registered design claims paired blocks with a common initial condition. That pairing holds
for the physical reset state (bit-identical, always) but not for the policy's first observation
(differs in 47 of 48 blocks). The blocks within a run are therefore not independent replicates:
each one's initial condition is a function of the arm's own history.

Three specific consequences.

1. **The negative control is not interpretable as a floor.** It is measuring the leak.
2. **Blocks are serially dependent within a run.** The block bootstrap resamples blocks as if
   exchangeable. With 3 processes per cell, the honest independent replicate count is 3, not 96.
3. **The headline effects carry a bias whose sign is, per Section 1.5, away from zero.** The size
   of that bias is not bounded by anything measured in the frozen study except the block-0 subset.

---

## 4. The leak-free estimate the study already contains

Block 0 of each run is uncontaminated. There are 12 such blocks, 3 per cell, 48 environment pairs
per cell. Recomputing the primary estimand (paired post-onset fall-rate difference among jointly
onset-eligible pairs, unshielded minus oracle) on block 0 only:

| Cell | Eligible pairs | Block-0 effect, pp | 95% CI, env-pair bootstrap 20k | Frozen headline, all blocks, pp |
|---|---:|---:|---|---:|
| Standing, motor | 48 | -8.33 | [-18.75, +0.00] | -14.47 |
| Standing, observation | 47 | +4.26 | [+0.00, +10.64] | +6.21 |
| Walking, motor | 48 | -18.75 | [-31.25, -8.33] | -6.24 |
| Walking, observation | 48 | +14.58 | [+6.25, +25.00] | +11.57 |
| **Interaction, obs minus motor** | 191 | **+23.02** | **[+13.63, +32.47]** | **+19.24** |

Per-process block-0 effects, as a clustering sanity check (n = 3, no interval claimed):

| Cell | process_01 | process_02 | process_03 |
|---|---:|---:|---:|
| Standing, motor | -12.50 | -18.75 | +6.25 |
| Standing, observation | +6.67 | 0.00 | +6.25 |
| Walking, motor | 0.00 | -31.25 | -25.00 |
| Walking, observation | +12.50 | +12.50 | +18.75 |

**All four signs agree with the frozen headline, and the interaction, which is the paper's headline
claim, reproduces at +23.02 pp with a 95 percent interval excluding zero.** Three of four cells hold
direction in all three processes; standing/motor flips in process_03 at n = 16 pairs.

Caveat, stated because it limits what this table can carry: the 48 pairs per cell are 16 parallel
environments in each of 3 blocks, so they share the block's sampled disturbance parameters. The
environment-pair bootstrap therefore understates uncertainty, and the true intervals are wider than
shown. This subset is evidence that the finding is not manufactured by the leak. It is not a
replacement for the frozen study, and it must not be reported as the paper's primary result.

---

## 5. What would settle it

The leak is identified, not merely suspected, so the requirement is a design fix plus a re-run, not
a diagnostic sweep. Both stages are cheap. Measured cost of the study as run: 24 arm-runs,
mean 229.5 s each, **1.53 h total wall clock** (from `arm_*.meta.json`
`run_finished_unix_s - run_started_unix_s`).

### Stage 1: fix probe. Two runs, about 10 minutes

The harness already has the knob. `reset_settle_ticks` is a protocol parameter, currently `0`, and
the settle loop at `reliability_closed_loop.py` L683-689 steps the environment with zero actions and
re-reads the observation before `initial_obs` is captured at L690. Setting it to `1` should refresh
the stale root buffers from the freshly written reset state.

- Design: one cell (`stand_obs`, process_01 protocol), both arms, 48 blocks, 16 envs,
  `reset_settle_ticks = 1`. No new protocol freeze, no analysis.
- Acceptance: cross-arm `max abs diff` of `initial_obs` and `onset_obs` is exactly `0.0` in **all 48
  blocks**, not just block 0. That is the same check Section 1.4 runs, and it is binary.
- If it passes, the fix is one parameter and Stage 2 is a straight re-run.
- If it does not pass, the buffers need more than one step, or the refresh is elsewhere. Fall back
  to constructing a fresh environment per block, or to running each block in its own process. Both
  are more expensive but neither is prohibitive at 48 blocks.
- Cost: 2 arm-runs at 3.82 min plus simulator startup, **about 10 to 15 minutes of GPU**.

ASSUMPTION: that one settle tick suffices. Not verified. That is exactly what the probe is for, and
the probe is binary, so it cannot be argued around.

### Stage 2: re-run the full frozen study under the fix

- Design: the identical 12 protocols, identical seeds, identical registry, only
  `reset_settle_ticks` changed. Re-freeze under a new `study_id`
  (`phoenix_causal_viability_replication_v2`) with the old one retained; do **not** overwrite v1.
- The pre-registered criterion is unchanged and is carried across verbatim. Under the fix the
  pre-onset negative control should be **exactly 0.000 pp with bit-identical fall vectors in every
  block**, which is a strictly stronger pass than "the interval includes zero".
- Cost: 24 arm-runs, **1.53 h wall clock**, matching the original run.

### Stage 3, only if the negative control is nonzero after the fix

If the fix leaves a residual, the floor has to be measured rather than asserted, and the seed count
follows from the observed block standard deviations. One-sample, two-sided alpha 0.05, 80 percent
power, `n = (z_0.975 + z_0.80)^2 sd^2 / delta^2 = 7.849 sd^2 / delta^2`, target `delta = 1.0` pp:

| Cell | Block sd, pp | Blocks needed | Have | Processes needed (32 disturbed blocks each) |
|---|---:|---:|---:|---:|
| Standing, motor | 3.204 | 81 | 96 | 3 |
| Standing, observation | 3.072 | 75 | 96 | 3 |
| Walking, motor | 8.969 | 632 | 96 | 20 |
| Walking, observation | 7.181 | 405 | 96 | 13 |

**20 independent processes per cell covers all four**, against 3 today. Cost: 20 processes x 4 cells
x 2 arms = 160 arm-runs at 3.82 min, **about 10.2 hours of GPU**. The standing cells are already
adequately powered to 1 pp; only the walking cells drive the number, and walking/motor drives it
alone. A reduced version resolving 1 pp in the two walking cells only is 20 x 2 x 2 = 80 arm-runs,
about 5.1 hours.

### Not recommended

Re-specifying the frozen criterion after seeing the data. The gate failed, it stays failed, and it
failed for a real reason that a re-specification would bury rather than fix.

Reporting `result_table.md` as the paper's primary result without the re-run. The leak is real, its
bias direction is away from zero (Section 1.5), and Section 4 shows the effect survives but does not
bound the contamination on the frozen numbers.

---

## 6. Reproducing the numbers in this document

Every number above was read or computed from the frozen artifacts under
`reliability_eval/causal_viability_replication/` (and, for Section 1.6,
`reliability_eval/causal_viability_replication_preflight_v6/`) with numpy only. No simulator was
started.

The computations are short numpy reductions over `process_*/<cell>/arm_{unshielded,oracle}.npz`,
using `blocks[i].disturbed` from the matching `protocol.json` as the block mask and the
16-environment mean of `pre_onset_fall`, `post_onset_fall`, and `fell` as the block statistic.
`reset_state`, `initial_obs`, and `onset_obs` comparisons are elementwise max-abs over the stored
arrays. Bootstraps use `numpy.random.default_rng` with the seeds noted in-line (20260824 for
Section 2, 7 for Section 4); the permutation test is 200,000 sign flips over the 96 block
differences. The observation layout used throughout is
`[base_lin_vel 0:3, base_ang_vel 3:6, projected_gravity 6:9, velocity_command 9:12, joint_pos 12:24,
joint_vel 24:36, last_action 36:48]`, confirmed against the artifacts as described in Section 1.3.

## 7. Summary for the paper

- Section 6 of `paper/outline.md` takes **branch (b)**. Branch (a), the noise-floor reading, is
  falsified by the block-0 control.
- P4 as currently drafted, "a stated noise floor of about 2 pp", must be deleted. There is no noise
  floor. There is a leak with an identified channel.
- P5's gated mechanism resolves to: stale root-state buffers in the reset observation carry the
  previous block's terminal dynamics, which the treatment sets, into the next block's pre-onset
  window.
- The outline's own instruction for branch (b) applies: this triggers a design fix and re-run, or a
  re-lock with `paper-contribution-locker`. It is not a rewrite of one paragraph.
- The re-run is 1.53 h of compute behind a 10 minute probe, so the fix is affordable well inside the
  2026-09-15 deadline, contingent on the Stage 1 probe passing.
