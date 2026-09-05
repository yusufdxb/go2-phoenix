# The onset residual: measured, bounded, and reported

Source of every number below: `reliability_eval/causal_viability_replication_v2/onset_residual_audit_n5.json`
and `combined_summary_n5.json`, produced from `registry_n5.json`, plus
`analysis/selection_bias/process_level.json` for process-level inference. All of
them are rendered into `paper/numbers.md` by `scripts/paper_numbers.py`; nothing
in this file is retyped by hand. This is drafting material for Section 6
(Threats to Validity), not final typeset prose.

**Sample.** 5 independent process seeds, 20 protocols, 960 independent blocks,
640 of them disturbed, 10,240 disturbed environment pairs. `process_04` and
`process_05` were added under the pre-registration in
`analysis/PREREG_n5_extension.md`, which was committed before either ran and
declared its own kill criterion. They pinned the bit-identical experimental
source snapshot the first three did, so all five processes executed the same
experiment.

**Two inferential levels, never mixed.** The frozen block bootstrap resamples
blocks within each process and never resamples processes, so its intervals are
conditional on the seeds actually run. Blocks inside one process share a policy
load, a physics batch and a process RNG, and are not independent replicates of
the process. Every table below therefore carries both that interval and the
process-level interval (mean of the per-process means, Student t on n = 5), and
labels which is which. Where they disagree, the process-level one is the one a
reviewer asking "would this replicate on new seeds" should read.

Two honest qualifications on that. The process-level interval is the *correct*
level for a claim about new seeds, but it is not uniformly the *wider* one: for
`stand_obs` and `walk_obs` on the full sample its half-width is 0.79x and 0.60x
the block-level one, because with 4 degrees of freedom the sample SD of five
process means can come out small. The measured between-process variance component
is close to zero (one-way ANOVA over processes gives ICC ~ 0 in three of four
cells, largest F = 1.78, p = 0.135), which is consistent with that. And the
interval is parametric: at n = 5 the sign test, the exact sign-flip randomization
test and the exact Wilcoxon signed-rank all bottom out at p = 0.0625, the minimum
attainable value at five paired units, so no exact distribution-free test can
reject at alpha = 0.05 no matter how large the effect. The data are as extreme as
the design permits; the inference rests on approximate normality of five process
means, which Shapiro-Wilk does not contradict (p in [0.07, 1.00]).

Under Holm correction over all 18 process-level intervals reported here, all 14
substantive quantities still reject; the weakest is the leak-free `stand_obs`
cell at raw p = 3.59e-3, Holm-adjusted 1.79e-2. The four pre-onset negative
controls correctly do not reject.

## What is and is not aligned across the paired arms

The batched-block harness removes temporal carryover by construction: block `i`
owns environments `i*16 .. i*16+15` and lives for exactly one block, so no block
has a predecessor whose simulator state it could inherit. Measured across all 20
process-cell arm pairs:

- Reset states are bit-identical between arms in every pair.
- Initial observations are bit-identical between arms in every pair.
- Onset observations are not, in most blocks of most pairs.

The v1 leak channel (a stale reset observation carrying the previous block's
terminal state) is therefore closed. What remains is a smaller channel acting
after reset.

## The residual is positively measured, not inferred by elimination

Earlier characterisation attributed the residual to simulator-internal history
by elimination, after ruling out observation, reset state, per-block RNG,
action-term memory and actuator-delay memory. That is an argument from a
shrinking list, not a measurement. We replace it with a falsifiable prediction
and its test.

If the channel is within-tick coupling through the single shared GPU PhysX batch
that advances all 768 environments together, then whether a block's onset
observation diverges must be decided by that block's onset **tick** alone: a
perturbation seeded when the earliest-onset environments are treated needs a
finite number of ticks to reach environments that have not yet reached their own
onset, so divergence must be an upward closed set in onset tick, with one
threshold per arm pair. Temporal carryover across blocks would instead order
divergence by block index, which the batched harness does not define, and
disturbance status or environment index would order it if the channel were
per-environment.

**The prediction holds in 18 of 20 arm pairs exactly, and in 956 of 960 blocks
overall.** In 18 pairs a single onset-tick threshold separates the divergent
blocks from the bit-identical ones with no exceptions. In the remaining two
pairs, both in `process_05` (`walk_motor` 1 block, `walk_obs` 3 blocks), a small
number of blocks fall on the wrong side of the best-fitting threshold. Under a
null in which the divergent set is an arbitrary subset of blocks of the observed
size, the joint probability of separation this good across all 20 pairs is on
the order of 10^-212.

**We report the 4 exceptions rather than describing the rule as exceptionless.**
At n = 3 the predicate held in 12 of 12 pairs with zero mismatches, and the
earlier draft of this section said "without exception". That statement does not
survive two more seeds and has been withdrawn. What survives is weaker and still
sufficient: onset tick explains divergence almost completely, and no competing
ordering (block index, disturbance status, environment index) explains any of the
residual variation once onset tick is known. The consequence for the selection
argument is stated below rather than buried.

One quantitative caveat carries over: the implied propagation delay is not a
single constant. The per-pair delay brackets admit no common value, so the delay
depends on the magnitude of the seeding perturbation and on the specific
dynamics, not on the solver alone.

## Magnitude on the registered estimand

The registered pre-onset negative control is a paired block-level fall-rate
difference in a window where the oracle has not engaged, so its true value is
exactly zero. Any non-zero estimate is residual imbalance or noise. At n = 5
(160 disturbed blocks per cell); see Table III of `paper/numbers.md`:

| cell | block bootstrap | process level (n=5) |
|---|---|---|
| stand_motor | +0.000 pp [+0.000, +0.000] | +0.000 pp [+0.000, +0.000] |
| stand_obs | +0.000 pp [+0.000, +0.000] | +0.000 pp [+0.000, +0.000] |
| walk_motor | +0.039 pp [+0.000, +0.117] | +0.039 pp [-0.069, +0.148] |
| walk_obs | +0.039 pp [+0.000, +0.117] | +0.039 pp [-0.069, +0.148] |

The two walking cells pass the frozen criterion by touching zero under the block
bootstrap, and straddle it properly under process-level inference. The
underlying count is small enough to name exactly: **pre-onset fall status
differs for 4 of 10,240 disturbed environment pairs**, the pairs the registered
estimand actually uses, and for 6 of 15,360 environment pairs counting the
nominal blocks the estimand never touches. Three disturbed blocks are affected,
none of them inside the leak-free subset. The two new processes contributed zero
additional discrepancies.

The largest residual, +0.039 pp, is 0.6% of the smallest primary effect in the
study (`walk_motor`, -6.60 pp) and 0.2% of the largest (`walk_obs`, +17.21 pp).

> **Note on a corrected number.** An earlier draft quoted this residual as 2
> disturbed environment pairs. That figure was hand-typed and wrong: the audit
> had only ever derived the all-blocks count, and the disturbed-only figure was
> never computed. It is now derived in
> `onset_residual.audit_replicate` as `pre_onset_fall_difference_environments_disturbed`,
> covered by a regression test, and emitted by `scripts/paper_numbers.py`. The
> correct figure at n = 3 was 4 of 6,144; at n = 5 it is 4 of 10,240.

## Contamination-free sensitivity analysis

Because divergence is very nearly the upper tail in onset tick, the
bit-identical blocks form a subset on which the two arms are provably identical
up to onset: the subset is defined by `max |onset_obs_u - onset_obs_o| == 0`
measured directly, not by the threshold model, so the 4 predicate exceptions
above do not put a contaminated block into the subset. Recomputing the
registered primary estimand there is a leakage-free replication of the headline.
It is post hoc and is reported as a sensitivity analysis, not as a replacement
for the registered estimand.

| cell | subset blocks | block bootstrap | process level (n=5) |
|---|---|---|---|
| stand_motor | 35 of 160 | -22.74 pp [-26.89, -18.63] | -22.80 pp [-27.49, -18.10] |
| stand_obs | 91 of 160 | +10.00 pp [+8.38, +11.65] | +9.90 pp [+5.42, +14.38] |
| walk_motor | 38 of 160 | -6.87 pp [-9.67, -4.16] | -7.06 pp [-8.62, -5.51] |
| walk_obs | 53 of 160 | +17.24 pp [+14.66, +19.90] | +17.08 pp [+12.65, +21.52] |

All four cells keep their sign and **all four exclude zero at both inferential
levels**. At the fault-family level the subset also survives process-level
inference: motor -15.09 pp [-19.81, -10.37], obs +12.52 pp [+10.79, +14.24],
interaction +27.61 pp [+21.33, +33.89].

**This is the claim that n = 3 could not support.** At three processes the
`stand_obs` leak-free cell gave +9.32 pp [-2.88, +21.52] at the process level, an
interval crossing zero, and the earlier draft reported it as not replicating at
that level. Two further pre-registered process seeds resolve it: the five process
estimates are +9.22, +4.46, +14.29, +9.92 and +11.61 pp, and the interval is
+9.90 [+5.42, +14.38]. The subsets are still the early-onset blocks rather than a
random sample, so this bounds the residual's influence without being an unbiased
estimate of the same population.

### Selection into the subset, and what it can and cannot bias

The obvious objection is that the subset was selected in a way that could
manufacture the result. Subset membership is very nearly onset tick itself:
across all 960 blocks, membership equals `1{onset_tick <= threshold}` for a
single per-arm-pair threshold in 956 cases, with 4 exceptions in 2 of the 20
pairs. Onset tick is drawn at design time, before either arm runs, so
conditioning on membership is overwhelmingly conditioning on a coarsening of a
pre-randomised covariate, which opens no collider path.

The 4 exceptions mean we can no longer say membership is *exactly* a
deterministic function of a pre-treatment covariate, and we do not say it. Two
things bound what those 4 blocks can do. They are 0.4% of the sample. And the
subset itself is defined by a measured bit-identity, not by the fitted threshold,
so an exception is a block the threshold model mispredicts, not a contaminated
block admitted to the clean set: the count of leak-free disturbed pairs carrying
a pre-onset discrepancy is 0.

That reduces the question to one channel: effect modification by onset tick. At
the process level it is null in all four cells, with implied extrapolation bias
of -1.43, -0.69, -0.31 and -0.51 pp against effects of -23.53, +9.89, -6.60 and
+17.21 pp. Three further checks agree. Subset-minus-complement differences cross
zero in all four cells (+0.72, +1.06, -0.56, -0.05 pp). A threshold sweep that
discards the fitted threshold and takes `{onset <= q}` on a grid from q = 115 to
q = 195 is flat, so the result is not an artefact of where the threshold happened
to land. Arm-label permutation within environment pairs on the leak-free subset
gives p = 2e-4 in all four cells, the minimum resolvable at 5,000 permutations.

**Post-onset exposure is a block-level constant.** Onset is shared by both arms
in all 960 blocks, so every within-block contrast is evaluated at identical
exposure and window length cannot bias it. Exposure differences between subset
and complement act only through effect modification by onset, the null channel
bounded above.

## The observation-fault effects are horizon-conditional

The episode horizon is 500 ticks and onset is drawn from [100, 200], so the
post-onset window varies from 300 to 400 ticks by design. In the observation
cells the unshielded arm is still falling near the horizon while the oracle arm
has stopped: 1.61% of `stand_obs` and 1.76% of `walk_obs` unshielded environment
pairs fall at elapsed tick 300 or later, against 0.00% for the oracle in both.

The dependence is monotone and has not plateaued by W = 300. Sweeping the
window: `stand_obs` runs +1.80 (W=100), +5.54 (W=200), +8.28 (W=300), +9.89
(full); `walk_obs` runs +6.27, +11.06, +15.46, +17.21. The motor cells are flat
across the same sweep (-22.87 to -23.61, and -7.22 to -6.60). **The sign is
horizon-invariant; the observation magnitude is not.** Because the maximum onset
is 200, W = 300 truncates no block, so the remaining gap is falls landing later
than onset + 300: roughly 16% of the reported `stand_obs` effect and 10% of
`walk_obs` is late-window exposure that the common-horizon check does not remove.

Recomputing on that common 300-tick window, the longest every block can supply in
full, moves `stand_obs` from +9.89 to +8.28 pp and `walk_obs` from +17.21 to
+15.46 pp; the motor cells move by 0.08 pp. At the fault-family level, motor goes
-15.07 to -15.15 pp and obs +13.55 to +11.87 pp, both still excluding zero at the
process level. Every sign and every gate outcome is unchanged, but the
observation effect sizes are a function of how long the episode is watched, and
we report them as such rather than quoting only the full-window figure.

## The outcome is falls, and only falls

The registered outcome is the post-onset fall rate and nothing else. This must be
stated in the paper rather than left for a reviewer to derive, because the
artifacts contain a field that looks like a second outcome and is not:
`task_complete` is defined as `~fell` (`scripts/reliability_closed_loop.py:1191`),
so `task_complete + fell = 1` exactly in every cell. It is the primary outcome
restated, not corroboration, and it may not be quoted as a secondary outcome. The
one other candidate, `return_until_first_fall`, is unusable in the observation
cells because the injected observation noise enters the reward terms directly
(`stand_obs` median -1588.76 unshielded against -26.93 oracle), so it measures
the corruption rather than the task.

The consequence bounds the claim. The fallback under test is a null-action
controller: at full blend the commanded action is exactly zero
(`reliability_closed_loop.py:1134`), pinning joint targets to the default stand
pose, open-loop with respect to observations. Under observation corruption that
severs the fault's only causal pathway, which is why the oracle arm's post-onset
fall rate is 0.51% (`stand_obs`) and 2.78% (`walk_obs`) against 38.18% and 59.30%
in the motor cells. A walking robot that freezes into a stance stops falling and
also stops walking, and **this study measures no cost for that**. "Benefit" here
means "fewer post-onset falls" and cannot be read as "safer", "better", or
"preferable"; the paper says so in the introduction, not only in the limitations.

## Provenance details a reviewer will check, stated before they ask

- **`bundle_id` differs between `process_01-03` and `process_04/05`** (`stand`:
  `a867c991a28fccff` against `1d09a70edc057bc9`). The bundle id hashes the code
  commit as well as the inputs. Diffing `bundle.json`, the only fields that differ
  are `code_commit` and `code_dirty`; every entry in `files{}` (checkpoint, ONNX,
  shield artifact, resolved env config) is byte-identical across all five
  processes. Nothing experimental changed.
- **`code_dirty` is `true` on all 16 new arm runs** and `false` on the original
  24. The 11-file `source_snapshot_sha256` in `EXPERIMENT_SOURCE_PATHS` is
  identical across all 40 arms, so the experimental surface was clean; the dirty
  flag reflects the untracked output directories the run itself was creating.
- **"5 independent process seeds" means exactly one thing.** Each replicate
  re-seeds the environment, torch and numpy RNGs
  (`reliability_closed_loop.py:552-555`); the seed is shared by the four cells
  within a replicate, which is why the process, not the cell, is the clustering
  unit. It is not an independent re-instantiation of a policy, an environment, a
  fallback design or a fault model. The generalization is over RNG streams, and
  the Limitations section scopes the claim to one policy family, one simulator and
  one fallback accordingly.

## What we do not claim

We do not claim the harness is bit-exact. It is not: most arm pairs diverge in a
majority of blocks at onset. We claim that the divergence enters after reset
through a mechanism identified by a positive test rather than by elimination,
that the test now has 4 exceptions in 960 blocks which we report, that the
residual's effect on the registered pre-onset negative control is at most
+0.039 pp with a process-level interval straddling zero, and that the primary
effects reproduce in sign and magnitude on the blocks it provably did not reach.
Eliminating the residual entirely would require one physics batch per block, a
48-fold increase in simulator launches, judged not worth the compute against a
residual of this size.

## Reproducing this section

```
# registered estimand and gate, n=5
PYTHONPATH=src python scripts/reliability_replication.py analyze \
  --registry reliability_eval/causal_viability_replication_v2/registry_n5.json \
  --output  reliability_eval/causal_viability_replication_v2/combined_summary_n5.json

# residual audit, n=5
PYTHONPATH=src python scripts/reliability_onset_residual.py \
  --registry reliability_eval/causal_viability_replication_v2/registry_n5.json \
  --output  reliability_eval/causal_viability_replication_v2/onset_residual_audit_n5.json

# the full selection-bias suite, including process-level inference
cd analysis/selection_bias && \
  REGISTRY=../../reliability_eval/causal_viability_replication_v2/registry_n5.json \
  OUT=RESULTS_n5.txt ./run_all.sh

# every number this section quotes, rendered from the three artifacts above
PYTHONPATH=src python scripts/paper_numbers.py --out paper/numbers.md
```

CPU only. No GPU and no simulator re-run: all four read the frozen arm arrays.
