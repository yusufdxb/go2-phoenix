# The onset residual: measured, bounded, and reported

Source of every number below: `reliability_eval/causal_viability_replication_v2/onset_residual_audit.json`,
produced by `scripts/reliability_onset_residual.py --registry reliability_eval/causal_viability_replication_v2/registry.json`.
The registered gate output it is compared against is `combined_summary.json` in the same directory.
This file is drafting material for Section 6 (Threats to Validity), not final typeset prose.

## What is and is not aligned across the paired arms

The batched-block harness removes temporal carryover by construction: block `i` owns
environments `i*16 .. i*16+15` and lives for exactly one block, so no block has a
predecessor whose simulator state it could inherit. Measured across all 12 process-cell
arm pairs of the v2 replication:

- Reset states are bit-identical between arms. Maximum absolute difference is exactly
  0.0 in 12 of 12 pairs.
- Initial observations are bit-identical between arms. Maximum absolute difference is
  exactly 0.0 in 12 of 12 pairs.
- Onset observations are not. Between 4 and 42 of 48 blocks per pair carry a non-zero
  difference, with per-pair maximum absolute differences from 2.87 to 13.18.

The v1 leak channel (a stale reset observation carrying the previous block's terminal
state) is therefore closed. What remains is a smaller channel that acts after reset.

## The residual is positively measured, not inferred by elimination

Earlier characterisation attributed the residual to simulator-internal history by
elimination, after ruling out observation, reset state, per-block RNG, action-term memory
and actuator-delay memory. That is an argument from a shrinking list, not a measurement.
We replace it with a falsifiable prediction and its test.

If the channel is within-tick coupling through the single shared GPU PhysX batch that
advances all 768 environments together, then whether a block's onset observation diverges
must be decided by that block's onset **tick** alone: a perturbation seeded when the
earliest-onset environments are treated needs a finite number of ticks to reach
environments that have not yet reached their own onset, so divergence must be an upward
closed set in onset tick, with one threshold per arm pair. Temporal carryover across
blocks would instead order divergence by block index, which the batched harness does not
define, and disturbance status or environment index would order it if the channel were
per-environment.

The prediction holds exactly. **In 12 of 12 arm pairs a single onset-tick threshold
separates the divergent blocks from the bit-identical ones with no exceptions**: every
block with onset at or above the threshold diverges and every block below it is
bit-identical. Divergence is uncorrelated with block index, disturbance status, and
environment index once onset tick is known. Under a null in which the divergent set is an
arbitrary subset of blocks of the observed size, the joint probability of perfect
separation in all 12 pairs is on the order of 10^-124.

The mechanism is therefore identified as within-tick spatial coupling in the shared
physics batch, not temporal carryover. One quantitative caveat: the implied propagation
delay is not a single constant. The per-pair brackets on the delay, in ticks after the
earliest onset in the batch, run from (12, 18] to (86, 88] and admit no common value, so
the delay depends on the magnitude of the seeding perturbation and on the specific
dynamics rather than on the solver alone.

## Magnitude on the registered estimand

The registered pre-onset negative control is a paired block-level fall-rate difference in
a window where the oracle has not engaged, so its true value is exactly zero. Measured on
v2, pooled over three processes (96 disturbed blocks per cell):

| cell | pre-onset negative control | 95% CI |
|---|---|---|
| stand_motor | +0.000 pp | [+0.000, +0.000] |
| stand_obs | +0.000 pp | [+0.000, +0.000] |
| walk_motor | +0.065 pp | [+0.000, +0.195] |
| walk_obs | +0.065 pp | [+0.000, +0.195] |

The two walking cells pass the frozen criterion by touching zero, not by straddling it,
and we state it that way. The underlying count is small enough to name exactly: across
all 12 pairs, pre-onset fall status differs for **6 of 9,216 environment pairs**, and
within the disturbed blocks the registered estimand actually uses, for **4 of 6,144
environment pairs**. The largest residual, +0.065 pp, is 1.1% of the smallest primary
effect in the study (walk_motor, -5.95 pp) and 0.4% of the largest (walk_obs, +17.52 pp).

## Contamination-free sensitivity analysis

Because divergence is exactly the upper tail in onset tick, the bit-identical blocks form
a subset on which the two arms are provably identical up to onset. Recomputing the
registered primary estimand on that subset alone is a leakage-free replication of the
headline. It is post hoc and is reported as a sensitivity analysis, not as a replacement
for the registered estimand.

| cell | registered (n=96) | contamination-free subset |
|---|---|---|
| stand_motor | -23.73 pp [-26.32, -21.20] | -23.28 pp [-30.56, -16.57] (n=17) |
| stand_obs | +9.61 pp [+8.09, +11.23] | +9.27 pp [+6.98, +11.70] (n=44) |
| walk_motor | -5.95 pp [-7.87, -4.03] | -7.19 pp [-10.40, -4.00] (n=24) |
| walk_obs | +17.52 pp [+15.31, +19.88] | +16.19 pp [+13.19, +19.28] (n=32) |

All four cells keep their sign, all four contamination-free intervals exclude zero, and
every registered point estimate falls inside its contamination-free interval. The
sign-flip between fault families, which is the paper's contribution, survives on data the
residual cannot have touched. The subsets are small, so the intervals are wider, and the
subsets are the early-onset blocks rather than a random sample, so this bounds the
residual's influence without being an unbiased estimate of the same population.

### Selection into the subset, and what it can and cannot bias

The obvious objection is that the subset was selected in a way that could manufacture the
result. It was not, and the reason is structural rather than statistical. Subset membership
is not merely correlated with onset tick, it **is** onset tick: across all 576 blocks,
membership equals `1{onset_tick <= threshold}` for a single per-arm-pair threshold with
zero exceptions, and across the 458 distinct (arm pair, onset tick) strata not one stratum
contains blocks of differing membership. Onset tick is drawn at design time, before either
arm runs, so conditioning on membership is conditioning on a coarsening of a pre-randomised
covariate and opens no collider path.

That reduces the whole question to one channel: effect modification by onset tick. It is
absent in all four cells (Pearson and Spearman both, all p > 0.3), which bounds the implied
extrapolation bias at -0.84, -0.07, -1.18 and +0.60 pp against effects of -23.73, +9.61,
-5.95 and +17.52 pp. Three further checks agree. Subset-minus-complement differences cross
zero in all four cells. A threshold sweep that discards the fitted threshold and takes
`{onset <= q}` on a grid from q=115 to q=195 is flat, so the result is not an artefact of
where the threshold happened to land. Block-level sign-flip permutation at the registered
analysis unit gives p = 3.05e-05 (exact, n=17), 0, 3.95e-04 and 0.

Two caveats belong here rather than in a rebuttal.

**Post-onset exposure is a block-level constant.** Onset is shared by both arms in all 576
blocks, so every within-block contrast is evaluated at identical exposure and window length
cannot bias it. Exposure differences between subset and complement act only through effect
modification by onset, which is the null channel bounded above.

**The intervals above are conditional on the three process seeds.** The block bootstrap
resamples blocks within each process and never resamples processes, so between-process
variance is absent from every interval in this document. Recomputing at the process level
(t interval on three process means) is materially wider, and for one cell it matters:

| cell | subset, block bootstrap | subset, process-level (n=3) |
|---|---|---|
| stand_motor | -23.28 pp [-30.56, -16.57] | -23.36 pp [-34.89, -11.83] |
| stand_obs | +9.27 pp [+6.98, +11.70] | +9.32 pp [-2.88, +21.52] |
| walk_motor | -7.19 pp [-10.40, -4.00] | -7.56 pp [-10.96, -4.16] |
| walk_obs | +16.19 pp [+13.19, +19.28] | +16.13 pp [+5.93, +26.34] |

The `stand_obs` subset replication does not survive process-level inference; its three
process estimates are +9.22 (n=30), +4.46 (n=7) and +14.29 (n=7). We state that rather than
report the narrower interval alone. The registered gate is unaffected, because its only
interval-based criterion is at the fault-family level, and there the subset survives
process-level inference: motor -15.06 pp [-27.42, -2.70], obs +12.08 pp [+7.70, +16.47],
interaction +27.14 pp [+10.41, +43.87]. All four registered full-sample cells also exclude
zero at the process level.

**The observation-fault effects are horizon-conditional.** The episode horizon is 500 ticks
and onset is drawn from [100, 200], so the post-onset window varies from 300 to 400 ticks by
design. In the observation cells the unshielded arm is still falling near the horizon while
the oracle arm has stopped: 1.31 percent of `stand_obs` and 1.99 percent of `walk_obs`
unshielded environment pairs fall at elapsed tick 300 or later, against 0.00 percent for the
oracle in both. Recomputing on a common 300-tick window, the longest every block can supply,
moves `stand_obs` from +9.61 to +8.31 pp and `walk_obs` from +17.52 to +15.54 pp; the motor
cells move by at most 0.13 pp. Every sign and every gate outcome is unchanged, but the
observation effect sizes are a function of how long the episode is watched, and we report
them as such.

## What we do not claim

We do not claim the harness is bit-exact. It is not: 11 of 12 arm pairs diverge in a
majority of blocks at onset. We claim that the divergence enters after reset through a
mechanism we have identified by a positive test rather than by elimination, that its
effect on the registered pre-onset negative control is at most +0.065 pp with an interval
touching zero, and that the primary effects reproduce in sign and magnitude on the blocks
it provably did not reach. Eliminating the residual entirely would require one physics
batch per block, which is a 48-fold increase in simulator launches and was judged not
worth the compute against a residual of this size.

## Reproducing this section

```
PYTHONPATH=src python scripts/reliability_onset_residual.py \
  --registry reliability_eval/causal_viability_replication_v2/registry.json \
  --output reliability_eval/causal_viability_replication_v2/onset_residual_audit.json
```

CPU only. No GPU and no simulator re-run: the audit reads the frozen arm arrays.
