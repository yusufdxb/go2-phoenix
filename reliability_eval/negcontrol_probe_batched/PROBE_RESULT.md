# Stage 1 probe, fourth attempt: `--batched-blocks`

Run 2026-09-02. Cell `stand_obs`, protocol seed 2026074002, process seed 2026074101, 48 blocks,
16 environments per block, both arms. Protocol blocks verified **bit-identical** to
`negcontrol_probe_invalidate` and therefore to
`causal_viability_replication/process_01/stand_obs` (`blocks identical: True`, 48 blocks).

Change under test: a design change, not a flag on the old design. The three previous probes all
tried to scrub state between sequentially replayed blocks. This one removes the sequence. Every
block is given its own private slice of a single larger environment batch, so environment `i`
belongs to block `i // 16` and lives exactly one block. There is no previous block for simulator
state to carry from. All per-block scalars (onset tick, disturbance magnitude, disturbed flag)
became per-environment vectors, and one pass over the 500 tick horizon runs all 48 blocks at once.

## Verdict: the pre-onset negative control PASSES, exactly

```
reset_state  global max abs diff = 0.0000000000 | blocks nonzero =  0/48 | block0 = 0.0
initial_obs  global max abs diff = 0.0000000000 | blocks nonzero =  0/48 | block0 = 0.0
onset_obs    global max abs diff = 2.8715746403 | blocks nonzero =  4/48 | block0 = 2.5504631996
pre-onset negative control, all 48 blocks     = +0.0000 pp  (u=4 falls, o=4, 0/48 vectors differ)
pre-onset negative control, 32 disturbed only = +0.0000 pp
```

The frozen gate criterion is `pre_onset_negative_controls_include_zero`. This is the strictly
stronger outcome that Section 5 of `NEGATIVE_CONTROL_ANALYSIS.md` named as the pass condition:
**exactly 0.000 pp with bit-identical pre-onset fall vectors in every block**, not merely an
interval that includes zero. Both arms record the same 4 pre-onset falls, in the same 4
environments.

Comparison against the third probe, same cell, same seeds, same blocks:

| Quantity | probe 3 (invalidate) | this probe (batched) |
|---|---:|---:|
| `initial_obs` blocks nonzero | 0/48 | 0/48 |
| `onset_obs` blocks nonzero | 47/48 | **4/48** |
| `onset_obs` global max abs diff | 6.4477 | 2.8716 |
| pre-onset negative control, 48 blocks | +0.9115 pp | **+0.0000 pp** |
| blocks with differing pre-onset fall vector | 9/48 | **0/48** |

## The residual, measured rather than assumed

`onset_obs` is still not identical in 4 of 48 blocks, so the probe's own secondary acceptance
clause ("`onset_obs` exactly 0.0 in all 48 blocks") is not met. The residual has a clean structure:

```
divergent blocks : 0, 2, 33, 47      onset ticks 191, 197, 189, 188
clean blocks     : the other 44      onset ticks 100 .. 186
```

Divergence at onset occurs **if and only if the block's onset tick is at least 188**, and when it
occurs it affects all 16 environments of that block. It is not associated with disturbance status
(2 of the 4 are nominal), nor with pre-onset falls (the 4 pre-onset falls sit in blocks 1, 9, 27
and 44, all of which are clean).

Mechanism: all 768 environments share one GPU physics batch. The earliest onsets are at tick 100,
and from that tick the oracle arm starts engaging its fallback, so the two arms' batches stop being
identical. Numerical differences then propagate across the shared solver until, after roughly 88
ticks, they reach the environments that are still pre-onset. Nothing carries between blocks, because
no environment has a second block; the coupling is spatial within one tick, not temporal across
blocks.

This was confirmed, not inferred, by a determinism control: the unshielded arm was run a second time
in a separate process from the identical command
(`reliability_eval/negcontrol_probe_batched_rerun/`). **Every recorded array is bit-identical
between the two runs.** The simulator is therefore run-to-run deterministic for a fixed execution
history, so the 4 block residual cannot be run noise and must come from the arms differing.

The residual does not touch the estimand it could have biased: it moved no pre-onset fall in any
block, which is why the negative control is exactly zero rather than merely small.

## The headline survives the fix

Registered estimand, paired block-level post-onset fall-rate difference among jointly onset-eligible
environment pairs, unshielded minus oracle. Eligibility excluded 4 of 768 pairs.

| Set | This probe | Frozen v1, same cell |
|---|---:|---:|
| All 48 blocks | **+6.02 pp** [+3.91, +8.30] | |
| 32 disturbed blocks | **+9.04 pp** [+6.45, +11.89] | |
| v1 frozen `stand_obs` headline | | +6.21 pp [+4.91, +7.56] |

Positive means the fallback reduces post-onset falls. Direction and magnitude reproduce under a
negative control that is now exactly zero, so the v1 estimate for this cell was not an artefact of
the leak. Intervals here are 20,000 sample environment-pair bootstraps (seed 20260824) and, as in
v1, understate uncertainty because the 16 environments in a block share its sampled disturbance.

## Cost

One pass of 500 ticks at 768 environments, against 48 sequential passes of 500 ticks at 16
environments. Measured wall clock is about one minute per arm, against 216.9 s and 209.4 s for the
two sequential arms of probe 3. The full 24 arm-run study should therefore cost well under the
1.53 h the sequential design took.

## What is not established here

- Only `stand_obs` was run. The other three cells are untested under batching.
- The 4 block onset residual is characterised but not eliminated. If a future analysis needs
  `onset_obs` identical in all 48 blocks, the batch would have to be partitioned so that no
  environment is pre-onset while another is post-onset.
- This is a probe, not the study. `study_id phoenix_causal_viability_replication_v2` has not been
  frozen or run.
