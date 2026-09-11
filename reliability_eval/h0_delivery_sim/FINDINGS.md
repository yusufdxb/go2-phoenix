# Harvested sim failures: the substrate is real, the gate criterion is wrong

Run 2026-09-11. Reproduce with:

```
python scripts/harvest_sim_failures.py \
    --checkpoint checkpoints/phoenix-flat-v4/latest.pt \
    --env-config configs/env/flat_v4.yaml \
    --num-envs 64 --num-failures 32 --max-steps 6000 --pre-onset-steps 600 \
    --push-velocity 1.0 --push-interval-s 2.0 4.0
python scripts/h0_delivery_probe.py --pool data/failures/sim_harvest
```

## What was built

`scripts/harvest_sim_failures.py` records genuine failure trajectories from
rollouts: the policy walks, a mid-episode push disturbs it, and the episodes that
end in a fall are written in the standard trajectory Parquet schema. Every field
the replay pipeline needs is observed rather than invented, and the same
`FailureDetector` that labels real robot telemetry labels these, so simulator and
hardware failures enter the curriculum through one code path.

Verified end to end on the harvested files: `TrajectoryPool.from_directory` loads
them, mode filtering works (the mechanism phase II's mode-conditioned arm needs),
`FailureCurriculum.assign(64)` at fraction 0.5 returns 32 seeded and 32 nominal,
`resolve_seed` finds a row 0.50 s before onset, and `load_initial_state` returns
all-finite fields. The pool is consumable.

## Two bugs this run found, both in the harvester

**Mid-episode versus reset disturbance.** The first attempt used the
`perturbation:` overlay, which modulates a reset-mode event term. It produced 24
falls between step 5 and step 33, every one with onset at row 0, and the reset
bridge correctly rejected all 24 for having no pre-onset interval. Driving the
interval-mode `push_robot` term instead produces falls at steps 539, 1102 and
1674, after 11 to 33 seconds of ordinary walking.

**Auto-reset contamination.** Isaac resets a terminating environment inside
`env.step()`. Reading termination state before the step is one step stale, and
reading it after returns a fresh spawn for exactly the environments that just
failed. The first windows therefore held 320 rows of a frozen 0.155 m height
followed by a 0.400 m spawn spike. `PreResetCapture` fixes it. Post-fix the same
check shows zero spawn-height rows, 66 distinct heights across the pre-onset
window, and a smooth monotonic collapse over the final rows.

## The yield is the binding constraint

| outcome | count |
|---|---:|
| terminations seen | 148 |
| harvested | **3** |
| rejected, onset within 100 rows of episode start | 71 |
| detector never fired | 74 |

Two measured problems, neither of them a harvester bug.

**The rule-based detector misses about half of real falls.** 74 of 148
terminated episodes produced no `FailureDetector` event, despite Isaac's
`base_contact` term firing. This is a recall measurement on genuine falls and it
matters well beyond the harvest: that same detector is the thing now wired into
the deploy path to label real hardware captures. If it misses half the falls on
the robot, half the hardware failures will still go unlabelled.

**Half the falls happen too soon after reset** to leave a usable pre-onset
window, which is the same structural problem the synthetic pool had.

## H0 still fails, but the criterion is measuring the wrong thing

| trajectory | mode | onset | seed row | z | verdict |
|---|---|---:|---:|---:|---|
| sim_fall_0000 | slip | 172 | 147 | -0.10 | FAIL_NOT_DISTINCT |
| sim_fall_0001 | slip | 239 | 214 | -0.48 | FAIL_NOT_DISTINCT |
| sim_fall_0002 | collapse | 588 | 563 | +2.55 | FAIL_DIRECTION |

An offset sweep over 0.0 to 2.0 s passes 0 of 3 at every offset. But the pattern
is not the synthetic pool's pattern:

* the two slip cases move in the predicted direction at nearly every offset and
  simply never reach |z| > 2
* the collapse case's |z| grows with offset, +2.55 at 0.5 s to +3.95 at 2.0 s,
  while the direction inverts, because further back in time the robot was
  genuinely standing higher

That last row is the tell. **Seeding further before onset finds a healthier
robot, which is the entire point of pre-onset seeding, and my criterion scores it
as a failure.** The preregistered H0 asks whether seeded states are distinct from
*nominal reset states*. During the robustness pass I re-anchored on each
trajectory's own prior history, which is statistically sounder but answers a
different question: whether the seeded state is already exhibiting the failure. A
curriculum that seeds states already exhibiting the failure can only teach
recovery, never avoidance.

Against the criterion as preregistered, these states pass clause (a) easily:
nominal reset is height 0.400 m with zero joint velocity, and these sit at 0.154
to 0.204 m with joint velocity norms around 7.6.

So the correct conclusion is not "the harvested substrate fails H0". It is that
**H0 needs restating before it can judge a pre-onset seeding strategy at all**,
and that restatement has to happen before Gate B, not after.

## What the synthetic-pool result still establishes

Nothing here rehabilitates the synthetic pool. There the seeded state was a
random healthy walking row drawn from the same distribution as row 0 of the same
trajectory, unrelated to the failure, with an excursion fraction of about -1% and
direction at chance. Here the seeded state lies on the actual trajectory into the
failure with excursion fractions of 15% and 41%. Those are different situations.

With n = 3 the harvested pool is far too small to conclude anything statistically,
and it is not yet a usable curriculum pool.

## Next, in order

1. **Raise detector recall**, or replace the label source with the simulator's own
   termination terms. 50% recall is the binding constraint on yield and it also
   caps how much real hardware data can ever be labelled.
2. **Restate H0** so it asks the preregistered question: distinct from a nominal
   reset, and on a trajectory that leads to the failure. Not "already failing".
3. **Then scale the pool**, which is a parameter change once 1 and 2 are settled.
4. Only then Gate B.
