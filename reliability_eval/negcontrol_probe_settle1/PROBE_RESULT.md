# Stage 1 probe result: reset_settle_ticks = 1

Run 2026-08-24. Cell `stand_obs`, protocol seed 2026074002, process seed 2026074101,
48 blocks, 16 envs. Protocol blocks verified bit-identical to
`causal_viability_replication/process_01/stand_obs`; the only substantive parameter
difference is `reset_settle_ticks` 0 to 1. All four controller artifacts hash-match the
frozen bundle.

## Verdict: PROBE FAILED. One settle tick does not close the leak, and it introduces a
## second, worse problem.

The acceptance test (cross-arm max abs diff of `initial_obs` and `onset_obs` exactly 0.0
in all 48 blocks) was never reached. The unshielded arm aborted during the reset settle
of block 11:

```
[cl] block 10/48 id=9 disturbed fell=3/16 pre=0 post=3 engaged=0/16
Traceback (most recent call last):
  File "scripts/reliability_closed_loop.py", line 696, in run_arm
    raise RuntimeError("FAIL CLOSED: environment terminated during reset settling")
RuntimeError: FAIL CLOSED: environment terminated during reset settling
```

The harness guard at L695-696 fired correctly. Full log: `probe_run.log`.

## The finding that matters more than the abort

The settle loop at L691-697 steps the environment with ZERO actions. For a standing
policy, zero action is not a hold, it is a release: the robot collapses and the
environment terminates.

That failure is ARM-ASYMMETRIC, which is the exact property the negative control exists
to rule out. Evidence from this run: the unshielded arm died in the settle of block 11,
while the oracle arm ran past block 24 without a single settle termination before it was
stopped manually. The oracle fallback holds the robot through the settle tick; the
unshielded arm has nothing holding it.

So `reset_settle_ticks = 1` as currently implemented would replace a stale-buffer leak
with a treatment-dependent termination, which is strictly worse: the leak biases
estimates, an arm-dependent abort destroys the pairing outright.

## What this rules in and out

- RULED OUT: the one-parameter fix. Stage 2 cannot be a straight re-run.
- NOT YET TESTED: settling with the POLICY action instead of zeros. This is the obvious
  next candidate and is a small change at L691-697: run the same policy forward for the
  settle ticks rather than commanding zeros. It keeps both arms in their nominal control
  regime during the settle, so it should not be arm-asymmetric. UNVERIFIED.
- STILL AVAILABLE: the fallbacks named in NEGATIVE_CONTROL_ANALYSIS.md, a fresh
  environment per block or one process per block. Both are leak-free by construction
  because there is no previous block to go stale, at a cost of 48 simulator startups.

## Recommended next probe

Change the settle loop to step the policy rather than zeros, then re-run this identical
probe. Acceptance is unchanged and still binary: cross-arm max abs diff exactly 0.0 in
all 48 blocks, both arms completing. Cost is the same 10 to 15 minutes.

ASSUMPTION, unverified: that a policy-action settle refreshes the stale root buffers at
all. If it does not, the per-block fresh environment is the fallback, and the leak
question is settled by construction rather than by a parameter.
