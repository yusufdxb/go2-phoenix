# Stage 1 probe, second attempt: policy-action settle, reset_settle_ticks = 1

Run 2026-08-24, 19:25 to 19:33. Same cell, seeds, and 48 blocks as the frozen
`causal_viability_replication/process_01/stand_obs`; blocks verified bit-identical.
Change under test: the settle loop steps the POLICY instead of commanding zeros.

## Verdict: the arm asymmetry is FIXED. The leak is NOT. Probe FAILED.

### What was fixed

Both arms completed all 48 blocks (unshielded 3 min 53 s, oracle 3 min 52 s). No
`FAIL CLOSED: environment terminated during reset settling`. The zero-action version died
in the settle of block 11 of the unshielded arm while the oracle arm sailed past block 24,
which was the disqualifying asymmetry. Stepping the policy removes it, as predicted: both
arms are in their nominal control regime during the settle and no fallback is applied.

### What was not fixed, and got measurably worse

Acceptance was cross-arm `max abs diff` of `initial_obs` and `onset_obs` exactly `0.0` in
all 48 blocks. Measured:

```
initial_obs: global max abs diff = 27.3604393005 | blocks nonzero = 47/48
onset_obs:   global max abs diff =  5.8050246239 | blocks nonzero = 47/48
block 0 max diff = 0.0
pre-onset negative control effect = +1.3021 pp (n=48 blocks, 11 blocks nonzero)
reset_state identical across arms = True
```

Block 0 is still the only clean block, exactly as in the frozen v1 study. The leak
signature is unchanged in kind and larger in degree: v1 showed 5.34 on 9 contaminated
dims, this run shows 27.36 on 45 of 48 dims.

### Why stepping made it worse, which identifies the real defect

`reset_state` is bit-identical across arms, so the PHYSICS state at reset is correct. The
contamination is in the OBSERVATION assembled at reset: root linear velocity, angular
velocity, and projected gravity are read from buffers still holding the previous block's
terminal values. That is the original diagnosis and it survives.

The settle step therefore feeds the policy an already-leaked observation. Different
observations produce different actions, different actions produce genuinely different
physics, and one tick of that propagates the contamination from 9 observation dims into 45,
including joint states that were previously clean. Settling does not overwrite the stale
buffer, it launders it into the dynamics.

This rules out the whole family of "step the environment a few times after reset" fixes.
More settle ticks would amplify further, not converge.

## What remains

1. RECOMMENDED: a fresh environment per block, or one process per block. Leak-free by
   construction, because there is no previous block whose buffers can go stale. Cost is 48
   simulator startups per arm rather than one. UNVERIFIED but structurally sound.
2. Refresh the observation buffers at reset without stepping: a second `env.reset()`, or
   assembling the reset observation directly from `robot.data` (which is demonstrably
   correct, since `reset_state` matches). Cheaper than option 1 if it works, and it targets
   the actual defect rather than routing around it. UNVERIFIED.
3. Keep v1's estimates and report the leak as a bounded bias. NOT RECOMMENDED: the leak
   tracks the treatment sign in all four cells, so it biases magnitudes away from zero,
   which is the direction that flatters the headline.

Option 2 is the next probe and is the same 8 minutes of GPU. Option 1 is the fallback that
cannot fail for structural reasons.

The pre-registered criterion was not re-specified at any point.
