# Stage 1 probe, third attempt: `--reset-invalidate-buffers`

Run 2026-08-27. Cell `stand_obs`, protocol seed 2026074002, process seed 2026074101, 48 blocks,
16 environments, both arms. Change under test: after `env.reset()`, every lazy
`TimestampedBuffer` on the articulation data is stamped `-1.0` (the documented not-updated-yet
sentinel) and the observation is recomputed, instead of being served from the previous block's
cache. This is a distinct mechanism from the two settle probes: nothing is stepped.

Protocol blocks verified bit-identical to `causal_viability_replication/process_01/stand_obs`
(`blocks identical: True`, 48 blocks). The only substantive parameter difference is
`reset_invalidate_buffers` false to true; `reset_settle_ticks` stays at `0`.

## Verdict: the reset-observation leak is CLOSED. The probe still FAILS acceptance.

The pre-registered acceptance test was cross-arm `max abs diff` of BOTH `initial_obs` and
`onset_obs` exactly `0.0` in all 48 blocks. Measured:

```
reset_state  global max abs diff = 0.0000000000 | blocks nonzero =  0/48 | block0 = 0.0
initial_obs  global max abs diff = 0.0000000000 | blocks nonzero =  0/48 | block0 = 0.0
onset_obs    global max abs diff = 6.5035066605 | blocks nonzero = 47/48 | block0 = 0.0
pre-onset negative control, all 48 blocks     = +0.9115 pp   (u=8 falls, o=1, 9/48 vectors differ)
pre-onset negative control, 32 disturbed only = +1.1719 pp
```

`initial_obs` half of the acceptance test PASSES outright. `onset_obs` fails.

## What the fix demonstrably did

Per-dimension cross-arm `max abs diff` of `initial_obs`, this probe against the frozen v1 run of
the identical cell and seeds:

| Observation block | v1 frozen | this probe |
|---|---:|---:|
| base_lin_vel (0:3) | 0.261549 | **0.000000** |
| base_ang_vel (3:6) | 3.561405 | **0.000000** |
| projected_gravity (6:9) | 1.418244 | **0.000000** |
| velocity_command (9:12) | 0.0 | 0.0 |
| joint_pos (12:24) | 0.0 | 0.0 |
| joint_vel (24:36) | 0.0 | 0.0 |
| last_action (36:48) | 0.0 | 0.0 |

The stale-buffer channel identified in `NEGATIVE_CONTROL_ANALYSIS.md` Section 1.3 was real, it was
exactly the 9 dimensions predicted, and invalidating the buffers removes it completely. The harness
logged `[cl] invalidated 26 lazy data buffers after reset`, both arms completed all 48 blocks
(unshielded 216.9 s, oracle 209.4 s), and no `FAIL CLOSED` guard fired.

## What the fix did not do, and what that proves

`onset_obs` and the negative control barely moved:

| Quantity | v1 frozen | this probe |
|---|---:|---:|
| `onset_obs` global max abs diff | 6.447702 | 6.503507 |
| `onset_obs` blocks nonzero | 47/48 | 47/48 |
| pre-onset negative control, 48 blocks | +1.0417 pp | +0.9115 pp |
| blocks with differing pre-onset fall vector | 8/48 | 9/48 |

So the arms now enter every block with a bit-identical observation and still arrive at the
registered onset tick, 100 to 200 ticks later, in materially different physical states. Per-block
divergence at onset:

```
block  0: nonzero elems    0/768
block  1: nonzero elems  637/768  proj_gravity=0.839 joint_pos=1.369 joint_vel=0.707 last_action=2
block  2: nonzero elems  639/768  proj_gravity=0.724 joint_pos=1.418 joint_vel=0.956 last_action=2
block  5: nonzero elems  667/768  proj_gravity=0.488 joint_pos=1.482 joint_vel=0.978 last_action=2
block 20: nonzero elems  653/768  proj_gravity=0.572 joint_pos=1.228 joint_vel=0.787 last_action=2
block 47: nonzero elems  630/768  proj_gravity=0.133 joint_pos=0.995 joint_vel=0.523 last_action=2
```

The decisive control is the nominal blocks. The oracle arm never engages in a nominal block
(measured: `oracle engaged envs in nominal blocks = 0`, against 512 engaged environments overall),
so in those 16 blocks the two arms execute the identical control law on a bit-identical starting
observation. They still diverge:

```
disturbed blocks (32): onset_obs nonzero in 31/32
nominal   blocks (16): onset_obs nonzero in 16/16,  max abs diff 3.286269
```

Identical inputs, identical control, divergent trajectories. The remaining channel therefore
cannot be the observation assembly, cannot be the fallback, and cannot be the RNG, all three of
which are now excluded by measurement rather than by argument.

**Conclusion: there is a second cross-block carryover channel, and it lives in simulator state that
the harness neither resets nor records.** `reset_state` (root state 13 + joint pos 12 + joint vel
12) is bit-identical in all 48 blocks of both arms, so the entry state that the harness can see is
clean; the entry state the solver actually has is not. ASSUMPTION, not verified here: the residual
carrier is PhysX-internal state that survives a state write, for example contact manifolds, solver
warm-start caches, or contact-sensor history. Distinguishing among those would need instrumentation
this harness does not have, and the result would not change the remedy.

Block 0 was never evidence that the observation channel was the only channel. Block 0 has no
predecessor, so every carryover channel is simultaneously clean there. This probe is what separates
them, and it separates them cleanly.

## Status of the frozen gate

`pre_onset_negative_controls_include_zero` FAILED in v1 and stays FAILED. Nothing here re-specifies
it. The criterion was not relaxed at any point, and this probe was scored against the acceptance
test exactly as it was written in `NEGATIVE_CONTROL_ANALYSIS.md` Section 5, Stage 1.

## Where this leaves the re-run

Three distinct mechanisms have now been tried against the pre-onset leak and all three failed the
same binary acceptance test:

1. Zero-action settle (`8401416`): treatment-dependent abort. Worse than the leak.
2. Policy-action settle (`0434a23`): abort fixed, leak amplified from 9 to 45 contaminated dims.
3. Buffer invalidation (this probe): observation channel fully closed, trajectory divergence and
   the negative control essentially unchanged.

Per the project's own stop rule, no fourth in-place mechanism is attempted. The three attempts have
jointly established the shape of the problem: the leak is not in how the observation is assembled,
it is in the environment being reused across blocks at all.

The only remaining option that is leak-free by construction is the one
`NEGATIVE_CONTROL_ANALYSIS.md` already named and this probe now promotes from fallback to sole
candidate: **a fresh environment, or a fresh process, per block**, so that no previous block exists
whose state can carry. Cost is 48 simulator startups per arm rather than one. That is a design
change to the harness and a re-costing of the full study, not a parameter, and it is not made here.

`phoenix_causal_viability_replication_v2` is NOT started. Starting it under any of the three tried
mechanisms would reproduce the leak.

## Reproducing

```
python scripts/reliability_closed_loop.py --freeze \
  --out-dir reliability_eval/negcontrol_probe_invalidate \
  --disturbance obs --policy-name stand --cell-id stand_obs --replicate-id probe_invalidate \
  --protocol-seed 2026074002 --process-seed 2026074101 \
  --reset-settle-ticks 0 --reset-invalidate-buffers

for arm in unshielded oracle; do
  python scripts/reliability_closed_loop.py --arm $arm \
    --out-dir reliability_eval/negcontrol_probe_invalidate \
    --disturbance obs --policy-name stand --cell-id stand_obs --replicate-id probe_invalidate \
    --protocol-seed 2026074002 --process-seed 2026074101
done
```

Run on an NVIDIA (Blackwell) consumer GPU, IsaacLab 4.5.22, torch 2.10.0+cu128, rsl_rl 5.0.1,
numpy 1.26.4. Total GPU time 7 min 6 s across the two arms, plus simulator startup. All numbers in
this document are elementwise reductions over `arm_unshielded.npz` and `arm_oracle.npz` in this
directory with numpy only.

## Mechanism verification, done before the run rather than after

Read from the installed IsaacLab source, not from memory:

- `isaaclab/utils/buffers/timestamped_buffer_warp.py`: `TimestampedBufferWarp.__init__` sets
  `self.timestamp = -1.0`, documented as "indicating that the buffer has not been updated yet".
  The sentinel is real.
- `isaaclab_physx/assets/articulation/articulation_data.py`: every lazy property guards on
  `if self._<buf>.timestamp < self._sim_timestamp`, so `-1.0` forces recompute for any
  `_sim_timestamp >= 0`, and `_sim_timestamp` starts at `0.0` and only increases.
- The recompute reads the simulation, not the cache: `root_com_vel_w` calls
  `self._root_view.get_root_velocities()`; `root_com_lin_vel_b` and `root_com_ang_vel_b` rotate
  that by `root_link_quat_w`; `projected_gravity_b` is derived from `root_link_quat_w`.
- All the buffers are instance attributes created in `_create_buffers`, so the harness's
  `vars(data)` walk reaches them. It reported 26.

One known side effect, recorded because it is a real change and not a nil one: `_joint_acc` is a
finite-difference buffer and invalidating its timestamp perturbs its `time_elapsed`. `joint_acc`
enters only the `dof_acc_l2` reward term, never the 48-dim policy observation and never a
termination, and the perturbation is arm-identical by construction, so it cannot produce the
divergence reported above.
