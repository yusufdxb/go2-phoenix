# H0 delivery gate: FAILED

Run 2026-09-11 on `feat/causal-viability-replication`. Reproduce with:

```
python scripts/h0_delivery_probe.py
```

CPU only, no simulator. Raw record: `h0_delivery.json`. The acceptance criterion,
the descriptor set and the per-mode direction predictions were written into the
probe's docstring before the probe was run.

## Verdict

**H0 fails.** The August delivery fix is necessary but not sufficient. With the
production strategy (`failure_onset_minus_seconds`, offset 0.5 s) the curriculum
still seeds a nominal gait state for 5 of the 6 failure modes. Phase II must not
start until this is closed, for exactly the reason Phase I exists as a warning.

| verdict | n | modes |
|---|---:|---|
| PASS | 3 | command_mismatch (3/3) |
| FAIL_DIRECTION | 8 | attitude, collapse, slip, stumble |
| FAIL_NOT_DISTINCT | 4 | attitude, collapse, slip |
| NOT_DELIVERABLE | 3 | contact_loss |
| RESOLVE_FAILED | 2 | onset at row 0, bridge correctly refuses |
| NO_ONSET | 5 | all five real-hardware captures |

## Why it fails: the offset overshoots the failure

Every synthetic trajectory is built as exactly 50 stable rows followed by a
failure segment (`ashfall/src/ashfall/synth/generator.py`, `n_stable=50` in all
six generators). The control period is 0.02 s, so the production 0.5 s offset
steps back **25 rows** from onset. For five of six modes that lands back inside
the stable prefix:

| mode | onset row | seeded row | lands in |
|---|---:|---:|---|
| stumble | 50 | 25 | stable prefix |
| contact_loss | 55 | 30 | stable prefix |
| collapse | 62 | 37 | stable prefix |
| attitude | 69 | 44 | stable prefix |
| slip | 70 | 45 | stable prefix |
| command_mismatch | 80 | 55 | failure segment (5 rows in) |

Rows inside the stable prefix are produced by the same `_stable_row` call that
produces row 0. The seeded state is therefore a fresh draw from the *same*
distribution Phase I was seeding, just at a different index.

The measurement confirms this rather than assuming it. Taking each trajectory's
own row 0 as baseline and its onset row as the far end, the seeded state covers
this fraction of the nominal-to-failure excursion on the mode's own descriptor:

* all 18 resolvable trajectories: median **-0.6%**, range -8.9% to +112.5%
* excluding command_mismatch: median **-1.3%**, range **-8.9% to +3.7%**
* direction matches the mode's prediction in **4 of 12**, against a 50% chance rate

A treatment that moves the intended descriptor by about -1% of its available
range, in the predicted direction less often than a coin, is not a delivered
treatment.

## The one mode that works, and why

`command_mismatch` passes 3/3 with delivery fractions of 92.5%, 105.5% and
112.5%. Its failure segment is 60 rows long and its flag does not raise until 30
rows in, so onset minus 25 rows still sits 5 rows *inside* the failure. It is the
only mode whose failure develops slowly enough to survive a 0.5 s rollback.
This is a property of the data, not of the fix.

## contact_loss cannot be delivered by this bridge at all

`generate_contact_loss_failure` overwrites **only** `contact_forces`; it emits an
otherwise ordinary stable row. `phoenix.replay.state_adapter.restore_state`
writes root pose, root velocity, joint position, joint velocity and the velocity
command. It cannot write contact forces, and contact state is not a settable
initial condition. No choice of seed row delivers this mode. It is reported as
NOT DELIVERABLE rather than as a failure of the fix, and it should be dropped
from the Phase II mode set or re-generated with a signature that lives in the
robot state.

## Clause (a) passes, for a reason that does not help

H0's first clause asks whether seeded states are distinguishable from nominal
reset states. They are, but not because they are failures. Nominal reset is
height exactly 0.400 m, tilt exactly 0 deg and joint velocity exactly 0 (from
`UNITREE_GO2_CFG.init_state` and the `reset_base` / `reset_robot_joints` event
terms, which randomize x, y, yaw and root velocity but never roll, pitch, height
or joint velocity). The seeded states sit at height 0.294 to 0.307 m, tilt 0.49
to 2.26 deg, joint velocity norm 4.0 to 7.8.

So every curriculum reset already differs from a normal reset by about 10 cm of
body height and a nonzero joint velocity. That is the synthetic pool's walking
gait differing from the simulator's spawn pose. It is a distribution shift the
curriculum injects on every seeded environment, unrelated to failure, and it is
a confound Phase II should control rather than a sign of success.

## The real hardware captures are unusable as-is

All five real captures, including both Gate 7 live runs, return NO_ONSET: their
`failure_flag` column is False in every row and `failure_mode` is null
throughout, so `resolve_seed` correctly refuses them. The 2026-04-21 Gate 7 run
saturated slew at 33% on the real robot, and nothing in the recorded trajectory
marks it.

Worse, four of the seven fields `restore_state` needs were never populated by the
capture path. In `gate7_live_2026-04-21_18-33-17.parquet` (5,961 rows):

| field | state |
|---|---|
| `base_quat` | real (IMU) |
| `base_ang_vel_body` | real (IMU gyro) |
| `joint_pos`, `joint_vel`, `action` | real (motor state) |
| `base_pos` | **identically zero** |
| `base_lin_vel_body` | **identically zero** |
| `command_vel` | **identically zero** |
| `contact_forces` | **identically zero** |

Seeding from this today would place the robot at the terrain origin at z = 0,
with zero body velocity and a zero velocity command held for the episode. The GO2
publishes no base position or linear velocity that the logger subscribed to, so
this is a capture-path gap, not corruption.

## What has to change before Phase II

1. **Pick the seed row against the failure, not against the clock.** A fixed
   0.5 s rollback is wrong when failure development time varies by a factor of 6
   across modes. Either use `failure_onset_minus_steps` with a small offset, or
   define the offset as a fraction of each trajectory's own pre-onset failure
   development window. Whatever is chosen, this probe is the gate it must pass.
2. **Re-generate or drop contact_loss.** Its signature is not in the delivered
   state.
3. **Regenerate the pool with a longer stable prefix**, or accept that the
   seedable window is 5 to 30 rows wide and set the offset accordingly.
4. **Control the gait-versus-spawn-pose shift** in clause (a), or the Phase II
   treatment arm carries a 10 cm height offset the control arm does not.
5. **Fix the hardware capture path** before any real failure can enter the loop:
   populate `base_pos`, `base_lin_vel_body`, `command_vel` and `contact_forces`,
   and run the failure detector over the capture so `failure_flag` is set.

Item 5 is the one that matters most for the project's story, because until it is
fixed no real robot failure can ever be fed back into training.
