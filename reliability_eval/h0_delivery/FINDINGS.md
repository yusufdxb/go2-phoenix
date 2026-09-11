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
| FAIL_NOT_DISTINCT | 12 | attitude, collapse, slip, stumble |
| NOT_DELIVERABLE | 3 | contact_loss |
| RESOLVE_FAILED | 2 | onset at row 0, bridge correctly refuses |
| NO_ONSET | 5 | all five real-hardware captures |

> **Estimator corrected 2026-09-11, same day, after an adversarial re-analysis of
> this probe.** The first cut anchored each comparison on the single row 0, which
> is one noisy draw. At these effect sizes that anchor was unstable: the excursion
> fraction's median flipped sign (-1.3% to +0.33%) depending on the anchor, and the
> "direction correct in 4 of 12" figure it produced is a coin flip
> (binomial P(X<=4 | n=12) = 0.19), not evidence of anti-correlation as the first
> write-up implied. The reported statistic is now a z-score against each
> trajectory's own history strictly before the seeded row. **The verdict did not
> change and the evidence for it is stronger.** `FAIL_DIRECTION` disappeared as a
> category because direction carries no information when the state is not distinct.

## Why it fails: the offset overshoots the failure

Every synthetic trajectory is built as exactly 50 stable rows followed by a
failure segment (the synthetic pool's generator, `n_stable=50` in all six
generators). The control period is 0.02 s, so the production 0.5 s offset
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

The measurement confirms this rather than assuming it. Each seeded value is
scored against the distribution of the same descriptor over the trajectory's own
rows strictly before the seeded row. That window is causal and pool-agnostic, and
where the seeded row sits inside the failure segment it absorbs failure rows,
which widens the window and makes the test harder to pass. So it is conservative
for the cases that pass.

| mode | z of seeded state vs its own pre-seed history |
|---|---|
| attitude | -0.57, +0.71, +0.23 |
| collapse | +1.38, -0.56, -0.54 |
| slip | +1.24, -0.54, +0.78 |
| stumble | -0.58, -0.08, +0.46 |
| contact_loss | -0.64, +0.84, -0.04 (undeliverable regardless) |
| **command_mismatch** | **+3.06, +3.49, +3.16** |

**Zero of the 15 non-command_mismatch trajectories reach |z| > 2.** Every one of
them seeds a state that is statistically indistinguishable from a random row of
its own stable prefix. That is not a delivered treatment; it is the Phase-I
behaviour at a different row index.

## The one mode that works, and why

`command_mismatch` passes 3/3 at z = +3.06, +3.49, +3.16, covering 92% to 114% of
the excursion. Its failure segment is 60 rows long and its flag does not raise
until 30 rows in, so onset minus 25 rows still sits 5 rows *inside* the failure.
It is the
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

Worse, four of the seven fields `restore_state` needs are zero in every capture.
In `gate7_live_2026-04-21_18-33-17.parquet` (5,961 rows):

| field | state | why |
|---|---|---|
| `base_quat` | real | IMU |
| `base_ang_vel_body` | real | IMU gyro |
| `joint_pos`, `joint_vel`, `action` | real | motor state |
| `base_pos` | zero | hardcoded `np.zeros(3)` at `ros2_policy_node.py:517` |
| `base_lin_vel_body` | zero | hardcoded `np.zeros(3)` at `ros2_policy_node.py:457`, logged at `:519` |
| `contact_forces` | zero | hardcoded `np.zeros(4)` at `ros2_policy_node.py:525` |
| `command_vel` | zero | **not a defect**, see below |

Seeding from this today would place the robot at the terrain origin at z = 0 with
zero body velocity.

**`command_vel` is not a capture bug.** It is correctly wired: `:307` initializes
it and `_on_cmd_vel` at `:354-357` updates it from `/cmd_vel`, and it is logged
live at `:523`. It is zero because these were stand tests and nothing published
`/cmd_vel`. That is a genuine recorded zero, not a missing field. It is still a
data trap worth closing, because a zero column cannot be distinguished from
"operator commanded zero" by any downstream consumer; a `teleop_active` column
would separate them.

**The other three are recoverable now, and the comment that justifies zeroing
them is wrong for this robot.** `ros2_policy_node.py:510-512` says base position
and contact forces "aren't observable on stock GO2 without odometry / foot
sensors". On this platform both are published:

* `unitree_go/msg/LowState` line 10 declares `int16[4] foot_force`, and
  `lowstate_bridge_node.py:63` already subscribes to exactly that message on
  `/lowstate`. It extracts `motor_state` and `imu_state` at `:74-88` and drops
  `foot_force` on the floor. Contact force is one field access away.
* `/utlidar/robot_odom` publishes `nav_msgs/Odometry` at 151 Hz
  (`docs/go2_field_notes.md:49`), which carries both base position and body twist.

So the capture path is cheaply fixable rather than blocked on absent sensors.
That materially improves the outlook: the reason no real failure can enter the
loop is a wiring gap of a few lines plus one extra subscription, not a hardware
limitation.

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
   extract `foot_force` from the `LowState` the bridge already receives, subscribe
   `/utlidar/robot_odom` for base position and body twist, and invoke the failure
   detector in the deploy path so `failure_flag` is actually evaluated. Today
   `FailureDetector` is never imported by `sim2real`: `ros2_policy_node.py:44`
   imports only `FailureThresholds` for the attitude gate, and `:526-527` writes
   `failure_flag=False, failure_mode=None` as hardcoded literals. Correct the
   stale comment at `:510-512` while doing it.

Item 5 is the one that matters most for the project's story, because until it is
fixed no real robot failure can ever be fed back into training.
