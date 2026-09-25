# Phoenix walking policy: simulation result (2026-09-25)

PhoenixVelocity uses a 45-dimensional proprioceptive observation, a velocity
command, and a checkpoint manifest that records the training and deployment
contract. The code in this branch covers training, evaluation, a MuJoCo sim2sim
gate, and the controller handoff. No Phoenix walking checkpoint has run on the
real GO2.

## Seed 46 checkpoint selection

The 9,000-iteration run learned useful behavior early, then collapsed. The
held-out Isaac Lab evaluation used 256 environments for 1,050 steps. Values
below come from the recorded evaluation output and the training diagnostic
table; generated checkpoints and raw logs are not versioned in this branch.

| Iteration | Training mean episode length | Isaac falls | Stand height, mean | Slow-command linear error, mean |
|---:|---:|---:|---:|---:|
| 1,000 | 1,000.0 steps | 0/256 | 0.326 m | 0.151 m/s |
| 1,500 | 990.8 steps | 1/256 | 0.329 m | 0.145 m/s |
| 8,500 | 7.04 steps | 23,947/23,947 evaluated episodes | not selected | not selected |

At iterations 1,000 and 1,500, a fixed-command Isaac sweep found no calf
hard-limit proximity at +0.6 or +0.8 rad/s yaw. This check targeted the
counter-clockwise turn that exposed a calf-limit failure in an earlier seed.
The later checkpoint is unusable: its evaluated episodes all ended in falls.

## MuJoCo gate v3

Both early checkpoints failed the frozen v3 gate. All simulated scenario safety
checks passed, but the manifest and command-envelope checks failed. The
manifest records training ranges of x velocity [-0.3, 0.6] m/s, y velocity
[-0.2, 0.2] m/s, and yaw [-0.5, 0.5] rad/s. The proposed gate requires
symmetric coverage of |x| 0.5 m/s, |y| 0.3 m/s, and |yaw| 0.8 rad/s. The
manifest validator also rejects the y and yaw deploy caps because they exceed
the training envelope. The joint-sign audit passed.

The v3 gate therefore blocks hardware deployment. Narrowing deploy commands to
the trained envelope and running the gate again is a separate evaluation; no
passing verdict is claimed here. The controller and actuator changes in this
branch have offline tests, but no new real-robot walking result.

See [the gate definition and scenario results](sim2sim_gate.md) for the earlier
seed 42/44 diagnoses and positive controls.
