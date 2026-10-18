# Friction-aware walking retrain, 2026-10-17 (simulation evidence)

**Simulation only. The hardware A/B/C comparison on the GO2 has not been performed.**

The first walking commands on the GO2 under-tracked (about 60 % of the forward distance and
20 % of the commanded turn). An active breakaway probe on the robot measured joint static
friction that the training simulator did not model (see `EVIDENCE.md`). This directory
holds the evidence for two policies retrained with a joint-friction actuator model, and
for the incumbent policy they are compared against.

## The three policies

| Arm | Checkpoint | What changed | Seed |
|---|---|---|---|
| A | `walk_vnext3c_seed47/model_2750.pt` | Incumbent walking policy; no joint friction in training | 47 |
| B | `fm_B_s62/model_3749.pt` | +1000 iterations from A with `joint_friction_model: hardware`: per-joint Coulomb friction from the probe measurements, scaled x0.75 to x1.6, per-env jitter 0.15, load-dependent term 0 to 0.1, viscous 0 to 0.05 per joint group | 62 |
| C | `fm_C_s60/model_3749.pt` | +1000 iterations from A with `joint_friction_model: broad`: Coulomb friction drawn uniformly from 0 to 1.6 N m (not tied to the measured values), load-dependent term 0 to 0.1, viscous 0 to 0.1 | 60 |

B and C resume from A's weights and curriculum state with the same command curriculum,
rewards and other randomization; only the friction model and seed differ (full configs:
`training/*/spec.json`; Isaac Lab, 4096 envs, PPO, 50 Hz control). Both finished the
curriculum at level 8 of 8 (`training/*/curriculum_final_state.json`). The legacy
`joint_friction_enabled: false` key in the specs is the earlier friction randomization,
which the code rejects in combination with `joint_friction_model`.

Three seeds were trained per arm (B: 60, 61, 62; C: 60, 61, 62) and several checkpoints
of each were evaluated. `summary.md` lists all 26 candidates. B62_3749 and C60_3749 are
the best of their arm by `seen err + unseen err` on the MuJoCo evaluation below,
so their MuJoCo numbers come from the same evaluation they are ranked on.

## Results

`abc_table.md` has the full rows for the three arms; `summary.md` has every candidate.

MuJoCo replay over the identified friction family (physics hypotheses that reproduce the
hardware walks; `err` = mean |1 - achieved/commanded| over profiles):

| Arm | seen err (hardware presets) | unseen err (other magnitudes) | robust err (points outside the family) | falls | vx03_4s ratio | yaw05_3s ratio |
|---|---|---|---|---|---|---|
| A | 0.585 | 0.562 | 0.411 | 0 | 0.54 | 0.22 |
| B | 0.258 | 0.228 | 0.145 | 0 | 0.84 | 0.59 |
| C | 0.128 | 0.113 | 0.082 | 0 | 0.98 | 0.71 |

Isaac Lab, eval seed 7, two environments shared by all arms: "hardware friction" is B's own
training spec (the identified joint-friction model), "frictionless" is A's own training spec
(no joint friction). Each of those arms is therefore scored partly in its training
environment. Lower error is better; ratio 1.0 = on command.

| Arm | env | slow lin err (m/s) | yaw err (rad/s) | yaw_p0.6 ratio | vx_0.5 ratio |
|---|---|---|---|---|---|
| A | hardware friction | 0.108 | 0.092 | 0.70 | 0.79 |
| B | hardware friction | 0.062 | 0.073 | 0.88 | 0.96 |
| C | hardware friction | 0.063 | 0.073 | 0.99 | 1.07 |
| A | frictionless | 0.055 | 0.069 | 0.91 | 0.98 |
| B | frictionless | 0.082 | 0.080 | 1.08 | 1.13 |
| C | frictionless | 0.114 | 0.085 | 1.17 | 1.23 |

B and C track better than A when friction is present and overshoot when it is absent: they
have learned to push through friction. Which world the robot is closer to is what the
hardware comparison has to decide.

## Export and parity

Each arm was exported to ONNX and checked against the PyTorch policy on 2,000 observations
(`candidates/*/export/parity_gate.json`; tolerance max |diff| 1e-5, cosine 0.9999). All three
pass the sim-to-sim gate v3 (`candidates/*/export/sim2sim_gate/gate_report.json`).

| Arm | max abs diff | worst-sample cosine | sim2sim gate |
|---|---|---|---|
| A | 1.19e-06 | 0.99999999999915 | PASS |
| B | 1.19e-06 | 0.99999999999963 | PASS |
| C | 1.67e-06 | 0.99999999999975 | PASS |

## Artifacts and hashes

`SHA256SUMS` lists the B and C training checkpoints and all three ONNX exports. The binaries
are not in this repository; they are kept by the author and match these hashes (also
recorded in each `phoenix_manifest.json`).

## Provenance

| Commit | What it holds |
|---|---|
| `d76d6f7` | joint-friction actuator model, its Isaac Lab and MuJoCo implementations, the identified friction family |
| `fc25b3f` | candidate evaluation driver (export, sim2sim gate, parity, MuJoCo); recorded in A's manifest |
| `8246211` | Isaac Lab evaluation driver; recorded in B's manifest |
| `c3a33d8` | candidate summary with the Isaac table; recorded in C's manifest |

These commits are on a development branch that is not yet published, so the code that
produced these files cannot be rebuilt from this repository yet. Absolute local paths in the
JSON files were shortened to repository-relative paths; no values were changed.

## Not shown

- Any result on the robot for B or C. The hardware A/B/C comparison has not been performed.
- That the identified friction family is the robot's true dynamics: it is fitted to a few
  short free-floor walks.
- Whether the overshoot in the frictionless environment matters on the robot.
