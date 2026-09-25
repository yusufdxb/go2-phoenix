# Sim2sim gate (MuJoCo) for Phoenix walking checkpoints

A checkpoint goes to the gantry only after it passes this gate. The gate drives the
exported ONNX through the same observation builder and action post-processing the
hardware node uses, on a MuJoCo GO2 with the real actuator limits, and returns one
PASS/FAIL verdict.

## One command

```bash
PYTHONPATH=src python scripts/sim2sim_gate.py \
    --onnx <run_dir>/policy.onnx \
    --manifest <run_dir>/phoenix_manifest.json \
    --out runs/sim2sim_gate/<name>
```

Needs `mujoco`, `onnxruntime`, `numpy`, `pyyaml`, and for video `imageio[ffmpeg]`.
The gate runs on CPU. A full run (9 scenarios,
about 125 simulated seconds at 1 kHz, plus video) takes about 30 s of wall time.

Exit code: `0` PASS, `1` FAIL, `3` DIAGNOSTIC (a gate setting was overridden, so no
verdict is issued), `2` setup error (unreadable manifest, ONNX input size does not
match the manifest).

Options:

| flag | effect |
|---|---|
| `--no-video` | skip the MP4s |
| `--gate v1\|v2\|v3` | gate version, default `v3` |
| `--gate-config <yaml>` | explicit gate file, overrides `--gate` |
| `--scenarios a b` | subset; makes the run DIAGNOSTIC |
| `--latency-ms X` | target delay; makes the run DIAGNOSTIC |
| `--action-clip X\|none` | override the deploy clamp; makes the run DIAGNOSTIC |

## Inputs

`--onnx`: one input `[batch, obs_dim]`, one output (`action` preferred). Observation
normalization must be inside the graph (as `phoenix.sim2real.export` writes it).

`--manifest`: one of three schemas.

1. `phoenix-checkpoint-manifest/v1` (the normal case, from
   `phoenix.velocity.contract.build_manifest`). The observation is
   `phoenix.velocity.observation.build_actor_observation` (45-D) and the targets
   `actions_to_joint_targets` after the manifest's action clip. The manifest is validated
   with `validate_manifest_for_mode(..., "velocity")` against the gate's required
   command envelope; any problem fails the gate. Optional deploy block:

   ```json
   "deploy": {"kp": 25.0, "kd": 0.5, "action_clip": 100.0}
   ```

   `kp`/`kd` may be a scalar, 12 values in `joint_order`, or a mapping by group
   (`{"hip": .., "thigh": .., "calf": ..}`) or by joint name; both must be given.
   `action_clip` must be > 0 or `null` (no clamp; fails gate v2 SAFETY). Without the
   block the gate uses Kp 25 / Kd 0.5 / clip 1.0 and records each as an assumption.
2. `phoenix-legacy-checkpoint-manifest/v0` (H25 only): 48-D, `base_lin_vel` fed
   zeros, then the same 45-D builder.
3. `phoenix-sim2sim-deploy-spec/v1`: explicit spec for a policy from another stack
   (rl_sar-style terms and scales, optional newest-first history). Examples:
   `configs/sim2sim/positive_control/*.json`.

## Output

`<out>/gate_report.json`, `<out>/videos/<scenario>.mp4` (480x360, 25 fps, rendered on
the CPU with Mesa llvmpipe EGL so it never touches the GPU; if rendering fails the
report says `video: null` and gives `video_error`).

Report keys: `verdict`, `would_be_verdict`, `failures` (list of
`<scenario>:<check>` or `spec:<check>`), `diagnostic_reasons`, `gate` (config path,
sha256, the commit that last touched it), `policy` (ONNX and manifest sha256), `spec`
(everything the deploy path used, including assumptions), `spec_checks`,
`joint_audit`, and `scenarios.<name>.{metrics, checks, pass, video}`.

## What the gate simulates

- Plant: MuJoCo Menagerie GO2 (`assets/mujoco/unitree_go2`, see
  `phoenix.sim2sim.model.MENAGERIE_PROVENANCE`), profile `real_go2`: published
  inertials, armature hip/thigh 0.01 and calf 0.02 (unitree_rl_mjlab), no passive
  damping or friction loss, condim 3, foot friction 0.8 unless a scenario sets it.
- Control: policy at 50 Hz; PD `tau = kp (q_des - q) - kd qd` at 1 kHz from the
  current joint state; DCMotor torque-speed clip per group: hip/thigh 23.5 Nm and
  30.1 rad/s, calf 45.43 Nm and 15.70 rad/s (IsaacLab#7479, Unitree URDF).
- Start: the spec's default joint pose, lowest foot 5 mm above the floor, zero
  velocity (the state after a FixStand ramp to the trained pose).
- Fall: base height < 0.20 m or |roll| or |pitch| > 0.8 rad.

## Gate v3 (default)

`configs/sim2sim/gate_v3.yaml` was introduced after v2 passed the collapsed
seed42/seed44 @8999 checkpoints. It is post hoc for those checkpoints and was
set before the seed46 run. It is v2
plus one blocking SAFETY check, `responsiveness`: in `forward_0p3`, `forward_0p5` and
`yaw_0p6` the achieved mean of vx (or wz) over the tracking window must have the
command's sign and be >= 0.5 x |command| (0.15 m/s, 0.25 m/s, 0.30 rad/s). A fall fails
it. The check value is the achieved fraction `sign(cmd) * mean / |cmd|`.

v3 results (responsiveness fractions for fwd 0.3, fwd 0.5, yaw 0.6):

| policy | v3 SAFETY | responsiveness | other SAFETY failures |
|---|---|---|---|
| rl_sar robot_lab | PASS | 1.00, 1.02, 1.14 | none |
| rl_sar himloco | FAIL | 0.92, 0.94, **0.00** | none (does not turn in place) |
| H25 stand | FAIL | -0.01, -0.01, 0.00 | as v2 |
| walk-v1 seed42 @8999 (post-hoc) | FAIL | 0.00, 0.00, 0.00 | none |
| walk-v1 seed44 @8999 (post-hoc) | FAIL | 0.00, 0.00, 0.00 | none |
| walk-v1 seed42 @3000 (post-hoc) | FAIL | 0.93, 1.00, 0.84 | yaw_0p6 near_limit_fraction.calf 0.033 |

## Finding: seed42 @3000 calf near-limit in yaw_0p6 is real policy behavior

- Model ranges match: MuJoCo calf range = URDF = Isaac USD = (-2.7227, -0.83776) on all
  four legs. No model mismatch, nothing changed in the model.
- Which limit: the UPPER calf limit (knee near straight), FR_calf and RL_calf (one
  diagonal). RL_calf reaches 0.0001 rad from the hard stop, FR_calf 0.0124. The lower
  limit is never approached (margin >= 1.01 rad).
- The policy requests it: calf targets up to 0.246 rad PAST the hard upper stop (RL,
  30 control steps; FR, 17). Measured calf is past Isaac's soft upper limit (-0.932,
  factor 0.9) on 32 (FR) and 22 (RL) steps. Events cluster at t = 2 to 3 s, 7 to 8 s.
- Robust to the plant: worse on the `isaac_matched` profile (near fraction 0.335,
  0.035 rad into the MuJoCo soft stop), present with 20 ms latency, larger at +0.8
  rad/s (targets 0.313 rad past the stop). Absent for clockwise yaw (-0.6, -0.8: margin
  >= 0.16 rad) and at +0.3 rad/s.
- Why Isaac reports 0.0011: `scripts/eval_velocity.py` counts joints
  within 5 % of the SOFT range of the SOFT limits, averaged over all 12 joints, all envs
  and randomly sampled commands. The gate counts control steps where ANY calf is within
  0.05 rad of the HARD limit, in one scenario (pure +0.6 rad/s turn in place). The
  gate's band lies entirely outside Isaac's soft range. Not comparable numbers.
- Deploy consequence: at +0.6 rad/s the requested calf target exceeds the hard limit
  by more than the earlier H25 abort band (`LIMIT_ABORT_BAND_RAD` 0.175) on 5 of
  600 steps (19 at +0.8). The current deploy path clips such requests and
  latches only on sustained clipping. The candidate still fails the near-limit
  gate. The behavior is a policy result, not a model or threshold artifact.
- Not verified: the same trace in Isaac at a fixed +0.6 rad/s command (GPU reserved).

## Gate v2

`configs/sim2sim/gate_v2.yaml` was set before the walk-v1 checkpoint evaluation.
It uses the same plant, scenarios and numbers as v1, with checks split into tiers.

- SAFETY (blocking, the verdict): no fall and finite actions in every scenario;
  pre-clip saturation <= 0.05 measured as `|raw| >= deploy.action_clip` (no clip =
  FAIL); torque saturation <= 0.05 per group; zero hard-limit violations; near-limit
  fraction <= 0.02 per group; stand height >= 0.24 m; manifest valid; command envelope
  covers |vx| 0.5, |vy| 0.3, |wz| 0.8 in both signs (the first lab session's commands);
  joint sign audit.
- PERFORMANCE (reported, non-blocking): v1 tracking thresholds unchanged (0.15 m/s,
  0.20 rad/s; stress 0.25, 0.30), with named rows `stand_yaw_drift` (stand_20s) and
  `turn_in_place` (yaw_0p6). Report key `tiers.performance`.

Report: `verdict` is the SAFETY verdict; `failures` are SAFETY failures;
`performance_failures` and `tiers.performance.{rows, named, verdict}` hold the rest.

## Gate v1

Thresholds and scenarios are in `configs/sim2sim/gate_v1.yaml`. They were set
before any policy was run through the gate. A changed gate requires a new file.

Scenarios: `stand_20s`, `forward_0p3`, `forward_0p5`, `lateral_0p2`, `yaw_0p6`,
`lateral_step` (0.3 m/s lateral steps with a sign flip), `friction_0p4`,
`friction_1p0`, `payload_2kg`.

Checks per scenario: no fall; finite actions; pre-clip saturation rate
(|raw action| > clip) <= 0.05; torque saturation <= 0.05 per joint group; zero
hard-limit violations; near-limit (< 0.05 rad) fraction <= 0.02 per group; velocity
tracking RMSE on 0.5 s smoothed body velocity, skipping 1.5 s after each command
change (nominal: 0.15 m/s and 0.20 rad/s; stress: 0.25 and 0.30); stand only: mean
base height >= 0.24 m.

Spec checks: manifest valid for velocity mode; the trained command envelope covers
|vx| 0.5, |vy| 0.3, |wz| 0.6 in both signs; joint sign audit passes.

Metrics also reported (not gated): peak applied and demanded torque over limit per
group, joint velocity over limit fraction, minimum limit margin per group, max |raw
action|, |raw| > 1 rate, planar displacement, tracked mean vx/vy/wz, time to fall.

## Joint order and sign audit

`phoenix.sim2sim.joint_audit`, run inside every gate call and in
`tests/test_sim2sim_joint_audit.py`:

- Orders derived by name: Isaac breadth-first (`JOINT_ORDER`), per-leg FL/FR/RL/RR
  (MuJoCo MJCF, Newton, mjlab, legged_gym), Unitree SDK FR/FL/RR/RL.
  per-leg to SDK = `[3,4,5,0,1,2,9,10,11,6,7,8]`, identical to unitree_rl_mjlab's
  GO2 `joint_ids_map`; Isaac to SDK = `[3,0,9,6,4,1,10,7,5,2,11,8]`.
- Sign: with the base pinned, +0.1 rad through the spec's own deploy path on each
  policy joint must move only that joint by about +0.1 rad and move that leg's foot
  as the axis requires (hip: foot +y; thigh: foot -x; calf: leg extends).
- Limit: the audit trusts joint NAMES. It catches sign inversions and wrong stated SDK
  maps, not a manifest that names the policy's joints in the wrong order; that is
  established by the manifest being written from the training config.

## Validation under gate v2 (2026-09-24)

| policy | SAFETY verdict | performance |
|---|---|---|
| rl_sar go2/robot_lab | PASS | FAIL: stand_yaw_drift 0.204 > 0.20 (turn_in_place 0.100 PASS) |
| rl_sar go2/himloco | PASS | FAIL: turn_in_place 0.597 > 0.20 (stand_yaw_drift 0.012 PASS) |
| Phoenix H25 stand | FAIL: manifest, envelope x3, stand height 0.229 m, pre-clip saturation 0.82 to 0.98 in all 9 scenarios | FAIL |

The v1 rerun reproduced the v1 numbers below. Raw generated videos and
checkpoints are not included in this branch.

## Validation under gate v1 (2026-09-24)

| policy | real-robot evidence | verdict | what failed |
|---|---|---|---|
| rl_sar go2/robot_lab | rl_sar README (maintainer) | FAIL | stand_20s yaw rate RMSE 0.204 > 0.20: turns at +0.19 rad/s with zero command |
| rl_sar go2/himloco | rl_sar README (maintainer) | FAIL | yaw_0p6 yaw RMSE 0.597: does not turn in place (turns only while translating) |
| Phoenix H25 stand (clamp 1.0) | fell on F1 | FAIL | spec checks; stand height 0.229 m; pre-clip saturation 0.82 to 0.98 in every scenario |
| H25 without clamp (DIAGNOSTIC) | F1 deploy path | would-be FAIL | falls at 0.54 s; calf torque saturated 75 % |

Neither hardware-attested policy passes gate v1; each fails one behavior. Both walk
without falling in every scenario, with no pre-clip saturation and at most 0.5 %
calf torque saturation. A mirrored-policy diagnostic flipped the robot_lab drift sign
(+0.192 to -0.198 rad/s), so the drift comes from the policy, not from a harness
asymmetry.

## Positive control for the gantry

Use rl_sar `go2/robot_lab` or `go2/himloco` with rl_sar's own deployer
(`rl_real_go2`), per the numbers above: robot_lab tracks 0.3 and 0.5 m/s forward and
0.6 rad/s yaw but will not stand still; himloco stands still but turns only while
walking. Their real-robot status comes from the upstream maintainers; Phoenix
has not repeated those runs on this robot. unitree_rl_mjlab ships no GO2
checkpoint in the version examined for this comparison.
