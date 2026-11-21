# 03 — Phase C: sim-to-real validation (interim, 2026-10-12)

Status: **C1 (reproduce baseline) done offline. C2 partially done (existing perturbation
model only). C3/C4 BLOCKED (hardware).** No robot commands were sent.

## VERIFIED

### Replay provenance
- Replay script: `scripts/desk/sim_replay_walk.py`. It exists only on mewtwo,
  excluded via `.git/info/exclude`, sha256 `0a95d9d2…`; a copy is in `~/Projects/phoenix-friction-data/desk_scripts/`.
  Run from worktree `~/workspace/wt/phoenix-walk-v1-hw` @ `06c5a37`. `git diff ed50fe7 06c5a37` is
  empty for `src/phoenix/sim2sim`, `walk_command.py`, `configs/sim2sim/gate_v3.yaml` and the
  sim deploy config, so the simulation code equals the 10-07 state.
- MuJoCo 3.5.0, onnxruntime 1.24.4 (`~/Sim/isaac-sim-venv`), CPU only, deterministic, 1 initial
  condition per preset. MuJoCo `dof_frictionloss` = scale × measured Fs per joint (`fmeas`).
- Policy identity, sim vs robot: `policy.onnx` `6d36796f…` and `policy.onnx.data` `b6ec50ed…`
  are identical in the Archive (sim) and the Jetson bundle `walk2750-34104ff` (robot);
  `phoenix_manifest.json` `c6d4a320…` matches the robot's lock check. The deploy YAMLs differ only in
  file paths; `mode_switch.enabled: false`. Bridge PD kp 25 / kd 0.5 in both.
- Observation: 45-D `phoenix-velocity-obs/v2`, no base_lin_vel term. The policy node's
  "base_lin_vel … CONSTANT ZERO" warning is generic and doesn't apply to this schema; the same log confirms
  "45-D, no base_lin_vel". **Not a sim/real confound.**
- Outputs: mewtwo `~/workspace/friction-validation-repro/replay_20261012/`, copied to
  `~/Projects/phoenix-friction-data/replay_20261012/` (`*_results.json`, `*_summary.md`, logs).

### Hardware walking reference (session `20261007_34104ff860c8`)
| trial | commanded | measured | source | n |
|---|---|---|---|---|
| `W_vx03_4s` | 1.20 m | 0.71 m (59 %) | **not in any log**: `base_pos` all zero, `odom_valid=False` | 1 |
| `W_yaw05_3s` ×2 | 85.9° | 17.6°, 19.0° (20.5 %, 22.1 %) | IMU quaternion yaw. Gyro integral agrees within 1° (16.9°, 18.2°) | 2 |

Yaw was recomputed from `policy.parquet` (the same method as `~/phoenix/yaw_an.py`). The robot was
in a harness; the operator judgement for all three trials is "no".

### Replay results (forward = 4 s preset distance / 1.2 m; yaw = turn / 86°)
| condition | forward | yaw | note |
|---|---|---|---|
| nominal (no joint friction) | 89 % | 76 % | matches published |
| fmeas=1 | 66 % | 47 % | matches published upper yaw |
| fmeas=1, fstiff | 66 % | 34 % | matches published lower yaw |
| fmeas=1.25 / 1.25 fstiff | 60 % | 27 % / 25 % | |
| fmeas=1.5 / 1.5 fstiff | 55 % | 24 % / 23 % | |
| fmeas=2 | 45 % | 18 % | |
| **drag=30, spring=5 (no joint friction)** | **59 %** | **26 %** | trunk drag N·s/m + yaw spring N·m/rad, rope proxy |
| drag=25, spring=5 (no joint friction) | 62 % | 26 % | |
| calf=0.8, spring=4 (no joint friction) | 67 % | 19 % | calf kp/kd ×0.8 |
| unload=0.2, fmeas=1 | 66 % | 62 % | 20 % of weight carried by harness |
| robot | 59 % | 20–22 % | |

Full one-factor sweep: `defaultsweep_summary.md`. Foot friction (0.3–1.0) and actuator
latency (0–20 ms) each change tracking by ≤ 4 points forward and ≤ 12 yaw. They don't explain the gap.

## INFERRED
- **The published 89/76 → 66/34–47 numbers are reproduced exactly from code and policy
  identical to what ran on 10-07.**
- **The friction-scale hypothesis isn't identified by this data.** At least three different
  mechanisms each bring the replay to about 60 % / 20–26 %: (a) measured joint friction ×1.25–1.5,
  (b) harness drag plus yaw torsion with zero joint friction, (c) a weak calf actuator plus yaw torsion.
  Three short trials (one forward, two yaw), all in a harness, can't separate them.
- The harness is a confound in the walking comparison regardless of friction. Its force on
  the trunk was never measured. Harness unloading (`unload`) moves sim yaw *away* from the robot
  (62 % vs 47 % at fmeas=1), so the harness's net effect depends on its geometry.
- Sim yaw is far more sensitive to friction than sim forward tracking. A yaw-only discrepancy is
  therefore weak evidence for friction specifically.

## HYPOTHESES (untested)
- Load-dependent joint friction (the original hypothesis). It remains one of several candidate
  explanations.
- Harness rope drag/torsion of the magnitude above (25–30 N·s/m, 4–5 N·m/rad). It's physically
  unverified; the values were chosen only to show the explanations can't be told apart.
- Calf actuator torque deficit (`calf` gain factor below 1).

## BLOCKED / OPEN
- **0.71 m provenance:** ask the operator how it was measured (tape, video?). Until then the
  forward ratio is an unverified hand measurement, n=1.
- **C3/C4 (held-out hardware validation):** robot motion authority is held by another stack;
  requires approval per test. Prerequisite for any decisive test: walk **without a harness
  load path** (or with the harness force measured) and with independent position measurement
  (e.g. motion capture or a calibrated camera). Otherwise harness and friction stay confounded.
- **Friction-DR retrain already exists:** mewtwo `results/friction_dr/fric_T_s50–54`,
  `fric_C_s50–54`, `scripts/eval_friction_dr.py`, `friction_dr_promotion_guard.py`, and a gantry A/B
  campaign (`cb6180d`: "campaign not ready"). Not audited in this pass. Phase C5 must start from
  these, not from scratch.
