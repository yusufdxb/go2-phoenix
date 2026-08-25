# Native (C++) Runtime Audit, go2-phoenix deploy path

**Scope.** Read-only architecture audit of the deployment path, done to establish what a native C++
runtime must reproduce bit-for-bit and what should stay in Python. Target architecture: Python trains
and evaluates, C++ executes on the robot, a deterministic native safety layer holds final authority.

**Branch audited:** `feat/in-training-rate-limiter` @ `3f8af81`, working tree dirty (uncommitted
reliability research; not modified by this audit).
**Method:** every claim below cites `file:line` from a file actually read. Nothing was executed except
read-only introspection of `deploy/*.onnx` and `deploy/*.npz`. No training, no GPU job, no git state change.

---

## 1. Training path, do not disturb

| Component | File | Note |
|---|---|---|
| PPO runner | `src/phoenix/training/ppo_runner.py:125` | wraps env in `RslRlVecEnvWrapper(env, clip_actions=1.0)` |
| Eval harness | `src/phoenix/training/evaluate.py:161` | same `clip_actions=1.0` |
| Checkpoint load | `src/phoenix/training/checkpoint.py` | `load_runner_checkpoint`, used by every harness |
| Agent cfg | `src/phoenix/training/agent_cfg.py` | `build_runner_cfg`; `empirical_normalization` flag lives here |
| Slew metric | `src/phoenix/training/slew.py:19-36` | `slew_saturation_rate`, offline metric only |
| Env cfg builder | `src/phoenix/sim_env/go2_env_cfg.py:568-612` | layers YAML onto the upstream Isaac Lab task |
| In-MDP rate limiter | `src/phoenix/sim_env/rate_limited_action.py:29-101` | Isaac Lab `JointPositionAction` subclass |
| Pure clamp | `src/phoenix/sim_env/rate_limit.py:31-54` | imports `MAX_DELTA_PER_STEP_RAD` **from the deploy side** (`rate_limit.py:26`) |

**Upstream truth the deploy path is pinned to** (these are *not* in this repo, they are Isaac Lab):

- Observation term order and unit scaling: `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`, `class ObservationsCfg.PolicyCfg`, `base_lin_vel, base_ang_vel, projected_gravity, velocity_commands, joint_pos (joint_pos_rel), joint_vel (joint_vel_rel), actions (last_action), height_scan`. **No per-term scale factors.**
- `height_scan = None` for the flat task: `.../config/go2/flat_env_cfg.py:51`.
- Action scale 0.25, `use_default_offset=True`: `.../config/go2/rough_env_cfg.py:84` (upstream base is 0.5 in `velocity_env_cfg.py ActionsCfg`; the Go2 cfg overrides it).
- `enable_corruption` is **True** in training and only disabled in the `_PLAY` variants (`.../config/go2/rough_env_cfg.py:117`, `.../config/go2/flat_env_cfg.py:65`). The repo comment at `src/phoenix/sim_env/go2_env_cfg.py:464` states the Go2 task ships `enable_corruption = False`; that is true only of the `_PLAY` cfg.

**Do not disturb:** any of the above. Changing the observation term order, the action scale, the
`use_default_offset` reference pose, or `MAX_DELTA_PER_STEP_RAD` invalidates every exported checkpoint
and every shield artifact simultaneously.

---

## 2. Observation construction, the bit-comparable contract

Single builder: `src/phoenix/sim2real/observation.py`.

### Term ordering and layout (`observation.py:85-97`)

| # | Term | Dims | Source at deploy | dtype | Scale |
|---|---|---|---|---|---|
| 0 | `base_lin_vel` | 3 | **hardcoded zeros**, `ros2_policy_node.py:462` | float32 | 1.0 |
| 1 | `base_ang_vel` | 3 | `/imu/data` `angular_velocity.{x,y,z}`, `ros2_policy_node.py:454-461` | float32 | 1.0 |
| 2 | `projected_gravity` | 3 | `_projected_gravity_from_quat(ori.x,y,z,w)`, `ros2_policy_node.py:446`, impl `:709-721` | float32 | 1.0 |
| 3 | `velocity_command` | 3 | `/cmd_vel` `[linear.x, linear.y, angular.z]`, `ros2_policy_node.py:346-349` | float32 | 1.0 |
| 4 | `joint_pos - default_q` | 12 | `observation.py:90`; `default_q` from `cfg.control.default_joint_pos` in `joint_order` (`observation.py:63-65`) | float32 | 1.0 |
| 5 | `joint_vel` | 12 | `/joint_states.velocity` remapped by name (`ros2_policy_node.py:435-437`) | float32 | 1.0 |
| 6 | `last_action` | 12 | previous raw ONNX `action` output (`ros2_policy_node.py:491`); zeros on step 0 (`observation.py:66,83-84`) | float32 | 1.0 |

Total = `3+3+3+3+3*12 = 48` (`observation.py:70`). Assertion at `observation.py:95-96`.

Then, in the node: optional zero-padding of `obs_pad_zeros` (default **187**, `ros2_policy_node.py:165`) for
the rough-terrain height scanner, reshaped to `(1, 48+pad)`, `ros2_policy_node.py:482-487`. All deployed
`configs/sim2real/*.yaml` set `obs_pad_zeros: 0`. Shape sanity check against the ONNX input at
`ros2_policy_node.py:169-178`.

### Joint order

`cfg.joint_order` (identical in every deploy config, e.g. `configs/sim2real/deploy_stand_h25.yaml:59-71`):

```
0 FL_hip 1 FR_hip 2 RL_hip 3 RR_hip 4 FL_thigh 5 FR_thigh 6 RL_thigh 7 RR_thigh 8 FL_calf 9 FR_calf 10 RL_calf 11 RR_calf
```

Remapping is **by name**, not positional: `JointOrder.remap` builds an int64 index vector from the
incoming `/joint_states.name` list and raises `KeyError` on any missing joint (`observation.py:48-55`).
The lowstate bridge publishes in Unitree motor order `FR,FL,RR,RL × hip/thigh/calf`
(`lowstate_bridge_node.py:35-48,73-76`), the two orders differ, and only the name lookup reconciles them.

### Projected gravity

`ros2_policy_node.py:709-721`:
```
gx = -2*(x*z - w*y);  gy = -2*(y*z + w*x);  gz = -(1 - 2*(x*x + y*y))
```
Quaternion is ROS `(x,y,z,w)`. Byte-identical duplicate at `verify_deploy.py:115-127` (written in the
algebraically equivalent `2*(...)*-1.0` form). This equals `R(q)^T @ (0,0,-1)`, matching Isaac Lab's
`mdp.projected_gravity`. Sign convention was previously wrong and is now pinned by
`tests/test_projected_gravity.py` (5 tests).

### Callers of `ObservationBuilder`

1. `ros2_policy_node.py:473-481`, single-policy path.
2. `ros2_policy_node.py:558-575`, mode-switch path, builds **two** obs vectors (one per policy) that differ only in `last_action`.
3. `verify_deploy.py:98-106`, parquet replay for the parity gate (deliberately reuses the same builder so obs bugs are caught too).

---

## 3. Observation normalization

**It is a property of the checkpoint, not a config constant.** This is the single most load-bearing
recent change.

- rsl_rl 3.x stores `EmpiricalNormalization` buffers **inside** `actor_state_dict` as
  `obs_normalizer._mean` / `obs_normalizer._var` (`export.py:44-46`, `export.py:245-292`).
- Transform: `(x - mean) / (sqrt(var) + eps)` with `eps = 1e-2`, `export.py:42`, `_Normalizer.__init__` at `export.py:390-397`. **The epsilon is added to the std, not under the sqrt.**
- The exporter bakes the normalizer **into the ONNX graph** as `Sub` then `Div` nodes when present, and exports a raw-obs policy when absent (`export.py:362-364`, `_load_normalizer` at `export.py:399-414`).
- `checkpoint_has_obs_normalizer` (`export.py:295-330`) is the single source of truth both the exporter and every eval harness now call.
- Commit `e0a899c` fixed `scripts/reliability_rollout.py` (was hardcoding `empirical_normalization: True`); commit `9748073` fixed the same bug in `scripts/reliability_closed_loop.py:300`. Asking rsl_rl for normalization on a checkpoint without buffers produces an untrained normalizer whose forward is `(x - 0)/(1 + 1e-2)`, a silent 1% shrink of every observation. Both harnesses now record the resolved flag in run metadata.

**Verified by introspection of the shipped artifacts:**

| Artifact | Graph nodes | Normalizer present |
|---|---|---|
| `deploy/stand_v3_latent.onnx` | `Sub, Div, Gemm, Elu, Gemm, Elu, Gemm, Elu, Gemm, Concat` | **YES**, `normalizer.mean [48]`, `normalizer.std [48]`, both **inline** (not external data) |
| `deploy/flat_v4_latent.onnx` | `Gemm, Elu, Gemm, Elu, Gemm, Elu, Gemm, Concat` | **NO** |

 Two shipped artifacts in the same directory disagree about whether normalization exists. A C++
loader must never assume either.

**Clipping:** there is **no observation clipping** anywhere on the deploy path, not in
`observation.py`, not in the exported graph. (Upstream Isaac Lab clips only `height_scan` to
`(-1, 1)`, which the flat task removes.)

---

## 4. ONNX export

CLI: `python -m phoenix.sim2real.export` (`export.py:49-76`).

| Property | Value | Citation |
|---|---|---|
| Input name / shape / dtype | `obs`, `["batch", 48]`, float32 | `export.py:122-128`; verified in both `deploy/*.onnx` |
| Output names | `action` `["batch",12]`; with `--emit-latent` also `latent` `["batch",384]` | `export.py:109-114`; verified |
| Opset | 17 (default `--opset 17`) | `export.py:74`; both artifacts report `opset_import ('' , 17)`, `ir_version 10` |
| Producer | `pytorch 2.10.0+cu128` | ONNX metadata of both artifacts |
| Dynamic axes | batch dim only, on all of `obs`/`action`/`latent` | `export.py:110,114` |
| Architecture | `Linear(48,512) ELU Linear(512,256) ELU Linear(256,128) ELU Linear(128,12)` | `_build_actor_mlp` `export.py:214-224`; confirmed by initializer shapes |
| Latent taps | inputs of Linear #2 and #3 → `Concat(256 + 128) = 384` | `default_tap_indices` `export.py:167-187`; graph shows `Concat['val_8','val_12']`; artifact provenance records `tap_indices: [2,3]` |
| Activation | `Elu(alpha=1.0)` | graph attributes |
| Gemm attrs | `transB=1, alpha=1.0, beta=1.0` | graph attributes |
| TorchScript twin | `<output>.pt` written alongside via `torch.jit.trace` | `export.py:133-135` |

### External-data implication (`deploy/flat_v4_latent.onnx.data`)

Both shipped ONNX files are 2–3 KB with a ~746 KB sidecar `.onnx.data`. The following initializers have
`data_location == EXTERNAL`: `actor.{0,2,4}.bias`, `actor.{0,2,4,6}.weight`. Only `actor.6.bias` and (for
stand_v3) `normalizer.mean/std` are inline.

**Implications for a C++ loader:**

1. The `.onnx` file **alone is not a model.** Shipping it without the `.data` sidecar produces a model that loads its graph and then fails (or silently reads garbage) at weight resolution.
2. External-data paths are resolved **relative to the directory of the `.onnx` file**. `onnxruntime` C++ `Ort::Session` does this automatically only when the sidecar sits next to the model; if the model is loaded from a memory buffer (`CreateSessionFromArray`), external data will **not** resolve unless the base path is supplied explicitly. Prefer path-based session creation, or call `AddExternalInitializers` / set the model-external-data base directory.
3. Any deploy provisioning step must checksum **both** files. `docs/lab_card_stand_feet_on_ground.md:11-12,23-24` already mandates shipping both and sha256-matching them.
4. If the native runtime converts to TensorRT, the conversion must happen after external data resolution; TensorRT's ONNX parser reads external data relative to the model path passed to `parseFromFile`, not `parse`.

### Export-time parity check

`export.py:138-162`: 16 batches of 4 random `standard_normal` float32 inputs, torch vs `onnxruntime`
CPU EP, max abs diff over **every** output (action and latent) must be `<= 1e-4`, else return code 2.
`export.py:155-158` documents why the latent is gated as strictly as the action.

---

## 5. ONNX parity checks, tolerances actually in force

| Gate | What it compares | Tolerance today | Citation |
|---|---|---|---|
| Export-time | torch vs onnxruntime, random inputs | `1e-4` (hard fail) | `export.py:159-161` |
| `verify_deploy` CLI | ONNX vs TorchScript, obs replayed from a real parquet | `--tol` default `1e-4` | `verify_deploy.py:137`; `ParityReport.passed` at `verify_deploy.py:39-41` |
| `reliability_verify_deploy.py` | ONNX latent vs Isaac-recorded latent → OOD score → **arbiter blend trace** | latent/score diffs are *reported*; **trip-decision and blend mismatches have a budget of zero** | `scripts/reliability_verify_deploy.py:107-126,161-165` |
| Float64-fit vs float32-deploy monitor | Mahalanobis score path | `decision_disagreement` is the gate; `max_rel_err` diagnostic | `reliability/deploy.py:371-400` |

**Recorded results** (`deploy/*.verify.json`):

| Artifact pair | frames | max latent abs diff | max score rel diff | trip mismatch | blend mismatch |
|---|---|---|---|---|---|
| `stand_v3_latent.onnx` / `shield_stand_v3.npz` | 134,400 | 1.72e-05 | 1.61e-05 | 0 | 0 |
| `flat_v4_latent.onnx` / `shield_flat_v4.npz` | 6,400 | 2.29e-04 | 1.51e-06 | 0 | 0 |

Known limitation, self-documented: `EVIDENCE.md` records that the ONNX↔TorchScript gate compares two
exports of the **same** `_ExportablePolicy` wrapper, so it verifies runtime numeric parity but **cannot
catch a wrong wrapper**, which is exactly how the dropped-normalizer bug survived for months.
`reliability_verify_deploy.py` (obs replayed from the Isaac study through the shipped ONNX) is the only
gate that closes that hole, and it exists only on the latent-emitting path.

---

## 6. `ros2_policy_node.py` decomposition (733 LOC)

### ROS plumbing vs deterministic control logic

| Lines | Block | Class |
|---|---|---|
| 60-76 | argparse | plumbing |
| 78-103 | `main`, rclpy init/spin/shutdown | plumbing |
| 110-178 | config parse, `ObservationBuilder` construction, ORT session, obs-dim sanity | mixed |
| 180-221 | mode-switch session loading + `ModeSwitchCfg` | mixed |
| 222-291 | shield construction, config whitelist, fail-closed checks | mixed |
| 297-334 | state fields, `Node`, QoS, subs/pubs, 50 Hz timer | plumbing |
| 336-356 | 4 subscription callbacks (timestamping + latch on estop) | plumbing |
| 358-370 | `_latch_abort` (+ parquet footer flush) | **deterministic** |
| 372-386 | `_ready_to_command_motion` binding | **deterministic** |
| 388-512 | `_control_step`, the whole control law | **deterministic** |
| 514-534 | `_log_step` parquet append | telemetry |
| 536-629 | `_compute_mode_switch_target` | **deterministic** |
| 631-679 | `_apply_shield` + telemetry publish | **deterministic** (+ telemetry) |
| 681-689 | `_clip_to_limits`, `_publish_default_pose` | **deterministic** |
| 691-706 | `shutdown` | plumbing |
| 709-729 | `_projected_gravity_from_quat`, `_rpy_from_quat_xyzw` | **deterministic** |

Roughly 300 of 733 lines are deterministic control/safety logic; the rest is ROS bring-up, config
parsing, and logging.

### QoS

One profile for everything the node owns: `QoSProfile(depth=1, BEST_EFFORT)` (`ros2_policy_node.py:318`).
The bridge deliberately mirrors BEST_EFFORT with `depth=10` (`lowcmd_bridge_node.py:118-129`), a
RELIABLE subscriber against a BEST_EFFORT publisher gets nothing. The deadman node uses **RELIABLE**
(`deadman_joy_node.py:61-65`) while the wireless estop node uses **BEST_EFFORT** for the same topic
(`wireless_estop_node.py:67-72`). See open questions.

### Python stays / Native moves

| Component | Verdict | Rationale |
|---|---|---|
| Observation assembly (`observation.py`) | **Native** | Runs at 50 Hz on the critical path; must be bit-comparable with the exported graph's input; trivially portable (7 concats, no branching). |
| `_projected_gravity_from_quat` | **Native** | Same tick, 6 flops, sign convention is a proven historical bug source; must live with the obs builder. |
| ONNX inference | **Native** | Real-time budget; onnxruntime C++ removes GIL and Python allocator jitter; enables TensorRT EP later. |
| `_clip_to_limits` / `per_step_clip_array` | **Native** | This is the last-authority actuation limit. Must be in the deterministic layer, not behind a GC. |
| `is_ready_to_command_motion`, `startup_state`, `estop_is_active`, `sensor_is_stale` (`safety.py`) | **Native** | These are the safety predicates. Final authority must be deterministic, allocation-free, and independent of the Python runtime being alive. Pure functions with an injected clock, a mechanical port. |
| `_latch_abort` state machine | **Native** | Same argument; it is the abort latch. |
| Attitude abort + NaN abort (`ros2_policy_node.py:440-452`) | **Native** | Per-tick predicates on the actuation path. |
| `DeployMonitor.score_one` (`reliability/deploy.py:205-214`) | **Native** | 384×384 float32 matvec every tick; already written allocation-free specifically for this reason (`deploy.py:16-22`); benchmarked p999 = 0.0122 ms (`deploy/shield_stand_v3.npz.bench.json`). |
| `SimplexArbiter.update` (`arbiter.py:106-156`) | **Native** | Integer counters and a ramp; it gates actuation; must not diverge from the calibrated trace. |
| `DeployShield` arming window (`deploy.py:262-274`) | **Native** | Tick-counted; part of the calibrated operating point. |
| Blend arithmetic `(1-b)*learned + b*default_q` (`ros2_policy_node.py:660`) | **Native** | On the actuation path. |
| `mode_switch.py` state machine | **Native (only if used)** | Deterministic, per-tick, gates the published target. But `mode_switch.enabled: false` in every shipped config and it is mutually exclusive with the shield (`ros2_policy_node.py:284-291`). Port last, or not at all. |
| `motor_crc.py` CRC + `PHOENIX_FOR_MOTOR` reorder | **Native** | Already a port *from* C++ (`motor_crc.py:3-12`); the ctypes struct is a 812-byte C layout (`motor_crc.py:124`). Going native removes the ctypes round-trip entirely. |
| `lowcmd_bridge_node` tick logic | **Native** | It is the actuation gate: estop latch, stale-command watchdog, hold-pose fallback, slew clip, CRC. This is the piece with the strongest determinism argument in the repo. |
| `lowstate_bridge_node` | **Native** (thin) | 40 lines of field copying at 50+ Hz; native removes one serialization hop and one process. Low value but low cost. |
| Artifact loading (`deploy.load_artifact`) | **Native** | Startup-time only, but it holds the fail-closed dimension/version checks that must run before actuation. Alternative: keep as an offline validator and have C++ read a flattened binary. |
| `deadman_joy_node` / `wireless_estop_node` | **Native** | They *are* the deadman. A Python process death that stops the heartbeat currently fails safe (downstream treats stale as asserted), so this is not urgent, but the deadman should not depend on the Python GC either. |
| **Config parsing (YAML), argparse, logging setup** | **Python** | Startup-only. No real-time argument. YAML in C++ costs a dependency and buys nothing. Emit a validated flat binary/JSON from Python instead. |
| **`TrajectoryLogger` parquet writing** | **Python** | Off critical path in principle, but see parity risk R14: it is currently called *synchronously inside the 50 Hz tick*. Move it off-thread, don't rewrite it in C++. |
| **Shield telemetry publish** (`_publish_shield_telemetry`) | **Python** | Pure observability. Publishing it natively is fine but there is no determinism argument. |
| **`verify_deploy.py`, `export.py`, `bench_export.py`** | **Python** | Offline tooling. Porting these is pure resume optics. |
| **`reliability/ood_monitor.py` (fit-time scorers), `metrics.py`, `features.py`, `study.py`, `replication.py`, `oracle_screen.py`** | **Python** | Offline fitting/analysis, float64, Ledoit-Wolf, SVD. Explicitly designed to be fit offline and deployed as constants (`deploy.py:9-14`). Porting is actively harmful, the C++ side should consume the artifact, never fit it. |
| **`ShieldRuntime` + `TemporalFilter`** (`reliability/runtime.py`, `ood_monitor.py:238-283`) | **Python** | Note: **the deploy path does not use these.** `DeployShield.step` bypasses the temporal filter entirely (`deploy.py:262-274`, `filtered_score = raw`). They exist for the offline study. Do not port. |
| **`failure_detector.py` (`FailureDetector`)** | **Python** | Only `FailureThresholds` (the dataclass of constants) is used on-robot (`ros2_policy_node.py:295,449`); the stateful detector is offline replay tooling. Port the two constants, not the class. |
| **`replay/`, `adaptation/`, `demo/`, `real_world/synthesize_failure.py`** | **Python** | Not on the deploy path at all. |

---

## 7. Low-level bridge

### `lowcmd_bridge_node.py` (376 LOC)

| Behaviour | Value | Citation |
|---|---|---|
| Dry-run default | publishes `/lowcmd_dry` unless `--live` | `lowcmd_bridge_node.py:139-141,277-279` |
| Requires LowState first | silent until one valid `/lowstate` seen | `:184-187` |
| Command validation | rejects `len != 12` and any non-finite | `:154-163` |
| Estop | `estop_is_active` fail-closed; latch is **sticky**, never lifts | `:191-206` |
| Stale-command watchdog | `watchdog_s` default 0.2 s | `:228-232`, `:294-299` |
| Hold behaviour | target = last measured q, gains `hold_kp=20, hold_kd=1.0` | `:209-211`, `:282-293` |
| Active gains | `kp=25, kd=0.5` (CLI only, no config key) | `:280-281` |
| Reorder | `phoenix[PHOENIX_FOR_MOTOR[k]]` | `:215-218` |
| Slew clip | `per_step_clip_array(target, last_measured, 0.175)` | `:221-223` |
| Publish | mode `0x01`, `dq=0`, `tau=0`, head `0xFE 0xEF`, `level_flag 0xFF`, CRC | `:236-264` |
| estop timeout resolution | CLI → `safety.estop_timeout_s` in deploy.yaml → 0.5 s | `:329-339` |

### `lowstate_bridge_node.py` (111 LOC)

- Publishes `/joint_states` in Unitree motor order `FR,FL,RR,RL × hip/thigh/calf` with names populated (`:35-48,73-76`).
- **Quaternion convention:** Unitree `imu_state.quaternion` is `[w,x,y,z]`, fanned into ROS `Imu.orientation (x,y,z,w)` (`:82-86`). Documented as confirmed against `read_low_state.cpp` (`:16-18`).
- Gyro/accel passed through body-frame, unrotated (`:19-21,87-92`).

### `motor_crc.py` (181 LOC)

- `LowCmdRaw` ctypes struct, asserted `sizeof == 812` (`:124`).
- CRC: poly `0x04C11DB7`, init `0xFFFFFFFF`, **no reflection, no final XOR**, over `(sizeof>>2)-1` little-endian uint32 words (`:101-127`).
- `PHOENIX_FOR_MOTOR = (1,5,9, 0,4,8, 3,7,11, 2,6,10)` (`:42-55`), motor index k pulls phoenix index `PHOENIX_FOR_MOTOR[k]`.
- `motorCmd` array is **20** entries; slots 12..19 are left zero (`:90,141-143`).

---

## 8. Safety predicates, complete enumeration

All in `src/phoenix/sim2real/safety.py` unless noted.

| Predicate | Semantics | Threshold | Where configured |
|---|---|---|---|
| `estop_is_active` (`:13-37`) | True if never seen, or age > timeout, or value True | `estop_timeout_s` | `configs/sim2real/*.yaml safety.estop_timeout_s`, **0.5** in `deploy_stand_h25/v2/v3/v3_shielded`, **0.8** in `deploy.yaml:75` and `deploy_v3b.yaml:52`. Code default 0.5 (`ros2_policy_node.py:135`); bridge default 0.5 (`lowcmd_bridge_node.py:93,339`) |
| `sensor_is_stale` (`:40-55`) | True if never seen or age > timeout | `sensor_timeout_s` = **0.2** in all configs | `safety.sensor_timeout_s`; code default 0.2 (`ros2_policy_node.py:136`) |
| `deadman_should_estop` (`:58-75`) | True if no input, input stale, or button released | `joy_timeout_s` = **0.5** | CLI `--joy-timeout-s` (`deadman_joy_node.py:111-115`, `wireless_estop_node.py:116-120`), **not** in YAML |
| `per_step_clip` / `per_step_clip_array` (`:78-109`) | element-wise clip to `current ± max_delta` | `MAX_DELTA_PER_STEP_RAD = 0.175` | **hardcoded constant** `safety.py:95`. Imported by `ros2_policy_node.py:51`, `lowcmd_bridge_node.py:67`, `sim_env/rate_limit.py:26` |
| `is_ready_to_command_motion` (`:112-164`) | composite gate; reasons `estop_publisher_missing`, `external_estop`, `estop_heartbeat_stale`, `sensor_missing`, `sensor_stale` | as above | as above |
| `startup_state` (`:167-211`) | `waiting` / `ready` / `abort(first_message_timeout_<csv>)` | `first_message_timeout_s` = **15.0** in all configs | `safety.first_message_timeout_s`; falls back to deprecated `startup_grace_s` with a `DeprecationWarning` (`ros2_policy_node.py:141-156`) |
| Max runtime | latches `max_runtime` | **120 s** in all configs | `safety.max_runtime_s` (`ros2_policy_node.py:128,392-393`) |
| Attitude abort | `abs(pitch) > 0.8` **or** `abs(roll) > 0.6` rad | `FailureThresholds.pitch_rad=0.8`, `roll_rad=0.6` | **hardcoded dataclass defaults** `real_world/failure_detector.py:32-33`; instantiated with no overrides at `ros2_policy_node.py:295` |
| NaN abort | any non-finite in `q` or `qd` | n/a | `ros2_policy_node.py:440-443` |
| Bridge command validity | `len == 12` and all finite | n/a | `lowcmd_bridge_node.py:155-161` |
| Bridge LowState validity | all `motor_state[i].q` finite | n/a | `lowcmd_bridge_node.py:169-171` |
| Bridge stale-command | age > `watchdog_s` | **0.2 s** | CLI `--watchdog-s` only (`lowcmd_bridge_node.py:294-299`), **not** in YAML |
| Shield config whitelist | only `{enabled, artifact}` accepted | n/a | `ros2_policy_node.py:255-262` |
| Shield latent presence | ONNX must emit `latent` | n/a | `ros2_policy_node.py:237-246` |
| Shield dim match | artifact `latent_dim` == ONNX latent width | n/a | `reliability/deploy.py:328-332` |
| Shield artifact version | must equal `ARTIFACT_VERSION = 2` | n/a | `reliability/deploy.py:47,316-318` |
| Monitor constant finiteness | mean/whitener all finite | n/a | `reliability/deploy.py:189-190` |
| Arbiter hysteresis invariant | `clear_threshold < trip_threshold` | n/a | `arbiter.py:63-64`, `deploy.py:88-89` |
| Mode-switch hysteresis invariant | `enter_stand_thresh < enter_walk_thresh` | 0.05 / 0.15 | `mode_switch.py:40-41`; `configs/sim2real/deploy.yaml:22-23` |

**Thresholds with no config path at all** (must be surfaced in the native runtime's config schema):
`MAX_DELTA_PER_STEP_RAD`, `pitch_rad`, `roll_rad`, bridge `kp/kd/hold_kp/hold_kd`, bridge `watchdog_s`,
deadman `joy_timeout_s`, deadman button index / wireless button mask.

---

## 9. Watchdogs, slew limiting, freshness, estop

### Watchdog inventory

| Watchdog | Owner | Window | On expiry |
|---|---|---|---|
| estop heartbeat | policy node | 0.5–0.8 s | latch abort `estop_heartbeat_stale`, publish default pose once |
| estop heartbeat | lowcmd bridge | 0.5 s | sticky latch → hold at measured q with `hold_kp/hold_kd` |
| IMU freshness | policy node | 0.2 s | latch abort `sensor_stale` |
| joint_states freshness | policy node | 0.2 s | latch abort `sensor_stale` |
| policy-command staleness | lowcmd bridge | 0.2 s | hold at measured q |
| first-message | policy node | 15 s | latch abort `first_message_timeout_<missing>` |
| max runtime | policy node | 120 s | latch abort `max_runtime` |
| joy/wireless input | deadman nodes | 0.5 s | publish `estop=True` |

**Post-abort behaviour is deliberate and load-bearing:** after a latch the policy node publishes the
default pose **once** and then returns every tick without publishing (`ros2_policy_node.py:395-401`).
The comment records the failure mode: per-tick rebroadcast of `default_q` fought the real posture and
caused a Jetson brownout. The bridge's 0.2 s stale watchdog then holds at last-measured q. A native
runtime must preserve this exact "publish once, then go silent" semantics.

### Slew-rate limiting, training/deploy consistency

Three call sites, one constant:

1. Deploy, policy node: `per_step_clip_array(target, q, 0.175)` against **measured q**, `ros2_policy_node.py:681-684`.
2. Deploy, bridge: same helper against **last measured q**, after the Unitree reorder, `lowcmd_bridge_node.py:221-223`.
3. Training, in-MDP: `rate_limit_targets(processed, current_q, max_delta)` in `process_actions`, held across all decimation substeps, `rate_limited_action.py:57-80`, clamp at `rate_limit.py:31-54`, constant imported from the deploy module at `rate_limit.py:26`.

Consistency is structural (one constant, one clamp form), **not** proven by a cross-check test, see
parity risk R11. Training also supports `clip_mode: "prev_command"` (`rate_limited_action.py:59-70`),
which **does not** match the deploy bridge; `measured_q` is the default and the documented winner
(`docs/lab_card_stand_feet_on_ground.md:62-70`).

`slew_saturation_rate` (`training/slew.py:19-36`) is the offline metric on the same threshold.

### Estop chain

`deadman_joy_node` (RELIABLE QoS, `/joy` buttons[4] default) **or** `wireless_estop_node`
(BEST_EFFORT, `keys & 0x02` = L1) → `/phoenix/estop` (std_msgs/Bool @ 10 Hz) → consumed independently
by both the policy node (`ros2_policy_node.py:324-326,351-356`) and the bridge
(`lowcmd_bridge_node.py:137,174-180`). Double-checking is explicit and intentional
(`lowcmd_bridge_node.py:24-26`).

`scripts/estop_publisher.sh` exists but is a bare 10 Hz `False` heartbeat and is **not** a deadman
(`deadman_joy_node.py:9-14`; the lab card at `docs/lab_card_stand_feet_on_ground.md:37` mandates the
real deadman).

---

## 10. Reliability / PHM runtime, how a health verdict reaches actuation

### Chain (single-policy path only)

```
ONNX session.run(["action","latent"])            ros2_policy_node.py:489
  → latent (384,) float32
  → DeployShield.step(latent)                     ros2_policy_node.py:647
      → DeployMonitor.score_one  ||W(x-mu)||^2    reliability/deploy.py:205-214
      → arming gate (first 15 ticks: no arbiter)  reliability/deploy.py:262-269
      → SimplexArbiter.update(raw)                arbiter.py:106-156
      → ShieldDecision{blend, state, raw, filtered}
  → target = (1-blend)*learned + blend*default_q  ros2_policy_node.py:660
  → per_step_clip_array(target, q, 0.175)         ros2_policy_node.py:498,684
  → publish Float64MultiArray                     ros2_policy_node.py:500-502
```

The verdict is an **advisory blend weight**, never an abort. The shield cannot stop the robot; it can
only pull the target toward `default_q`. Everything it does is still subject to the slew clip and to
every hard predicate above it.

### CURRENT safety precedence ordering (as it actually is, `_control_step` 388-512)

| Rank | Gate | Action on trigger |
|---|---|---|
| 1 | `elapsed > max_runtime` | latch `max_runtime` |
| 2 | already `_estopped` | **return, publish nothing** |
| 3 | `startup == "waiting"` | publish `default_q`, return |
| 4 | `startup == "abort"` | latch + publish `default_q`, return |
| 5 | `is_ready_to_command_motion` false (estop missing/stale/asserted, sensor missing/stale) | latch + publish `default_q`, return |
| 6 | non-finite `q`/`qd` | latch `nan_in_joint_state` + `default_q`, return |
| 7 | `abs(pitch)>0.8` or `abs(roll)>0.6` | latch `attitude ...` + `default_q`, return |
| 8 | *(policy inference runs)* |, |
| 9 | **reliability shield blend** | blend learned target toward `default_q` |
| 10 | slew clip vs measured q | clamp |
| 11 | publish |, |

Out-of-band, higher than all of the above: `_on_estop` latches immediately on message receipt
(`ros2_policy_node.py:355-356`), i.e. between ticks.

Downstream of the node, the bridge applies its own precedence
(`lowcmd_bridge_node.py:184-226`): no LowState → silent; estop latched **or** command stale → hold at
measured q with soft gains; else reorder → slew clip → CRC → publish.

**So the shield sits at rank 9 of 11, below every hard predicate and above nothing except the clip.**
It is architecturally advisory. If the eventual design wants "deterministic native safety layer has
final authority", the current ordering already satisfies that on the *predicate* side; what changes is
where the code lives, not the precedence.

### Evidential status of the shield

- Stand-v3 artifact (`deploy/shield_stand_v3.npz` meta): trip 6385.4, clear 475.9, K=3, arming 15 ticks, handoff 10, recover 25, `latch: true`; nominal episode FPR 0.0417, 100% of falls warned, median lead 0.64 s, median full-fallback lead 0.44 s. Fit on `checkpoints/phoenix-stand-v3-h25-final/latest.pt` (sha256 recorded), tap indices [2,3], 60,000 fit frames, 192 calib episodes.
- Flat-v4 artifact: trip 7389.7, K=10, arming 15; `falls_warned`/`median_lead_s` are **NaN** and `evaluation` is **empty**, this artifact has no measured fall-warning evidence attached.
- Latest closed-loop walking study (`reliability_eval/closed_loop_walk/results.json`, corrected-normalization re-run @ `3f8af81`, n=32 disturbed blocks × 16 envs): unshielded−shielded fall rate 0.105 [0.064, 0.146] excludes zero; **dose-matched sham−shielded 0.0039 [−0.023, 0.033] does NOT exclude zero**; oracle−shielded ≈ 0. Read honestly: the shield reduces falls, but the corrected study does not separate that benefit from a dose-matched sham, and a perfect (oracle) detector does no better. Commit `3a3a66e` explicitly withdrew the monitor-timing claim. **All of this is simulation.**

---

## 11. Logging / telemetry

| Sink | Content | Citation |
|---|---|---|
| Parquet trajectory log | 13-column fixed schema, one row per 50 Hz tick | `real_world/trajectory_logger.py:42-73`; appended at `ros2_policy_node.py:514-534` |
|, fields written as zeros on real GO2 | `base_pos`, `base_lin_vel_body`, `contact_forces` (no odometry / foot sensors) | `ros2_policy_node.py:516-517,522-524,530` |
| Footer flush on abort | `_latch_abort` closes the writer so a SIGKILL leaves a readable parquet | `ros2_policy_node.py:364-370`; regression-tested in `tests/test_ros2_policy_node.py:53-117` |
| Shield telemetry | `Float64MultiArray [blend, raw_score, trip_threshold, state_code]` on `/phoenix/shield` | `ros2_policy_node.py:664-679`; topic in `configs/sim2real/deploy_stand_v3_shielded.yaml:75-76` |
|, non-finite score handling | transported as `-1.0` (arbiter already saw the true `inf`) | `ros2_policy_node.py:672` |
| Startup / abort logs | `logger.warning("ABORT: %s")`, shield-engagement warning (first engagement only) | `ros2_policy_node.py:361`, `:653-659` |
| Bench artifact | `deploy/shield_stand_v3.npz.bench.json`: latent_dim 384, 20k ticks, p50 0.0074 ms, p999 0.0122 ms, max 0.0269 ms vs 2.0 ms budget / 20 ms period, peak traced growth 1.47 KB. **x86_64, numpy 1.26.4, not the Orin.** | `deploy/shield_stand_v3.npz.bench.json` |

There is **no** loop-timing / jitter telemetry on the deploy path: nothing records tick period, inference
latency, or timer overrun. That is a gap for a real-time native port.

---

## 12. Tests covering the deploy path

| Test file | Tests | Covers |
|---|---|---|
| `tests/test_safety.py` | 33 | every predicate in `safety.py`, fail-closed semantics |
| `tests/test_observation.py` | 5 | builder layout, dim, name remap |
| `tests/test_projected_gravity.py` | 5 | gravity sign convention |
| `tests/test_mode_switch.py` | 18 | state machine, hysteresis, blend alpha |
| `tests/test_motor_crc.py` | 11 | CRC, struct size, phoenix↔unitree reorder round-trip |
| `tests/test_lowcmd_bridge.py` | 7 | `_build_config` resolution order, estop/watchdog logic |
| `tests/test_ros2_policy_node.py` | 4 | **only** `_latch_abort` parquet-footer flush (rclpy can't be constructed in CI, `:1-8`) |
| `tests/test_verify_deploy.py` | 7 | `verify_parity` with injected inference fns |
| `tests/test_export_normalizer.py` | 13 | normalizer-buffer discovery, `checkpoint_has_obs_normalizer` regressions |
| `tests/test_rate_limit.py` | 5 | pure clamp |
| `tests/test_slew_saturation.py` | 7 | offline metric |
| `tests/test_reliability_deploy.py` | 23 | artifact save/load, fail-closed checks, `DeployMonitor`, `DeployShield` arming, `parity_report` |
| `tests/test_reliability_arbiter.py` | 13 | hysteresis, dwell, ramps, re-trip, non-finite |
| `tests/test_reliability_runtime.py` | 7 | `ShieldRuntime` (offline path) |
| `tests/test_ood_monitor.py` | 12 | scorers, shrinkage, temporal filter (offline) |
| `tests/test_reliability_bundle.py` | 13 | artifact bundle manifest |
| `tests/test_failure_detector.py` |, | `FailureThresholds` / detector |
| `tests/test_trajectory_logger.py` |, | parquet schema round-trip |
| `tests/test_sim_integration.py` |, | `pytestmark = pytest.mark.sim`, excluded from CI (`:18`) |

**The single biggest test gap:** `_control_step`, the whole precedence ordering in §10, has **zero**
test coverage, because the node class cannot be constructed without rclpy
(`tests/test_ros2_policy_node.py:1-8`). Every gate is unit-tested *individually* in `test_safety.py`,
but their **composition and ordering** is untested. A native port is the opportunity to fix this: extract
the control step as a pure function of (sensor snapshot, clock) → (command, abort reason) and test the
whole ladder.

---

## 13. Configuration

`configs/sim2real/*.yaml`, six variants. Schema (all present in every file):

| Key | Consumed at |
|---|---|
| `policy.onnx_path` | `ros2_policy_node.py:88-91`, `verify_deploy.py:149` |
| `policy.torchscript_path` | `verify_deploy.py:150` only |
| `policy.device` | **never read**, see open questions |
| `policy.obs_pad_zeros` | `ros2_policy_node.py:165`, `verify_deploy.py:151` |
| `policy.mode_switch.*` | `ros2_policy_node.py:184-211` |
| `reliability.enabled` / `reliability.artifact` | `ros2_policy_node.py:227-266`; whitelist-enforced at `:255-262` |
| `control.rate_hz` | `ros2_policy_node.py:127,334`; `lowcmd_bridge_node.py:321` |
| `control.action_scale` | `ros2_policy_node.py:126,493` |
| `control.default_joint_pos` | `ros2_policy_node.py:121-125`, `verify_deploy.py:153-155` |
| `joint_order` | `ros2_policy_node.py:120` |
| `topics.*` | `ros2_policy_node.py:320-332`; `lowcmd_bridge_node.py:323` (joint_command only) |
| `safety.*` | `ros2_policy_node.py:128-156,325`; `lowcmd_bridge_node.py:324-327` |

Path from config to node: `--config <path>` → `yaml.safe_load` (`ros2_policy_node.py:84`) → plain dict,
no schema validation except the reliability whitelist. `--onnx` overrides `policy.onnx_path`. The bridge
reads only `control.rate_hz`, `topics.joint_command`, `safety.emergency_stop_topic`,
`safety.estop_timeout_s` and tolerates a missing file entirely (`lowcmd_bridge_node.py:319-320`).

**Config drift across variants:** `estop_timeout_s` is 0.5 in four files and 0.8 in two
(`deploy.yaml:75`, `deploy_v3b.yaml:52`). `default_joint_pos` hips are `0.0` in five files and
`±0.1` in `deploy_stand_h25.yaml:45-49`, the latter carries a comment (`:40-44`) stating that
`±0.1` is the **correct** training value and that the others are wrong. Only `deploy_stand_h25.yaml`
and `deploy_stand_v3_shielded.yaml` are current deliverables.

---

## 14. Real-hardware evidence, exactly as recorded, not upgraded

| Claim | Status as recorded | Source |
|---|---|---|
| Deploy chain (`ros2_policy_node`, `lowcmd_bridge_node`, estop, parity gates) ran live end-to-end on the GO2, 2026-04-18 | **HARDWARE-VALIDATED (that it ran)** | `EVIDENCE.md` §Verified; `README.md:55` |
| Outcome of that live run: **30.23%** per-step slew saturation at `cmd_vel = 0` | **HARDWARE-MEASURED** | `EVIDENCE.md` §Verified |
| Later live number quoted as ~33% | **HARDWARE-MEASURED** | `README.md:44,58`; `sim_env/rate_limit.py:11` |
| 2026-04-20 dry-run: 16.67% slew sat localized to rear thighs | **HARDWARE-MEASURED**, interpretation ("posture-mismatch, not policy fault") is **INFERRED** | `EVIDENCE.md` §Inferred |
| Gate 7, 10 s live stand × 3 on the real GO2 | **NOT VALIDATED / PENDING HARDWARE** | `EVIDENCE.md` §Not validated; `README.md:58` |
| Gate 8, flat walking on the real GO2 | **NOT ATTEMPTED** | `EVIDENCE.md` §Not validated |
| stand-v3-h25 32/32 success, slew 3.30% nominal / 2.91% DR | **SIMULATED** | `EVIDENCE.md` §Verified (explicitly "sim only" in `README.md:52`) |
| stand-h25-lat-noise 32/32, slew 3.65% / 4.23% | **SIMULATED** | `configs/sim2real/deploy_stand_h25.yaml:11-13` |
| ONNX↔torch parity 5.722e-06 (h25), 4.77e-06 (v3-h25-final), 3.8e-06 (v2) | **BENCHMARKED (offline, not on the Jetson)** | `configs/sim2real/deploy_stand_h25.yaml:14`; `EVIDENCE.md` |
| Mode-switch runtime | **UNIT-TESTED; hardware status explicitly "not validated"** | `EVIDENCE.md` §Verified + §Not validated |
| Failure-curriculum adaptation against real parquets | **NEVER RUN**, `failure_sample_fraction: 0.0` | `EVIDENCE.md` §Not validated; `README.md:169-177` |
| Reliability shield (any regime) | **SIMULATED ONLY.** No shield run has touched hardware. `deploy_stand_v3_shielded.yaml:5` describes shield runs as planned "August CaresLab" work | `configs/sim2real/deploy_stand_v3_shielded.yaml:1-6`; `reliability_eval/closed_loop_walk/results.json` |
| Shield real-time budget (p999 0.0122 ms vs 2 ms) | **BENCHMARKED on x86_64 (`Linux ... x86_64`, python 3.10.12)**, NOT on the Orin | `deploy/shield_stand_v3.npz.bench.json` |
| Feet-on-ground stand test | **PLANNED**, lab card written, explicitly says "THIS is the first real hardware validation of the Phoenix stand, every prior number was sim" | `docs/lab_card_stand_feet_on_ground.md:106-108` |

**Nothing in this repo is hardware-validated as *working*.** The only hardware validation is that the
stack runs and that it produced a failing slew number. Every success metric is simulation.

---

## 15. Evidence table, component by component

Tags used: `impl` (implemented), `unit` (unit-tested), `bench` (benchmarked), `sim` (validated in
simulation), `hw` (hardware-validated), `pend-hw` (pending hardware validation).

| Component | File | Tags | Justification |
|---|---|---|---|
| `ObservationBuilder` | `sim2real/observation.py` | impl, unit, pend-hw | 5 tests; fed the live 2026-04-18 run but never validated against a passing hardware outcome |
| Projected gravity | `ros2_policy_node.py:709-721` | impl, unit | `tests/test_projected_gravity.py`, 5 tests |
| ONNX exporter | `sim2real/export.py` | impl, unit, bench | `tests/test_export_normalizer.py` 13 tests; export-time parity 1e-4 gate |
| Normalizer reconstruction | `export.py:245-330,379-414` | impl, unit | 13 tests incl. 4 regression tests pinned by `e0a899c` |
| Exported artifacts | `deploy/*.onnx(+.data)` | impl, bench | opset 17, verified graph structure and initializer layout |
| `verify_deploy` parity gate | `sim2real/verify_deploy.py` | impl, unit | 7 tests; tol 1e-4 |
| `reliability_verify_deploy` E2E gate | `scripts/reliability_verify_deploy.py` | impl, sim | zero-mismatch results on 134,400 (stand) / 6,400 (flat) recorded frames |
| Safety predicates | `sim2real/safety.py` | impl, unit | 33 tests |
| `_control_step` composition | `ros2_policy_node.py:388-512` | impl, pend-hw | **no test coverage of the ordering** |
| Mode switch | `sim2real/mode_switch.py` | impl, unit | 18 tests; disabled in every config; hardware "not validated" per EVIDENCE.md |
| `lowcmd_bridge_node` | `sim2real/lowcmd_bridge_node.py` | impl, unit, hw(ran) | 7 tests on config/logic; ran live 2026-04-18 |
| `lowstate_bridge_node` | `sim2real/lowstate_bridge_node.py` | impl, hw(ran) | no dedicated test file |
| `motor_crc` | `sim2real/motor_crc.py` | impl, unit, hw(ran) | 11 tests; correctness ultimately proven by firmware acceptance |
| Deadman nodes | `deadman_joy_node.py`, `wireless_estop_node.py` | impl, unit(predicate only) | `deadman_should_estop` tested; node bodies untested |
| Slew limiter (deploy) | `safety.py:98-109` | impl, unit, hw(measured saturating) | measured 30.23% saturation on hardware |
| Slew limiter (training) | `sim_env/rate_limited_action.py` | impl, unit(clamp), sim | 5 tests on the clamp; in-MDP behaviour sim-only |
| `DeployMonitor` / `DeployShield` | `reliability/deploy.py` | impl, unit, bench, sim | 23 tests; bench on x86_64; sim study only |
| `SimplexArbiter` | `reliability/arbiter.py` | impl, unit, sim | 13 tests |
| Shield artifacts | `deploy/shield_*.npz` | impl, sim | stand_v3 carries measured evaluation; **flat_v4 carries none** (`falls_warned: NaN`, empty `evaluation`) |
| OOD scorers (offline) | `reliability/ood_monitor.py` | impl, unit | 12 tests; offline only |
| `ShieldRuntime` / `TemporalFilter` | `reliability/runtime.py` | impl, unit | 7 tests; **not on the deploy path** |
| Trajectory logger | `real_world/trajectory_logger.py` | impl, unit, hw(ran) | schema tests + abort-flush regression |
| Failure detector | `real_world/failure_detector.py` | impl, unit | only the thresholds reach the robot |
| Configs | `configs/sim2real/*.yaml` | impl | no schema validation; documented drift between variants |

---

## 16. Parity risk register

Places where a naive C++ reimplementation silently diverges from Python. Ordered by how quietly it breaks.

| # | Risk | Where Python does it | C++ requirement |
|---|---|---|---|
| **R1** | **Normalizer presence is per-checkpoint, not per-config.** `stand_v3_latent.onnx` has `Sub/Div`; `flat_v4_latent.onnx` does not. A C++ runtime that hand-implements normalization double-normalizes one and correctly normalizes the other. | `export.py:362-364` bakes it into the graph | Never implement normalization in C++. Feed raw obs to the graph. If you must know, read the graph for a `Sub/Div` prefix, do not read a config flag. |
| **R2** | **Normalizer epsilon is added to the std, not under the sqrt.** `(x-mean)/(sqrt(var)+1e-2)`. The common form `(x-mean)/sqrt(var+eps)` differs by orders of magnitude on low-variance dims. | `export.py:393` | Only matters if R1 is violated. It will be. |
| **R3** | **External data.** `.onnx` is 2 KB; the weights are in `.onnx.data`. Buffer-based session creation will not resolve them. | onnxruntime Python resolves relative to the model path | Use path-based `Ort::Session`; verify both files' sha256 at startup; fail closed if the sidecar is missing. |
| **R4** | **`base_lin_vel` is fed as zeros at deploy but is real in training.** The policy was trained with true body linear velocity in obs dims 0..2; the robot has no estimator, so the node feeds `np.zeros(3)`. | `ros2_policy_node.py:462` | Reproduce the zeros **exactly**. Do not "improve" this by wiring in an estimator, that changes the input distribution the policy was trained and calibrated on, and invalidates the shield artifact. Any change here is a research decision, not a port decision. |
| **R5** | **`default_joint_pos` must equal the training asset's default pose, and five of six configs are wrong.** Isaac Lab's `use_default_offset=True` makes it both the action offset and the `joint_pos_rel` reference. Training uses hips L=+0.1 / R=−0.1. | `deploy_stand_h25.yaml:40-49` (correct) vs `deploy.yaml:31-34` etc. (all-zero hips) | Refuse to run on a config whose `default_joint_pos` does not match a value carried in the model/artifact provenance. Do not silently accept a plausible-looking pose. |
| **R6** | **Joint order is resolved by NAME, and the two orders genuinely differ.** Policy order is `FL,FR,RL,RR × hip/thigh/calf`; the LowState bridge emits `FR,FL,RR,RL × hip/thigh/calf`. A positional copy is a silent leg swap. | `observation.py:48-55`; `lowstate_bridge_node.py:35-48` | Build the index vector from names once at startup, assert every joint present, and never index positionally. |
| **R7** | **Two distinct reorder maps exist.** `PHOENIX_FOR_MOTOR` (`motor_crc.py:42-55`) is the *command-side* permutation and is **not** the inverse of the observation-side name remap. Using one where the other belongs is a silent left/right swap that will look like a plausible gait. | `lowcmd_bridge_node.py:215-218` uses `PHOENIX_FOR_MOTOR`; `ros2_policy_node.py:435` uses name remap | Keep both, name them unambiguously, and unit-test the round-trip (`unitree_to_phoenix(phoenix_to_unitree(v)) == v`) in C++ too. |
| **R8** | **Quaternion conventions differ at three boundaries.** Unitree `imu_state.quaternion` = `[w,x,y,z]`; ROS `Imu.orientation` = `(x,y,z,w)`; Isaac Lab base quat = `(w,x,y,z)`; the parquet schema stores `base_quat` as **xyzw** (`trajectory_logger.py:46`). | `lowstate_bridge_node.py:82-86` | If the native runtime reads `/lowstate` directly (skipping the bridge), it inherits `[w,x,y,z]` and must not reuse the ROS-ordered gravity formula unchanged. |
| **R9** | **Gravity formula sign.** `gz = -(1 - 2(x²+y²))`, the negation applies to the whole term. The previously-shipped version had gx/gy flipped (`ros2_policy_node.py:714-716`). | `ros2_policy_node.py:709-721` | Port the expression literally and port `tests/test_projected_gravity.py` with it. |
| **R10** | **No action clipping at deploy, but training clips to [−1,1].** `RslRlVecEnvWrapper(env, clip_actions=1.0)` (`ppo_runner.py:125`, `evaluate.py:161`) means the action the env sees, and therefore the `last_action` term Isaac Lab feeds back, is clipped. The node scales the **raw, unbounded** Gemm output and feeds the **raw** value back as `last_action` (`ros2_policy_node.py:490-493`). The exported graph has no tanh/clip. This is a real train/deploy divergence in the current Python, and a C++ port that "helpfully" clips would diverge from the *Python* while converging on *training*. | `ros2_policy_node.py:490-493` | Decide explicitly, document the decision, and make both sides agree. Do not resolve it silently in the port. |
| **R11** | **Slew clip reference differs by node.** Policy node clips vs the **remapped policy-order q**; the bridge clips vs the **Unitree-order last-measured q**, after the reorder. Both use 0.175. Applying the clip once, in one order, is not equivalent to applying it twice in two orders when the two `q` snapshots are from different messages. | `ros2_policy_node.py:684`; `lowcmd_bridge_node.py:221-223` | Preserve the double clip, or prove equivalence. Do not collapse the two. |
| **R12** | **float32 vs float64.** Obs and the ONNX path are float32 (`observation.py:86-92`). The published command is cast to **float64** for the ROS message (`ros2_policy_node.py:501`) then back to float32 in the bridge (`lowcmd_bridge_node.py:158`) then to C `float` in the struct. `DeployMonitor` is float32 with `casting="unsafe"` (`deploy.py:210`); the offline scorer is float64 (`ood_monitor.py:51`). The arbiter compares in Python `float` (=double) (`arbiter.py:113-114`). | throughout | Match every width at every boundary. In particular the arbiter comparison `score > trip_threshold` must be done in double against a double threshold read from the artifact's JSON, not a float32 round-trip. |
| **R13** | **numpy broadcasting semantics.** `per_step_clip_array` is `np.clip(target, current-d, current+d)` with array bounds (`safety.py:98-109`); `_Normalizer.forward` broadcasts a `(48,)` buffer over `(N,48)`. A C++ loop is fine, but note `np.clip` with **NaN** in `target` returns NaN (it does not clamp), whereas `std::clamp` on NaN is UB. Non-finite `q` is caught earlier (`ros2_policy_node.py:440`) but non-finite *target* is not. | `safety.py:98-109` | Explicitly define NaN behaviour on the clip and fail closed. |
| **R14** | **Parquet logging is synchronous inside the 50 Hz tick.** `self._logger.append(...)` is called on the control path (`ros2_policy_node.py:504-511`), and `TrajectoryLogger` flushes row groups. Any C++ port that keeps this inline inherits an unbounded-latency I/O call in the control loop; any port that moves it off-thread changes the abort-time flush semantics that `tests/test_ros2_policy_node.py` pins. | `ros2_policy_node.py:504-511`, `364-370` | Move it off-thread **and** preserve the "flush on abort before the process can die" guarantee. |
| **R15** | **`DeployShield` bypasses the temporal filter.** `ShieldRuntime` (offline) applies EWMA smoothing (`runtime.py:86`); `DeployShield.step` sets `filtered_score = raw` (`deploy.py:271-274`). Porting `ShieldRuntime` instead of `DeployShield` would silently change the operating point the artifact was calibrated at. | `deploy.py:262-274` | Port `DeployShield` only. |
| **R16** | **Arming window is stateful and tick-counted, not time-counted.** 15 ticks after construction/reset (`deploy.py:255-260`). At a jittery 50 Hz that is not 0.30 s. The artifact records `control_dt_s: 0.02`. | `deploy.py:262-269` | Count ticks, not milliseconds. |
| **R17** | **Non-finite score must count as "above trip" and never toward "clear".** `above_trip = (not isfinite(score)) or score > trip` (`arbiter.py:113-114`); `DeployMonitor.score_one` returns `+inf` on any non-finite latent (`deploy.py:211-212`). A C++ `>` comparison against NaN is **false**, which inverts this. | `arbiter.py:113-114` | Explicit `!std::isfinite(score) \|\| score > trip`. This is the single most dangerous naive-port bug in the file. |
| **R18** | **CRC struct layout.** `sizeof(LowCmd) == 812` is asserted (`motor_crc.py:124`); the ctypes `_MotorCmd` is `uint8 mode` followed by five floats with **default (non-packed) alignment**, i.e. 3 pad bytes after `mode`, 36 bytes per motor. The CRC covers `(sizeof>>2)-1` words, little-endian. | `motor_crc.py:59-127` | Reuse the original C struct, don't re-derive it. Assert `sizeof == 812` at compile time. Note the array is **20** motors, slots 12..19 zeroed. |
| **R19** | **CRC is non-standard.** Poly `0x04C11DB7`, init `0xFFFFFFFF`, **no input/output reflection, no final XOR**, and the inner loop XORs the polynomial on each set data bit (`motor_crc.py:101-117`), this is not zlib CRC32 and not the standard MPEG-2 CRC either. | `motor_crc.py:101-117` | Port the loop literally; do not substitute a table-driven standard CRC32. |
| **R20** | **Post-abort silence, not rebroadcast.** After a latch the node publishes `default_q` once and then returns without publishing. Re-broadcasting every tick previously caused motor fight → Jetson brownout. | `ros2_policy_node.py:395-401` | Preserve exactly. This is a behaviour that looks like a bug and is not. |
| **R21** | **Bridge estop latch is sticky and unrecoverable.** Once `_estop_latched` is True it is never cleared (`lowcmd_bridge_node.py:177-206`); the node must be restarted. | `lowcmd_bridge_node.py:199-206` | Do not add auto-recovery. |
| **R22** | **`ShieldState` enum order is load-bearing on the wire.** `_publish_shield_telemetry` encodes state as `("nominal","handoff","fallback","recovering").index(...)` (`ros2_policy_node.py:668,677`), independent of the enum's declaration order in `arbiter.py:44-48` (which happens to match). | `ros2_policy_node.py:668` | Pin the wire encoding explicitly. |
| **R23** | **`obs_pad_zeros` code default is 187, config default is 0.** A native runtime that omits the key inherits a 235-dim obs and fails the shape check, loudly, thankfully (`ros2_policy_node.py:169-178`). | `ros2_policy_node.py:165` | Make the native default 0 and require the shape check. |
| **R24** | **Attitude thresholds come from a dataclass default, not the config.** `FailureThresholds()` with no overrides (`ros2_policy_node.py:295`) → pitch 0.8, roll 0.6 rad, and roll ≠ pitch. | `failure_detector.py:32-33` | Surface both in the native config, keep the asymmetry. |
| **R25** | **Timer semantics.** `create_timer(1.0/rate_hz, ...)` is a best-effort rclpy timer with no overrun detection and no deadline miss handling; there is no measurement of actual tick period anywhere. A native fixed-period loop will have a *different* (better) timing distribution than the one the sim study assumed (`control_dt_s: 0.02`). | `ros2_policy_node.py:334` | Instrument the native loop and record actual periods; do not assume the Python timing was 20 ms. |

---

## 17. Open questions, unresolved from the code

1. **`policy.device` is set in every config and read by nothing.** Both sessions are hardcoded to `CPUExecutionProvider` (`ros2_policy_node.py:167,200-205`; `verify_deploy.py:160`). Is CPU-only intentional on the Orin, or is this dead config? This determines whether the native port should plan for a CUDA/TensorRT EP.
2. **Which deploy config is canonical?** `deploy_stand_h25.yaml` claims to supersede `deploy_stand_v3.yaml` (`:25`), but `deploy_stand_v3_shielded.yaml`, the only shield-enabled config, is built on the superseded `stand_v3` pose (`:42-54`, all-zero hips) and points at `deploy/stand_v3_latent.onnx`. Is the shielded config knowingly running the wrong `default_joint_pos` (R5)?
3. **Is `deploy/flat_v4_latent.onnx` shippable?** Its shield artifact has `falls_warned: NaN`, `median_lead_s: NaN` and an empty `evaluation` block, and `checkpoints/phoenix-flat-v4/NEGATIVE_RESULT.md` records v4 as not shipped. Yet it is in `deploy/` next to the stand artifacts.
4. **Deadman QoS mismatch.** `deadman_joy_node` publishes `/phoenix/estop` **RELIABLE** (`:61-65`); the wireless node publishes it **BEST_EFFORT** (`:67-72`); both consumers subscribe BEST_EFFORT. A RELIABLE publisher and BEST_EFFORT subscriber are compatible, so this works, but was the asymmetry deliberate?
5. **Bridge gains have no config path.** `kp=25, kd=0.5, hold_kp=20, hold_kd=1.0, watchdog_s=0.2` are CLI-only (`lowcmd_bridge_node.py:280-299`). Should the native runtime read them from `deploy.yaml` (behaviour change) or preserve CLI-only (safer)?
6. **Action clipping (R10).** Was the missing `[−1,1]` clip at deploy a deliberate decision or an oversight? Nothing in the repo addresses it. This must be answered before the port, not during it.
7. **Was the Orin real-time budget ever measured?** `deploy/shield_stand_v3.npz.bench.json` is x86_64. Is there an Orin measurement anywhere? None found in the repo.
8. **EVIDENCE.md references artifacts that are not in `docs/`:** `docs/pre_lab_gates_2026-04-17.md`, `docs/pre_lab_stand_rollout_2026-04-17.json`, `docs/retrain_flat_scratch_2026-04-19.md`, `docs/superpowers/specs/2026-04-19-phoenix-gate8-mode-switch-design.md`. `docs/` contains only `architecture.{dot,svg}`, `changelog.md`, `deploy_mode_switch_runbook.md`, `lab_card_stand_feet_on_ground.md`, `structure.md`, `sweep_design_2026-05-17.md`. Were these removed, gitignored, or never committed? Several parity numbers quoted in EVIDENCE.md are unverifiable from the repo as it stands.
9. **Does the C++ runtime replace the ROS graph or live inside it?** The current design has four processes (deadman, lowstate bridge, lowcmd bridge, policy node) coupled by DDS at 50 Hz. Collapsing them into one native process removes three serialization hops and the DDS QoS failure modes, but also removes the independent double-check of the estop (`lowcmd_bridge_node.py:24-26`), which is currently a genuine safety property. This is an architecture decision, not a port decision.
10. **What is the intended relationship between the shield and the abort latch?** Today the shield can only blend toward `default_q` and can never latch (§10). If the native safety layer is to have "final authority", does the shield gain abort authority, or does it remain advisory?

---

## 18. Bottom line

- **Moves to C++ (real-time or determinism argument):** observation construction, projected gravity, ONNX inference, slew clip, all of `safety.py`, the abort latch, attitude/NaN aborts, `DeployMonitor` + `SimplexArbiter` + `DeployShield`, the blend arithmetic, the whole lowcmd bridge tick, `motor_crc`, the lowstate translation, and the deadman.
- **Stays in Python (no such argument):** YAML/argparse/logging, `export.py`, `verify_deploy.py`, `bench_export.py`, all shield *fitting* (`ood_monitor.py`, `metrics.py`, `features.py`, `study.py`, `replication.py`, `oracle_screen.py`), `ShieldRuntime` + `TemporalFilter` (unused at deploy), `FailureDetector` (only its constants ship), parquet logging (move off-thread, don't rewrite), shield telemetry, and everything under `replay/`, `adaptation/`, `demo/`.
- **The port's real risk is not performance, it is R1/R4/R5/R10/R17**, five places where a competent C++ engineer doing the obvious thing produces a robot that moves plausibly and is wrong.
- **The port's real opportunity** is that `_control_step`'s precedence ladder currently has zero test coverage. Extracting it as a pure function is worth more than the latency win.
