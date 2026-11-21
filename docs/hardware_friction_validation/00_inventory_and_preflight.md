# 00 — Inventory and preflight (2026-10-12)

Scope of this pass: SSH discovery, repository/provenance inventory, offline reproduction of
the 2026-10-07 FP friction estimates. **No robot commands were sent; nothing on the Jetson
was modified.** Robot-side work is BLOCKED (see below).

## VERIFIED

### Access
- Jetson reached from the MacBook as `unitree@192.168.0.70` from the LabLink device record
  `Jetson` (`~/.config/lablink/config.yaml`), strict host-key check against the LabLink pin
  `SHA256:0apC8UMreQU1yZ+dFZQZnv+GPQ1COuX52mar7Hh/9JU` (ED25519). The pin is
  trust-on-first-use, not independently verified. `lablinkd` CLI socket is absent on the Mac,
  so plain `ssh -o HostKeyAlias=lablink.lab.Jetson -o UserKnownHostsFile=~/.local/state/lablink/remote/lab.Jetson.known_hosts` was used.
- Jetson: hostname `unitree-jetson-payload`, L4T R36.4.7, Ubuntu 22.04.5, kernel
  5.15.148-tegra aarch64, ROS 2 Humble, CycloneDDS. Root disk 81 % used (44 GB free).
- Mewtwo (`ssh mewtwo`, Tailscale): RTX 5070 12 GB; Isaac Sim + IsaacLab under `~/Sim`;
  MuJoCo importable from `~/.local` (py3.12) and `~/Sim/isaac-sim-venv`, **not** from
  `~/workspace/go2-phoenix/.venv`.

### Robot state at inspection (~19:40 local)
- A non-Phoenix stack is live and owns motion: go2-nav `system.launch.py`
  (`hardware_bridge_node`, `safety_arbiter_node`, `lidar_hazard_node`, nav2, slam_toolbox),
  come-here `go2_bridge_node` (motion-authority topic), MOSAIC (`mosaic_node`,
  `mosaic_guardian`, `mosaic_posture`), whip voice/VLM. Load average ≈ 8.
- No Phoenix process running.

### Provenance of the 2026-10-07 friction result
| Item | Location | Note |
|---|---|---|
| Published summary | upstream `main` `0bfba9f` (EVIDENCE.md, README.md) | docs only |
| Result commit | `go2-phoenix-archive` `feat/phoenix-walk-v1-hw` @ `ed50fe7` | `docs/hardware/2026-10-07_friction_probe/` (README + 5 per-run JSONs) |
| Probe code as run | Jetson payload `04f3d46` (`PAYLOAD_SYNC.txt`: from mewtwo, clean, synced 2026-10-07T01:15:40Z) | `friction_probe.py`, `friction_probe_node.py` byte-identical to `ed50fe7` |
| Raw traces | Jetson `~/yusuf/go2-phoenix/logs/hw_sessions/20261007_*/FP_*` | 6 sessions; copied, not moved |
| Operator run logs | Jetson `~/phoenix/FP_run*.log`, `fp_run.sh`, `fp_launch.sh` | |
| Policy bundle for FP | `~/phoenix/walk2750-stand-34104ff/deploy_walk_iter2750_stand.yaml` | via `env_09cce76_stand.sh` |

Mewtwo's local `feat/phoenix-walk-v1-hw` is ~20 commits ahead of the archive remote
(unpushed WIP: gantry A/B, free-floor gate, walk grants). This branch is based on the archive
head `ed50fe7`, not on those.

Offline snapshot (not in git): `~/Projects/phoenix-friction-data/jetson_snapshot_20261012/`
(65 MB, `SHA256SUMS`, 152 files). The 6 `bridge.jsonl` streams (~1.1 GB total) were left on
the Jetson.

### Reproduction of the published estimates
`PYTHONPATH=src python scripts/reproduce_fp_estimates.py --root <snapshot>/hw_sessions --out <data>/repro_20261012.json`
→ all 5 stored `fp_estimate.json` reproduced from `fp_trace.npz`, max |Δ| = 0 on
`fs, bias, kinetic, tau±, fs_tau_est, bias_tau_est`. `FP_20261007T005239Z` has 5 ticks and no
estimate (aborted run1). `tests/test_friction_probe.py`: 13 passed (macOS, numpy only).

| run (docs) | session | joints with two-sided Fs (N·m) |
|---|---|---|
| run2 | FP_20261007T005825Z | FR_hip 0.353 |
| run3 | FP_20261007T010358Z | FR_thigh 0.249 |
| run4 | FP_20261007T011122Z | FL_hip 0.338, FL_thigh 0.171, RR_hip 0.261, RR_thigh 0.339, RL_thigh 0.331 |
| run5 (invalid, floor) | FP_20261007T011645Z | FR_calf 1.875, FL_calf 2.255, RR_calf 0.960, RL_calf 0.612 |
| run6 | FP_20261007T011929Z | FR_calf 0.422, FL_calf 0.622, RR_calf 0.962, RL_calf 0.607, RL_hip 0.241 |

The published ranges equal the run2/3/4/6 values. The run mapping is by identical values and timestamp order.

### Method facts (from code at `ed50fe7`, header of each trace)
- Bridge PD kp 25, kd 0.5, 50 Hz; one joint ramped 0.05 rad/s each way; breakaway = 0.02 rad
  of motion from the pre-ramp median; max offset 0.3 rad.
- Torque = **commanded PD torque** `kp(target−q) − kd·dq`, not a measured torque. The
  robot-reported `tau_est` (lowstate) is recorded alongside; its Fs agrees within
  0.00–0.03 N·m.
- Gravity/bias removed by direction symmetry: `Fs = (τ+ − τ−)/2`, `bias = (τ+ + τ−)/2`.
- Both directions are tested; no temperature or motor current is in the trace
  (`npz` keys: `t, t_mono_ns, q, dq, target, kp, kd, tau_est, active, phase, joint_order, q0`).

## INFERRED
- **Sample size is the main weakness.** Each hip and thigh rests on one two-sided
  measurement. Calves have one valid run. There's no repeat within a joint
  across sessions, so no uncertainty interval can be computed from the 10-07 data.
- **The only repeatability data is incidental.** The rear calves in run5 (not in floor
  contact) vs run6, about 3 min apart, differ by 0.002 and 0.005 N·m.
- **Run5's front calves aren't evidence of load-dependent joint friction.** They show a
  large bias (+1.55/+1.79 N·m) and Fs 4× the unloaded value. Floor contact adds
  foot–ground friction and body constraint to the resisted torque, so joint friction can't
  be separated from it. It's confounded, not a controlled loaded measurement.
- `Fs` includes everything between commanded torque and joint motion: gearbox, motor
  cogging, and any torque-tracking error of the motor driver. It isn't purely joint friction.
- The symmetric-friction assumption is untested: τ+ and τ− friction can't be separated by
  this design without an independent gravity/load reference.

## HYPOTHESES (untested)
- 1.25–1.5× effective friction under load. Not measured; see `ed50fe7` README.
- `tau_est` is an independent (current-based) torque measurement. Unitree's derivation of
  `tau_est` hasn't been checked; its close agreement with PD torque could also mean it's
  model-derived.

## BLOCKED / OPEN
- **Robot-side preflight and every active test:** the Jetson is running another session's live
  stack that holds motion authority. Phoenix can't get exclusive authority without stopping
  it, which needs the user's decision and coordination with that session.
- ~~MuJoCo replay not located~~ Resolved: it's mewtwo-only `scripts/desk/sim_replay_walk.py`, reproduced
  exactly. See `03_phase_c_sim2real_validation.md`.
- **Walking-trial measurements:** yaw (17.6°/19.0°) reproduced from IMU logs. The 0.71 m forward
  distance has no logged source (no odometry); provenance unknown. See `03_…`.
- **Load-measurement hardware:** no calibrated force/torque sensor or loading fixture has
  been identified. Phase A loaded tests can't be designed until one exists.
- **Not recorded:** temperature and motor current in the FP traces. Check whether
  `bridge.jsonl` holds lowstate motor temperature.

## Live Jetson inspection, 2026-10-12 ≈19:57Z (passive: DDS discovery + 6 s subscription only)
- **Processes:** the MOSAIC / go2-nav / come-here stack seen at 19:40 had stopped by 19:57.
  `come-here.service` is inactive and disabled. Only the idle `ros2` daemon remains. I didn't stop it,
  and I don't control whether it restarts.
- **Robot mode:** sport `mode 0`, `error_code 1001`, lying (calves ≈ −2.8 rad, body height 0.07 m).
  `/lowcmd` has **1 publisher** (bare-DDS app, not on the Jetson, i.e. the GO2's internal controller)
  sending pure damping (kp 0; kd 3 on front-leg motors 0–5, kd 1 on rear-leg motors 6–11). Phoenix needs this released (`motion_switch.py`)
  to own `/lowcmd`.
- **Rates:** `/lowstate` 500 Hz, `/lowcmd` 500 Hz, `/sportmodestate` 300 Hz, `/utlidar/robot_odom` 150 Hz
  (same position as sportmodestate), `/utlidar/imu` 250 Hz, `/lf/lowstate` 20 Hz. Silent:
  `/lio_sam_ros2/mapping/odometry`, `/uslam/*/odom`, `/wirelesscontroller` (remote off or idle).
- **`motor_state` fields:** mode, q, dq, ddq (always 0), tau_est (values seen only in 0.025 N·m
  steps), q_raw, dq_raw, ddq_raw, temperature (integer °C; rear hips 53, others 30–35 at idle), lost, reserve.
  **No motor current.** `foot_force` raw counts (≈ 90–100 while lying), `foot_force_est` 0, so
  uncalibrated. Battery SOC 49 %, 28.5 V.
- **Cameras:** no `/dev/video*` on the Jetson. `/frontvideostream` (GO2 front camera, H.264) exists.
  No external camera identified.
- **Clock:** Jetson UTC, NTP inactive, not synchronized (≈ 0.4 s from the Mac including SSH latency).
- **Recorder:** `scripts/harness_record.sh` (rosbag2 of `/lowstate`, `/joint_states`, `/phoenix/estop`, …;
  camera only if `/dev/video*`). rosbag2 is present. It wasn't used on 10-06/07: the bridge log (50 Hz) has no
  tau_est, temperature or odometry.
- **Safety architecture (code on payload):** `harness_preflight.sh` ledger A–H, fail-closed, no
  chaining, typed operator phrase per live stage; W requires `LIVE W <preset> ON GANTRY`;
  deadman = GO2 remote (`phoenix_wireless_estop`) → `/phoenix/estop`; bridge holds on stale
  lowstate, estop, limit-abort band 0.175 rad, slew limit, torque margin. Presets: vx03_2s, vx03_4s,
  vx05_3s, yaw05_3s (CCW only), vxm03_2s.

## Follow-up 2026-10-12 ≈20:14Z
Rear-hip temperature transient (53 → 46 °C), battery drain (49 → 43 % idle), motion-authority restart paths,
instrumentation readiness and the FREE_FLOOR audit are in `07_readiness_and_free_floor_audit.md`. Video provenance
is in `06_…`, and joint-level residuals in `08_…`.
