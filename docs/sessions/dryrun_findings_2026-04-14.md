# Phoenix Dry-Run Findings (2026-04-14)

Pipeline plumbing now works end-to-end on the Jetson with the GO2
powered on in **sport mode** (no live `/lowcmd` writes — output diverted
to `/lowcmd_dry`). The infrastructure can take a policy. **The current
baseline policy (`checkpoints/phoenix-base/policy.onnx`) cannot.**

## Pipeline status — green

| Stage | Verified |
|---|---|
| `scripts/estop_publisher.sh` heartbeat | publisher count on `/phoenix/estop` ≥ 1 |
| `lowstate_bridge_node` | `/joint_states` 500.96 Hz, `/imu/data` 500.25 Hz |
| `lowcmd_bridge_node` (dry, no policy) | `/lowcmd_dry` 49.998 Hz holding `motor_cmd[i].q == motor_state[i].q` with `kp=20 kd=1` |
| `lowcmd_bridge_node` (dry, policy active) | `/lowcmd_dry` 43.3 Hz with `kp seen [20, 25]` (transitions hold ↔ active correctly), mean cmd.q − state.q = 0.15 rad |
| `ros2_policy_node` | starts cleanly, ONNX session created, parquet written at 47.7 Hz, 1536 rows over 32 s, `action` finite |
| `data/failures/dryrun_<TS>.parquet` | written and readable, full schema present |

QoS gotcha caught and fixed: the bridge originally subscribed to
`/joint_group_position_controller/command` with `RELIABLE`. The Phoenix
policy publishes `BEST_EFFORT`. DDS treats this combination as
incompatible; the bridge stayed in hold-pose mode forever. Bridge now
matches the policy's `BEST_EFFORT` profile across all four
subscriptions and the LowCmd publisher (commit `5d7df2c`).

## Policy status — not safe to run live

The baseline policy was trained on `Isaac-Velocity-Rough-Unitree-Go2-v0`
which uses obs_dim=235 = 48-dim proprio + 187-dim height-scanner
readings. The deploy node zero-pads the missing 187 dims (per the
runbook's section 4 caveat). On hardware-adjacent obs the policy is
unstable:

| Metric | Value | Implication |
|---|---|---|
| Raw action `min`/`max` | −10.7 / +11.3 | Wide range |
| Action norm at row 0 | 2.82 | Reasonable initial output |
| Action norm at row middle | 20.05 | Drift |
| Action norm at row last | 19.16 | Settled into a high-amplitude regime |
| `joint_pos` range observed | −1.77 … 0.95 rad | The robot's joints DID move (sport mode controller maintaining stand against the policy's pretend disturbance) |
| `joint_vel` range observed | −7.7 … +7.1 rad/s | Already near per-step clip ceiling |
| Per-step delta saturating ±0.175 rad clip | **18,332 of 18,432 motor-steps** (99.5%) | Every step on every motor would slew at maximum rate |

Translation: the safety clips would prevent joint damage (the bridge
caps `cmd.q` at `state.q ± 0.175 rad` per step), but each leg would
slam at 8.75 rad/s in arbitrary directions on every step. That isn't a
gait — it's incoherent thrashing. Not safe to put weight on.

This is consistent with the failure mode the runbook predicts in
section 4 ("the stubbed height scan means the policy is blind to
terrain edges"), pushed to the extreme: with 187 dims of constant zero
the policy's distribution is so far off-manifold that it produces
saturated commands.

## What can fix this (not done in this session)

In rough preference order — pick one before live:

1. **Retrain on `Isaac-Velocity-Flat-Unitree-Go2-v0`** (obs_dim=48, no
   height scanner). A flat-surface policy with the same network
   architecture would consume the 48-dim proprio directly — no padding,
   no obs distribution shift. This is the cleanest path to an
   on-hardware first run.
2. **Implement a real height scanner** from `/utlidar/cloud_deskewed`
   (or whatever produces a 187-dim raycast around the robot in the
   training conv). Matches what the policy was trained on but is real
   engineering work.
3. **Train a stand-only policy** as the very first hardware test. No
   `/cmd_vel` consumption; outputs converge to a fixed stand pose.
   Eliminates the locomotion-instability question entirely while
   exercising the full pipeline.

## What survived to be reused

* `scripts/dryrun_pipeline.sh` — the launcher used here. Runs the whole
  stack for N seconds, captures topic samples mid-run, writes a
  parquet, tears down cleanly. Leave this as the standard offline
  validator before any future policy is taken to live.
* `scripts/lowcmd_inspect.py` — standalone subscriber that confirms
  whether the bridge is in hold mode (kp=20, q≈state.q) or active mode
  (kp=25, q drifting from state.q toward the 0.175 clip). Quickest way
  to diagnose QoS-style "bridge isn't getting commands" symptoms.

## Environment delta from earlier today

* `pyarrow` was missing on the Jetson (Phoenix's `pyproject.toml:dependencies`
  lists it, but `pip install -e .` was never run). Installed
  `pyarrow==14.0.2` from a copy at `T7:LABWORK/PHOENIX/wheels/`.
* `ros2 topic hz` and `ros2 topic echo` default to RELIABLE QoS, which
  doesn't match Phoenix's BEST_EFFORT topics. Pass
  `--qos-profile sensor_data` for hz against any Phoenix topic.

## Resume in next session

1. Read this doc + `T7:LABWORK/PHOENIX/RESUME_NEXT.md`.
2. Decide which fix path (1 / 2 / 3 above) to take. (1) is the smallest
   diff with the most upside.
3. Whatever new ONNX you produce, run `scripts/dryrun_pipeline.sh 30`
   and re-check `action` stability + slew saturation in the parquet
   BEFORE going `--live`. Less than ~5 % saturated steps is the bar.
