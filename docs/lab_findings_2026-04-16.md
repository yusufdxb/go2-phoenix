# Lab findings — 2026-04-16

Branch: `deploy-run-2026-04-14`
Operator: yusufdxb
Robot: Unitree GO2 + Jetson companion (lab)
Baseline policy: `checkpoints/phoenix-flat/policy.onnx` (Flat-v0, obs_dim=48)

---

## Section 0 — Preflight sync — PASS

- **T7 mount note:** Prompt path was `/media/T7 Storage`; actual mount is `/media/cares/T7 Storage`. Lab-PC udev mounts removable drives under `/media/cares/`. Future prompts should use the corrected path.
- **Jetson IP correction:** Prompt and prior memory said `unitree@192.168.0.4`; actual is `unitree@192.168.0.2`. Memory updated.
- **Sync method change:** Original `tar | ssh "rm -rf && tar x"` would have destroyed local untracked `checkpoints/phoenix-base/` on the Jetson. Replaced with `rsync -av` (no `--delete`).
- **Pre-sync Jetson HEAD:** `c68f5b8` (wireless_estop_node). Strict ancestor of T7 HEAD `7c93986`, so fast-forward via rsync was safe.
- **Post-sync Jetson HEAD:** `7c93986 Add verify_deploy parity gate + reset_bridge tests + lab-day prompt`.
- Branch correct (`deploy-run-2026-04-14`), `policy.onnx` present, `phoenix-base/` preserved.
- 58 MB transferred.

## Section 1 — Verify deploy artifacts — PENDING

## Section 2 — Dryrun saturation gate — PENDING

## Section 3 — Low-level mode toggle — PENDING

## Section 4 — Gamepad deadman test — PENDING

## Section 5 — First live policy run (stand) — PENDING

## Section 6 — Ground run 30s — PENDING

## Section 7 — Post-run artifacts — PENDING

---

## Mewtwo-side follow-ups (2026-04-16 PM)

### Root cause: flat-v0 export was under-trained

The policy at `checkpoints/phoenix-flat/policy.onnx` (shipped `c5e34a9`)
was trained for only 500 iters, ~3 min wall-time. Final-iter
`error_vel_xy` = 0.76 m/s — ~8x the sub-0.1 m/s target for a converged
Isaac-Velocity-Flat baseline. Isaac Lab references typically need
5-10M env steps, ours saw ~1M.

**Tell at deploy time:** canonical-stand ONNX inference emits
|action|∞ = 2.01, far above one-third of the slew-clip threshold. A
converged policy should be well under 0.3 with a zero-velocity
command and default pose. Confirmed in <1s via the new bench.

### Changes landed on `audit-fixes-2026-04-16`

1. **`configs/train/ppo_flat.yaml`**: `max_iterations` 500 → 2500 (5x).
   `save_interval` 50 → 100 to keep checkpoint count manageable.
   Algorithm hyperparameters match rsl_rl / Isaac-Velocity-Flat-v0
   defaults; audit found no other deltas worth changing.
2. **`src/phoenix/sim2real/bench_export.py`** (new, 39 unit tests →
   now 45): canonical-stand post-export gate. Wired into
   `scripts/deploy.sh` between `export` and `ros2_policy_node` launch —
   fails `deploy.sh` before any ROS bringup if |a|∞ ≥ 0.3.
3. **`ros2_policy_node.main()`**: `--onnx` now optional, falls back to
   `cfg["policy"]["onnx_path"]`. Matches `verify_deploy` / `dryrun_pipeline.sh`
   pattern.
4. **`ros2_policy_node.shutdown()`**: guards `_publish_default_pose()`
   with `rclpy.ok()` and catches publish exceptions. Fixes the
   `RCLError: Failed to publish` on every clean exit observed today.
5. **`scripts/dryrun_pipeline.sh`**: `ros2 topic hz --qos-profile sensor_data`
   → `--qos-reliability best_effort` (Humble doesn't accept the profile
   form in Humble's topic CLI).
6. **`configs/sim2real/deploy.yaml`**: `estop_timeout_s` 0.5 → 0.8. Still
   well under human reaction time, tolerates the BT hiccups that
   tripped fail-closed in today's session.

### Retrain kickoff

Retrain kicked off in background on mewtwo (RTX 5070):
`scripts/train.sh configs/train/ppo_flat.yaml --headless`. Monitor via
tensorboard at `checkpoints/phoenix-flat/<ts>/`. Success gate is
`error_vel_xy < 0.1 m/s` at final iter. Deploy gate is the new bench
(|a|∞ < 0.3 on canonical stand).
