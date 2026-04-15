# Phoenix Lab Prompt — 2026-04-16

**Target:** Unitree GO2 + Jetson companion (ROS 2 Humble), at the lab
**Branch on T7:** `deploy-run-2026-04-14` (do NOT merge to main until flat-v0 is hardware-validated)
**Baseline policy:** `checkpoints/phoenix-flat/policy.onnx` (Flat-v0, obs_dim=48, trained 2026-04-15 on mewtwo)

---

## SCOPE: LAB-ONLY

You are running in the lab, on or next to the GO2. **Everything in this prompt requires the physical robot or Jetson.** If a step doesn't need the hardware, it doesn't belong here — it was already done on mewtwo before the session.

**Do NOT do any of the following in this session (they are mewtwo-side and already complete as of 2026-04-15):**
- Training or retraining (PPO, adaptation, fine-tune)
- ONNX / TorchScript export
- Writing new tests or refactoring Python
- Editing `configs/env/*.yaml`, `configs/train/*.yaml`, or anything under `src/phoenix/training/`
- Pushing to GitHub or opening PRs
- Modifying `reset_bridge.py`, `curriculum.py`, or adaptation logic
- Running `pytest` outside of the two commands in Section 2 below

If you notice a defect in one of the above, **log it in `docs/lab_findings_2026-04-16.md` and move on.** Fixes happen back on mewtwo.

---

## LAB TASKS (in order)

### 0. Preflight sync

```bash
# T7 → Jetson over the lab LAN:
cd "/media/T7 Storage" && tar c go2-phoenix | \
  ssh unitree@192.168.0.2 "cd ~/yusuf && rm -rf go2-phoenix && tar x"
ssh unitree@192.168.0.2 "cd ~/yusuf/go2-phoenix && git status && git log -1 --oneline"
```

Expect `deploy-run-2026-04-14` branch, HEAD at `c5e34a9` or newer (flat-v0 deploy config). If not, stop.

---

### 1. Verify the deploy artifacts (offline, no robot motion)

Before the GO2 powers on, confirm the exported models are internally consistent. The `verify_deploy` gate was added on mewtwo 2026-04-15 specifically for this step.

```bash
cd ~/yusuf/go2-phoenix
python3 -m phoenix.sim2real.verify_deploy \
    --parquet data/failures/synth_slippery_trained.parquet \
    --deploy-cfg configs/sim2real/deploy.yaml \
    --tol 1e-4 \
    --max-steps 200
```

Expected: `PASS` with `max_diff` < 1e-4. Non-zero exit code = STOP; don't proceed to hardware.

Also spot-check `scripts/dry_run_policy.py` one more time (Jetson env may differ from mewtwo):

```bash
python3 scripts/dry_run_policy.py
```

Must report all four scenarios pass (clean start, attitude abort, estop latch, NaN abort).

---

### 2. Dryrun saturation gate

Goal: confirm the Flat-v0 policy does **not** saturate the slew clip rate-limiter on the real robot. The old Rough-v0 policy saturated on 99.5% of motor steps — that's the regression this gate catches.

Procedure (robot ON STAND, motors unpowered or low-level disabled):

```bash
# Terminal A — policy in dryrun mode (publishes joint cmds to a sink topic)
ros2 launch phoenix_bringup dryrun_pipeline.launch.py

# Terminal B — inspector
python3 scripts/lowcmd_inspector.py --topic /phoenix/lowcmd --window 500
```

Gate: **< 5% of steps saturate slew clip.** If higher, stop and log findings. Do not enable motors.

---

### 3. Low-level mode toggle

Confirm we can switch the GO2 into low-level control mode and back out cleanly. This must work before motors are ever enabled with the policy in the loop.

```bash
# On the GO2 (via Jetson):
ros2 run unitree_mode_ctrl mode_ctrl --ros-args -p target:=low
# Wait 5s, then:
ros2 run unitree_mode_ctrl mode_ctrl --ros-args -p target:=high
```

Verify:
- Both transitions succeed (exit code 0).
- `/lowstate` topic is publishing at ~500 Hz while in low mode.
- Robot stays in stand pose during both transitions (no joint drift > 0.05 rad).

Do **not** proceed if the transition is jerky or if `/lowstate` rate drops below 400 Hz.

---

### 4. Gamepad deadman test

Verify the wireless controller → `/phoenix/estop` path works before enabling motors. The `wireless_estop_node` was added 2026-04-14.

```bash
# Terminal A:
ros2 run phoenix_ros wireless_estop_node
# Terminal B:
ros2 topic echo /phoenix/estop
```

Test both directions:
1. Press deadman button → topic emits `False` (run-enabled).
2. Release deadman → topic emits `True` (estop latched).
3. Unplug controller → topic emits `True` within 500 ms of disconnect.

All three must work. If any fail, stop; motors stay off.

---

### 5. First live policy run (STAND ONLY)

Only proceed if Sections 0–4 all passed.

```bash
# Terminal A — policy node with built-in logging
python3 -m phoenix.sim2real.ros2_policy_node \
    --deploy-cfg configs/sim2real/deploy.yaml \
    --log-parquet data/hardware/run_2026-04-16_stand.parquet

# Terminal B — external estop (redundant with controller deadman)
bash scripts/estop_publisher.sh

# Terminal C — teleop
# Operator sends small /cmd_vel commands (<= 0.2 m/s) via joystick.
```

Gates:
- Policy runs for the full 120s without auto-aborting on attitude/NaN.
- No slew saturation warnings from the lowcmd inspector.
- Parquet is written on completion (`ls -la data/hardware/`).

**STOP HERE.** Do not lower the robot. Ask the human operator for explicit approval before Section 6.

---

### 6. (If human approves) Ground run, 30 seconds

Same launch as Section 5 but with the robot on the ground. Set `safety.max_runtime_s: 30` as a temporary override:

```bash
# Edit deploy.yaml max_runtime_s → 30 for this run only.
```

If the robot walks without falling or tripping an abort, capture the parquet; this is the **first real failure-curriculum seed.**

---

### 7. Post-run artifacts

Commit the run log and findings to the `deploy-run-2026-04-14` branch on the Jetson clone. Do NOT push to origin; the push happens back on mewtwo after review.

```bash
git add data/hardware/run_2026-04-16_*.parquet docs/lab_findings_2026-04-16.md
git commit -m "lab run 2026-04-16: flat-v0 on stand + (maybe) ground"
```

Then sync the Jetson clone back to T7 so the seeds reach mewtwo:

```bash
# From the Jetson:
rsync -av --delete ~/yusuf/go2-phoenix/data/hardware/ \
    /media/T7\ Storage/LABWORK/PHOENIX/hardware_runs/
```

---

## FAILURE MODES TO WATCH FOR

| Symptom | Likely cause | Action |
|---|---|---|
| `verify_deploy` reports max_diff ~1e-2 | ONNX↔torch export drift | Stop. Re-export on mewtwo with fresh `policy.pt`. |
| Slew clip saturates > 5% on dryrun | Wrong policy (rough instead of flat) or wrong `obs_pad_zeros` | Check `deploy.yaml`: `onnx_path` = phoenix-flat, `obs_pad_zeros: 0`. |
| Low mode transition hangs | SDK / firmware | Power-cycle the GO2; do not force-kill. |
| Deadman release doesn't latch estop | `wireless_estop_node` not running, or QoS mismatch | Confirm BEST_EFFORT QoS; confirm `/phoenix/estop` has exactly 1 publisher. |
| Attitude abort fires on stand (no motion) | IMU orientation wrong (GO2 placed on its side) | Reseat; check `/imu/data` orientation. |
| Node logs "NaN in joint_state" | Lowstate bridge dropped a frame | Restart `lowstate_bridge_node`. Do NOT disable the NaN check. |

---

## DO NOT, UNDER ANY CIRCUMSTANCES

- Raise `safety.max_runtime_s` above 120 on any hardware run.
- Disable the deadman or estop latching in code.
- Merge `deploy-run-2026-04-14` to `main` during the lab session.
- Start training on the Jetson.
- Push to GitHub from the Jetson.
- Edit `reset_bridge.py`, `curriculum.py`, or anything under `src/phoenix/adaptation/` or `src/phoenix/training/`.

---

## END-OF-SESSION CHECKLIST

- [ ] `docs/lab_findings_2026-04-16.md` populated with observations from each section.
- [ ] All hardware parquets committed to `deploy-run-2026-04-14`.
- [ ] T7 rsync of `data/hardware/` completed.
- [ ] Robot powered down, tether removed.
- [ ] Controller deadman confirmed as released (estop True) on shutdown.
