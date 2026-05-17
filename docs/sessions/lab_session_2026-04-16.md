# Phoenix Lab Session — 2026-04-16

**Target:** Unitree GO2 + Jetson Orin NX (ROS 2 Humble), CaresLab WiFi (Jetson @ 192.168.0.2)
**Branch on Jetson at start:** `deploy-run-2026-04-14` @ `7c93986`
**Branch on Jetson at end:** `deploy-run-2026-04-14` @ `69476ca` (audit-fixes applied)
**Session outcome:** Deploy chain hardware-verified end-to-end. Flat-v0 policy is broken at training. Gate 7 (live stand) not attempted.

---

## What changed on Jetson

- `/home/unitree/yusuf/go2-phoenix` advanced **7c93986 → 69476ca** via `git am /tmp/audit-fixes.patch` — a single-commit `format-patch` of `b074e40` "Audit pass: fail-closed safety + real replay + gravity sign fix", fetched from `origin/audit-fixes-2026-04-16` on the PC T7 clone.
- No config edits. No safety defaults touched. Untracked `checkpoints/phoenix-flat/` preserved.
- One throwaway parquet written: `data/failures/dryrun_2026-04-16_22-54-09.parquet` (62 KB, 512 rows).
- `gh` CLI installed on the PC via `apt-get install gh` (v2.4.0) for the fetch step. Authed then logged out at session end.

## Exact commands, in order

```bash
# === PC (mewtwo-side clone on T7) ===
sudo apt-get install -y gh
gh auth login -w                              # auth as yusufdxb
cd "/media/careslab/T7 Storage/go2-phoenix" && git fetch origin
git format-patch 7c93986..origin/audit-fixes-2026-04-16 \
    --stdout > /tmp/audit-fixes.patch
scp /tmp/audit-fixes.patch unitree@192.168.0.2:/tmp/audit-fixes.patch

# === Jetson ===
cd ~/yusuf/go2-phoenix && git am /tmp/audit-fixes.patch

# offline gates
python3 -m pytest tests/test_safety.py tests/test_projected_gravity.py \
  tests/test_apply_variations.py tests/test_lowcmd_bridge.py          # 60 pass in 0.67s

# ROS 2 must be sourced explicitly — `bash -c` non-interactive skips bashrc:
source /opt/ros/humble/setup.bash
source ~/unitree_ros2/cyclonedds_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/unitree/unitree_ros2/cyclonedds_ws/src/cyclonedds.xml
export PYTHONPATH="$PWD/src:$PYTHONPATH"

python3 scripts/dry_run_policy.py --config configs/sim2real/deploy.yaml   # 4/4 pass
ros2 topic hz /lowstate                                                   # 491 Hz
python3 -m phoenix.sim2real.wireless_estop_node                           # deadman test
ros2 topic echo /phoenix/estop --qos-reliability best_effort

bash scripts/dryrun_pipeline.sh 30                                        # saturation gate (FAIL)

python3 -m phoenix.sim2real.verify_deploy \
    --parquet data/failures/dryrun_2026-04-16_22-54-09.parquet \
    --deploy-cfg configs/sim2real/deploy.yaml \
    --tol 1e-4 --max-steps 200                                            # PARITY PASS

# === PC cleanup ===
gh auth logout --hostname github.com
```

---

## Hardware-verified (this session, on real GO2 + ROS graph)

| Item | Evidence |
|---|---|
| `/lowstate` alive | 491.4 Hz, 1 pub / 1 sub, no drops across 492 samples |
| Phoenix topics free of conflict | `/imu/data`, `/joint_states`, `/cmd_vel`, `/phoenix/estop`, `/joint_group_position_controller/command` all Unknown pre-bringup — `come_here` + `helix_bringup` running but don't race |
| Deadman press → run | L1 held on `/wirelesscontroller` → `/phoenix/estop = false` (stream confirmed) |
| Deadman release → stop | L1 released → `true` |
| **Mid-grip controller death** | Held L1 then powered off controller → observed `false → true` transition at echo line 6539, stream locks `true` thereafter — stale-heartbeat contract proven |
| `dryrun_pipeline.sh 30` four-stage bringup | All 4 stages up cleanly in 5 s; 30-s run; parquet written; shutdown clean |
| ONNX↔TorchScript parity on on-robot parquet | `verify_deploy` PASS: `max_diff=2.861e-06`, `tol=1e-4`, 200 steps |

## Hardware-verified behavioral findings (log, not fails)

- BT link between GO2 controller and robot has occasional >0.5 s gaps during continuous L1 hold. These trip `estop_timeout_s` (0.5 s default) and produce transient `true` blips in `/phoenix/estop` (observed e.g. lines 6227→6231, 6439→6463 — both ~20 ms wide = single publish period, then resumes `false`). This is **correct fail-closed behavior**; operationally it means live runs will micro-estop on BT hiccups. Consider either (a) accept — bridge hold-pose is smooth, (b) raise `estop_timeout_s` to ~0.8 s, (c) track on `mewtwo`.
- Robot's idle-stand joint positions drift from Phoenix's `default_joint_pos`: RR thigh reads **0.62 rad** when deploy.yaml expects **1.0 rad** (0.38 rad / ~22° offset). RL thigh similarly at 0.79 vs 1.0. Calves also lightly off. Either the Unitree idle stand differs from the sim default, or the joint-order permutation in the observation path has a subtle issue. Not a failure on its own, but worth a mewtwo-side check against the training env's `default_joint_pos`.

## Dry-run-verified (Jetson offline or single-node ROS)

- Patch `69476ca` applies cleanly on `7c93986`; untracked artifacts preserved.
- **60 / 60 pytest pass** on Jetson in 0.67 s: `test_safety` + `test_projected_gravity` (byte-match cross-helper — **locks out the old gx/gy sign flip**) + `test_apply_variations` + `test_lowcmd_bridge` config sentinels.
- `dry_run_policy.py`: all four scenarios pass — 47.5 Hz rate, attitude abort collapses commands to default, estop latch collapses commands to default, NaN joint state aborts cleanly. **Flag:** normal-scenario `max|cmd−default|=0.175` — already pegged at the slew clip. First hint of the gate-6 result.
- ONNX↔TorchScript parity of the shipped flat-v0 export: byte-clean.

## FAILED / unverified

- **Saturation gate FAILED: 99.63 %** of motor-steps at the ±0.175 rad slew clip over 10.2 s of live-robot-obs / dry-pipeline output (512 rows @ 50.04 Hz).  04-14 baseline was 99.5 %. **The flat-v0 retrain did not fix the saturation problem.**
- **Root cause pinned to training, not deploy.** A canonical "upright, stationary, at stand pose, no cmd_vel, zero last_action" observation fed directly to the ONNX session yields:
  - action norm **4.34**
  - |action|∞ = **2.01**
  - **7 / 12 motors over the 0.7 saturation threshold** (`|a| × 0.25 ≥ 0.175`)
  - Calves output `-1.6 / -1.56 / -1.68 / -1.82` — policy wants calves "more negative" when they're already at the -1.5 rad stand position.
- A healthy flat-v0 policy should emit near-zero actions on that observation. This one doesn't.
- **Gate 7 (live `--live` stand run) not attempted.** Saturated output would max-slew every motor on every step — incoherent, not a controller. The safety chain would contain it (bridge's hold-pose on estop, ±0.175 clip), but a live run proves nothing useful and adds hardware risk.
- **Deploy-config-driven ONNX path** for `ros2_policy_node` CLI: still `required=True` for `--onnx` in `parse_args` (`src/phoenix/sim2real/ros2_policy_node.py:51`). `verify_deploy` and `dryrun_pipeline.sh` resolve from `deploy.yaml`, but the main policy node does not. Minor drift; logged only.

## Artifacts on Jetson

- `/tmp/audit-fixes.patch` — 2486 lines, md5 `02a2ca572937a1d97a36102c21c0b182`
- `data/failures/dryrun_2026-04-16_22-54-09.parquet` — 62 KB, 512 rows, 50.04 Hz, 10.2 s span, full TrajectoryLogger schema
- `/tmp/dryrun_{estop,lsb,lcb,policy}.log` — stage logs (policy has an `RCLError: Failed to publish` on shutdown — cosmetic)
- `/tmp/dryrun_samples.log` — empty due to `ros2 topic hz --qos-profile sensor_data` syntax not supported in this ROS 2 Humble; a script bug
- Commit `69476ca` on `deploy-run-2026-04-14` (local, **not** pushed)

## Narrowest honest claim the repo can make after this session

> *"As of 2026-04-16, the Phoenix deploy chain (bridges, safety, estop heartbeat, gravity, replay, ONNX/TorchScript parity, dry-run pipeline) is hardware-verified end-to-end on the GO2 + Jetson. The shipped `checkpoints/phoenix-flat/policy.onnx` is byte-parity clean and loads correctly, but the trained weights produce saturated output on both canonical and hardware observations (99.63 %), making live actuation unsafe. The chain is ready; the policy is not."*

## Highest-leverage next fix, on `mewtwo`

**Retrain the flat-v0 policy properly.** 500 iters × 3 min wall-time was insufficient — the previous session (`c5e34a9`) shipped an under-trained policy. Three specific things to check:

1. **Training length.** 500 iters × 2048 envs ≈ 1 M steps. Isaac-Velocity-Flat baselines typically need 5 – 10 M steps to converge. Bump iters at least 5×.
2. **Sanity-check `configs/train/ppo_flat.yaml`** (new file from b074e40) — reward terms, observation normalization, action scaling vs the reference Flat-v0 baseline. The action-std-at-converge of 0.97 from the 04-15 training log is high but plausible; the real tell is whether final-iter `error_vel_xy` is sub-0.1 m/s on commanded motion. It shipped at 0.76 — too high.
3. **Add a 1-second bench test post-export** on mewtwo before shipping: feed the canonical-stand obs to the exported ONNX and require `|action|∞ < 0.3` (one-third of saturation threshold). If that fails, don't bother exporting. We caught it in <1 second today.

Alternatives from `docs/dryrun_findings_2026-04-14.md` still apply if a long retrain isn't viable: stand-only policy (very narrow, very safe first hardware test) or real height-scanner integration (bigger engineering).

## Small housekeeping (non-blocking, mewtwo-side)

- `scripts/dryrun_pipeline.sh`: `ros2 topic hz --qos-profile sensor_data TOPIC` is unsupported in this ROS 2 Humble. Use `--qos-reliability best_effort` or drop the hz probe.
- `ros2_policy_node.main()`: make `--onnx` optional and fall back to `cfg["policy"]["onnx_path"]` — matches the pattern `verify_deploy` and `dryrun_pipeline.sh` already use.
- `ros2_policy_node.shutdown()`: `_publish_default_pose()` after rclpy context is gone → `RCLError: Failed to publish` on every clean exit. Guard the publish with a context-valid check.
- `docs/lab_findings_2026-04-16.md` (exists on T7 at commit `5605cf1`, not yet on Jetson): when T7 is rebased onto `audit-fixes-2026-04-16`, append this session's findings; rsync back to mewtwo.
- Decide on `estop_timeout_s`: 0.5 s fails closed on BT hiccups (observed today). 0.8 s would tolerate typical hiccups while still being well inside human reaction time.

## Session status

- PC gh CLI: **logged out**.
- Jetson background processes: cleaned up (no stray `wireless_estop_node` or `topic echo`).
- No push to GitHub. No merge to `main`. Nothing on Jetson is destructive or committed to a shared branch.
