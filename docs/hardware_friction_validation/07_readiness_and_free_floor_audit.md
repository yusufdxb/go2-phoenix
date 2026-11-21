# 07 — Jetson readiness, rear-hip temperature, motion authority, FREE_FLOOR audit (2026-10-12)

No motor activation. No Jetson deployment change. Passive DDS subscriptions plus mewtwo-side tests only.

## 1. Rear-hip temperature (passive telemetry)
| motor | 19:56Z | 20:14Z (30 s, 14 994 samples) |
|---|---|---|
| RR_hip | 53 °C | 46–47 °C |
| RL_hip | 53 °C | 46 °C |
| all other 10 motors | 30–35 °C | 30–33 °C (steady) |

Board NTC 44/41 °C. Robot lying in sport-mode damping (mode 0, error 1001) the whole time.

- **VERIFIED:** both rear hips are cooling (−7 °C in 18 min) while the others are flat. That's a thermal
  transient, not a constant offset. The robot was under other stacks earlier today (MOSAIC/go2-nav running at 19:40Z).
- **VERIFIED:** no historical motor-temperature data exists. Phoenix 10-06/07 logs (bridge, parquet,
  LIMP, DIMU) have no temperature field, and the MOSAIC bags record `/sportmodestate` only.
- **INFERRED:** most consistent with residual heat from recent rear-hip work. A localized sensor or
  thermal fault can't be ruled out without a cold baseline.
- **Required:** record all 12 motor temperatures (rosbag `/lowstate`) at session start after ≥ 1 h
  idle (cold baseline), and before and after every run. No temperature gate exists in Phoenix. Any threshold
  must come from Unitree documentation, not be invented here.

## 2. Battery
SOC 49 % (19:56Z) → 43 % (20:14Z) while idle, i.e. about 20 %/h with the payload on. **Correction to doc 05:** there's no
85 % gate. Existing gates are the harness floor (> 20 %, every motors-live stage) and the A/B prereg
(block start ≥ 30 %). 85–91 % was the SOC *measured* on 10-06/07. Matching it (charge to ≥ 85 %, log
SOC per run, interleave arms) is a condition-matching recommendation, not an existing gate.

## 3. Motion authority
- **VERIFIED:** at 19:57Z no ROS process runs on the Jetson except the idle ros2 daemon. `/lowcmd` has one
  publisher, a bare-DDS app off the Jetson (the GO2's internal controller, damping kp 0).
- **VERIFIED (no auto-restart found):** no enabled robot-related systemd unit or timer (`come-here.service` is
  disabled; `Restart=on-failure` only while running), no crontab, no autostart, no tmux. `.bashrc`
  only sources ROS environments.
- **Remaining paths (not excludable by software):** manual launch by another session (MOSAIC/go2-nav/whip,
  e.g. via the 5080 operator workspace `jetson` wrapper), the Unitree app/remote (WebRTC API topics
  present), and a motion-switcher request re-engaging sport mode.
- **Gate gap:** `preflight_eval.contention_checks` (both `04f3d46` and `06c5a37`) recognises only
  `come-here.service` and node names containing `come_here`. It wouldn't flag `mosaic_*`,
  `hardware_bridge_node`, `safety_arbiter_node`, `go2_bridge_node`, whip nodes, or a non-ROS `/lowcmd`
  publisher. **Required before any live stage:** a reviewed amendment extending the check (unit and node lists,
  plus `/lowcmd` publisher count and endpoint origin after release). Until then, run a manual check
  immediately before each live step: process list, `ros2 node list`, `ros2 topic info -v /lowcmd`.

## 4. Instrumentation readiness
| item | state |
|---|---|
| `/lowstate` 500 Hz (q, dq, tau_est, temperature, lost, raw foot counts) | available; `tau_est` quantized ≈ 0.024 N·m; **no motor current**; foot force uncalibrated |
| rosbag2 on Jetson | available; `scripts/harness_record.sh` exists (records `/lowstate`, `/joint_states`, `/phoenix/estop`, …); not used 10-06/07 |
| bridge `bridge.jsonl` / `policy.parquet` / `walk_cmd.jsonl` | verified 10-07 (q, dq, targets, gains, clips, estop, IMU, actions) |
| IMU yaw | verified (quaternion vs gyro ≤ 1°) |
| external camera | **none identified**; no `/dev/video*` on Jetson |
| floor markers / fiducials | **none identified** |
| clock sync | Jetson NTP inactive, boots at 1970. Align video by motion onset; set and record the clock each session |
| harness load cell | **none**; tension UNMEASURED |
| disk | Jetson / 81 % (44 GB free); rosbag `/lowstate` ≈ 0.5 MB/s (estimate, verify in dry run) |

## 5. Motors-off gates B–D (CONDITIONAL GO: prepared, NOT executed)
Code inspection of the payload `harness_preflight.sh` and modules:
- **B:** `dryrun_pipeline.sh` launches `lowcmd_bridge_node` **without `--live`**, so it publishes `/lowcmd_dry` only.
  `--live` additionally requires config, lock, expected SHA and telemetry. The script states "Nothing in here can publish /lowcmd".
  The estop heartbeat runs with `PHOENIX_DRYRUN_ONLY=1`.
- **C:** launches only the deadman node (publishes `/phoenix/estop` Bool) plus `hw_probe deadman`; "no lowcmd bridge is
  started". It needs the operator holding the GO2 remote.
- **D:** launches only `lowstate_bridge_node` (publishes `/joint_states`, `/imu/data`, `/phoenix/foot_force`) plus `hw_probe rates`.
- None of B–D sends a motion-switcher or sport request.
- **Not executed**, because:
  1. No operator is confirmed present (required for C; requested for all).
  2. **Code identity:** the payload is `04f3d46` (FP commits on top of `34104ff`; `src` diff = friction-probe
     files only, plus FP harness additions). The walk config's `PHOENIX_EXPECT_SHA` is `34104ff`, so `need_sha` halts.
     Running B–D needs one of: (a) restage `34104ff` (deployment change), (b) run at `04f3d46`
     (new ledger at that commit), or (c) stage `06c5a37` for FREE_FLOOR (deployment change). That's your decision.
  3. Stages write session evidence under `logs/hw_sessions/` on the Jetson. That's a write, but not a deployment change.

## 6. FREE_FLOOR implementation audit (`06c5a37`, mewtwo; REVIEW ONLY)
Reviewed in a scratch clone (`~/workspace/friction-validation-repro/ff_review`, branch name set locally only).
Mewtwo's own worktree, with its uncommitted `EVIDENCE.md`, is untouched.

**Tests:** `test_free_floor_gate`, `test_walk_grant`, `test_deploy_contract`, `test_harness_preflight`,
`test_free_floor_rehearsal`: 163 passed and 3 skipped in the worktree (2 skips = dirty tree, 1 = gitignored H25
artifacts). In the clean clone the 2 rehearsal tests **pass**. They failed first on a detached HEAD
because the gate refuses an unexpected branch name, which is correct behaviour. Full rehearsal (fake GO2, `ROS_LOCALHOST_ONLY=1`, domain 87): see §7.

**What it enforces (verified in code):** incumbent-only identity (model_2750 hashes; seed 50 explicitly denied);
`vx03_2s` only; a GO STOP/deadman drill (≤ 3 attempts, newest must be GO) before the walk; a write-once
run-start record before any motor process; one walk per session; a review latch blocking every later walk
(FREE_FLOOR and gantry W) until a named human review; 5 typed operator confirmations plus the phrase
`LIVE FREE_FLOOR <x> NO GANTRY`; interactive TTY required; tape measurement recorded; odometry marked
unreliable; walking grants (one authority at a time; hand-launched nodes refused); disclosed deadman
latency (up to ≈ 0.6 s of motion after release).

**Control path:** `git diff 34104ff 06c5a37` shows 0 changed lines in `observation.py`, `safety.py`, `go2_model.py`,
`export.py`. `lowcmd_bridge_node.py` (+81) and `ros2_policy_node.py` (+95/−1) add only grant verification and
a latch (hold, then sit, above the approved preset). Policy inputs, gains, slew and limits are unchanged.

**Gaps against the requirements:**
1. The contention gate is come-here-only (§3).
2. No temperature gate; battery floor 20 % only.
3. No stopping-distance or clearance measurement step ("fall area clear" is a typed assertion only).
4. Forward measurement is tape only; no camera or synchronized translation record.
5. No yaw. Untethered yaw needs a new grant preset and review (NO-GO by your decision).
6. Staging `06c5a37` changes the gantry W path too (walking grants), so a paired comparison should run **both**
   arms on `06c5a37` and re-run the ledger A–G at that commit. That isn't the 10-07 code identity, though the control path is unchanged.
7. The commits are labelled `wip` and unpushed. They need your review before staging.

## 7. Full FREE_FLOOR rehearsal (fake GO2, mewtwo, `ROS_LOCALHOST_ONLY=1`, domain 87; no robot)
`scripts/free_floor_rehearsal.py` at `06c5a37` (clean clone): **28/28 scenarios PASS, 155 `[ok]`, 0 `[FAIL]`,
rc 0, 1068 s.** Log: mewtwo `~/workspace/friction-validation-repro/ff_rehearsal_all.log`.
- Scenarios:
  - happy reject/accept; stale env; wrong session; artifact swap; review forgery; missing raw log; crash after begin;
  - drill retry / bridge / policy / logger death; no, early or late deadman release; stale lowstate;
  - walk with the bridge dying before or during the command, policy or cmd death, deadman release, lowstate stop, harness SIGKILL;
  - gantry W ok / bad preset / seed 50; grant mutual exclusion; hand-launched node refused.
- What this proves: the software sequencing, latches and refusals behave as designed against a simulated
  GO2. It **doesn't** prove deadman latency, stopping distance or any hardware behaviour.
  `E2E_STOP_MAX_S` (1.0 s) and `E2E_STOP_DESIGN_S` (0.6 s) are design figures, not robot data.

## 8. Offline gate amendment (decision 5: implemented, NOT enabled, NOT staged, NOT committed)
A separate scratch clone, mewtwo `~/workspace/friction-validation-repro/ff_gates` (based on `06c5a37`, uncommitted).
Patch: `../ff_gates_amendment.diff` (6 files, +99/−4). The new test is also copied to `../test_untethered_gate_amendment.py`.
1. **Contention (applies to every stage, including gantry):**
   - `COMPETING_NODE_SUBSTRINGS` adds the stacks seen live on 10-12 (`mosaic`, `whip`, `hardware_bridge`,
     `safety_arbiter`, `go2_bridge`, `lidar_hazard`, nav2 servers, `slam_toolbox`).
   - A new check, **"no unrecognized node in the ROS graph"**: every node must start with `phoenix_` or `_ros2cli_daemon_`.
     Plain `_ros2cli_` isn't allowed, because `ros2 topic pub` runs as `_ros2cli_<pid>`.
   - Gap still open: a non-ROS (bare-DDS) `/lowcmd` publisher is invisible to a node list. That needs a post-release
     `/lowcmd` publisher-count check inside the bridge (not implemented).
2. **CW yaw preset** `yawm05_3s` (wz −0.5, 3 s) in `walk_command.PRESETS`. No grant mode lists it, so no launcher can authorize it.
3. **Untethered yaw gate (disabled):** `FREE_FLOOR_YAW_PRESETS = (yaw05_3s, yawm05_3s)` and `FREE_FLOOR_YAW_ENABLED = False`.
   `check("walk", preset=yaw…)` always refuses. Even with the flag flipped it still refuses until:
   - (a) a real (non-rehearsal) FREE_FLOOR `vx03_2s` walk has a GO result and a valid human *accept*;
   - (b) an IMU-heading yaw analyzer replaces `YAW_EVAL_MISSING`.

   `walk_grant.ALLOWED_PRESETS[FREE_FLOOR]` is unchanged (`stand_zero`, `vx03_2s`), a second independent refusal.
- **Tests** (gate, grant, walk_command, contention, harness_preflight, deploy_contract, gantry_ab, rehearsal, amendment):
  357 collected, 352 passed, 2 failed, 3 skipped. The 2 failures
  (`test_correct_model_2750_is_accepted_…`, `test_a_go_drill_opens_the_walk_…`) fail identically on the
  **unmodified** clone (stage-F hardware-evidence lookup) and pass in the mewtwo worktree, so they're
  environment-dependent and not caused by the amendment. ruff check clean.
- Not done: the full rehearsal on the amended code (its code-identity gate refuses a dirty tree, by design).
  It needs a reviewed commit first.
- Two existing tests were adjusted:
  - The contention fixture node `/lowcmd_bridge` became the real name `/phoenix_lowcmd_bridge`.
  - `yaw05_3s` moved out of the generic "allows only" refusal loop; it now has its own refusal test.
