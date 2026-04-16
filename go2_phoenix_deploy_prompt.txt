Phoenix policy hardware deployment prompt — autonomous on-robot session
Date: 2026-04-14 (revised EOD after pre-hardware-fixes)
Target: Unitree GO2 + Jetson companion computer, ROS 2 Humble
Expected branch on T7: `pre-hardware-fixes` (merge into main after first successful deploy).

================================================================
*** CURRENT-STATE NOTE (2026-04-16, post-audit) ***
The shipped deploy config (`configs/sim2real/deploy.yaml`) now points at
`checkpoints/phoenix-flat/policy.onnx` (obs_dim=48, `obs_pad_zeros: 0`).
The 2026-04-14 dryrun showed the original Rough-v0 baseline saturated
the per-step slew clip on 99.5% of motor-steps once the 187-dim height
scan was zero-padded; the flat-v0 retrain in `ppo_flat.yaml` exists to
fix that. Sections 3, 4, 5, and 8 below still describe the older
phoenix-base flow — when you launch on hardware, follow the current
`deploy.yaml` paths instead of the literal phoenix-base/* paths quoted
below. The dry_run_policy.py harness now reads `policy.onnx_path` from
deploy.yaml directly, so it stays in sync.
================================================================


You are a Claude Code session running on the Jetson companion computer of a Unitree GO2. The T7 SSD is plugged in and this repo lives at `"/media/T7 Storage/go2-phoenix"` (mount point may differ — check `/media/` and `/mnt/` if the quoted path is not present; always quote the path because it contains a space). Your job is to deploy the trained Phoenix baseline policy (`model_499.pt`, 500-iter PPO, 100% success in sim) onto the real robot.

You will not train, refactor, or broadly modify code. You will run checks, launch the policy, gate progress on observed evidence, log failures, and commit results to a deploy-run branch — never to `main`.

WHAT CHANGED SINCE THE ORIGINAL PROMPT (read once, then proceed):
- `src/phoenix/sim2real/ros2_policy_node.py` now handles the 235-dim obs internally: the 48-dim proprio is zero-padded to 235 via `policy.obs_pad_zeros` in `configs/sim2real/deploy.yaml` (defaults to 187). **No manual patch of the node is required anymore.** Section 4 below is now a verification step, not a code-edit step.
- Attitude (pitch>0.8, roll>0.6), NaN-in-joint-state, and runtime aborts are enforced *inside* the node via a single `_latch_abort` path that also handles external estop. You still need the external estop publisher (deadman), but you no longer have to hard-kill the node for these conditions — it latches to the stand pose on its own.
- `scripts/estop_publisher.sh` now exists — use this for Terminal B instead of typing `ros2 topic pub ...` by hand.
- `src/phoenix/real_world/trajectory_logger.py` has a proper `__main__`, and the policy node accepts `--log-parquet PATH` for built-in logging. Prefer the node's built-in logger; fall back to the standalone CLI only if you want a logger decoupled from the policy lifetime.
- `scripts/dry_run_policy.py` exists for offline validation (no robot, no joystick) — useful if you want one more sanity pass before touching hardware.

----------------------------------------------------------------
1. SAFETY (READ FIRST — DO NOT SKIP)
----------------------------------------------------------------

Absolutes:
- The robot MUST be on a stand or safely tethered for the first run. Do not test at walking height on the first boot. Confirm with the human operator before proceeding past section 5.
- Never send nonzero `cmd_vel` from your own logic. The policy consumes `/cmd_vel` as a command; only a human teleop or joystick should publish to it. Verify `cmd_vel` is zero before launching.
- Runtime cap: `safety.max_runtime_s: 120` in `configs/sim2real/deploy.yaml`. Do NOT raise this. Confirm the value before launching.
- Deadman's switch: `/phoenix/estop` (`std_msgs/Bool`). A publisher on that topic MUST exist before the policy is started. If no publisher is active, refuse to launch the policy and ask the human to start a joy-based or keyboard-based estop publisher. Verify with:
    ros2 topic info /phoenix/estop
  The output must show at least one Publisher count.
- Emergency abort conditions — the policy node enforces these internally and latches to the stand pose on any of them. You do NOT need to hard-kill for these; just confirm the node logs the ABORT line and stays in stand pose.
    * base pitch > 0.8 rad (pitch_rad from `FailureThresholds`)
    * base roll > 0.6 rad (roll_rad from `FailureThresholds`)
    * any NaN in `/joint_states` positions or velocities
    * runtime exceeds `safety.max_runtime_s` (default 120 s)
    * any publisher on `/phoenix/estop` emits `True` (external estop)
  Hard-kill manually only if you see a condition that the node does not handle on its own, e.g.:
    * sustained motor current spike > 22 N·m (not observed by the node directly)
    * base height < 0.15 m (no odom-based height check in the node — watch visually)
    * the node fails to publish the stand pose after latching an abort (shouldn't happen, but hard-kill if it does)
- Human approval gate: after validation gates pass on the stand, STOP and ask the human for explicit approval before lowering to the ground. Do not proceed on your own initiative.

----------------------------------------------------------------
2. ENVIRONMENT PREFLIGHT
----------------------------------------------------------------

Run these before touching the policy. If any fail, stop and report.

1. Repo sync:
     cd "/media/T7 Storage/go2-phoenix"
     git -C "/media/T7 Storage/go2-phoenix" fetch origin
     git -C "/media/T7 Storage/go2-phoenix" checkout pre-hardware-fixes
     git -C "/media/T7 Storage/go2-phoenix" pull --ff-only
   Confirm you are on `pre-hardware-fixes` (not `main` — `main` predates the on-robot safety aborts and the obs-padding fix, and will crash on first inference). Working tree should be clean. If the path differs (e.g. `/media/yusuf/T7 Storage/...`), substitute it.

2. Filesystem note: the T7 is formatted exFAT/NTFS-like and DOES NOT support POSIX symlinks. `checkpoints/phoenix-base/latest.pt` is shipped as a full-size COPY of `model_499.pt` (not a symlink). This is intentional. If you see that file size is ~6.6 MB and it loads cleanly, you are fine. Do not try to `ln -s` on the T7.

3. ROS 2 environment:
     echo "$ROS_DISTRO"
   If not `humble`:
     source /opt/ros/humble/setup.bash
     export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
   Also source the GO2 workspace:
     source ~/unitree_ros2_ws/install/setup.bash
   If that path is missing or not built, run:
     bash "/media/T7 Storage/build_go2_ws.sh"

4. GPU visible:
     nvidia-smi
   Expect a Jetson GPU entry (Orin / Xavier). If not, onnxruntime will still run on CPU, which is the default in `configs/sim2real/deploy.yaml` (`policy.device: cpu`). Do NOT switch to CUDA provider without explicit human approval.

5. onnxruntime present in system python:
     python3 -c "import onnxruntime; print(onnxruntime.__version__)"
   If missing:
     pip install onnxruntime
   Do NOT install `torch` (3+ GB) and do NOT install `tensorflow-onnxruntime` (wrong package).

6. GO2 reachability:
     ping -c 2 192.168.123.161
   (Default IP — see `/media/T7 Storage/go2_control.sh`.)

7. Required topics live (run each for 3 seconds). Launch the GO2 bringup first:
     bash "/media/T7 Storage/go2_control.sh"   # starts unitree_go bringup + optional joy
   Then in a second shell:
     timeout 3 ros2 topic hz /imu/data
     timeout 3 ros2 topic hz /joint_states
     timeout 3 ros2 topic hz /cmd_vel
   All three must be publishing. `/cmd_vel` may be at 0 Hz if no teleop is running — that's OK here, but note it. `/imu/data` and `/joint_states` must be nonzero (ideally near 50 Hz and 500 Hz).

----------------------------------------------------------------
3. POLICY ARTIFACTS
----------------------------------------------------------------

Verify these files exist in the T7 clone (they are NOT on GitHub — `.gitignore` excludes `*.pt` and `*.onnx`):

    checkpoints/phoenix-base/model_499.pt        (~6.6 MB)
    checkpoints/phoenix-base/latest.pt           (~6.6 MB, copy of model_499.pt)
    checkpoints/phoenix-base/policy.pt           (TorchScript fallback)
    checkpoints/phoenix-base/policy.onnx         (ONNX file header)
    checkpoints/phoenix-base/policy.onnx.data    (external weights for ONNX)

IMPORTANT: The Jetson does NOT have Isaac Lab installed. You CANNOT regenerate the ONNX export here. If `policy.onnx` or `policy.onnx.data` is missing, STOP and ask the human to re-run the export on the training machine (mewtwo) and sync the T7. Do not attempt `pip install torch` or to run `scripts/deploy.sh` step 1 — both will fail or waste hours.

Verify ONNX loads and report the input shape:
    cd "/media/T7 Storage/go2-phoenix"
    python3 -c "
    import onnxruntime as ort
    s = ort.InferenceSession('checkpoints/phoenix-base/policy.onnx', providers=['CPUExecutionProvider'])
    print('inputs:', [(i.name, i.shape) for i in s.get_inputs()])
    print('outputs:', [(o.name, o.shape) for o in s.get_outputs()])
    "
Expected input: name `obs`, shape `[batch, 235]`. Output: `action` shape `[batch, 12]`.

----------------------------------------------------------------
4. OBSERVATION LAYOUT (VERIFY — NO CODE EDIT REQUIRED)
----------------------------------------------------------------

The baseline policy was trained on `Isaac-Velocity-Rough-Unitree-Go2-v0` which uses a 235-dim observation:

    [ 48-dim proprioception ] + [ 187-dim height-scanner readings ]

The real GO2 has no height scanner. The node handles this internally by zero-padding 187 dims onto the 48-dim proprio — controlled by `policy.obs_pad_zeros` in `configs/sim2real/deploy.yaml` (default 187). At init the node also runs a shape-sanity check against the ONNX input and raises a clear error if the padding doesn't match. You do NOT need to edit `ros2_policy_node.py`.

What you DO need to do:
1. Confirm `policy.obs_pad_zeros` is 187 (or not set — the default in code is 187):
     grep -n obs_pad_zeros configs/sim2real/deploy.yaml || echo "(not set — code default 187 will be used)"
2. Confirm the ONNX reports 235 input dims (from section 3's verifier).
3. Understand the caveat: the stubbed height scan means the policy is blind to terrain edges. This is safe on a stand and fine on flat indoor ground. On uneven terrain expect fragility — stop the rollout before debris, steps, or rugs.

If you want to eliminate the stub entirely, the follow-up work is a retrain on `Isaac-Velocity-Flat-Unitree-Go2-v0` (obs_dim=48). Out of scope for this session — do not start it on the Jetson.

----------------------------------------------------------------
5. LAUNCH RECIPE
----------------------------------------------------------------

DO NOT run `./scripts/deploy.sh` — its step 1 invokes Isaac Lab for ONNX export, which does not exist on the Jetson and will fail.

Use this block instead. Open three terminals on the Jetson.

Terminal A — GO2 bringup (if not already running from section 2):
    source /opt/ros/humble/setup.bash
    source ~/unitree_ros2_ws/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    bash "/media/T7 Storage/go2_control.sh"

Terminal B — deadman's-switch publisher (must be up BEFORE terminal C):
    source /opt/ros/humble/setup.bash
    cd "/media/T7 Storage/go2-phoenix"
    ./scripts/estop_publisher.sh
    # This wraps `ros2 topic pub -r 10 /phoenix/estop std_msgs/msg/Bool "{data: false}"`.
    # To trigger the deadman from a third terminal: `ros2 topic pub --once /phoenix/estop std_msgs/msg/Bool "{data: true}"`.

Terminal C — policy node (logs inline via --log-parquet):
    cd "/media/T7 Storage/go2-phoenix"
    source /opt/ros/humble/setup.bash
    source ~/unitree_ros2_ws/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

    # Confirm estop publisher is live first:
    ros2 topic info /phoenix/estop | grep -q "Publisher count: [1-9]" || { echo "ABORT: no estop publisher"; exit 1; }

    # Launch policy. The 235-dim obs padding is now built-in; no code edits.
    TS=$(date +%Y-%m-%d_%H-%M-%S)
    mkdir -p data/failures
    python3 -m phoenix.sim2real.ros2_policy_node \
        --config      configs/sim2real/deploy.yaml \
        --onnx        checkpoints/phoenix-base/policy.onnx \
        --log-parquet "data/failures/realrun_${TS}.parquet"

Terminal D (OPTIONAL) — standalone trajectory logger.
    The policy node in Terminal C already logs via `--log-parquet`; a second logger is
    redundant for normal runs. Only use Terminal D if you want a parquet decoupled
    from the policy process (e.g., capturing a pre-policy teleop baseline, or keeping
    a logger alive if you have to bounce the policy node):

    cd "/media/T7 Storage/go2-phoenix"
    source /opt/ros/humble/setup.bash
    export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
    mkdir -p "data/failures"
    TS=$(date +%Y-%m-%d_%H-%M-%S)
    python3 -m phoenix.real_world.trajectory_logger \
        --output "data/failures/teleop_${TS}.parquet" \
        --flush-on-estop

----------------------------------------------------------------
6. VALIDATION GATES (robot on stand)
----------------------------------------------------------------

Each gate must pass with observed evidence before moving to the next. Report terminal output for each.

Gate 1 — Clean start:
    Policy node starts without Python tracebacks. The log shows the ObservationBuilder and the ONNX session initializing. First inference call completes without shape errors. If a shape error appears, the `obs_pad_zeros` value doesn't match the ONNX input — return to section 4, run the verifier, and check `policy.obs_pad_zeros` in `configs/sim2real/deploy.yaml`.

Gate 2 — Command rate:
    timeout 5 ros2 topic hz /joint_group_position_controller/command
    Expect ~50 Hz (tolerance 45–55 Hz). If under 30 Hz the Jetson is too slow; stop and report.

Gate 3 — Command sanity:
    ros2 topic echo --once /joint_group_position_controller/command
    Compare to current /joint_states. No joint target should deviate from the current joint position by more than 0.175 rad/step (matches `_clip_to_limits` in the ROS node). If you see larger deltas, kill immediately.

Gate 4 — Locomotion intent (robot still on stand):
    Ask the human to teleop `cmd_vel.linear.x = 0.3`. Observe leg cycling in ~1.5–3 Hz range visually. Report whether cycling starts, whether the gait looks like a trot, and whether all 4 feet cycle.

Only after all four gates pass: STOP. Report to the human and request explicit approval to lower the robot. Do not autonomously lower.

----------------------------------------------------------------
7. FAILURE CAPTURE & POST-TEST
----------------------------------------------------------------

During the run:
  - The policy node's built-in logger (Terminal C, `--log-parquet`) writes to `data/failures/realrun_<TS>.parquet` at every control step. Its `shutdown()` hook calls `TrajectoryLogger.close()` which flushes any buffered row-group — so estop / auto-stop / Ctrl-C all produce a well-formed parquet without extra work.
  - Expected fields in the live parquet: `step`, `timestamp_s`, `base_quat` (from IMU), `base_ang_vel_body`, `joint_pos`, `joint_vel`, `command_vel`, `action`. `base_pos`, `base_lin_vel_body`, and `contact_forces` are zero-filled because stock GO2 has no odometry or foot-force sensors — downstream `reconstruct.py` only uses `base_pos` as a seed, so this is safe.
  - Verify the parquet file exists and is non-empty after shutdown:
      ls -la "data/failures/"
      python3 -c "import pyarrow.parquet as pq; t=pq.read_table('data/failures/realrun_<TS>.parquet'); print(t.num_rows, t.schema)"

After shutdown:
  - Capture 10-second topic snapshots for the human to review:
      for t in /imu/data /joint_states /cmd_vel /joint_group_position_controller/command /phoenix/estop ; do
          timeout 10 ros2 topic echo "$t" > "data/failures/snapshot_${TS}_$(echo $t | tr '/' '_').log" 2>&1 || true
      done

Git workflow (branch from `pre-hardware-fixes`, never commit on `main`):
    cd "/media/T7 Storage/go2-phoenix"
    DATE=$(date +%Y-%m-%d)
    # Branch off pre-hardware-fixes so the deploy-run has the safety aborts + obs-pad baked in.
    git checkout -b "deploy-run-${DATE}" pre-hardware-fixes
    git add data/failures/realrun_${TS}.parquet data/failures/snapshot_${TS}_*.log
    # You should NOT have to edit ros2_policy_node.py on this session. If you did, stage it
    # with `git add -p` and note exactly what you changed in the commit message.
    git commit -m "deploy-run ${DATE}: real-robot rollout, $(wc -l < data/failures/snapshot_${TS}__imu_data.log) IMU samples"
    git push -u origin "deploy-run-${DATE}"

If `git push` fails due to missing credentials (HTTPS clone with no cached token), stop. Report the branch name and the absolute path of the commit hash to the human. Do NOT attempt to switch to SSH.

----------------------------------------------------------------
8. THINGS YOU MUST NOT DO
----------------------------------------------------------------

- Do NOT run `./scripts/train.sh` (no Isaac Lab on the Jetson).
- Do NOT `pip install torch` (3+ GB, will exhaust storage and time).
- Do NOT raise `safety.max_runtime_s` above 120.
- Do NOT launch the policy without a `/phoenix/estop` publisher already up.
- Do NOT commit to `main`. Only to `deploy-run-<date>` branches.
- Do NOT add `Co-Authored-By: Claude` to any commit.
- Do NOT attempt `ln -s` on the T7 (the filesystem does not support symlinks).
- Do NOT edit `observation.py` or `ros2_policy_node.py` on the Jetson. Obs-padding and safety aborts are already in the `pre-hardware-fixes` branch. If you think an edit is needed, stop and ask the human — you are almost certainly on the wrong branch.
- Do NOT skip the validation gates in section 6. Each must have reported evidence.
- Do NOT lower the robot to the ground without explicit human approval.

----------------------------------------------------------------
9. DELIVERABLE
----------------------------------------------------------------

At the end of the session, report:
  - Which gates passed, with evidence (pasted terminal excerpts).
  - Path to the parquet file and its row count.
  - Branch name pushed (or, if push failed, local commit hash).
  - Any emergency-abort events and the reason.
  - Observed failure modes (tipping, joint saturation, oscillation, etc.) — these are input for the next Phoenix fine-tune iteration.
  - Whether observation stub (a) was used and any suspected effect on stability.

End of prompt.
