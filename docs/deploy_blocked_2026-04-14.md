# Phoenix Hardware Deploy — Session Blocked (2026-04-14)

Status: **blocked at runbook section 2 step 7** (required topics missing). No policy rollout occurred. Robot was not commanded. No failure-mode data produced.

## TL;DR

The deploy prompt assumes a bridge layer exists between the Unitree GO2 firmware and standard ROS 2 topic names. That bridge does not exist on this Jetson. Writing it is a precondition for the next attempt.

## What passed preflight

| Check | Result |
|---|---|
| Repo on `pre-hardware-fixes` @ `a79c187` | PC + Jetson match origin |
| onnxruntime on Jetson | 1.18.1 (downgraded from 1.23.2 — see below) |
| ONNX input/output shape | `obs [batch,235]` → `action [batch,12]` ✓ |
| ONNX inference latency | 0.14 ms/run on Jetson Orin CPU, ~7 kHz ceiling |
| `configs/sim2real/deploy.yaml` | `obs_pad_zeros` default 187 (unset, OK); `safety.max_runtime_s=120`; `dead_mans_switch=true` |
| GO2 reachable | `ping 192.168.123.161` 0% loss, ~7 ms |
| Policy artifacts | `model_499.pt`, `latest.pt`, `policy.onnx`, `policy.onnx.data`, `policy.pt` all present on Jetson, md5 matches PC |

## What blocked

Runbook section 2 step 7 requires `/imu/data`, `/joint_states`, `/cmd_vel` to be publishing. On the Jetson right now none of those exist, and `/joint_group_position_controller/command` has no subscriber.

Actual topic surface on Jetson DDS bus (109 topics):

- Unitree native (via `unitree_sdk2_ros` bridge): `/lowstate`, `/lowcmd`, `/sportmodestate`, `/wirelesscontroller`, `/utlidar/imu`, `/api/sport/*`
- helix_bringup: `/helix/*` (diagnostics, no motion control)
- come_here_bringup (mock): its own nodes, `use_mock:=true` so no motor contact
- Standard ROS names Phoenix needs: **none**

## Why the bridge isn't trivial

The GO2's native `/lowstate` → `/joint_states + /imu/data` translation is the easy half. The hard half is the command path:

- Phoenix publishes `std_msgs/Float64MultiArray` of 12 joint targets to `/joint_group_position_controller/command` at 50 Hz.
- The GO2 firmware listens on `/lowcmd` (`unitree_go/msg/LowCmd`) for a structured message with `motor_cmd[20]`, each containing `mode`, `q`, `dq`, `kp`, `kd`, `tau`, and a `crc` field computed over the whole struct.
- Stock `go2_ros2_sdk/go2_driver_node` only subscribes to `cmd_vel_out` (Twist, high-level sport mode). It does **not** consume low-level joint commands.

The GO2 must also be put into **low-level mode** (controller L2+A then L2+B, or via sport_lease API) before `/lowcmd` is honored. In sport mode, `/lowcmd` is ignored.

## What needs to exist before re-running the prompt

1. **Low-level bridge node.** Subscribes to `/joint_group_position_controller/command`. Maps each index via `deploy.yaml:joint_order` (FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, …). Constructs `unitree_go/msg/LowCmd` with:
   - `motor_cmd[i].mode = 0x01` (servo position mode)
   - `motor_cmd[i].q = target`, `dq = 0`
   - `motor_cmd[i].kp`, `kd` from config (suggest kp=25, kd=0.5 as GO2 stand-safe defaults; tune before walking)
   - `motor_cmd[i].tau = 0`
   - `crc = unitree_sdk2::crc32_core(cmd)` — reference `~/go2_ws/src/unitree_ros2/example/src/src/go2/go2_stand_example.cpp`
   - Publish to `/lowcmd` at 50 Hz
2. **Mode management.** Helper to flip to low-level mode on startup and back to sport mode on abort. Or document: "manually L2+A L2+B before launching."
3. **Topic surface.** Launch `go2_ros2_sdk`'s `robot.launch.py` with the minimum set (`rviz2:=false nav2:=false slam:=false foxglove:=false`) after exporting `ROBOT_IP=192.168.123.161`. Remap `imu:=imu/data` so the deploy.yaml path works unchanged. Alternatively change `deploy.yaml:topics.imu` from `/imu/data` to `/imu`.
4. **Safety-abort coverage.** The policy node's abort latch publishes the stand pose to `/joint_group_position_controller/command`. Verify the new bridge correctly routes the stand-pose Float64MultiArray into `/lowcmd` with safe kp/kd, and that the stand pose is actually safe when the GO2 is held at height (i.e., it doesn't just drop the hips).
5. **onnxruntime pin.** Pin `onnxruntime==1.18.1` in `pyproject.toml` (or in the runbook's preflight), and stash a copy of `onnxruntime-1.18.1-cp310-cp310-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl` at `checkpoints/wheels/` (gitignored) or T7:LABWORK/PHOENIX/wheels/. Reason below.

## Environment notes for next session

- **onnxruntime 1.23.2 (shipped default) crashes on Jetson Orin CPU.** The PyPI prebuilt wheel trips a `std::vector` assertion in ORT's CPU-detect code because it doesn't recognize Cortex-A78AE (CPU part 0xd42). 1.19.2 has the same bug. **1.18.1 works** (affinity pthread warnings print but are non-fatal). The runbook says "if missing, `pip install onnxruntime`"; that command without version pinning will reinstall the broken version.
- **exFAT-FUSE `scp` corrupts text files.** Transferring `go2-phoenix` from the T7 to Jetson via `scp -r` wrote many text files as byte-correct-length but zero-filled. The binary `.pt`/`.onnx` checkpoints were fine. Workaround that worked on the first attempt: `tar c go2-phoenix | ssh ... tar x`. Reason is a FUSE read pattern bug, not a network bug. Don't trust `scp -r` from `/media/careslab/T7 Storage/`.
- **Jetson has no internet** (DNS resolution fails on "CaresLab" hotspot; `github.com` unreachable). Plan transfers from the PC, not clones on the Jetson.
- **The T7 is mounted on the PC at `/media/careslab/T7 Storage`** (quoted — contains a space), not `/media/T7 Storage` as the prompt says. The runbook's "mount point may differ" caveat applies.
- **SSH to Jetson:** `sshpass -p '123' ssh unitree@192.168.0.2` (CaresLab WiFi; previous memory entry for 172.20.10.6 hotspot is stale).
- **Repo location on Jetson:** `/home/unitree/yusuf/go2-phoenix` (user requested `yusuf/` prefix).
- **Active ROS 2 sessions on the Jetson** (running when this session started, user said leave them): `helix_bringup helix_sensing.launch.py` and `come_here_bringup come_here.launch.py use_mock:=true`. Kill these before a real low-level deploy — their DDS peers will collide with any low-level controller and the come-here `go2_bridge_node` could end up publishing into a shared `/cmd_vel`.

## Suggested first steps next session

1. Write `src/phoenix/sim2real/lowcmd_bridge_node.py` mirroring `go2_stand_example.cpp`. Keep it tiny: subscriber + publisher + a CRC helper. Add a unit test that feeds a fixed Float64MultiArray and asserts the resulting `LowCmd` has the right `q[i]` ordering. Don't put it behind ros2_control — the simpler the bridge, the fewer moving parts at 50 Hz.
2. Dry-run the bridge + policy offline using `scripts/dry_run_policy.py` — it already synthesizes `/imu/data` + `/joint_states`. Modify to also assert on `/lowcmd` output.
3. Only then re-run the hardware prompt, explicitly confirming low-level mode before Terminal C.

## Resume trigger

Next Claude session: say `continue phoenix deploy` and point to `T7:LABWORK/PHOENIX/RESUME_NEXT.md`.

## Artifacts from this session

- This doc (committed in repo on branch `deploy-run-2026-04-14`).
- `T7:LABWORK/PHOENIX/RESUME_NEXT.md` — trigger pointer.
- No parquet, no topic snapshots, no rollout data.

Commit: see branch `deploy-run-2026-04-14`, parent `a79c187`.
