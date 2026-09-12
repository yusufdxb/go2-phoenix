# H25 stand-only hardware run card

**Scope: a commanded-zero STAND, feet on the ground. Walking is refused in code.**
Every step below that moves a motor is its own command, asks for a typed phrase, and
refuses unless every earlier stage counts for the same commit and the same lock.

| item | value |
|---|---|
| deliverable | `checkpoints/phoenix-stand-h25-lat-noise` (model_799) |
| deploy config | `configs/sim2real/deploy_stand_h25.yaml`, `safety.stand_only: true`, `base_lin_vel_source: zeros` |
| lock | `configs/sim2real/locks/deploy_stand_h25.lock.yaml` |
| `policy.onnx` | `5331dbb7c4b8194e7f2238c14b985ea16af2f47aa7130e04bade3a0f9e9289eb` |
| `policy.onnx.data` | `cda0e9b4847c4208ec1aadd447c1851856b9d6f2e38dbadfaae1c23e214dee92` |
| `policy.pt` | `24d42d606f916b70ee5f48c665b3405202d14683daa00dd4c8018dcee7273622` |
| checkpoint (`latest.pt` = model_799) | `c904aebbb1c62eeca77317dcde9260067f3979be3931dee4db263eac4d9fa8f8` |
| config, semantic sha256 | `92ea5ff7b4f939efee8aeecd60ad9410448de36dbaf023ba912cd842511feb32` |

**The one question, one command:** `scripts/harness_preflight.sh status` exits 0 and
prints `READY FOR STAND-ONLY LIVE GO2 TEST (stage F): YES` only when stages A to E all
count. With no argument the script also runs stage A on the machine it is on.

## 0. Workstation, before leaving

```bash
cd ~/workspace/go2-phoenix
git switch feat/causal-viability-replication
git status --short                      # must print nothing
scripts/harness_preflight.sh            # stage A must end "STAGE A: GO"; F stays NO here
SHA=$(git rev-parse HEAD)
SESSION=logs/hw_sessions/$(date -u +%Y%m%d)_${SHA:0:12}
```

## 1. Put the code and the bundle on the payload (over the cable, no `git fetch`)

```bash
PAYLOAD_REPO=/home/unitree/go2-phoenix          # confirm on site: ls -d ~/go2-phoenix ~/workspace/go2-phoenix
scripts/stage_payload_repo.sh jetson-cable:$PAYLOAD_REPO           # last line: manifest AND import verified
STAGE_A_SESSION=$SESSION scripts/stage_payload_bundle.sh \
    checkpoints/phoenix-stand-h25-lat-noise \
    configs/sim2real/deploy_stand_h25.yaml \
    jetson-cable:/home/unitree/phoenix/stand-h25-${SHA:0:7}        # last line: transfer AND activation verified
```

## 2. Payload shell (every terminal)

```bash
cd $PAYLOAD_REPO
source /opt/ros/humble/setup.bash
source ~/unitree_ros2/cyclonedds_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://<xml whose NetworkInterface is enP8p1s0>   # docs/go2_field_notes.md section 1
ros2 daemon stop
export PHOENIX_EXPECT_SHA=<the full commit stage_payload_repo.sh printed>
export PHOENIX_BUNDLE=/home/unitree/phoenix/stand-h25-<first 7 characters of that commit>
export DEPLOY_CFG=$PHOENIX_BUNDLE/deploy_stand_h25.yaml
export PHOENIX_DEADMAN=wireless                                         # GO2 remote, L1
```

## 3. Motors OFF: stages A to D

Robot powered, lying folded on the fall mat. Nothing in these stages publishes `/lowcmd`.

```bash
scripts/harness_preflight.sh A --payload   # lock, activation, ORT 1.18.1, payload ONNX vs workstation torch
scripts/harness_preflight.sh B 30          # dryrun: bridge to /lowcmd_dry, synthetic heartbeat (NOT a deadman)
scripts/harness_preflight.sh C             # hold, release, hold the remote's L1 when prompted
scripts/harness_preflight.sh D 10          # /lowstate, /joint_states, /imu/data rates, gaps, content
scripts/harness_preflight.sh status        # must say: ready for live hold stage E: YES
```

## 4. Physical setup for every live stage

* Feet on the ground, on the fall mat, starting folded. Tether clipped or a spotter's
  hands at the harness. 2 m clear radius.
* Person 1 holds the GO2 remote with L1 held and watches only the robot. Person 2 runs
  the terminal and spots. Releasing L1 is the stop.
* Low-level mode: release the sport service with the same procedure used for the
  2026-04-21 live session. Never call motion_switcher `SelectMode` with anything but
  `mcf` or `ai`; `normal` wedges the robot until a power cycle.
* A suspended or stand-fixture setup is an out-of-distribution diagnostic only. It is
  never this gate and its evidence never counts.

## 5. Motors LIVE: stages E to H

```bash
scripts/harness_preflight.sh E 10    # bridge LIVE, holds measured posture 10 s, no policy
scripts/harness_preflight.sh status
scripts/harness_preflight.sh F       # H25 cmd=0 stand, 2 s of policy authority
scripts/harness_preflight.sh status
scripts/harness_preflight.sh G       # 5 s
scripts/harness_preflight.sh status
scripts/harness_preflight.sh H       # 10 s, three attempts, each started and ended by the operator
scripts/harness_preflight.sh status
```

Each stand attempt: bridge starts in HOLD, then the policy node gets exactly the stage's
authority window, then its single abort notice returns the bridge to HOLD. The script
asks whether the robot stood without collapse, oscillation or buzz, then asks for Enter
to DAMP (kp 0): the robot sinks onto the mat, so the spotter must be ready.

## HALT: release L1 immediately, then Ctrl-C the terminal

* A joint snaps, or buzz or hard saturation on any motor.
* The base tips past about 25 degrees or starts a divergent oscillation.
* Any process dies, `/joint_states` or `/imu/data` drop out, the Jetson reboots or browns out.
* The robot moves while stage E says HOLD, or at all before the policy window starts.
* Anything unexpected. A halted stage is a NO-GO; re-run it, never skip it.

## What fails closed on its own (you should see these, not fight them)

| event | final bridge does |
|---|---|
| L1 released, or deadman heartbeat older than 0.5 s | latches HOLD of measured posture |
| `/phoenix/estop` published by anything but the one real deadman node | refuses to arm, latches HOLD |
| LowState older than 0.2 s | latches HOLD, then DAMP 0.2 s later |
| NaN command, wrong joint order, wire version mismatch | latches HOLD |
| requested target beyond a hard joint limit by more than 0.175 rad | latches HOLD |
| measured joint beyond a hard limit by more than 0.175 rad, or NaN LowState | latches DAMP |
| nonzero velocity command | policy node aborts; bridge latches HOLD |
| policy abort of any kind, including the end of its window | HOLD, never a drive to the stand pose |
| Ctrl-C on the bridge | DAMP for 0.2 s, then exit |

## What a pass means, and what it does not

* Pass criteria are the gating checks each stage prints: every process alive, the exact
  locked artifacts and commit, fresh sensors, the real deadman armed, continuous policy
  authority for the whole window, no fault before its end, attitude inside the policy's
  abort thresholds, and the operator's confirmation that the robot stood.
* The hardware slew clip percentage is REPORTED from the bridge telemetry
  (`bridge_final_slew_clip_activation_v1`) and never gates. The old 3.65% / 4.23% and 5%
  numbers are legacy figures of a different quantity; see the config header.
* Passing H does not permit walking. The walking prerequisites are listed in
  `phoenix.sim2real.deploy_contract.WALKING_PREREQUISITES`, starting with validating
  `/utlidar/robot_odom` from a bag with known motion.

## Evidence to keep

The whole session directory: every `stage_<X>.json`, every run directory with
`bridge.jsonl` (one line per bridge tick: raw action, requested and clipped targets,
final LowCmd target, per-joint clip flags and margins, measured q and dq, command,
LowState, IMU, joint-state and estop ages, abort reason, observation source, the
base_lin_vel and velocity command actually fed), the node logs and the probe JSON.
