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
PAYLOAD_REPO=/home/unitree/yusuf/go2-phoenix    # CONFIRMED on site 2026-09-17. Not ~/go2-phoenix.
scripts/stage_payload_repo.sh jetson-cable:$PAYLOAD_REPO           # last line: manifest AND import verified
STAGE_A_SESSION=$SESSION scripts/stage_payload_bundle.sh \
    checkpoints/phoenix-stand-h25-lat-noise \
    configs/sim2real/deploy_stand_h25.yaml \
    jetson-cable:/home/unitree/phoenix/stand-h25-${SHA:0:7}        # last line: transfer AND activation verified
```

## 2. Payload: one prep command, then one `source` per terminal

`scripts/payload_prep.sh` does every payload-side step that cost time on 2026-09-17,
checks each one actually took effect, and writes `payload_env.sh`. It moves no motors,
starts no bridge and publishes nothing. Get the UTC string from the workstation first,
because the payload has no RTC.

```bash
# On the WORKSTATION:
date -u +'%Y-%m-%d %H:%M:%S'

# On the PAYLOAD, from the synced repo:
cd /home/unitree/yusuf/go2-phoenix
./scripts/payload_prep.sh \
    --utc "<the string above>" \
    --expect-sha <full commit stage_payload_repo.sh printed> \
    --bundle /home/unitree/phoenix/stand-h25-<first 7 of that commit>
```

It stops `come-here.service` and proves it stopped (it starves the payload, 132 topics
against 109 and `/lowstate` at 349 Hz against 500, AND it can command the robot); sets and
sanity-checks the clock; points `CYCLONEDDS_URI` at the repo's own
`configs/payload/cyclonedds.xml` and verifies the NIC named in it exists and is UP (a name
that does not exist makes every Go2 topic advertise and carry no data, while
`ros2 topic list` looks perfectly healthy); confirms `python3` imports phoenix from the
synced tree, printing the `.pth` fix rather than `pip install -e .`, which fails on this
payload's pre-PEP-660 setuptools; and runs the contention probe. It exits non-zero and
prints `NOT READY` if any of that fails.

Then, in **every** terminal:

```bash
cd /home/unitree/yusuf/go2-phoenix
source payload_env.sh
```

That sets ROS, `RMW_IMPLEMENTATION`, the `file://` `CYCLONEDDS_URI`, `PHOENIX_EXPECT_SHA`,
`PHOENIX_BUNDLE`, `DEPLOY_CFG`, `PHOENIX_DEADMAN=wireless`, and stops the stale `ros2`
daemon (which otherwise reports the previous environment's topic list).

## 3. Motors OFF: stages A to D

Robot powered, lying folded on the fall mat. Nothing in these stages publishes `/lowcmd`.

```bash
scripts/harness_preflight.sh A --payload   # lock, activation, ORT 1.18.1, payload ONNX vs workstation torch
scripts/harness_preflight.sh B 30          # dryrun: bridge to /lowcmd_dry, synthetic heartbeat (NOT a deadman)
scripts/harness_preflight.sh C             # hold, release, hold the remote's L1 when prompted
                                           # C is OWED: it has never counted. The probe is now
                                           # observation-driven (it advances on a SUSTAINED estop
                                           # state, not on terminal timing), which is what cost
                                           # three failed C runs on 2026-09-17. No clock
                                           # coordination between the two people is needed.
scripts/harness_preflight.sh D 10          # /lowstate, /joint_states, /imu/data rates, gaps, content
scripts/harness_preflight.sh status        # must say: ready for live hold stage E: YES
```

## 3b. Optional: answer the live stages from the GO2 remote

By default every live stage is confirmed at the TERMINAL, so one person types while the
other holds L1 and watches the robot. `src/phoenix/sim2real/operator_remote.py` exists to
remove that relay, and it is now wired in behind an opt-in flag:

```bash
export PHOENIX_OPERATOR_REMOTE=1          # per terminal, before stages F to H
export PHOENIX_OPERATOR_REMOTE_TIMEOUT_S=120
```

With it set, the three prompts in each stand attempt come through the remote instead:

| prompt | remote gesture |
|---|---|
| ready to start | hold L1, press **Start** once |
| did it stand? | **A** = yes, **B** = NO-GO |
| end the attempt | **release L1** (bridge DAMPs) |

It fails closed in every direction: a timeout, an ambiguous press (A and B together) and
a released deadman all HALT the stage. A held button cannot confirm anything, because the
recognizer requires a rising edge. The evidence lands in `remote_<stage><k>_<mode>.json`.

**UNVERIFIED ON HARDWARE.** This path has never run against a real GO2. Leave the flag
unset for the first attempts of the session, confirm the stage passes the ordinary way,
and only then try it if the terminal relay is slowing you down.

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

## The 66.7% clip figure is a FOLDED-START ARTIFACT, not a policy defect

Do not spend the session chasing it. Folded calf readings (-2.77 to -2.82 rad) sit below
the audited URDF limit (-2.7227), so HOLD clips to exactly the limit and the margin reads
zero. The 100% figures are SLEW clipping (target versus measured q, +/- 0.175) with the
motors off and q never moving.

Cheapest confirmation, once stage F or G has run: take the clip rate over the FINAL
SETTLED SECOND of one live stand only, not over the whole window and not from a folded
start.

## What a pass means, and what it does not

* Pass criteria are the gating checks each stage prints: every process alive, the exact
  locked artifacts and commit, fresh sensors, the real deadman armed, continuous policy
  authority for the whole window, no fault before its end, attitude inside the policy's
  abort thresholds, and the operator's confirmation that the robot stood.
* The hardware slew clip percentage is REPORTED from the bridge telemetry
  (`final_target_vs_policy_request_clip_activation_v1`) and never gates. The old 3.65% / 4.23% and 5%
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
