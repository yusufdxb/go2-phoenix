# Lab findings — 2026-04-21

Branch: `audit-fixes-2026-04-16` at `3c52b05` + one-line patch (this session).
Operator: lab-PC SSH session driving the Jetson at `192.168.0.2`; human
next to the robot for mode-switching / physical intervention.
Baseline from previous session: stand-v2 DRY saturation **16.67%** on
the 04-20 dryrun (see `lab_findings_2026-04-20.md`), concentrated on
rear thighs — predicted to "unmask" to a real number once the GO2 was
switched to low-level mode and motors engaged.

---

## Headline

**First live Gate 7 run. 120 s, 5961 rows at 49.7 Hz, parquet footer
clean. Saturation 33.06% — worse than the DRY prediction, concentrated
on FL/RL/RR thigh and FL calf.** Gate 7 remains blocked. The cause is
not one thing; it is two failure modes stacked:

1. **Stand-posture offset.** With the GO2 in low-level mode on the
   stand, rear thighs hang unloaded at q=0.630 (RL) / 0.954 (RR) vs
   trained default 1.0. RL's 0.37 rad gap alone is 2× the 0.175 rad
   per-step slew clip — saturation is guaranteed on that joint
   regardless of what the policy outputs.
2. **Out-of-distribution policy output.** On a stand the feet do not
   contact ground, `qd ≈ 0`, and the proprioceptive observation is
   outside anything stand-v2 saw in training (sim had ground contact
   always). The policy responds with large asymmetric actions —
   mean|a|=0.90 on FL_thigh, 0.79 on RR_thigh, 0.69 on FR_calf. With
   `action_scale=0.25`, |0.25·a| crosses the 0.175 clip on its own,
   even for joints whose measured q is essentially at default.

Both are training-distribution problems. Neither is fixable on the
Jetson.

---

## What was executed

1. **Synced the Jetson to `audit-fixes-2026-04-16 @ 3c52b05`** (this
   branch carries the post-lab 04-20 fixes: parquet-footer abort flush
   and `safety.first_message_timeout_s` discovery gate). rsync from
   T7 over SSH; Jetson had `deploy-run-2026-04-14` on checkout.

2. **Discovered a cyclonedds env mismatch on first launch attempt.**
   The 04-20 estop_publisher/bridges were run without
   `CYCLONEDDS_URI`; this session's policy node *did* set it. The
   split DDS configs meant the policy saw `/phoenix/estop` (topic
   info reported Publisher count: 1, Subscription count: 2) but its
   steady-state heartbeat check fired within 0.5 s anyway, aborting
   on `estop_heartbeat_stale` before inference started.

3. **Restarted the entire pipeline under a single unified env**
   (`CYCLONEDDS_URI=.../cyclonedds_ws/src/cyclonedds.xml`,
   `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`) in the order:
   `estop_publisher` → `lowstate_bridge_node` → `lowcmd_bridge_node --live`
   → `ros2_policy_node`. First attempt with canonical timeouts
   (`estop_timeout_s: 0.5`, `sensor_timeout_s: 0.2`) aborted on
   `sensor_stale` ~12 s after policy start. Jetson then hung /
   rebooted under the post-abort `_publish_default_pose` load (see
   "Bugs observed #1" below) — **GO2 went "shaky" on the stand during
   the reboot window**. Operator switched GO2 to sport mode; robot
   stabilized before any further action from the lab-PC side.

4. **Patched `ros2_policy_node.py` to stop the per-tick default-pose
   rebroadcast after abort.** One-liner:
   ```python
   if self._estopped:
       return   # was: self._publish_default_pose(); return
   ```
   The initial abort-time publish still fires once, which is bounded
   by the bridge's per-step clip (0.175 rad). The steady-state
   rebroadcast at 50 Hz was the brownout driver: measured rear-thigh
   posture ≈0.63 rad, commanded default_q=1.0 → bridge walked target
   up 0.175 rad per tick, kp=25 drew sustained current against the
   stand, Jetson brownout / reboot.

5. **Created a Jetson-local `deploy_stand_v2.yaml`** (archived at
   `docs/attachments/2026-04-21/jetson_deploy_stand_v2.yaml`) with
   `estop_timeout_s: 2.0`, `sensor_timeout_s: 1.0` — the 04-18/04-20
   workaround for cyclonedds heartbeat jitter under ONNX warmup
   load. `first_message_timeout_s: 15.0` left at canonical. Canonical
   timeouts remain in the T7-side `configs/sim2real/deploy_stand_v2.yaml`.

6. **Relaunched, ran the full 120 s Gate 7 rollout to `max_runtime`
   abort.** Clean parquet footer, no brownout, no reboot. Artifact:
   `data/failures/gate7_live_2026-04-21_18-33-17.parquet` (5961 rows,
   119.84 s, 49.7 Hz effective).

7. **Extracted per-joint rest posture** from the live parquet for
   use as the training default in a stand-v3 retrain. Medians are in
   `docs/attachments/2026-04-21/rest_posture_extraction.json`.

---

## Saturation table (parquet `gate7_live_2026-04-21_18-33-17`)

5961 rows, 119.84 s, 49.7 Hz effective; 12 joints × 5961 = 71532
motor-steps. **Overall saturation: 33.06%** (23645 / 71532).

| joint               | sat%    | max\|Δ\| | mean\|Δ\| | q_mean  | target_mean | dominant cause |
|---------------------|--------:|-------:|--------:|--------:|------------:|----------------|
| FL_hip_joint        |   0.1%  | 0.189  |  0.076  | -0.044  |    0.094    | — |
| FR_hip_joint        |   0.0%  | 0.188  |  0.067  |  0.025  |   -0.101    | — |
| RL_hip_joint        |   0.0%  | 0.163  |  0.116  |  0.084  |    0.233    | — |
| RR_hip_joint        |   0.0%  | 0.046  |  0.024  |  0.090  |    0.124    | — |
| **FL_thigh_joint**  | **99.8%** | **0.223** | **0.213** |  0.778  |    0.575    | **policy mag** (a=-0.90) |
| FR_thigh_joint      |   0.0%  | 0.046  |  0.015  |  0.669  |    0.655    | — |
| **RL_thigh_joint**  | **100.0%** | **0.470** | **0.465** |  0.636  |    1.098    | **posture offset + a=+0.39** |
| **RR_thigh_joint**  | **100.0%** | **0.309** | **0.281** |  0.954  |    1.198    | **policy mag** (a=+0.79) |
| **FL_calf_joint**   | **96.7%** | **0.211** | **0.200** | -1.593  |   -1.398    | **both** (q low, a=+0.41) |
| FR_calf_joint       |   0.0%  | 0.154  |  0.101  | -1.470  |   -1.327    | — |
| RL_calf_joint       |   0.0%  | 0.163  |  0.128  | -1.512  |   -1.363    | — |
| RR_calf_joint       |   0.0%  | 0.118  |  0.073  | -1.527  |   -1.436    | — |

Action magnitude (policy output, pre-scale):
- raw range: min -1.047, max 1.047, mean 0.254, std 0.619
- per-step vector-norm: mean 1.902, max 2.034
- top-5 by mean|a|: **FL_thigh 0.90, RR_thigh 0.79, FR_calf 0.69, RL_hip 0.53, RL_calf 0.55**
  — asymmetric on axes that are supposed to be symmetric.

---

## What this really means — two independent fix axes

### Axis 1 — policy output magnitude (retrain)

Four saturating joints (FL_thigh, RR_thigh, FL_calf partially, RR_hip) are
driven by the policy emitting large actions even when q is at or near
default. Stand-v2 trained with `action_rate=-0.5` / `joint_acc=-1e-6` —
already 10x / 4x stronger than `base.yaml`. The on-stand OOD state
breaks through the smoothness prior anyway.

**Planned response:** stand-v3 fine-tune from stand-v2's
`model_499.pt` with `action_rate=-2.0` / `joint_acc=-5e-6`,
`init_noise_std=0.05`. Configs are in place:

- `configs/env/stand_v3.yaml`
- `configs/train/ppo_stand_v3.yaml`
- `scripts/train_stand_v3.sh`
- `scripts/export_stand_v3.sh`
- `configs/sim2real/deploy_stand_v3.yaml`

Expected outcome: FL_thigh / RR_thigh / FR_calf saturation falls
sharply. Overall probably lands at ~10–15% — still above Gate 7's
<5% bar, but unambiguous progress on axis 1.

### Axis 2 — stand-posture distribution mismatch (v4 or pivot)

RL_thigh's 0.37 rad gap between measured rest (0.630) and trained
default (1.0) cannot be closed by action regularization. It's a
sim/real scenario mismatch: training did "stand on ground, feet
loaded"; Gate 7 tests "stand on stand, feet dangling".

Two paths forward, both out-of-scope for v3:

- **v4 idea (proper sim fix):** add a `rel_floating_envs` fraction in
  base.yaml that suspends a subset of training episodes above the
  ground (no foot contacts). Policy learns to hold default when
  unloaded. Requires wiring the init_state path (currently unwired
  per base.yaml audit line 7). Day-sized project.
- **Floor-testing pivot:** run Gate 7 on the floor where feet contact
  the ground and sim matches reality. GO2 is safely tetherable at
  walking height via the harness in the lab. Out of scope per the
  04-14 deploy prompt's "MUST be on a stand or safely tethered for
  the first run" — but we've now had one clean on-stand live run, so
  this is a plausible next-session option if operator comfort allows.

### Axis 3 — stand asymmetry (mechanical, investigate)

Measured rear-thigh rest posture is **asymmetric**: RL=0.630,
RR=0.954 (a 0.32 rad difference). Fronts are also skewed: FL=0.773,
FR=0.664. The dog and stand are both supposed to be symmetric. Before
the next hardware run, check:
- Is the stand level? (floor, clamp heights, any lean?)
- Are foot-stops symmetric?
- Any cables pulling on one side?
- Motor calibration drift on RL_thigh / FL_thigh?

A 0.3 rad stand asymmetry will appear as a saturation floor on every
retrain in this fixture, independent of policy quality.

---

## Bugs observed

1. **Post-abort `_publish_default_pose` rebroadcast caused a Jetson
   brownout on the first launch attempt.** The policy's
   `_control_step` was publishing default_q to the bridge every tick
   while `self._estopped` was True. Bridge slewed target toward
   default_q at 0.175 rad/step with kp=25; on the real stand with
   rear thighs hanging at 0.63 rad vs target 1.0 rad, motors drew
   sustained current that tripped the Jetson's power path. Patched
   this session (one-liner) — the single on-abort publish still
   fires, which is bounded.

2. **cyclonedds env split between bridges started in different shells.**
   First launch had `estop_publisher.sh` without `CYCLONEDDS_URI`
   and the policy node with it; heartbeat callbacks landed but the
   policy's steady-state freshness check tripped anyway. Resolved
   pragmatically by launching all four processes from a single shell
   with a unified env block — but this is fragile and worth a wrapper
   script. The `dryrun_pipeline.sh` already has the right pattern;
   generalizing it to a `live_pipeline.sh` would close the gap.

3. **Policy process does not self-exit on max_runtime abort.** With
   the patch above in place, after `max_runtime` the policy just
   returns from `_control_step` forever, consuming CPU at idle. A
   follow-up should make `_latch_abort("max_runtime")` also call
   `rclpy.shutdown()` or set a stop flag. Operator `kill -INT`
   worked around it this session.

---

## Artifacts

On T7 (`/media/cares/T7 Storage/go2-phoenix/`):
- `data/failures/gate7_live_2026-04-21_18-33-17.parquet` — **primary
  evidence**, 5961 rows, 33.06% sat
- `data/failures/gate7_live_2026-04-21_18-19-45.parquet` — aborted
  first attempt, 512 rows, kept to prove the parquet-flush fix works
  on abort
- `docs/attachments/2026-04-21/rest_posture_extraction.json` —
  per-joint median rest posture from the primary parquet; input for
  the stand-v3 training env config
- `docs/attachments/2026-04-21/jetson_deploy_stand_v2.yaml` — Jetson
  copy with bumped timeouts (canonical stays in T7-side
  `configs/sim2real/deploy_stand_v2.yaml`)
- `src/phoenix/sim2real/ros2_policy_node.py` — patched (post-abort
  rebroadcast removed)
- `configs/env/stand_v3.yaml` + `configs/train/ppo_stand_v3.yaml` +
  `configs/sim2real/deploy_stand_v3.yaml` — new retrain kit
- `scripts/train_stand_v3.sh` + `scripts/export_stand_v3.sh` — run
  these on a GPU box (mewtwo or the 5080 lab PC)

On Jetson: same `audit-fixes-2026-04-16 @ 3c52b05` snapshot plus the
one-line patch. Stand-v2 checkpoints unchanged.

---

## Concrete next actions (supersede 04-20 list)

1. **Run `./scripts/train_stand_v3.sh` on a 5080/5070 box.** See the
   T7-root runbook `phoenix_next_session_2026-04-21.txt` for the lab
   PC kick-off procedure.
2. **Before the next hardware session, level-check the stand and
   inspect for mechanical asymmetry** (axis 3 above). A 0.32 rad L/R
   rest offset will defeat even a perfect policy.
3. **Plumb a `live_pipeline.sh`** mirroring `dryrun_pipeline.sh` so
   the unified-env launch isn't a shell-history one-off.
4. **Make `max_runtime` actually exit the policy process.**
5. **Decide: hardware retry on stand (v3) or pivot to floor-testing?**
   Floor is in-distribution for the sim, on stand is not. If v3
   lands sat at ~10%, floor may pass Gate 7 with room to spare while
   on-stand remains stuck.
