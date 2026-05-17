# LAB CARD NEXT: Gate 7 + Gate 8 + Block 3 (script-driven)

Supersedes `docs/LAB_CARD_2026-04-22.md` for the next CaresLab session. Same
block structure, same gates. Difference: every preflight / logging / post-block
check is a single command. The operator runs scripts; the lab card is for
decision making, not finger-on-keyboard plumbing.

Print this. Paper beats a tablet when the robot is moving.

## Goals (priority order)

1. **Gate 7 on floor**: stand-v3 at cmd=0, <5% slew sat
2. **Gate 8 mode-switch**: STAND -> WALK -> STAND clean
3. **Block 3 parquet collection**: 5 to 10 parquets, >=4 distinct failure modes

Anything unachieved gets logged, not retried in session. Time > completeness.

## Safety (non-negotiable, manual)

- [ ] Fall mat under robot at all times
- [ ] Tether rigged for Blocks 2 and 3; optional for Block 1
- [ ] Gamepad deadman in operator's hand
- [ ] `/phoenix/estop` latch verified BEFORE first live command
- [ ] Clear 2 m radius
- [ ] `max_runtime_s: 120` in deploy yaml

## Topic capture rationale

`scripts/harness_record.sh` records the following topics (skip-if-absent):

| Topic | Why |
|---|---|
| `/lowstate` | raw firmware state, full fidelity ground truth |
| `/lowcmd` | live motor cmd (lowcmd_bridge in `--live` mode) |
| `/lowcmd_dry` | dry-run motor cmd (lowcmd_bridge default) |
| `/joint_states` | `lowstate_bridge_node` publish (see src/phoenix/sim2real/lowstate_bridge_node.py L61) |
| `/imu/data` | `lowstate_bridge_node` publish |
| `/cmd_vel` | operator command (Twist), subscribed by policy_node |
| `/phoenix/estop` | deadman latch, std_msgs/Bool |
| `/joint_group_position_controller/command` | policy_node publish (see ros2_policy_node.py L254) |

`/phoenix/policy_state` is referenced in plans but is NOT currently published
by `ros2_policy_node`. Reserved for future state-machine telemetry; recorder
silently skips it if absent.

---

## Preflight (60 min, one command)

```
bash scripts/harness_preflight.sh
```

This wraps:

| Step | What the script does | Halt on fail |
|---|---|---|
| P1 | `rsync --checksum` T7 -> Jetson for `checkpoints/phoenix-stand-v3/` + `configs/sim2real/deploy_stand_v3.yaml`, md5 verify policy.onnx + latest.pt + policy.pt | yes |
| P2 | `git fetch origin $PHOENIX_BRANCH && git merge --ff-only` (default branch: `main`; override via env) | yes |
| P3 | `pytest tests/ -x -q -m "not sim and not ros"` | yes |
| P4 | `python -m phoenix.sim2real.verify_deploy --parquet <ref> --deploy-cfg <cfg> --tol 1e-4` | yes |
| P5 | `bash scripts/dryrun_pipeline.sh 25` (brings up three bridges), parses `/joint_states hz` from samples, requires >= 200 Hz | yes |
| P6 | manual prompt: press-latch E-stop, confirm `/phoenix/estop` holds; release; confirm fresh-and-False seen | yes |
| P7 | manual prompt: robot on mat, low sit, feet unloaded | yes |

Logs land in `/tmp/harness_preflight_<ts>/`.

### Steps NOT automated (still manual checkboxes)

- E-stop physical press / release (P6): requires a human hand on the wireless E-stop. The script prompts and waits for explicit `y` confirmation.
- Physical robot pose (P7): same; the script can't see the robot. Prompted.
- Tether rigging, fall mat placement, clear-radius check.

Self-test on mewtwo: `bash scripts/harness_preflight.sh --mock`

---

## Block 1: Gate 7 on floor (45 min)

**Config:** `deploy_stand_v3.yaml`, `policy.mode_switch.enabled: false`.

**Per hold:**

```
bash scripts/harness_record.sh gate7 hold_$N 15
```

(writes to `data/lab/<date>/gate7_hold_$N_<ts>/`)

Then in another terminal: bring up policy_node, enable motors, robot stands for 10 s, abort. Repeat 3x.

**Decision gate** (read `slew_saturation_pct` from the post-shutdown parquet):

| sat | call |
|---|---|
| <5% | Gate 7 CLEARED. Tag `v0.4.0-gate7` post-session. |
| 5 to 15% | LAND. Proceed to Block 2. |
| >15% | ABORT Block 1. Fall back to stand-v2 for Block 2. |

What NOT to do: retrain, swap configs, chase the number. 45 min hard cap.

---

## Block 2: Gate 8 mode-switch (90 min)

**Config:** same yaml, `policy.mode_switch.enabled: true`; stand-v3 at cmd=0, v3b at cmd!=0, hysteresis `[0.05, 0.15]`, 25-tick blend.

**v3b paths:** `checkpoints/phoenix-flat/2026-04-16_21-39-16/model_999.pt`; ONNX at `checkpoints/phoenix-flat/v3b/policy.onnx` md5 `674ea7ca0907ce9877f518413f582f69` (verify on Jetson via `md5sum`; the preflight P1 already did this if v3b is under the synced ckpt dir).

**Dryrun (no motor enable):**

```
bash scripts/harness_record.sh gate8 dryrun_step 30
```

In another terminal publish `cmd_vel`: 0 -> 0.3 -> 0. Watch logs for STAND -> TRANS_TO_WALK -> WALK -> TRANS_TO_STAND -> STAND.

If states never fire or blend misbehaves: FALLBACK to v3b single-policy. Document and move on. 20 min cap on dryrun + decide.

**Live G8a (step cmd, 30 min):**

```
bash scripts/harness_record.sh gate8 g8a_step_$N 30
```

(3 reps; step 0 -> hold 5s -> 0.3 m/s -> hold 5s -> 0 -> hold 5s)

**Live G8d (lin+yaw combined, 30 min):**

```
bash scripts/harness_record.sh gate8 g8d_combined_$N 30
```

(3 reps; `(vx=0.2, vyaw=0.5)`)

**Decision gate:**

| Behavior | Call |
|---|---|
| mode-switch fires, no falls in 6 reps | Gate 8 CLEARED. Tag `v0.5.0-gate8`. |
| fires but 1+ falls | LAND as partial. |
| does not fire | Fall back to v3b single-policy. |

---

## Block 3: Parquet collection for loop closure (2 to 3 hr)

**Goal:** 5 to 10 parquets, >=4 distinct failure modes. Diversity > quantity.

### Failure mode menu (pick 4+)

| Mode | Perturbation | Detector label |
|---|---|---|
| slip | towel / plastic under one foot | SLIP |
| push_lat | lateral shove | ATTITUDE or recovery |
| ramp_dn | 10 deg ramp, walk down | ATTITUDE (pitch) |
| ramp_up | 10 deg ramp, walk up | ATTITUDE (pitch back) |
| step_up | 3 cm book / block | contact transient |
| yaw_shove | shove hindquarters | yaw / ATTITUDE |
| stop_sudden | cmd step 0.5 -> 0 instant | stumble |
| deadman_freeze | hold deadman mid-walk | freeze + recover |

### Per scenario

```
bash scripts/harness_record.sh block3 pqa_<mode>_$N 30
```

Then in policy terminal: robot walking at `(vx=0.3, vyaw=0)`. Apply perturbation. Do NOT catch the robot. Abort after failure or 10 s post-perturbation.

Rename or copy the produced parquet into `data/failures/` with the canonical name `pqa_<mode>_<ts>.parquet` so the diversity validator picks it up. (The recorder writes a rosbag2 store + camera + manifest; the policy_node writes the parquet via its `--log-parquet` flag, same as in `dryrun_pipeline.sh`.)

### Diversity check (do this BEFORE leaving the lab)

```
python3 scripts/harness_diversity.py data/failures/
```

Prints a coverage matrix and returns non-zero if any of:
- fewer than 4 distinct mode prefixes
- any mode with < 2 good replicates
- any parquet with rows <= 200
- any parquet with zero detector events

If it fails, extend Block 3. Do not pack up with 8 slip parquets and call it a day.

Self-test: `python3 scripts/harness_diversity.py --self-test` (writes synthetic parquets to a tmp dir, exits 0).

---

## EOD (one command)

```
bash scripts/harness_eod.sh
```

Rsyncs Jetson -> T7 (with `--checksum`) for `data/failures/*.parquet` and `data/lab/<today>/`, then re-md5s every file on both sides. Exits non-zero on any mismatch. Prints a per-file summary.

Self-test: `bash scripts/harness_eod.sh --mock` (writes to `/tmp/t7_mock` instead of the T7 mount).

### Manual EOD items (not scripted)

- [ ] T7 -> mewtwo rsync after you get home
- [ ] One-line per-gate summary in `docs/lab_findings_<date>.md`
- [ ] Push gate tags if earned
- [ ] Robot in sit, motors off, mcf mode
- [ ] Pack mat + tether

---

## Rollback matrix

| If this breaks | Do this |
|---|---|
| harness_preflight P1 md5 mismatch | re-rsync, check T7 source for corruption |
| harness_preflight P3 pytest fail | do NOT proceed; investigate or revert HEAD |
| harness_preflight P4 parity fail | re-export ONNX on Jetson, rerun P4; if still fails use stand-v2 ONNX |
| harness_preflight P5 `/joint_states` hz < 200 | restart bridges (cyclonedds discovery race), rerun preflight |
| harness_record can't see topics | three bridges not up; run `dryrun_pipeline.sh` manually first |
| harness_diversity FAIL | extend Block 3, do not pack |
| harness_eod checksum mismatch | re-rsync the specific path, re-verify; do not power-cycle T7 mid-transfer |

---

## Operator command sequence at the lab

1. `bash scripts/harness_preflight.sh`
2. For each Gate 7 hold (x3): `bash scripts/harness_record.sh gate7 hold_$N 15` + bring up policy_node in another terminal.
3. For Gate 8: `bash scripts/harness_record.sh gate8 dryrun_step 30`, then `..._g8a_step_$N`, then `..._g8d_combined_$N`.
4. For each Block 3 scenario: `bash scripts/harness_record.sh block3 pqa_<mode>_$N 30`.
5. `python3 scripts/harness_diversity.py data/failures/`
6. `bash scripts/harness_eod.sh`

Done. Robot back to sit, motors off, mcf mode, pack up.
