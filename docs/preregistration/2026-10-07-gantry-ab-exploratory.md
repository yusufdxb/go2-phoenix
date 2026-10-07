# Preregistration: EXPLORATORY gantry A/B, model_2750 vs friction-DR seed 50 (2026-10-07)

Written and committed before any hardware data of this campaign exist. EXPLORATORY: this is
not a promotion gate and cannot promote a policy. Its outputs are (1) whether arm B is
obviously better, obviously worse, or neither on the real GO2, (2) model-identification
evidence (calf sag, joint tracking) and (3) the design of a separate confirmatory campaign,
which must be preregistered before that campaign runs.

## Status of the inputs (unchanged, not reinterpreted)

- Friction-DR study `2026-10-07-joint-friction-dr.md`: hypothesis PASS in MuJoCo replay;
  promotion guard FAIL for every seed (`results/friction_dr/promotion_guard.json`,
  `2026-10-07-joint-friction-dr.OUTCOME.md`). Seed 50 is NOT promoted.
- Nothing in this file changes any rule or result of that study.

## Purpose

Test whether `fric_T_s50` (arm B) transfers to the real GO2 better than the incumbent
`model_2750` (arm A), under identical gantry conditions, at the smallest command amplitudes
already flown on hardware (2026-10-06/07: vx 0.3 m/s, wz 0.5 rad/s).

## Arms (exact artifacts; the deploy configs differ only in these four paths)

| Arm | Checkpoint (sha256) | policy.onnx | policy.onnx.data | Stand config / lock | Walk config / lock |
|---|---|---|---|---|---|
| A | `walk_vnext3c_seed47/2026-09-26_21-06-20/model_2750.pt` e03013d19ccc5df92a58762f8b76566a745c131534e8949ff1275739378198e4 | 6d36796f857a490b3d446492581aa1952fa9afdb689bef88b09410db10512c32 | b6ec50ed6e295aecc0bb3b6d801d3a089c890af491956b1f4f1358c25079797b | `configs/sim2real/deploy_walk_iter2750_stand.yaml` / `locks/deploy_walk_iter2750_stand.lock.yaml` | `configs/sim2real/proposed/deploy_walk_iter2750.yaml` / `locks/deploy_walk_iter2750.lock.yaml` |
| B | `fric_T_s50/2026-10-07_09-03-11/model_3749.pt` 9948900b4568cb09bd0119b58c395f8aa0309179cf5467ac43bae0cd4945ba9a | 4b7063b47bac0b36d4696258da613da08a14cebe57c05f026d88aca73872034f | 13946b354f0dc7cfb92951164f4e88ccfa01b4ed13ccaa6bce243f65f394e00a | `configs/sim2real/deploy_walk_fric_s50_3749_stand.yaml` / `locks/deploy_walk_fric_s50_3749_stand.lock.yaml` | `configs/sim2real/proposed/deploy_walk_fric_s50_3749.yaml` / `locks/deploy_walk_fric_s50_3749.lock.yaml` |

Archives (read-only, SHA256SUMS): A `~/Archive/phoenix-deploy-candidates-2026-10-03`,
B `~/Archive/phoenix-deploy-candidates-2026-10-07`. The exporter used for B reproduces A's
locked policy.onnx and policy.onnx.data byte for byte. Manifests differ only in
checkpoint hash, provenance and the DR description (no deploy-relevant field).
Identical for both arms: bridge, policy node, harness, gains (kp 25, kd 0.5), handover
(kd step + 1 s kp blend), stand-up 2 s, presets, gantry, rope, floor, operator, spotter.

## Setup (both arms, every run)

Gantry walk as in vault `LABCARD_2026-10_first-walk.md`: feet on the floor under the
gantry, safety rope attached to the trunk and the gantry, slack (carries no weight) but
short enough to catch a fall. Hard requirements carried over unchanged: gantry for every
motion stage from D on, D2 already measured (2026-10-06), floor mu >= 0.4 (incline test),
battery above 20 % (harness-enforced), no other stack on the robot (contention check).
"Ground walking" (any walking command without the gantry rope attached) is out of scope.

## Sequence

1. Desk: workstation stage A GO for BOTH stand locks at the campaign commit; payload bundles
   staged and derived for both arms (`scripts/gantry_ab_campaign.sh stage-check`).
2. Ladder, arm A then arm B, each in its OWN session directory
   (`logs/hw_sessions/<date>_<sha12>_abA` / `_abB`, so stage records never overwrite each
   other): A --payload, B (60 s dry), C (deadman), D (30 s), E (10 s hold), F (2 s stand),
   G (5 s stand). Walking for an arm is refused in code until that arm's F and G are GO
   for its exact artifacts. E2 for each arm on its ladder G (decision input below).
3. Blocks, fixed order **A B B A B A** (counterbalances drift in battery, temperature,
   rope and operator; each arm gets one early, one middle, one late block). Each block:
   `G` (5 s zero-command stand, stand config) then `W vx03_4s` then `W yaw05_3s` (walk
   config). 3 blocks per arm = 3 stands, 3 forward and 3 yaw runs per arm.
4. STOP drill, once per arm, in that arm's LAST block, after the yaw run: one `G` stand in
   which the operator releases L1 about 2 s into policy authority (announced, robot on
   the gantry). Not repeated.
5. No other preset, no vx 0.5, no backward or lateral command, no repeated attempt of a
   preset beyond the one allowed repeat below.

Commands (`src/phoenix/sim2real/walk_command.py`): vx03_4s = vx 0.3 m/s for 4 s (about
1.2 m), yaw05_3s = wz +0.5 rad/s (CCW) for 3 s (about 86 deg); each with 2.0 s zero lead-in,
ramps at 0.5 m/s^2 / 1.0 rad/s^2, 3.0 s zero tail; policy window ends with the bridge sitting
the robot. Warmup/reset: robot starts every run folded on the floor under the gantry, the
bridge stands it up (2 s), the policy takes over; after the sit the operator re-centres the
robot on the start mark by hand (motors at the sit pose), checks the rope, then the next
run. At least 60 s between runs (motor temperature), identical for both arms.

## Validity (decided by `scripts/gantry_ab_analyze.py` from raw logs, no manual exclusion)

A run is VALID iff: the bridge manifest's policy.onnx / policy.onnx.data / checkpoint hashes
equal the planned arm's; `live` true and `code_identity_problems == []`; exactly one
contiguous policy episode; LowState fresh on every policy tick; no fault other than the
normal end latch (`policy_abort:authority_window_complete`, `estop_asserted`,
`bridge_shutdown`); for W: `walk_cmd` summary outcome `complete`; the operator record says
the rope never carried weight and nobody touched the robot; for vx03_4s: a tape reading
exists. For G: the E2 settled window passes (`actuator_check.settle_window`).
An INVALID or aborted run may be repeated ONCE, immediately, in the same block; every
attempt is logged and reported; the first valid attempt is used. Two invalid attempts =
missing slot.

## Abort (operator releases L1) and campaign STOP

Abort a run: foot slip, stumble, sideways drift > 0.3 m, oscillation or buzz, turn in the
wrong direction, the rope taking weight, tilt visibly > 15 deg, battery at 20 %.
STOP the whole campaign: any fall fault or any bridge fault other than the normal end
latch; a node crash, Jetson reboot or brownout; a joint limit clip under the policy;
two aborted runs of the same arm and preset; battery below 30 % at the start of a block
(remaining blocks not run); any LabLink identity mismatch. If one arm alone triggers a
STOP condition, that arm is terminated and the other arm's remaining blocks may still run.
Never "retry until pass": a failed physical safety check is recorded as observed.

## Data that must be logged (per run, existing pipeline unless stated)

`bridge.jsonl` (manifest with artifact hashes and code identity; per tick: mode, faults,
q, dq, requested and final target, kp/kd effective, slew_clip, limit_clip, limit_margin,
torque_margin_nm, tilt, lowstate/cmd/imu/estop ages, estop state, counters),
`policy.parquet` (timestamps, IMU quaternion, gyro, joint pos/vel, command, action, odom
validity), `walk_cmd.jsonl`, operator answers, battery and contention records, and a
campaign record `logs/gantry_ab/<date>_<sha12>/campaign.jsonl` written by
`scripts/gantry_ab_campaign.sh` (block, planned arm, run dir, battery, tape forward and
lateral in metres, rope/contact flags, free-text note). Motor tau_est: GO2 LowState carries
it, but the current bridge and policy logs do not record it. It is NOT collected in this
campaign unless `PHOENIX_AB_EFFORT=1` is set for the WHOLE campaign before block 1
(passive /joint_states bag); it is never collected for one arm only.

## Metrics (all computed by `scripts/gantry_ab_analyze.py`)

Primary (per W run):
- Yaw tracking `Y` = IMU yaw change from the last zero-command policy tick before motion to
  1.0 s after the command returns to zero, divided by the commanded yaw (integral of the fed
  wz over policy ticks). Gyro-integral yaw reported as a cross-check.
- Forward tracking `F` = tape forward displacement (start mark to end mark along the initial
  heading, robot folded before and sat after the run) divided by the commanded distance
  (integral of the fed vx). Odometry only if `odom_valid` (reported, never substituted).

Secondary (exploratory, reported for every valid run):
- yaw-rate overshoot = max of the 0.1 s moving-average gyro wz during the motion segment
  divided by the commanded wz; settling time = time after the command reaches zero until
  |gyro wz| < 0.1 rad/s for 0.5 s; yaw drift during vx runs; tape lateral drift.
- joint tracking during motion: RMS(final target - q) per joint group; swing ratio
  (p95-p5 of q over p95-p5 of target), the same definition as the MuJoCo replay.
- zero-command posture (G stands): max tilt, per-joint dq RMS, calf sag (below).
- oscillation: per-joint dq power fraction in 4-15 Hz over the G window and W tail.
- saturation: slew_clip and limit_clip tick counts under policy, min torque margin.
- timing: bridge tick interval p50/p99/max and count > 10 ms; policy step interval p99/max
  and count > 30 ms; max lowstate/cmd/imu age.
- STOP: latency from the first tick with the deadman released (estop asserted) to the first
  non-policy tick (STOP drill and any abort); latency from the end of the policy window to
  the first sit tick (every run).

Calf sag (model identification). Robot 0.135 (E2 `calf_err_hw`, 2026-10-06 G/H, robot on
the floor without gantry) and sim 0.182 (E2 `ref_nominal`, MuJoCo) are NOT the same
measurement: windows differ (1 s after the gain blend, 3-8 s, pooled G/H vs t >= 2 s of a
10 s sim stand), hardware uses the post-safety target, the sim starts from the trained pose
while the robot starts folded, and robot mass and support were not controlled. This
campaign measures, per G stand, `S = mean over the 4 calves of the median over the E2
settled window of (final target - q)` (rad, positive = sag) with the SAME function as the
sim reference (`actuator_check.summarize_hw` vs `summarize_trace`), the rope slack during
the stand, plus the mean-instead-of-median variant used by the desk replay. Sim references
per arm: A 0.182, B 0.176 (MuJoCo nominal; boundaries A 0.239/0.255, B 0.231/0.245).
Reported per arm as hardware minus sim; no gate is attached to it here.

## Decision rule (exploratory; fixed now; computed by the analysis script)

Needs >= 2 valid runs per arm and preset, else INSUFFICIENT. Per preset, over valid runs:
- **B obviously better**: on BOTH presets, every B run exceeds every A run (min B > max A)
  and median(B) - median(A) >= 0.15; AND B had no safety event (abort, fault, limit clip
  under the policy, fall) that A did not; AND every B yaw-rate overshoot <= 1.3; AND B's
  G stands are not worse (max tilt <= A max + 2 deg, max calf dq RMS <= 1.5 x A max).
  Consequence: write and commit a CONFIRMATORY preregistration (more blocks, the
  promotion guard defined before data, held-out presets) before any further B campaign.
- **B obviously worse**: on EITHER preset, max B < min A and median(B) - median(A) <= -0.15;
  OR B had a safety event that A did not. Consequence: stop B; record.
- Otherwise **INCONCLUSIVE**: no claim; use the model-identification data to choose the
  next experiment.
With 3 runs per arm, complete separation happens by chance with probability 1/20 per preset
when the arms are equal. This is a screening rule, not a significance test.
Model identification questions (descriptive): hardware Y/F per arm vs that arm's MuJoCo
replay at measured friction x1.0 / x1.25 / x1.5; hardware calf sag per arm vs sim; whether
the hardware-to-sim ratio is the same for both arms (if both fall short by the same factor,
the simulator is missing drivetrain or contact behaviour that friction DR did not cover).

## What this campaign may NOT do

Promote a policy; define a new promotion rule from these results; run on the ground without
the gantry; run any preset not listed; change the friction-DR preregistration.

## Amendment 1 (2026-10-07, before any campaign data; desk rehearsal finding)

A localhost rehearsal (fake GO2) showed that the bridge manifest's `live` flag is also
true in rehearsal, so "`live` true" alone does not identify a hardware run. Added to the
validity rule: the bridge manifest's `code_identity.source` must be `payload_sync` with
`manifest_verified` true (every one of the 14 hardware runs of 2026-10-06/07 has this; the
rehearsal has `git`), the run's code sha must equal the campaign commit, and the runner
records `rehearsal` on every run record (any rehearsal record is invalid). No metric,
arm, order, threshold or decision rule changed.

## Amendment 2 (2026-10-07, before any campaign data; independent pre-hardware audit)

Three analysis gaps closed; no metric, arm, order, threshold or decision rule changed.
1. Attempts are counted physically: the attempt index of a run is its position among ALL
   records of its (block, slot) in the campaign file, and any physical attempt beyond the
   second is invalid. The runner never re-runs a block that was started; an interrupted
   block is closed with `gantry_ab_campaign.sh abandon <k> <reason>` and its missing slots
   stay missing.
2. The STOP drill is not exempt from fault checks: any fault other than the normal end
   latch during the drill makes it invalid and counts as a safety event of that arm (only
   the operator's deliberate deadman release is expected).
3. Abnormal bridge faults or limit clips in live ladder stages (verified payload runs)
   count as safety events of the arm whose artifacts ran. Campaign STOP conditions that are
   computable from the logs (faults, limit clips, two aborted runs of one arm and preset)
   are listed in the analysis summary.
The 60 s rest between runs and the gantry rope remain operator responsibilities (not
sensed); calling `scripts/harness_preflight.sh W` directly during the campaign is a
protocol violation (the runner restricts presets; the harness alone does not).

## Amendment 3 (2026-10-07, before any campaign data; audit follow-up)

A deadman release is logged as `policy_abort:external_estop` (policy node latch, recorded by
the actuator gate); Amendment 2 would have counted every STOP drill as a fault. Now:
`policy_abort:external_estop` is part of the normal end latch, and WHEN the deadman was
released is judged from the ticks (estop asserted on a policy-mode tick). A STOP drill is
valid only if that release happened during policy authority; in a G or W run such a release
is an aborted run (invalid, safety event) even if the operator did not say so. Verified on
the real 2026-10-06/07 hardware logs: every G/H/W run stays valid with unchanged metrics; the
two FP runs carrying `external_estop` released after the policy window (not flagged). No
metric, arm, order, threshold or decision rule changed.

## Correction to Amendment 3 (2026-10-07, before any campaign data)

Amendment 3's sentence "the two FP runs carrying `external_estop` released after the policy
window" is FALSE. Both FP runs (`20261007_fa58780dd4f8/FP_20261007T005825Z` ticks 4769-4770,
`20261007_04f3d46edbfe/FP_20261007T011645Z` ticks 16226-16227) were deadman releases DURING
policy authority: the last policy tick has estop false and the next tick is already `hold`
with estop true and `hold_cause policy_abort:external_estop` (3.2-3.3 ms later). The bridge
never logs estop true on a policy-mode tick, so Amendment 3's release-timing rule cannot fire
on real data: every real STOP drill would be marked invalid and an un-reported abort in a G
stand would not be detected from the ticks. STATUS: the analyzer's STOP-drill / abort
detection is KNOWN DEFECTIVE and the campaign must not start until it is fixed under a new
amendment, with a test built from this real tick sequence. Two fixes to this mechanism
(Amendment 2 item 2, Amendment 3) failed; the next fix waits for explicit instruction.

## Amendment 4 (2026-10-07, before any campaign data; analysis correction only)

Scope: a measurement/analysis correction of deadman-release detection. It does NOT change
the hypotheses, arms or policy identities, commands or presets, safety limits, decision
thresholds, number of runs or blocks, or promotion status. Seed 50 remains exploratory and
not promoted. Amendments 1-3 and the Correction to Amendment 3 stand as written.

Event semantics, established from all 16 real bridge logs of 2026-10-06/07 before this
amendment: the bridge never logs estop true on a policy-mode tick; the tick counter never
skips; every planned end shows, on the FIRST non-policy tick, `hold` with `hold_cause
policy_abort:authority_window_complete` and estop false (11 runs; policy episode within
17 ms of the authority window); both real deadman releases under policy show, on that same
first non-policy tick, `hold` with `hold_cause policy_abort:external_estop`, estop true and
`estop_age_s` 0.36-0.38 ms. A release can therefore only be observed on the transition out
of policy control.

Correction. Deadman-release detection considers the transition from the final
policy-controlled tick to the first subsequent recorded tick, because the release event may
only become observable after policy mode has ended. Per run, from `bridge.jsonl` only:
- `t_policy_last`: `t_mono_ns` of the last tick of the single policy episode.
- `t_release_obs`: `t_mono_ns` of the first subsequent tick; it is a release iff that tick has
  estop true and `hold_cause` in {`policy_abort:external_estop`, `estop_asserted`}. A planned
  end iff estop false and `hold_cause policy_abort:authority_window_complete` with the policy
  episode within 0.1 s of the authority window the harness passed (`--authority-s`: G and
  STOP drill 5 s, W profile duration + 4.0 s; ladder F 2 s, G 5 s). Any other cause is a
  fault exit. A release must also lie inside the authority window, and the bridge's receipt
  time of the release message `t_release_rx = t_release_obs - estop_age_s` must satisfy
  `t_policy_last < t_release_rx <= t_release_obs`.
- `t_zero`: the first tick from `t_release_obs` on whose published command demonstrably moves
  nothing: mode `hold`, no requested (policy) target, the published target equal to the
  measured q at hold entry (or limit-clipped), and the same target published on the next
  tick, also in `hold`. (A constant target alone is not evidence: policy targets also repeat
  between 50 Hz policy steps.)
- STOP latency = `t_zero - t_release_obs`. The origin is unchanged from the registered metric
  (the first tick with the deadman released); the end point additionally requires the
  zero-command evidence above (with the current bridge the first non-policy tick is the hold
  entry, so both coincide). Also reported, not gated: `t_release_obs - t_policy_last` (bound
  on when the release reached the bridge) and `t_zero - t_release_rx`. The physical L1-to-bridge
  delay is not observable in bridge logs and is not claimed.
Classification: G and W runs must end as a planned end; a release (or a release with no
policy episode) is an aborted run from the logs, invalid and a safety event of that arm,
whatever the operator answered; operator/log disagreement is reported as a discrepancy and
an operator-reported abort is never discarded. A STOP drill is valid only as a release with
`t_zero` found; a release without `t_zero` fails the drill and is a safety event. Live ladder
F/G runs whose policy episode does not end as a planned end are ladder safety events
(Amendment 2 item 3). Fail closed: missing estop field or first post-policy tick,
non-increasing timestamps, a skipped tick counter, estop not false on a policy tick, an
estop/cause combination other than the two above, a planned end outside the window, or
`t_release_rx` outside its bound make the run INVALID and, for G/W/STOP drill, a safety
event; nothing is estimated or interpolated. The runner prints the same per-run log
classification after each run and treats a non-clean classification like an operator-reported
abort (the one allowed repeat); validity is still decided only by the analysis script.
Regression test: `tests/test_gantry_ab.py`, built from the real tick sequence of
`20261007_fa58780dd4f8/FP_20261007T005825Z` ticks 4765-4774 (bridge.jsonl sha256
80befcf1d6ef164520a2d273584c39d5b99f0a79cd70f639799da5effbcd549c).
