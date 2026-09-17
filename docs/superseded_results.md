# Superseded and invalid results

This record exists so that no earlier result in this repository is read as if it
still stands. Nothing under `data/`, `reliability_eval/`, `logs/` or
`checkpoints/` was modified. Recorded evidence stays exactly as it was produced;
this document says which of it no longer supports the conclusion it was used for.

## Branch and evidence integrity

**Corrected source of truth: `feat/causal-viability-replication`.**

At the time of writing it is 77 commits ahead of `origin/main` and 8 behind.
Those 8 were inspected individually rather than merged.

| what `origin/main` has that this branch does not | disposition |
|---|---|
| Public-hygiene scrubs replacing an exact GPU model and a hostname in docs, configs and one comment | **Already satisfied independently.** A `git grep` for both strings across this branch returns 0 tracked files, so nothing is lost by not merging. |
| Deletion of `scripts/loop_closure.sh`, `dry_run_policy.py`, `harness_eod.sh`, `cuda_128_smoke.sh`, `lowcmd_inspect.py`, and committed media blobs | **Deliberately not taken.** `loop_closure.sh` is being repaired here under item 9 of this pass; merging `main` would delete the file this pass is fixing. |

**Do not merge `origin/main` into this branch blindly.** The merge is a
destructive delete of work in progress, not a fast-forward. When this line
eventually reaches `main`, the script deletions must be reconciled as an explicit
decision about whether the repaired orchestration ships publicly, not resolved by
a merge driver.

No commit in this pass rewrites published history.

## Conclusions that are superseded or invalid

### 1. Every rough-versus-slippery contrast

The `terrain:` block was declared by `rough.yaml`, `slippery.yaml` and
`flat.yaml` and was discarded by `build_env_cfg` with no warning. Terrain comes
entirely from `env.task_name`. Both configs therefore built the SAME upstream
terrain and differed only in `domain_randomization.friction_range`.

Affected: the failure-fraction sweep, the multi-seed pilot, the n=7 and n=11
seed-scaling results, and the mode-subset ablation, in every place they are
described as a terrain contrast. The *friction* contrast they actually ran is
real and the recorded numbers are unchanged; the label on the axis was wrong.

The n=11 null was already known to be uninformative for an unrelated reason: the
curriculum seeded row 0 of each trajectory, a nominal gait state, so the
treatment was never delivered. It now has two independent reasons to be
uninformative, and remains not evidence against failure curricula.

### 2. Every simulator slew-saturation percentage

`phoenix.training.slew.slew_saturation_rate` compared successive RAW POLICY
ACTION deltas against 0.175. Deployment builds
`target = default_q + action_scale * action` and clips that target against
MEASURED joint position. Those are different quantities.

Fixed. `slew_saturation_rate` no longer exists. `slew_clip_activation_rate`
replaces it and calls the shared deploy helper
`phoenix.sim2real.safety.per_step_clip_array`, so the metric and the limiter
cannot drift apart. The old definition survives under the unambiguous name
`legacy_raw_action_delta_saturation_rate`, purely to reproduce the figures
below; `phoenix.training.evaluate` reports it as `legacy_raw_action_delta_pct`
alongside the corrected `slew_saturation_pct`, and stamps
`slew_metric_definition: "deploy_clip_activation_v2"` into every new metrics
JSON. A metrics JSON with no `slew_metric_definition` field is a legacy-metric
file.

`docs/sweep_design_2026-05-17.md` named `slew_saturation_rate` as the source of
the gate metric (it now names the corrected one), so every sim slew figure
recorded in this repository before this pass, including the
0.33%, 3.30%, 2.91%, 3.65% and 4.23% numbers, is a LEGACY-definition figure. The
recorded values are not edited. They must not be compared against a hardware
slew number or against the corrected metric without restating both.

**Open question, and it matters.** How the 2026-04-21 hardware figure of 33.06%
was derived is NOT determinable from this repository: `docs/lab_findings_2026-04-21.md`
is not present on this branch. If the hardware number counted actual clip
activations, which is what the deploy path physically does, then the headline
0.33% to 33% sim-to-real gap was partly a comparison between two different
metrics rather than entirely a physics gap. This is not asserted here, because it
is not verified. It is the single highest-value thing to resolve before that gap
is cited again. Resolving it requires the original lab-findings artifact or a
re-derivation from the Gate 7 parquet.

**Resolved 2026-09-12, by re-derivation from the Gate 7 parquet.** With
`scripts/slew_layer_audit.py`, `data/failures/gate7_live_2026-04-21_18-33-17.parquet`
(5,961 rows at 49.7 Hz, stand-v2 policy) gives a deploy-equivalent clip activation of
33.05% of joint-samples with the pose that run actually used (every hip 0.0, section 3)
and 33.06% with the training pose, against a LEGACY raw-action-delta figure of 0.00%.
The recorded 33.06% is therefore a clip-activation number, and the headline "0.33% in
sim against 33% on hardware" compared a legacy-metric sim figure with a clip-activation
hardware figure. How much physics gap remains is unmeasured until a corrected-metric
sim figure exists for the same policy and pose. The other April captures agree in kind:
`gate7_live_2026-04-21_18-19-45` 16.81% and `stand_v2_dryrun_2026-04-20_18-02-42`
16.67% clip activation, both 0.00% legacy.

### 3. The Gate 7 root-cause attribution

The 2026-04-21 run's slew blowup was attributed solely to the per-step rate
limiter existing at deploy but not in training. Two further mismatches were live
in that same run:

* `deploy_stand_v2.yaml`, the config that run used, set all four hip joints to
  0.0 while training used +0.1 left and -0.1 right. Because the action term uses
  `use_default_offset=True`, that value is simultaneously the joint-position
  observation reference and the action offset.
* The policy node fed `base_lin_vel` to the policy as zeros, although it is a
  trained observation term.

The single-cause attribution is superseded. Which of the three dominates is now a
measurement rather than an argument: `scripts/deploy_ablation.py` runs all eight
combinations against the same unmodified checkpoint.

### 4. H0 gate v1 verdicts on rollout-harvested data

`scripts/h0_delivery_probe.py` was re-anchored during a robustness pass onto each
trajectory's own prior history. That is a better estimator of a different
question. It asks whether the seeded state is ALREADY EXHIBITING the failure,
where the preregistered H0 asks whether it differs from an ordinary reset.

Consequence, measured: on harvested rollout failures a collapse case's z score
grew from +2.55 to +3.95 as the seed row moved earlier, while its direction
inverted, because earlier in time the robot was genuinely standing higher.
Seeding earlier finds a healthier robot, which is the entire point of pre-onset
seeding, and v1 scored it as a failure.

**Superseded: the v1 FAIL verdicts on harvested rollout failures.** They are not
evidence that rollout-harvested seeds are unusable.

**Still standing: the v1 verdict on the synthetic pool.** There the seeded state
was a random healthy walking row drawn from the same distribution as row 0 of the
same trajectory, unrelated to the failure, with an excursion of about -1% and
direction at chance. That is a different situation and the conclusion holds.

The replacement is `phoenix.reliability.hazard_gate`, protocol `h0-gate-v2`,
which separates delivery, precursor validity and recoverability. It is a new
versioned gate. No v1 code or v1 recorded evidence was modified.

### 5. The first harvest yield figures

The initial harvest reported 3 kept from 148 terminations. That run used detector
success as a DATA INCLUSION RULE, so 74 genuine simulator terminations were
discarded for being unlabelled. Those counts describe the old inclusion rule, not
the failure population. The simulator termination is ground truth and the
detector is now measured against it.

### 6. The harvested sim-failure pool, and every gate result seeded from it

`snapshot_manager_state` removed only the env origin's Z while `restore_state`
added the full XYZ, so store and restore were not inverses. The three files in
`data/failures/sim_harvest/` were captured through that broken path. Their
recorded `base_pos` x/y are world-grid coordinates, measured at `[6.603, -8.670]`,
`[-8.344, -4.417]` and `[1.353, 1.134]`, so a seeded robot would be placed up to
about 10 m from its own tile.

**Superseded: the pool itself, and the H0 verdicts and offset sweep computed from
it earlier on 2026-09-11.** Those were the FAIL_NOT_DISTINCT and FAIL_DIRECTION
results and the 0.0 to 2.0 s sweep.

They are re-harvested rather than relabelled. Relabelling would require assuming
the flat grid cloner sets origin z = 0, and nobody measured that. The data files
are left on disk untouched.

Note the synthetic pool was never affected: `synthesize_failure.py` already
subtracted the full XYZ, so the two producers now agree. The conclusion about the
synthetic pool in section 4 is unchanged by this.

### 7. The re-harvest's termination counts, and three of its windows' contents

`data/failures/sim_harvest/harvest_report.json` (schema 1.0, commit 5783416,
sha256 `46aec003...8d09c`) counts 148 genuine terminations. It is 74 physical
falls, each recorded twice: the real window at step s, then a one-row window on
the same environment at step s + 1 with the same `base_contact` term. All 74
one-row records follow a long record in exactly that way.

**Mechanism, measured.** `TerminationManager.compute` rebuilds its buffers every
step and `ContactSensor.reset` zeroes the contact history, so neither survives a
reset by itself (read in the Isaac Lab source). But the contact sensor fills its
history lazily, on the first `.data` read after it is marked outdated, from
whatever PhysX last simulated, and `_reset_idx` marks a reset environment
outdated without stepping physics. The harvest's post-step snapshot read contact
data in exactly that gap, writing the pre-reset base contact force into the new
episode's history, and `illegal_contact`, a max over that history, fired again
one step later. `scripts/diag_post_reset_termination.py`, recorded in
`data/failures/sim_harvest/diagnostics/post_reset_termination_2026-09-12.json`
(32 environments, 400 steps per phase, random actions plus the harvest's interval
push, one process):

| phase | terminations | on the first step after the env's own reset |
|---|---|---|
| read contact data after every step, as the harvest did | 479 | 239, all episode length 1, base height at least 0.396 m, stale force in history slot 1 |
| never read it after a step | 247 | 0 caused in-phase (its one length-1 termination, at its first step, follows a reset on the last step of the previous phase) |
| read it, then re-reset the sensor for envs reset that step | 222 | 0 |

**A second defect in the same windows.** The harvest cleared a window only when
`terminated` was set, and a time-out reset has `terminated=False`. With 20 s
episodes (1000 control steps) and 600-row windows, 49 of the 74 physical windows
span a reconstructed time-out reset. Checked on the one written window predicted
to show it: `sim_fall_0001_env058_step001102` jumps from a fallen z of 0.183 m to
a spawn height of 0.398 m, and 1.05 m in x/y, between rows 496 and 497, which is
step 999. Its detector onset, row 239, is a fall from the previous episode.

**Corrected accounting**, recomputed from the 1.0 report through the same
classifier the live loop now uses, in
`data/failures/sim_harvest/harvest_report.v2_recomputed.json` (schema 2.0):

| quantity | 1.0 as recorded | 2.0 recomputed |
|---|---|---|
| termination ticks | 148, reported as terminations | 148 |
| post-reset artifacts | not separated | 74 |
| physical terminations | 148, reported as genuine failures | 74 |
| written | 3 | 3, of which 1 spans a time-out reset |
| rejected, no usable pre-onset window | 71 | 71 |
| rejected, window under two rows | 74 | 0: these were the artifacts |
| detector recall | 1.000, 74 of 74 | 1.000, 74 of 74; over the 25 unspliced windows, 25 of 25 |
| detector mean lead | 10.178 s | 10.178 s over all; 6.974 s over the 25 unspliced windows |

**Superseded:** 148 as a failure population; the explanation that 74 terminations
had a window under two rows because most falls happen close to an episode reset;
the 10.178 s mean lead and the per-mode split, since 49 of the windows behind them
are spliced; and `sim_fall_0001_env058_step001102` as a single-episode failure
trajectory, together with anything seeded from it. The 74 of 74 recall stands as
a count. Section 5's 74 terminations discarded for being unlabelled were very
likely the same artifacts, because a one-row window can never fire the detector
and the figures are identical, but that run's report was not kept, so this is
inferred rather than recomputed.

**Fixed** in `scripts/harvest_sim_failures.py`: the contact sensor is re-reset for
environments reset in the step, history is cleared on every reset, every
termination tick goes through `is_post_reset_artifact` and `TerminationLedger` with
an episode generation and length recorded per record, and a report whose tick
accounting does not close is refused. No 1.0 artifact was modified. Nothing was
re-harvested: the corrected loop has not been run in Isaac, only the diagnostic
that confirms its contact fix.

### 8. Hardware-readiness criteria in force before the 2026-09-12 readiness pass

* **The H25 stand slew figures and their 5% gate.** `deploy_stand_h25.yaml` quoted
  "slew 3.65%" (nominal) and "slew 4.23%" (full DR) against "gate <5%". Both are
  legacy raw-action-delta figures (section 2). The config now keeps them verbatim
  under an explicit LEGACY heading and no longer uses them, or the 5% threshold, as a
  pass criterion. Hardware slew evidence is now the final LowCmd bridge's per-joint
  clip activation (`final_target_vs_policy_request_clip_activation_v1`), reported by the stand
  stages and not gated, because no corrected-metric simulator baseline exists for
  this checkpoint.
* **The canonical-stand bench threshold of 0.3.** H25 measures 0.482 and would fail
  it. The threshold compares action units against a radian slew clip (the same
  confusion as section 2) and a static single observation is not evidence about a
  feedback-stabilised stand. It is recorded as non-gating in the H25 lock; stage A
  instead gates on the output reproducing the locked value on the running runtime
  and on the first target step staying inside one slew cap.
* **`scripts/harness_preflight.sh` P1..P7.** P1 (T7 rsync) and P2 (`git fetch` plus
  fast-forward of `main`, mutating the payload) are gone; P4 picked the newest
  parquet on disk as its parity reference; P5 wrapped the dryrun in `|| true` and
  parsed `/tmp` files a previous run could have left; P7 asked for "feet unloaded
  (suspended slightly...)", contradicting the feet-on-ground stand card. A green run
  of that script is not evidence of anything. The staged gates A..H replace it.
* **`scripts/dryrun_pipeline.sh` before this pass** hardcoded `deploy.yaml` whatever
  the selected config, launched processes without checking they stayed alive, and
  reported through fixed `/tmp` paths. No earlier dryrun "pass" was checked against
  a specific config, lock or commit.
* **The previous LowCmd bridge's behaviour**, recorded here because every earlier
  bringup ran it: no absolute joint limits; clipping and holding against the last
  LowState however old; the estop heartbeat timed with the ROS wall clock (a
  backwards `date -s` on the payload made a dead publisher look fresh); NaN or
  malformed commands dropped rather than failed closed; and the policy node's
  default-pose messages (startup wait, abort) followed at up to 0.175 rad per tick.
  See `phoenix.sim2real.actuator_gate` for what replaced each.

### 2026-09-17 s9: the attitude threshold was briefly shared by sim and hardware

Between `e93f1b9` and the fix recorded here, `FailureThresholds`' defaults moved
from pitch 0.8 / roll 0.6 rad to 0.40 rad on both, to serve the hardware
attitude intervention. `phoenix.training.evaluate` builds its rollout analyzer
from the same dataclass, so for that window SIM rollouts were scored at the
HARDWARE bar: a stricter one than every historical sim number in this repo (the
32/32 stand successes, the H25 evaluations, every rollout metrics JSON under
`docs/`).

No sim result was produced or published in that window, so nothing on record is
invalid. The coupling is now removed rather than merely documented:

* `DEFAULT_ATTITUDE_INTERVENTION_RAD` (0.40 rad) stays the HARDWARE
  intervention, below the run card's 25 degree operator-halt instruction.
* `phoenix.real_world.failure_detector.sim_analysis_thresholds()` carries the
  SIM analysis bar (pitch 0.8 / roll 0.6), and `evaluate.py` and
  `phoenix.replay.variant_writer` both use it.

The two are separate quantities answering separate questions and neither should
move because the other did. Old and new sim failure counts are comparable again.
Pinned by `tests/test_variant_writer.py::test_a_tilt_below_the_sim_bar_is_not_a_failure`.

## Hardware-unverified

Nothing in this pass ran on hardware; no robot was connected at any point.

* The odometry twist frame contract. `nav_msgs/Odometry` expresses `twist` in
  `child_frame_id`, and the field notes record `child_frame_id=base_link`, which
  implies the twist is already body-frame and must NOT be rotated. Any rotation
  applied on that path is correct only at zero attitude if that reading is wrong.
* `foot_force` units and per-foot ordering. `unitree_go/msg/LowState` declares
  `int16[4] foot_force` with no documented calibration. Treated as raw counts.
* Odometry Z is boot-relative, so it is not an absolute height above ground and
  must not be used for collapse detection without a validated ground-relative
  source.
* Whether the corrected hip pose changes behaviour on the real robot.
* Whether the deploy-equivalent slew metric reproduces the hardware figure.
* The 2026-09-12 readiness pass (actuator gate, telemetry, staged gates): every
  ROS-side path was exercised only against localhost rehearsal fakes, never a GO2.
  Specifically unverified: what the GO2 motor firmware does when the LowCmd stream
  stops after a damping command; that `/lowstate` reaches the payload at the rates
  the field notes measured under the full bringup; that `wireless_estop_node` sees
  L1 continuously while held; that the policy behaves from Unitree's folded pose,
  which is not the training reset pose; and the low-level mode release procedure on
  this payload.
