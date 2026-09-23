# Experiment

Preregistration for Phoenix v2. Everything in this file is fixed before any scientific
run. A change after the first scientific run needs a dated entry under "Amendments",
with the reason, and the report must list it.

Status on 2026-09-22: **nothing in this plan has been run.** Phase 0 is blocked on a
deployment-fidelity fix; phases 1 to 3 need Isaac Lab and the GO2. Revision 2 of this
file (same day) folds in an adversarial methods review; the changes are listed at the end.

## Scope: stand first, walking second

The incumbent is a stand-only policy. **Stage S** runs the whole protocol on the stand
task: it can answer "does a targeted distribution built from hardware residuals improve
behaviour under changed actuation, without harming nominal", for standing. It cannot
answer the locomotion part of the research question. **Stage W** repeats the identical
protocol, unchanged, once a walking policy passes phase 0; only Stage W can support a
claim about locomotion. Neither stage has started.

## Phase 0: deployment fidelity and a frozen limiter (precondition)

The audit found that the execution layer rewrote 88.8 % of joint-samples in the only live
policy run and 59.7 % in the incumbent's own simulated training distribution. At
`kp = 25` the bridge's measured-q slew clip (0.175 rad) caps PD torque near 4.4 N m. The
controlled degradation multiplies that cap by `s`, so until the limiter is fixed the
manipulated variable is not "actuator authority" but "a torque cap", and the monitor
(which excludes clipped samples) would see least on exactly the degraded joint.

1. **Measure.** Log per-joint demanded torque `kp |q_sent - q|` against `kp * 0.175` in the
   incumbent's sim evaluation and in one nominal live stand.
2. **Choose and freeze one limiter** (robot owner's decision, recorded here as an amendment
   before phase 1): keep the measured-q clip and retrain until requests stay inside it;
   switch to a command-rate limit (`clip_mode: prev_command`, implemented in sim), keeping
   hard joint limits, the abort band and the firmware torque limits; or retune `kp`/`kd`
   and the action scale together. Switching the limiter changes the safety envelope and
   needs explicit review.
3. **The frozen limiter is identical** in sim training, sim evaluation and the bridge.
   Its identity (mode and constants) is recorded in the training run directory and the
   bridge manifest; a mismatch makes the run inadmissible.
4. **Gate** (`phoenix.monitor.fidelity.PREREGISTERED`): at most 5 % of joint-samples altered
   by more than 1 mrad, RMS alteration at most 0.01 rad, at least 10 s of continuous
   authority, and at most 5 % altered on every individual joint. Sim first,
   then one nominal live run.
5. **Stop rule.** If no limiter option passes, the study stops and reports the fidelity
   result. No adaptation experiment runs on a policy the robot does not execute.

## Phase 1: monitor validation

**`s_train`** is fixed before phase 1 by a sim-only pilot on the phase-0 incumbent: the
largest `s` in {0.8, 0.7, 0.6, 0.5} at which the incumbent's primary score under RR_thigh
degradation drops by at least 0.10. If none does, the study stops with the finding "the
incumbent absorbs this degradation; there is nothing to adapt to".

**1a Simulation.** RR_thigh degraded at `s_train` and at 1.0, plus FL_calf and RL_hip at
`s_train` for localisation: 10 sessions of 60 s each. Nominal false-positive set: 20 sessions
of 120 s, each after an env restart so session-level offsets are present. Calibration
uses a separate 10-session nominal set. The simulator writes bridge-format tick records
(`phoenix.monitor.sim_records`, implemented and unit-tested; wiring it into `evaluate.py` is owed and must follow the
evaluator repair on the velocity branch). **Gate before hardware:** the injected joint is
localised in at least 8 of 10 sessions per injected joint, and no DEGRADED joint appears
in the false-positive set. Otherwise stop and fix the monitor, which then needs a new
preregistered validation.

Monitor settings, frozen: 1 s windows, `alpha = 0.01`, `min_effect = 0.10`, 8 of 10
windows, hysteresis 2 of 10, at most 2 localised joints, at least 60 % valid samples per
joint window.

**1b Hardware, at the same `s_train`.** Calibration: at least 10 nominal stands over at
least two days. Then 10 degraded sessions and 10 nominal sessions in randomised order.
**These hardware sessions, not the simulated ones, drive the Phoenix arm.**

## Phase 2: the simulation comparison (the statistical test)

**Arms.** All warm-start from the phase-0 incumbent with `configs/train/phoenix_finetune.yaml`
(300 PPO iterations, identical settings and env count):

| Arm | Distribution |
|---|---|
| A1 continued | incumbent recipe |
| A2 broad | every joint's stiffness and damping U[0.5, 1.15] |
| A2j joint-broad | incumbent recipe + RR_thigh U[0.5, 1.0] on half the envs (tests "narrowing": same joint, wide range) |
| A3 Phoenix | incumbent recipe + the targeted term from `phoenix_loop.py condition` on hardware 1b data |
| A4 oracle | as A3, centred on the true `s_train`, with the median width of the A3 ranges |

For A3, each training seed gets its own overlay, built from a bootstrap resample (with
replacement, seeded by the training seed) of the 1b degraded sessions, so the monitor's
uncertainty propagates into the arm; the resampled session IDs are recorded in the
overlay's provenance. A4's width is set after the A3 overlays are built and before any
arm is trained.

No online-adaptation (history-conditioned) arm is run. Every conclusion is therefore
limited to memoryless actors, and the paper makes no claim against RMA-style methods.

**Conditions.** nominal; RR_thigh at `s_train`; held-out RR_thigh at `s_train - 0.1` (if at
least 0.5) and at `s_train + 0.15`, both fixed now and evaluated for every arm; specificity
FL_calf at `s_train`. Evaluation overlays live in `configs/env/phoenix_v2/eval_*.yaml`
(one per value, written before training).

**Primary endpoint.** Per episode, a continuous score in [0, 1]: the fraction of the 20 s
episode during which the robot has no trunk contact and true base roll and pitch stay
within 0.40 rad (Stage W adds commanded-velocity tracking error within a bound fixed from
the incumbent's nominal run). 256 episodes per seed per condition; the seed's score is the
mean. The evaluator commit that fixes the Isaac Lab 3.0 quaternion order is pinned here
before phase 2 starts (amendment).

**Seeds.** 10 independent training seeds per arm, fixed now, from the compute available
(5 arms x 10 seeds x 300 iterations). With the normal approximation this gives 80 % power
at two-sided alpha 0.05 for standardised effects of about 1.3 and larger; smaller effects
may be missed, and a null result will be reported with that caveat.

**Hypotheses, all co-primary** (an intersection-union test: each at alpha 0.05, no
correction, all three required):

* H1a: A3 > A2 on the degraded condition.
* H1b: A3 > A1 on the degraded condition (targeting beats just training more).
* H1c: A3 is non-inferior to A1 on nominal, margin 0.03.

Test: 95 % Welch t interval of the difference in seed means
(`phoenix.validate.candidate_gate.compare_arms`), with the standardised effect size; a
percentile bootstrap interval is reported alongside. H1a and H1b pass when the lower bound
is above zero; H1c when it is above -0.03.

**Secondary, reported with intervals, no p-values:** A3 vs A2j (does narrowing beat a wide
range on the right joint); A3 vs A4 (cost of the estimator); held-out severities;
specificity as a difference in differences, (A3 - A2) on RR_thigh minus (A3 - A2) on FL_calf,
expected positive; nominal of every arm against A1.

Hardware candidate: the A3 seed with the median degraded score, chosen before held-out or
specificity results are opened.

## Phase 3: hardware transfer

On the floor, harnessed, operator holding the deadman. Degradation at `s_train` on
RR_thigh via `--experiment-degradation RR_thigh:<s_train>` with the other two locks
(`docs/runbooks/HARDWARE.md`).

| Run | Policy | Condition |
|---|---|---|
| A | incumbent | nominal |
| B | incumbent | degraded `s_train` |
| C | A2, median seed | degraded `s_train` |
| D | A3, selected seed | degraded `s_train` |
| E | A3, selected seed | degraded `s_train - 0.1` if at least 0.5 |
| F | A3, selected seed | nominal |

5 trials of 20 s per run, order randomised within a session. Reported per run: trials
passing the fidelity gate, mean primary score with its range, and successes out of 5
with the exact binomial interval (5/5 has a 95 % lower bound of 0.48: hardware is a
transfer check, not the statistical test). Trials failing the fidelity gate are kept and
reported; in a sensitivity analysis they count as score 0. D must hold a PROMOTE decision
before it becomes the new incumbent.

## Decision rules for promotion (fixed now)

The promotion gate is a different test from H1c: it compares ONE candidate artifact to
the incumbent over episodes (so a tight 0.02 margin is resolvable with 256 episodes),
while H1c compares arms over 10 seed means, where between-seed variance needs the wider
0.03 margin.

Candidate gate (`phoenix.validate.candidate_gate.PREREGISTERED`, applied per episode on
the continuous score): degraded improvement at least 0.05 with the bootstrap lower bound
above zero; nominal lower bound at least -0.02; held-out not worse by more than 0.02;
parity max_abs at most 1e-5 on the exact checkpoint. Conditioning: observed 2.5 to 97.5
percentile spread plus 0.05 each side, capped at [0.3, 1.0], half the envs nominal, at
most two joints, never from a global shift (`phoenix.condition.distribution`). No
threshold is tuned on phase 2 or phase 3 data.

## Threats

* **Floor effect** if the degraded joint's torque ceiling is below its stance torque: the
  phase-0 torque log decides; `s_train` selection then avoids values the incumbent cannot
  survive at all only if the pilot shows it (a total failure at every `s` is itself a
  reported result, not a reason to change the arms).
* **Load sharing.** In a four-foot stance a weak joint changes the load on others, which
  can bias `s_hat`, move the flag to another joint, or trigger a global-shift refusal.
  Phase 1a measures this before hardware.
* **Estimator bias under policy compensation.** Not measured by the unit tests (the one
  test with feedback shows none); A4 measures its effect on the outcome.
* **Software degradation is not a real fault.** It scales PD gains; a weak motor may
  saturate instead. Claims are limited to the applied intervention.
* **A3 inherits the incumbent's broad [0.85, 1.15] randomisation** under the targeted term,
  so its effective range is wider than the overlay states; A2j and A4 share the same base.

## Revision 2 changes (2026-09-22, before any run)

A3 now conditioned on hardware 1b data at `s_train` (was: simulated data at another
severity); phase 0 freezes one limiter for sim and bridge, with a stop rule and a per-joint
fidelity limit; `s_train` fallback and stop rule; continuous primary endpoint with 256
episodes (was: binary, 64); co-primary H1b against continued training and non-inferiority
against A1 with margin 0.03 (was: against A0, 0.02); added A2j; decided: no
online-adaptation arm; fixed 10 seeds (was: pilot-sized); Welch t interval primary;
per-seed bootstrap overlays for A3; oracle width matched to A3; held-out severities fixed
in advance for all arms; calibration across days and a session-offset false-positive set.

## Amendments

### Amendment 1 (2026-09-22, before any Phase B/C simulation run)

**Limiter family chosen by the robot owner: command-rate limiter** (`clip_mode:
prev_command`, `|q_sent[k] - q_sent[k-1]| <= dq_max`), applied before an unchanged
absolute hard envelope (URDF joint limits, abort band 0.175 rad, E-stop, deadman,
LowState/command watchdogs, hold/damp). Kp, Kd and action scale are NOT changed in the
same step.

**Offline finding recorded before simulation** (Phase A, `results/phoenix_v2/limiter_offline/`):
Isaac Lab's `RslRlVecEnvWrapper(clip_actions=1.0)` clamps raw actions to [-1, 1] in
training and evaluation; the exported ONNX and the deploy policy node do not. That clamp
is part of the trained plant. Layer 2 ("after policy-node transformation") is therefore
defined as `default + 0.25 * clip(raw, -1, 1)`, and the deploy node must apply the same
clamp. Execution fidelity is measured between layer 2 and the sent target (layer 3).

**Physical standing success, per 20 s simulated episode** (replaces "reached timeout"):
no trunk contact; |roll| and |pitch| <= 0.40 rad on every step, computed from
`projected_gravity_b` (no quaternion-order dependence; Isaac Lab 3.0 `root_quat_w` is
xyzw, verified in the harness); no layer-2 request beyond the abort band; and the
per-episode fidelity gate (altered <= 5 % of joint-samples overall and on every joint at
1 mrad tolerance, RMS alteration <= 0.01 rad). The continuous primary score is unchanged
(fraction of the episode with no trunk contact and roll/pitch within 0.40 rad).

**Phase C selection rule for dq_max, fixed now.** Development data only: the incumbent
checkpoint on (i) its training distribution (`stand_v3_h25`, DR on, seed 1001) and (ii)
the nominal condition (DR off, seed 1002), 256 episodes each, grid
{0.02, 0.035, 0.05, 0.075, 0.10, 0.175} rad/step. Choose the SMALLEST dq_max for which,
on both conditions, the altered fraction is <= 1 % overall and <= 5 % on every joint, and
the mean primary score is within 0.02 of the hard-envelope-only simulation reference.
If no grid value passes, stop and report; do not widen the grid after seeing results.
No held-out condition (degraded joints, evaluation seeds) is used for selection.
The chosen value is frozen with its commit and config hash in Amendment 2.

### Amendment 2 (2026-09-22, after the Phase B/C development sweep, before any held-out run)

**dq_max = 0.075 rad/step (3.75 rad/s at 50 Hz), frozen.** Source:
`results/phoenix_v2/sim_limiter/*/summary.json` (incumbent checkpoint, 256 episodes per
cell, dev seeds 1001 DR / 1002 nominal). Applying Amendment 1's rule:

| dq_max | DR altered | DR worst joint | DR score (hard-only 0.9703) | nominal altered | nominal score (hard-only 1.000) | rule |
|---|---|---|---|---|---|---|
| 0.020 | 2.88 % | 13.2 % | 0.9995 | 2.48 % | 1.000 | fails altered |
| 0.035 | 1.00 % | 2.3 % | 0.9930 (+0.023) | 0.88 % | 1.000 | fails "within 0.02" |
| 0.050 | 0.50 % | 1.3 % | 0.9971 (+0.027) | 0.42 % | 1.000 | fails "within 0.02" |
| **0.075** | **0.28 %** | **0.6 %** | **0.9894 (+0.019)** | **0.20 %** | **1.000** | **passes** |
| 0.100 | 0.17 % | 0.3 % | 0.9628 (-0.008) | 0.12 % | 1.000 | passes |
| 0.175 | 0.05 % | 0.1 % | 0.9620 (-0.008) | 0.04 % | 1.000 | passes |

Reading of the rule, stated openly: "within 0.02 of the hard-only reference" was applied
as written, two-sided. The small bounds (0.035, 0.05) fail it because they IMPROVE the
DR score by more than 0.02: at those bounds the limiter acts as a low-pass filter on a
policy whose raw output is outside [-1, 1] on 84 % of samples, which changes closed-loop
behaviour, the thing the rule exists to exclude. A one-sided reading ("not worse by more
than 0.02") would have chosen 0.035. The choice between the readings was made after
seeing the table; both are reported, and 0.075 is the literal one.

**Tracking abort (effort protection), rule fixed before its data is read.** With the
measured-q clip gone, `|sent - q|` is no longer bounded by 0.175 rad. The gate latches
hold if any joint's `|sent - q|` stays above `tracking_abort_rad` for 0.2 s.
`tracking_abort_rad` = the smallest multiple of 0.05 rad that is at least 1.25 x the
largest 0.2 s-sustained `|sent - q|` on any joint in the dq_max = 0.075 development runs
(nominal and DR, saved per step). If that value exceeds 0.94 rad (effort limit 23.5 N m
/ kp 25), the abort is reported as unable to add protection beyond motor saturation.

### Amendment 3 (2026-09-22, before any walking policy is trained)

**Stage W baseline (Phase L), recipe fixed now.** The smallest legitimate walking
baseline, not a locomotion contribution: `Isaac-Velocity-Flat-Unitree-Go2-v0` through
`configs/env/phoenix_v2/walk_flat.yaml` (the repository's `base.yaml` velocity ranges
vx [-1, 1] m/s, vy [-0.6, 0.6] m/s, yaw rate [-1, 1] rad/s, 2 % standing envs, its DR and
observation noise) with the frozen limiter in the MDP (`prev_command`, 0.075 rad/step)
and the [-1, 1] action clamp; PPO recipe `configs/train/ppo_walk_v2.yaml` (the v3b flat
recipe, 1500 iterations, 4096 envs, seed 42). Trained from scratch: no stand weights.

**Gate L, fixed now.** 256 episodes of 20 s per condition, evaluation seeds 4001 (DR off)
and 4002 (training DR), commands drawn from the training ranges and resampled every
10 s. Per-episode walking success: no trunk contact; |roll| and |pitch| <= 0.40 rad on
every step; the execution-fidelity gate against the layer-2 request; mean planar
velocity error <= 0.25 m/s and mean yaw-rate error <= 0.30 rad/s, each excluding the
first 1.0 s after episode start and after each command resample. Standing check: the
same policy under zero command meets the Amendment 1 standing success. Gate L passes
when walking success >= 0.90 (DR off) and >= 0.80 (DR on), standing success >= 0.90
(DR off), and the simulated deployment path (v2 gate, clamp, true body velocity as an
idealised odometry source) reaches walking success >= 0.90 (DR off). If the gate
fails, report the failure; the recipe is not tuned against the evaluation seeds.

### Amendment 4 (2026-09-22): Stage S stop rule triggered by the s_train pilot

The preregistered pilot ran on the phase-0 incumbent (H25 under the frozen v2 deploy
path, simulated deployment harness, nominal physics, 128 episodes per value, seed 3001,
RR_thigh kp and kd scaled by the deploy gate exactly as the hardware degradation does):

| s | primary score | drop vs s = 1.0 | success | safety latch |
|---|---|---|---|---|
| 1.0 | 1.000 | - | 128/128 | none |
| 0.8 | 1.000 | 0.000 | 128/128 | none |
| 0.7 | 1.000 | 0.000 | 128/128 | none |
| 0.6 | 1.000 | 0.000 | 128/128 | none |
| 0.5 | 0.958 | 0.042 | 121/128 | `degradation_joint_saturated` in 7/128 |

No value reaches the 0.10 drop, and 0.5 is the smallest scale the controlled
degradation permits (`MIN_SCALE`). By the stop rule written before the pilot, **Stage S
stops: the incumbent absorbs this degradation; there is nothing to adapt to.** Phases
1a (as a gate), 1b, 2 and 3 are not run for the stand policy. Anything run on the stand
policy after this point is labelled exploratory and is not evidence for or against H1.
Stage W (walking) is unaffected and repeats the protocol, pilot first, once a walking
policy passes Gate L.

### Amendment 5 (2026-09-22): Gate L fails; Stage W stops before hardware

`checkpoints/phoenix-walk-v2/2026-09-22_09-47-26/model_1499.pt` (recipe of amendment 3,
trained once, 1500 iterations) evaluated on the preregistered seeds
(`results/phoenix_v2/gate_l/`):

| condition | walking success | fidelity pass | altered | planar err (m/s) | yaw err (rad/s) |
|---|---|---|---|---|---|
| DR off (4001) | 0/256 | 0/256 | 31.5 % | 0.252 | 0.310 |
| DR on (4002) | 0/256 | 0/256 | 30.3 % | 0.268 | 0.360 |
| zero command (stand check) | 0/256 (stand success) | 0/256 | 1.9 % | 0.019 | 0.017 |

Gate L fails on every criterion's fidelity term and on tracking. Exploratory
diagnostic (not a gate, `results/phoenix_v2/gate_l_diag/nolimit`): the same checkpoint
with the limiter removed walks worse (walking success 14.8 %, planar error 0.35 m/s at a
mean commanded speed of 0.39 m/s), so the policy relies on the limiter as part of its
plant, exactly as H25 relied on the measured-q clip; the 0.075 rad/step bound chosen on
standing data is inside the walking gait's natural target rate. **Stage W stops:** no
walking policy exists that passes Gate L, so phases L (hardware), M and N are not run.
The recipe is not retuned on the evaluation seeds.

Next single experiment (not run): train the walking baseline with NO soft limiter in the
MDP (action-rate penalty only), then choose a deployment dq_max for it with the
amendment 1 selection rule on development seeds only, and re-run Gate L with fresh
evaluation seeds. It tests whether the fidelity failure is caused by training against a
binding limiter or by the bound itself.

### Amendment 6 (2026-09-22, before any run it governs): scorer defect, limiter criterion, recipe W

Written after amendment 5 and before the W1 training run, the limiter re-selection and
any evaluation below. Directed by the robot owner where stated.

**6.1 Harness defect in the Gate L tracking scorer.** `score_walk_episodes` (amendment 3)
excludes 1 s after every step on which the command changes. The walking task samples a
heading and derives `ang_vel_z` from the heading error on every step
(`heading_command: true`, `rel_heading_envs: 1.0`, Isaac Lab defaults; the YAML
`heading_stiffness` was never wired), so the yaw command changes on almost every step and
the settled window is empty: in `results/phoenix_v2/gate_l/nominal`, 196 of 256 episodes
have no settled sample and score an infinite tracking error (184/256 in
`gate_l_diag/nolimit`). The amendment 5 verdict is unchanged, because every episode also
failed the fidelity term (31 % of samples rate-limited), but its planar and yaw-rate
error figures describe a biased subset of episodes and are relabelled INVALID in the
results report; the artifacts are kept. Fix: `score_walk_v2` treats only command jumps
(resamples) as changes, and recipe W commands yaw rate directly
(`heading_command: false`, now wired in `_apply_commands`), holding it until resample,
which is what the joystick sends on the robot. Regression test:
`test_walk_v2_keeps_a_settled_window_under_a_drifting_yaw_command`.

**6.2 Limiter selection criterion, restated (owner-directed).** Amendment 1 required the
mean primary score to be "within 0.02 of the hard-envelope-only reference". Amendment 2
applied it two-sided and wrote down the intent as excluding any closed-loop change,
better or worse; that reading excluded 0.035 and 0.05, which improved the DR score by
0.023 and 0.027, and selected 0.075. The robot owner directs the criterion to be
one-sided: the score term guards against the limiter harming the task, and intervention
itself is bounded by the altered-fraction terms, which measure it directly. This is a
change of criterion, not a clarification of the original text, and it is written
knowing that on the amendment 2 data it would have selected 0.035. It therefore runs on
new development seeds; the amendment 2 table is not reused for selection.

For a policy `P`, development conditions `C` (its training distribution with DR, and
nominal with DR off), the grid `G = {0.02, 0.035, 0.05, 0.075, 0.10, 0.175}` rad/step,
`alt(d, c)` the fraction of joint-samples altered by more than 1 mrad, `alt_j(d, c)` the
same per joint, and `S(d, c)` the mean primary score (walking: mean walking success
indicator, `score_walk_v2`), with `S(hard, c)` from the same seeds and no soft limiter:

    dq_max(P) = min { d in G : for all c in C,
                      alt(d, c) <= 0.01  and  max_j alt_j(d, c) <= 0.05  and
                      S(d, c) >= S(hard, c) - 0.02 }

If the set is empty, the policy has no admissible limiter and is not deployed. The
two-sided result is reported beside it. H25 is re-selected on seeds 1101 (DR) and
1102 (nominal); a walking policy on seeds 5101 (DR) and 5102 (nominal), after it passes
Gate W-H (6.4). The value is frozen per policy in a further amendment before any fresh-
seed evaluation that uses it.

**6.3 Recipe W: no soft limiter in the training MDP.** Training plant: policy, clamp
[-1, 1], scale 0.25, DC-motor PD, PhysX joint limits (contract v3,
`docs/research/DEPLOY_CONTRACT.md`). Smoothness pressure lives in the objective only.
W1 is amendment 3's recipe with exactly two changes: `action.rate_limit.enabled: false`
and `command.heading_command: false` (`configs/env/phoenix_v2/walk_w1.yaml`,
`configs/train/ppo_walk_w1.yaml`, seed 42, 1500 iterations). Later candidates change one
diagnosed dimension at a time, each recorded in `results/phoenix_v2/walk_ledger.jsonl`
(commit, resolved env and train config, seed, checkpoint sha256, contract version,
limiter used at evaluation, development metrics). No run is deleted from the ledger.

**6.4 Gates and seeds.** Development evaluation (tuning allowed): seeds 5001 (DR off),
5002 (training DR), 5003 (zero command), 256 episodes of 20 s. Final evaluation, used
once per frozen candidate: 6001 (DR off), 6002 (DR on), 6003 (zero command).

* **Gate W-H (hard envelope only, Phase 6):** walking success `score_walk_v2` >= 0.90 DR
  off and >= 0.80 DR on; zero-command standing success >= 0.90. Walking success
  thresholds are provisional (`WALK_V2_PROVISIONAL`) and are frozen by amendment 7
  before the first final-seed evaluation.
* **Gate W-L (frozen deploy limiter, Phase 7):** the same, with the policy's frozen
  `dq_max`, plus the amendment 1 fidelity gate per episode (altered <= 5 % overall and on
  every joint, RMS <= 0.01 rad) inside walking success.
* **Gate W-D (exact deploy stack around Isaac Lab, Phase 9):** ONNX Runtime, deploy
  observation builder, `policy_action_map`, command wire and the real `ActuatorGate`,
  conditions nominal, training DR, held-out friction, held-out actuator scale, held-out
  command combinations; walking success >= 0.90 nominal and >= 0.80 on each other
  condition. Held-out values fixed in amendment 7.

Stage W's s_train pilot, monitor validation and arms follow only after Gate W-D, as in
the preregistration.

### Amendment 7 (2026-09-22): H25 limiter re-selected and frozen under the 6.2 rule

`results/phoenix_v2/sim_limiter_a6/` (H25 `model_799`, 256 episodes per cell, new dev
seeds 1101 DR / 1102 nominal, `scripts/phoenix_v2_select_dq.py`, which reproduces the
amendment 2 table exactly when pointed at the old sweep):

| dq_max | DR altered | DR worst joint | DR score (hard-only 0.9837) | nominal altered | nominal score (1.000) | one-sided | two-sided |
|---|---|---|---|---|---|---|---|
| 0.020 | 2.98 % | 13.9 % | 0.9996 (+0.016) | 2.51 % | 1.000 | fail | fail |
| **0.035** | **0.998 %** | **2.3 %** | **0.9930 (+0.009)** | **0.88 %** | **1.000** | **PASS** | PASS |
| 0.050 | 0.49 % | 1.3 % | 0.9939 (+0.010) | 0.42 % | 1.000 | PASS | PASS |
| 0.075 | 0.27 % | 0.6 % | 0.9967 (+0.013) | 0.20 % | 1.000 | PASS | PASS |
| 0.100 | 0.16 % | 0.3 % | 0.9840 (+0.000) | 0.12 % | 1.000 | PASS | PASS |
| 0.175 | 0.05 % | 0.1 % | 0.9753 (-0.008) | 0.04 % | 1.000 | PASS | PASS |

**H25 dq_max = 0.035 rad/step (1.75 rad/s), frozen.** On these seeds both readings of the
criterion agree. Margins are thin and are stated, not used to reopen the choice: the DR
altered fraction is 0.998 % against the 1 % bound; per-episode fidelity passes in 92 %
of DR episodes (RMS alteration 0.0091 rad against 0.01), against 100 % at 0.075. Also on
these seeds H25 violates the 0.40 rad attitude bound in 27 % of DR episodes with no soft
limiter at all (physical success 0.73 hard-only), a property of the policy under its own
training randomisation.

Catastrophic tracking-error watchdog for this bound, amendment 2 rule on the saved
dq_max 0.035 runs (`results/phoenix_v2/sim_limiter_a6_steps/`, bit-identical to the
sweep cells): largest 0.2 s-sustained `|sent - q|` 1.115 rad, so **1.40 rad for 0.2 s**
(PD effort saturates at 0.94 rad at kp 25; the watchdog is not torque protection).
Deploy config `configs/sim2real/deploy_stand_h25_v3.yaml`, semantic sha256 `419efa18...`,
lock `configs/sim2real/locks/deploy_stand_h25_v3.lock.yaml` (verified against the
artifacts, no problems). The v2 config (0.075, 1.55 rad) is kept and superseded.
Phase E on hardware, when the robot is reachable, uses the v3 config.

### Amendment 8 (2026-09-22, W2 at iteration ~1170, before any W2 checkpoint past 1500 is evaluated)

W2 (W1 recipe, 3000 iterations, seed 42) reproduces W1-full value for value through
iteration 1499 (deterministic training), so everything below is fixed before any W2
information that W1-full did not already give.

**8.1 Sign / asymmetry audit, done on the live W1 plant, no implementation bug**
(`results/phoenix_v2/asymmetry_audit/audit.json`, `scripts/phoenix_v2_asymmetry_audit.py`):
the command sampler is symmetric (102,400 samples: 50,198 positive vs 50,096 negative
vx; tail probabilities P(vx > a) vs P(vx < -a) within |z| < 1.3 for a in 0.1..0.95;
standing 2.06 %; no curriculum term); Isaac Lab's tracking rewards are exactly mirror-
symmetric on the six sign cases; body +x points at the head and a velocity written
along the heading reads with the correct sign and earns the matching command's reward.
The default stance is mildly back-biased: after 1 s of zero action the whole-body COM is
1.9 cm (sd 1.4) behind the feet's support centroid and the trunk is 3 deg nose-up. The
command-sign flip on the observation (diagnostic, `--flip-command-obs x`) flips the
behaviour: W1-full reads the command and only knows how to execute its negative sign.
Under forward commands W1-full leans nose-down and sees more attitude violations (6.5 %
vs 0 %), thigh contacts (4.6 % vs 0 %) and trunk contact (0.7 % vs 0 %) than backward.
Directional bins (`phoenix.monitor.walk_directional`) are diagnostic only; Gate W-H is
unchanged.

**8.2 W2 decision rule.** `F(it)` = strong-forward bin (vx_cmd > 0.4) mean achieved vx /
mean commanded vx, nominal dev seed 5001, 128 episodes, W2 checkpoints 1500..2999.
(a) W2 passes dev Gate W-H: freeze, fresh-seed evaluation, then Phase 7.
(b) Else, if `F(2999) >= 0.5` and `F(2999) - F(2600) >= 0.05`: training-extension study
E, resumed from W2's final checkpoint in blocks of 500 iterations, at most 3 blocks
(4500 total); after each block dev Gate W-H and `F`; stop at a gate pass (freeze, fresh
seeds), at a block improving `F` by less than 0.03 (plateau, go to W3), or after 3 blocks.
(c) Else: W3.

**8.3 W3, symmetric command curriculum (one change from W1).** `command.curriculum`
(`phoenix.sim_env.curriculum_command`, stage machine `phoenix.sim_env.command_curriculum`,
unit-tested): all three command ranges scaled by stage factors 0.2, 0.4, 0.6, 1.0, so
every stage is symmetric; advance only when forward AND backward tracking ratios, each
the mean achieved/commanded vx over the last 2000 qualifying segments of its own sign
(|vx_cmd| >= 0.5 x stage max, at least 500 per sign, first 1 s excluded), are both
>= 0.7 on 2 consecutive checks (every 50 iterations), at least 100 iterations in a stage;
a stage that has not qualified after 1000 iterations stays (no promotion on time) and
the run is reported as failing to advance. Budget 3000 iterations, W1 otherwise.
Development: one run, seed 42; after it, only the stage factors, the ratio threshold and
the maximum stage duration may be changed, once, and the change is recorded. Then frozen
and trained on fresh seeds 101, 102, 103, each evaluated on dev Gate W-H. W3 works if at
least 2 of 3 pass; the candidate is the passing seed with the median dev walking
success, and only it sees the final seeds (6001-6003).

**8.4 W4, exploration (only if W3 fails).** One variable family: PPO `entropy_coef`
0.005 -> 0.01, W3's frozen curriculum kept, same dev + 3-seed protocol.

**8.5 Stopping rule.** If W4 also fails (fewer than 2 of 3 seeds pass dev Gate W-H),
Stage W stops with the finding that this flat-ground PPO recipe (Isaac Lab GO2 flat task,
clamp [-1, 1], scale 0.25, no soft limiter) did not produce a symmetric walking baseline
within the declared budget. No further locomotion variables are tried in this study.

### Amendment 9 (2026-09-22): W2 passes dev Gate W-H; thresholds and candidate frozen

W2 (W1 recipe, 3000 iterations, seed 42, 36 min, deterministic match to W1-full through
iteration 1499) on the development seeds, hard envelope only: walking success **0.941**
(DR off, 5001), **0.918** (DR on, 5002), stand **1.000** (5003). Gate W-H (0.90 / 0.80 /
0.90) PASSES, so amendment 8.2 takes branch (a). Checkpoint `model_2999.pt`, sha256
`94790929ea9f8a78...`, run `checkpoints/phoenix-walk-w2/2026-09-22_13-34-53`.

Directionally (`results/phoenix_v2/walk_diag/w2_nominal/directional.json`) it is
symmetric: strong forward +0.701 -> +0.697 (segment success 0.987), strong backward
-0.703 -> -0.694 (1.000), both mild bins 1.000. Forward emerged between iterations 1600
and 1800 and converged by about 2000 (`results/phoenix_v2/walk_curve/w2/`).

**Walking success thresholds frozen** at the values `WALK_V2_PROVISIONAL` has carried
since before W1 (they were never changed after seeing a result): planar error <= 0.25
m/s, yaw-rate error <= 0.30 rad/s, settle 1.0 s, progress ratio >= 0.80, minimum base
height >= 0.20 m, effort saturation <= 1 %, joint speed <= 30 rad/s, plus the amendment 1
stand criteria (no trunk contact, |roll| and |pitch| <= 0.40 rad, no abort-band request,
per-episode execution fidelity, no safety hold, full episode length).

**Final-seed evaluation** (once, this candidate): 6001 DR off, 6002 DR on, 6003 zero
command, 256 episodes, hard envelope only.

**Phase 7 for W2**: the amendment 6.2 rule on development seeds 5101 (DR) and 5102
(nominal), grid unchanged {0.02, 0.035, 0.05, 0.075, 0.10, 0.175}; if no value passes,
the rule says the policy has no admissible limiter and is not deployed. Gate W-L then
runs on fresh seeds 6011-6013 with the selected bound.

**Phase 9 held-out conditions, fixed now** (exact deploy stack, seeds 6101+): (A) nominal;
(B) training randomisation; (C) held-out friction, static and dynamic 0.2 and 1.8
(training 0.3 to 1.5); (D) held-out actuator scale, motor strength 0.75 and 1.25
(training 0.85 to 1.15); (E) held-out command combinations, vx and vy and yaw rate all
near their corners simultaneously (|vx| in [0.7, 0.9], |vy| in [0.4, 0.5], |wz| in
[0.7, 0.9], signs drawn independently).

### Amendment 10 (2026-09-22): W2 passes Gate W-H on final seeds but has no admissible limiter

**Gate W-H, final seeds, once, hard envelope only** (`results/phoenix_v2/walk_final/w2/`):
walking success **0.9023** (6001, gate 0.90), **0.8906** (6002, gate 0.80), stand
**1.0000** (6003, gate 0.90). PASS. Phoenix has a walking policy that does not depend on
downstream target shaping.

**Phase 7 (`results/phoenix_v2/walk_limiter/`, dev seeds 5101/5102, amendment 6.2 rule):
the admissible set is EMPTY.** Altered fraction per bound (DR / nominal): 0.02 28.7 /
36.0 %, 0.035 21.4 / 30.8 %, 0.05 19.7 / 28.5 %, 0.075 19.1 / 23.7 %, 0.10 16.7 /
19.4 %, 0.175 7.4 / 8.5 %, against the 1 % limit; walking success falls from 0.926 / 0.902
(hard envelope only) to 0.000 at every bound. W2's own target rate is the reason: median
per-tick target change 0 rad, p99 0.42 rad, maximum 0.500 rad, which is the full span the
[-1, 1] clamp allows at action scale 0.25, and raw outputs are outside [-1, 1] on 66 % of
steps. The policy is bang-bang, so a stand-derived command-rate bound is not a seatbelt
for it. **By the rule, W2 is not deployed.** It stays the reference walking baseline.

**W5, smoothness in the objective (one dimension, declared ladder).** W2's recipe with a
larger `action_rate` weight, the term whose quantity IS the per-tick target change
(`action_rate_l2` on the clamped action; target change = 0.25 x action change). Ladder,
in order: `-0.25` (5x), then `-0.5` (10x); 3000 iterations, seed 42, everything else
identical, grid for the limiter unchanged. **Promotion rule, fixed now:** take the
SMALLEST penalty in the ladder whose policy (i) passes dev Gate W-H and (ii) has a
non-empty admissible limiter set under amendment 6.2 on dev seeds 5101/5102. Then its
final-seed Gate W-H and Gate W-L run once, and Phase 9 follows. If neither rung
satisfies both, Stage W reports: a flat-ground policy trained with no soft limiter meets
the walking gate but cannot be executed within a stand-derived command-rate envelope, and
no further locomotion variable is tried in this study (amendment 8.5 stands).

Hardware, rechecked at this point: no 192.168.123.0/24 interface exists on the
workstation and none of .161 / .18 / .15 answer. Hardware phases stay BLOCKED.

### Amendment 11 (2026-09-22): the W5 ladder fails; Stage W continues in simulation on W2

**W5 ladder result.** Rung 1 (`action_rate` -0.25) and rung 2 (-0.5), 3000 iterations
each, seed 42, everything else as W2: both FAIL dev Gate W-H with walking success
0.000 / 0.000 and stand 1.000. Neither left the stand-still plateau that W2 escaped
between iterations 1600 and 1800; their joint-step jump fractions (1.9 % and 0.7 % above
0.075 rad) are free, because a standing policy is trivially smooth. No rung satisfies
both promotion conditions, so by amendment 10 no smoother walking candidate exists in
this study and no further locomotion variable is tried (amendment 8.5 stands).

**Deployability finding (Stage W).** In this recipe family a flat-ground policy either
tracks the command with bang-bang joint targets (W2: p99 per-tick target change 0.42 rad,
maximum 0.500 rad, the full span the clamp allows) or is smooth and does not walk (W5).
W2's own p99 target rate, 21 rad/s, is the same order as the actuator's velocity limit
(sim 30 rad/s, GO2 datasheet 21 rad/s), while measured joint speed stays at p99 6.9 rad/s:
the PD plant low-passes the command. A command-rate limiter chosen on standing data is
therefore not a seatbelt for walking; it is a second controller. This is reported as a
result, not repaired by widening the frozen grid.

**How the program continues.** Hardware is blocked three ways: the GO2 subnet does not
exist on the workstation, `deploy_contract.WALKING_ENABLED` is False, and the walking
prerequisites (validated odometry above all) have no hardware evidence. The adaptation
study therefore continues **in simulation only**, on W2 as the Stage W baseline, and
every result it produces is labelled SIM VERIFIED and carries the two limits above: W2
is not deployable under the frozen limiter rule, and no hardware run exists. The deploy
path used for it is the exact one (ONNX Runtime, deploy observation builder,
`policy_action_map`, command wire, `ActuatorGate`) with the gate's rate bound opened to
0.6 rad/step, which never binds (0.0000 altered targets, `results/phoenix_v2/walk_deploy/`),
recorded in every manifest as `limiter_max_delta_override`.

**Stage W s_train pilot, as preregistered** (phase 1): W2 through that path, nominal
physics, RR_thigh gain scaled by the deploy gate exactly as on hardware, s in
{1.0, 0.8, 0.7, 0.6, 0.5}, 128 episodes, seed 3101. Endpoint: the Stage W continuous
primary score (`phoenix.monitor.stand_metrics.walk_primary_score`: the fraction of the
episode with no trunk contact, roll and pitch within 0.40 rad, and, outside the 1 s
settling windows, planar velocity error within the frozen 0.25 m/s bound). `s_train` is
the largest s whose score drops by at least 0.10 against s = 1.0; if none does, Stage W
stops with the same finding the stand stage reached.

### Amendment 12 (2026-09-22): Stage W stops at the s_train pilot, the same rule as Stage S

The preregistered pilot ran on W2 through the exact deploy path (nominal physics,
128 episodes per value, seed 3101, RR_thigh kp and kd scaled by the deploy gate exactly
as the hardware degradation does, `results/phoenix_v2/walk_strain_pilot/`):

| s | Stage W primary score | drop vs 1.0 | walking success | safety-hold episodes | fidelity pass | mean planar error |
|---|---|---|---|---|---|---|
| 1.0 | 0.9888 | - | 0.9375 | 0/128 | 1.000 | 0.095 m/s |
| 0.8 | 0.9780 | 0.0108 | 0.9219 | 2/128 | 0.984 | 0.105 m/s |
| 0.7 | 0.9775 | 0.0113 | 0.9141 | 4/128 | 0.969 | 0.109 m/s |
| 0.6 | 0.9709 | 0.0179 | 0.9062 | 5/128 | 0.961 | 0.110 m/s |
| 0.5 | 0.9579 | 0.0309 | 0.8906 | 9/128 | 0.930 | 0.120 m/s |

No value reaches the 0.10 drop, and 0.5 is the smallest scale the controlled degradation
permits (`MIN_SCALE`, a hardware-safety bound). **By the stop rule written before the
pilot, Stage W stops: the walking baseline absorbs this degradation too, so there is
nothing for a targeted distribution to repair.** Phases 1a, 1b, 2 and 3 are not run for
walking. The effect is real but small and monotone in severity (safety-hold episodes
0 to 7 %, fidelity 100 to 93 %, tracking 0.095 to 0.120 m/s); it is simply far below the
preregistered bar.

**Both stages of the study have now stopped at the same place, for the same reason**: the
one controlled degradation this program is allowed to apply, a single joint's PD gains
scaled to at least 0.5, does not move either policy's preregistered endpoint by 0.10.
The central hypothesis is therefore still untested, and is not refuted: the study never
reached the comparison it was designed to make.

**Next scientific decision, recorded, not acted on here.** The evidence points at the
intervention, not at Phoenix's machinery. In the same deploy path, a GLOBAL actuator
weakening to 0.75 on every joint did move walking substantially (success 0.953 -> 0.719,
`results/phoenix_v2/walk_deploy/w2_open/d_actuator_weak/`), while one joint at 0.5 did
not. A follow-up study would have to preregister a stronger or multi-joint intervention
(and justify it against the hardware-safety bound that fixes `MIN_SCALE` at 0.5) and a
more sensitive endpoint than the 0.10 drop, before any of it is run. Nothing in the
present study is reinterpreted to get a positive result.

### Amendment 13 (2026-09-22, before any screening run): intervention screening, Stage W

Both stages stopped at the same rule, and amendment 12 recorded that the evidence points
at the intervention rather than at Phoenix's machinery. This amendment opens the
follow-up that amendment 12 said a follow-up would have to preregister. The full protocol
is `docs/research/INTERVENTION_SCREENING.md`, committed before the first screening cell
ran; this entry is the formal amendment and states only what changes.

**13.1 W2 is frozen.** No retrain, resume or re-export for the duration.
`model_2999.pt` sha256 `94790929ea9f8a78...`, ONNX `fd0d3f3087365453...`, train yaml
`b9dba652ec30b917...`, resolved env `b1bc6fc7bde9150c...`, training commit `20b46c9`,
contract v3. The immutable reference is the existing `W2` row of
`results/phoenix_v2/walk_ledger.jsonl`; no new row is written.

**13.2 The sensitivity endpoint changes, before the screening runs.** The Stage W
continuous primary score (`walk_primary_score`, bar 0.10) is demoted to a secondary
metric. It is nearly blind to this intervention class, and the evidence is entirely from
runs that predate this amendment: `walk_deploy/w2_open` nominal 0.9531 walking success at
primary score 1.0000, versus `d_actuator_weak` 0.7188 at 0.9858. A condition that removes
a quarter of the successful episodes moves that score by 0.014, because it scores step
time inside a 0.40 rad attitude bound and a 0.25 m/s settled tracking bound, neither of
which a weakened robot violates for most of an episode. No intervention inside the
permitted safety envelope can reach 0.10 on it, so retaining it would guarantee a stop
irrespective of the physics.

**New primary endpoint: walking success** (`walk2_success_rate`, `score_walk_v2`), whose
thresholds were frozen in amendment 9 and are not touched here, and which is already the
Gate W-H endpoint. **New sensitivity bar: a drop of at least 0.15** against the
screening's own nominal on matched seeds; at 384 episodes per cell the pooled binomial
standard error near p = 0.9 is 0.0153, so the bar is about 10 standard errors.

**13.3 The controlled degradation may name a joint SET, with a floor that rises with
reach** (robot owner's decision). `MIN_SCALE` stays 0.50 for a one-joint spec; the new
`MIN_SCALE_MULTI` is **0.70** for two or more joints, every affected joint taking the
same scale. Nothing else in the gate is weakened: reduction only, POLICY mode only, 2 s
ramp, triple-locked arming, per-tick logging, and the hard limits, abort band, tracking
watchdog, E-stop, deadman and freshness rules unchanged. The saturation latch is now
tracked **per joint**, so a wider set cannot dilute it. No value below a floor is
screened, for any reason.

**13.4 Families, severities, seeds.** A nested extent ladder crossed with severity:
C1 `RR_thigh` (1 joint) at 0.8/0.7/0.6/0.5; C2 `leg_RR` (3); C3 `rear` (6); C4 `all` (12),
each at 0.90/0.85/0.80/0.75/0.70. One shared nominal cell. C5 (latency) is **not**
screened: the deployment layer has no reversible latency mechanism, and a response lag is
a different question from delivered authority. Screening development seeds **7001-7003**,
128 robots per seed; **7101 and above** reserved for confirmatory and held-out use.

**13.5 Base physics is DR off for every screening cell**, so the intervention is the only
variable. This supersedes the exploratory global-0.75 figure quoted in amendment 12:
`walk_deploy_d_actuator_weak.yaml` changes motor strength **and** pins friction to 0.8
**and** adds actuator latency 1-5 steps, against a DR-off nominal, on 64 episodes of one
seed. **That 0.953 -> 0.719 result is confounded and is not a dose-response point.**

**13.6 Selection rule, frozen before results.** A cell qualifies on all of: pooled drop
>= 0.15; the drop >= 0.15 in each of the three seeds; pooled walking success remaining
>= 0.40; safety hold <= 0.15, attitude violation <= 0.30, abort band == 0, fidelity
>= 0.90; and its family monotone within 0.05 across adjacent severities. Among qualifying
cells, take the fewest affected joints, then the least severe severity, tie-broken by the
smaller monotonicity violation. Preferring fewer joints is recorded as a choice made for
interpretability and safety envelope, not for scientific advantage; the opposite
consideration is that a joint subset gives the targeted arm an axis broad randomisation
lacks, so **if C4 also qualifies it is carried as a preregistered secondary
intervention**. The grid points immediately milder and immediately stronger than the
selected severity are held out.

**13.7 Stop rule.** If no cell qualifies, the adaptation experiment stops with the
finding that no safe actuator intervention within the tested envelope produced the
required measurable degradation. No floor is lowered, no family added, no threshold
revisited, and the endpoint is not changed a second time.

**13.8 Hardware.** Rechecked 2026-09-22 at the start of this work: no
`192.168.123.0/24` interface on the workstation (`eno1` 192.168.8.189/24, `wlp9s0`), the
robot subnet routes to the default gateway, and .161 / .18 / .15 do not answer. Hardware
phases stay BLOCKED, not failed. The screening and anything following it are SIM VERIFIED
at best until that changes.

### Amendment 14 (2026-09-22): the degradation saturation latch is re-sized for walking

Written after the first (latch-armed) pass of the amendment 13 screen and before the
pass that supersedes it. It changes one experiment-specific safety constant and nothing
else. The screening endpoint, bar, families, severities, seeds and selection rule of
amendment 13 are UNCHANGED.

**The finding.** The saturation latch fires when a degraded joint's requested target sits
at least `DEGRADATION_PIN_BAND_RAD` (0.175 rad) beyond its measured position for
`SATURATION_LATCH_S` (0.5 s). That band is the historical slew cap and was reasoned about
for STANDING, where a joint pinned that far is a sagging leg. It is not a sag detector for
a bang-bang walking policy. On the screen's own nominal walking telemetry, **with no
degradation applied at all** (seeds 7001-7003, 12 sessions, 144 joint-sessions):

* per-joint p99 of `|requested - q|` reaches **0.69 rad**, about four times the band;
* **9 of 144 joint-sessions (6.2 %) sustain the latch condition for 0.5 s or more**, one
  of them for 10.18 s.

Nominal never latches only because the latch is armed exclusively when a degradation spec
is present. Arm a no-op (scale 1.0) spec and a healthy robot would latch. In the C2 cell
(`leg_RR` at 0.80) the trip count was **6 of 144, no higher than nominal's 9 of 144**.

**What that did to the first pass.** The C2 family cleared the magnitude bar (pooled
walking-success drop 0.164 to 0.190) but failed the `not_catastrophic` rule on safety
holds (0.206 to 0.234 against the 0.15 bound) and fidelity (0.77 to 0.79 against 0.90).
Splitting its episodes shows the effect is not graded locomotion degradation at all:

| `leg_RR` at 0.80 | episodes | walking success | min base height |
|---|---|---|---|
| latch fired | 90 / 384 | 0.000 | 0.106 m |
| latch did not fire | 294 / 384 | 0.956 | 0.281 m |
| nominal reference | 384 | 0.922 | 0.282 m |

In roughly three quarters of episodes the weakened leg walks as well as nominal; in the
rest the gate latches, the robot sinks and scores zero. Only 1.4 % of un-held degraded
episodes ever breach the 0.20 m height bound. The measured "degradation" was substantially
the fraction of episodes tripping a latch sized for a different task.

**The change, sized by this program's own rule.** Amendments 2 and 7 sized the
catastrophic-tracking watchdog by taking the largest sustained `|sent - q|` observed on
nominal development runs and rounding up with margin (observed 1.115 rad, frozen 1.40).
The same rule, applied to the latch statistic on nominal walking development seeds
7001-7003: the largest 0.5 s-sustained `|requested - q|` is **0.499 rad** (RR_calf);
times the same 1.25 margin gives 0.624; rounded up on the same 0.05 grid:

> **`DEGRADATION_PIN_BAND_WALK_RAD` = 0.65 rad, sustained 0.5 s, for walking policies.**

At kp 25 a joint held 0.65 rad from its target is demanding 16.3 N m and not moving, which
is what "pinned" was meant to mean. It remains stricter than the general
catastrophic-tracking watchdog (1.25 rad for 0.2 s), which is unchanged and runs whether
or not a degradation is armed.

**Scope of the change.** `DEGRADATION_PIN_BAND_RAD` (0.175 rad) is unchanged and remains
the default, so every standing configuration is bit-identical to before. The band is now a
gate parameter (`GateParams.degradation_pin_band`), recorded in every manifest and in every
tick record, so no run can be read back without knowing which bound was in force. Nothing
else in the envelope is touched: hard joint limits, abort band, tracking watchdog, E-stop,
deadman, LowState freshness, reduction-only, POLICY-mode-only, the 2 s ramp and the
triple-locked arming all stand. The floors (`MIN_SCALE` 0.50, `MIN_SCALE_MULTI` 0.70) are
unchanged.

**Consequence for the screen.** The amendment 13 screen is re-run in full with the walking
band. The first pass is kept, not deleted, under
`results/phoenix_v2/intervention_screen_pinband_0p175/`, and is reported as the evidence
for this amendment. No screening threshold or rule is changed with it.

### Amendment 15 (2026-09-22): intervention frozen; the monitor rule and its gate, before any gate run

Phases G to J. Written after the screen selected, and **before any of the 120 s monitor
sessions it governs exist**. Nothing below may be changed once the first of those sessions
has run; a failure is reported as a failure.

**15.1 The intervention, frozen (Phase G).**

| field | value |
|---|---|
| family | C3, both rear legs |
| target expression | `rear:0.70` |
| joints | RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf |
| severity | 0.70 on kp and kd, uniform over the six |
| floor that bounded it | `MIN_SCALE_MULTI` = 0.70 |
| simulator implementation | deploy `ActuatorGate`, the same object as on the robot, via `--degrade rear:0.70 --allow-degradation --degradation-pin-band 0.65` |
| hardware implementation | the same, behind the triple lock; hardware is BLOCKED, so unexercised |
| development severity | 0.70 |
| held-out severity, reserved and unused | 0.75 |
| secondary intervention, preregistered | C4 `all:0.80` and `all:0.75` |
| selection evidence | `results/phoenix_v2/intervention_screen/screening.{json,md}` |

Walking success 0.6328 against a nominal 0.9219, a drop of 0.2891, every seed at least
0.2734. Selection is not revisited now that training is about to begin.

**15.2 The monitor is reframed around persistent actuator-response change (Phase H).**
`phoenix.monitor.health` asks which single motor is bad and treats a many-joint change as
a veto (`GLOBAL_SHIFT`), which `condition.distribution.build_targeted_spec` then refuses.
That question is both unanswerable here and no longer the right one: on nominal W2 walking
telemetry undegraded joints score anywhere from 0.75 to 1.63, and with one joint truly at
0.80 it was the lowest-scoring joint in a minority of sessions; and the screen eliminated
the single-joint family outright.

`phoenix.monitor.response_shift` instead scores each of a FIXED hypothesis space of
physically meaningful joint groups: the per-window median of `s_hat` over the group's
joints, then the median over windows. A group flags when it is persistently below its
calibrated threshold AND its members moved together (the interquartile spread of member
medians is inside its calibrated nominal range). The reported answer is the LARGEST
flagged group, which is the widest extent the evidence supports.

**Hypothesis space, frozen: the twelve multi-joint groups** (`all`, `rear`, `front`,
`diag_a`, `diag_b`, `leg_FR/FL/RR/RL`, `hips`, `thighs`, `calfs`). The twelve singletons
are dropped: they are the noisiest candidates in this regime and no screened family can
produce one. Every candidate costs false-alarm budget, so the space is small and fixed in
advance rather than searched.

**Configuration, frozen:** `alpha` 0.05, `min_effect` 0.06, persistence 7 of the last 10
windows, `min_usable` 8 windows, `spread_quantile` 0.90, `min_spread_bound` 0.10, 1 s
windows, at least 60 % valid samples per joint-window.

**15.3 Why the rule is frozen UNCHANGED despite a 0.50 development false-flag rate.**
On the screening's 20 s sessions the detector flagged something in half of held-out
nominal sessions. That is a calibration-depth artifact, not a fault in the rule, and the
diagnosis is recorded here because it is what justifies changing nothing: no group's
threshold was clamped by the `min_effect` floor, and the observed per-window alarm rate on
held-out nominal sessions was **0.105 to 0.351 against the 0.05 the threshold was set
for**, because that threshold was a 5 % quantile estimated from about 114 windows. With an
honestly estimated 0.05, the 7-of-10 persistence rule puts a session-level false alarm
near 1e-6 across twelve groups. The single remedy applied is therefore **more and longer
calibration sessions**, and no threshold, quantile, persistence count or aggregation is
touched.

**15.4 Calibration and validation sessions (Phase I).** Frozen W2, exact deploy stack, DR
off, 120 s per session (about 119 windows, against the 19 the screening cells gave), one
independent session per simulated robot, 12 robots per run, walking latch band 0.65 rad.
Calibration, nominal validation and degraded sets are seed-disjoint, and the evaluator
refuses overlapping calibration and validation sets.

| set | seeds | sessions | condition |
|---|---|---|---|
| calibration | 7101, 7102 | 24 | nominal |
| nominal validation | 7103, 7104 | 24 | nominal |
| degraded | 7105, 7106 | 24 | `rear:0.70` |
| held-out severity | 7107, 7108 | 24 | `rear:0.75`, reported not gated |

**15.5 The gate, frozen (Phase J).** Applied ONCE, to the sets above. All four required:

* **G1 false alarms.** Session-level nominal false-flag rate (any group flagged) **<= 0.05**.
* **G2 detection.** The selected group `rear` reaches SHIFTED in **>= 0.80** of degraded sessions.
* **G3 extent.** The reported group is `rear` in **>= 0.70** of degraded sessions.
* **G4 severity usefulness.** Median severity estimate within **0.10** of the applied 0.70,
  and the reported `[lo, hi]` contains 0.70 in **>= 0.70** of detected sessions.

**If any of the four fails, targeted adaptation STOPS at Phase J.** The monitor is not
retuned to pass, no threshold is revisited, the hypothesis space is not re-cut, and the
session sets are not extended. Two claims are then reported separately, because the
development evidence already separates them: that the residual carries usable severity
information (the group estimate moved monotonically, 0.806 to 0.956 for an applied 0.70 to
0.90), and that the detector could not operationalise it at the session level. The
recorded finding in that case is: **Phoenix could estimate actuator-response severity
monotonically in simulation, but could not reliably distinguish nominal from degraded
walking at the session level.**

### Amendment 16 (2026-09-22): the monitor fails its frozen gate; targeted adaptation stops at Phase J

The amendment 15.5 gate, applied once, to the session sets of amendment 15.4
(`results/phoenix_v2/monitor_gate/shift_monitor.json`). Calibration 24 nominal sessions
of 120 s (2856 windows per group, against the 114 the screening cells gave); validation
24 held-out nominal, 24 degraded at `rear:0.70`, 24 held-out at `rear:0.75`. Every session
cleared `min_usable`; the smallest had 10 usable windows, the nominal sets 119 each.

| criterion | bar | result | verdict |
|---|---|---|---|
| G1 nominal false-flag rate | <= 0.05 | **0.1250** (3 of 24) | **FAIL** |
| G2 detection of `rear` | >= 0.80 | **0.7083** | **FAIL** |
| G3 reported group is `rear` | >= 0.70 | **0.3750** | **FAIL** |
| G4a median severity bias | <= 0.10 | 0.0746 (estimate 0.775 for an applied 0.70) | PASS |
| G4b reported range covers truth | >= 0.70 | 0.7647 (17 detected sessions) | PASS |

**Three of five fail, so by the rule written before these sessions existed, targeted
adaptation STOPS.** Phases K through P are not run: no conditioner is built, no targeted
distribution is generated, and no training arm (no adaptation, broad DR, Phoenix targeted,
oracle) is trained. The monitor is not retuned, no threshold is revisited, the hypothesis
space is not re-cut and the session sets are not extended.

**Deeper calibration helped, and was not enough.** Against the development pass on 20 s
sessions, the false-flag rate fell from 0.50 to 0.125 and detection rose from 0.08 to
0.708. The diagnosis in amendment 15.3 was therefore correct in direction and insufficient
in size: the rule was not broken, the calibration was thin, and fixing the calibration did
not close the gap. The three false alarms were `front`, `hips` and `leg_FL`, one session
each, all groups the intervention never touched.

**Two claims, reported separately, because the evidence separates them.**

1. **Severity estimation works, and is the positive result of this phase.** The group
   statistic recovered the applied scale to within 0.075 at the development severity and
   0.095 at the held-out severity, monotonically (0.775 for an applied 0.70, 0.845 for an
   applied 0.75), with its reported interval covering the truth in 76 % and 88 % of
   detected sessions. It passed both halves of its criterion. The residual carries usable
   information about how much actuator authority was lost.

2. **Session-level detection and localisation do not work.** The detector fires on healthy
   walking once in eight sessions, misses the degradation in three of ten, and names the
   right six joints in fewer than four of ten. It cannot be trusted to decide *whether* to
   adapt or *what* to adapt, which is what the Phoenix arm would have been conditioned on.

Stated as the finding: **Phoenix could estimate actuator-response severity monotonically
in simulation, but could not reliably distinguish nominal from degraded walking at the
session level.** That is a more informative failure than "the monitor did not work", and
it localises the remaining problem: the detection threshold, not the estimator.

**What this does NOT say.** It does not test whether a monitor-targeted distribution beats
broad randomisation; that comparison was never reached. It says nothing about hardware,
which stayed BLOCKED throughout. It does not show that no detector could work on this
signal, only that this one, frozen in advance, did not.

**The next scientific decision, recorded and not acted on here.** The estimator passing
while the detector fails points at the decision rule, not the residual. Any follow-up
would have to preregister a different detector (for example one that scores a single
preregistered group rather than selecting among twelve, which would remove the
multiple-comparison load that produced all three false alarms) and re-run this same frozen
gate on fresh seeds before any adaptation arm is trained. Nothing in the present study is
reinterpreted to get a positive result.

### Amendment 17 (2026-09-22): detector v2, one preregistered attempt, before any of its sessions exist

Detector v1 failed the amendment 15.5 gate at three of five criteria while its severity
estimator passed both of its (amendment 16). This is ONE further attempt at the decision
rule. It is written and committed **before any session it is validated on has been
generated**, and it is the last: if it fails, the Phoenix adaptation line stops and no
detector v3 is built in this study.

Nothing else moves. W2 is unchanged, the frozen intervention (`rear:0.70`) is unchanged,
the frozen intervention screen is untouched, the gate thresholds are the amendment 15.5
ones, and no adaptation policy is trained before the gate passes.

**17.1 Why v1 failed, and what that licenses changing.** v1 thresholded twelve groups
independently on per-WINDOW quantiles and reported the largest that flagged. Its three
false alarms were all groups the intervention never touched. Two defects: a
multiple-comparison load of twelve tests per session with no family-wise control, and
calibration at window level for a gate that measures a SESSION-level rate, with the
observed per-window alarm rate on held-out nominal running 0.105 to 0.351 against the
0.05 its thresholds were set for. v2 changes the decision rule only. **The severity
estimator is reused unchanged**: it is the part that passed, and it is not redesigned.

**17.2 Architecture, frozen (`phoenix.monitor.response_shift_v2`).**

*Stage 1, detection.* Session statistic `S = median over the session's windows of (median
over all twelve joints of s_hat)`. A session is SHIFTED when `S < tau_global`, where
`tau_global` is the `alpha` quantile of `S` over NOMINAL DEVELOPMENT SESSIONS, floored so
it is never closer to 1 than `min_effect`. Calibrating at session level is what makes the
false-alarm rate the quantity the gate measures. Taking the median over windows IS the
persistence rule: a session flags only when more than half its windows are below the bound.

*Stage 2, extent, only if stage 1 fired.* Five groups, frozen: `all`, `front`, `rear`,
`left`, `right`, i.e. the whole robot and its two complementary anatomical bisections.
Each group's session statistic is standardised against its own nominal development
distribution, `z_g = (mu_g - S_g) / sigma_g`, so groups of different sizes are comparable.
The reported group is `argmax_g z_g` subject to `z_g > tau_fw`; if nothing clears it the
extent is UNRESOLVED and no group is named.

*Family-wise control.* `tau_fw` is the `1 - alpha_fw` quantile of `max_g z_g` over the
nominal development sessions, i.e. the null distribution of the most extreme group, which
is what "take the best group" actually tests. This replaces v1's independent per-group
thresholding, and it is derived from development sessions only.

*Group selection is by evidence, not by size.* A group's statistic is a median over its
members, so a whole-robot change moves `all` fully while a rear-only change moves it
halfway, and the larger group also has the smaller `sigma_g`. No size preference or
tie-break is wired in. **Nothing privileges `rear`**, and 17.5 checks that directly.

*Severity mapping, unchanged from v1:* the selected group's session shift, with the
2.5/97.5 percentile of its per-window series as the reported range.

*Configuration, frozen:* `alpha` 0.05, `min_effect` 0.04, `alpha_fw` 0.05,
`window_quantile` 0.5, `min_usable` 8 windows, `min_sigma` 0.01, 1 s windows, at least
60 % valid samples per joint-window.

**17.3 Excluded hypotheses, and why.** The four individual legs are excluded: a leg is
three of twelve joints and a median over twelve does not move when a quarter of them do,
at any severity, so stage 1 can never fire on a single-leg change and a single-leg
hypothesis would raise `tau_fw` for every other group while never being selectable. The
screen also eliminated the one-leg family as an intervention (C2 `leg_RR` reached 0.1250
against the 0.15 bar). Diagonal pairs and the per-joint classes are excluded because no
screened family produced them. The twelve singletons are excluded as in v1.

**17.4 Declared limitation, before the gate.** Because stage 1 aggregates with a median,
**v2 detects an actuator-response shift affecting at least half the robot and cannot
detect one confined to a single leg**, at any severity. That is a property of the
aggregate, not a tuning choice. v2 is not a general fault detector. The selected
intervention affects exactly six of twelve, as does the specificity condition.

**17.5 Sessions, on fresh seeds.** No session used in detector development or in the
failed v1 gate (7101-7108) is reused. 120 s, DR off, frozen W2, exact deploy stack,
walking latch band 0.65 rad.

| set | seeds | sessions | condition |
|---|---|---|---|
| calibration (development) | 7109, 7110 | 48 | nominal |
| nominal validation | 7111, 7112 | 24 | nominal |
| degraded, **gated** | 7113, 7114 | 24 | `rear:0.70` |
| held-out severity | 7115, 7116 | 24 | `rear:0.75` |
| **specificity** | 7117, 7118 | 24 | `front:0.70` |

The specificity set degrades the FRONT legs. It exists to check that v2 infers the
affected group from telemetry rather than defaulting to the study's selected answer. It is
**reported, not gated**: the verdict is read from the `rear:0.70` set alone, as amendment
15.5 specifies. A `front:0.70` session reported as `rear` would be a serious finding and
is reported as one whatever the gate says.

**17.6 The gate, unchanged from amendment 15.5**, applied ONCE to the sets above, with the
verdict read from the `rear:0.70` condition: nominal false-flag <= 0.05; detection
>= 0.80; reported group correct >= 0.70; median severity bias <= 0.10; reported range
covers the applied scale in >= 0.70 of detected sessions. All five required.

**17.7 Outcomes, fixed now.** If v2 PASSES, the study proceeds immediately to the
adaptation comparison: monitor output to a targeted actuator distribution, then
matched-compute no-adaptation versus broad DR versus Phoenix targeted, with an optional
oracle-targeted diagnostic kept clearly secondary. If v2 FAILS, **the Phoenix adaptation
line stops**; no detector v3 is built in this study, no threshold is revisited, and the
reported finding is that actuator-response magnitude was estimable while reliable
condition detection and localisation were not achieved.
