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
