# Research question

## Problem

A learned locomotion policy is trained against one actuator model and deployed on a
robot whose actuators then change: wear, heat, a loose transmission, a weak motor. The
common answers are to train once over a broad randomised actuator distribution
(robust but conservative, and it spends its samples on changes the robot does not
have), or to infer the change online from proprioceptive history (RMA, UP-OSI, the
fault-tolerant locomotion line), which only works for changes inside the training
distribution. Phoenix asks whether the robot's own measurement of *how* its
actuation changed can decide what the next training run randomises.

## Research question

> Can persistent actuator tracking residuals observed on a deployed quadruped be used
> to define a targeted simulation training distribution that improves locomotion
> under the robot's changed hardware dynamics without degrading nominal performance?

Public one-liner: *can a quadruped learn from changes in its own hardware?*

## Hypothesis

At matched fine-tuning compute, a candidate fine-tuned on the distribution Phoenix builds
from residuals measured on the robot (the targeted arm) scores higher under the degraded
condition than both a candidate fine-tuned on broad per-joint actuator randomisation that
covers the degraded value and a candidate that simply trained longer on the incumbent's
recipe, and is non-inferior to the latter under nominal conditions.

The one-joint model (`TOY_MODEL.md`) predicts that a policy which cannot identify the
change is brittle to it, that training around the measured value helps at that value,
and that doing so costs nominal performance. On one joint it does NOT predict that the
targeted-plus-nominal mixture Phoenix uses beats broad randomisation; that has to come
from what one joint cannot show (twelve joints, a fixed budget), and is exactly what the
experiment tests.

## Independent variable

The fine-tuning distribution. Five arms, all warm-started from the same incumbent with
identical PPO settings, iterations and env count (`configs/train/phoenix_finetune.yaml`):
A1 continued (incumbent recipe), A2 broad (every joint U[0.5, 1.15]), A2j joint-broad
(RR_thigh U[0.5, 1.0] on half the envs), A3 Phoenix (targeted term built by
`phoenix.condition` from hardware residuals), A4 oracle (targeted term centred on the true
applied scale).

The degradation is controlled: a software reduction of one joint's `kp`/`kd` by `s` on the
robot (`phoenix.sim2real.degradation`) and the same factor on that joint's actuator
stiffness and damping in Isaac Lab. It is not a physical fault and is never reported as
one. Its meaning depends on the execution limiter, which phase 0 freezes first.

## Dependent variables

Primary: per episode, the fraction of the episode spent with no trunk contact and true
base roll and pitch within 0.40 rad (for walking, also commanded-velocity tracking within
a bound fixed from the incumbent's nominal run); the mean over 256 episodes is the seed's
score. Secondary: deployment fidelity, attitude RMS, tracking error, and for the monitor
the detection rate, time to DEGRADED, localisation accuracy, false positives per nominal
hour and the bias of `s_hat` against the applied `s`. Compute is reported in environment
steps.

## Baselines

A0 the incumbent without fine-tuning (reference for hardware runs), A1 continued
training, A2 broad randomisation, A2j joint-broad. No online-adaptation arm is run, so no
claim is made against RMA-style methods and conclusions are limited to memoryless actors.

## Evaluation

Simulation is the statistical test: 5 arms x 10 training seeds x conditions nominal,
degraded at `s_train`, two held-out severities fixed in advance, and a wrong-joint
specificity condition. Hardware is the transfer check: six runs of five trials each,
reported per trial with exact binomial intervals. The protocol runs on the stand task
first (which cannot answer the locomotion part of the question) and is repeated unchanged
on a walking policy (`EXPERIMENT.md`).

## Falsification condition

The hypothesis is rejected if, at 10 seeds per arm, the 95 % Welch interval of A3 - A2 or
of A3 - A1 on the degraded score does not lie above zero, or if the interval of A3 - A1 on
nominal has a lower bound below -0.03. The detection claim is rejected if phase 1a
localises the injected joint in fewer than 8 of 10 sessions, or if any DEGRADED joint
appears in the nominal false-positive set. If A4 passes and A3 does not, the result is
reported as "targeting helps, the estimator is the bottleneck". If A2j matches A3, the
result is "choosing the joint matters, narrowing the range does not", and "narrowed" is
dropped from the claim. If phase 0 or the `s_train` pilot hits its stop rule, that is the
reported result.

## Novelty relative to the literature

Details and citations: `RELATED_WORK.md`. Updating a simulator's parameter distribution
from real data and retraining is prior art (SimOpt, BayesSim, DROPO; ASAP for a
deploy, collect, retrain cadence). Training once over per-joint actuator faults is prior
art, including on hardware (Gravina et al. 2026, DreamFLEX, AcL). Online latent
adaptation is prior art (UP-OSI, RMA). What Phoenix adds, if the experiment supports it:
a per-joint response residual measured through the safety layer on a deployed policy;
a distribution that is *narrowed* to the measured change rather than widened; a
promotion gate on degraded improvement and nominal non-inferiority; and a controlled
degradation that is the same parameter on the robot and in the simulator, so the whole
loop can be scored against a known ground truth. None of these is claimed as a
contribution until the experiment in `EXPERIMENT.md` has been run.
