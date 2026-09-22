# Phoenix v2 results, 2026-09-22

Status words as in `EVIDENCE.md`. Every number here is read from a file under
`results/phoenix_v2/`; per-step arrays and per-robot bridge telemetry are kept on the
workstation (gitignored, paths below). Nothing in this document ran on the GO2: the robot
was not reachable from the workstation during this work. Preregistration and all five
amendments: `docs/research/EXPERIMENT.md`.

## Answer to the research question, as of this date

Not answered. The program stopped at two preregistered gates before any adaptation
experiment:

* **Stage S (stand) stopped by the s_train stop rule** (amendment 4): the stand policy,
  executed faithfully, absorbs RR_thigh actuator degradation down to the smallest
  permitted scale (primary score drop 0.042 at s = 0.5, needed 0.10). There is nothing
  for a targeted distribution to repair.
* **Stage W (walking) stopped at Gate L** (amendment 5): the walking baseline trained
  under the frozen limiter fails execution fidelity (31 % of samples rate-limited) and
  tracks poorly.

What the work did establish is about deployment fidelity, which the adaptation claim
depends on.

## 1. Limiter selected, and why

Command-rate limiter, `|q_sent[k] - q_sent[k-1]| <= 0.075 rad` per 50 Hz tick
(3.75 rad/s), anchored on the last target actually sent, followed by the unchanged hard
envelope (URDF limits, 0.175 rad abort band, E-stop, deadman, watchdogs, hold/damp) and
a non-rewriting tracking abort (1.55 rad for 0.2 s). Selected by the amendment 1 rule on
development simulation only (`results/phoenix_v2/sim_limiter/`); the two-sided reading of
"within 0.02 of hard-only" was applied as written, which excluded 0.035 and 0.05 because
they changed closed-loop behaviour by more than 0.02 (they improved it). Frozen in
amendment 2, commit `5f18a78`; deploy config `configs/sim2real/deploy_stand_h25_v2.yaml`,
semantic sha256 `6312a359...`, lock `configs/sim2real/locks/deploy_stand_h25_v2.lock.yaml`.
The tracking-abort sizing rule returned 1.55 rad, above effort saturation (0.94 rad at
kp 25): it catches a joint that is not following, it does not bound torque.

## 2. Old versus new execution distortion

**A second deploy-contract defect was found first.** Isaac Lab clamps raw actions to
[-1, 1] before the action term and before `last_action` is observed
(`RslRlVecEnvWrapper(clip_actions=1.0)`, `ActionManager.action`); the exported ONNX and
the deploy node did neither. On the real runs 42 to 80 % of raw outputs were outside
[-1, 1] (max 9.7). The v2 node applies the clamp to the target and to the fed-back
action.

Offline replay of the recorded GO2 requests (open loop; `results/phoenix_v2/limiter_offline/`;
the measured-q replay reproduces the recorded sent targets to 4.8e-8 rad):

| run | request | limiter | altered | RMS (rad) | D |
|---|---|---|---|---|---|
| F1 (live, folded start) | as deployed | measured-q 0.175 | 88.8 % | 0.803 | 0.95 |
| F1 | trained-plant (clamped) | command-rate 0.075 | 29.0 % | 0.271 | 0.86 |
| B 9df76d7 (motors off) | as deployed | measured-q 0.175 | 66.7 % | 0.529 | 1.76 |
| B 9df76d7 | trained-plant | command-rate 0.075 | 0.42 % | 0.033 | 0.014 |
| B 01355b2 / cc13103 | trained-plant | command-rate 0.075 | 0.11 % | 0.002 | 0.001 |

No limiter produced a hard-limit violation (the envelope runs last). With the raw request,
F1's abort band would have fired at tick 8 (0.16 s), not tick 29: the node's clip hid an
illegal request for 21 ticks. F1's residual 29 % is the folded handover, which v2 no longer
permits (stand-up ramp, then a rate-limited handover).

In simulation, H25 on nominal physics, 256 episodes (`results/phoenix_v2/sim_limiter/`):

| limiter | altered | worst joint | fidelity pass | physical success | base height |
|---|---|---|---|---|---|
| measured-q 0.175 (as trained) | 35.7 % | 99.5 % | 0/256 | 0/256 | 0.168 m |
| command-rate 0.075 | 0.20 % | 0.46 % | 256/256 | 256/256 | 0.245 m |
| hard envelope only (reference) | 0 | 0 | 256/256 | 256/256 | 0.247 m |

Standing needs a p99 target-to-position gap of 0.49 rad (about 12 N m at kp 25); the
measured-q clip caps it at 0.175 rad (4.4 N m) and the robot stands 8 cm lower.

## 3. Isaac Lab standing result

Attitude from projected gravity; Isaac Lab 3.0 `root_quat_w` verified xyzw (error 6e-7,
wxyz 2.0), confirming audit H6. **Through the exact deployment path** (deploy
`ObservationBuilder`, ONNX Runtime on the shipped artifact, the node's action map, the
command wire and the real `ActuatorGate`, whose gains drive the simulated actuators;
`results/phoenix_v2/sim2sim/`): v2 config 64/64 physical success nominal, 62/64 under the
training randomisation, 64/64 pass the hardware fidelity gate on the gate's own
telemetry; the legacy config 0/64, every robot latching `target_beyond_limit` (the F1
signature) from a correct standing start. Factorial (`results/phoenix_v2/sim2sim_factorial/`):

| action clamp | limiter | success | altered | base height |
|---|---|---|---|---|
| no | measured-q | 0/64 (abort, fall) | 91.1 % | - |
| no | command-rate | 0/64 (abort, fall) | 93.6 % | - |
| yes | measured-q | 0/64 (fidelity) | 41.9 % | 0.162 m |
| yes | command-rate | 64/64 | 0.14 % | 0.245 m |

Both changes are necessary; the limiter's isolated effect (clamp held on) is 0 to 64/64
fidelity and 0.16 to 0.245 m stance height. Independent parity (a numpy re-implementation
of the actor from the raw checkpoint tensors, sharing no code with the export): max
|delta| 6.0e-6 against the shipped ONNX on 4882 real GO2 observations; ONNX normaliser
constants equal the checkpoint's (`results/phoenix_v2/parity/`).

## 4. Real GO2 standing result

NOT RUN. The robot subnet was not reachable. The v2 path has only simulation evidence.
Runbook: `docs/runbooks/HARDWARE.md`, "Phase E".

## 5-8. Monitor calibration, controlled degradation, detection, sim conditioning

Not run as preregistered (stage S stopped before phase 1). Exploratory, not a test
(`results/phoenix_v2/exploratory_monitor_sim_s0p5/`, deploy path, nominal physics,
calibrated on 10 nominal 60 s sessions): the monitor fails its own preregistered gate.
5 of 20 nominal 120 s sessions and 3 of 10 RR_thigh-at-1.0 sessions end with a DEGRADED
joint (s_hat 0.73 to 0.82); RR_thigh at 0.5 is localised in 5 of 10 sessions, SUSPECT at a
median 6 s and DEGRADED at 9 s, with s_hat 0.79 for a true 0.5 (biased towards nominal
under policy compensation); FL_calf and RL_hip at 0.5 trip the gate's saturation latch
before the monitor sees them. The monitor is not ready for hardware calibration. Phase I
(sim conditioning fidelity) was not run.

## 9-11. Training arms, broad DR versus Phoenix, non-inferiority

Not run. The Phoenix arm is defined on hardware phase-1b data, which does not exist, and
stage S stopped by rule.

## 12-13. Walking

Walking baseline trained (amendment 3 recipe,
`checkpoints/phoenix-walk-v2/2026-09-22_09-47-26/model_1499.pt`, gitignored). Gate L
(`results/phoenix_v2/gate_l/`): walking success 0/256 nominal and 0/256 with DR, 0/256
fidelity passes (31 % of samples rate-limited). The planar error 0.25 m/s and yaw-rate
error 0.31 rad/s quoted here earlier are INVALID (amendment 6.1: the scorer's settled
window was empty for 196 of 256 episodes under heading-derived yaw commands, so the
figures describe a biased subset); the fidelity failure alone fails the gate. Without the
limiter it walks worse (exploratory; its tracking figures carry the same defect). No
walking hardware run. Continued after amendment 5: see the 2026-09-22 addendum below.

## 14-17. Closed loop, hypothesis, claims

No closed-loop run occurred (steps 1 to 10 of the program's closed-loop definition: only
step 1 happened, on 2026-09-22, and it faulted). The central hypothesis was neither
supported nor refuted.

Strongest defensible claims:
* The incumbent's deployment rewrote most of its own policy, in simulation and on the
  robot, for two separable reasons: a measured-position clip that is also a 4.4 N m torque
  cap, and a missing action clamp that the training plant always applied. The missing
  clamp alone reproduces the F1 abort in simulation from a correct standing start.
* With the trained clamp and a 0.075 rad/step command-rate limiter, the same H25 policy
  runs through the exact deploy code in simulation with 0.14 % of targets altered and
  64/64 physical success. SIM VERIFIED, not hardware verified.
* In simulation, that stand policy absorbs a 50 % RR_thigh gain reduction.

Claims Phoenix cannot make: anything about hardware standing under v2; any monitor
detection or localisation performance; any sim-to-real correspondence of the degradation;
any benefit of targeted over broad randomisation; any walking capability; any closed loop.

## 22-23. Blockers and the next experiment

* Hardware access for phase E (the v2 path on the GO2).
* The monitor's false-positive rate in nominal simulation (session-level offsets).
* A walking baseline that passes Gate L.
* The stand task gives no adaptation target: a Phoenix test on standing needs a
  degradation the policy cannot absorb, and the only one permitted, one joint at 0.5,
  it absorbs.

Next single experiment: the walking recipe with no soft limiter in the MDP, a deployment
dq_max chosen for it on development seeds by the amendment 1 rule, and Gate L on fresh
seeds (amendment 5). Walking is where a single weak joint is most likely to matter.

## Artifact paths

Tracked: `results/phoenix_v2/{limiter_offline,sim_limiter,sim_limiter_steps,sim2sim,
sim2sim_factorial,strain_pilot,parity,gate_l,gate_l_diag,exploratory_monitor_sim_s0p5}`.
Workstation only: `*/steps.npz`, `*/bridge/robot*.jsonl` (gate telemetry, schema v2),
`checkpoints/phoenix-walk-v2/`, the GO2 logs under the main checkout's
`logs/payload_evidence_20260922/`. No video was recorded.

## Test count

`PYTHONPATH=src PHOENIX_SKIP_HEAVY=1 pytest tests -m "not sim and not ros"`, 2026-09-22,
after this work: 1732 passed, 17 skipped, 4 deselected, 0 failed (workstation with the
checkpoint directory present).


## Addendum, 2026-09-22 (later): walking without the soft limiter

Amendments 6 to 8 (`EXPERIMENT.md`). Deploy contract v3 (`DEPLOY_CONTRACT.md`) verified
against the live Isaac Lab plant with zero difference on every action and observation
term (`results/phoenix_v2/contract/w1/`); the mode-switch path of the policy node fixed
to use the same clamp. H25's limiter re-selected on new seeds under the one-sided rule:
0.035 rad/step, watchdog 1.40 rad (amendment 7). Walking candidates, all in
`results/phoenix_v2/walk_ledger.jsonl`:

| candidate | change | dev walking success (DR off / DR on) | stand | note |
|---|---|---|---|---|
| W1 (stopped at 500) | no soft limiter, yaw rate commanded | 0 / 0 | 1.00 | stopped prematurely (my error), kept |
| screens SA, SB (300 it) | feet_air_time 0.25; action_rate -0.01 | not evaluated | - | inconclusive: too short |
| W1-full (1500 it) | as W1 | 0.27 / 0.29 | 1.00 | walks backward only |

W1-full per direction (`results/phoenix_v2/walk_diag/w1_full_nominal/directional.json`):
strong backward -0.70 -> -0.74 m/s (segment success 0.99), strong forward +0.70 ->
+0.05 m/s (0.00). Learning curve over its checkpoints: backward appears between
iterations 1000 and 1100; forward stays near zero to the end. The sign/asymmetry audit
found no implementation bug (amendment 8.1).

### Walking, continued (same day)

| candidate | change from the one before | dev Gate W-H (DR off / DR on / stand) |
|---|---|---|
| W1-full | no soft limiter in the MDP, yaw rate commanded directly | 0.27 / 0.29 / 1.00 |
| W2 | 3000 iterations instead of 1500 | **0.941 / 0.918 / 1.000 PASS** |
| W5 rung 1 | W2 with action_rate -0.25 | 0.000 / 0.000 / 1.000 |
| W5 rung 2 | W2 with action_rate -0.5 | 0.000 / 0.000 / 1.000 |

W2 also passes Gate W-H on the fresh final seeds (0.9023 / 0.8906 / 1.0000) and is
directionally symmetric. W1's backward-only behaviour was not a bug (amendment 8.1) and
not an exploration trap: forward locomotion emerged between iterations 1600 and 1800.

Phase 7 then failed for W2 and for both W5 rungs, for opposite reasons (amendments 10
and 11): W2 commands bang-bang joint targets that no bound in the frozen grid can pass
as a seatbelt, and the smoother rungs do not walk at all. Phase 9 through the exact
deploy stack, with the gate's bound opened so it never binds, executes W2 faithfully
(0.0000 altered targets) at 0.953 nominal and 0.859 under training randomisation, and
fails the held-out friction (0.156 at 0.2, 0.703 at 1.8) and weak-actuator (0.719 at
0.75) conditions. The Stage W s_train pilot then stopped the stage by the same rule that
stopped Stage S (amendment 12): W2 absorbs RR_thigh at 0.5 (primary score drop 0.031).

**Status of the research question after this work: still unanswered, and untested.** The
program now has the two things it lacked, a faithful deployment contract and a valid
walking baseline, and it is stopped by the intervention being too weak, not by its own
machinery. Nothing here ran on the robot.
