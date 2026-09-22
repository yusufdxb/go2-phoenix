# Intervention screening, Stage W

Preregistration. Written and committed **before any screening run**. It is governed by
`EXPERIMENT.md` amendment 13, which is the formal amendment; this file is the protocol
in full. Nothing below may be changed once the first screening cell has run. A change
after that needs its own dated amendment naming what it changes and why.

Date: 2026-09-22. Branch `research/phoenix-hardware-adaptation-v2`.

---

## 0. Why this study exists

Both stages of Phoenix v2 stopped at the same rule (amendments 4 and 12): the one
controlled degradation the program was allowed to apply, **a single joint's PD gains
scaled to at least 0.5**, does not move either policy's preregistered endpoint by 0.10.
Stage W's pilot measured a primary-score drop of 0.031 at `s = 0.5` on RR_thigh.

That is a fact about the **intervention**, not about Phoenix's machinery. This study
asks one question, and only this one:

> What is the least severe actuator-condition change that produces a reproducible,
> non-catastrophic and monotonic degradation in W2 locomotion?

The screening is **not training**. W2 is frozen throughout (section 1). No policy is
fine-tuned, no monitor is calibrated, and no adaptation arm is run inside this study.
If the screening finds a qualifying intervention, the adaptation experiment follows
under its own preregistration. If it does not, Stage W stops and says so.

---

## 1. Phase A: the frozen baseline

W2 is the Stage W baseline and is **not retrained, resumed or re-exported** during this
study. Every screening cell loads exactly these artifacts.

| Field | Value |
|---|---|
| candidate | `W2` |
| run directory | `checkpoints/phoenix-walk-w2/2026-09-22_13-34-53` (gitignored) |
| checkpoint | `model_2999.pt` |
| checkpoint sha256 | `94790929ea9f8a78eb8b19c8ff445f8f919bf14221aefde4768ad2441c54a8f8` |
| exported actor | `policy.onnx`, sha256 `fd0d3f30873654530462fc40b8917267202cc779d85d69ee6eaa42782fc0fdbb` |
| train yaml sha256 | `b9dba652ec30b9172b1550ba870797e8a270736dfd01a8a2cf01df94752d77e3` |
| resolved env sha256 | `b1bc6fc7bde9150c350cacc72a8e16c309807b167c588dc69d7434fe8643163e` |
| training commit | `20b46c9db4c7b619d4f0c21bb5e67856a9b42985` |
| seed | 42, 3000 iterations |
| action contract | v3 (`DEPLOY_CONTRACT.md`): clamp [-1, 1], action scale 0.25, DC-motor PD, PhysX joint limits |
| deploy config | `configs/sim2real/deploy_walk_w2_sim.yaml` |
| hard envelope | URDF joint limits, abort band 0.175 rad, tracking watchdog 1.25 rad / 0.2 s, E-stop, deadman, LowState freshness, hold/damp |

The immutable reference entry already exists as the `W2` row of
`results/phoenix_v2/walk_ledger.jsonl` (recorded 2026-09-22T18:17:18Z, ledger commit
`20b46c9`), which carries all of the above plus its three development evaluations. This
study adds no new W2 row; it cites that one.

Frozen Gate W-H result on the final seeds, for reference (amendment 10): walking success
**0.9023** (6001, DR off), **0.8906** (6002, DR on), stand **1.0000** (6003).

---

## 2. Phase D: the endpoint, and why it changes

### 2.1 The preregistered endpoint is abandoned, before this screening runs

The Stage W sensitivity endpoint was the continuous primary score
(`phoenix.monitor.stand_metrics.walk_primary_score`) with a drop of at least 0.10. It is
**retained as a secondary metric and abandoned as the primary**, for one reason, which is
readable off data that already exists and was NOT produced by this screening:

| existing run | walking success | continuous primary score |
|---|---|---|
| `walk_deploy/w2_open/a_nominal` (nominal) | 0.9531 | 1.0000 |
| `walk_deploy/w2_open/d_actuator_weak` (global 0.75, confounded) | 0.7188 | 0.9858 |
| `walk_strain_pilot/rr_thigh_1p0` | 0.9375 | 1.0000 |
| `walk_strain_pilot/rr_thigh_0p5` | 0.8906 | 0.9579 |

A condition that removes a quarter of W2's successful episodes moves the continuous
score by **0.014**. The reason is structural, not incidental: `walk_primary_score` is the
fraction of steps with no trunk contact, roll and pitch inside 0.40 rad, and settled
planar velocity error inside 0.25 m/s. A weakened GO2 still stays upright and still
tracks within 0.25 m/s most of the time; it fails on progress ratio, base height and
sustained tracking, which that score does not test. The endpoint cannot reach 0.10 for
any intervention inside the permitted safety envelope, so keeping it would guarantee a
stop regardless of the physics.

This is an endpoint change made **before** the screening and justified **only** on runs
that predate it. It is recorded here, in amendment 13, and in the final report.

### 2.2 The primary endpoint

**Walking success**, `walk2_success_rate`, from
`phoenix.monitor.stand_metrics.score_walk_v2`. It is the conjunction of eight per-episode
physical checks (stand criteria, settled planar error <= 0.25 m/s, settled yaw-rate error
<= 0.30 rad/s, progress ratio >= 0.80, minimum base height >= 0.20 m, effort saturation
<= 1 %, joint speed <= 30 rad/s, target-jump fraction). Its thresholds were frozen by
amendment 9 and have not been changed since before W1; nothing in them is tuned here.

It is already the Gate W-H endpoint, so the screening measures the same quantity the
walking baseline was accepted on.

### 2.3 The sensitivity bar

**A condition qualifies on magnitude when walking success drops by at least 0.15**
against the screening's own nominal cell, measured on the same seeds through the same
path.

Chosen before any screening run, on these grounds:

* **Noise.** 128 episodes per seed, 3 seeds, 384 episodes per cell. At p ~ 0.9 the
  binomial standard error of a pooled cell is `sqrt(0.9 x 0.1 / 384) = 0.0153`. A 0.15
  drop is about 10 standard errors, so it cannot be a sampling artifact.
* **Headroom.** Combined with the floor in section 5, a 0.15 drop leaves a policy that
  still succeeds on a large minority of episodes, which is what makes an improvement
  measurable in both directions.
* **Comparability.** It is the same order as the abandoned 0.10 bar, on an endpoint whose
  observed nominal-to-collapse range is about 0.95 down to 0.16 (friction 0.2,
  `walk_deploy/w2_open/c_friction_low`).

### 2.4 Secondary metrics, reported for every cell, no thresholds attached

`walk_primary_score_mean`, `walk2_failures_by_check` (which of the eight checks failed),
`walk2_mean_lin_err_m_s`, `walk2_mean_yaw_err_rad_s`, `walk2_median_progress_ratio`,
`walk2_p05_min_base_height_m`, `walk2_p95_max_joint_speed_rad_s`,
`walk2_mean_effort_saturation`, `fidelity_pass_rate`, `safety_hold_episode_rate`,
`attitude_violation_episode_rate`, `abort_band_episode_rate`, `altered_fraction`,
`mean_base_height_m`, `effort_saturation_fraction`, `gate_faults`.

---

## 3. The mechanism, and what changed to allow it

### 3.1 One mechanism, applied at the deploy layer

Every screened intervention is a **uniform reduction of the PD gains (`kp` and `kd`) of a
named joint set, applied by the deploy `ActuatorGate` in POLICY mode**, i.e. by
`phoenix.sim2real.degradation.DegradationSpec` through
`scripts/phoenix_v2_sim2sim_deploy.py --degrade ... --allow-degradation`. This is the
same code path, and the same object, that would run on the GO2.

It is deliberately NOT the Isaac-side `motor_strength_scale` / `targeted_actuator` route,
even though the two compose into the same DCMotor stiffness
(`phoenix_v2_sim2sim_deploy.py:222-225` reads `act.stiffness` after the startup events,
then writes `kp_p * kp_fac` each tick). The gate path is chosen because:

* it carries the **saturation latch** (`degradation_joint_saturated`), which is part of
  how the intervention behaves on a robot and which fired in 0 to 7 % of episodes in the
  Stage W pilot. An env-side injection has no latch and would understate the failure rate;
* it is the only form that is representable on hardware (Phase U);
* it keeps sim and hardware on one code path, so a screening number and a hardware number
  mean the same thing.

### 3.2 The multi-joint extension and its floor

Until this study `DegradationSpec` was hard-limited to one joint, and "one joint at a
time" was part of its written safety argument. Every family stronger than the one that
already failed needs more than one joint.

The robot owner has permitted the multi-joint form **with a floor that rises with reach**:

| reach | floor | constant |
|---|---|---|
| exactly 1 joint | 0.50 | `MIN_SCALE` (unchanged) |
| 2 or more joints | 0.70 | `MIN_SCALE_MULTI` (new) |

Rationale, recorded: a single weak joint is load-shared by the other eleven; a whole leg
or the whole robot at the same scale removes that margin, so the permitted envelope is
narrower the further the intervention reaches. **No value below a floor is screened, for
any reason.** Lowering a floor to manufacture a failure is out of scope for this study
and for any follow-up.

Everything else about the gate is unchanged and is not weakened: reduction only, POLICY
mode only, 2 s ramp-in, triple-locked arming on the real bridge
(`PHOENIX_EXPERIMENT=controlled_degradation` + stage label `X*` + the CLI spec), per-tick
logging of the scale actually applied, and hard limits, abort band, tracking watchdog,
E-stop, deadman and freshness rules untouched. The **saturation latch is now tracked per
joint**, so widening the set cannot dilute it: any one affected joint pinned for
`SATURATION_LATCH_S` latches HOLD.

Implemented and unit-tested before this preregistration was committed
(`tests/test_controlled_degradation.py`, 46 tests).

---

## 4. Phase C: the families

One interpretable ladder: **extent** (how many joints) crossed with **severity** (the
scale). Not a combinatorial sweep. The extent ladder is nested, each family containing
the one above it, so a monotone reading across families is meaningful.

| family | target | joints | n | severities |
|---|---|---|---|---|
| **C1** single joint, the known-failed reference | `RR_thigh` | RR_thigh | 1 | 0.8, 0.7, 0.6, 0.5 |
| **C2** one leg | `leg_RR` | RR hip, thigh, calf | 3 | 0.90, 0.85, 0.80, 0.75, 0.70 |
| **C3** both rear legs | `rear` | RR + RL, all joints | 6 | 0.90, 0.85, 0.80, 0.75, 0.70 |
| **C4** global | `all` | all twelve | 12 | 0.90, 0.85, 0.80, 0.75, 0.70 |

Rear-right is the root of the ladder purely for continuity with the original study; no
result is read as being about that leg specifically.

C1 is re-run inside this screening, on the screening's own seeds and nominal reference,
so that the known-failed reference is comparable with the rest of the table rather than
quoted from a different seed. It is not expected to pass and does not get a second
chance: a pass would be a finding about seeds, and would be reported as such.

**C5, latency / response lag, is NOT screened.** Recorded decision, with the reason: the
deployment layer has no reversible actuator-latency mechanism (the Isaac-side
`actuator_latency_steps` has no counterpart in the `ActuatorGate`), so it fails the
"representable in the deployment software layer" and "safe and reversible for later
hardware work" criteria; and a response lag is a different research question from a
change in delivered authority. Phoenix does not need every possible fault.

---

## 5. Phase E: the selection rule, frozen before any result

A cell is **(family, severity)**. A cell **qualifies** when all of the following hold.

1. **Magnitude.** Pooled walking-success drop vs the screening nominal >= **0.15**.
2. **Reproducibility.** The drop is >= 0.15 in **each** of the three seeds individually,
   not only pooled.
3. **Headroom.** Pooled walking success remains >= **0.40**. Phoenix needs a repair
   problem, not a dead robot.
4. **Not catastrophic.** Pooled `safety_hold_episode_rate` <= 0.15,
   `attitude_violation_episode_rate` <= 0.30, `abort_band_episode_rate` == 0, and
   `fidelity_pass_rate` >= 0.90.
5. **Ordered family.** Its family's dose-response is monotone within noise: for every
   adjacent pair of severities `s_i > s_{i+1}` in that family,
   `success(s_{i+1}) <= success(s_i) + 0.05`. A family whose response reverses by more
   than that is a poor experimental intervention even if one of its points is dramatic,
   and no cell in it qualifies.

**Selection, among qualifying cells:**

1. Prefer the family with the **fewest affected joints**.
2. Within that family, take the **least severe** (largest scale) qualifying severity.
3. Tie-break between families of equal size: the smaller maximum monotonicity violation.

Justification for preferring fewest joints, recorded now so it cannot be re-read later:
physical interpretability, and the safety envelope, since the raised multi-joint floor
bites less the fewer joints are involved. It is **not** chosen for scientific advantage.
Note the opposite consideration explicitly: a smaller joint set gives the Phoenix arm an
extra axis (which joints) that broad randomisation lacks, so a subset intervention makes
the targeted-versus-broad contrast easier for Phoenix, while a global intervention forces
targeting to win on parameter range alone. To keep that answerable either way:

> **If a global family (C4) also qualifies, it is carried as a preregistered SECONDARY
> intervention, and the adaptation study reports it alongside the primary.** This is fixed
> now, before results, so that "did the joint-subset axis do the work" cannot become a
> post-hoc question.

**Held-out severities**, fixed now: for the selected family, the grid point immediately
milder and the grid point immediately stronger than the selected severity are reserved as
held-out and are **not** used for development, monitor calibration or conditioner design.
If the selected severity is at an end of its grid, only the one neighbour exists and only
it is reserved.

**Stop rule.** If no cell qualifies, the Phoenix adaptation experiment stops, and the
report reads: *no safe actuator intervention within the tested envelope produced the
required measurable degradation.* No floor is lowered, no family is added, no threshold
is revisited, and the endpoint is not changed again.

---

## 6. Phase F: how the screening runs

| setting | value |
|---|---|
| policy | frozen W2 (section 1), exact deploy stack |
| path | `scripts/phoenix_v2_sim2sim_deploy.py --walk` (ONNX Runtime, deploy observation builder, `policy_action_map`, command wire, real `ActuatorGate`) |
| deploy config | `configs/sim2real/deploy_walk_w2_sim.yaml` |
| env config | `configs/env/phoenix_v2/walk_deploy_a_nominal.yaml` (**DR off**) for every cell |
| limiter | `--limiter-max-delta-override 0.6`, the bound used for every W2 result, which never binds (0.0000 altered) |
| episodes | 128 robots x 20 s per seed; 384 per cell |
| seeds | **7001, 7002, 7003** (screening development, not used anywhere else in this program) |
| telemetry | 4 robots per run, for the later monitor work |
| output | `results/phoenix_v2/intervention_screen/<family>_<severity>/seed<N>/` |

**Base physics is DR off, deliberately.** The intervention is then the only variable that
differs between a cell and the nominal cell. This fixes a defect in the exploratory global
result that motivated the study: `configs/env/phoenix_v2/walk_deploy_d_actuator_weak.yaml`
changes motor strength **and** pins friction to 0.8 **and** adds actuator latency 1-5
steps, against a DR-off nominal, on 64 episodes of one seed. Its 0.953 -> 0.719 figure is
therefore **confounded and is not a dose-response point**; it is superseded by this
screening and is reported that way.

Seeds reserved and not used for development: **7101 and above**, for the confirmatory and
held-out evaluations of whatever follows.

---

## 7. What is reported

Per cell: family, target expression, affected joints, severity, floor that bounded it,
walking success (per seed and pooled), drop vs nominal (per seed and pooled), every
secondary metric of section 2.4, and the qualification verdict against each of the five
rules of section 5.

Per family: the dose-response curve of walking success against severity, and the
monotonicity check.

Plus: the selected intervention and which rule selected it; every family that failed and
which rule it failed; the held-out severities; and, if nothing qualifies, the stop.

Failed and invalid cells are kept and labelled, never deleted.

---

## 8. What this study does not do

* It does not retrain, resume or re-export W2.
* It does not calibrate the monitor or build a conditioner.
* It does not reintroduce the stand-derived soft rate limiter into the walking path.
* It does not lower `MIN_SCALE` or `MIN_SCALE_MULTI`.
* It does not run on hardware. Hardware is BLOCKED: rechecked 2026-09-22, no
  `192.168.123.0/24` interface exists on the workstation (interfaces `eno1`
  192.168.8.189/24, `wlp9s0`), the robot subnet routes to the default gateway, and none of
  .161 / .18 / .15 answer. Everything here is labelled SIM VERIFIED at best.
* It does not claim that a software gain reduction is motor damage.
