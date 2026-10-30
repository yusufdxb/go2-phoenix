# Preregistration: why actuator Model B over-sticks (2026-10-30)

Committed before any v2 probe recording and before the simulated dry run. Constants live in
`configs/sim2real/actuator_probe_v2_prereg.yaml`; `scripts/probe_eval.py` reads them from there.
"Model C" below is the actuator model, unrelated to the A/B/C locomotion campaign.

## Why

The A/B actuator evaluation (2026-10-30, `docs/hardware/2026-10-30_actuator_friction_model_B/`)
accepted Model B with limits: B tracks better than A but sticks longer than the robot. B's
fitted Coulomb friction (`actuator_model_B_frozen.yaml`: hip 0.522, thigh 0.319, calf 1.15 N m)
is 1.7x, 1.1x and 1.9x the medians of the static friction measured on 2026-10-07
(`configs/physics/go2_joint_friction_2026-10-07.yaml`). Two explanations compete:
(a) the fit inflated F, because a position-RMSE objective with 2 s state resets rewards
holding still; (b) a single Coulomb value (Fs = Fc) cannot produce the transition, so any
F that fits position must over-stick somewhere. They predict different things on a probe
built to measure the transition, which the old FP probe (one slow ramp per direction) was not.

## Claim (primary, one sentence)

On the VAL run of the v2 probe, replaying with the measured per-joint static friction (Bm)
instead of B's fitted friction cuts the median absolute breakaway-threshold error by >= 30 %
in at least 2 of the 3 joint families, with the 12-joint paired sign-flip test at p < 0.05.

Comparator: B (frozen). If the claim fails, explanation (a) is not supported, and that
result is reported as such.

## Arms

| Arm | Purpose | Definition |
|---|---|---|
| Bm (treatment) | friction magnitude taken from physical measurement, no trajectory fit | `actuator_model_B_measured.yaml`, per joint, B0 structure, B's constraint settings |
| B (comparator) | the fitted magnitude under test | `actuator_model_B_frozen.yaml`, unchanged |
| A (no-friction reference) | floor; shows the probe resolves friction at all | DCMotor PD only |
| Bm_fam (bundling control, secondary) | Bm differs from B in magnitude AND per-joint granularity; this arm keeps granularity per family | family medians of Bm: 0.2995 / 0.290 / 0.6145 N m |
| Oracle B0 (ceiling, never reported as validation) | best any constant per-joint Coulomb can do on these traces | per-joint F fitted on VAL itself (in-sample by construction) |
| Positive control P1 (magnitude) | the pipeline must detect a known magnitude error | simulated v2 probe with Bm friction as "hardware"; replaying it, Bm must score median |threshold error| <= 0.025 N m (one slow-rate quantum) and H_mag must come out supported for B |
| Positive control P2 (slip detector) | a null on micro-slip must mean "no slips", not "blind detector" | unit test: a 1-DOF Karnopp plant (Fs = 1.5 Fc) under the v2 PD yields micro-slip events; a pure Coulomb plant and a noise-only trace at the hardware settle noise yield ~0 |
| Sham / dose | not applicable | no detector or trigger decides when to act; every model replays identical excursions |

P1 and P2 must pass before the hardware run. If P1 fails, the eval script is broken and
the run waits.

## Protocol (frozen)

v2 probe, `src/phoenix/sim2real/friction_probe.py::RateProbeSequencer`. Hung, unloaded robot,
all 12 joints in sequence. Stance pose q0 via `PHOENIX_FP_STANDUP_S`. Gains kp 25, kd 0.5 on
every joint. Per joint: settle 2 s, then 2 repeats of
[+0.05, -0.05, +0.20, -0.20] rad/s excursions. Each excursion: rise until 0.5 s past
breakaway (probe rule: 20 mrad from the pre-window median) or the 0.30 rad cap (limit
margin 0.05 rad), hold 1.0 s, fall back to q0 at the same rate, hold at q0 for 2.0 s.
Existing abort, e-stop and deadman paths are unchanged. Telemetry: probe npz (t, q, dq,
target, tau_est, kp, kd, phase, rate, excursion) plus bridge.jsonl (IMU rpy/quaternion,
battery voltage, motor temperature, lowstate tick).

Split: CAL = the first complete 12-joint run, VAL = a second complete run, same session.
VAL is the primary analysis set and is read exactly once for Model C. Any fitting (Model C)
uses CAL only. VAL is a same-session holdout, not a cross-day one; that limit is reported.

## Replay

1 ms MuJoCo steps on the fixed-base GO2 (`isaac_matched`), logged targets held between 50 Hz
ticks, logged kp/kd, zero delay. Each excursion is initialised from the measured q, dq of all
12 joints at its start and runs to the end of its zero dwell with no reset. Gravity is tilted
by the run-mean bridge roll/pitch. Same code path for A, B, Bm, Bm_fam.

## Metrics (per excursion, probed joint, 50 Hz samples)

1. Position RMSE [rad] over the whole excursion.
2. Velocity RMSE [rad/s]: sim qvel vs logged dq.
3. Breakaway-threshold error: `breakaway()` from `phoenix.sim2real.actuator_replay` (first tick
   3 mrad past the window start in the motion direction; tracking error then), applied to the
   rise (motion direction) and the fall (return breakaway, opposite direction). Error =
   kp (e_model - e_hw) [N m].
4. Breakaway-timing error: t_model - t_hw [s] from the window start.
5. Missed breakaway: hw breaks inside the window, model does not. 6. False breakaway: the
   reverse. Both as a fraction of windows.
7. Direction-dependent error: mean signed threshold error of + windows minus - windows.
8. Micro-slip: on per-tick |dq| (50 Hz differences) a moving tick is >= 0.5 mrad; an event is
   a maximal same-sign run of moving ticks with >= 3 still ticks on both sides; a micro-slip
   has 2 <= |displacement| < 20 mrad. Frequency = events per excursion, displacement = mean
   |event|. Recall = fraction of hw events matched by a model event of the same sign starting
   within 0.10 s. Noise floor = the same detector on the probed joint's settle phase.
9. Holding error: |mean(q_model) - mean(q_hw)| over the last 1.0 s of each zero dwell [mrad];
   hw offset from q0 reported alongside.

The 20 mrad breakaway rule is a robustness re-run of 3 to 6, not a second primary.
Aggregation: per joint = median over its VAL windows; per family = median of its 4 joints.

## Confound answers

- A1 onset: report the fraction of breakaways in the first 2 ticks of a window. If above 10 %,
  the threshold metric is measuring the initial-state offset and is reported as such.
- A2 exposure: every model replays the same excursions, windows and dwell lengths; a window
  is dropped for all models or none (an aborted excursion is dropped).
- A3 replication: the simulation is deterministic, so the unit is the joint (n = 12, 4 per
  family). The exact two-sided sign-flip floor is 2/4096 at n = 12 but 0.125 at n = 4, so
  family-level results are effect sizes only, never p-values.
- A4 bundling: B vs Bm differ in magnitude and in granularity; Bm_fam separates the two.
  Structure, constraint, gains, dt, init and gravity are identical.
- A6 outcome: the outcome is simulator fidelity of the stick-to-slide transition and
  position tracking, the thing this study is about. Effect on locomotion transfer is out
  of scope (no Isaac Lab, no PPO).
- A8 floor: a 50 Hz tick at 0.20 rad/s is 4 mrad = 0.10 N m at kp 25, at 0.05 rad/s
  0.025 N m. Hip friction is ~0.3 N m, so fast-rate thresholds are coarse; H_vel requires
  a difference above 0.10 N m.
- A9 selection: every threshold above is fixed in this file before data. One amendment is
  allowed after P1/P2 and the in-sample 10-07 pipeline check, dated, before any v2 hardware
  data; none after.
- A10 leakage: B and Bm both come from the 2026-10-07 sessions, so neither is fitted on v2
  data. Model C fits on CAL only. The oracle arm is fitted on VAL and is labelled oracle.
  Motor temperature and battery voltage drift since 10-07 is a known confound (recorded,
  exploratory).

## Decision rules (Phase 4)

- H_mag, excessive fitted friction (primary): the claim above. If Bm_fam also passes,
  magnitude, not granularity, carries it.
- H_inad, constant friction inadequate: B and Bm both reach < 25 % pooled micro-slip
  recall, AND the hw micro-slip rate is at least 2x its settle noise floor. If the hw rate
  is below 2x the floor, the result is "no resolvable micro-slip" and H_inad is undetermined.
- H_vel, velocity dependence: for >= 2 families the hw fast vs slow breakaway threshold
  differs by more than max(15 %, 2 repeat SDs, 0.10 N m).
- H_dir, direction or loading: a + vs - model error difference > 15 % with the same sign in
  every VAL repeat and rate is flagged and investigated (gravity load at q0) before any model
  change. Loading effects are exploratory only.
- No Stribeck, backlash or neural residual by default.

## Model C (Phase 5), smallest supported first

1. C1: per-joint Coulomb from CAL breakaway thresholds (median kp e_hw over CAL windows),
   B0 structure. Always built.
2. C2: Fs != Fc (Karnopp in qfrc_applied), only if H_inad holds and C1 fails the slip rule.
3. C3: C1 plus dof damping fitted on CAL slide phases, only if H_vel holds.

The chosen C is evaluated once on VAL. ACCEPT it if on VAL: median |threshold error| <= 25 %
of the hw median threshold in every family; missed + false <= 10 % of windows; position
RMSE <= 1.05x the better of B and Bm per family; micro-slip recall >= 50 % when the hw rate
is >= 0.1 per excursion.

Overall recommendation: ACCEPT B as-is if B meets the Model C acceptance rule; MODIFY to the
first of Bm / C1 / C2 / C3 that meets it; REJECT the constant-friction family if none does and
H_inad holds.

## Kill criterion and stopping rule

- Primary dead if the 12-joint sign-flip p >= 0.05 or fewer than 2 families reach the 30 % cut.
  Reported as a null, with the oracle B0 bound.
- The run is unusable for thresholds if fewer than 75 % of hw windows show a breakaway. Then
  stop and report.
- A VAL run with fewer than 12 complete joints is repeated once for the missing joints only.
- No physical command before the operator is present and authorises the probe batch.

## Amendment 1 (2026-10-30, the single allowed amendment)

Written after P1, P2 and the in-sample 10-07 pipeline check, before any v2 hardware data.
The text above stays as registered; where it conflicts with this section, this section rules.
No amendment follows this one.

1. Command stream. Replay targets come from the bridge's logged command stream
   (`bridge.jsonl`); the probe npz targets are the fallback only when no bridge log exists.
2. Window. A breakaway window is a ramp plus the dwell that follows it (rise + dwell_peak,
   fall + dwell_zero), so a breakaway that happens after the ramp stops is still inside it.
3. C1. As registered, C1 was the median of kp e_hw over CAL windows. On P1 that closed form
   is biased by about +0.09 N m against the known plant friction (it reads the tracking error
   at the first moving tick, which already includes PD lag). C1 is now the per-joint F that
   minimises the CAL breakaway-threshold error on a grid of replays (B0 structure, same
   grid as the oracle arm). The closed form is reported as a diagnostic only.
4. 10-07 grouping. The in-sample check on the 2026-10-07 v1 recordings maps v1 phases to v2
   ones (rest = zero dwell, one rate, 0.05 rad/s). That check is a pipeline check, never
   evidence for or against any hypothesis.
5. P2. As registered: "a 1-DOF Karnopp plant (Fs = 1.5 Fc) under the v2 PD yields micro-slip
   events". That failed: with the GO2 hip/thigh inertia (about 0.022 kg m^2 at q0) and kd 0.5,
   Fs = 1.5 Fc breaks away once and then slides. P2 is now (a) planted-event recall: five
   stick-slip steps of 3 to 15 mrad on a stuck joint with the 10-07 settle noise, all five
   found and nothing else; (b) a pure Coulomb plant and noise-only traces (3e-5 and 3e-4 rad)
   yield no events; (c) a Karnopp characterisation kept as assertions: Fs = 2 Fc at hip/thigh
   inertia on a slow ramp yields >= 5 same-direction micro-slips, the same at calf inertia
   (0.005 kg m^2) yields none. A micro-slip null on hardware is read against (c).
6. Tilt. Gravity tilt is the run-mean bridge `sensors.rpy` roll/pitch (ASSUMPTION: radians).
   The 10-07 bridge logged no sensors, so those runs replay level.
7. H_vel. As registered, H_vel compared raw hw fast vs slow thresholds. On P1, whose plant
   is constant Coulomb by construction, that rule came out supported: fast minus slow was
   +0.150 N m (hip) and +0.143 N m (thigh) against a 0.10 N m bar, from PD lag and the 50 Hz
   quantum, not from friction. H_vel is now model-relative: the hw fast minus slow shift
   minus the same shift in the model's replay must pass the registered bar in >= 2 families,
   for both B and Bm. The raw comparison is reported alongside.
8. Gaps. An excursion with an internal sample gap above 0.10 s is dropped for every model.
9. Exclusions. Excursions with a documented physical reason (joint at its limit, a splayed
   leg, ground contact) are dropped for every model through `probe_eval.py --exclude
   RUN:JOINT`, and each exclusion is listed with its reason in the report. The 10-07 check
   applies the source README's exclusions; it also showed why: run5 (front feet on the floor)
   had front calf thresholds of 1.9 to 2.3 N m, against 0.4 to 0.9 N m on the hung run6.
10. Feet off the floor. Before each run the operator confirms by eye that all four feet are
    clear, and the lab card records the bridge `foot_force` values at the hung pose. The
    report prints per-run foot_force median and max per leg. A leg whose foot_force rises
    clearly above its hung baseline during a run is checked against the operator's notes;
    a confirmed contact excludes that leg's joints in that run under item 9. (The foot_force
    scale on a hung GO2 is not characterised, so no numeric trigger is registered.)
11. Known pose limit. Seven of the twelve Bm values (all hips and thighs except RL_hip) were
    measured on 2026-10-07 in the folded hung pose; v2 runs every joint at the stance q0
    (`PHOENIX_FP_STANDUP_S=2.0`, as in 10-07 run6). A Bm threshold error on those seven
    joints may be load or pose dependence rather than magnitude, and is read under H_dir
    (exploratory), not as evidence against H_mag's mechanism.
