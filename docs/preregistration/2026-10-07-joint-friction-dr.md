# Preregistration: joint-friction DR fine-tune of model_2750 (2026-10-07)

Committed before any treatment or control training. SIMULATION evidence only; the
hardware walk decides whether any of this transfers.

## Why

On the gantry (2026-10-06) model_2750 tracked ~60 % of a 0.3 m/s forward command and
~20 % of a 0.5 rad/s yaw command (sim 89 % / 76 %). The hung-robot FP probe (2026-10-07,
`docs/hardware/2026-10-07_friction_probe/`) measured joint static friction of
0.17-0.35 N m (hips, thighs) and 0.42-0.96 N m (calves). The policy was trained with
zero joint friction. MuJoCo replay with the measured values x1.25-1.5 matches the robot.

## Claim (one sentence)

Under MuJoCo replay of the hardware walking presets with the measured GO2 joint friction
x1.25, fine-tuning model_2750 for 1000 iterations with joint-friction DR raises yaw
tracking (yaw05_3s) by >= 0.15 absolute, then forward tracking (vx03_4s) by >= 0.10
absolute, relative to the identical 1000-iteration fine-tune without friction DR.

## Arms

| Arm | Purpose | Definition |
|---|---|---|
| T (treatment) | the claim | `--spec vnext3f`: vnext3 + joint friction U(0.1, 0.6) N m hips/thighs, U(0.3, 1.5) N m calves, static = dynamic, per env and joint at startup |
| C (no-treatment, dose-matched) | isolates friction DR from "1000 more iterations" | `--spec vnext3`, everything else identical |
| I (incumbent, reference) | the deployed checkpoint | model_2750 (sha256 e03013d1...), evaluated, not retrained |
| Positive control P1 | the friction model acts in N m in Isaac | `scripts/isaac_friction_probe.py`, run 2026-10-07 before this file: 15/16 cells recovered within 1 %; FR_hip at 1.2 N m never broke away on the up ramp (0 moving samples, estimate undefined; above the hip training max 0.6). Strict probe verdict FAIL on that cell, recorded as such. |
| Positive control P2 | friction degrades tracking in the outcome simulator | I at fmeas=1.25 must have yaw05_3s fraction <= 0.60 (it is 0.34-0.47 at fmeas 1.0, 2026-10-07). If not, the design cannot show the effect and the study stops. |
| Sham / oracle | not required | no detector or trigger decides when to act; the intervention is a training distribution |

Both T and C: faithful resume of model_2750 (optimizer, adaptive LR, curriculum level 8
from `curriculum_final_state.json`), `--resume-continue-iterations`, `--max-iterations 3750`,
8192 envs (spec default), health guard on. Seeds 50, 51, 52, 53, 54 (fresh; 47-49 used
before; same seed replays bit-identically). Pairing = seed. Evaluated checkpoint =
`model_3750.pt` of every run, fixed in advance (no checkpoint selection).

## Outcome measurement

`scripts/desk/sim_replay_walk.py` (local, sha256 0a95d9d2e65bf5cd...), gate config
`configs/sim2sim/gate_v3.yaml` (sha256 edde53e1...), deploy config = the model_2750
walk config with the policy paths swapped to each run's export (kp 25, kd 0.5). The
replay is deterministic, so one value per checkpoint and condition.

- E1 (primary 1): `yaw_deg / 86` for preset yaw05_3s, condition `fmeas=1.25`.
- E2 (primary 2): `fwd_m / 1.2` for preset vx03_4s, condition `fmeas=1.25`.
- Secondary (exploratory, labelled so permanently): same at `fmeas=1.0` and `fmeas=1.5`,
  vx03_2s, steady_vx_frac / steady_wz_frac, calf sag, Isaac training reward.

## Confound checklist

- A1 onset: no detector; N/A.
- A2 exposure: every checkpoint runs the identical deterministic command profiles and duration.
- A3 seeds: n = 5 paired. One-sided exact sign-flip p floor = 1/32 = 0.031 < 0.05.
- A4 bundling: T and C differ only in the three friction DR fields
  (`tests/test_velocity_friction_dr.py::test_vnext3f_differs_from_vnext3_only_in_friction_dr`).
  Extra training is shared by both arms (C exists for this).
- A5 positive control: P1 (Isaac friction in N m) and P2 (friction hurts I in MuJoCo).
- A6 outcome: the outcome of interest is real-robot tracking on the gantry. MuJoCo replay
  is a PROXY in a second simulator (not the training simulator). Headline wording stays
  "in sim replay"; only the hardware walk can confirm.
- A7 dose: equal iterations, envs, optimizer state and curriculum start for T and C.
- A8 ceiling/floor: I is at 0.34-0.47 yaw at fmeas 1.0 and lower at 1.25, so no ceiling.
- A9 selection: endpoints, condition (x1.25, the middle of the 1.25-1.5 band that matched
  the robot), checkpoint (3750) and seeds fixed here before training. The training
  friction ranges were chosen from the hardware measurement, not from replay results.
- A10 leakage: the eval friction (measured x1.25) lies inside the training range by design
  (the point is to train on the plausible robot). It is not fit on replay outcomes. A
  held-out check is the secondary fmeas=1.5 (calf 1.44 N m, near the range edge).

## Analysis plan

- Per seed s: d_k(s) = E_k(T, s) - E_k(C, s).
- Fixed-sequence test (controls family-wise error without Holm): E1 first, then E2 only
  if E1 passes. Each: one-sided exact sign-flip permutation p over the 5 paired d,
  alpha 0.05, AND mean d >= the minimum effect (0.15 for E1, 0.10 for E2).
- Report mean d with a paired bootstrap 95 % CI (10,000 resamples of seeds) and every
  per-seed value. Also report I.

## Kill criterion (numeric)

If E1 fails (mean d < 0.15 or p > 0.05) the claim is dead, it is recorded as a null in
the repo and vault, and no T checkpoint is promoted on the basis of this study. If E1
passes and E2 fails, only the yaw claim stands.

## Promotion guards (for choosing a hardware candidate; not part of the claim)

A T checkpoint can be proposed for the gantry only if: no fall in any replay condition;
frictionless (`nominal`) replay within 0.10 of the paired C on vx03_4s and yaw05_3s;
sim2sim gate v3 PASS; no health-guard abort. Among T seeds passing all guards, the one
with the largest E1 + E2 is proposed (an engineering choice, labelled post hoc).

## Stopping rule

All 10 runs train to 3750 unless the health guard aborts; an aborted run is reported as
such and not replaced. No extra seeds are added after looking at results.

## Amendment 1 (2026-10-07 13:20Z, after the first run finished, before any T/C outcome was computed)

The final checkpoint of a 1000-iteration resume from 2750 is named `model_3749.pt`
(rsl_rl names the last save by the last iteration index; `latest.pt` points to it); no
`model_3750.pt` exists. "Evaluated checkpoint = model_3750" above means the FINAL
checkpoint of every run, i.e. `model_3749.pt`. No other checkpoint is evaluated. Only the
incumbent replay (P2) had been computed when this was found.
