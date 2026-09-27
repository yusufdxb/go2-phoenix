# Actuator-gain robustness: clean model_2750 vs model_1750 (2026-09-27, simulation only)

## Question
Did `walk_vnext3_seed47` model_1750 (trained under the compounding motor-gain DR bug, see
`training_validity.md`) gain useful robustness from that corrupted, widening gain distribution, or does the
clean `walk_vnext3c_seed47` model_2750 (stationary DR U(0.85, 1.15)) remain equal or better under realistic
actuator variation?

## Design (fixed before results)
Evaluation only; no training, no hardware. Actuator "gain" = the DC-motor PD stiffness and damping.

| condition | Isaac (`eval_velocity.py --gain`) | MuJoCo (`sim2sim_robustness.py`) |
|---|---|---|
| nominal | pattern:nominal | pattern nominal |
| intended | range:0.85:1.15 (per env, per joint, per reset, kp/kd independent) | 5 seeded per-joint draws |
| extrapolation | range:0.7:1.3 | 5 seeded per-joint draws |
| weak rear | RL/RR joints x0.75 | same |
| weak calves | calves x0.75 | same |
| left/right | left x0.85, right x1.15 | same |
| all weak | all joints x0.7 | same |

- Isaac: 256 envs, seeds 1-3, for model_2750, model_1750 and the control (walk_vnext_seed47 model_1000,
  needed by criterion 4) under every condition (63 evals). MuJoCo: gate v3 + still-stance cases.
- Frozen five criteria: the unmodified checker (`logs/vnext2/check_criteria.py`) per condition x seed,
  criterion 4 against the control under the same condition and seed; for ranges Isaac seed s is paired with
  MuJoCo draw s-1, and MuJoCo gate / C1 are also reported over all 5 draws. Thresholds unchanged.
- Validation: all 63 Isaac and 30 MuJoCo outputs complete; realized per-joint gains match every requested
  condition exactly (patterns) or span the range (ranges); MuJoCo nominal reproduces the stored gate reports
  and cold results bit-for-bit; an Isaac rerun of the same seed/condition was bit-identical.

## Results

Frozen contract (seeds passing / 3):

| condition | model_2750 | model_1750 |
|---|---|---|
| nominal | 3/3 | 3/3 |
| intended U(0.85, 1.15) | 3/3 | 3/3 |
| extrapolation U(0.7, 1.3) | 3/3 | 3/3 |
| weak rear x0.75 | 0/3 (C4) | 3/3 |
| weak calves x0.75 | 0/3 (C4, C5) | 3/3 |
| left/right 0.85/1.15 | 2/3 (C4) | 3/3 |
| all weak x0.7 | 0/3 (C1, C2, C4, C5) | 3/3 |

MuJoCo gate v3: both PASS nominal, 5/5 intended draws, 5/5 extrapolation draws, weak rear, left/right.
model_2750 FAILS weak calves and all weak (`yaw_0p6:no_fall`); model_1750 passes both.

Isaac, falls: zero for both models in nominal, intended, extrapolation, weak rear (2750: 0.13% one seed),
left/right. model_2750 falls under weak calves (0.13%) and all weak (0.9% overall, up to 3.9% of envs on
right-lateral vy -0.3, up to 1.2% on CCW yaw); model_1750 zero everywhere. No trunk contacts, no numerical
failures, all episode ends time-outs in the random-command eval.

In-distribution differences (Isaac, mean of 3 seeds, intended range): model_2750 lower action |a| p99
(1.67 vs 1.87), better backward -0.3 (0.956 vs 0.925), lateral +0.3 lower (0.924 vs 0.972), forward 0.3
lower (0.971 vs 0.999); still-stance starts 1.00 for both.

## Failure mode (model_2750, out-of-distribution weak calves)
MuJoCo trace (yaw_0p6, calves x0.75): calf target-minus-position error grows (mean 0.28 rad vs 0.14-0.23
nominal, max 0.60-0.67 rad), the trunk sags from 0.32 m to 0.20 m during the counter-clockwise turn, and the
base-height fall check trips at 11.4 s; no torque saturation, no joint-limit contact. Sweeps: model_2750
survives CCW turning (30 s) at calves x0.80 and above and all joints x0.75 and above; fails at calves x0.75 /
all x0.70 and below. Clockwise turning, forward 0.5 and standing survive calves x0.70. model_1750 survives
every case down to x0.60. Isaac shows the same asymmetry (CCW-yaw and right-lateral falls, calves near the
hard limit during CCW turns). Consistent with Phoenix's earlier CCW-yaw calf sensitivity.

## Conclusions
1. Within the intended actuator variation (U(0.85, 1.15)) and even random per-joint variation to
   U(0.7, 1.3), the two models are equivalent on the frozen contract, gate v3 and falls; model_2750 is
   better on backward tracking and action magnitude.
2. model_1750 has a REAL, reproducible robustness advantage (both simulators, deterministic, several
   patterns) that is confined to structured, correlated gain loss beyond the training range: >= 20-25%
   weaker subsets of motors (rear legs, calves, one side) or all motors 30% weaker. Most plausible cause:
   its training plant, through the compounding bug, contained many envs with gains down to ~0.4x, i.e. an
   accidentally much wider (not better-shaped) gain distribution.
3. model_2750 remains the designated simulation candidate (valid plant, in-distribution equal or better).
   model_1750 should NOT be discarded: it stays a hardware candidate / fallback until real GO2 actuator
   variation is known.

## Limitations
One training seed per model; 3 Isaac eval seeds; fixed patterns are chosen stress cases, not measured GO2
degradation; "gain" perturbation models PD/torque-constant mismatch, not torque-limit (battery) sag; MuJoCo
single-robot results are deterministic per draw; simulation only.

## Blockers before hardware
- Characterize real GO2 actuator variation (per-joint torque/PD tracking, temperature and battery effects)
  to decide whether >= 20-25% correlated weakness is in scope; that decides 2750 vs 1750.
- model_2750's CCW-turn weak-calf sensitivity: if in scope, it needs a training fix (e.g. wider stationary
  DR or correlated per-leg scaling), a new preregistered run, not a threshold change.
- Single clean training seed; criterion-4 forward-tracking near-misses.
- Deploy branch: the velocity ROS node must pass `DeployManifest.obs_schema_version` to ControllerFSM (not
  yet wired); hardware lab card and operator ladder not yet run for this policy.
