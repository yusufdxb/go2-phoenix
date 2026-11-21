# 02 — Audit: friction-DR training study and gantry A/B campaign (2026-10-12)

Offline only. No training was started; mewtwo checkpoints, sources and logs are unchanged.
Re-evaluation outputs: mewtwo `~/workspace/friction-validation-repro/dr_robustness_20261012/`,
copied to `~/Projects/phoenix-friction-data/dr_robustness_20261012/` (`robustness_table.txt`, `agg.py`).

## VERIFIED

### The study as registered
Location: mewtwo `~/workspace/wt/phoenix-walk-v1-hw` (branch `feat/phoenix-walk-v1-hw` @ `06c5a37`,
unpushed). Preregistration `docs/preregistration/2026-10-07-joint-friction-dr.md` (`3e8df13`,
amendment 1 `8e11852`), outcome note `…joint-friction-dr.OUTCOME.md`.

- Arms: **T** = vnext3f, i.e. vnext3 + joint friction U(0.1, 0.6) N·m hips/thighs and U(0.3, 1.5) N·m calves
  (static = dynamic); **C** = vnext3, dose-matched; both a 1000-iteration faithful resume of
  model_2750 → `model_3749.pt`, seeds 50–54. **I** = incumbent model_2750 (`e03013d1…`).
- Evaluation: `scripts/eval_friction_dr.py` → MuJoCo `sim_replay_walk.py` under `nominal`, `fmeas=1.0/1.25/1.5` only.
- Stored result (`results/friction_dr/analysis.json`): E1 (yaw at fmeas=1.25) mean paired
  d = +0.225, CI [0.139, 0.318]; E2 (forward) +0.208, CI [0.167, 0.247]; both p = 1/32. That's the
  smallest p possible with 5 pairs.
- Promotion guard (`promotion_guard.json`): **FAIL for all 5 T seeds.** Under frictionless
  replay, T differs from its paired C by more than 0.10 in forward on every seed and in yaw on seeds 50 and 51.
  No checkpoint is promoted. `fric_T_s50` is labelled exploratory and gantry-only.
- Recompute: re-running the replay for all 11 policies reproduces the stored fmeas=1.25 yaw
  for T s50–54 exactly (0.7400, 0.6038, 0.6452, 0.6338, 0.6376).

### New: robustness to physics outside the training family
Same replay and presets; each policy uses its own `results/friction_dr/<label>/deploy.yaml`. One deterministic
run per cell. Values are fraction of command (forward `vx03_4s`, yaw `yaw05_3s`); T and C are means over 5 seeds.

| condition | I fwd/yaw | T fwd/yaw | C fwd/yaw | Δyaw T−C (T>C) | Δfwd T−C (T>C) |
|---|---|---|---|---|---|
| nominal (frictionless) | 0.89/0.76 | **1.06**/0.93 | 0.89/0.79 | +0.14 (5/5) | +0.16 (5/5) |
| fmeas=1.25 (training family) | 0.60/0.27 | 0.80/0.65 | 0.60/0.43 | +0.23 (5/5) | +0.21 (5/5) |
| drag=30, spring=5 (rope proxy) | 0.59/0.26 | 0.73/0.38 | 0.62/0.28 | +0.10 (5/5) | +0.11 (5/5) |
| drag=25 | 0.63/0.73 | 0.77/0.92 | 0.65/0.80 | +0.12 | +0.12 |
| spring=4 | 0.89/0.29 | 1.06/0.46 | 0.89/0.35 | +0.11 | +0.17 |
| calf=0.8 (weak calf PD) | 0.67/0.64 | 0.90/0.87 | 0.70/0.65 | +0.22 | +0.19 |
| calf=0.8, spring=4 | 0.67/0.19 | 0.90/0.40 | 0.70/0.25 | +0.14 (4/5) | +0.19 |
| all=0.85 | 0.83/0.69 | 1.02/0.91 | 0.83/0.77 | +0.14 | +0.19 |
| lat=20 ms | 0.86/0.73 | 1.02/0.93 | 0.87/0.75 | +0.18 | +0.15 |
| foot fric=0.3 | 0.93/0.88 | 1.11/0.97 | 0.97/0.84 | +0.13 | +0.14 |
| jdamp=0.3 | 0.77/0.69 | 0.94/0.87 | 0.77/0.72 | +0.14 | +0.17 |
| unload=0.2 | 0.89/0.85 | 1.05/0.94 | 0.92/0.83 | +0.11 | +0.14 |

No falls in any of the 11 × 12 cells.

### Gantry A/B campaign (`2026-10-07-gantry-ab-exploratory.md`, model_2750 vs fric_T_s50)
- "Not ready" (`cb6180d`): the analyzer's STOP-drill and unreported-abort detection was defective on
  real ticks. A deadman release appears only as the transition from the last policy tick to a `hold`
  tick with `policy_abort:external_estop`; the bridge never logs estop=true on a policy tick. Two
  earlier fixes (amendments 2 and 3) failed.
- Amendment 4 (`a3118dc`, `d029653`, disclosure `a8aaec7`) rewrote the detection from all 16 real
  bridge logs. `tests/test_gantry_ab.py`: **30 passed** (re-run 2026-10-12). No later document declares
  the campaign ready, no campaign data exists, and later commits moved to the incumbent-only
  FREE_FLOOR path (`158ecea`).

## INFERRED
- **Most of T's advantage is generic, not friction-specific.** T beats C by about +0.10 to +0.23 under every
  condition, including frictionless replay, where T overshoots forward (1.06). The friction-specific
  increment (fmeas=1.25 minus nominal) is about +0.09 yaw and +0.05 forward. A weak calf actuator earns T
  about the same benefit as friction (+0.22 yaw), and the rope proxy earns less (+0.10).
- So a hardware A/B in which T tracks better **wouldn't identify friction as the cause**. T is
  predicted to track better under friction, harness forces and actuator weakness alike, and it
  over-tracks (about 106 %) if nothing is wrong.
- The promotion guard failing is substantive, not a technicality: it flags exactly this
  nominal-behaviour change.
- The A/B campaign runs on the gantry, so it inherits the harness confound.

## HYPOTHESES (untested)
- T's generic gain reflects a learned bias toward larger actions or gait frequency. Not inspected
  (would need action and joint traces from the replay outputs).

## BLOCKED / NO-GO
- **Physical A/B with fric_T_s50: NO-GO.** Its own promotion guards fail; it's preregistered as
  exploratory and gantry-only; FREE_FLOOR code denies its hashes explicitly; and it can't discriminate
  the competing explanations.
- **Retraining:** not proposed. No identified physics model justifies it yet.
