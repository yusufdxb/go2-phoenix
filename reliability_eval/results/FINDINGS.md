# Reliability Shield, Phase 3 (Isaac twin) Findings

Policy: `checkpoints/phoenix-flat/latest.pt` (flat-terrain velocity-tracking GO2, rsl_rl PPO,
actor [512,256,128] ELU). Sim: Isaac-Velocity-Flat-Unitree-Go2-v0, Isaac Lab 4.5.22.
Grid: 7 conditions x 3 seeds, 128 envs x 400 steps (8 s) @ 50 Hz. Per-file checkpoint SHA-256
and library versions in `raw/*.meta.json`. Failure oracle = env base-contact termination that
is NOT a timeout (`done & ~time_out`), independent of any monitor score.

## Held-out OOD design (strictly outside training DR)
Training randomized friction [0.3,1.5] and base mass +/-2 kg; motor strength was never
randomized. Shifts pinned beyond that: friction {0.2, 0.06}, added mass {+4, +7 kg},
motor strength {0.6, 0.45}. Nominal is a clean fixed condition (friction 0.9, no added
mass, no push).

## Behavioral fall rate (env oracle, mean [95% CI] over seeds)
| condition | fall rate |
|---|---|
| nominal | 0.154 [0.123, 0.185] |
| friction_moderate | 0.307 [0.264, 0.351] |
| friction_severe | 0.521 [0.507, 0.534] |
| mass_moderate | 0.047 |
| mass_severe | 0.026 |
| motor_moderate | 0.003 |
| motor_severe | 0.232 [0.160, 0.304] |

Only low friction and severe motor-weakening reliably *induce falls*; the policy tolerates
added mass and mild motor loss (fall rate at/below nominal). Lead-time claims are therefore
strongest on friction (the failure-inducing shift).

## Finding 1, Detection works and beats every baseline (AUROC, mean [95% CI])
| condition | latent-Maha (shield) | obs-Maha | obs-magnitude | value-signal | action-sat | random |
|---|---|---|---|---|---|---|
| friction_moderate | **0.957** [.950,.965] | 0.876 | 0.535 | 0.287 | 0.511 | 0.500 |
| friction_severe | **0.980** [.974,.986] | 0.929 | 0.592 | 0.278 | 0.505 | 0.499 |
| mass_moderate | **0.797** [.772,.822] | 0.567 | 0.486 | 0.448 | 0.550 | 0.500 |
| mass_severe | **0.893** [.884,.903] | 0.698 | 0.515 | 0.483 | 0.561 | 0.500 |
| motor_moderate | **0.912** [.909,.916] | 0.771 | 0.526 | 0.588 | 0.569 | 0.500 |
| motor_severe | **0.963** [.954,.972] | 0.886 | 0.609 | 0.619 | 0.556 | 0.500 |

Policy-latent OOD scoring separates held-out dynamics shifts from nominal far better than
observation magnitude, the critic value signal, and action saturation (all near chance;
value is *below* chance on friction), and beats observation-space Mahalanobis on every shift.
This is the core defensible result. The internal-state signal carries information the raw
observation and the policy's own value head do not surface.

## Finding 2, Naive frame-level calibration is NOT actionable
Calibrating the Simplex trip at a 1% *frame* FPR engages the shield on **100% of nominal
episodes**. Latent scores are temporally autocorrelated, so 1% per-frame false positives
arrive in runs and trip the arbiter within essentially every 400-frame episode. At that
operating point the "warn before failure" and "intervention" numbers are meaningless
(the shield is always on).

## Finding 3, Episode-level calibration IS actionable (the principled fix)
Recalibrating the threshold at the episode level (see `sweep.json`), on friction_severe:

| threshold %ile | persistence K | nominal-episode FPR | falls warned | median lead (s) |
|---|---|---|---|---|
| 99.9 | 3 | 0.042 | 1.000 | 0.73 |
| 99.95 | 5 | **0.004** | **0.987** | **0.72** |
| 99.99 | 3 | 0.004 | 0.958 | 0.70 |
| 99.99 | 5 | 0.000 | 0.755 | 0.64 |

At p=99.95 / K=5: **0.4% of nominal episodes false-alarm, 98.7% of friction-induced falls
warned before onset, ~0.72 s (about 36 control steps) median lead**, enough head-room for a
Simplex handoff. The warning is real and early once the operating point controls
episode-level, not frame-level, false alarms.

## Verdict
Policy-latent OOD scoring provides useful, statistically-defensible early warning of
behavioral failure and outperforms observation magnitude, value signal, action saturation,
and observation-space Mahalanobis. The actionable-shield claim holds ONLY under episode-level
FPR calibration; the naive frame-level operating point is not deployable. Smallest principled
redesign already validated: calibrate the trip threshold to a target nominal *episode* FPR
(and use persistence K), not a per-frame FPR.

## Honest limitations
- Single policy (flat GO2, a mediocre walker: 15% nominal fall rate). Results should be
  reconfirmed on the stronger stand-v3 policy and on a better walker.
- Mass / mild-motor shifts rarely induce falls, so their lead-time estimates rest on few
  events; detection AUROC for them is still high (monitor sees the shift the policy absorbs).
- Sim only (Isaac twin). Real-GO2 confirmation is the August hardware step.
- Intervention is a counterfactual proxy (did the shield engage before onset?), not a
  re-simulation with the fallback controller in the loop.
- dt assumed 0.02 s (50 Hz) for lead-time; consistent with the flat env decimation.

## Reproduce
```
# rollouts (Isaac venv): scripts/reliability_run_grid.sh  (ENVS=128 STEPS=400 SEEDS="0 1 2")
# metrics (phoenix venv): PYTHONPATH=src .venv/bin/python scripts/reliability_eval.py
# operating-point sweep:  PYTHONPATH=src .venv/bin/python scripts/reliability_sweep.py
```
Raw arrays in `raw/*.npz` (+ `.meta.json` with checkpoint SHA-256 + versions).
