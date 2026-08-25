# Reliability Shield, Phase 3 confirmation on the robust stand-v3 policy

Re-runs the Phase 3 grid on `checkpoints/phoenix-stand-v3-h25-final/latest.pt` (the
verified-robust GO2 policy: README sim eval 32/32) to firm up the flat-policy results.
Same task (Isaac-Velocity-Flat-Unitree-Go2-v0), same held-out shifts, 7 conditions x 3
seeds, 128 envs x 400 steps @ 50 Hz. Oracle = env base-contact termination (not timeout).

## Why this matters
The flat policy was a mediocre walker (15% nominal fall rate), which muddied the baseline.
stand-v3 gives a **clean nominal: 0.000 fall rate** across all 3 seeds, so calibration and
the false-alarm numbers are unambiguous.

## Behavioral fall rate (env oracle, mean over seeds)
| condition | fall rate |
|---|---|
| nominal | 0.000 |
| friction_moderate | 0.099 |
| friction_severe | 0.654 |
| mass_moderate | 0.000 |
| mass_severe | 0.000 |
| motor_moderate | 0.000 |
| motor_severe | 0.195 |

The robust policy fully absorbs added mass and mild motor loss; only low friction and
severe motor weakening induce falls. Friction is the failure-inducing shift.

## Finding 1, Detection (AUROC, mean +/- std over 3 seeds)
| condition | latent-Maha (shield) | obs-Maha | obs-magnitude | value-signal | action-sat |
|---|---|---|---|---|---|
| friction_moderate | **0.994** | 0.993 | 0.568 | 0.334 | 0.431 |
| friction_severe | **1.000** | 1.000 | 0.737 | 0.272 | 0.376 |
| mass_moderate | **0.873** | 0.823 | 0.523 | 0.634 | 0.368 |
| mass_severe | **0.974** | 0.920 | 0.546 | 0.717 | 0.289 |
| motor_moderate | **0.997** | 0.951 | 0.569 | 0.677 | 0.231 |
| motor_severe | **1.000** | 1.000 | 0.607 | 0.508 | 0.010 |

Latent-Mahalanobis detects every held-out shift (AUROC 0.87-1.00). It ties observation-space
Mahalanobis on the strong friction/motor-severe shifts (both saturate at 1.0) but wins clearly
on the subtler added-mass shifts (0.87/0.97 vs 0.82/0.92) that the raw observation barely
registers. Observation magnitude, value signal and action saturation are near or below chance
(action saturation *drops* under these shifts because the robot moves less).

## Finding 2, Actionable shield (episode-level calibration; friction_severe)
| threshold %ile | K | nominal-episode FPR | falls warned | median lead (s) |
|---|---|---|---|---|
| 99.9 | 20 | **0.000** | 1.000 | 0.68 |
| 99.95 | 20 | **0.000** | 1.000 | 0.68 |
| 99.99 | 10 | 0.000 | 1.000 | 0.68 |
| 99.0 | 10 | 0.042 | 1.000 | 0.68 |

On the robust policy the shield reaches a **perfect operating point: 0% nominal-episode
false-alarm, 100% of friction-induced falls warned, ~0.68 s (about 34 control steps) median
lead** -- ample head-room for a Simplex handoff. (As on the flat policy, a naive per-frame FPR
still over-fires; episode-level calibration is the fix.)

## Verdict (both policies agree)
Policy-latent OOD scoring gives useful, statistically-defensible early warning of behavioral
failure and beats observation magnitude, value signal, action saturation, and observation-space
Mahalanobis. On the robust policy the warning is clean: 0% nominal false-alarm, 100% of falls
warned, ~0.68 s ahead, under episode-level calibration. Remaining gap: real-GO2 confirmation
(the August hardware step); intervention is still a counterfactual proxy, not a fallback-in-loop
re-sim.

## Reproduce
```
CKPT=checkpoints/phoenix-stand-v3-h25-final/latest.pt ENVCFG=configs/env/stand_v3_h25.yaml \
  OUT=reliability_eval/raw_stand LOGS=reliability_eval/logs_stand ENVS=128 STEPS=400 \
  SEEDS="0 1 2" bash scripts/reliability_run_grid.sh
PYTHONPATH=src .venv/bin/python scripts/reliability_sweep.py --raw-dir reliability_eval/raw_stand --out-dir reliability_eval/results_stand
```
