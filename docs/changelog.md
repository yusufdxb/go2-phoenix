# Changelog

Release-style summary of past Phoenix milestones.

## 2026-05-17: sweep + harness system

Added a 12-cell sim-side benchmark sweep over friction range, lateral push,
and action-rate weight to stress-test the stand-v3 baseline. Added a
four-script lab-day harness (preflight, recorder, diversity validator, EOD
rsync).

- `scripts/sweep_run.py`, `configs/train/sweep_stand_v3_stress.yaml`
- `scripts/harness_{preflight,record,diversity,eod}.{sh,py}`
- `docs/sweep_design_2026-05-17.md`
- +28 tests, full no-sim suite at 228 passed.

No `src/phoenix/*` changes; no policy retrain.

## 2026-05-17: H0 bridge fix

`reset_bridge` made configurable: opt-in velocity write plus a selectable
seed-row strategy (first / explicit / last_failure_minus_k). Defaults
preserve legacy behavior bit-for-bit. Closes a synth-distribution mismatch
where the bridge silently dropped initial forward velocity and seeded from
row 0 even when the failure label fired at row 60+.

- `src/phoenix/adaptation/reset_bridge.py`, `tests/test_reset_bridge.py`
- +6 tests, suite 189 passed.

## 2026-04-21: stand-v3 retrain

A live hardware Gate 7 attempt on stand-v3 saturated the per-step slew clip
at 33% on the rear thighs. Two coupled causes were diagnosed (a stand-posture
offset plus out-of-distribution policy output). v3 attacks the latter via a
4x action_rate plus 5x joint_acc penalty. Sim slew dropped to 0.33% at cmd=0.

## 2026-04-19: two-policy mode switch shipped

The single-policy v3b replacement path was exhausted across four retrain
attempts (`flat-scratch`, `flat-v3b-ft`, `flat-slewhinge`,
`flat-slewhinge-w5`). Root cause: reward-landscape dominance, not init
conditioning. Pivoted to a runtime mode switch: `stand-v2` and `v3b` loaded
together, hysteresis plus a 25-tick blend, routed on `cmd_vel` magnitude.
Opt-in flag, zero retraining, 179 unit tests green. See
`docs/deploy_mode_switch_runbook.md` for how to flip it on.

## 2026-04-18: first live hardware dryrun

`sim2real.ros2_policy_node` ran end-to-end on a live GO2. The per-step slew
clip saturated at 30.23% specifically when `cmd_vel = (0, 0, 0)`. Root cause:
the rough-v0 baseline was trained on 235-dim obs (proprioception plus height
scan) and zero-padded at deploy, so the policy could not respect the slew
cap. A flat-v0 retrain in `ppo_flat.yaml` was the next attempt.

## 2026-04-17: pre-lab gates cleared for phoenix-stand

| Gate | Metric | Result |
|---|---|---:|
| 0a, sim rollout | success at 20.0 s mean length | 16 / 16 |
| 0b, ONNX staging | hashes match deploy path | pass |
| 0c, verify_deploy parity | max torch / ort abs-diff | 3.8e-06 (26x under 1e-4 tol) |

## 2026-04-14: baseline + warm-start adaptation result

| Policy | Terrain | Mean return | Success | Episodes |
|---|---|---:|---:|---:|
| rough-v0 baseline (500 iters) | rough | 18.95 | 100% | 16 |
| phoenix-base | slippery | 15.90 | 90.6% | 64 |
| phoenix-adapt | slippery | 16.64 | 100% | 64 |
| phoenix-adapt | rough | 17.56 | 96.9% | 64 |

The adaptation is plain warm-start PPO on the slippery overlay, not a
failure-curriculum result (`adaptation.yaml` ships with
`failure_sample_fraction: 0.0` until a hardware-captured parquet exists).
