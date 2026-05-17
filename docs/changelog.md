# Changelog

Release-style summary of past Phoenix milestones. For day-by-day
narratives + raw metrics, see `docs/sessions/`.

## 2026-05-17 — sweep + harness system (no lab)

Added a 12-cell sim-side benchmark sweep over friction range,
lateral push, and action-rate weight to stress-test the stand-v3
baseline before the next lab slot. Added a four-script lab-day
harness (preflight, recorder, diversity validator, EOD rsync) so
the next CaresLab session is execution-only.

- `scripts/sweep_run.py`, `configs/train/sweep_stand_v3_stress.yaml`
- `scripts/harness_{preflight,record,diversity,eod}.{sh,py}`
- `docs/sweep_design_2026-05-17.md`, `docs/LAB_CARD_NEXT.md`
- +28 tests, full no-sim suite at 228 passed.

No `src/phoenix/*` changes; no policy retrain.

## 2026-05-17 — H0 bridge fix

`reset_bridge` made configurable: opt-in velocity write + selectable
seed-row strategy (first/explicit/last_failure_minus_k). Defaults
preserve legacy behavior bit-for-bit. Closes a synth-distribution
mismatch where the bridge silently dropped initial forward velocity
and seeded from row 0 even when the failure label fired at row 60+.
See `Daily Claude Logs/2026-05-17.md` for the spike + verification trail.

- `src/phoenix/adaptation/reset_bridge.py`, `tests/test_reset_bridge.py`
- +6 tests, suite 189 passed.

## 2026-04-22 — Day 1 of the 5-day Gate-9 sprint

Floor-pivot lab session: Gate 7 (stand-v3 on floor), Gate 8
(mode-switch on floor), Block 3 (failure-mode parquet collection).
Operator card archived at `docs/sessions/LAB_CARD_2026-04-22.md`.

## 2026-04-21 — stand-v3 retrain

Live Gate 7 attempt on stand-v3 saturated per-step slew at 33% on
RL/RR thigh. Two coupled causes diagnosed (stand-posture offset +
OOD policy output). v3 attacks the latter via 4x action_rate + 5x
joint_acc penalty. Sim slew dropped to 0.33% at cmd=0.

See `docs/sessions/lab_findings_2026-04-21.md`.

## 2026-04-19 — two-policy mode switch shipped

Single-policy v3b replacement path was exhausted across four
retrain attempts (`flat-scratch`, `flat-v3b-ft`, `flat-slewhinge`,
`flat-slewhinge-w5`). Root cause: reward-landscape dominance, not
init conditioning. Pivoted to a runtime mode switch:
`stand-v2 + v3b` loaded together, hysteresis + 25-tick blend,
routed on `cmd_vel` magnitude. Opt-in flag, zero retraining,
179 unit tests green.

See `docs/sessions/retrain_flat_scratch_2026-04-19.md` for the
ablation that disproved the fine-tune-destabilization hypothesis,
and `docs/deploy_mode_switch_runbook.md` for how to flip it on.

## 2026-04-17 — pre-lab gates cleared for phoenix-stand

| Gate | Metric | Result |
|---|---|---:|
| 0a — sim rollout | success @ 20.0 s mean length | 16 / 16 |
| 0b — ONNX staging | hashes match deploy path | pass |
| 0c — verify_deploy parity | max torch / ort abs-diff | 3.8e-06 (26x under 1e-4 tol) |

See `docs/sessions/pre_lab_gates_2026-04-17.md`.

## 2026-04-18 — first live hardware dryrun

`sim2real.ros2_policy_node` ran end-to-end on a live GO2. Per-step
slew clip saturated at 30.23% specifically when `cmd_vel = (0, 0, 0)`.
Root cause: rough-v0 baseline trained on 235-dim obs (proprio + height
scan) and zero-padded at deploy — the policy could not respect the
slew cap. Flat-v0 retrain in `ppo_flat.yaml` was the next attempt.

## 2026-04-14 — baseline + warm-start adaptation result

| Policy | Terrain | Mean return | Success | Episodes |
|---|---|---:|---:|---:|
| rough-v0 baseline (500 iters) | rough | 18.95 | 100% | 16 |
| phoenix-base | slippery | 15.90 | 90.6% | 64 |
| phoenix-adapt | slippery | 16.64 | 100% | 64 |
| phoenix-adapt | rough | 17.56 | 96.9% | 64 |

The adaptation is plain warm-start PPO on the slippery overlay,
not a failure-curriculum result (`adaptation.yaml` ships with
`failure_sample_fraction: 0.0` until a hardware-captured parquet
exists). See `docs/sessions/dryrun_findings_2026-04-14.md`.
