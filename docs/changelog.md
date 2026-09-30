# Changelog

Release-style summary of past Phoenix milestones.

## 2026-09-30: evidence reconciliation (docs only)

Annotated earlier entries in place rather than rewriting them. Every sim slew
percentage in this log (for example 0.33%) used a legacy raw-action-delta
metric that has since been replaced by deploy-equivalent clip activation, so
it is not comparable to hardware slew figures. The single-cause Gate 7
root-cause attribution is superseded by three candidate train/deploy
mismatches (a deploy-only per-step rate limiter, all four hips at 0.0 in
`configs/sim2real/deploy_stand_v2.yaml` against +/-0.1 in training, and
`base_lin_vel` fed to the policy as zeros); which dominates is pending
`scripts/deploy_ablation.py` on the `feat/causal-viability-replication`
branch. Record: [superseded results](https://github.com/yusufdxb/go2-phoenix/blob/feat/causal-viability-replication/docs/superseded_results.md).

- `docs/changelog.md`, `README.md`, `EVIDENCE.md`
- No code, config, or data changes; README and EVIDENCE test count set to the
  measured 236.

## 2026-05-17: sweep + harness system

Added a 12-cell sim-side benchmark sweep over friction range, lateral push,
and action-rate weight to stress-test the stand-v3 baseline. Added a
four-script lab-day harness (preflight, recorder, diversity validator, EOD
rsync).

- `scripts/sweep_run.py`, `configs/train/sweep_stand_v3_stress.yaml`
- `scripts/harness_{preflight,record,diversity,eod}.{sh,py}`
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

> **Superseded (2026-09-30).** The 0.33% sim figure is a legacy raw-action-delta
> metric, not deploy-equivalent clip activation, so it is not comparable to the
> 33% hardware figure ([record](https://github.com/yusufdxb/go2-phoenix/blob/feat/causal-viability-replication/docs/superseded_results.md#2-every-simulator-slew-saturation-percentage)).
> The two-cause diagnosis above is also superseded: three train/deploy
> mismatches were live in that run (a deploy-only per-step rate limiter, all
> four hips at 0.0 in `deploy_stand_v2.yaml` against +/-0.1 in training, and
> `base_lin_vel` fed as zeros), and which dominates is pending
> `scripts/deploy_ablation.py` on the `feat/causal-viability-replication`
> branch ([record](https://github.com/yusufdxb/go2-phoenix/blob/feat/causal-viability-replication/docs/superseded_results.md#3-the-gate-7-root-cause-attribution)).

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

> **Note (2026-09-30).** No ablation isolating this single cause is recorded in
> this repo. For the later 2026-04-21 run, single-cause attribution is
> superseded by three candidate train/deploy mismatches (deploy-only per-step
> rate limiter, hips at 0.0 at deploy against +/-0.1 in training,
> `base_lin_vel` fed as zeros), with the ablation pending
> ([record](https://github.com/yusufdxb/go2-phoenix/blob/feat/causal-viability-replication/docs/superseded_results.md#3-the-gate-7-root-cause-attribution)). Whether those
> mismatches also applied to this rough-v0 run is not verified.

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
