# stand-v2 fine-tune — 2026-04-19

## Summary

Fine-tuned `phoenix-stand/2026-04-16_22-04-28/model_999.pt` on
`configs/env/stand_v2.yaml` (+10× action_rate, +4× joint_acc vs base, init
noise 0.1) for 500 iters, seed 42. Exported ONNX staged as the Gate 7
candidate at `checkpoints/phoenix-flat/policy.onnx`. First seed passed all
applicable gates — seed fallback sweep not needed.

## Run dir
`checkpoints/phoenix-stand-v2/2026-04-19_11-20-36` (final checkpoint
`model_499.pt`, ~15 min wall @ 10240 envs on RTX 5070).

## Gates

| # | Gate | Threshold | Result |
|---|---|---|---|
| G1 | 32/32 success + mean ep len | 1.0 / 20.0 s | **1.0 / 20.0 s** ✓ |
| G3 | `slew_saturation_pct` @ cmd=0 | <0.05 | **0.00337** ✓ |
| G6 | ONNX↔TorchScript parity | max_diff <1e-4 on 200 steps | **3.82e-06** ✓ |

Rollout config: `configs/env/stand.yaml`, 16 envs × 32 episodes (via
`phoenix.training.evaluate --slew-saturation-max 0.05`). Full JSON at
`docs/rollout_stand_v2_2026-04-19.json`.

## Hardware-vs-sim slew saturation

| Source | cmd_vel | slew_saturation_pct |
|---|---|---|
| stand-v1 on GO2 hardware (2026-04-18 CaresLab dryrun) | 0 | **30.23%** (6× over <5% gate) |
| stand-v2 in sim (this run, stand.yaml rollout)        | 0 | **0.34%** (**~89× reduction**) |

Headroom for sim-to-real gap is ~15× the hardware gate. A sim→hw ratio of
1.5× (a generous gap on a stand-only task) would still put us at ~0.5%,
well under 5%.

## Side-effects on tracking (bonus — stand task doesn't command motion)

| metric | value |
|---|---|
| `mean_episode_return` | 29.07 |
| `mean_lin_vel_error`  | 0.008 m/s |
| `mean_ang_vel_error`  | 0.017 rad/s |

The policy is tracking the zero command extremely tightly — it sits on
the stand point with negligible drift.

## Training curve

- `Train/mean_reward`: 0.12 → 28.81 (converged by iter ~100, plateau
  through iter 499)
- `Train/mean_episode_length`: 11.9 → 1000.0 steps (full episode survival)
- `Episode_Reward/action_rate_l2` (per-step weighted): stable -0.004 to
  -0.008 across training; weight 0.5 × policy's natural smoothness made
  the penalty visible but not dominating
- `Episode_Reward/dof_acc_l2`: stable -0.001 to -0.002

## Staging

- stand-v2 ONNX staged as `checkpoints/phoenix-flat/policy.onnx` (Gate 7
  candidate path hardcoded in `configs/sim2real/deploy.yaml`)
- Run dir rsync'd to T7 at
  `/media/yusuf/T7 Storage/go2-phoenix/checkpoints/phoenix-stand-v2/2026-04-19_11-20-36/`
- Staged ONNX rsync'd to T7 at
  `/media/yusuf/T7 Storage/go2-phoenix/checkpoints/phoenix-flat/`
- v3b artifacts retained at `checkpoints/phoenix-flat/v3b/` (Gate 8
  fallback; flat-v3b-ft replaces it in Phase 2)

## Next

Gate 7 lab day (CaresLab) uses this stand-v2 ONNX. Pre-lab checklist per
`Vault/Projects/go2-phoenix/NEXT_STEPS_2026-04-17.md` §1 (T7→Jetson sync)
is the next ping point. No hardware work in this session.

Phase 2 (`flat-v3b-ft` fine-tune) proceeds from here.
