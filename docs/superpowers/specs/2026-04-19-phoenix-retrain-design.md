---
title: Phoenix retrain — stand-v2 + flat-v3b-ft
date: 2026-04-19
status: approved
supersedes: 2026-04-18-stand-v2-and-first-message-gate-design.md (extends)
---

# Phoenix retrain — stand-v2 + flat-v3b-ft

## Objective

Unblock Gate 7 by producing two hardware-safe checkpoints:

1. A real **stand specialist** (`phoenix-stand-v2`) distinct from the flat policy, trained with +10× action_rate / +4× joint_acc smoothness penalties and low init noise.
2. A **smoothness-tuned flat policy** (`phoenix-flat-v3b-ft`) fine-tuned from v3b with the same reward knobs plus modest extra cmd=0 exposure.

Both must clear a <5% slew-saturation gate at cmd=0 on sim rollout without regressing velocity tracking beyond v3b baseline.

## Background

- **v3b** (`checkpoints/phoenix-flat/2026-04-16_21-39-16/model_999.pt`) passes sim (32/32 @ 20 s, stand + flat) but on GO2 hardware at `cmd_vel=[0,0,0]` saturates ±0.175 rad/step slew clip on **30.23%** of motor steps. Gate is <5%. Gate 7 blocked.
- **v4** retrain (commit `61fae38`, checkpoints at `phoenix-flat-v4/2026-04-17_21-42-51/`) attempted entropy 0.005→0.01 + init_noise 0.5→1.0 + rel_standing_envs 0.02→0.15 + iters 5000. Lost decisively: return -13%, lin_vel_err +21%, **ang_vel_err +67% worse**. Kept as negative-result reference.
- **phoenix-stand** never existed as a distinct specialist — the ONNX at `phoenix-stand/` was bit-identical to `phoenix-flat/policy.onnx`.
- Existing `configs/env/stand_v2.yaml` + `configs/train/ppo_stand_v2.yaml` (commit `bf17f21`) already encode the stand-specialist recipe but have never been run.

## Runs

| Run | Base checkpoint | Env config | Train config | Iters | Seeds |
|---|---|---|---|---|---|
| **stand-v2** | `phoenix-stand/2026-04-16_22-04-28/model_999.pt` | `configs/env/stand_v2.yaml` (exists) | `configs/train/ppo_stand_v2.yaml` (exists) | 500 | 42 (fallback sweep: 7, 123) |
| **flat-v3b-ft** | `phoenix-flat/2026-04-16_21-39-16/model_999.pt` (v3b) | `configs/env/flat_v3b_ft.yaml` (NEW) | `configs/train/ppo_flat_v3b_ft.yaml` (NEW) | 500 | 42 (fallback sweep: 7, 123) |

### New configs to create

**`configs/env/flat_v3b_ft.yaml`:**

```yaml
# Flat-v3b fine-tune — smoothness recipe + modest cmd=0 exposure.
#
# v3b (checkpoints/phoenix-flat/2026-04-16_21-39-16) passed sim but
# saturated slew clip 30.23% at cmd=0 on GO2 hardware 2026-04-18.
# This fine-tune mirrors stand-v2's action-smoothness knobs and bumps
# rel_standing_envs 0.02 -> 0.10 so the policy actually sees cmd=0
# during training. NOT the v4 over-reach (that went to 0.15 AND
# doubled init_noise AND doubled entropy, which wrecked yaw tracking).
#
# Evaluation criterion: slew_saturation_pct < 0.05 at cmd=0 AND
# lin_vel_err <= 0.10 m/s AND ang_vel_err <= 0.10 rad/s on flat.yaml.

defaults:
  - flat

reward:
  action_rate: -0.5     # was -0.05 (base.yaml)
  joint_acc: -1.0e-6    # was -2.5e-7 (base.yaml)

command:
  rel_standing_envs: 0.10   # was 0.02 (base.yaml); v4 tried 0.15 and lost
```

**`configs/train/ppo_flat_v3b_ft.yaml`:** clone of `ppo_stand_v2.yaml` with:
- `run.name: "phoenix-flat-v3b-ft"`
- `env.config: "configs/env/flat_v3b_ft.yaml"`
- `max_iterations: 500`
- `init_noise_std: 0.1`
- `entropy_coef: 0.005` (unchanged — v4's bump to 0.01 was a mistake)
- `seed: 42`

All other PPO hyperparameters match `ppo_stand_v2.yaml` exactly.

## Gates

All gates evaluated via `phoenix.training.evaluate` + `phoenix.sim2real.verify_deploy`. Each run is accepted only if it passes every applicable gate.

| # | Gate | Threshold | Applies to | Tool |
|---|---|---|---|---|
| G1 | Sim survival — stand.yaml | 32/32 @ 20 s | both | `evaluate --env-config configs/env/stand.yaml --num-envs 16 --num-episodes 32` |
| G2 | Sim survival — flat.yaml | 32/32 @ 20 s | flat-v3b-ft | `evaluate --env-config configs/env/flat.yaml --num-envs 16 --num-episodes 32` |
| G3 | **Slew saturation @ cmd=0** | `< 0.05` | both | `evaluate --env-config configs/env/stand.yaml --slew-saturation-max 0.05` |
| G4 | Tracking — linear vel | `lin_vel_err ≤ 0.10 m/s` | flat-v3b-ft | via `evaluate` on flat.yaml |
| G5 | Tracking — angular vel | `ang_vel_err ≤ 0.10 rad/s` | flat-v3b-ft | via `evaluate` on flat.yaml |
| G6 | ONNX parity | `max_diff < 1e-4` on 200 steps | both | `verify_deploy --tol 1e-4 --max-steps 200` |

stand-v2 is **not** required to pass G2/G4/G5 (specialist, not commanded to move).

## Seed policy

1. Run seed 42 first for each config (sequential: stand-v2 then flat-v3b-ft).
2. If all applicable gates pass → accept, skip the sweep.
3. If any gate fails → run seeds 7 and 123 for that config, keep best-of-3 by composite score:
   `score = slew_saturation_pct + (lin_vel_err / 0.10) + (ang_vel_err / 0.10)` (lower is better; tracking terms zeroed for stand-v2)
4. If none of the 3 seeds pass → STOP. Do not ship a regressed policy. Revisit design.

## Artifacts & staging

1. Commit the two new configs on `audit-fixes-2026-04-16` BEFORE kicking off training (separate commit from training artifacts).
2. For each accepted run:
   - Export ONNX via `phoenix.sim2real.export_onnx`.
   - Run G6 parity gate.
   - `rsync -avL --delete` the run dir to T7.
3. **Gate 7 lab-day staging:** stage `phoenix-stand-v2` ONNX as `checkpoints/phoenix-flat/policy.onnx` (Gate 7 is a standing-only test; the specialist is the safer pick).
4. Stage flat-v3b-ft separately at `checkpoints/phoenix-flat/v3b-ft/` as the Gate 8 candidate. Update `latest.pt` symlink to the flat-v3b-ft `model_999.pt`.
5. Keep v3b artifacts in place at `phoenix-flat/v3b/` — do not delete.

## Reporting

1. Single-row summary table per run (old vs new) in the style of commit `61fae38` post-mortem.
2. Append to vault: `Projects/go2-phoenix/RETRAIN_2026-04-19.md` with before/after numbers for every gate metric, configs used, seeds tried, and Gate 7 staging decision.
3. Update `Projects/go2-phoenix/status.md`: flip from "Gate 7 BLOCKED on saturation gate" to "Gate 7 ready — retrained stand-v2 specialist, hw dryrun pending".
4. Update `project_go2_phoenix.md` memory with the new checkpoint paths and Gate 7 readiness.

## Out of scope

- No DR / IMU-noise / friction curriculum (ruled out in Q2-A; reserved for a follow-on redesign if hardware still fails after this retrain).
- No new reward terms beyond action_rate + joint_acc scaling.
- No PPO hyperparameter changes besides `init_noise_std` and `max_iterations`.
- No from-scratch training. Both runs are fine-tunes from proven checkpoints.
- No hardware dryrun or Gate 7 lab day execution — that's the follow-on CaresLab session after this retrain is green.

## Success definition

This retrain is "done" when:
- Both runs have accepted checkpoints passing all applicable gates.
- ONNX + parquet metrics artifacts committed/rsynced.
- Gate 7 staging is in place (`phoenix-flat/policy.onnx` = stand-v2 ONNX, T7 synced).
- Vault notes + memory updated.

At that point `PHOENIX_NEXT_STEPS.md §1` (T7→Jetson sync) is the next actionable step on the next lab day.
