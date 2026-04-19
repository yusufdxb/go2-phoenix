---
title: Phoenix retrain — stand-v2 + flat-v3b-ft
date: 2026-04-19
status: approved (revised 2026-04-19 after codex critique — Phase 0 added)
supersedes: 2026-04-18-stand-v2-and-first-message-gate-design.md (extends)
---

# Phoenix retrain — stand-v2 + flat-v3b-ft

> **Revision 2026-04-19 (post-codex critique):** Codex surfaced that the
> `reward` section in Phoenix env YAMLs is **not applied** to the Isaac
> Lab env cfg — `src/phoenix/sim_env/go2_env_cfg.py:41` lists
> `_UNWIRED_TOP_LEVEL = ("reward", "termination")` and only logs a
> warning. Both the existing `stand_v2.yaml` (commit `bf17f21`) and the
> newly proposed `flat_v3b_ft.yaml` set `reward.action_rate: -0.5` and
> `reward.joint_acc: -1.0e-6`, but **neither takes effect** — upstream
> `UnitreeGo2RoughEnvCfg` reward weights win. Without wiring, running
> either fine-tune reproduces v3b's training distribution and the
> hardware failure. **Phase 0 (reward-wiring) is a hard prerequisite.**

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

## Phase 0 — Wire the `reward` section (prerequisite, blocks everything else)

**Problem.** `src/phoenix/sim_env/go2_env_cfg.py` applies `env`, `command`,
`domain_randomization`, `perturbation`, and `seed` from the YAML, but
`reward` is on the `_UNWIRED_TOP_LEVEL` list — silently ignored except for a
warning log. Every YAML-level `reward.*` override is a no-op today.

**What to build.** An `_apply_rewards(env_cfg, data.get("reward", {}))` helper
that, for each YAML key in `reward:`, sets
`env_cfg.rewards.<mapped_term_name>.weight = <value>`. The YAML key ↔ Isaac
Lab reward term name mapping lives inside the helper (the upstream GO2 task
uses names like `action_rate_l2`, `joint_acc_l2`, `track_lin_vel_xy_exp`;
a YAML key like `action_rate` maps to `action_rate_l2`). Unknown keys raise,
not warn — we do not want more silent drift.

**Code changes.**
1. Add `_apply_rewards(env_cfg, data.get("reward", {}))` to
   `src/phoenix/sim_env/go2_env_cfg.py` following the same pattern as
   `_apply_commands` / `_apply_perturbation`.
2. Call it from `build_env_cfg` **after** `_apply_perturbation`.
3. Remove `"reward"` from `_UNWIRED_TOP_LEVEL`. `termination` stays unwired
   (separate, out-of-scope PR per the existing module docstring).
4. Update module docstring to reflect the new wired list.

**Tests (must pass before any retrain).**
- `test_apply_rewards_sets_weights`: load `configs/env/stand_v2.yaml`, build
  `env_cfg`, assert `env_cfg.rewards.action_rate_l2.weight == -0.5` and
  `env_cfg.rewards.joint_acc_l2.weight == -1.0e-6`.
- `test_apply_rewards_missing_term_raises`: synthetic config with
  `reward.bogus_term: -1.0` must raise, not silently pass.
- `test_unwired_warning_no_longer_fires_for_reward`: loading any YAML with a
  `reward` section does not emit the "unwired" warning; loading a YAML with
  `termination` still does.

**Reproducibility note.** This Phase 0 change **invalidates v3b as a
reproducible baseline** — v3b was trained under upstream reward weights,
so once rewards are wired, "re-running v3b's config" would produce a
different run. Keep the v3b checkpoint untouched at
`phoenix-flat/2026-04-16_21-39-16/` as the frozen reference. The v3b
comparisons in the Gates section stay valid because they compare against
the v3b *checkpoint*, not against a re-trained v3b.

**Commit.** Phase 0 code + tests land as a single commit on
`audit-fixes-2026-04-16` with subject `sim_env: wire reward section
(removes silent no-op; blocks retrain)`, before any new training
artifacts are produced.

**Gate for Phase 0.** Tests green locally + a 1-iter smoke retrain that
confirms the TensorBoard `Rewards/action_rate_l2` scalar changes when the
YAML weight changes (sanity: the value actually propagated).

Only after Phase 0 is green do Runs 1 + 2 below execute.

---

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

- Wiring the `termination`, `observation.noise`, `robot.init_state`, or
  `robot.actuator` sections — separate PRs. Phase 0 only wires `reward`.
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
