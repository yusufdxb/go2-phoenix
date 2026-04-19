# flat-slewhinge fine-tune — 2026-04-19 (NEGATIVE RESULT)

## Summary

Phase 2b of the 2026-04-19 retrain session. Added a new Phoenix-owned
reward term `slew_sat_hinge_l2` (per-motor squared hinge, threshold
0.15 rad, 0.025 rad below the ±0.175 hardware slew clip) via a new
`_NEW_TERM_FACTORIES` dispatcher in `go2_env_cfg.py`. Fine-tuned v3b
at two weights: `w=-50` (spec default) and `w=-5` (lighter fallback
after w=-50 failed tracking). **Both runs failed all gates.** Not
exported, not staged. Hinge reward proven effective (2.7× slew
reduction), but fine-tuning v3b destabilizes its tracking solution
regardless of weight.

## Runs

| Run | Weight | Run dir |
|---|---|---|
| flat-slewhinge | -50.0 | `checkpoints/phoenix-flat-slewhinge/2026-04-19_12-18-33/` |
| flat-slewhinge-w5 | -5.0 | `checkpoints/phoenix-flat-slewhinge-w5/2026-04-19_12-29-13/` |

Both: 500 iters @ 10240 envs, seed 42, resume from v3b
(`phoenix-flat/2026-04-16_21-39-16/model_999.pt`), ~15 min wall each.

## Gates (both weights)

| # | Gate | Threshold | v3b | flat-v3b-ft | w=-50 | w=-5 |
|---|---|---|---|---|---|---|
| G1 | success_rate | 32/32 | 32/32 | 29/32 | **31/32** | **30/32** |
| G1 | mean_ep_length_s | 20.0 | 20.0 | 19.0 | **19.5** | **19.6** |
| G2 | slew_saturation_pct | <0.05 | 0.302 hw | 0.335 sim | **0.123** | **0.186** |
| G3 | mean_lin_vel_error | ≤0.10 m/s | 0.091 | 0.619 | **0.623** | **0.572** |
| G4 | mean_ang_vel_error | ≤0.10 rad/s | 0.087 | 0.435 | **0.607** | **0.576** |
| G5 | ONNX parity | n/a (not exported) | — | — | — | — |

Full JSONs:
- `docs/rollout_flat_slewhinge_flat_2026-04-19.json`
- `docs/rollout_flat_slewhinge_w5_flat_2026-04-19.json`

## Root cause — fine-tune destabilization, not reward-shape

The hinge reward works exactly as intended — w=-50 achieved a
**2.7× reduction** in slew saturation over flat-v3b-ft (33.5% → 12.3%)
despite being a 500-iter fine-tune. The shape of the penalty (per-motor
squared hinge at 85% of clip) correctly targets the failure mode that
`action_rate_l2` could not.

But **tracking collapses identically** at both weights:

| weight | lin_vel_err | ang_vel_err |
|---|---|---|
| v3b baseline | 0.091 | 0.087 |
| flat-v3b-ft (`action_rate_l2 -0.5`) | 0.619 | 0.435 |
| flat-slewhinge w=-50 | 0.623 | 0.607 |
| flat-slewhinge w=-5 | 0.572 | 0.576 |

Three different reward configurations (action_rate L2, slew_hinge -50,
slew_hinge -5) all converge to lin_vel_err ≈ 0.58-0.62 m/s — a 6-7×
regression from v3b. The common factor is **the fine-tune itself**,
not the reward.

### Training-curve evidence

- w=-50: `mean_reward` first=-56.8, **min=-1983**, last=0.85
- w=-5:  `mean_reward` first=-5.7, min=-183.2, last=6.32

Even at w=-5 (10× lighter than the default), mean_reward dipped below
-180 before recovering. With `init_noise_std=0.1` + LR=1e-3 + 10240
envs + adaptive KL schedule, the early-training gradient is large
enough that ANY new reward pressure drives the policy off v3b's
precise tracking solution. 500 iters isn't enough for the policy to
rebuild tracking precision — by end of training survival is ~95% and
slew is reduced, but tracking error is stuck ~6× over v3b.

## Decision

**Do NOT ship either checkpoint.** The w=-50 run passed survival (31/32)
but failed tracking by 6×. It would be strictly worse than v3b on
hardware for any walking task.

**Do NOT run w=-500 fallback** (spec §Seed/weight policy decision tree
says run w=-500 when G2 fails). The tracking data at w=-50 and w=-5
show tracking is already destroyed regardless of weight — going heavier
will make tracking worse, not better. No weight in `{-5, -50, -500}`
solves both gates simultaneously.

**Do NOT run seed sweep** (7, 123). The failure is architectural
(fine-tune destabilization), not seed-variance.

## Path forward (follow-on session)

Two viable options for Gate 8:

### Option A — deploy-layer mode-switch (recommended)
Ship stand-v2 and v3b as-is. Add a ROS 2 policy node that switches
between them based on commanded velocity magnitude:
- `|cmd_vel| < ε` → stand-v2 (Gate 7's proven policy, slew 0.34%)
- `|cmd_vel| ≥ ε` → v3b (proven walker, lin_vel_err 0.091 m/s)

Engineering investment: ~100 LOC + hysteresis logic + switch-discontinuity
testing. No GPU retrain needed. Needs its own spec (design hysteresis
boundary, which policy owns startup, how to handle mid-episode switch).

### Option B — from-scratch training with slew_sat_hinge active from iter 0
Train a new flat-v0 policy from random init with the slew-hinge term
active throughout. No v3b prior to destabilize — the policy grows up
under the slew constraint. ~2 hr GPU at 10240 envs, 2500 iters. Higher
probability of a clean single-policy answer but much more expensive than
the mode-switch.

Neither is a this-session task. Document the negative result, freeze
the artifacts, move to a fresh session with a new spec.

## Artifacts

Code (kept; forms the template for future Phoenix-owned reward terms):
- `src/phoenix/sim_env/rewards.py` — new module, `slew_sat_hinge_l2` function
- `src/phoenix/sim_env/go2_env_cfg.py` — `_NEW_TERM_FACTORIES` dispatcher
- `tests/test_slew_sat_hinge.py` — 6 unit tests (1 is the drift-guard)
- `tests/test_go2_env_cfg_rewards.py` — +2 factory integration tests

Configs (kept for reproducibility / future weight sweeps):
- `configs/env/flat_slewhinge.yaml` (w=-50)
- `configs/env/flat_slewhinge_w5.yaml` (w=-5)
- `configs/train/ppo_flat_slewhinge.yaml`
- `configs/train/ppo_flat_slewhinge_w5.yaml`

Training artifacts (retained for diagnosis, NOT shipped):
- `checkpoints/phoenix-flat-slewhinge/2026-04-19_12-18-33/`
- `checkpoints/phoenix-flat-slewhinge-w5/2026-04-19_12-29-13/`

Neither ONNX-exported. No T7 sync (would only be diagnostic data —
done if needed for a deeper post-mortem).

## Gate 7 status

Unaffected. `checkpoints/phoenix-flat/policy.onnx` = stand-v2 (staged in
Phase 1). Gate 7 lab day proceeds as planned; Gate 8 becomes a follow-on
session.
