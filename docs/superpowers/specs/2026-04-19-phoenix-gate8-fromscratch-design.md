---
title: Phoenix Gate 8 retrain — from-scratch + slew-sat hinge
date: 2026-04-19
status: draft
supersedes: 2026-04-19-phoenix-gate8-slewhinge-design.md (both fine-tune runs failed; root cause was fine-tune destabilization)
---

# Phoenix Gate 8 retrain — from-scratch + slew-sat hinge

## Objective

Produce `phoenix-flat-scratch` — a v3b replacement trained **from random
init** (no `--resume`) with `slew_sat_hinge_l2` active from iteration 0.
Gate targets on `flat.yaml`:

- `slew_saturation_pct < 0.05` at nonzero `cmd_vel`
- `mean_lin_vel_err ≤ 0.10 m/s`
- `mean_ang_vel_err ≤ 0.10 rad/s`
- 32/32 survival @ 20 s

If all four clear, stage the ONNX as the Gate 8 velocity policy. Gate 7
(stand) continues to use stand-v2; the deploy-layer mode-switch decision
remains a separate downstream choice.

## Background — why from-scratch now

### Observed failure pattern across 3 fine-tunes from v3b (2026-04-19)

| run | reward pressure | lin_vel_err | slew_sat | outcome |
|---|---|---|---|---|
| v3b baseline | upstream action_rate_l2 (-0.05) | 0.091 | 0.302 hw | reference |
| flat-v3b-ft | +10× action_rate_l2, +4× joint_acc | 0.619 | 0.341 sim | FAIL |
| flat-slewhinge (w=-50) | slew_sat_hinge_l2 only | 0.623 | 0.123 sim | FAIL |
| flat-slewhinge-w5 (w=-5) | slew_sat_hinge_l2 only | 0.572 | 0.186 sim | FAIL |

Three orthogonal reward landscapes all collapsed tracking to
`lin_vel_err ≈ 0.57–0.62 m/s` after 500 iters of fine-tuning from v3b.
The hinge reward **works** (2.7× slew reduction at w=-50) — but the
tracking collapse is independent of reward shape. That localizes the
failure to **fine-tune destabilization**:

- v3b's converged tracking basin is fragile under any new reward
  gradient at `init_noise_std=0.1 + LR=1e-3 + 10240 envs`.
- 500 iters is below the reset-and-reconverge horizon for tracking
  precision once the policy leaves v3b's local basin.

### Why from-scratch breaks the pattern

A fresh policy has no basin to leave. Growing up under
`action_rate_l2 + slew_sat_hinge_l2` from iter 0 means the policy
never learns the "clip a few motors hard, keep others quiet" local
optimum that both fine-tune attempts converged to (it's only
reachable as a perturbation *off* v3b). The rest of the training
recipe is unchanged from the v3b-producing run, so any delta vs v3b
is attributable to the new reward term.

## Design

### Config pair

**New file:** `configs/env/flat_scratch.yaml`

```yaml
defaults:
  - flat
reward:
  slew_sat_hinge: -50.0
```

Inherits `flat.yaml` (which inherits `base.yaml`). The only delta from
v3b's training env is the additional `slew_sat_hinge` term. All other
reward weights, DR ranges, command ranges, and `rel_standing_envs=0.02`
stay at base.

**New file:** `configs/train/ppo_flat_scratch.yaml`

```yaml
run:
  name: "phoenix-flat-scratch"
  output_dir: "checkpoints"
  log_interval: 1
  save_interval: 100
  max_iterations: 2500
  seed: 42
  device: "cuda:0"

env:
  config: "configs/env/flat_scratch.yaml"

algorithm:
  class_name: "PPO"
  value_loss_coef: 1.0
  use_clipped_value_loss: true
  clip_param: 0.2
  entropy_coef: 0.005
  num_learning_epochs: 5
  num_mini_batches: 4
  learning_rate: 1.0e-3
  schedule: "adaptive"
  gamma: 0.99
  lam: 0.95
  desired_kl: 0.01
  max_grad_norm: 1.0

policy:
  class_name: "ActorCritic"
  init_noise_std: 0.5
  actor_hidden_dims: [512, 256, 128]
  critic_hidden_dims: [512, 256, 128]
  activation: "elu"

runner:
  num_steps_per_env: 24
  empirical_normalization: true

logging:
  tensorboard: true
  wandb: false
  wandb_project: "go2-phoenix"
```

### Design decisions

| knob | value | rationale |
|---|---|---|
| from-scratch | **no `--resume`** | avoids fine-tune destabilization — the whole point |
| `max_iterations` | 2500 | matches `ppo_flat.yaml` (the run that produced v3b's architecture) |
| `init_noise_std` | 0.5 | matches `ppo_flat.yaml` — fresh policy needs exploration; 0.1 is the fine-tune value |
| `entropy_coef` | 0.005 | stable default; v4's 0.01 bump was a documented mistake |
| `rel_standing_envs` | 0.02 | base — walking focus; stand handled by stand-v2 + mode-switch |
| `slew_sat_hinge` weight | -50.0 | matches `flat_slewhinge.yaml` — apples-to-apples reward pressure |
| `action_rate_l2` weight | -0.05 | upstream default — retain to discourage base chatter; hinge handles per-motor clip |
| seed | 42 | project convention |
| `num_envs` | 10240 | matches v3b; `num_envs` lives in Isaac Lab cmd line, not this yaml |

### What is NOT in this spec

- No deploy-layer mode-switch design. That is an independent ROS 2
  policy-node change; a separate spec covers it if/when we go that
  route (see Phase 2b spec §Scope).
- No reward-weight sweep. We commit to `w=-50` because the prior
  sweep already ranked `-50 > -5` on slew reduction and tracking was
  independent of weight. If from-scratch at `w=-50` also collapses
  tracking, that's a harder failure and warrants a new spec, not a
  sweep.
- No architecture changes. Actor/critic dims, activation, observation
  space are identical to v3b.
- No additional DR or perturbation. `configs/env/flat_scratch.yaml`
  inherits `flat.yaml` verbatim aside from the new reward term.

## Budget

- Training wall-time: ~2 hours on RTX 5070 (extrapolated from the 45-min
  flat-v4 run at 5000 iters and the 10-min flat-slewhinge runs at
  500 iters; 2500 iters × ~2.5s/iter ≈ 105 min, add overhead).
- Evaluation: ~5 min (`phoenix.training.evaluate` on 16 envs × 32 eps
  per task, ~3 tasks).
- ONNX export + parity: <1 min.
- T7 sync: <1 min.

## Success criteria — gates (apples-to-apples with prior runs)

All gates evaluated with the post-`warp-array-fix` `evaluate.py`
(introduced 2026-04-17; measured metrics are not silently-zero
anymore):

- **G1** 16 envs × 32 eps on `stand.yaml` (cmd=0): 32/32 survival,
  `slew_saturation_pct ≤ 0.05`. This is a nice-to-have; from-scratch
  isn't expected to match stand-v2 at cmd=0 (different task).
- **G2** 16 envs × 32 eps on `flat.yaml` (nonzero cmd): 32/32 survival.
- **G3** G2 rollout: `slew_saturation_pct < 0.05` (primary gate).
- **G4** G2 rollout: `mean_lin_vel_err ≤ 0.10 m/s`.
- **G5** G2 rollout: `mean_ang_vel_err ≤ 0.10 rad/s`.
- **G6** `phoenix.sim2real.verify_deploy`: `max_diff < 1e-4` over 200
  steps on `synth_slippery_trained.parquet`.

If **G3 OR (G4 AND G5) fail**, the run is a negative result: write
post-mortem, do not stage ONNX, do not sync to T7.

If **all gates green**: stage ONNX to `checkpoints/phoenix-flat/gate8/`
(sibling of the Gate 7 stand-v2 `policy.onnx` that remains in place),
rsync to T7, tag `v0.3.0-gate8-candidate`.

## Risk log

1. **2hr run fails at iter ~2000 via OOM / DR divergence / Isaac Lab
   upgrade regression.** Mitigation: save_interval=100 so we always
   have a recent checkpoint; a failed run still produces a post-mortem
   with the longest-lived checkpoint evaluated.
2. **Tracking converges but slew stays above 5%.** Means the weight
   is too low or the hinge threshold is too loose. Bump `w` to `-200`
   in a follow-on spec, not as an in-run change.
3. **Slew clears but tracking lands in the `0.3–0.5 m/s` zone
   (better than fine-tune, worse than v3b).** Borderline result —
   still useful for Gate 8 proof-of-life; document clearly in
   post-mortem, do not overstate.
4. **Save rotation collides with `latest.pt` symlink.** Mirror v3b
   convention: `latest.pt` auto-updates to the newest `model_*.pt`.
5. **ONNX export uses wrong checkpoint.** Pin export to
   `checkpoints/phoenix-flat-scratch/<ts>/latest.pt` — never a hard-coded
   iter.
