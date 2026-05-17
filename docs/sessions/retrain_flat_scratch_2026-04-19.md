# flat-scratch retrain — 2026-04-19

**Status:** Negative result on Gate 8 tracking gates. Disproves the "fine-tune destabilization" hypothesis.

**Spec:** `docs/superpowers/specs/2026-04-19-phoenix-gate8-fromscratch-design.md`
**Plan:** `docs/superpowers/plans/2026-04-19-phoenix-gate8-fromscratch.md`
**Run dir:** `checkpoints/phoenix-flat-scratch/2026-04-19_18-14-44/`
**Checkpoint evaluated:** `model_2499.pt` (via `latest.pt` symlink)

## Training summary

- From random init, 2500 iters, seed 42, `slew_sat_hinge_l2 @ w=-50` active from iter 0.
- Wall time: **21:04** on RTX 5070 @ 10240 envs, 245.76 M env steps, 200 k steps/s.
- Final training-rollout metrics (TB, with full DR + full cmd range):
  - `Mean reward: 7.13` (vs v3b's ~35 in its 1000-iter training)
  - `Metrics/base_velocity/error_vel_xy: 1.22`, `error_vel_yaw: 1.19`
  - `Episode_Reward/slew_sat_hinge_l2: -0.062` (non-trivial penalty still paid)
  - `Episode_Termination/base_contact: 0.0005` (near-zero falls during training)
- No crashes. No NaNs. Save rotation clean; `latest.pt` resolves.

## Gate results (phoenix.training.evaluate, 16 envs × 32 eps, warp-array-fix post-2026-04-17)

### G2–G5 on `configs/env/flat.yaml` (nonzero cmd)

| Gate | Target | Measured | Verdict |
|---|---|---|---|
| G2 survival | 32/32 @ 20 s | 32/32, 20.0 s | ✅ |
| **G3 slew_saturation_pct** | **< 0.05** | **0.00254** | ✅ (20× under) |
| G4 mean_lin_vel_error | ≤ 0.10 m/s | 0.579 | ❌ |
| G5 mean_ang_vel_error | ≤ 0.10 rad/s | 0.658 | ❌ |

Metrics file: `docs/rollout_flat_scratch_flat_2026-04-19.json`.

### G1 on `configs/env/stand.yaml` (cmd=0) — secondary

| metric | measured |
|---|---|
| survival | 32/32 @ 20 s |
| slew_saturation_pct | 0.00243 |
| mean_lin_vel_error | 0.034 m/s |
| mean_ang_vel_error | 0.057 rad/s |
| track_lin_vel_xy_exp reward | 0.991 |

Metrics file: `docs/rollout_flat_scratch_stand_2026-04-19.json`.

Scratch is a perfectly competent **stand** policy (same slew + tracking band as stand-v2) and a **tracking-failed** velocity policy.

## Comparison across all 2026-04-19 runs

| run | init | iters | lin_err | ang_err | slew | survival |
|---|---|---|---|---|---|---|
| v3b baseline | scratch (pre-hinge era) | 1000 | **0.091** | **0.087** | 0.302 hw | 32/32 |
| flat-v3b-ft | fine-tune v3b | 500 | 0.619 | 0.435 | 0.341 | 29/32 |
| flat-slewhinge (w=-50) | fine-tune v3b | 500 | 0.623 | 0.607 | 0.123 | 31/32 |
| flat-slewhinge-w5 (w=-5) | fine-tune v3b | 500 | 0.572 | 0.576 | 0.186 | 30/32 |
| **flat-scratch (this run)** | **scratch** | **2500** | **0.579** | **0.658** | **0.00254** | **32/32** |

Slew is now 50× better than any prior run and 120× better than v3b's hardware number. Tracking is in the same ~0.58-0.66 m/s band as every other hinge-active run. **Init regime (scratch vs fine-tune) did not move tracking.**

## What this disproves

The spec hypothesized fine-tune destabilization as the root cause: that v3b's tracking basin was fragile under new reward gradients at `init_noise_std=0.1`, and that growing a policy up under the hinge from iter 0 would avoid the failure.

**That hypothesis is wrong.** The tracking-collapse pattern persists with:
- fresh random init
- `init_noise_std=0.5`
- 5× the iteration budget (2500 vs 500)
- no warm-start from any prior policy

The collapse is reproducible across four orthogonal training setups sharing only one thing: `slew_sat_hinge_l2` as a significant component of the reward. The real root cause is **reward-landscape dominance**, not init conditioning.

## What this proves (observed reward-landscape dominance)

Tracking-reward per-term at iter 2500 on `flat.yaml` vs `stand.yaml`:

| term | flat.yaml (cmd≠0) | stand.yaml (cmd=0) |
|---|---|---|
| track_lin_vel_xy_exp | **0.339** | 0.991 |
| track_ang_vel_z_exp | 0.158 | 0.484 |
| slew_sat_hinge_l2 (implied) | trained to near-zero | trained to near-zero |

The policy maximizes expected return by **choosing to stand even when commanded to move**. At `w=-50`, the cost of saturating motors to achieve tracking exceeds the exponential-kernel reward for hitting the velocity target. The PPO objective is being satisfied — we asked for the wrong objective.

## Budget variance

Plan said ~2hr. Actual: 21 min. RTX 5070 at 200 k steps/s against the flat-v0 env is faster than the flat-v4 precedent (flat-v4 was 45 min at 5000 iters, same env, same batch size). Likely difference: driver updates since the last flat-v4 run, or Isaac Sim 2026-04-16 CUDA 12.8 migration giving a throughput boost. Either way, the 2hr estimate in the spec was conservative by ~6×.

## Paths forward (three options)

All require new specs — no in-plan retry per the plan's §"If any of G2–G5 fails" clause.

### Option A — Reward redesign (hinge threshold / shape)

The current hinge fires at step-delta > 0.15 rad (85% of the ±0.175 rad/step clip). Possible redesigns:
- Raise threshold to 0.17 rad so only *near-saturation* motors are penalized. Current threshold plus w=-50 creates a broad penalty zone that the policy can avoid only by not moving.
- Replace squared-hinge with a barrier function that is zero below threshold and sharply negative only in the last 5% to the clip.
- Add a curriculum: start at `w=-5` for the first 500 iters (let tracking form), anneal up to `w=-50` over the next 1000. Avoids the reward dominating before tracking converges.

**Pro:** Keeps the single-policy approach. No deploy-layer complexity.
**Con:** Another 20-min training run per candidate. Unknown how many iterations of redesign before we land a recipe.

### Option B — Deploy-layer mode switch (recommended)

Ship `stand-v2` for cmd=0 + `v3b` for `|cmd_vel| > eps` with hysteresis in `ros2_policy_node.py`. Neither policy is retrained. Slew saturation is only a problem at cmd=0 per the 2026-04-18 dryrun (30.23% at cmd=0, bearable at nonzero cmds after slew-annealing already in the node). Stand-v2 solves cmd=0. v3b handles nonzero cmds as well as it ever did.

**Pro:** Zero additional training. Both policies already exist. ~100 LOC in `ros2_policy_node.py`. Directly addresses the hardware failure mode.
**Con:** Mode-switch introduces a discontinuity the robot has to survive. Needs hysteresis + blending design.

### Option C — Task-redesign (joint-velocity command instead of base-velocity)

Fundamentally harder. Out of scope for the immediate Gate 7/8 line. Park as a research direction.

## Recommendation

**Option B.** The data now says single-policy with slew hinge doesn't converge at any weight, any init, any reward shape we've tried this session. Shipping a competent stand + a competent walker is cheaper, faster, and matches what every production quadruped team does. Option A is a potentially bottomless research well; Option B is an engineering decision with a known LOC budget.

If B passes Gate 7 + an updated Gate 8 hardware dryrun, Gate 8 is done. The v3b slew-saturation problem is already mitigated at the deploy layer by stand-v2 for cmd=0; the nonzero-cmd slew percentage on real hardware was never measured in isolation and may be fine (the 30.23% was cmd=0 dryrun).

## Action items

- [ ] No ONNX export. Do not touch `checkpoints/phoenix-flat/policy.onnx` — stand-v2 stays at Gate 7.
- [ ] Do not rsync `phoenix-flat-scratch/` to T7 (saves disk; not deploy-bound).
- [ ] Commit configs + both metrics JSONs + this post-mortem on `audit-fixes-2026-04-16`.
- [ ] Update memory `project_go2_phoenix.md` with the disproved hypothesis.
- [ ] User decides Option A vs B vs C. If B, write the mode-switch spec next.
