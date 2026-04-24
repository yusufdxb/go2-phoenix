---
title: Phoenix Gate 8 retrain — slew-sat hinge reward
date: 2026-04-19
status: approved (Phase 2b follow-on to the 2026-04-19 retrain spec)
supersedes: none (extends 2026-04-19-phoenix-retrain-design.md)
---

# Phoenix Gate 8 retrain — slew-sat hinge reward

## Objective

Produce a v3b replacement (`phoenix-flat-slewhinge`) that clears
`slew_saturation_pct < 0.05` at nonzero `cmd_vel` while preserving
v3b's tracking (`mean_lin_vel_err ≤ 0.10 m/s`,
`mean_ang_vel_err ≤ 0.10 rad/s`). Hardware `cmd_vel ≈ 0` is handled
by a separate deploy-layer mode-switch to stand-v2 (explicitly **out
of scope** of this spec; needs its own ROS 2 policy-node design).

## Background

### Prior attempts in this session (2026-04-19)

- Phase 0 (commits `fa9ce5e`→`3b82f86`): wired the reward YAML
  section into the Isaac Lab env cfg. No-ops were silent before
  this. Live A/B smoke confirmed weight propagation
  (`Episode_Reward/action_rate_l2` ratio 10.00/10.04 matches weight
  ratio 10×).
- Phase 1 (`phoenix-stand-v2`): fine-tuned the stand specialist.
  All gates green, 32/32 @ 20 s, `slew_saturation_pct = 0.00337` at
  `cmd=0`. Staged at `checkpoints/phoenix-flat/policy.onnx` as the
  Gate 7 policy. Hardware-ready.
- Phase 2 (`phoenix-flat-v3b-ft`, **FAILED**): fine-tuned v3b with
  `+10× action_rate_l2`, `+4× joint_acc`, `rel_standing_envs` 0.10.
  All flat.yaml gates failed: lin_vel_err 0.619 m/s (7× worse than
  v3b's 0.091), ang_vel_err 0.435 rad/s (5× worse than v3b's 0.087),
  slew_sat 34.1% (worse than v3b's 30.2% hardware failure).
  Not shipped.

### Root cause of Phase 2 failure

`action_rate_l2 = sum_i (a_t^i - a_{t-1}^i)^2`. Single-motor slew
saturation contributes a tiny slice of this L2 norm; the other 11
motors sitting near zero keep the total small. Evidence from the
flat-v3b-ft rollout: `action_rate_l2` per-step mean = -0.00134
(tiny) while `slew_saturation_pct = 0.341`. The policy converged to
"clip a few motors hard, keep others quiet" — which minimizes L2
while saturating the per-motor clip. **The penalty shape is wrong,
not its weight.** Scaling +10× or even +100× wouldn't fix it.

## The fix — new reward term `slew_sat_hinge_l2`

A per-motor squared-hinge penalty on the *actual* per-motor slew
delta. Fires only when a motor's step-delta approaches the ±0.175
rad/step hardware clip; penalty is proportional to the square of
how far it's past the hinge threshold.

### Implementation

**New file:** `src/phoenix/sim_env/rewards.py`

```python
"""Phoenix-owned reward term functions, not provided by upstream
Isaac Lab. Added 2026-04-19 for Gate 8 slew-saturation targeting.
"""

from __future__ import annotations

import torch

from phoenix.sim2real.safety import MAX_DELTA_PER_STEP_RAD

_DEFAULT_THRESHOLD = 0.15  # 85% of MAX_DELTA_PER_STEP_RAD = 0.175 rad


def slew_sat_hinge_l2(
    env,
    threshold: float = _DEFAULT_THRESHOLD,
) -> torch.Tensor:
    """Per-motor squared-hinge penalty on action deltas approaching
    the per-step slew clip.

    For each env at each step, computes |a_t^i - a_{t-1}^i| per
    motor i, applies a hinge at ``threshold`` (default 0.15 = 85% of
    MAX_DELTA_PER_STEP_RAD 0.175 rad), squares, and sums across
    motors. Returns a per-env positive-magnitude tensor of shape
    [num_envs]; caller applies negative weight via RewTerm.

    Targets the same failure mode that ``slew_saturation_pct`` in
    phoenix.training.evaluate measures on rollouts. Unlike
    action_rate_l2 (an L2 norm across all motors) this penalty
    fires per-motor — one motor hitting the clip is enough to
    activate it.
    """
    action = env.action_manager.action            # [E, num_actions]
    prev   = env.action_manager.prev_action       # [E, num_actions]
    delta  = torch.abs(action - prev)
    excess = torch.clamp(delta - threshold, min=0.0)
    return (excess ** 2).sum(dim=-1)              # [E]
```

### Registration

Add a new `RewTerm` to `UnitreeGo2RoughEnvCfg.rewards` in the
phoenix env-cfg builder:

```python
from phoenix.sim_env.rewards import slew_sat_hinge_l2

# in _apply_rewards (or a sibling helper if structure prefers):
# register the term if the YAML key is present.
```

Integration detail: since `slew_sat_hinge_l2` is **not** an upstream
Isaac Lab term (unlike the 8 entries already in `_REWARD_TERM_MAP`),
`_apply_rewards` must be extended to handle "new terms to attach"
in addition to "existing terms to reweight". Proposed shape:

```python
# go2_env_cfg.py
from phoenix.sim_env.rewards import slew_sat_hinge_l2
from isaaclab.managers import RewardTermCfg as RewTerm

_NEW_TERM_FACTORIES = {
    "slew_sat_hinge": lambda weight: RewTerm(
        func=slew_sat_hinge_l2,
        weight=float(weight),
        params={"threshold": 0.15},
    ),
}
```

Then in `_apply_rewards`, for each YAML key:
- if key is in `_REWARD_TERM_MAP` → setattr existing term's weight
  (existing behavior, unchanged)
- elif key is in `_NEW_TERM_FACTORIES` → build RewTerm via factory,
  setattr on `env_cfg.rewards`
- else → raise KeyError (existing behavior, unchanged)

### Tests

All in `tests/test_slew_sat_hinge.py` (new):

- `test_zero_below_threshold`: all deltas < 0.15 → reward tensor is
  all zeros
- `test_squared_above_threshold`: single motor at 0.175 → reward =
  (0.025)² = 6.25e-4 on that env
- `test_sums_across_motors`: 3 motors at clip → reward = 3 × 6.25e-4
- `test_per_env_independent`: different envs get different
  penalties depending on their action deltas
- `test_threshold_parameter`: raising threshold to 0.175 → no
  motor at exactly the clip triggers penalty (strict inequality)
- `test_registration_via_new_factory`: loading a YAML with
  `reward.slew_sat_hinge: -50.0` builds the RewTerm and attaches
  it to `env_cfg.rewards.slew_sat_hinge_l2` with weight = -50.0
  (integration-style test using the fake env cfg from
  `test_go2_env_cfg_rewards.py`)

The first five tests use pure-torch fakes; the sixth extends the
existing fake-env-cfg fixture.

## Run spec — `phoenix-flat-slewhinge`

| | |
|---|---|
| Baseline checkpoint | `checkpoints/phoenix-flat/2026-04-16_21-39-16/model_999.pt` (v3b) |
| Env config (NEW) | `configs/env/flat_slewhinge.yaml` |
| Train config (NEW) | `configs/train/ppo_flat_slewhinge.yaml` |
| Iters | 500 |
| Envs | 10240 |
| Seed | 42 (weight-sweep fallback, see §Seed policy) |
| Wall | ~15 min on RTX 5070 |

**`configs/env/flat_slewhinge.yaml`:**

```yaml
# Flat-slewhinge: v3b + direct per-motor slew-clip penalty.
#
# See docs/superpowers/specs/2026-04-19-phoenix-gate8-slewhinge-design.md
# for why L2 action_rate was insufficient (flat-v3b-ft negative result).

defaults:
  - flat

reward:
  # Isolate the new lever. action_rate_l2 / joint_acc stay at base
  # (v3b familiar weights) so any delta vs v3b is attributable to
  # the new term.
  slew_sat_hinge: -50.0

# command.rel_standing_envs stays at base 0.02. Deploy mode-switches
# to stand-v2 at |cmd_vel| < epsilon; no need to shift v3b cmd=0.
```

**`configs/train/ppo_flat_slewhinge.yaml`:** clone of
`ppo_flat_v3b_ft.yaml` with `run.name: "phoenix-flat-slewhinge"`,
`env.config: "configs/env/flat_slewhinge.yaml"`, otherwise identical
(seed 42, 500 iters, `init_noise_std 0.1`, `entropy_coef 0.005`).

## Gates

| # | Gate | Threshold | Where | Tool |
|---|---|---|---|---|
| G1 | Sim survival | 32/32 @ 20 s | flat.yaml | `phoenix.training.evaluate --env-config configs/env/flat.yaml --num-envs 16 --num-episodes 32` |
| G2 | `slew_saturation_pct` | <0.05 | flat.yaml (nonzero cmds dominate) | same rollout |
| G3 | `mean_lin_vel_error` | ≤0.10 m/s | flat.yaml | same rollout |
| G4 | `mean_ang_vel_error` | ≤0.10 rad/s | flat.yaml | same rollout |
| G5 | ONNX parity | max_diff <1e-4 on 200 steps | `verify_deploy` | `phoenix.sim2real.verify_deploy` |
| — | stand.yaml survival | informational only | stand.yaml | `phoenix.training.evaluate --env-config configs/env/stand.yaml` (no `--slew-saturation-max`) |

Rollout config: 16 envs × 32 episodes.

## Seed / weight policy

Start seed 42 at `slew_sat_hinge = -50.0` (from `flat_slewhinge.yaml`).

Decision tree on failure:
- **G2 fails** (slew still above 5%) → retry at weight **-500** (10×
  heavier penalty). Create `flat_slewhinge_w500.yaml` overlay +
  sibling ppo config.
- **G3 or G4 fails** (tracking regressed) → retry at weight **-5**
  (10× lighter).
- **Both G2 AND (G3 or G4) fail** at w=-50 → retry at w=-500 first
  (biggest lever on the primary gate).
- **At any weight all four gates pass** → accept, skip rest of
  sweep.
- If none of `{-5, -50, -500}` passes → STOP, revisit design.
  Candidate next moves: widen hinge threshold from 0.15 to 0.17,
  or double iters to 1000. Do **not** ship a regressed policy.

Seed 7 / 123 fallback at the best-scoring weight only if we're
close but not crossing the gate (e.g., G2 at 0.052 vs 0.05 target).

## Artifacts & staging

1. Commit the Phase 2b code changes as ordered commits on
   `audit-fixes-2026-04-16`, **before** any training artifact:
   1. New `src/phoenix/sim_env/rewards.py` + 5 new tests in
      `tests/test_slew_sat_hinge.py`. Commit message:
      `sim_env: add slew_sat_hinge_l2 reward function + tests`.
   2. Extend `_apply_rewards` + `go2_env_cfg.py` with
      `_NEW_TERM_FACTORIES` dispatch; +1 integration test. Commit:
      `sim_env: wire slew_sat_hinge RewTerm via new-term factory`.
   3. `configs/env/flat_slewhinge.yaml` + `configs/train/ppo_flat_slewhinge.yaml`.
      Commit: `configs: add flat_slewhinge (v3b + slew-sat hinge penalty)`.
2. For each accepted run:
   - Export ONNX (`phoenix.sim2real.export --verify`).
   - Run parity gate G5.
   - Rsync run dir to T7.
3. **Staging path:** `checkpoints/phoenix-flat/gate8/policy.onnx`
   (new sibling dir under `phoenix-flat/`). Do NOT touch
   `phoenix-flat/policy.onnx` — that's stand-v2, staged for Gate 7.
   Gate 8 deploy config is a follow-on spec (the mode-switch node),
   which will reference `phoenix-flat/gate8/policy.onnx`.
4. Sync the staged `gate8/` dir to T7 at
   `/media/yusuf/T7 Storage/go2-phoenix/checkpoints/phoenix-flat/gate8/`.
5. Keep v3b artifacts at `phoenix-flat/v3b/` unchanged.

## Reporting

1. Single-row summary table per weight tried (starting-weight + any
   fallbacks), old vs new, in the style of the v4 / flat-v3b-ft
   post-mortems.
2. Append a Phase 2b section to
   `Vault/Projects/go2-phoenix/RETRAIN_2026-04-19.md` with
   before/after metrics and the accepted weight.
3. Write `docs/retrain_flat_slewhinge_2026-04-19.md` (or dated to
   actual session date) as the in-repo post-mortem.
4. Update `project_go2_phoenix.md` memory status line to note Gate 8
   candidate produced; flag the deploy-mode-switch spec as the
   remaining blocker before Gate 8 hardware.

## Out of scope

- **Deploy-layer mode-switch** (`|cmd_vel| < ε` → stand-v2,
  otherwise → flat-slewhinge). Separate spec needed — hysteresis
  boundary, switch-discontinuity handling, ROS 2 policy-node
  lifecycle. Without it, flat-slewhinge is not hardware-deployable
  at cmd≈0 even if every sim gate passes.
- Gate 7 (stand-v2) is unchanged. `phoenix-flat/policy.onnx` stays
  stand-v2.
- No hardware dryrun or Gate 8 lab day — follow-on session.
- No changes to the Phase 0 wiring beyond extending `_apply_rewards`
  for new-term factories.
- No changes to `termination` / `observation.noise` / `robot.*`
  unwired sections.

## Known risks

1. **Threshold 0.15 rad may be too tight.** If v3b already operates
   near 0.15 on non-clipping motors, the hinge fires excessively
   and tracking degrades at all three tried weights. Mitigation:
   if all weights fail similarly (all three produce tracking
   regression without slew fix), redesign with threshold 0.17
   (effectively "only penalize motors at or above the hardware
   clip").
2. **500 iters may be insufficient.** Same budget that failed
   flat-v3b-ft. The reward signal is now per-motor and much more
   targeted, but if seed-42-at-w=-50 shows monotonic gate progress
   that didn't saturate at 500 iters, bump to 1000 and retry
   before escalating weights.
3. **Phoenix has no precedent for custom MDP reward functions.**
   All prior rewards are upstream Isaac Lab terms. The new
   `sim_env/rewards.py` module is a template for future Phoenix-
   owned reward terms but introduces a new code surface. The
   existing `_REWARD_TERM_MAP` covers upstream terms only;
   `_NEW_TERM_FACTORIES` is introduced specifically for Phoenix-
   owned terms.
4. **`env.action_manager.prev_action`** may not be the right
   attribute in Isaac Lab 3.x — verify before committing the
   reward function. If not, fall back to `env.action_manager._prev_action`
   or a closure that stashes prev-action explicitly.

## Success definition

Phase 2b is "done" when:
- Accepted checkpoint at some weight in `{-5, -50, -500}` passing
  all of G1-G5 at seed 42.
- ONNX exported, parity-gated, staged at `phoenix-flat/gate8/`, and
  T7-synced.
- In-repo post-mortem + vault note + memory update committed.
- Next-action is the deploy-layer mode-switch spec (out of this
  spec's scope).
