---
title: Phoenix Gate 8 — deploy-layer mode switch (stand-v2 @ cmd=0, v3b @ walk)
date: 2026-04-19
status: draft
supersedes: 2026-04-19-phoenix-gate8-fromscratch-design.md (training-side approach exhausted; all four reward-based runs produced either failed slew or failed tracking)
---

# Phoenix Gate 8 — deploy-layer mode switch

## Objective

Ship a two-policy ROS 2 runtime that (a) runs `stand-v2` when the
commanded velocity is near zero and (b) runs `v3b` when it isn't, with
a bounded hysteresis + linear-blend transition between the two. No
retraining. Hardware target:

- `|cmd_vel|` ≈ 0 → slew_saturation_pct ≤ 0.05 (stand-v2 already shows
  0.00254 @ cmd=0 in sim; 2026-04-18 dryrun confirmed the problem was
  specifically v3b @ cmd=0 at 30.23%).
- `|cmd_vel|` > 0.15 m/s or |yaw| > 0.25 rad/s → v3b's tracking
  (lin_vel_err 0.091 m/s / ang_vel_err 0.087 rad/s in sim).
- Transition between modes must not trip attitude abort, collapse
  abort, or saturate the per-step slew clip.

## Background

This session (2026-04-19) produced four training-side attempts at a
single-policy v3b replacement:

| run | init | slew | lin_err | ang_err |
|---|---|---|---|---|
| v3b baseline | scratch | 0.302 hw | 0.091 | 0.087 |
| flat-v3b-ft | ft v3b | 0.341 | 0.619 | 0.435 |
| flat-slewhinge (w=-50) | ft v3b | 0.123 | 0.623 | 0.607 |
| flat-slewhinge-w5 (w=-5) | ft v3b | 0.186 | 0.572 | 0.576 |
| flat-scratch | scratch | 0.00254 | 0.579 | 0.658 |

No reward weight in `{-5, -50}` at any init produced a policy that
cleared both slew and tracking gates simultaneously. See
`docs/retrain_flat_scratch_2026-04-19.md` for the reward-landscape-
dominance analysis that killed the single-policy path.

Parallel observation: the 2026-04-18 hardware dryrun saturated at
30.23% **specifically at `cmd_vel = [0, 0, 0]`**. The nonzero-cmd slew
was never measured in isolation on hardware; v3b's sim slew at nonzero
cmd (needs measurement) is likely already in-gate. The immediate
hardware failure mode is a zero-command failure mode, which stand-v2
fully solves.

## Design

### State machine (runs every 20 ms = 50 Hz)

```
      cmd_vel magnitude
      |cmd| < enter_stand_thresh (0.05)
   ┌───────────────────────────┐
   ▼                           │
STAND ───────────► TRANS_TO_WALK ───────► WALK
(stand-v2)   |cmd| >            (25 ticks,    (v3b)
              enter_walk_       linear blend)
              thresh (0.15)
   ▲                                        │
   │                                        ▼
TRANS_TO_STAND ◄──────────────── (|cmd| < enter_stand_thresh)
(25 ticks, linear blend)
```

- Hysteresis band `[0.05, 0.15]` prevents mode-flutter from noisy
  teleop or small command jitter.
- Magnitude function: `max(sqrt(vx^2 + vy^2), |vyaw| * yaw_scale)` with
  `yaw_scale = 0.3` to put a 1 rad/s yaw on par with a 0.3 m/s lin
  command. Tunable via deploy.yaml.
- Transition duration: 25 ticks = 0.5 s. Linear blend of joint
  **targets** (not actions) between the two policies.
- Both policies receive identical obs each tick (shared obs builder).
  Each maintains its own `last_action`; on mode-entry, the inactive
  policy's `last_action` is zeroed to match its training distribution.

### Two ONNX sessions, one active target

```python
stand_session = ort.InferenceSession(cfg["policy"]["stand_onnx_path"])
walk_session  = ort.InferenceSession(cfg["policy"]["walk_onnx_path"])

# every tick:
obs = build_obs(...)
stand_action = stand_session.run(["action"], {"obs": obs})[0][0]
walk_action  = walk_session.run(["action"],  {"obs": obs})[0][0]
stand_target = default_q + action_scale * stand_action
walk_target  = default_q + action_scale * walk_action

if state == STAND:       target = stand_target
elif state == WALK:      target = walk_target
elif state == TRANS_TO_WALK:
    alpha = blend_progress  # 0 → 1 over 25 ticks
    target = (1 - alpha) * stand_target + alpha * walk_target
elif state == TRANS_TO_STAND:
    alpha = blend_progress
    target = (1 - alpha) * walk_target + alpha * stand_target

target = clip_to_limits(target, q)  # shared slew+limit clipping
publish(target)
```

Both sessions run every tick. At 50 Hz on Jetson Orin NX 16 GB, each
ONNX inference is ~0.3 ms (measured); double that is well inside the
20 ms budget. No need for "sleeping" policies.

### Why blend joint targets, not actions

Action blending would hand the *blended action* back to neither policy
as `last_action`, breaking the temporal coherence both policies
expect. Target blending keeps each policy's internal state honest; the
blended target only hits the robot, never the obs vector.

### `last_action` contract during transitions

- In STAND state: `stand_session.last_action` is updated; `walk_session.last_action` is held at zero.
- Entering TRANS_TO_WALK: `walk_session.last_action` is reset to zero at tick 0, then updated each tick thereafter.
- In WALK state: `walk_session.last_action` is updated; `stand_session.last_action` is held at zero.
- Entering TRANS_TO_STAND: `stand_session.last_action` is reset to zero at tick 0, then updated each tick thereafter.
- Rationale: both policies are trained with obs that include
  `last_action`; feeding them their own action keeps the obs on-
  distribution. Cross-feeding would push one policy off-distribution
  during transitions.

### Safety (unchanged)

- Estop, sensor-freshness, attitude, first-message, max-runtime
  watchdogs all gate motion before the mode-switch logic. Same abort
  semantics as today. `_publish_default_pose()` on any failure
  regardless of mode.
- Per-step slew clip (`per_step_clip_array`, ±0.175 rad/step) is
  applied *after* blending. Guarantees hardware slew constraint
  regardless of policy output.
- Joint position-margin + velocity clip unchanged.

## Config changes

`configs/sim2real/deploy.yaml`:

```yaml
policy:
  # Legacy single-policy path (fallback for one-policy mode).
  onnx_path: "checkpoints/phoenix-flat/policy.onnx"  # unchanged

  # New: two-policy mode-switch.
  mode_switch:
    enabled: true
    stand_onnx_path: "checkpoints/phoenix-stand-v2/policy.onnx"
    walk_onnx_path:  "checkpoints/phoenix-flat/v3b/policy.onnx"  # (see migration below)
    enter_walk_thresh: 0.15   # m/s-equivalent
    enter_stand_thresh: 0.05  # m/s-equivalent (hysteresis band 0.10)
    yaw_scale: 0.3            # rad/s → m/s equivalent for magnitude
    transition_ticks: 25      # 0.5 s @ 50 Hz
    initial_state: "stand"    # always boot into stand
```

`mode_switch.enabled: false` → node behaves exactly as today (single
ONNX at `policy.onnx_path`). This keeps a clean one-flag rollback.

### ONNX migration

The v3b ONNX needs to be back-staged since stand-v2 currently lives at
`checkpoints/phoenix-flat/policy.onnx`. Proposed layout:

```
checkpoints/phoenix-flat/
  policy.onnx        # stays: Gate 7 stand-v2 (legacy single-policy path)
  v3b/
    policy.onnx      # NEW: v3b walking policy (re-exported from v3b .pt)
    policy.onnx.data
    policy.pt

checkpoints/phoenix-stand-v2/
  policy.onnx        # already present (Phase 1 export)
  policy.onnx.data
  policy.pt
```

Re-export v3b via:
```
./scripts/deploy.sh --checkpoint checkpoints/phoenix-flat/2026-04-16_21-39-16/model_999.pt \
                    --out-dir checkpoints/phoenix-flat/v3b/
```
Then verify parity: `phoenix.sim2real.verify_deploy` on each ONNX independently against its source `.pt`.

## Code changes

| File | Change |
|---|---|
| `src/phoenix/sim2real/mode_switch.py` | **new** — pure-Python state machine (no rclpy). `compute_state(prev_state, cmd_vel, cfg) → (new_state, blend_alpha)`. 100% CI-testable. |
| `src/phoenix/sim2real/ros2_policy_node.py` | modify — if `mode_switch.enabled`, load two sessions; per-tick run both, route through `mode_switch.compute_state`, blend targets. ~80 LOC delta. |
| `tests/test_mode_switch.py` | **new** — 8–10 tests on the state machine: thresholds, hysteresis, blend_alpha progression, boot in stand, abort handling. Pure-numpy. |
| `tests/test_ros2_policy_parity.py` | modify — extend the existing parity replay to exercise a cmd step 0 → 0.3 → 0 through the two-policy path and verify no NaN + clip respected. |
| `configs/sim2real/deploy.yaml` | modify — add `mode_switch` block, default `enabled: false` at commit time (turn on in lab). |
| `docs/deploy_mode_switch_runbook.md` | **new** — lab-day bringup checklist. |

## Gate plan

### Offline / CI gates

- **CI-1** — `pytest tests/test_mode_switch.py` — all state transitions, hysteresis, blend math. Targets: 100% branch coverage on `mode_switch.py`.
- **CI-2** — `pytest tests/test_ros2_policy_parity.py` — replay a pre-recorded cmd sequence `[0] * 100 + [0.3] * 100 + [0] * 100` through the two-policy node (mocked rclpy). Assert: no NaN, targets inside `[default_q - 2.0, default_q + 2.0]` rad, per-step delta ≤ 0.175 rad, exactly 2 state transitions (STAND→WALK, WALK→STAND) with 25-tick blends each.
- **CI-3** — `phoenix.sim2real.verify_deploy` on each of `stand-v2` and `v3b` ONNX independently. `max_diff < 1e-4` each. Existing tool, new inputs.

### Hardware gates (CaresLab bringup)

Preconditions: T7 sync + Jetson scp of both ONNX pairs (stand-v2 already synced 2026-04-19; v3b re-export + rsync needed).

- **G7** — **live stand** (mode_switch enabled, cmd=0 throughout). Target: 32/32 10-second holds, no abort. Stand-v2 only; walking session ticks but its output is never published.
- **G8a** — **cmd step 0 → 0.3 m/s → 0**. Forward walk for 5 s, stop for 5 s. Target: mode flips on schedule, no attitude abort, no collapse, slew_saturation_pct < 0.05 averaged over the window.
- **G8b** — **cmd step 0 → 0.5 m/s → 0**. Same schedule, faster walk.
- **G8c** — **yaw step 0 → 0.5 rad/s → 0**. Spin in place for 5 s.
- **G8d** — **combined**: `(0.3, 0, 0.3)` for 5 s → `(0, 0, 0)` for 5 s. Walk-and-turn.
- **G8e** — **rapid toggle**: 10 cycles of 1-second walk / 1-second stop. Confirms hysteresis isn't flutter-prone under aggressive input.

If G7+G8a+G8d pass, Gate 8 is DONE for this hardware session. G8b/c/e are nice-to-haves to characterize the envelope.

## Success criteria

**Full success:** CI-1, CI-2, CI-3 pass. Hardware G7, G8a, G8d pass
with slew_saturation_pct < 0.05 in both mode regimes.
→ Tag `v0.3.0-gate8-mode-switch`. Update `PHOENIX_NEXT_STEPS.md` to
replace the v3b-retrain ask with mode-switch status.

**Partial success:** CI all green, G7 pass, G8a shows stable walk but
slew 0.05-0.10. → Ship as `v0.3.0-rc1`; document the elevated slew as
known non-blocker, since it's bounded by v3b's prior sim number
(needs measurement) and the node's per-step slew clip provides a
hard floor.

**Negative:** transition trips attitude abort or collapse. → Back out
to single-policy + stand-v2 at cmd=0 only (no walking). Open a new
spec for a longer transition window (50 ticks) or a different blend
shape (cosine instead of linear).

## Risk log

1. **Both ONNX sessions double CPU load.** 2 × ~0.3 ms = ~0.6 ms per tick on Jetson, well under 20 ms budget. Sanity-check with `time.monotonic_ns()` wrap in parity replay.
2. **Target blend produces a discontinuity at transition boundaries.** Mitigated by the 25-tick linear interp + per-step slew clip. Worst case the clip itself smooths a step input; this is a feature, not a bug.
3. **`last_action` off-distribution during blend.** Worst case one policy is fed its own last_action from before the blend; continuity of `last_action` through the blend is preserved per the contract above. If sim replay shows attitude drift, lengthen transition to 50 ticks.
4. **ONNX path confusion.** `phoenix-flat/policy.onnx` currently IS stand-v2 (Phase 1 staging). The spec keeps that and adds `phoenix-flat/v3b/policy.onnx` for walking. Any tooling that reads `policy.onnx_path` unchanged still gets stand-v2 — existing single-policy bringup unaffected. `mode_switch.enabled: true` is the explicit opt-in.
5. **Teleop cmd jitter triggers mode flutter.** Hysteresis band 0.10 m/s-equivalent between enter_walk (0.15) and enter_stand (0.05). Rapid toggle gate G8e catches pathological cases before shipping.

## What is NOT in this spec

- No retraining of either policy.
- No changes to the failure detector, the parquet logger, or the estop pipeline.
- No attempt to unify the two policies into one via distillation. That's a separate research direction; only revisit if mode-switch deploys fine but stakeholders want a single ONNX for another reason.
- No deploy-layer changes on hardware beyond `ros2_policy_node.py` + `deploy.yaml`. Bridges, estop publisher, deadman, lowcmd_bridge — all unchanged.

## Budget

- Implementation: ~4 hours (mode_switch.py + tests + node wiring + parity replay extension).
- Lab bringup: 1 session. Gates G7/G8a/G8d = ~30 min of robot-on time plus usual dryrun overhead.
- No GPU time required.
