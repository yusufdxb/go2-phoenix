# flat-v3b-ft fine-tune — 2026-04-19 (NEGATIVE RESULT)

## Summary

Fine-tuned `phoenix-flat/2026-04-16_21-39-16/model_999.pt` (v3b) on a new
`flat_v3b_ft.yaml` env overlay (+10× `action_rate`, +4× `joint_acc`,
`rel_standing_envs` 0.02 → 0.10, `init_noise_std` 0.1) for 500 iters,
seed 42. **All flat.yaml gates failed.** Not exported, not staged. The
design is wrong, not the seed — no fallback sweep run.

## Run dir
`checkpoints/phoenix-flat-v3b-ft/2026-04-19_11-28-09` (final checkpoint
`model_499.pt`, ~15 min wall @ 10240 envs). Artifacts kept for
diagnosis, rsync'd to T7.

## Gates

| # | Gate | Threshold | Result | Status |
|---|---|---|---|---|
| G1 | stand.yaml survival | 32/32 @ 20 s | 32/32 @ 20 s | ✓ |
| G2 | flat.yaml survival | 32/32 @ 20 s | **29/32 @ 19.0 s** | ✗ |
| G3 | slew_saturation @ cmd=0 | <0.05 | **0.341** (34.1%) | ✗ |
| G4 | lin_vel_err on flat.yaml | ≤0.10 m/s | **0.619 m/s** (6× over) | ✗ |
| G5 | ang_vel_err on flat.yaml | ≤0.10 rad/s | **0.435 rad/s** (4× over) | ✗ |

Full metrics:
- `docs/rollout_flat_v3b_ft_stand_2026-04-19.json`
- `docs/rollout_flat_v3b_ft_flat_2026-04-19.json`

## v3b baseline comparison (from commit `61fae38`)

| metric | v3b | flat-v3b-ft | delta |
|---|---|---|---|
| `mean_episode_return` (flat.yaml) | 39.50 | 8.89 | **-78%** |
| `mean_lin_vel_err` (flat.yaml) | 0.091 m/s | 0.619 m/s | **7× worse** |
| `mean_ang_vel_err` (flat.yaml) | 0.087 rad/s | 0.435 rad/s | **5× worse** |
| `slew_saturation_pct` (stand.yaml) | (not measured in sim) | 0.341 | (hw was 0.302) |

flat-v3b-ft is strictly worse than v3b on every measurable axis. This is
a deeper regression than v4 caused — v4 kept lin_vel roughly intact and
only wrecked ang_vel; flat-v3b-ft wrecks both AND doesn't fix slew.

## Root cause

`action_rate_l2` is `||a_t - a_{t-1}||_2²` — L2 norm summed over all 12
motor deltas per step. A single motor slewing the full ±0.175 rad clip
contributes only ~0.03 to the L2 norm; the other 11 motors can sit near
zero and the total stays small. The policy found exactly this solution:
hit the per-motor clip on a few joints, keep others quiet, and the L2
penalty barely fires.

Evidence from the rollout JSON:
- `per_term_rewards.action_rate_l2` = -0.00134 per step (weight 0.5, so
  raw L2 norm ~0.00268 per step — tiny)
- `slew_saturation_pct` = 0.341 — a third of steps have at least one
  motor exceeding the ±0.175 rad/step clip

The metric we gated on (slew_saturation) is not the one we incentivized
(L2 action delta). They're not correlated enough at the per-motor-clip
scale. The +10× scaling didn't fix this because scale isn't the
problem — choice-of-norm is.

**`rel_standing_envs` 0.02 → 0.10 was also insufficient.** Even with 5×
more cmd=0 exposure, the policy's cmd=0 behavior at eval time is nearly
identical to v3b's (v3b hw 30.2%, flat-v3b-ft sim 34.1%). The fine-tune
seems to have preserved the cmd=0 out-of-distribution behavior from v3b
rather than shifted it.

## Decision

- Do NOT ship this checkpoint. No ONNX export. No staging. No T7 copy
  of the ONNX (the training dir is on T7 for diagnosis only).
- Do NOT run seed 7 / 123 fallback — architectural, not seed-variance.
- Gate 8 (walking) follow-up: separate spec with a direct
  slew-saturation reward term OR a rel_standing_envs curriculum ramp
  (start at 0.5, decay to 0.05). See "Phase 2b brainstorm" session
  that follows this.

## Gate 7 status

**Unaffected** — stand-v2 is staged at `checkpoints/phoenix-flat/policy.onnx`
and all gates were green. Gate 7 lab day proceeds as planned with the
stand specialist. See `docs/retrain_stand_v2_2026-04-19.md`.
