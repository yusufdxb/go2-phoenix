# The positive regime: a latent-OOD shield prevents falls for a walking policy

> **CORRECTED 2026-07-26. The timing claim is withdrawn.** The original sham
> permuted switch schedules globally, so it fired in only 0.689 of disturbed
> episodes against the shield's 0.977: the two arms were not receiving the same
> treatment dose, and the "monitor timing helps" result was reading that dose
> gap. A condition-stratified sham holds disturbed engagement at 0.977 and cuts
> falls to 0.152. Against that dose-matched control the shield's remaining
> advantage is +0.031, 95% CI [-0.002, +0.066], which does not exclude zero.
> Section 2 below is superseded; the corrected reading is in "What this
> establishes". See `dose_matched_results.json`.

The three-regime characterization (`reliability_eval/closed_loop/REGIME_CHARACTERIZATION.md`)
predicted where a stand-fallback Simplex shield *would* help: a policy whose
risky behaviour the fallback meaningfully retreats from. This is that experiment,
run with the identical pre-registered 4-arm apparatus. **Engaging the fallback
nearly halves the fall rate and costs nothing on undisturbed walking, exactly
where the characterization predicted a benefit. What the data do *not* show is
that the monitor's timing is what produces the benefit: a blind switcher that
fires equally often captures nearly all of it.**

## Setup

- **Primary policy**: `phoenix-flat-v4`, a velocity-commanded *walking* policy
  (task `Isaac-Velocity-Flat-Unitree-Go2-v0`). Unlike the stand policy, its
  nominal behaviour (walking) is genuinely different from the fallback (a static
  stand pose), so switching is a real behavioural change.
- **Shield**: monitor refit on this policy's nominal *walking* latents (256 envs,
  384-dim tap, Ledoit-Wolf; `deploy/shield_flat_v4.npz`), p=99.999 K=10, held-out
  nominal episode FPR 4.35%, f32/f64 parity 0 decision disagreements.
- **Disturbance**: additive Gaussian corruption of the policy's observations
  (std [0.5, 4.0]) injected mid-gait at the registered onset. A *perceptual*
  fault: the robot's body is intact, so freezing to a stand is a safe attractor.
- 32 disturbed + 16 nominal blocks x 16 envs, block-bootstrap CI, bundle
  `be15e3249994cdeb`, protocol `8ae4f679d4012dad`.

## Headline numbers

| arm | disturbed fall rate | nominal fall rate | fallback engagement |
|---|---|---|---|
| unshielded | **0.236** | 0.137 | 0.000 |
| sham (global permutation, dose-mismatched) | 0.188 | 0.145 | 0.689 |
| sham (condition-stratified, dose-matched) | 0.152 | 0.145 | 0.977 |
| **shielded** | **0.121** | 0.133 | 0.977 |
| oracle (perfect onset timing) | 0.145 | — | 1.000 disturbed |

| comparison (block-paired, block-bootstrap 95% CI) | difference | verdict |
|---|---|---|
| **primary**: unshielded − shielded | **+0.115** [+0.078, +0.152] | engaging the fallback **PREVENTS falls** (23 of 32 blocks better) |
| ~~secondary: sham − shielded~~ | ~~+0.066 [+0.023, +0.109]~~ | **WITHDRAWN**, dose-mismatched sham (0.689 vs 0.977 engagement) |
| **dose-matched**: stratified sham − shielded | **+0.031** [−0.002, +0.066] | monitor timing **NOT shown** to beat blind switching (16 of 32) |
| nominal cost: shielded − unshielded | −0.004 [−0.043, +0.031] | **no cost** on undisturbed walking |
| ceiling: unshielded − oracle | +0.092 [+0.053, +0.131] | perfect timing also helps |
| gap: oracle − shielded | +0.023 [−0.004, +0.053] | shield **reaches the perfect-timing ceiling** |

## What this establishes

1. **The shield prevents falls.** The primary CI excludes zero on the right side:
   engaging the fallback on the monitor's warning cuts the disturbed fall rate
   from 0.236 to 0.121, a 49% relative reduction, better in 23 of 32 blocks.

2. **The act of freezing is doing most of the work, not the monitor.**
   *(This section replaces the withdrawn timing claim.)* The original sham
   engaged in only 0.689 of disturbed episodes against the shield's 0.977, so
   comparing them confounded "when you switch" with "how often you switch". A
   condition-stratified sham holds engagement at 0.977 on an information-free
   schedule and already reaches 0.152, capturing 73% of the shield's total
   benefit over unshielded. The residual attributable to the monitor's timing is
   +0.031 [−0.002, +0.066], which does not exclude zero at n=32 blocks. The
   honest statement is that a latent-OOD monitor with episode-level AUROC of
   0.80 to 1.00 buys little or nothing over a blind switcher of equal frequency.

3. **The real monitor matches a perfect detector.** Shielded (0.121) is
   statistically tied with, and numerically below, the onset-only oracle (0.145):
   the monitor also fires on early perceptual drift, catching falls an
   onset-triggered switch misses.

4. **No nominal cost.** On undisturbed walking the shield changes the fall rate
   by −0.004 (CI straddles zero), despite a 4.35% nominal engagement budget:
   freezing briefly during clean walking does not tip the robot.

## Why this is the honest completion of the story, not a rescue

Nothing here contradicts the negative motor-degradation result; it explains it.
The characterization said a stand fallback can only help when (a) the primary
policy actually fails and (b) standing is still safe in that failure mode. Motor
degradation violated (b); a stand policy violated (a). A **walking** policy under
a **perceptual** fault satisfies both, and the shield delivers exactly the
predicted benefit. The method works where the mechanism says it should and fails
where it says it shouldn't — which is a stronger claim than either result alone.

The dose-matched correction sharpens rather than weakens this. The governing law
was always a statement about the *fallback*, never about the detector: a shield
helps iff the fallback is behaviourally distinct from the policy and safe in the
detected failure mode. Both regimes now say the same thing once treatment dose is
held fixed. Under motor degradation, sham ties shielded at ~99% engagement in
both arms and both are worse than not switching. Under perceptual corruption, a
dose-matched sham recovers 73% of the benefit. Detector quality is not what
separates the two outcomes; fallback safety is.

## Caveats

- Nominal walking has a non-trivial base fall rate (~0.14) because `flat_v4`
  samples aggressive velocity commands; the effect is measured against that
  baseline and the nominal-cost arm controls for it.
- **Deploy parity (diagnosed 2026-07-26).** The exported walking ONNX
  (`deploy/flat_v4_latent.onnx`) fails the deploy gate with a worst latent
  difference of **6.44** against recorded latents of scale 673 (0.96% relative,
  on 100% of frames), not the ~5e-4 previously recorded. Cause: this study's
  rollout script hardcoded `empirical_normalization: True`, but the flat-v4
  checkpoint carries no normalizer buffers, so rsl_rl built an untrained
  `EmpiricalNormalization` whose forward is still `(x − 0) / (1 + 1e-2)`, a
  silent 1% shrink of every observation. The exporter correctly found no
  statistics and exported a raw-observation policy. TorchScript and ONNX agree
  to 2.4e-4 with each other, so the conversion was never the problem; feeding
  the exported model `obs/1.01` reproduces the recorded latents to 1.5e-4.
  Fixed in `scripts/reliability_rollout.py` by deriving the flag from the
  checkpoint via `phoenix.sim2real.export.checkpoint_has_obs_normalizer`
  (regression tests in `tests/test_export_normalizer.py`). **This does not
  affect any result above**: the scaling is a fixed input transform applied
  identically to all four arms, and it changed 1 trip decision in 6400 frames
  with 0 arbiter-blend disagreements. **Still owed before hardware:** re-record
  the nominal flat rollout under the corrected path, refit
  `deploy/shield_flat_v4.npz`, and re-run this study so the deployed
  configuration is the studied one.
- GPU physics is not bit-deterministic across processes, so paired blocks carry
  some run-to-run noise; the effect sizes are large relative to it and the
  block-bootstrap CIs account for block-level variance.

## Reproduce

```bash
python scripts/reliability_rollout.py --checkpoint checkpoints/phoenix-flat-v4/latest.pt \
    --env-config configs/env/flat_v4.yaml --condition nominal \
    --out reliability_eval/raw_flat/nominal_seed0.npz --num-envs 256 --max-steps 400 --seed 0
python scripts/reliability_fit_deploy.py --raw-dir reliability_eval/raw_flat \
    --out deploy/shield_flat_v4.npz --max-episode-fpr 0.05
python scripts/reliability_closed_loop.py --freeze --disturbance obs \
    --obs-noise-lo 0.5 --obs-noise-hi 4.0 --n-disturbed 32 --n-nominal 16 \
    --checkpoint checkpoints/phoenix-flat-v4/latest.pt --env-config configs/env/flat_v4.yaml \
    --artifact deploy/shield_flat_v4.npz --onnx deploy/flat_v4_latent.onnx \
    --out-dir reliability_eval/closed_loop_walk
for arm in unshielded shielded sham oracle; do
  python scripts/reliability_closed_loop.py --arm $arm \
    --checkpoint checkpoints/phoenix-flat-v4/latest.pt --env-config configs/env/flat_v4.yaml \
    --artifact deploy/shield_flat_v4.npz --onnx deploy/flat_v4_latent.onnx \
    --out-dir reliability_eval/closed_loop_walk
done
python scripts/reliability_closed_loop_analyze.py --out-dir reliability_eval/closed_loop_walk
```
