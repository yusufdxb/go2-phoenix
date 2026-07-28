# The positive regime: a latent-OOD shield prevents falls for a walking policy

> **RE-RUN 2026-07-27 on a corrected artifact.** The numbers below come from a
> full re-run of the frozen protocol after an observation-normalization defect
> was fixed in both recording harnesses (see "Deploy parity" below). The 48
> scenario blocks are byte-identical to the 2026-07-24 freeze; only the shield
> artifact they were run against changed. **The primary result survives**
> (+0.105 against a previous +0.115, heavily overlapping intervals). The
> 2026-07-26 withdrawal of the monitor-timing claim is *reinforced*: against a
> dose-matched sham the shield's advantage is now +0.004, 95% CI
> [-0.023, +0.033], flatly null. The pre-correction arms, protocol, bundle and
> results are kept under `superseded_2026-07-24/` and re-analyse with the same
> script.

The three-regime characterization (`reliability_eval/closed_loop/REGIME_CHARACTERIZATION.md`)
predicted where a stand-fallback Simplex shield *would* help: a policy whose
risky behaviour the fallback meaningfully retreats from. This is that experiment,
run with the identical pre-registered 4-arm apparatus. **Engaging the fallback
cuts the fall rate almost in half and costs nothing on undisturbed walking,
exactly where the characterization predicted a benefit. What the data do *not*
show is that the monitor's timing is what produces the benefit: a blind switcher
that fires equally often captures all of it.**

## Setup

- **Primary policy**: `phoenix-flat-v4`, a velocity-commanded *walking* policy
  (task `Isaac-Velocity-Flat-Unitree-Go2-v0`). Unlike the stand policy, its
  nominal behaviour (walking) is genuinely different from the fallback (a static
  stand pose), so switching is a real behavioural change.
- **Shield**: monitor refit on this policy's nominal *walking* latents (256 envs,
  384-dim tap, Ledoit-Wolf; `deploy/shield_flat_v4.npz`), p=99.999 K=10, trip
  7389.71, held-out nominal episode FPR 4.32%, f32/f64 parity 4.7e-07 with 0
  decision disagreements.
- **Disturbance**: additive Gaussian corruption of the policy's observations
  (std [0.5, 4.0]) injected mid-gait at the registered onset. A *perceptual*
  fault: the robot's body is intact, so freezing to a stand is a safe attractor.
- 32 disturbed + 16 nominal blocks x 16 envs, block-bootstrap CI, bundle
  `d17589571893a0d6`, protocol `1fb0c491034e86be`.

## Headline numbers

| arm | disturbed fall rate | nominal fall rate | fallback engagement |
|---|---|---|---|
| unshielded | **0.238** | 0.172 | 0.000 |
| sham (global permutation, dose-mismatched) | 0.174 | 0.102 | 0.709 |
| sham (condition-stratified, dose-matched) | 0.137 | 0.129 | 0.982 |
| **shielded** | **0.133** | 0.137 | 0.982 |
| oracle (perfect onset timing) | 0.131 | 0.137 | 1.000 disturbed, 0.000 nominal |

| comparison (block-paired, block-bootstrap 95% CI) | difference | verdict |
|---|---|---|
| **primary**: unshielded - shielded | **+0.105** [+0.064, +0.146] | engaging the fallback **PREVENTS falls** (26 of 32 blocks better) |
| ~~secondary: sham - shielded~~ | ~~+0.041 [+0.008, +0.076]~~ | **WITHDRAWN**, dose-mismatched sham (0.709 vs 0.982 engagement) |
| **dose-matched**: stratified sham - shielded | **+0.004** [-0.023, +0.033] | monitor timing buys **nothing** over blind switching (12 vs 11) |
| nominal cost: shielded - unshielded | -0.035 [-0.059, -0.012] | **no cost**; the estimate sits inside the noise floor (see the control row) |
| ceiling: unshielded - oracle | +0.107 [+0.068, +0.148] | perfect timing also helps |
| gap: oracle - shielded | -0.002 [-0.029, +0.023] | shield **reaches the perfect-timing ceiling** |
| negative control: oracle - unshielded, nominal blocks | -0.035 [-0.074, +0.008] | identical treatment, true effect 0: the run-to-run noise floor |

**Superseded (2026-07-24, pre-normalization-fix artifact)**, same 48 blocks,
recomputable from `superseded_2026-07-24/`:

| comparison | superseded | corrected |
|---|---|---|
| primary: unshielded - shielded | +0.115 [+0.078, +0.152] | +0.105 [+0.064, +0.146] |
| dose-matched sham - shielded | +0.031 [-0.002, +0.066] | +0.004 [-0.023, +0.033] |
| nominal cost | -0.004 [-0.043, +0.031] | -0.035 [-0.059, -0.012] |
| ceiling: unshielded - oracle | +0.092 [+0.053, +0.131] | +0.107 [+0.068, +0.148] |
| gap: oracle - shielded | +0.023 [-0.004, +0.053] | -0.002 [-0.029, +0.023] |
| unshielded / shielded disturbed fall rate | 0.236 / 0.121 | 0.238 / 0.133 |

Every difference between those two columns is within the study's own noise floor.
No conclusion changes direction; the one that moves at all, the dose-matched
comparison, moves further toward the null it was already reported as.

## What this establishes

1. **The shield prevents falls.** The primary CI excludes zero on the right side:
   engaging the fallback on the monitor's warning cuts the disturbed fall rate
   from 0.238 to 0.133, a 44% relative reduction, better in 26 of 32 blocks.

2. **The act of freezing is doing the work, not the monitor.** The original sham
   engaged in only 0.709 of disturbed episodes against the shield's 0.982, so
   comparing them confounded "when you switch" with "how often you switch". A
   condition-stratified sham holds engagement at 0.982 on an information-free
   schedule and reaches 0.137, statistically indistinguishable from the shield's
   0.133. The residual attributable to the monitor's timing is +0.004
   [-0.023, +0.033]. The honest statement is that a latent-OOD monitor with
   episode-level AUROC of 0.80 to 1.00 buys nothing measurable over a blind
   switcher of equal frequency in this regime.

3. **The real monitor matches a perfect detector.** Shielded (0.133) is
   statistically tied with the onset-only oracle (0.131), gap -0.002
   [-0.029, +0.023]. Read together with (2) this says the ceiling itself is low:
   even perfect timing is not what generates the benefit.

4. **No nominal cost.** On undisturbed walking the shield changes the fall rate
   by -0.035, that is, in the *helpful* direction, but that is not a real effect:
   the negative-control contrast (oracle minus unshielded on the same nominal
   blocks, where the oracle never engages and the true difference is zero by
   construction) returns the same -0.035. The defensible claim is the one the
   study was designed to make: a 4.32% nominal engagement budget causes no
   detectable increase in falls during clean walking.

## Why this is the honest completion of the story, not a rescue

Nothing here contradicts the negative motor-degradation result; it explains it.
The characterization said a stand fallback can only help when (a) the primary
policy actually fails and (b) standing is still safe in that failure mode. Motor
degradation violated (b); a stand policy violated (a). A **walking** policy under
a **perceptual** fault satisfies both, and the shield delivers exactly the
predicted benefit. The method works where the mechanism says it should and fails
where it says it shouldn't, which is a stronger claim than either result alone.

The dose-matched correction sharpens rather than weakens this. The governing law
was always a statement about the *fallback*, never about the detector: a shield
helps iff the fallback is behaviourally distinct from the policy and safe in the
detected failure mode. Both regimes say the same thing once treatment dose is
held fixed. Under motor degradation, sham ties shielded at ~99% engagement in
both arms and both are worse than not switching. Under perceptual corruption a
dose-matched sham matches the shield outright. Detector quality is not what
separates the two outcomes; fallback safety is.

## Caveats

- Nominal walking has a non-trivial base fall rate (~0.14 to 0.17) because
  `flat_v4` samples aggressive velocity commands; the effect is measured against
  that baseline and the nominal-cost arm controls for it.
- **Run-to-run noise is about 0.035 in fall rate.** GPU physics is not
  bit-deterministic across processes, so two arms receiving identical treatment
  still diverge. The negative-control row measures that directly rather than
  assuming it away, and it is why the nominal-cost interval excluding zero is not
  read as an effect. The disturbed-block effects are three times that floor.
- **Deploy parity (diagnosed 2026-07-26, closed 2026-07-27).** The exported
  walking ONNX previously failed the deploy gate with a worst latent difference
  of 6.44 against recorded latents of scale 673, because both recording
  harnesses hardcoded `empirical_normalization: True` while the flat-v4
  checkpoint carries no normalizer buffers: rsl_rl then built an untrained
  `EmpiricalNormalization` whose forward is still `(x - 0) / (1 + 1e-2)`, a
  silent 1% shrink of every observation. The exporter, correctly finding no
  statistics, had exported a raw-observation policy, so the two paths computed
  different functions of the same observation. Both
  `scripts/reliability_rollout.py` and `scripts/reliability_closed_loop.py` now
  derive the flag from the checkpoint via
  `phoenix.sim2real.export.checkpoint_has_obs_normalizer` and record the resolved
  value in their metadata (regression tests in `tests/test_export_normalizer.py`).
  The rollout was re-recorded, the artifact refit, and this study re-run against
  it, so the deployed configuration is now the studied one. The gate passes:
  6400 frames, worst latent absolute difference **2.29e-04** (3.4e-07 relative),
  worst score relative difference 1.5e-06, **0 trip-decision and 0 arbiter-blend
  disagreements**. The exported ONNX itself never changed; it is byte-identical
  to a fresh re-export, and only the recording was ever wrong.
- **The exporter's own absolute `1e-4` parity bar is not meaningful for this
  policy, and it still fails.** Because flat-v4's observations are not
  normalized, its activations and pre-clip actions are of order `10^3`, so a
  float32 rounding floor of a few `1e-4` absolute is unavoidable. Measured on the
  recorded observations, ONNX differs from float32 torch by 4.9e-04 (action) and
  4.0e-04 (latent), while float32 torch differs from float64 torch by 6.7e-04 and
  2.4e-04. ONNX is no further from float64 than single-precision torch is, and
  every relative error is about 5e-07, at float32 eps. That bar should be
  relative rather than absolute; until it is, the deploy gate above (which
  compares against the *studied* latents and budgets zero decision
  disagreements) is the check that carries the weight.
- The primary comparison is against a *non-switching* baseline, so it establishes
  that switching helps, not that this monitor is the right way to decide when.
  Point (2) is the load-bearing negative and should always be quoted with the
  positive.

## Reproduce

```bash
# 1. record nominal walking latents (normalization is read from the checkpoint)
python scripts/reliability_rollout.py --checkpoint checkpoints/phoenix-flat-v4/latest.pt \
    --env-config configs/env/flat_v4.yaml --condition nominal \
    --out reliability_eval/raw_flat/nominal_seed0.npz --num-envs 256 --max-steps 400 --seed 0
# 2. fit the deployable artifact
python scripts/reliability_fit_deploy.py --raw-dir reliability_eval/raw_flat \
    --out deploy/shield_flat_v4.npz --max-episode-fpr 0.05
# 3. deploy gate: the exported path must reproduce the studied one
OPENBLAS_NUM_THREADS=1 python scripts/reliability_verify_deploy.py \
    --onnx deploy/flat_v4_latent.onnx --artifact deploy/shield_flat_v4.npz \
    --raw-dir reliability_eval/raw_flat
# 4. freeze the bundle + protocol, then run every arm against it
python scripts/reliability_closed_loop.py --freeze --disturbance obs \
    --obs-noise-lo 0.5 --obs-noise-hi 4.0 --n-disturbed 32 --n-nominal 16 \
    --checkpoint checkpoints/phoenix-flat-v4/latest.pt --env-config configs/env/flat_v4.yaml \
    --artifact deploy/shield_flat_v4.npz --onnx deploy/flat_v4_latent.onnx \
    --out-dir reliability_eval/closed_loop_walk
for arm in unshielded shielded sham sham_stratified oracle; do
  python scripts/reliability_closed_loop.py --arm $arm \
    --checkpoint checkpoints/phoenix-flat-v4/latest.pt --env-config configs/env/flat_v4.yaml \
    --artifact deploy/shield_flat_v4.npz --onnx deploy/flat_v4_latent.onnx \
    --out-dir reliability_eval/closed_loop_walk
done
python scripts/reliability_closed_loop_analyze.py --out-dir reliability_eval/closed_loop_walk
# the superseded 2026-07-24 study re-analyses the same way:
python scripts/reliability_closed_loop_analyze.py \
    --out-dir reliability_eval/closed_loop_walk/superseded_2026-07-24
```
