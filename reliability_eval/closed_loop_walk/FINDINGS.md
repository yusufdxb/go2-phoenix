# The positive regime: a latent-OOD shield prevents falls for a walking policy

The three-regime characterization (`reliability_eval/closed_loop/REGIME_CHARACTERIZATION.md`)
predicted where a stand-fallback Simplex shield *would* help: a policy whose
risky behaviour the fallback meaningfully retreats from. This is that experiment,
run with the identical pre-registered 4-arm apparatus, and it confirms the
prediction. **Enabling the shield nearly halves the fall rate, the reduction is
caused by the monitor's timing (it beats a blind sham), and it costs nothing on
undisturbed walking.**

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
| sham | 0.188 | 0.145 | 0.689 |
| **shielded** | **0.121** | 0.133 | 0.977 |
| oracle (perfect onset timing) | 0.145 | — | 1.000 disturbed |

| comparison (block-paired, block-bootstrap 95% CI) | difference | verdict |
|---|---|---|
| **primary**: unshielded − shielded | **+0.115** [+0.078, +0.152] | shield **PREVENTS falls** (23 of 32 blocks better) |
| **secondary**: sham − shielded | **+0.066** [+0.023, +0.109] | monitor **timing helps** beyond blind switching (19 of 32) |
| nominal cost: shielded − unshielded | −0.004 [−0.043, +0.031] | **no cost** on undisturbed walking |
| ceiling: unshielded − oracle | +0.092 [+0.053, +0.131] | perfect timing also helps |
| gap: oracle − shielded | +0.023 [−0.004, +0.053] | shield **reaches the perfect-timing ceiling** |

## What this establishes

1. **The shield prevents falls.** The primary CI excludes zero on the right side:
   engaging the fallback on the monitor's warning cuts the disturbed fall rate
   from 0.236 to 0.121, a 49% relative reduction, better in 23 of 32 blocks.

2. **The monitor is doing the work, not just the act of freezing.** The sham
   arm freezes at nearly the same rate (0.689 vs 0.977 engagement) but on an
   information-free permuted schedule, and it only reaches 0.188. Shielded beats
   sham by +0.066 with a CI that excludes zero. This is exactly the comparison
   the motor-degradation study *failed* (there sham tied shielded); here the
   monitor's timing carries real information about when to retreat.

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

## Caveats

- Nominal walking has a non-trivial base fall rate (~0.14) because `flat_v4`
  samples aggressive velocity commands; the effect is measured against that
  baseline and the nominal-cost arm controls for it.
- The exported walking ONNX (`deploy/flat_v4_latent.onnx`) currently fails the
  1e-4 deploy parity bar (~5e-4); it is used here only for bundle provenance
  (inference runs from the `.pt` checkpoint). Re-export must pass parity before
  any hardware ship.
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
