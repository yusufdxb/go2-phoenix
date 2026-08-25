# When does a runtime latent-OOD Simplex shield help? A three-regime characterization

The closed-loop study asked whether engaging a stand fallback on a latent-OOD
warning *prevents* falls. Across three disturbance regimes, run on the same
frozen policy and the same paired-block apparatus, the answer is a clean
function of one variable: **how different the fallback's behaviour is from what
the primary policy is already doing.** This document states that function and the
evidence for it. It is the honest, transferable result the project produces
before any positive-regime demo.

## The apparatus (identical across regimes)

Primary policy: `phoenix-stand-v3-h25-final`, a standing policy (commanded
velocity = 0), trained with full domain randomisation including motor scale in
[0.85, 1.15]. Fallback: the static default stand pose (blend the action toward
zero). Monitor: squared Mahalanobis on post-ELU policy latents, fit on nominal
only. Unit of analysis: the pre-registered scenario block. Disturbance injected
mid-episode at a registered onset tick (100-200), long after the shield arms.

## The three regimes

| regime | disturbance | unshielded disturbed fall rate | shield verdict |
|---|---|---|---|
| **actuator degradation** | motor stiffness+damping x [0.30, 0.55] | 0.254 | **harms**: 0.434, primary −0.180 [−0.227, −0.131]; oracle (perfect timing) 0.424, still −0.170 [−0.215, −0.123] |
| **perceptual corruption** | additive Gaussian on the policy's obs, std [0.5, 5.0] | ~0.10, and *falling* in severity (r = −0.67) | **no headroom**: the disturbance barely destabilises a standing policy |
| **OOD command** | forward velocity command [0.6, 2.0] m/s to a cmd=0 policy | ~0.00 | **no headroom**: the policy ignores the command and keeps standing |

(Motor: 32 disturbed + 16 nominal blocks x 16 envs, block-bootstrap CI. Obs and
command: 12 disturbed + 4 nominal-block pilots, unshielded arm, wide severity
range to map fall rate against severity.)

## The mechanism

A Simplex shield can only help when switching to the fallback changes the
robot's behaviour toward something safer. For a **standing** policy protected by
a **standing** fallback, that gap is nearly empty:

- Under **perceptual** or **command** disturbance, the standing policy keeps
  standing (its action is rate-limited and it was trained to be stationary), so
  it rarely falls in the first place. There is almost nothing for the fallback to
  prevent, the two behaviours coincide.
- Under **actuator** disturbance, the policy *does* fall, but the failure is
  physical: the motors that the fallback also depends on are weakened. Here the
  behaviours finally diverge, and they diverge the wrong way, the active policy
  compensates for the weak motors while the frozen pose cannot. The oracle arm
  (a perfect, false-alarm-free detector) confirms this is intrinsic to the
  fallback, not a timing or calibration artefact.

So the fall-prevention benefit of a stand-fallback shield is bounded above by the
behavioural gap between the primary policy and standing still, and for a stand
policy that gap only opens under disturbances that also disable the fallback.
**Predicting a fall is necessary but not sufficient: the fallback must be both
reachable and safer than continuing, in the specific failure mode detected.**

## What this predicts (and the next experiment)

The characterization makes a falsifiable prediction: a shield helps when the
primary policy's risky behaviour is one the fallback meaningfully retreats from.
That points squarely at a **locomotion** policy, where "stop and stand" is a
distinct, safer action than continuing to walk into a failure.

**CONFIRMED, for the fallback and not for the detector.** The identical 4-arm
study, rerun on the walking policy `phoenix-flat-v4` (monitor refit on its
nominal walking latents) under a perceptual observation-corruption disturbance,
produces a clean positive (`reliability_eval/closed_loop_walk/FINDINGS.md`): the
shield cuts the disturbed fall rate 0.238 -> 0.133 (primary +0.105
[+0.064, +0.146], 26/32 blocks), reaches the perfect-timing oracle ceiling (gap
-0.002 [-0.029, +0.023]), and costs nothing on undisturbed walking. What it does
**not** show is that the monitor's timing produces the benefit: the first sham
arm permuted switch schedules globally and so engaged in only 0.709 of disturbed
episodes against the shield's 0.982, a treatment-dose mismatch rather than a
timing contrast. Against a condition-stratified sham at matched dose the shield's
advantage is +0.004 [-0.023, +0.033] -- a blind switcher of equal frequency
captures all of it. The method works exactly where the mechanism says it should
and fails where it says it shouldn't, and in both regimes the operative variable
is the fallback's safety, not the detector's quality.

## The full picture

| regime | policy | fallback behaviourally distinct? | policy actually fails? | shield verdict |
|---|---|---|---|---|
| actuator degradation | stand | yes, but fallback also disabled | yes | **harms** |
| perceptual / command | stand | no (policy already stands) | no | **no headroom** |
| perceptual (obs OOD) | **walking** | **yes** | **yes** | **prevents falls; a dose-matched sham does too** |

The single governing variable is the top-right pair: the shield helps iff the
fallback is both **behaviourally distinct** from the primary policy and **safe**
in the detected failure mode. Both must hold; the walking + perceptual cell is
the only one of the three where they do.

## Reproduce

```bash
# motor (full study + oracle): reliability_eval/closed_loop/FINDINGS.md
python scripts/reliability_closed_loop.py --freeze --disturbance obs \
    --obs-noise-lo 0.5 --obs-noise-hi 5.0 --n-disturbed 12 --n-nominal 4 \
    --out-dir reliability_eval/closed_loop_obs_pilot
python scripts/reliability_closed_loop.py --arm unshielded --out-dir reliability_eval/closed_loop_obs_pilot
python scripts/reliability_closed_loop.py --freeze --disturbance command \
    --command-speed-lo 0.6 --command-speed-hi 2.0 --n-disturbed 12 --n-nominal 4 \
    --out-dir reliability_eval/closed_loop_cmd_pilot
python scripts/reliability_closed_loop.py --arm unshielded --out-dir reliability_eval/closed_loop_cmd_pilot
```
