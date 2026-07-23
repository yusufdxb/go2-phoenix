# Closed-loop intervention study — the shield does not prevent falls (and this is the useful result)

This is the experiment the whole reliability layer was building toward: not "does
the monitor *warn* before a fall" (Phase 3-4, yes) but "does engaging the
fallback on that warning *prevent* the fall". It was pre-registered, run across
three arms against a frozen controller bundle, and analysed at the block level.

**It falsifies the core premise, as built.** Enabling the shield *increases* the
fall rate, the increase is caused by the act of switching to the fallback rather
than by anything the monitor detects, and along the way the study exposed a
calibration error that would have shipped to the robot.

## Headline numbers (32 disturbed + 16 nominal blocks, 16 envs each)

| arm | disturbed fall rate | nominal fall rate | fallback engagement |
|---|---|---|---|
| unshielded | **0.254** | 0.004 | 0.000 |
| shielded | **0.434** | 0.008 | 0.988 |
| sham | 0.441 | 0.004 | 0.998 |
| oracle (perfect onset timing, 0 false positives) | **0.424** | 0.008 | 1.000 disturbed / 0.000 nominal |

| comparison (block-paired, block-bootstrap 95% CI) | difference | verdict |
|---|---|---|
| **primary**: unshielded − shielded | **−0.180** [−0.227, −0.131] | shield **HARMS** (27 of 32 blocks worse) |
| **secondary**: sham − shielded | +0.008 [−0.033, +0.049] | **no effect** — monitor timing is irrelevant |
| **oracle**: unshielded − oracle | **−0.170** [−0.215, −0.123] | perfect timing **STILL HARMS** (25 of 32 blocks worse) |
| oracle − shielded | +0.010 [−0.037, +0.057] | oracle is no better than the deployed shield |
| nominal cost: shielded − unshielded | +0.004 [−0.012, +0.023] | no measurable cost on undisturbed blocks |

The primary CI excludes zero on the wrong side of it. The secondary CI straddles
zero: the real shield and a fallback switched on an information-free schedule are
statistically indistinguishable. The oracle CI also excludes zero on the wrong
side: a detector that fires **exactly at the true disturbance onset and never
false-alarms** — the best case the monitor could ever aspire to — increases falls
by 17 points and is statistically tied with the miscalibrated deployed shield.

## What actually happened, in order

**1. The monitor over-fires on a hard reset — a calibration error, not a tuning
knob.** The shield engages on **100% of nominal blocks that never receive any
disturbance**, at a median switch tick of 18 (three ticks after the arming window
ends), while disturbances do not land until tick 100-200. A direct measurement of
the post-reset transient (`scripts/reliability_reset_transient.py`) shows the
fraction of environments scoring above the trip threshold never settles below
~12% even 100 ticks after a hard reset:

```
t=  0  100%   t= 12  52%   t= 24  17%   t= 45  12%   t= 57  12%
```

The Phase 4 threshold (p99.95, implying ~0.05% of frames over-threshold) was
calibrated on **free-running** rollouts whose resets happened asynchronously
mid-stream. The closed-loop study, and the real robot, start from a **hard
synchronised reset**. Those are different distributions, and the monitor was fit
on the wrong one. This is not fixable by lengthening the arming window: the
over-rate plateaus at ~12% per frame, so with K=3 persistence over a 500-tick
episode the episode-level engagement is ~100% regardless of arming.

**2. Because it is essentially always engaged, the study measures the fallback.**
The shield runs the static stand target (default joint pose, no active policy)
almost continuously from tick ~18 onward.

**3. The static fallback is worse than the active policy under motor
degradation.** When the motors weaken at the onset tick, the learned policy
actively compensates; the static position target does not. Handing control to it
raises the fall rate from 0.254 to 0.434.

**4. The sham arm proves the monitor contributes nothing.** A fallback engaged
with the shield's own switching frequency and timing, permuted across blocks so
it cannot know anything about the episode, produces the same fall rate as the
real shield (0.441 vs 0.434, CI includes zero). Whatever harm the shield does, it
does by switching, not by detecting.

## Why this is worth more than a positive result

A green checkmark would have said "the plumbing works". This says two concrete,
transferable things:

- **The nominal calibration set must match the deployment reset protocol.** A
  monitor fit on free-running rollouts silently mis-calibrates for a robot that
  starts from a standing reset. Any future version must calibrate on hard-reset
  nominal data and re-measure the arming window against it.
- **A static stand pose is not a safe fallback for a walking/standing policy
  under actuator degradation** — precisely the regime a reliability shield is
  meant to cover. The Simplex assumption that the fallback is a safe attractor is
  false here. The sham result means this is not rescued by better *timing*; it
  needs a better *fallback*.

## What this does NOT show

- It does not show the *idea* of latent-OOD-triggered intervention is worthless.
  It shows this instance fails for two locatable reasons. A version with (a) a
  reset-matched calibration and (b) a fallback that is actually stabilising under
  degradation is a different experiment.
- ~~It does not measure a correctly-firing shield.~~ **The oracle arm now
  measures exactly this** (see below): a shield firing only at the true onset,
  with zero false alarms, still increases falls by 17 points. The clean causal
  question is no longer open — a perfectly-timed static fallback harms.

## The oracle arm: a perfect detector does not rescue this fallback

The shielded and sham arms both engage on the reset transient (median tick ~18),
long before any disturbance, so a fair objection was: *maybe the whole result is
an artefact of the calibration error, and a correctly-firing shield would help.*
The oracle arm settles it. On disturbed blocks it switches to the static fallback
**exactly at the registered disturbance onset** and holds; on nominal blocks it
never switches at all. This is a detector with perfect recall, perfect timing,
and zero false positives — an upper bound no real monitor can beat.

It still loses. Oracle disturbed fall rate is **0.424 vs 0.254 unshielded**
(−0.170 [−0.215, −0.123], 25 of 32 blocks worse, 2 better), and it is
statistically tied with the deployed shield (oracle − shielded = +0.010,
CI includes zero). The two independent failure modes we diagnosed — calibration
mismatch and a powerless fallback — are therefore *not* additive competitors for
the blame. Removing the calibration error entirely (the oracle has none) leaves
the harm essentially unchanged. **The fallback is the whole story: under motor
degradation, handing a walking/standing policy to a frozen joint target removes
the active compliance the robot needs to stay up, whenever and however cleanly
you do it.** Recalibrating the monitor could not have saved this design; only a
fallback with real control authority under the disturbance could, and none exists
without training a new policy on degradation the current policies never saw.

## The honest next step (revised after the oracle arm)

The original plan was two fixes then a re-run: (1) recalibrate on hard-reset
nominal data so the shield fires on the disturbance, not the reset transient, and
(2) swap in an actively-stabilising fallback. **The oracle arm retires fix (1).**
A detector with perfect timing and zero false alarms already exists in this study,
and it does not beat unshielded — so recalibration, which at best turns the real
monitor into that oracle, cannot rescue the design. The calibration mismatch is a
real and shippable-to-the-robot bug worth reporting, but it is not what makes the
shield harmful.

That leaves only fix (2): a fallback with genuine control authority under motor
degradation. No such controller exists in this project. Every trained policy,
including the domain-randomised variants, saw motor scaling no lower than 0.85;
the disturbance here runs 0.30–0.55. Producing a fallback that survives it means
**training a new policy on out-of-envelope degradation**, which is a separate
research effort, not a tweak — and a *learned* fallback would in any case forfeit
the Simplex premise (no verified invariant set, no formal safety floor), turning
the contribution into risk-conditioned switching between learned controllers
rather than a runtime assurance shield.

**Decision: this instance is reported as a decisive negative result, not
re-run.** Two external reviews (one GO-with-conditions gated on exactly the oracle
experiment above, one NO-GO) converge here once the oracle arm fails the gate.
The transferable lessons stand on their own: calibrate on the deployment reset
distribution, and — the sharper one — *predicting a fall is not enough; a runtime
shield is only as good as the recovery authority of its fallback, and a static
"safe" pose has none under actuator degradation.*

## Reproduce

```bash
python scripts/reliability_closed_loop.py --freeze --out-dir reliability_eval/closed_loop
python scripts/reliability_closed_loop.py --arm unshielded --out-dir reliability_eval/closed_loop
python scripts/reliability_closed_loop.py --arm shielded   --out-dir reliability_eval/closed_loop
python scripts/reliability_closed_loop.py --arm sham       --out-dir reliability_eval/closed_loop
python scripts/reliability_closed_loop.py --arm oracle     --out-dir reliability_eval/closed_loop  # perfect-timing diagnostic
python scripts/reliability_closed_loop_analyze.py --out-dir reliability_eval/closed_loop
python scripts/reliability_reset_transient.py   # the calibration-mismatch diagnostic
```

bundle `7aedf38c7a7c9008`, protocol `93743d4ab6203736`.
