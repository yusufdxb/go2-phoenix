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

| comparison (block-paired, block-bootstrap 95% CI) | difference | verdict |
|---|---|---|
| **primary**: unshielded − shielded | **−0.180** [−0.227, −0.131] | shield **HARMS** (27 of 32 blocks worse) |
| **secondary**: sham − shielded | +0.008 [−0.033, +0.049] | **no effect** — monitor timing is irrelevant |
| nominal cost: shielded − unshielded | +0.004 [−0.012, +0.023] | no measurable cost on undisturbed blocks |

The primary CI excludes zero on the wrong side of it. The secondary CI straddles
zero: the real shield and a fallback switched on an information-free schedule are
statistically indistinguishable.

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
- It does not measure a correctly-firing shield. Because of the over-firing, no
  arm switched *in response to the disturbance*; both shielded and sham switch on
  the reset transient. The clean causal question ("does a shield that fires only
  on the disturbance help?") is still open — though the sham result predicts the
  static fallback would harm whenever it engages, regardless of timing.

## The honest next step

Not a tweak to this artifact. Two changes, then a fresh pre-registered study:

1. **Recalibrate on hard-reset nominal rollouts** and re-derive the arming window
   and threshold against that distribution, so the shield fires on the
   disturbance rather than on every reset.
2. **Replace the static stand fallback with a controller that is stabilising
   under the disturbance** (the stand-v3 policy itself, frozen, is a candidate —
   it at least acts), and re-run all three arms.

If, after both fixes, the shield still does not beat the sham, the latent-OOD
Simplex approach does not work for this platform and should be reported as such.

## Reproduce

```bash
python scripts/reliability_closed_loop.py --freeze --out-dir reliability_eval/closed_loop
python scripts/reliability_closed_loop.py --arm unshielded --out-dir reliability_eval/closed_loop
python scripts/reliability_closed_loop.py --arm shielded   --out-dir reliability_eval/closed_loop
python scripts/reliability_closed_loop.py --arm sham       --out-dir reliability_eval/closed_loop
python scripts/reliability_closed_loop_analyze.py --out-dir reliability_eval/closed_loop
python scripts/reliability_reset_transient.py   # the calibration-mismatch diagnostic
```

bundle `7aedf38c7a7c9008`, protocol `93743d4ab6203736`.
