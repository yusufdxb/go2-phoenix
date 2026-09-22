# The one-joint explanatory model

Code: `src/phoenix/condition/toy_model.py`. Tests: `tests/test_toy_model.py`.
This is a model, not evidence about the GO2. It exists to make the mechanism explicit
before the full robot, and to say in advance what the method should and should not be
able to do.

## Setup

One actuated joint holds a posture and tracks a periodic reference against a
gravity-like load, `tau_g = 3 N m * sin(q)`, inertia 0.06 kg m^2. The motor driver runs
the GO2's PD law at 1 kHz with the deploy gains, scaled by an authority factor `s`:

    tau = s * (kp (u - q) - kd dq),   kp = 25, kd = 0.5,   |tau| <= 23.5 N m

`s = 1` is nominal. This is exactly what the controlled degradation does on the robot
(`phoenix.sim2real.degradation` scales one motor's `kp`/`kd`) and what the targeted
sim term does (`scale_targeted_actuator_gains` scales one joint's DCMotor stiffness and
damping, which equal the deploy `kp`/`kd`).

The policy updates its target at 50 Hz from a one-period-old, noisy measurement:

    u = q_ref + b + k (q_ref - q_meas)

with two parameters: a feed-forward offset `b` (the sag compensation a stand policy
learns) and a proportional correction `k`. It has no memory, so it cannot identify `s`;
the deployed Phoenix actor has no history input either. "Training under a distribution
p(s)" means choosing `(b, k)` on a grid to minimise the expected tracking cost plus a
target-rate penalty (the action-rate term of the real reward).

## What it shows (run on 2026-09-22, `run_study()`, s_true = 0.6)

The residual estimator first. From the nominal policy's own tracking errors,
`s_hat = rms(e_nominal) / rms(e_degraded)` gives **0.617 for a true 0.6**. The tests
check it stays within 5 % for `s` in {0.5, 0.6, 0.8} with and without the feedback term.

Then four policies, each fitted under a different `p(s)`, evaluated at four values of
`s` (cost x 1e-4, lower is better):

| Trained on | s = 0.5 | s = 0.6 (degraded) | s = 0.7 | s = 1.0 (nominal) | fitted (b, k) |
|---|---|---|---|---|---|
| nominal U[0.95, 1.05] | 28.30 | 14.55 | 8.06 | **2.57** | (0.09, 0.8) |
| broad U[0.4, 1.2] | 16.83 | 9.32 | 6.28 | 5.57 | (0.12, 1.3) |
| targeted U[s_hat +/- 0.1] | **13.24** | **7.31** | 5.99 | 9.68 | (0.14, 1.0) |
| anchored (half nominal, half targeted) | 18.98 | 9.63 | **5.69** | 4.09 | (0.11, 1.0) |

The study was repeated with four seeds for the training samples (seeds 0 to 3; the
nominal policy and `s_hat` do not change, the others do):

| Trained on | cost at s = 0.6, seeds 0 / 1 / 2 / 3 | cost at s = 1.0, seeds 0 / 1 / 2 / 3 |
|---|---|---|
| nominal | 14.55 / 14.55 / 14.55 / 14.55 | 2.57 / 2.57 / 2.57 / 2.57 |
| broad | 9.32 / 8.52 / 9.63 / 9.63 | 5.57 / 5.43 / 4.09 / 4.09 |
| targeted | 7.31 / 7.31 / 7.32 / 7.31 | 9.68 / 9.68 / 10.23 / 9.68 |
| anchored | 9.63 / 9.63 / 8.52 / 8.52 | 4.09 / 4.09 / 5.43 / 5.43 |

Reading it honestly:

1. The nominal policy is brittle: its cost at `s = 0.6` is 5.7 times its nominal cost. Its
   sag compensation `b` is right for full authority and wrong for 60 %.
2. Training only around the measured value is consistently best at that value and at its
   unseen neighbours (0.5, 0.7), about 7.3 against 8.5 to 9.6 for every other
   distribution, and consistently worst at nominal (9.7 to 10.2 against 2.6).
3. A policy that cannot identify `s` can only pick a point on a degraded-vs-nominal
   trade-off; no single `(b, k)` is best everywhere.
4. **The anchored mixture Phoenix uses and broad randomisation are a tie on one joint.**
   With a two-parameter policy on a grid, both land on the same two neighbouring grid
   cells and swap places with the sample draw (seed 0: broad better at 0.6, anchored at
   nominal; seeds 2 and 3: the reverse).

So the one-joint model supports the brittleness and the cost of targeting, and it
supports gating on nominal. It does **not** support the hypothesis that Phoenix's
anchored distribution beats broad randomisation. If that happens on the GO2, it has to
come from what one joint cannot show: twelve joints, where a broad distribution spends
almost all of its samples on combinations the robot does not have, and a fixed
fine-tuning budget. The experiment tests exactly that, and a tie there would be
consistent with this model.

## What the model does not say

* Nothing about policies that identify `s` online (RMA-style); with memory, a broad
  distribution may lose nothing.
* Nothing about effect sizes on the GO2.
* The estimator is checked only in the linear regime. Near the effort limit, and when the
  policy's feedback compensates the error it is measured by, it is biased; the monitor
  reports a range, and the oracle arm of the experiment measures the damage.

## Reproduce

    PYTHONPATH=src python3 -c "from phoenix.condition.toy_model import run_study; print(run_study())"

About two minutes per seed on one CPU core; deterministic for a given `seed`
(`run_study(seed=k)`).
