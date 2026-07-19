# Phase 4 — Orin packaging, parity, and the real-time budget

Phase 3 answered *does policy-latent OOD scoring warn before the robot falls*.
Phase 4 answers *can the exact thing that was measured run on the robot*. Every
number below was produced by a script in `scripts/`, and every gate fails closed.

## 1. What ships

A single 588 KB artifact, `deploy/shield_stand_v3.npz`:

| contents | why |
|---|---|
| `mean` (384,) float32 | nominal latent centroid |
| `whitener` (384x384) float32 | `W = L^-1` of the shrinkage covariance |
| `meta` JSON | operating point + full provenance |

The deploy-time score is `||W (x - mu)||^2` — one matrix-vector product. The
covariance estimate, its Cholesky factor and the inversion happen once, offline,
in float64; only the whitener ships, in float32.

Provenance recorded in the artifact: source checkpoint and its SHA-256, env
config, tap indices, latent dim, Isaac Lab / rsl_rl versions, the selection rule
and its budget, and the measured per-condition evaluation.

## 2. A correction to the Phase 3 headline

Phase 3 reported **0% nominal-episode false alarms**. That number was optimistic.
The sweep split nominal *frames* randomly between fitting and calibration, and
consecutive frames of one episode are near-duplicates — so the calibration set
was scoring data the scorer had effectively already seen.

Phase 4 splits by **environment**, keeping whole episodes on one side of the
split. Under that leak-free protocol the achievable nominal-episode false-alarm
floor is not 0% but roughly **1.3–2.5%**, and the selected point sits at 3.75%.

The detection result itself is unaffected (AUROC is computed across conditions,
not across a fit/calibration split). What changes is the false-alarm figure, and
it changes in the direction that matters, so it is restated here rather than
left standing.

## 3. Selecting the operating point without cheating

The rule is fixed in advance and uses **nominal data only**:

> Among candidate `(threshold percentile, persistence K)` pairs whose *held-out*
> nominal per-episode engagement rate is within budget, minimise `K` first, then
> the threshold.

`K` is minimised first because it is the shield's *intrinsic detection delay* —
it cannot warn sooner than `K` ticks after the latent goes bad — so buying quiet
with a long persistence window forfeits exactly the fast failures the shield
exists to catch. That is not hindsight: an earlier run of this script used
"lowest threshold first", landed on `K=30`, and dropped `friction_severe`
coverage from 100% to 64%.

The shifted conditions are scored **once**, afterwards, as evaluation. No OOD
rollout influences the threshold.

**Selected:** percentile 99.99, `K=3`, trip 9198.1, clear 506.9, held-out
nominal episode FPR **0.0375**.

| condition | falls | warned | median lead |
|---|---|---|---|
| friction_moderate | 56 | 56 (100%) | 2.44 s |
| friction_severe | 1297 | 1297 (100%) | 0.68 s |
| motor_severe | 93 | 93 (100%) | 0.88 s |
| mass_moderate / mass_severe / motor_moderate | 0 | — | — |

Every fall the robust policy suffers under held-out shift is warned, with 0.68 s
(34 control ticks) of lead in the worst case.

## 4. Parity gates

**float64 fit vs float32 deploy** (16,000 samples spanning nominal and every
shift): max relative error `2.2e-6`, **0 trip-decision disagreements**.

**Exported ONNX vs the studied path** — the gate that matters, because a shield
can be perfectly calibrated and still worthless if the latent the *robot*
computes lives in a different space than the latent the monitor was *fit* on. An
action-only parity check cannot see that failure: the robot would walk correctly
while the monitor scored nonsense. `scripts/reliability_verify_deploy.py` replays
real recorded observations through the exported ONNX and compares latents,
scores, and full arbiter traces:

- 132,800 frames across all 7 conditions x 3 seeds
- max latent absolute difference `1.7e-5`
- max score relative difference `1.9e-5`
- **0 trip-decision disagreements, 0 arbiter-blend disagreements**

The `latent` output is held to the same `1e-4` export-parity bar as `action`.

## 5. Real-time budget

`scripts/reliability_bench_shield.py`, 20,000 ticks, full deploy path (score +
arbiter), GC disabled, pinned to one core:

| p50 | p99 | p99.9 | max |
|---|---|---|---|
| 7.7 us | 9.8 us | 11.6 us | **21.6 us** |

Worst tick = **0.11% of the 20 ms control period**. Traced-memory growth over
the run: 1.4 KB (the timing array itself), confirming the per-tick path does not
allocate.

Two things had to be fixed to get an honest number, and both are deployment
lessons rather than benchmarking trivia:

1. **Single-threaded BLAS.** A 384x384 matrix-vector product is far too small to
   amortise a thread fan-out; a multi-threaded BLAS pinned to one core spends its
   time in its own barriers. Setting `OPENBLAS_NUM_THREADS=1` *before* importing
   numpy took the worst-case tick from 3.6 ms to 0.024 ms — about 150x.
2. **CPU pinning.** Unpinned, the worst case was ~5 ms while the median stayed at
   8 us. That tail was core migration on a shared desktop, not the shield. The
   deployed control thread is pinned, so the benchmark pins too and records that
   it did.

## 6. On-robot wiring

`ros2_policy_node.py` gains an opt-in `reliability` block
(`configs/sim2real/deploy_stand_v3_shielded.yaml`). Per tick it runs the policy
for `["action", "latent"]`, scores the latent, and blends:

```
target = (1 - blend) * learned_target + blend * default_stand_pose
```

The fallback is the default stand pose — the same posture every abort path in
this node already commands, and the only controller here that is verified in the
Simplex sense. The blend ramps over `handoff_ticks` rather than snapping.

Fail-closed behaviour:

- ONNX without a `latent` output, or a width that disagrees with the artifact →
  refuses to construct. A policy that cannot be monitored must not run silently
  unmonitored.
- `reliability.enabled` together with `policy.mode_switch` → refuses, because the
  shield would be scoring a policy that is not the one driving the robot.
- Non-finite latent → score `+inf` → counts toward tripping, never toward
  release.
- Trip threshold and `K` are read from the artifact and are deliberately not
  overridable from the launch config; letting a launch file retune one of them
  independently is how a validated operating point quietly stops being validated.

Telemetry publishes `[blend, raw_score, trip_threshold, state_code]` on
`/phoenix/shield` every tick for the lab log.

## 7. What is still not proven

- **No hardware.** Everything here is sim plus a replay of sim observations
  through the shipped ONNX. The August CaresLab session is the first real-robot
  evidence.
- **Latency measured on mewtwo (x86), not on the Orin NX (aarch64).** The
  arithmetic is fixed and tiny, but the number must be re-measured on the target.
  The benchmark records `platform` / `machine` so the two runs are never confused.
- **Intervention remains a counterfactual.** The shield's warnings are scored
  against what the unshielded policy did; the fallback was never actually engaged
  mid-episode in a re-simulation. Closed-loop intervention is the honest next
  experiment.
- **One policy, one task.** stand-v3 on flat ground. Generalisation across
  policies and terrains is untested.

## Reproduce

```bash
# 1. fit the artifact (fails closed if no operating point meets the budget)
PYTHONPATH=src .venv/bin/python scripts/reliability_fit_deploy.py \
  --raw-dir reliability_eval/raw_stand --out deploy/shield_stand_v3.npz \
  --max-episode-fpr 0.05

# 2. export the latent-emitting policy
PYTHONPATH=src $HOME/Sim/isaac-sim-venv/bin/python -m phoenix.sim2real.export \
  --checkpoint checkpoints/phoenix-stand-v3-h25-final/latest.pt \
  --output deploy/stand_v3_latent.onnx --emit-latent --verify

# 3. end-to-end gate: exported path must reproduce the studied one exactly
PYTHONPATH=src $HOME/Sim/isaac-sim-venv/bin/python scripts/reliability_verify_deploy.py

# 4. real-time budget
PYTHONPATH=src .venv/bin/python scripts/reliability_bench_shield.py --budget-ms 2.0
```
