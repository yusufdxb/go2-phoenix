# Phase 4, Orin packaging, parity, and the real-time budget

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

The deploy-time score is `||W (x - mu)||^2`, one matrix-vector product. The
covariance estimate, its Cholesky factor and the inversion happen once, offline,
in float64; only the whitener ships, in float32.

Provenance recorded in the artifact: source checkpoint and its SHA-256, env
config, tap indices, latent dim, Isaac Lab / rsl_rl versions, the selection rule
and its budget, and the measured per-condition evaluation.

## 2. A correction to the Phase 3 headline

Phase 3 reported **0% nominal-episode false alarms**. That number was optimistic.
The sweep split nominal *frames* randomly between fitting and calibration, and
consecutive frames of one episode are near-duplicates, so the calibration set
was scoring data the scorer had effectively already seen.

Phase 4 splits by **environment**, keeping whole episodes on one side of the
split. Under that leak-free protocol the achievable nominal-episode false-alarm
floor is not 0% but roughly **0.5–2.5%**, and the selected point sits at 4.17%.

The detection result itself is unaffected (AUROC is computed across conditions,
not across a fit/calibration split). What changes is the false-alarm figure, and
it changes in the direction that matters, so it is restated here rather than
left standing.

## 3. Selecting the operating point without cheating

The rule is fixed in advance and uses **nominal data only**:

> Among candidate `(threshold percentile, persistence K)` pairs whose *held-out*
> nominal per-episode engagement rate is within budget, minimise `K` first, then
> the threshold.

`K` is minimised first because it is the shield's *intrinsic detection delay* , 
it cannot warn sooner than `K` ticks after the latent goes bad, so buying quiet
with a long persistence window forfeits exactly the fast failures the shield
exists to catch. That is not hindsight: an earlier run of this script used
"lowest threshold first", landed on `K=30`, and dropped `friction_severe`
coverage from 100% to 64%.

The shifted conditions are scored **once**, afterwards, as evaluation. No OOD
rollout influences the threshold.

**Selected:** percentile 99.95, `K=3`, trip 6385.4, clear 475.9, held-out
nominal episode FPR **0.0417** (8 of 192 episodes).

| condition | falls | warned | lead to decision | lead to full fallback | episodes engaged |
|---|---|---|---|---|---|
| friction_moderate | 56 | 56 (100%) | 2.40 s | 2.20 s | 73.5% |
| friction_severe | 1297 | 1297 (100%) | 0.64 s | 0.44 s | 99.9% |
| motor_severe | 93 | 93 (100%) | 0.84 s | 0.64 s | 100% |
| mass_moderate | 0 |, |, |, | 37.5% |
| mass_severe | 0 |, |, |, | 92.7% |
| motor_moderate | 0 |, |, |, | 100% |

Lead is measured to the **decision** tick (the end of the K-run, which is when
the arbiter actually trips), not to the start of the run, reporting from the run
start overstates the margin by K-1 ticks. The full-fallback column further
subtracts the handoff ramp and is the margin that physically matters.

**Read the engagement column before believing the warn column.** `motor_moderate`
engages on 100% of episodes and produces zero falls; `mass_severe` engages on
92.7% with zero falls; `friction_moderate` engages ~141 episodes for 56 falls.
Episode-level precision is therefore low, the monitor is demonstrably good at
detecting *that the physics changed*, and this study does **not** establish that
it detects *that a fall is coming*. The shifts are also static from reset, which
makes the AUROC result partly an environment-classification result. Treating this
as a safety claim requires the closed-loop intervention experiment (§8).

## 4. Parity gates

**float64 fit vs float32 deploy** (16,000 samples spanning nominal and every
shift): max relative error `2.2e-6`, **0 trip-decision disagreements**.

**Exported ONNX vs the studied path**, the gate that matters, because a shield
can be perfectly calibrated and still worthless if the latent the *robot*
computes lives in a different space than the latent the monitor was *fit* on. An
action-only parity check cannot see that failure: the robot would walk correctly
while the monitor scored nonsense. `scripts/reliability_verify_deploy.py` replays
real recorded observations through the exported ONNX and compares latents,
scores, and full arbiter traces:

- 134,400 frames across all 7 conditions x 3 seeds
- max latent absolute difference `1.7e-5`
- max score relative difference `1.6e-5`
- **0 trip-decision disagreements, 0 arbiter-blend disagreements**

The `latent` output is held to the same `1e-4` export-parity bar as `action`.

## 5. Real-time budget

`scripts/reliability_bench_shield.py`, 20,000 ticks, full deploy path (score +
arbiter), GC disabled, pinned to one core:

| p50 | p99 | p99.9 | max |
|---|---|---|---|
| 7.4 us | 10.1 us | 12.2 us | **26.9 us** |

Worst tick = **0.13% of the 20 ms control period**. Traced-memory growth over
the run: 1.5 KB (the timing array itself), confirming the per-tick path does not
allocate.

Two things had to be fixed to get an honest number, and both are deployment
lessons rather than benchmarking trivia:

1. **Single-threaded BLAS.** A 384x384 matrix-vector product is far too small to
   amortise a thread fan-out; a multi-threaded BLAS pinned to one core spends its
   time in its own barriers. Setting `OPENBLAS_NUM_THREADS=1` *before* importing
   numpy took the worst-case tick from 3.6 ms to 0.024 ms, about 150x.
2. **CPU pinning.** Unpinned, the worst case was ~5 ms while the median stayed at
   8 us. That tail was core migration on a shared desktop, not the shield. The
   deployed control thread is pinned, so the benchmark pins too and records that
   it did.

**This number is a floor, not a deployment guarantee.** Re-running the same
benchmark while another heavy process was active pushed the worst case back to
4.0 ms with an unchanged 7.9 us median. The benchmark isolates the shield (pinned
core, single-threaded BLAS, GC off, tight loop); the ROS node does none of those
and additionally pays for ONNX inference, callbacks, DDS and logging. The
deployment-relevant measurement is end-to-end loop latency on the Orin under
realistic load, and it has not been taken.

## 6. On-robot wiring

`ros2_policy_node.py` gains an opt-in `reliability` block
(`configs/sim2real/deploy_stand_v3_shielded.yaml`). Per tick it runs the policy
for `["action", "latent"]`, scores the latent, and blends:

```
target = (1 - blend) * learned_target + blend * default_stand_pose
```

The fallback is the default stand pose, the same posture every abort path in
this node already commands, and the only controller here with any
independent standing; see the caveat in section 8 about calling it "verified". The blend ramps over `handoff_ticks` rather than snapping.

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

## 7. Corrections made after an adversarial review (2026-07-19)

An external review (codex) of the committed Phase 4 work found three real
defects. All three were confirmed against the data and fixed; they are recorded
here rather than quietly patched.

1. **The shield engaged instantly on a healthy robot.** Calibration discards the
   first 15 ticks after each reset; the runtime armed at tick 0. The median
   nominal score at tick 0 is ~1.1e6 against a trip threshold of ~9.2e3, so
   **320 of 320** nominal environments engaged the fallback at startup. Every
   gate missed it because they all inherited the same warmup assumption and none
   modelled the state machine from startup. Fixed with an explicit arming window
   carried *in the artifact*, so calibration and runtime cannot drift apart
   again; replayed from tick 0, nominal engagement is now 11/384. Regression
   tests: `test_shield_cannot_engage_during_arming`, `test_reset_re_arms_the_shield`.
2. **The stated protocol was not the actual protocol.** `nominal_seed0` was a
   stale 64-env/300-step rollout that the grid's skip-if-exists logic preserved
   across a re-run at 128x400, and the artifact's provenance was read from its
   sidecar. The rollout was regenerated and the grid now validates shape before
   skipping.
3. **Lead time was measured from the wrong instant** (run start rather than the
   decision tick). Corrected above, and the full-fallback margin is now reported
   alongside.

Two further points from that review are accepted but **not** yet addressed, and
are listed as open gaps below: the low episode-level precision (§3) and the fact
that the selection rule was revised after seeing OOD coverage, which means the
current OOD numbers are developmental rather than confirmatory.

## 8. What is still not proven

- **No hardware.** Everything here is sim plus a replay of sim observations
  through the shipped ONNX. The August CaresLab session is the first real-robot
  evidence.
- **Latency measured on the workstation (x86), not on the Orin NX (aarch64).** The
  arithmetic is fixed and tiny, but the number must be re-measured on the target.
  The benchmark records `platform` / `machine` so the two runs are never confused.
- **Intervention remains a counterfactual, and this is the headline gap.** The
  shield's warnings are scored against what the unshielded policy did; the
  fallback was never actually engaged mid-episode in a re-simulation. Nothing
  here shows the shield *prevents* a fall, only that it *warns* before one. The
  planned experiment is a paired closed-loop study: pre-generated scenario blocks
  replayed across arms (unshielded / shielded / fallback-at-onset / sham-switch
  with matched switching statistics), disturbances injected after stable
  locomotion rather than present from reset, scenario blocks as the unit of
  analysis, and paired fall-probability reduction with hierarchical bootstrap CIs
  as the headline. The sham-switch arm is what separates "the monitor's timing
  carries information" from "switching to a stand sometimes helps".
- **The selection rule is developmental, not confirmatory.** It was revised after
  observing that the first rule cost severe-friction coverage. That is legitimate
  development, but it means these OOD numbers need a fresh confirmatory run on
  unseen seeds before they can be reported as a test result.
- **The fallback is not "verified" in the Simplex sense.** A static stand target
  has no demonstrated invariant set or recoverable region under the disturbances
  that trigger it. It should be called an OOD-triggered stand target until there
  is an empirical recoverability map.
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
