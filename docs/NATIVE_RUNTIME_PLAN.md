# Native Runtime Plan

Synthesis of the Phase 0 audits into a single implementation plan.

**Inputs.** `docs/native_runtime_audit.md` (Phoenix deploy-path audit),
`policy-health-monitor/docs/native_runtime_audit.md` (PHM C++ audit, with measured allocation
counts), and three design memos held outside the public repos (architecture, verification matrix,
benchmark methodology).

**Thesis.** Python owns training and experimentation. C++ owns on-robot execution. A deterministic
native layer holds final authority over actuation.

**Status of this document.** Partly implemented. `runtime/phoenix_core` now exists and its parity
gates have been run; see `runtime/README.md` for measured results. Everything else below is plan
only. Nothing here is hardware-verified, and no performance measurement has been taken. Every
number quoted from a prior artifact carries its provenance.

---

## 0. What the audits changed about the plan

Four items in the original brief did not survive contact with the code. They are listed first
because they are the ones a reviewer will check.

| Brief said | Audit found | Plan does |
|---|---|---|
| Replace PHM's `vector.erase(begin())` with a ring buffer; "remove avoidable hot-path allocation" | **Measured**: the erase shifts 29 × 24-byte vector *headers* (696 B of memmove), because move-assignment steals the heap pointers. It is ~0.1% of the call. At PHM's 10 Hz publish rate the whole detector is ~0.03% of the frame budget. | **Do not do it.** Rank-10 item, explicitly recommended against. The `flat` buffer (122,880 B/call) is the real allocation, and even that is deferred pending a stated rate requirement. |
| Port normalization to C++ with parity tests | Normalization is **baked into the ONNX graph** for `stand_v3` and **absent** from `flat_v4` (R1). A config-driven C++ normalizer double-normalizes one of them. | **Never normalize in C++.** Feed raw obs. Detect the `Sub/Div` prefix from the graph, not from a config flag. There is no `observation_normalizer` module. |
| Adopt precedence `ESTOP > watchdog > PHM STOP > PHM INTERVENE > limits > policy` | Omits the startup gate, sensor-validity gates, the NaN gate and the attitude gate, four gates that today sit **above** the shield. Adopting verbatim is a **safety regression**. | Use the corrected 11-rank ladder (§3). Limits are a **terminal filter**, not a precedence rank. |
| Consider `librobotguard`, a shared C11 safety kernel | The honest intersection between Phoenix's and PHM's safety logic is `age > timeout` plus an integer counter. PHM does not actuate anything, so it is not a safety consumer. The NaN convention, the one thing worth sharing, is **directionally opposite** in the two repos. | **NO-GO.** Replaced by `docs/safety_invariants.md`: a numbered rule list duplicated in both repos, each testing the same rule numbers against its own semantics. Re-open trigger recorded in §7. |

---

## 1. Scope

### Stays Python (no determinism or latency argument)
Training and Isaac Lab env, `export.py`, `verify_deploy.py`, `bench_export.py`, all shield *fitting*
(`ood_monitor.py`, `metrics.py`, `features.py`, `study.py`, `replication.py`, `oracle_screen.py`),
`ShieldRuntime`/`TemporalFilter` (offline, not on the deploy path), `replay/`, `adaptation/`,
`demo/`, YAML/argparse/logging.

### Moves to C++
Observation construction, projected gravity, ONNX inference, the gate ladder, all of `safety.py`,
the abort latch, `DeployMonitor` + `SimplexArbiter` + `DeployShield`, the blend arithmetic, the
slew clip, `motor_crc`, the LowState translation, and the deadman.

### The honest justification for C++
Ranked by defensibility, per the benchmark memo:

1. **Measurability.** Nothing today records tick period, inference latency, or overruns. `rclpy`'s
   timer has no overrun detection (R25). The native port is the first opportunity to *have* these
   metrics. This is a real win with nothing to do with speed.
2. **Independence from the Python runtime being alive.** A correctness argument.
3. **Tail latency and jitter**, a hypothesis about p99/p99.9 and deadline misses, *not* p50, to be
   tested by §6, not asserted.
4. **Testability.** `_control_step`'s ladder has **zero** test coverage today. Extracting it as a
   pure function is worth more than the latency win. This is the headline.

**Not claimed:** faster ONNX inference (ORT is already C++; same kernels, same work), a faster
384×384 matvec (numpy dispatches to BLAS), a higher control rate, or any change to fall rate,
tracking error, or shield quality.

---

## 2. Process topology

**Four processes become three. The sensor hop is deleted. The actuation hop is kept.**

| Process | Absorbs | Rate |
|---|---|---|
| `phoenix_policy_rt` | `lowstate_bridge_node` + `ros2_policy_node` | 50 Hz |
| `phoenix_actuator_rt` | `lowcmd_bridge_node` | 50 Hz |
| `phoenix_guard` | `deadman_joy_node` / `wireless_estop_node` | 10 Hz |

**Why the sensor hop goes.** `lowstate_bridge_node` provides zero safety property, it cannot
refuse, latch, or hold. It adds a DDS round-trip and re-stamps the message with its own `now`, so
the policy node's "arrival time" is already second-hand. Fusing it *improves* freshness fidelity.

**Why the actuation hop stays.** The estop double-check at `lowcmd_bridge_node.py:24-26` defends
against policy-process hang, crash, wrong-but-finite command, and operator kill. **Three of those
four are only covered by a process boundary.** The collapse would buy one intra-host DDS hop for a
12-element command, on a 20 ms budget, never measured on the Orin. Trading an enumerable safety
property for an unmeasured sub-millisecond saving is a bad trade.

**Recorded so it is not rediscovered:** if Orin telemetry ever shows that hop at p999 > 2 ms, the
fix is a shared-memory ring plus eventfd **keeping the process boundary**, not a merge.

**New risk created deliberately:** reading `/lowstate` directly inherits Unitree `[w,x,y,z]`
quaternion order instead of ROS `(x,y,z,w)` (R8). Mitigated by naming the boundary explicitly:
`projected_gravity_from_wxyz` vs `..._from_xyzw`, never a bare four-float signature.

---

## 3. Safety precedence

Ranks 0–7 are **decision** ranks: the first to fire determines the target and short-circuits the
rest. Ranks 9–10 are **terminal filters** applied to whatever the ladder produced, on every path
including aborts.

| Rank | Gate | Latching |
|---|---|---|
| 0 | External ESTOP (async, between or during ticks) | sticky |
| 1 | Runtime watchdog (`max_runtime`; N consecutive deadline misses; inference timeout; ORT exception) | sticky |
| 2 | Estop-chain integrity (heartbeat missing or stale) | sticky |
| 3 | Startup gate (not all of estop/imu/joint_state seen) | none while waiting; latch at 15 s |
| 4 | Sensor validity (stale, frozen-payload, **non-finite IMU**, L2 fix, joint-name completeness) | sticky |
| 5 | Command freshness (`/cmd_vel` stale, L1 fix): ramp command to zero, policy keeps running | non-latching |
| 6 | Attitude abort (`!isfinite \|\| \|pitch\|>0.8 \|\| \|roll\|>0.6`) | sticky |
| 7a | Shield STOP (blend saturated): full `default_q` via calibrated ramp, **commanded hold, not abort** | escalates to rank 1 only after persistence |
| 7b | Shield INTERVENE: `(1-blend)·learned + blend·default_q` | non-latching |
| 8 | Policy command |, |
| 9 | *filter* Output validity (non-finite target ⇒ hold previous, latch) | sticky |
| 10a | *filter* Joint position limits, **does not exist today** | counter-latching |
| 10b | *filter* Slew-rate limit, **applies to every path including aborts** (L3 fix) |, |
| 11 | Emit, or stay silent per the abort output contract |, |

**Shield authority.** Today the shield is strictly advisory: it can only blend toward `default_q`,
never latch, and sits at rank 9 of 11. The plan grants it **commanded-hold** authority, and
escalation to a latch only after the blend has been saturated *and* the score above trip for
`stop_escalation_ticks`. It never gains authority to suppress a hard predicate or to un-latch.

The reason it does not get instant-abort authority is evidential: its nominal-episode FPR is 0.0417,
and the corrected closed-loop study cannot separate its benefit from a dose-matched sham
(`0.0039 [-0.023, 0.033]`). Instant abort would convert a measured false-positive rate into
mid-motion unrecoverable stops.

**Post-abort silence (R20) is part of the ladder, not an afterthought.** After a latch: publish per
the cause table, then **zero further publishes, ever**. Per-tick rebroadcast of `default_q`
previously fought the real posture and caused a Jetson brownout.

**LLM boundary.** No higher-level model touches actuation in either repo today. The rule is
recorded in `docs/safety_invariants.md`: no reasoning system may influence any rank 0–7 gate or
un-latch anything. Enforcement point is the gate function's signature, it takes a `SensorSnapshot`,
a `CommandInput` and a `RuntimeHealth`, and there is no channel through which advisory input can
reach it except rank 7.

---

## 4. Module layout

```
runtime/
├── phoenix_core/          # ROS-free, allocation-free, noexcept, C++17. The whole point.
│   ├── types.hpp          # fixed-size PODs, error codes
│   ├── joint_map.hpp      # BOTH permutations, named unambiguously, round-trip tested
│   ├── attitude.hpp       # projected_gravity_from_wxyz / _from_xyzw
│   ├── observation.hpp    # 48-dim contract; no normalizer (R1)
│   ├── inference.hpp      # InferenceEngine interface
│   ├── ort_engine.cpp     # ONNX Runtime backend, path-based session (R3)
│   ├── shield.hpp         # DeployMonitor + SimplexArbiter, ported literally
│   ├── gate.hpp           # the §3 ladder as a PURE function
│   ├── filters.hpp        # position clamp + slew clip, separately tested
│   ├── motor_crc.hpp      # literal port (R19), static_assert(sizeof(LowCmd)==812)
│   └── telemetry.hpp      # numeric codes, no hot-path string formatting
├── phoenix_ros2/          # thin adapter: subs, conversion, params, lifecycle
└── phoenix_msgs/          # JointTargetStamped{float32[12], seq, steady_ns, gate_state}
```

**Dropped from the suggested skeleton:** `observation_normalizer` (R1, must not exist),
`safety_gate` as separate from `watchdog` (they are one pure function; splitting invites a bypass
path), `action_filter` as a single module (split into position and rate filters, separately tested).

**C++17**, not C++20, ROS 2 Humble's default, and `std::span` is the only C++20 feature the design
wanted. A minimal internal `span` view is not worth a toolchain compatibility risk on the Orin.

---

## 5. Parity strategy

**The parity oracle is built in Python first.** Extract `control_step()` as a pure function in
Python *before* writing any C++, so the port has an oracle from day one and the ladder gains test
coverage it has never had.

Tolerances are **declared before the comparison runs**, derived from dtype and operation chain,
never from an observed diff. The comparison program takes the tolerance as a required argument with
no default.

| Stage | Declared tolerance | Basis |
|---|---|---|
| Observation vector | **bit-exact** | permuted copies + one float32 subtract; no accumulation |
| Projected gravity | **bit-exact** | 9 flops in double, single narrowing. Achieved: 0 mismatches over 507 fixtures. |
| Roll / pitch | **≤ 2 ULP** + zero-width ambiguous band vs. the attitude threshold | Revised after measurement, see below |
| ONNX action/latent | **bit-exact** with pinned ORT config | same build/graph/threads means same kernels. Achieved: 0 / 118,800 elements over 300 frames, ORT 1.23.2 pinned on both sides. |
| Mahalanobis score | 1e-4 relative + **zero decision mismatches** + ambiguous-band report | `944·u ≈ 5.6e-5` worst case |
| Arbiter state/blend | **bit-exact, zero mismatches** | integers and doubles only; pins R17 |
| Final joint command | 1e-6 rad | float32 transport, 5 orders below the 0.175 slew limit |
| CRC word | **exact** | it is an integer |
| Self-determinism | **bit-exact against itself**, run first; abort the comparison if it fails | non-determinism invalidates cross-language comparison |

### What the parity harness actually found

The fixtures caught four divergences that inspection had not. Recorded because three of them are
exactly the "looks right, is wrong" class the port exists to prevent.

1. **Quaternion width.** `QuatXYZW` initially stored `float`. `sensor_msgs/Imu` carries **float64**
   and the Python evaluates in that width, narrowing once at observation assembly. Narrowing at the
   *input* instead cost ~1 ULP in projected gravity. Fixed by making the sensor-width types double
   (audit risk R12).
2. **`np.clip` does not treat non-finite values uniformly.** It is
   `minimum(maximum(a, lo), hi)`, so it propagates NaN but **clamps ±Inf to the bound**. The obvious
   C++ reading, "non-finite passes through", diverges on infinities.
3. **A NaN `current` yields NaN**, because both bounds become NaN. The port passed the target
   through instead.
4. **Bit-exact roll/pitch is not achievable, and the cause is not the port.** numpy's `arctan2` and
   glibc's `atan2` disagree by 1 ULP on identical double input
   (`0x1.08ab61898531ep+0` vs `0x1.08ab61898531fp+0`, verified directly). Transcendental functions
   are not required to be correctly rounded and these two implementations differ.

Item 4 is the one worth being careful about, because widening a tolerance after a failed comparison
is exactly the move this methodology forbids. The resolution is to widen to the smallest value the
mechanism justifies (2 ULP) **and then measure the consequence rather than assume it away**:
roll/pitch feed exactly one decision, the attitude gate, so the harness asserts that no fixture lies
within a few ULP of the threshold, i.e. the drift provably cannot change the verdict. Measured:
worst 2 ULP on both, 234/507 bit-exact, **zero** ambiguous frames. If that assertion ever fails, the
fix is a shared implementation of the transcendental, not a wider band.

**Known problem to confront up front:** `deploy/flat_v4_latent.onnx.verify.json` records a max
latent abs diff of **2.29e-04**, which already exceeds the 1e-4 gate. This must be resolved before
the C++ gate runs, not explained afterwards.

The replay set must *contain* non-finite latents. If the recorded replay has none, inject a declared
NaN/±Inf fixture and record that it ran, this is what catches R17.

**Gate:** any stage over tolerance, any decision mismatch, or any non-empty ambiguous band ⇒ the
runtime is **not parity-verified**, and no performance claim may be published. A faster runtime that
computes something else is not a faster runtime.

---

## 6. Benchmarking

Programs are specified but **produce no numbers in this plan**. Result fields are null until
measured.

Requirements carried from the methodology memo:
- **p99.9 costs ~100,000 samples.** The 120 s `max_runtime_s` cap means one on-robot run honestly
  supports p50 and p95, p99 only with a CI, and **p99.9 not at all**.
- Machine state (governor, load, pinning, thermal) recorded into every result file. Two runs of the
  same PHM binary on the same machine differed 1.66× on p50 purely from external load; max ranged
  65 µs to 13,045 µs across four runs of identical work.
- Discard warm-up frames; K ≥ 5 repetitions; report median-of-medians and inter-run spread.
- Both runtimes get the same RT treatment or neither does. Comparing pinned-`SCHED_FIFO` C++ against
  default Python measures the configuration, not the language.
- **The Orin row stays `NOT MEASURED`** until a validated result file exists in the tree. The
  comparison script must contain no code path that can produce an Orin number without one.
- Numerical drift is emitted into the *same file* as latency, so a speed number can never be
  published without its correctness number attached.

---

## 7. PHM work

Ordered by measured value, not by the brief.

1. **Fix the NaN fail-open.** Non-finite embeddings currently produce a **healthy** verdict, the
   monitor fails open on exactly the input class a broken upstream policy emits. Highest-value item
   in either repo.
2. **Build the Python↔C++ golden-vector parity test.** The package's entire premise is behavioural
   equivalence, currently enforced only by comments, and already partly false.
3. **Fix the QoS mismatch** (C++ subscriber RELIABLE, Python BEST_EFFORT) and add a no-data warning.
   A silently receiving-nothing monitor is worse than a crashed one.
4. **Warn or fail on `threshold = 0.0`**, which makes the detector permanently inert while looking
   alive.
5. **Demote the Eigen backend** from default. It is measured *slower* than plain (p50 43.7 vs
   36.4 µs) while materializing a full window×dim temporary. Fix by demoting, not by optimizing.
6. Ring buffer / `erase` removal: **deferred, with a stated re-open condition** (embedding rate above
   ~1 kHz, or a move to a hard-real-time thread).

**Integration contract.** Contract-only, no build coupling. Phoenix consumes a PHM verdict through a
documented message shape; the enum, topic, and QoS are pinned in `docs/safety_invariants.md` in both
repos, and each repo tests the same numbered rules against its own semantics.

**`librobotguard` re-open trigger:** a *third actuating* consumer independently needing the same
fail-closed ladder plus slew clip. Not before, generalizing from two hypothetical call sites is how
safety code becomes unreadable.

---

## 8. Latent bugs in the current Python path

Found during the audit, independent of the port. Listed so they are fixed deliberately rather than
absorbed silently into a rewrite.

| # | Bug | Severity |
|---|---|---|
| L1 | `/cmd_vel` has no freshness check. Dead teleop leaves the last velocity latched; the robot keeps executing it. | High on any walking config |
| L2 | The attitude abort cannot fire on a NaN IMU (`abs(NaN) > 0.8` is `False`, and IMU orientation is never finiteness-checked). | High |
| L3 | The abort path is the only path with no slew clip; the bound comes solely from the bridge. | Medium now, High after the port |
| L4 | The bridge has no startup grace for the estop topic; starting it before the deadman latches it permanently. | Medium |
| L5 | A code comment claims a default pose is published at abort; false for `external_estop` and `max_runtime`. | Doc |
| L6 | No joint **position** limit clipping anywhere, only rate limiting. | Medium |
| L7 | A non-finite target survives `np.clip` and is published; caught only by a different process. | Medium |
| L8 | The bridge computes ages in ROS time while the node uses `monotonic_ns`. Two clock domains in the actuation chain. | Low now, a trap later |

All eight are caught by a single `control_step()` pure-function harness, which is the argument for
building that harness first.

---

## 9. Evidence discipline

Carried forward unchanged from the audits:

- **Nothing in Phoenix is hardware-validated as *working*.** The only hardware evidence is that the
  stack ran on 2026-04-18 and produced a **failing** number (30.23% per-step slew saturation at
  `cmd_vel = 0`). Every success metric is simulation.
- The shield has never touched hardware in any regime.
- The corrected closed-loop walking study's dose-matched sham does **not** exclude zero; commit
  `3a3a66e` withdrew the monitor-timing claim. That withdrawal stands.
- `deploy/shield_stand_v3.npz.bench.json` is **x86_64**, not Orin.
- `deploy/flat_v4_latent.onnx`'s shield artifact carries `falls_warned: NaN` and an empty
  `evaluation` block.

No README wording may upgrade any of these.
