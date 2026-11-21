# 05 — Paired harnessed vs untethered protocol and hardware authorization gate (DRAFT, 2026-10-12)

Question: **does the same policy (model_2750) still under-track forward motion and yaw when the
gantry harness is completely absent?**

Status: protocol prepared offline. **Nothing below has been executed. Every motor-live step
needs separate explicit authorization.**

## 1. Policy identity (VERIFIED from the robot's own logs)
Every Phoenix live run on 2026-10-06/07 (stands F1, G1; walks `W_vx03_2s` 10-06 23:54Z,
`W_vx03_4s`, `W_yaw05_3s` ×2) logged the same lock check: checkpoint `e03013d1…`, ONNX `6d36796f…`
(+ data `b6ec50ed…`), manifest `c6d4a320…`, code `34104ff`. Bundle `~/phoenix/walk2750-34104ff`
re-verified on 2026-10-12: `sha256sum -c SHA256SUMS` all OK.

**Provenance gap (unresolved).** All four Phoenix walking runs were under a slack-rope
gantry (mewtwo EVIDENCE note `158ecea`: "no walking command has been sent without the gantry";
the W stage requires the typed phrase `LIVE W <preset> ON GANTRY`). No Phoenix log of an
untethered walk exists on the Jetson (no session after 2026-10-07 01:21Z) or on mewtwo (no
`FF_*` / `free_floor_*` records). The README demo (`33ef16f`, 10-09) shows "untethered steps".
The run behind it (date, preset, code path, logs) must be identified before it's used as
evidence of an untethered operating envelope.

## 2. Predictions (MuJoCo, model_2750, from `replay_20261012` / `dr_robustness_20261012`)
| explanation | untethered `vx03_2s` (0.6 m cmd) | untethered `yaw05_3s` (86°) | harnessed (all) |
|---|---|---|---|
| H-F: joint friction ≈1.25–1.5× measured, harness negligible | 0.30–0.31 m | 23–27 % | ≈ untethered |
| H-H: harness drag/torsion only, no joint friction | 0.50 m | 76 % | 0.33 m / 26 % |
| H-F1+H: measured friction ×1 plus harness | 0.35 m | 34–47 % | ≈ 60 % / 20 % |
| H-A: calf actuator ×0.8 (+ harness torsion) | 0.38 m | 64 % | 0.38 m / 19 % |
| Oct 7 hardware (gantry) | not measured | 20.5 %, 22.1 % (IMU) | 4 s: 0.71 m tape (59 %) |

*In this replay* the yaw test separates the explanations most (about 20 % vs 47–76 %). These are model-dependent hypotheses (doc 08: no simulated condition fits the joint-level data), not categories the result must fall into. Forward separates H-H from
H-F by about 0.15–0.20 m, well above camera/tape resolution (≈1–2 cm).

## 3. Measurement system
| signal | source | rate | status |
|---|---|---|---|
| commanded vx/wz | `walk_cmd.jsonl` (`phoenix_walk_command`) | 50 Hz | existing, verified |
| joint q, dq, targets, kp/kd, slew/limit clips, torque margin, estop/hold causes | bridge `bridge.jsonl` | 50 Hz | existing, verified 10-07 |
| IMU quaternion, gyro, joint pos/vel, action | `policy.parquet` | 50 Hz | existing; yaw method verified (IMU vs gyro ≤ 1°) |
| `/lowstate` full (tau_est, temperature, foot_force raw, lost) | `scripts/harness_record.sh` rosbag2 | 500 Hz | rosbag2 present; **not used 10-07**; tau_est quantized at about 0.025 N·m; no motor current; foot_force uncalibrated |
| `/sportmodestate`, `/utlidar/robot_odom` | rosbag2 | 300 / 150 Hz | supplementary only; same estimator; behaviour in released (low-level) mode unverified |
| forward translation | **external fixed camera + floor fiducials** | ≥ 60 fps | **equipment not identified**; tape measure as cross-check |
| harness tension | — | — | **UNMEASURED** (no rated load cell identified) |

Camera method: tripod-fixed camera, intrinsics from a checkerboard, floor homography from ≥4
surveyed floor markers (tape-measured grid or ArUco), AprilTag/ArUco on the robot's back. Before any trial, check accuracy by moving a tagged
object a tape-measured 0.50 m (accept ≤ 2 cm error). Time alignment: the Jetson isn't NTP-synced
(offset to Mac ≈ 0.4 s, unverified). Align on the walk-motion onset seen in both video
and `walk_cmd.jsonl` / joint telemetry (±1 frame).

Harness load cell: nothing may go inline on the fall-arrest path without a rating for
fall shock loads and qualified mechanical sign-off. Until then, harness tension is recorded as
unmeasured, and "rope slack" is an operator observation (video should show it).

## 4. Conditions that can't be fully controlled
- **Harness loading** (unmeasured; the rope was "reported slack", never shown to carry zero load).
- **Battery:** measured SOC was 91 % (10-06) and 85 % (10-07). On 10-12 it was 49 % → 43 % idle in 18 min.
  Existing gates: > 20 % per live stage (harness) and ≥ 30 % at A/B block start. **There's no 85 % gate.**
  This protocol *recommends* starting each block near the 10-06/07 SOC (≥ 85 %) to match conditions, logging SOC
  per run and interleaving arms. That's a protocol choice, not an existing threshold.
- **Motor temperature:** not logged 10-07. On 10-12 the rear hips read 53 °C, then 46 °C 18 min later, vs 30–33 °C steady
  elsewhere (cooling transient, see doc 07 §1). Record a cold baseline, log via rosbag, and interleave conditions so drift
  doesn't align with condition. No temperature threshold exists. Any abort value must come from a documented source.
- **Floor surface/contact:** use the same floor area and surface for both arms (ideally the gantry footprint with
  the rope detached). Note any change of surface.
- **Initial pose:** each run starts from the policy's own stand-up; record q at motion start.
- **Operator/rope geometry:** fixed rope anchor and length for harnessed runs; photograph each run.

## 5. Staged matrix (2026-10-12 revision; policy fixed = model_2750, code identity fixed per campaign)
The §2 predictions are model-dependent hypotheses from one replay. Results are reported as measured
ratios with uncertainty and **aren't** forced into H-F/H-H/H-A. A result outside every predicted band is a valid outcome.

**Identical measurement in both arms:**
- the same external camera, floor markers and homography;
- the same rosbag topic set (`/lowstate` 500 Hz plus bridge, policy and `walk_cmd` logs);
- the same IMU-yaw method;
- the same tape cross-check;
- the same start pose and floor area.

The only intended difference between arms is whether the rope is attached.

| stage | content | arm | motors | gate / approval |
|---|---|---|---|---|
| S0 | passive rosbag of idle robot (cold temperature baseline, rates, clock offset); camera intrinsics + floor homography; 0.50 m tagged-object accuracy check (≤ 2 cm) | — | **off** | GO (decision 2); needs camera + markers on site |
| S1 | motors-off ledger B, C, D | — | **off** | CONDITIONAL GO (decision 3): operator with remote; code-identity choice (doc 07 §5) |
| S2 | qualified standing: E–H (gantry), then FREE_FLOOR drill (zero command, deadman STOP) | A, then B | live | NO-GO until battery, temperature, motion-authority (doc 07 §3) and safety gates are met + explicit approval per stage |
| S3 | harnessed reference repeat: `vx03_2s` ×3, `yaw05_3s` ×3 with the S0 instrumentation | A | live | NO-GO until same gates (decision 7) + per-run approval |
| S4 | untethered forward: ONE `vx03_2s` via FREE_FLOOR, then human review | B | live | NO-GO (decision 5): needs staging of reviewed FREE_FLOOR code, contention amendment, approval |
| S5 | untethered yaw `yaw05_3s` (CCW), then `yawm05_3s` (CW) | B | live | NO-GO: gate implemented offline, disabled (doc 07 §8); needs accepted S4 + yaw analyzer + approval |
| S6 | paired blocks ABBAAB, n = 3 per arm per profile | A, B | live | after S3–S5 each reviewed |

On sample size: Oct 7 yaw repeatability was 20.5 % vs 22.1 % (n=2). n=3 per arm can resolve differences of about 10 points or more. That's an
estimate from n=2, so re-estimate it from S3 before fixing S6.

Per-run abort criteria (existing bridge/harness logic plus operator): deadman release; estop;
lowstate stale; tilt / limit-abort band (0.175 rad); any fault; slip, stumble or contact with an obstacle; operator
judgement. One run per authorization step; check data quality before the next run.

## 6. GO / NO-GO
| test | verdict | reason / prerequisite |
|---|---|---|
| Block 0 passive rosbag + camera accuracy check | **GO** (no motors) | robot idle/damped; needs camera and markers on site |
| Re-run stages B, C, D (motors OFF) on payload 34104ff | **GO, pending operator** | C needs the GO2 remote (no `/wirelesscontroller` traffic seen 10-12) and operator present |
| Stages E–G live stand (gantry) | **NO-GO now → GO on approval** | battery above the existing gates (> 20 % / ≥ 30 %) and, for matching, near the 10-06/07 SOC; temperature baseline recorded; other stacks confirmed stopped (come-here disabled ✔; MOSAIC/go2-nav stopped at 19:57Z, restart not controlled by me); operator + estop; explicit authorization |
| Harnessed `vx03_2s`, `vx03_4s`, `yaw05_3s` (arm A) | **NO-GO now → GO on approval** | existing gated path, previously flown; same prerequisites |
| Untethered zero-command stand | **NO-GO** | no Phoenix gate on the robot covers it; FREE_FLOOR drill exists only in unpushed mewtwo WIP `06c5a37` |
| Untethered `vx03_2s` | **NO-GO** | only via FREE_FLOOR (incumbent-only, vx03_2s-only, drill + tape + human review), which isn't staged on the Jetson. Staging new code is a deployment change that needs approval and a new stage A. Typing "ON GANTRY" untethered would falsify the gate and is refused. |
| Untethered `yaw05_3s`, `vx03_4s` | **NO-GO** | no gate permits them; needs a reviewed FREE_FLOOR grant amendment; the untethered envelope behind the README demo is undocumented |
| CW yaw (both arms) | **NO-GO** | preset `yawm05_3s` exists only in the offline amendment (doc 07 §8), unreviewed and ungrantable |
| Physical A/B with fric_T_s50 | **NO-GO** | see `02_training_ab_audit.md` |
| Retraining | **NO-GO** | no credible identified model |
