# 08 — Joint-level residuals: hardware vs simulated explanations (2026-10-12, offline)

The aggregate fit (forward and yaw ratio) can't separate the explanations (doc 03). This pass asks
whether the per-joint signals the robot already logged on 10-06/07 can.

## Method
- **Hardware:** the four gantry W runs (`bridge.jsonl`, 50 Hz policy ticks). Window = first non-zero `walk_cmd` →
  + preset duration. Per joint: `swing_ratio = (p95−p5 of q) / (p95−p5 of policy target)` and
  `rms_err = rms(target − q)` [rad]. 0 slew-clip and 0 limit-clip ticks in every window, so targets
  weren't altered by the bridge. Script: `scratchpad/hw_joint.py` → `phoenix-friction-data/hw_joint_metrics_20261012.json`.
- **Simulation:** the same metrics from the `sim_replay_walk.py` MuJoCo replay for each perturbation in `replay_20261012`.
- **Scores**, across 12 joints × 3 presets:
  - `swingRMSE` / `swingBias` = RMSE / mean of (sim − hw) swing ratio;
  - `errRMSE` / `errBias` = the same for `rms_err` [rad];
  - `aggRMSE` = RMSE of (sim − hw) forward/yaw achievement ratio.
- Output: `phoenix-friction-data/joint_model_comparison_20261012.txt` (45 conditions).

## Results (selected; full table in the data file)
| condition | swingRMSE | swingBias | errRMSE | aggRMSE |
|---|---|---|---|---|
| nominal (no friction, no harness) | 0.276 | +0.142 | 0.0302 | 0.441 |
| fmeas=1 (measured friction) | 0.269 | +0.111 | 0.0250 | 0.186 |
| fmeas=1.25 | 0.285 | +0.134 | 0.0242 | 0.039 |
| fmeas=1.5 | 0.336 | +0.147 | 0.0240 | 0.036 |
| fmeas=1.25, fstiff=1 | 0.300 | +0.128 | **0.0232** | **0.029** |
| drag=30, spring=5 (harness proxy, no friction) | 0.323 | +0.170 | 0.0296 | 0.032 |
| calf=0.8, spring=4 (weak calf + harness) | 0.386 | +0.235 | 0.0312 | 0.055 |
| calf=0.8 | 0.346 | +0.210 | 0.0373 | 0.304 |
| jdamp=0.3 (extra viscous joint damping) | **0.236** | +0.057 | 0.0289 | 0.360 |
| floss=0.3 | 0.260 | +0.116 | 0.0266 | 0.217 |

Hardware calf swing ratios: 0.36–0.54 (forward), 0.21–0.53 (yaw).

## Findings
- **VERIFIED (within this replay):**
  - **Swing.** Every simulated condition over-predicts joint swing (bias +0.06 to +0.24). The robot's joints
    move less, relative to the policy's targets, than any model makes them.
  - **Tracking error.** The conditions that fit the aggregate (friction ×1.25–1.5, harness drag+spring) aren't
    better than nominal at joint swing (0.29–0.34 vs 0.276). Friction conditions do give the lowest
    tracking-error mismatch (0.023–0.025 vs 0.030 nominal and harness), so on that one metric they edge
    out the harness-only proxy.
  - **Calf weakness.** Calf-weakness conditions are worst on both joint metrics.
  - **No single condition fits both levels.** The best swing fit (`jdamp=0.3`) has a poor aggregate fit (0.36).
- **INFERRED:**
  - The joint-level data weakly favour a joint-side loss (friction/damping) over the harness-only proxy,
    via tracking error. They argue against calf weakness.
  - The persistent swing bias in every condition points to a missing or mis-specified element that none of the
    perturbations captures. Candidates: velocity-dependent loss, actuator bandwidth/latency, or foot-contact
    compliance. These are untested.
  - n = 4 runs, one robot, one day, gantry only, so there's no uncertainty estimate.
- **Not identified:** the hypothesis "friction ×1.25–1.5" remains one of several fits. The untethered pair
  (doc 05) is still needed to remove the harness term. A new rosbag `/lowstate` (500 Hz `tau_est`) would allow the
  same comparison on torque, not just position.
