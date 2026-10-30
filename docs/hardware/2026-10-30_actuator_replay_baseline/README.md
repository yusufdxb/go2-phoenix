# Actuator replay baseline (model A), 2026-10-30

Offline replay of the 2026-10-07 hung friction-probe sessions through the unmodified
sim actuator. No fitting, no hardware run. Model A is the training DCMotor
(hip/thigh 23.5 N m / 30 rad/s, calf 45.43 N m / 15.7 rad/s) on the `isaac_matched`
MuJoCo profile (joint damping, frictionloss and armature all 0). The sim is
fixed-base, with no contacts and zero command delay. It resets to the measured state
every 2 s, and the per-tick bridge gains are applied.

Reproduce:

```
PYTHONPATH=src python scripts/actuator_replay.py \
  --split configs/sim2real/actuator_replay_split.yaml \
  --data-root <hw_sessions> --scene <unitree_go2/scene.xml> \
  --out logs/actuator_s2r/baseline_A_dt1ms --dt 0.001
```

The split was registered before any fit:

- train: 005239, 005825, 011122, 011645
- held out: 010358, 011929

The joint exclusions come from the friction-probe README and were recorded there before this analysis.

## Identity

All six sessions are stage FP with checkpoint sha256 `e03013d1…`. Code shas are
`cf78e2ec`, `fa58780d` and `04f3d46e`. The controller is the bridge PD with gains
logged per tick:

| mode | kp | kd |
|---|---|---|
| sit | 30 | 1.5 |
| policy/probe | 25 | 0.5 |
| hold | 20 | 1 |
| folded | 5 | 1 |
| standup | 60 | 5 |

## Findings

1. **Integration step.** At 5 ms the sit (kd 1.5) and standup (kd 5) gains are
   explicit-PD unstable on the calf: kd·dt/I is 1.5 and 5.0 for I = 0.005 kg m².
   Calf position RMSE is then about 0.17 rad. This is an artifact of the sim
   integration, not a property of the actuator. At 1 ms the sit RMSE drops to
   0.006 / 0.009 / 0.025 rad. Policy-mode gains (kp 25, kd 0.5) give the same result at 5 ms and 1 ms.
2. **Pos RMSE at 1 ms, hip / thigh / calf (rad):**
   - train: 0.006–0.014 / 0.009 / 0.018–0.029
   - held out: 0.007–0.008 / 0.007–0.009 / 0.023–0.026
3. **No saturation evidence.** Commanded torque stays at or below 5.8 N m and there are no
   velocity-limit exceedances. The hung data cannot identify torque or rate limits.
4. **The motor follows the PD law.** During probe rises and falls, tau_est = a·PD + b
   with a = 0.99–1.05, |b| = 0.02–0.06 N m and a residual of 0.03–0.06 N m.
5. **Stick-slip is not modelled.** On hardware the tracking error ramps until breakaway
   and the joint holds at rest with about 0 N m. Model A sags 0.01–0.025 rad under
   gravity and never sticks. tau_sim - tau_est RMSE is 0.22–0.35 N m on clean hips
   and thighs and 0.42–0.84 N m on clean calves. Steady-state target-q on the calf changes sign across
   sessions on hardware (-0.027 to +0.021 rad), while the sim stays at a constant -0.008 rad.
   This points to joint friction (model B) as the first effect to test.
6. **Lag is ambiguous.** Whole-session target-to-dq lag on hips and thighs is
   54–105 ms on hardware and 34–70 ms in the sim, so hardware lags about 30–50 ms more.
   Calf lag is noisy (median correlation 0.57–0.75) and is not used.

## Assumptions and limits

- ASSUMPTION: the base is level and fixed. The measured tilt is about 3°, which may
  bias the hip offsets.
- Excluded joints:
  - 010358 FR calf: folded at the joint limit.
  - 011122 RL hip: started splayed and one-sided.
  - 011645 FR and FL calves: the feet touched the floor.
- Hung-leg data only. Load dependence cannot be identified from these sessions.
- This is an offline replay, not a physical validation.
