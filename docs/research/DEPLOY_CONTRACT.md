# Deploy contract v3 (frozen 2026-09-22)

One action and observation contract for training, simulated evaluation, ONNX inference
and the ROS deploy path. Each line below was checked against the installed Isaac Lab
source or the deploy code, not recalled. Tests: `tests/test_deploy_contract_v3.py` (pure
functions, CI) and `scripts/phoenix_v2_contract_check.py` (the live Isaac Lab plant;
result in `results/phoenix_v2/contract/`).

## Action path

| # | Layer | Training / sim evaluation (Isaac Lab) | Deploy (ROS 2 node + LowCmd bridge) |
|---|---|---|---|
| 1 | raw policy output `raw` | actor mean (rsl_rl) | ONNX `action` output (the graph has no clamp) |
| 2 | clamp | `RslRlVecEnvWrapper(clip_actions=1.0).step`: `clamp(raw, -1, 1)` (`vecenv_wrapper.py:171`) | `policy_action_map`: `clip(raw, -1, 1)` (`control.action_clip: 1.0`) |
| 3 | `a` stored | `ActionManager.process_action`: `_action[:] = a` (`action_manager.py:389`) | `fed_back` returned by `policy_action_map` |
| 4 | scale and joint map | `JointPositionAction.process_actions`: `a * 0.25 + default_q` (`joint_actions.py:174`), joint order = `POLICY_JOINT_ORDER` | `default_q + 0.25 * a`, same order; reordered to Unitree motor order only on the wire |
| 5 | soft rate limiter | **none in training** (recipe W and later); sim evaluation may apply the frozen deploy limiter for Phase 7 | bridge `ActuatorGate`, `prev_command`: `clip(req, prev_sent +/- dq_max)`, anchored on the last target actually sent |
| 6 | hard envelope | PhysX joint limits; the clamp keeps every target within 0.25 rad of the default pose, which is inside the URDF limits | URDF joint-limit clip, 0.175 rad abort band on the layer-4 request, catastrophic tracking-error watchdog, E-stop, deadman, LowState / command watchdogs, hold / damp |
| 7 | sent target | `processed_actions`, held for 4 physics substeps (50 Hz policy, 200 Hz physics) | LowCmd `q`, one per 50 Hz tick |

## Recurrent term

`last_action` observes **layer 3**: post-clamp, pre-scale, pre-limiter
(`mdp.last_action` returns `env.action_manager.action`, `observations.py:671`). It is
zero after reset (`ActionManager.reset`). `action_rate_l2` penalises the difference of
the same layer-3 quantity. The deploy node feeds back `fed_back` from
`policy_action_map`, starting from zeros. A deploy limiter (layer 5) never changes
`last_action`: the policy is told what it asked for, as in training, where no limiter
exists.

## Observation

48 dims, no scaling, in this order (`OBS_TERM_ORDER`): `base_lin_vel` (body, m/s),
`base_ang_vel` (body, rad/s), `projected_gravity` (body, unit), `velocity_command`
(`vx, vy, wz`), `joint_pos - default_q`, `joint_vel`, `last_action`. Training adds
uniform noise to the first three and the two joint terms; the deploy builder adds none.

## What changed relative to the incumbent's deploy path

1. The ONNX output was never clamped on hardware (`action_clip` absent), so both the
   target and the fed-back `last_action` could reach 10 (fixed in v2, amendment 1).
2. The dual-policy mode-switch path in `ros2_policy_node.py` computed
   `default_q + scale * raw` and fed raw actions back, bypassing `policy_action_map`
   (fixed 2026-09-22; `test_mode_switch_path_uses_the_same_action_map` fails on the old
   code). Walking configs are still refused by `deploy_contract.WALKING_ENABLED`.
3. The measured-q clip (incumbent) is replaced by the command-rate limiter in every v2
   deploy config; the incumbent config keeps it for replay only.

## Contract versioning

`contract v3` = the table above with **no soft limiter in the training MDP**. Policies
trained with a soft limiter in the MDP are contract v2. They run under v3 only where
simulation shows they execute faithfully without their training limiter: H25 (trained
with measured-q 0.175) does, 64/64 through the exact deploy code with 0.14 % of targets
altered (`results/phoenix_v2/sim2sim_factorial/`); the amendment 3 walking baseline
(trained with prev_command 0.075) does not (Gate L, amendment 5).
