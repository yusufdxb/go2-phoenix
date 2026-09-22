# Evidence ledger

Last reviewed 2026-09-22. Every claim in the README maps to a row here. Status words:

* **IMPLEMENTED**: code exists and its unit tests pass.
* **OFFLINE VERIFIED**: checked against recorded data or files, no simulator or robot run.
* **SIM VERIFIED**: measured in Isaac Lab.
* **HARDWARE VERIFIED**: measured on the GO2.
* **NOT YET VERIFIED**: none of the above.

"Closed-loop validated" is reserved for a full GO2 → training → candidate → GO2 round
trip. It has not happened.

## Phoenix v2 loop

| Claim | Status | Evidence |
|---|---|---|
| The monitor separates policy request, policy-node clip, sent target and measured response | IMPLEMENTED, OFFLINE VERIFIED | `phoenix.monitor.layers`; `tests/test_monitor.py`; checked on the real F1 log (bridge input equals policy-node target on every tick) |
| Safety-altered samples are excluded from the residual | IMPLEMENTED | `tracking_pairs`, tests on policy-node and bridge clips |
| `s_hat` recovers an injected gain scale within 5 % | IMPLEMENTED, OFFLINE VERIFIED on synthetic telemetry and on the one-joint model only | `tests/test_monitor.py`, `tests/test_toy_model.py` |
| Persistence 8/10, hysteresis, global-shift veto, INSUFFICIENT_DATA instead of NOMINAL | IMPLEMENTED | `tests/test_monitor.py` |
| No false DEGRADED on a 120-window nominal run | OFFLINE VERIFIED on synthetic data only | `test_no_false_positive_on_long_nominal_run` |
| Monitor detects and localises a real degradation | NOT YET VERIFIED | needs phase 1 |
| Deployment-fidelity gate (aggregate and per-joint limits) | IMPLEMENTED, OFFLINE VERIFIED on the 2026-09-21/22 GO2 logs | F1: FAIL, 88.8 % altered, RMS 0.80 rad, 0.56 s authority |
| Live health vector beside a running robot (tails `bridge.jsonl`, read-only) | IMPLEMENTED; NOT YET VERIFIED during a robot session | `phoenix.monitor.live`, `scripts/phoenix_live_monitor.py`, `tests/test_live_monitor.py` |
| Simulator rollouts in the bridge record format | IMPLEMENTED (pure adapter); not wired into `evaluate.py` | `phoenix.monitor.sim_records`, `tests/test_sim_records.py` |
| Health report becomes a targeted overlay the env factory consumes | IMPLEMENTED, OFFLINE VERIFIED through the real config loader | `tests/test_condition_validate.py`, `tests/test_phoenix_loop_cli.py` |
| The sim `targeted_actuator` event scales only the targeted joint | NOT YET VERIFIED in Isaac Lab (the factor sampling is unit-tested) | `scale_targeted_actuator_gains` |
| Candidate gate and promotion rule | IMPLEMENTED, OFFLINE VERIFIED on synthetic evaluations | `tests/test_condition_validate.py` |
| Controlled degradation: one joint, gains only down, policy mode only, 2 s ramp-in, saturation latch, triple-locked plus telemetry required, logged per tick, sent gains equal logged gains | IMPLEMENTED | `tests/test_controlled_degradation.py` |
| Controlled degradation behaves as designed on the GO2 | NOT YET VERIFIED | never run on hardware |
| Targeted fine-tuning beats broad randomisation | NOT YET VERIFIED | experiment not run |

## Phoenix v2 deployment fidelity (2026-09-22, `docs/research/PHOENIX_V2_RESULTS.md`)

| Claim | Status | Evidence |
|---|---|---|
| Training and sim evaluation clamp raw actions to [-1, 1]; the shipped ONNX and the pre-v2 deploy node did not (target or fed-back `last_action`) | OFFLINE VERIFIED | Isaac Lab source; ONNX graph ops; 42-80 % of real raw outputs outside [-1, 1] |
| The measured-q replay reproduces the recorded GO2 targets | OFFLINE VERIFIED | max error 4.8e-8 rad, 4 runs, `results/phoenix_v2/limiter_offline/` |
| Command-rate limiter (0.075 rad/step) + clamp, tracking abort, telemetry schema v2 | IMPLEMENTED | `tests/test_phoenix_v2_limiter.py`, `deploy_stand_h25_v2.yaml` + lock |
| Isaac Lab `root_quat_w` is xyzw | SIM VERIFIED | quaternion check in every `results/phoenix_v2/sim_limiter/*/summary.json` |
| H25 stands with the measured-q clip at 35.7 % altered, 0/256 fidelity, 0.168 m | SIM VERIFIED | `results/phoenix_v2/sim_limiter/nominal_measured_q_0p175` |
| H25 stands with the v2 limiter: 256/256 physical success, 0.20 % altered | SIM VERIFIED | `results/phoenix_v2/sim_limiter/nominal_prev_command_0p075` |
| Through the exact deploy code (obs builder, ONNX, action map, wire, ActuatorGate): v2 64/64, legacy 0/64 with `target_beyond_limit` | SIM VERIFIED | `results/phoenix_v2/sim2sim/`, factorial in `sim2sim_factorial/` |
| Actor parity by an independent numpy forward pass, max 6.0e-6 on 4882 real observations | OFFLINE VERIFIED | `results/phoenix_v2/parity/` |
| The v2 path executes H25 faithfully on the GO2 | NOT YET VERIFIED | not run: robot unreachable |
| H25 absorbs RR_thigh gain reduction to 0.5 (score drop 0.042) | SIM VERIFIED | `results/phoenix_v2/strain_pilot/`; stage S stopped by rule (EXPERIMENT.md amendment 4) |
| The monitor passes its sim validation gate | FALSE in an exploratory run | 5/20 nominal false positives, 5/10 localisation, `exploratory_monitor_sim_s0p5/` |
| A walking policy passes Gate L | FALSE for the amendment 3 baseline | 0/256, 31 % rate-limited, `results/phoenix_v2/gate_l/`; its tracking figures are INVALID (scorer defect, amendment 6.1) |
| Deploy contract v3 is identical in training, sim evaluation and the deploy path | SIM VERIFIED (live Isaac Lab plant) | zero difference on `ActionManager.action`, the applied target and every observation term over 120 steps of pathological actions, `results/phoenix_v2/contract/w1/` |
| A walking policy (W2) passes Gate W-H with no soft limiter, hard envelope only | SIM VERIFIED | final seeds 0.9023 / 0.8906 / 1.0000, `results/phoenix_v2/walk_final/w2/`; dev 0.941 / 0.918 / 1.000 |
| That policy tracks both directions | SIM VERIFIED | strong forward +0.701 -> +0.697, strong backward -0.703 -> -0.694, `results/phoenix_v2/walk_diag/w2_nominal/directional.json` |
| The forward/backward asymmetry of W1 was an implementation bug | FALSE | sampler, rewards, frame, signs and command reading all audited symmetric, `results/phoenix_v2/asymmetry_audit/` |
| W2 has an admissible deployment rate limiter | FALSE | every bound in the frozen grid alters 7 to 36 % of joint-samples (limit 1 %) and drops walking success to 0, `results/phoenix_v2/walk_limiter/` |
| A smoother walking policy exists in this recipe family | FALSE for both declared rungs | action_rate -0.25 and -0.5 never leave the stand-still plateau, `results/phoenix_v2/walk_dev/w5_*` |
| The exact deploy stack executes the walking policy faithfully | SIM VERIFIED (limiter opened, recorded) | 0.0000 altered, 100 % fidelity and hardware-gate pass, no faults, 0.953 nominal / 0.859 DR, `results/phoenix_v2/walk_deploy/` |
| Walking absorbs the controlled single-joint degradation | TRUE, so Stage W stops | primary score drop 0.031 at s = 0.5 against the 0.10 rule, `results/phoenix_v2/walk_strain_pilot/` |
| Walking absorbs a global actuator weakening | FALSE, but that run is CONFOUNDED | its env config moves motor strength AND pins friction to 0.8 AND adds actuator latency 1-5 steps against a DR-off nominal, 64 episodes of one seed; superseded by the screen |
| The controlled degradation may name a joint SET, floor rising with reach (0.50 one joint, 0.70 for two or more) | IMPLEMENTED | `MIN_SCALE_MULTI`, per-joint saturation latch, `tests/test_controlled_degradation.py` |
| The degradation saturation latch's standing band is meaningless for walking | SIM VERIFIED | on NOMINAL walking, no degradation applied, per-joint p99 \|requested - q\| reaches 0.69 rad and 9 of 144 joint-sessions sustain the 0.175 rad condition for 0.5 s, one for 10.18 s; amendment 14, `intervention_screen_pinband_0p175/` |
| A single joint at 0.50 degrades W2 walking | FALSE | walking-success drop 0.0104 against a 0.9219 nominal, three of four severities at or above nominal, `results/phoenix_v2/intervention_screen/` |
| A whole leg at 0.70 degrades W2 walking enough to adapt to | FALSE | drop 0.1250 against the preregistered 0.15 bar |
| Both rear legs at 0.70 degrade W2 walking measurably and safely | TRUE, and it is the selected intervention | walking success 0.6328 vs 0.9219, drop 0.2891, every seed >= 0.2734, no abort-band episodes; fidelity 0.9036 against a 0.90 bound is a thin margin |
| The group monitor estimates the applied actuator scale | SIM VERIFIED | within 0.075 at an applied 0.70 and 0.095 at 0.75, monotone (0.775, 0.845), interval covers truth in 76 % / 88 % of detected sessions |
| The group monitor separates nominal from degraded walking at session level | FALSE | frozen gate: false-flag 0.125 (bar 0.05), detection 0.708 (bar 0.80), correct extent 0.375 (bar 0.70); `results/phoenix_v2/monitor_gate/` |
| Targeted adaptation beats broad randomisation for walking | NOT TESTED | the study stopped at the monitor gate before any arm was trained |

## Incumbent policy and deploy stack

| Claim | Status | Evidence |
|---|---|---|
| Incumbent is H25 stand, `phoenix-stand-h25-lat-noise/2026-06-22_21-08-20/model_799`, trained on zero velocity commands only | OFFLINE VERIFIED | configs and weight-lineage check, audit H1 |
| Walking is refused in the deploy path | IMPLEMENTED | deploy contract, policy node, actuator gate, audit H2 |
| ONNX / TorchScript / checkpoint parity for the locked H25 artifacts, max_abs <= 1.7e-6 (tol 1e-5) | OFFLINE VERIFIED, also on the real F1 inputs | lock file, `parity_gate.json`, stage A |
| H25 survives 20 s in sim without trunk contact ("32/32") | SIM VERIFIED | that is the whole meaning of the old success metric, audit H5 |
| H25 holds attitude in sim | SIM VERIFIED with the v2 limiter (0/256 violations nominal); under its trained measured-q clip 1/256 nominal and 10.5 % of DR episodes violate 0.40 rad | `results/phoenix_v2/sim_limiter/`; attitude from projected gravity. The legacy evaluator's wxyz reading is still in `evaluate.py` and not used by v2 |
| H25 stands on the GO2 | NOT YET VERIFIED | the one live attempt (F1, 2026-09-22) faulted after 0.58 s on `target_beyond_limit:RR_thigh_joint` from a folded start |
| The robot executes the policy's requests | FALSE for the incumbent | 88.8 % altered on hardware, 59.7 % in sim, audit H4 |
| Stand-up ramp to the training stance before policy authority | IMPLEMENTED, never run with motors live | `9df76d7` |
| Staged hardware gates A to H | IMPLEMENTED; A to E GO and F NO-GO on 2026-09-21/22 | payload stage records (kept out of git) |

## Test suite

`PYTHONPATH=src PHOENIX_SKIP_HEAVY=1 pytest tests -m "not sim and not ros"` on
2026-09-22 after the v2 limiter work: see the latest count in
`docs/research/PHOENIX_V2_RESULTS.md`. Earlier the same day: 1700 passed, 18 skipped,
1 failed (`test_bundle_staging_refuses_evidence_from_another_commit`, which needs a
gitignored `parity_gate.json`; it passes when the checkpoint directory is present).
Without `PYTHONPATH=src` an editable install elsewhere can shadow this checkout.

## Superseded

Earlier claims (failure replay loop, reliability shield, stand-v3 slew percentages under
the legacy metric, the April hardware slew figure) are indexed in
[`docs/legacy/README.md`](docs/legacy/README.md) and
[`docs/superseded_results.md`](docs/superseded_results.md).
