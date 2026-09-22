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
| A walking policy passes Gate L | FALSE for the only one trained | 0/256, 31 % rate-limited, `results/phoenix_v2/gate_l/` |

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
