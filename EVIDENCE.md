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

## Incumbent policy and deploy stack

| Claim | Status | Evidence |
|---|---|---|
| Incumbent is H25 stand, `phoenix-stand-h25-lat-noise/2026-06-22_21-08-20/model_799`, trained on zero velocity commands only | OFFLINE VERIFIED | configs and weight-lineage check, audit H1 |
| Walking is refused in the deploy path | IMPLEMENTED | deploy contract, policy node, actuator gate, audit H2 |
| ONNX / TorchScript / checkpoint parity for the locked H25 artifacts, max_abs <= 1.7e-6 (tol 1e-5) | OFFLINE VERIFIED, also on the real F1 inputs | lock file, `parity_gate.json`, stage A |
| H25 survives 20 s in sim without trunk contact ("32/32") | SIM VERIFIED | that is the whole meaning of the old success metric, audit H5 |
| H25 holds attitude in sim | NOT YET VERIFIED | the evaluator reads Isaac Lab 3.0 xyzw quaternions as wxyz; attitude flags are corrupted, audit H6 |
| H25 stands on the GO2 | NOT YET VERIFIED | the one live attempt (F1, 2026-09-22) faulted after 0.58 s on `target_beyond_limit:RR_thigh_joint` from a folded start |
| The robot executes the policy's requests | FALSE for the incumbent | 88.8 % altered on hardware, 59.7 % in sim, audit H4 |
| Stand-up ramp to the training stance before policy authority | IMPLEMENTED, never run with motors live | `9df76d7` |
| Staged hardware gates A to H | IMPLEMENTED; A to E GO and F NO-GO on 2026-09-21/22 | payload stage records (kept out of git) |

## Test suite

`PYTHONPATH=src PHOENIX_SKIP_HEAVY=1 pytest tests -m "not sim and not ros"` on
2026-09-22: 1700 passed, 18 skipped, 1 failed. The failure,
`test_bundle_staging_refuses_evidence_from_another_commit`, needs a gitignored
`parity_gate.json` that is absent from a fresh checkout; it is not caused by v2.
Without `PYTHONPATH=src` an editable install elsewhere can shadow this checkout.

## Superseded

Earlier claims (failure replay loop, reliability shield, stand-v3 slew percentages under
the legacy metric, the April hardware slew figure) are indexed in
[`docs/legacy/README.md`](docs/legacy/README.md) and
[`docs/superseded_results.md`](docs/superseded_results.md).
