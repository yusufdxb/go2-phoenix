# Deployment runbook

The deploy path is unchanged from the H25 staged-gate work except for three additions:
per-motor gains and `tau_est` in the bridge telemetry, the controlled-degradation option
(off by default, see `HARDWARE.md`), and the rule that a candidate becomes the new
incumbent only with a PROMOTE decision.

## Artifacts and locks

A deployable policy is a directory with `policy.onnx`, `policy.onnx.data`, `policy.pt`
and the source checkpoint, pinned by a lock under `configs/sim2real/locks/` (hashes of
every artifact plus the deploy config's semantic hash). The policy node and the LowCmd
bridge refuse to start in live mode without the lock, the expected commit and a
telemetry path. Staging, payload preparation and the gate ladder A to H are in
`docs/h25_stand_hardware_run_card.md`.

## Promotion

```bash
PYTHONPATH=src python3 scripts/phoenix_loop.py promote --decision decision.json \
    --lock configs/sim2real/locks/<candidate>.lock.yaml
```

Exit 0 only when the decision says PROMOTE for exactly the checkpoint the lock pins.
Experiment arms deployed for comparison (for example the broad-randomisation policy) are
not promoted; they need the phase-3 preconditions in `docs/research/EXPERIMENT.md`
(parity, and a fidelity PASS on every admissible trial).

## After every run

```bash
PYTHONPATH=src python3 scripts/phoenix_loop.py fidelity logs/hw_sessions/<session>/<stage>/bridge.jsonl
```

A FAIL is recorded and kept; the run is not evidence about the policy or the actuators.
