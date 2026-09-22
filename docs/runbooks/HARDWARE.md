# Hardware runbook: the Phoenix experiment stages

For bring-up, staging, the gate ladder A to H, the physical setup, HALT and the field
notes, follow `docs/h25_stand_hardware_run_card.md` and `docs/go2_field_notes.md`. This
page only adds the experiment stages (labels starting with `X`). None of it has been run.

## Preconditions (all required)

1. Phase 0 passed: the policy passes the deployment-fidelity gate on a nominal live run.
   With the current H25 incumbent it does not (88.8 % altered on 2026-09-22).
2. Stages A to H pass for the exact commit and lock.
3. Nothing else is commanding the robot (`come-here.service` stopped; the contention
   probe of every live stage checks this).
4. Robot on the floor in a harness, operator holding the deadman, stand-up by the
   operator as in the run card.

## Controlled degradation

A software reduction of one motor's `kp` and `kd`, applied only while the bridge is in
policy mode. Hold, damp and stand-up keep their nominal gains, so every fail-closed path
behaves exactly as without it. Bounds: one joint, scale 0.5 to 1.0 (a reduction only).
It is off unless all three locks agree:

```bash
# lock 1: the environment; lock 2: a stage label starting with X; lock 3: the spec
export PHOENIX_EXPERIMENT=controlled_degradation
python3 -m phoenix.sim2real.lowcmd_bridge_node ... \
    --stage X1 \
    --telemetry <session>/X1/bridge.jsonl \
    --experiment-degradation RR_thigh:0.6
```

Any lock missing refuses startup with a message naming it, and a degradation without
`--telemetry` is refused too. The scale ramps in over 2 s after policy authority begins,
and the gate latches HOLD if the degraded joint stays pinned at the slew limit for 0.5 s
(`degradation_joint_saturated`). The spec is in the telemetry
manifest, and every tick record carries the per-motor scale actually applied. To reverse
it, restart the bridge without the flag. It is not a model of any particular motor fault
and must never be described as motor damage.

Expect a softer leg: at `s = 0.6` the joint's effective torque ceiling under the slew
clip drops from about 4.4 N m to about 2.6 N m. The attitude intervention (0.40 rad) and
the operator's deadman are the backstops.

## Stage sequence

| Stage | What | Pass condition |
|---|---|---|
| X0 | nominal live run, 60 s, 10 repeats | fidelity PASS; used for calibration |
| X1 | controlled degradation at `s_train`, 60 s, 10 sessions, interleaved with 10 nominal | the monitor's detection and localisation rates are recorded, not gated |
| X2 | runs A to F of `docs/research/EXPERIMENT.md`, 5 trials of 20 s each, randomised order | trials failing the fidelity gate are kept and reported, and scored 0 in the sensitivity analysis |

During each trial, optionally watch the health vector live (read-only):

```bash
PYTHONPATH=src python3 scripts/phoenix_live_monitor.py <trial>/bridge.jsonl --baseline <X0 baseline>
```

After each trial:

```bash
PYTHONPATH=src python3 scripts/phoenix_loop.py fidelity <trial>/bridge.jsonl --out <trial>/fidelity.json
PYTHONPATH=src python3 scripts/phoenix_loop.py assess <trial>/bridge.jsonl \
    --baseline <X0 baseline> --regime stand --out <trial>/health.json
```

## Abort

Release the deadman. The bridge latches hold, then damps. Do not re-arm a degraded
stage after an attitude abort without writing down what happened first.
