# Training runbook

Needs Isaac Lab (default `~/Sim/IsaacLab`, override `ISAACLAB_PATH`) and a free GPU.
Check the GPU is free before starting (`nvidia-smi`); a training run competes with any
other GPU process. Nothing in this runbook has been run for Phoenix v2 yet.

## 1. Build the Phoenix overlay from a monitored run

```bash
PYTHONPATH=src python3 scripts/phoenix_loop.py calibrate nominal_*/bridge.jsonl \
    --regime stand --out runs/baseline_stand.json
PYTHONPATH=src python3 scripts/phoenix_loop.py assess degraded/bridge.jsonl \
    --baseline runs/baseline_stand.json --regime stand --out runs/health.json
PYTHONPATH=src python3 scripts/phoenix_loop.py condition runs/health.json \
    --parent-env ../stand_v3_h25 --baseline runs/baseline_stand.json \
    --out configs/env/conditioned/rr_thigh_<date>.yaml
```

Each step refuses rather than guessing: calibration refuses runs that failed the
deployment-fidelity gate or have fewer than 20 usable windows per joint; `assess`
refuses a baseline from another command regime; `condition` refuses a run that failed
the fidelity gate, a global shift, or more than two degraded joints (exit 3 means "no
targeted distribution", which is a valid outcome).

The overlay's `defaults:` path is relative to the overlay's own directory. Its
`phoenix_condition` block records the health report, the telemetry file hashes and the
baseline, so the training run can be traced back to the robot data.

## 2. Train the arms

Every arm uses `configs/train/phoenix_finetune.yaml` (300 iterations, identical PPO
settings) and warm-starts from the incumbent, so the overlay is the only difference:

```bash
INC=checkpoints/phoenix-stand-h25-lat-noise/2026-06-22_21-08-20/model_799.pt
for s in $(seq 1 10); do
  scripts/phoenix_train_candidate.sh continued   configs/env/phoenix_v2/continued.yaml            $INC $s
  scripts/phoenix_train_candidate.sh broad       configs/env/phoenix_v2/broad_actuator.yaml       $INC $s
  scripts/phoenix_train_candidate.sh jointbroad  configs/env/phoenix_v2/joint_broad_rr_thigh.yaml $INC $s
  scripts/phoenix_train_candidate.sh phoenix     configs/env/conditioned/rr_thigh_seed$s.yaml     $INC $s
  scripts/phoenix_train_candidate.sh oracle      configs/env/phoenix_v2/oracle_rr_thigh_0p6.yaml  $INC $s
done
```

Ten seeds per arm, fixed by `docs/research/EXPERIMENT.md`. The Phoenix arm needs one
overlay per seed, each built from a seeded bootstrap resample of the hardware degraded
sessions (the resampling step is not yet scripted; `condition` builds one overlay from
one health report). The oracle overlay is a placeholder until its width is set from the
Phoenix overlays. Each run directory gets
`train.yaml`, the leaf `env.yaml` and `env_resolved.yaml` (the full merged tree).

## 3. Evaluate

Evaluate every candidate and the incumbent on `configs/env/phoenix_v2/eval_nominal.yaml`,
`eval_rr_thigh_0p6.yaml` and the held-out overlays, 256 episodes each, with
`phoenix.training.evaluate`. Blocker: the evaluator's attitude check reads Isaac Lab 3.0
quaternions in the wrong order (audit H6); until that fix lands, the primary endpoint's
attitude condition cannot be computed.

## 4. Export, parity, gate

```bash
./scripts/deploy.sh <candidate.pt>                   # ONNX export
python3 scripts/parity_gate.py --checkpoint <candidate.pt> --onnx <policy.onnx> \
    --deploy-cfg <deploy.yaml> --parquet <real capture> --json-out parity.json
PYTHONPATH=src python3 scripts/phoenix_loop.py gate --candidate cand_eval.json \
    --incumbent inc_eval.json --candidate-sha256 <sha> --parity parity.json --out decision.json
```

Evaluation JSON format for `gate`: `{"checkpoint_sha256": ..., "conditions": {"degraded":
{"episodes": [...]}, "nominal": {...}, "held_out": {...}}}` with one primary-endpoint value
per episode.

## Configuration reference

| Key | Meaning |
|---|---|
| `domain_randomization.motor_strength_scale: [lo, hi]` | every joint's stiffness and damping scaled per env at startup |
| `domain_randomization.targeted_actuator.joints: {name: [lo, hi]}` | the targeted joint(s), at most two, range inside [0.3, 1.0] |
| `domain_randomization.targeted_actuator.nominal_fraction` | share of envs left at the parent recipe (0.5 default; 0.0 in evaluation overlays) |
| `domain_randomization.targeted_actuator.scale_damping` | scale damping with stiffness (true, matching the hardware intervention) |
| `domain_randomization.targeted_actuator_seed` | RNG seed for the targeted factors |

The targeted term is applied even when `domain_randomization.enabled` is false, so an
evaluation overlay can switch every other randomisation off and keep the degradation.
