# Repository structure

The Phoenix loop maps onto five places in the code.

| Stage | Where | Python context |
|---|---|---|
| RUN / REDEPLOY | `src/phoenix/sim2real/`: policy node, command wire, final actuator gate (`actuator_gate.py`), LowCmd/LowState bridges, telemetry, deploy contract and locks, preflight gates A to H, controlled degradation (`degradation.py`) | system Python + ROS 2 |
| DETECT | `src/phoenix/monitor/`: `layers.py` (four-layer record), `residual.py` (residual, calibration, `s_hat`), `health.py` (persistence, health vector), `fidelity.py` (deployment-fidelity gate) | plain numpy |
| CONDITION | `src/phoenix/condition/distribution.py` (health report to overlay), `toy_model.py` (explanatory model); sim event `scale_targeted_actuator_gains` in `src/phoenix/sim_env/go2_env_cfg.py` | numpy; the event needs Isaac Lab |
| TRAIN | `src/phoenix/training/ppo_runner.py` (`--resume` warm start), `configs/train/phoenix_finetune.yaml`, `scripts/phoenix_train_candidate.sh`, arm overlays in `configs/env/phoenix_v2/` | Isaac Lab |
| VERIFY | `src/phoenix/validate/` (candidate gate, promotion), `src/phoenix/training/evaluate.py`, `scripts/parity_gate.py`, `src/phoenix/sim2real/export.py` | numpy / Isaac Lab |

Operator entry point for the offline steps: `scripts/phoenix_loop.py`
(`fidelity`, `calibrate`, `assess`, `condition`, `gate`, `promote`).

Other top-level directories: `configs/` (layered YAML; `env/phoenix_v2/` holds the
experiment's arms and evaluation conditions), `docs/research/` (question, experiment,
related work, audit, outline, demo), `docs/runbooks/`, `docs/legacy/` (index of demoted
work), `tests/` (torch-free and ROS-free), `checkpoints/`, `data/`, `logs/` (all
gitignored).

## Two Python contexts

Isaac Lab Python (`$ISAACLAB_PATH/isaaclab.sh -p`) and system Python with ROS 2 never
share a process. Data crosses as files: `.onnx`, `bridge.jsonl`, `.parquet`, overlay
`.yaml`, decision `.json`. No module imports both `torch` and `rclpy`.

## Tests

```bash
PYTHONPATH=src PHOENIX_SKIP_HEAVY=1 pytest tests -m "not sim and not ros"
```

`PYTHONPATH=src` matters: an editable install of another checkout would otherwise be
imported instead.
