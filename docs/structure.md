# Repository structure

A map for first-time visitors.

| Path | Purpose |
|---|---|
| `src/phoenix/sim_env/` | Isaac Lab GO2 environment |
| `src/phoenix/velocity/` | Walking observation, task and checkpoint contract |
| `src/phoenix/training/` | PPO training and evaluation records |
| `src/phoenix/sim2sim/` | MuJoCo model, scenarios and gate |
| `src/phoenix/sim2real/` | ONNX export, controller and safety boundaries |
| `src/phoenix/real_world/` | Failure detector and trajectory logger |
| `src/phoenix/replay/` and `src/phoenix/adaptation/` | Failure replay and fine-tuning |
| `configs/` | Environment, training, gate and deploy configuration |
| `scripts/` | Supported command-line entry points |
| `tests/` | Offline and integration tests |
| `docs/` | Architecture, evidence and result reports |
| `checkpoints/` and `data/` | Generated artifacts, generally ignored by Git |

## The two Python contexts

Phoenix runs in two Python environments that never share a process:

| Context | Where | Modules that import from it |
|---|---|---|
| Isaac Lab Python | `$ISAACLAB_PATH/isaaclab.sh -p` | `sim_env`, `training`, `replay`, `adaptation`, `demo.benchmark`, `sim2real.export` |
| System Python + ROS 2 | `/opt/ros/humble` + `pip install -e ".[real]"` | `sim2real.ros2_policy_node`, `real_world`, `demo.video_compose` |

Data crosses the boundary as files: `*.onnx`, `*.parquet`, `*.mp4`.
No module imports `torch` *and* `rclpy`.

## CI scope

Most offline tests run without Isaac Lab or ROS 2. Run the filtered
suite with:

```bash
pytest tests -m "not sim and not ros"
```

Isaac Lab and ROS 2 paths are exercised manually on the hardware.

## Where to start reading code

For a single-pass orientation:

1. `src/phoenix/sim_env/go2_env_cfg.py`: what the env looks like
2. `src/phoenix/training/ppo_runner.py`: how it trains
3. `src/phoenix/sim2real/export.py` + `verify_deploy.py`: sim-to-real handoff with parity
4. `src/phoenix/sim2real/ros2_policy_node.py` + `safety.py`: on-robot loop with fail-closed semantics
5. `src/phoenix/real_world/failure_detector.py`: what gets flagged as a failure
6. `src/phoenix/adaptation/curriculum.py` + `reset_bridge.py`: close the loop
