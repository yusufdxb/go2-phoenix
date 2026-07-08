# Repository structure

A map for first-time visitors.

```
go2-phoenix/
├── README.md                 hero: what Phoenix is + quick start
├── pyproject.toml            package metadata + optional [sim] / [real] / [dev]
├── CITATION.cff              cite the project
├── LICENSE                   MIT
│
├── src/phoenix/              the library
│   ├── sim_env/              GO2 env factory on Isaac Lab's rough-terrain task
│   ├── training/             PPO (rsl_rl) trainer + evaluation rollouts
│   ├── sim2real/             ONNX export with parity gate, ROS 2 policy node, safety predicates
│   ├── real_world/           rule-based failure detector, Parquet trajectory logger
│   ├── replay/               Halton variation sampler + Isaac Sim reconstruction
│   ├── adaptation/           failure-curriculum fine-tuning + reset bridge
│   └── demo/                 side-by-side video pipeline (ffmpeg)
│
├── configs/
│   ├── env/                  layered env YAMLs (defaults: chain)
│   ├── train/                PPO + adaptation + sweep specs
│   ├── sim2real/             deploy.yaml (ROS 2 + bridges + safety)
│   ├── replay/               Halton variation specs
│   └── _generated/           materialized per-cell sweep configs (gitignored)
│
├── scripts/
│   ├── train.sh              entry point: train a policy
│   ├── deploy.sh             entry point: hand off to Jetson
│   ├── replay.sh             entry point: reconstruct a parquet in sim
│   ├── adapt.sh              entry point: failure-curriculum fine-tune
│   ├── demo.sh               entry point: render the side-by-side video
│   ├── sweep_run.py          spec-driven benchmark sweep runner
│   └── harness_*.{sh,py}     lab-day preflight, recorder, diversity, EOD
│
├── tests/                    unit tests (torch-free, ROS-free; CI-safe)
│   └── test_sim_integration.py   marked @pytest.mark.sim, runs locally only
│
├── docs/
│   ├── architecture.{dot,svg}    rendered system diagram
│   ├── deploy_mode_switch_runbook.md   how to flip the two-policy mode switch on
│   ├── structure.md              this file
│   └── changelog.md              release-style summary of past gates
│
├── checkpoints/              .pt + .onnx artifacts (gitignored)
├── data/                     parquets, videos, npz (gitignored)
├── media/                    demo + render clips (gitignored)
└── docker/                   CPU-only testbox image for CI
```

## The two Python contexts

Phoenix runs in two Python environments that never share a process:

| Context | Where | Modules that import from it |
|---|---|---|
| Isaac Lab Python | `$ISAACLAB_PATH/isaaclab.sh -p` | `sim_env`, `training`, `replay`, `adaptation`, `demo.benchmark`, `sim2real.export` |
| System Python + ROS 2 | `/opt/ros/humble` + `pip install -e ".[real]"` | `sim2real.ros2_policy_node`, `real_world`, `demo.video_compose` |

Data crosses the boundary as files: `*.onnx`, `*.parquet`, `*.mp4`.
No module imports `torch` *and* `rclpy`.

## CI scope

`tests/` is torch-free + ROS-free by construction. Run the full
CI-safe suite with:

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
